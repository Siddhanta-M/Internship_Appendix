import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist      # Velocity command message
import math
from sensor_msgs.msg import LaserScan    # LiDAR scan message
from nav_msgs.msg import Odometry        # Robot odometry message
from tf_transformations import euler_from_quaternion
from rclpy.qos import QoSProfile, ReliabilityPolicy
import csv
import os


class CmdVelPublisher(Node):
    """
    ROS2 node that controls the robot by combining:

    1. Trajectory tracking
       - Generates a desired figure-eight trajectory.
       - Computes velocity commands using a nonlinear controller.

    2. Obstacle avoidance
       - Continuously monitors the LiDAR.
       - Overrides the trajectory controller whenever an obstacle
         is detected closer than 0.4 m.

    The resulting velocity commands are published to:

        /robot_2/cmd_vel_des

    Additionally, controller performance is logged to a CSV file
    for later analysis.
    """

    def __init__(self):

        # Initialise the ROS2 node
        super().__init__('path_traveller')

        # ---------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------

        # Publish desired linear and angular velocities.
        self.publisher_ = self.create_publisher(
            Twist,
            '/robot_2/cmd_vel_des',
            10
        )

        # ---------------------------------------------------------
        # Odometry subscription
        # ---------------------------------------------------------

        # BEST_EFFORT reliability is commonly used for odometry
        # because occasional packet loss is acceptable while
        # minimising communication delay.
        odom_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            '/robot_2/odom',
            self.odom_callback,
            odom_qos
        )

        # ---------------------------------------------------------
        # LiDAR subscription
        # ---------------------------------------------------------

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/robot_2/scan',
            self.scan_callback,
            10
        )

        # Indicates whether the initial robot pose has been stored.
        self.initialized = False

        # ---------------------------------------------------------
        # Controller parameters
        # ---------------------------------------------------------

        # Declare configurable ROS2 parameters.
        self.declare_parameter("a", 1.0)
        self.declare_parameter("a_2", 0.1)
        self.declare_parameter("l_x", 0.5)
        self.declare_parameter("k_x", 1.0)
        self.declare_parameter("l_y", 0.5)
        self.declare_parameter("k_y", 1.0)

        # Read parameter values into Python variables.
        self.a = self.get_parameter("a").value
        self.a_2 = self.get_parameter("a_2").value
        self.l_x = self.get_parameter("l_x").value
        self.k_x = self.get_parameter("k_x").value
        self.l_y = self.get_parameter("l_y").value
        self.k_y = self.get_parameter("k_y").value

        # ---------------------------------------------------------
        # Control loop timing
        # ---------------------------------------------------------

        # Execute controller every 50 ms (20 Hz).
        self.timer_period = 0.05

        self.timer = self.create_timer(
            self.timer_period,
            self.timer_callback
        )

        # ---------------------------------------------------------
        # CSV logging
        # ---------------------------------------------------------

        # Create a CSV file if it does not already exist.
        file_exists = os.path.exists("output.csv")

        self.csv_file = open("output.csv", "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        # Write column headings only once.
        if not file_exists:
            self.csv_writer.writerow([
                "x_error",
                "y_error",
                "x",
                "y",
                "linear_velocity",
                "angular_velocity"
            ])

        # Store latest LiDAR scan.
        self.latest_scan = None

        # Internal controller time.
        self.time = 0.0

    def yaw_from_quaternion(self, q):
        """
        Convert the robot orientation from quaternion form
        into a yaw angle (heading).
        """

        _, _, yaw = euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        return yaw

    def scan_callback(self, msg):
        """
        Save the newest LiDAR scan.
        """
        self.latest_scan = msg

    def odom_callback(self, msg):
        """
        Update the robot pose using odometry.
        """

        # Current robot position.
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        # Current orientation.
        q = msg.pose.pose.orientation
        self.theta = self.yaw_from_quaternion(q)

        # Save the robot's initial pose only once.
        #
        # The desired trajectory will later be generated
        # relative to this starting position.
        if not self.initialized:
            self.x0 = self.x
            self.y0 = self.y
            self.theta0 = self.theta
            self.initialized = True

    def controller(self):
        """
        Generate the desired trajectory and compute the
        required velocity commands.

        Returns:
            linear velocity
            angular velocity
            x tracking error
            y tracking error
        """

        t = self.time

        a = self.a
        a_2 = self.a_2
        l_x = self.l_x
        k_x = self.k_x
        l_y = self.l_y
        k_y = self.k_y

        # Scale factor controlling trajectory speed.
        j = 0.4 / (a * math.sqrt(2))

        # -----------------------------------------------------
        # Desired figure-eight trajectory
        # -----------------------------------------------------

        x_local = a * math.sin(j * t)

        y_local = (
            a *
            math.sin(j * t) *
            math.cos(j * t)
        )

        # Desired trajectory velocities.
        dx_local = j * a * math.cos(j * t)
        dy_local = j * a * math.cos(2 * j * t)

        # Transform from local coordinates into world coordinates.
        x_d = (
            self.x0 +
            x_local * math.cos(self.theta0) -
            y_local * math.sin(self.theta0)
        )

        y_d = (
            self.y0 +
            x_local * math.sin(self.theta0) +
            y_local * math.cos(self.theta0)
        )

        dx_d = (
            dx_local * math.cos(self.theta0) -
            dy_local * math.sin(self.theta0)
        )

        dy_d = (
            dx_local * math.sin(self.theta0) +
            dy_local * math.cos(self.theta0)
        )

        # -----------------------------------------------------
        # Tracking errors
        # -----------------------------------------------------

        ex = x_d - self.x - 0.1
        ey = y_d - self.y

        # -----------------------------------------------------
        # Nonlinear feedback controller
        # -----------------------------------------------------

        # Hyperbolic tangent limits the control effort,
        # preventing excessively large corrections.
        x_hol = (
            dx_d +
            l_x * math.tanh((k_x / l_x) * ex)
        )

        y_hol = (
            dy_d +
            l_y * math.tanh((k_y / l_y) * ey)
        )

        # Convert desired holonomic motion into
        # differential-drive commands.
        linear = (
            math.cos(self.theta) * x_hol +
            math.sin(self.theta) * y_hol
        )

        ang = (
            (1 / a_2) *
            (
                -math.sin(self.theta) * x_hol +
                math.cos(self.theta) * y_hol
            )
        )

        return linear, ang, ex, ey

    def timer_callback(self):
        """
        Main control loop.
        Executes every 50 ms.
        """

        # Advance controller time.
        self.time += self.timer_period

        # Wait until initial pose is known.
        if not self.initialized:
            return

        # Wait until at least one LiDAR scan has arrived.
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        ranges = scan.ranges

        # -----------------------------------------------------
        # Find the nearest obstacle
        # -----------------------------------------------------

        closest_range = float('inf')
        closest_angle = 0.0

        for i, r in enumerate(ranges):

            # Ignore invalid measurements.
            if not math.isfinite(r):
                continue

            if r < closest_range:
                closest_range = r

                closest_angle = (
                    scan.angle_min +
                    i * scan.angle_increment
                )

        msg = Twist()

        # -----------------------------------------------------
        # Obstacle avoidance
        # -----------------------------------------------------

        # If an obstacle is closer than 0.4 m,
        # temporarily ignore the trajectory controller.
        if closest_range < 0.4:

            # Slow oscillating forward motion.
            msg.linear.x = 0.3 * math.sin(self.time) ** 2

            # Turn away from the obstacle.
            if closest_angle == 0:
                msg.angular.z = -1.5
            else:
                msg.angular.z = -1.5 * closest_angle

        # -----------------------------------------------------
        # Normal trajectory tracking
        # -----------------------------------------------------

        else:

            (
                msg.linear.x,
                msg.angular.z,
                xerror,
                yerror,
            ) = self.controller()

            # Record controller performance.
            self.csv_writer.writerow([
                xerror,
                yerror,
                self.x,
                self.y,
                msg.linear.x,
                msg.angular.z
            ])

        # Publish velocity command.
        self.publisher_.publish(msg)


def main(args=None):
    """
    Main entry point.
    """

    # Initialise ROS2.
    rclpy.init(args=args)

    # Create controller node.
    node = CmdVelPublisher()

    try:
        # Run until interrupted.
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    # ---------------------------------------------------------
    # Stop the robot before exiting
    # ---------------------------------------------------------

    stop_msg = Twist()

    # Zero velocities stop the robot safely.
    node.publisher_.publish(stop_msg)

    # Clean up resources.
    node.destroy_node()

    rclpy.shutdown()


# Program entry point.
if __name__ == '__main__':
    main()

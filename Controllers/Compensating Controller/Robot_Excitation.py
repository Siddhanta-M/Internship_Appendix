import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist       # Velocity command message
import math
from sensor_msgs.msg import LaserScan     # LiDAR scan message


class CmdVelPublisher(Node):
    """
    ROS2 node that generates velocity commands for robot_2.

    The node has two operating modes:

    1. Excitation mode:
       - Generates a continuously changing velocity signal.
       - Used for testing robot dynamics, system identification,
         and controller response.

    2. Obstacle avoidance mode:
       - Uses LiDAR to detect nearby obstacles.
       - Overrides the excitation signal when an obstacle is
         too close.

    Velocity commands are published to:

        /robot_2/cmd_vel_des
    """

    def __init__(self):

        # Initialise ROS2 node.
        super().__init__('cmd_vel_publisher')

        # ---------------------------------------------------------
        # Publisher
        # ---------------------------------------------------------

        # Publish desired robot velocity commands.
        #
        # Twist contains:
        #   linear.x  -> forward/backward velocity
        #   angular.z -> rotational velocity
        self.publisher_ = self.create_publisher(
            Twist,
            '/robot_2/cmd_vel_des',
            10
        )

        # ---------------------------------------------------------
        # LiDAR subscription
        # ---------------------------------------------------------

        # Subscribe to the robot's LiDAR sensor.
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/robot_2/scan',
            self.scan_callback,
            10
        )

        # ---------------------------------------------------------
        # Timing
        # ---------------------------------------------------------

        # Controller update rate:
        #
        # 0.05 seconds = 20 Hz
        self.timer_period = 0.05

        self.timer = self.create_timer(
            self.timer_period,
            self.timer_callback
        )

        # Store the most recent LiDAR scan.
        self.latest_scan = None

        # Internal time variable used by the excitation signal.
        self.time = 0.0


    def scan_callback(self, msg):
        """
        Store the latest LiDAR measurement.

        The newest scan is saved and used during the next
        controller update.
        """

        self.latest_scan = msg


    def excite(self):
        """
        Generate an excitation velocity signal.

        The signal is composed of multiple sine and cosine
        components with different frequencies.

        This creates a rich input signal useful for:
            - robot system identification
            - actuator testing
            - dynamic response measurements

        Returns:
            linear velocity
            angular velocity
        """

        # Forward velocity excitation.
        #
        # Multiple frequencies are combined so the robot
        # experiences different acceleration patterns.
        linear = (
            0.18 *
            (
                math.sin(self.time)
                +
                math.sin(3.6 * self.time)
                +
                math.cos(self.time / 2.3)
            )
        )

        # Rotational velocity excitation.
        #
        # The combination of frequencies causes the robot
        # to continuously change turning behaviour.
        ang = (
            0.3 *
            (
                math.sin(2.81 * self.time)
                +
                2 * math.sin(self.time / 3)
                +
                math.cos(3 * self.time)
            )
        )

        return linear, ang


    def timer_callback(self):
        """
        Main controller loop.

        Runs every timer_period seconds.
        """

        # Update internal time.
        self.time += self.timer_period

        # Wait until LiDAR data is available.
        if self.latest_scan is None:
            return


        scan = self.latest_scan
        ranges = scan.ranges


        # ---------------------------------------------------------
        # Find closest obstacle
        # ---------------------------------------------------------

        closest_range = float('inf')
        closest_angle = 0.0

        # Search through all LiDAR measurements.
        for i, r in enumerate(ranges):

            # Ignore invalid readings.
            if not math.isfinite(r):
                continue

            # Keep the closest valid measurement.
            if r < closest_range:

                closest_range = r

                # Convert beam index into an angle.
                closest_angle = (
                    scan.angle_min +
                    i * scan.angle_increment
                )


        # Create a new velocity command message.
        msg = Twist()


        # ---------------------------------------------------------
        # Obstacle avoidance mode
        # ---------------------------------------------------------

        # If an obstacle is closer than 0.75 m,
        # override the excitation signal.
        if closest_range < 0.75:

            # Move forward slowly while turning away.
            #
            # The sine-squared term keeps the velocity
            # positive while smoothly varying its magnitude.
            msg.linear.x = (
                0.3 *
                math.sin(self.time) ** 2
            )


            # Determine turning direction.
            if closest_angle == 0:

                # Obstacle directly ahead:
                # rotate strongly.
                msg.angular.z = -1.5

            else:

                # Turn proportionally away from the obstacle.
                msg.angular.z = (
                    -1.5 *
                    closest_angle
                )


        # ---------------------------------------------------------
        # Excitation mode
        # ---------------------------------------------------------

        else:

            # No obstacle detected, so apply the
            # test excitation signal.
            (
                msg.linear.x,
                msg.angular.z
            ) = self.excite()


        # Send velocity command to the robot.
        self.publisher_.publish(msg)



def main(args=None):
    """
    Main program entry point.
    """

    # Initialise ROS2.
    rclpy.init(args=args)

    # Create velocity publisher node.
    node = CmdVelPublisher()


    try:

        # Keep node active and process callbacks.
        rclpy.spin(node)


    except KeyboardInterrupt:
        pass


    # ---------------------------------------------------------
    # Stop robot before shutting down
    # ---------------------------------------------------------

    stop_msg = Twist()

    # Zero velocity command.
    node.publisher_.publish(stop_msg)


    # Clean up ROS2 resources.
    node.destroy_node()

    rclpy.shutdown()



# Program entry point.
if __name__ == '__main__':
    main()

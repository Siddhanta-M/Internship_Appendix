import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist      # Velocity command message
from sensor_msgs.msg import LaserScan    # LiDAR scan message

import math


class WallHugger(Node):
    """
    ROS2 node that implements a wall-following controller.

    The robot attempts to maintain a fixed distance from either
    the left or right wall while driving forward.

    If an obstacle is detected closer than a specified safety
    distance, normal wall following is temporarily overridden
    by a simple obstacle avoidance behaviour.
    """

    def __init__(self):

        # Initialise the ROS2 node.
        super().__init__('wall_hugger')

        # ---------------------------------------------------------
        # ROS interfaces
        # ---------------------------------------------------------

        # Publisher for velocity commands.
        self.cmd_pub = self.create_publisher(
            Twist,
            '/robot_2/cmd_vel_des',
            10
        )

        # Subscribe to the robot's LiDAR.
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/robot_2/scan',
            self.scan_callback,
            10
        )

        # ---------------------------------------------------------
        # Controller parameters
        # ---------------------------------------------------------

        # Desired distance from the wall.
        self.declare_parameter("target_distance", 0.6)

        # Controller update period (seconds).
        self.declare_parameter("time_period", 0.05)

        # Constant forward velocity.
        self.declare_parameter("forward_speed", 0.2)

        # Distance error gain.
        self.declare_parameter("kd", 0.5)

        # Wall angle error gain.
        self.declare_parameter("ktheta", 0.5)

        # Turning speed parameter.
        self.declare_parameter("turn_speed", 0.6)

        # Minimum safe obstacle distance.
        self.declare_parameter("safe_distance", 0.3)

        # ---------------------------------------------------------
        # Read parameter values
        # ---------------------------------------------------------

        self.target_distance = self.get_parameter("target_distance").value
        self.time_period = self.get_parameter("time_period").value
        self.forward_speed = self.get_parameter("forward_speed").value
        self.kd = self.get_parameter("kd").value
        self.ktheta = self.get_parameter("ktheta").value
        self.turn_speed = self.get_parameter("turn_speed").value
        self.safe_distance = self.get_parameter("safe_distance").value

        # Choose whether to follow the left or right wall.
        #
        # True  -> left wall
        # False -> right wall
        self.follow_left = self.declare_parameter(
            "follow_left",
            True
        ).value

        # side is used throughout the controller:
        #
        #  1  = left
        # -1  = right
        self.side = 1 if self.follow_left else -1

        # Store the latest received scan.
        self.latest_scan = None

        # ---------------------------------------------------------
        # Create periodic controller
        # ---------------------------------------------------------

        # Execute every time_period seconds.
        self.timer = self.create_timer(
            self.time_period,
            self.timer_callback
        )

    def scan_callback(self, msg):
        """
        Store the newest LiDAR scan.
        """
        self.latest_scan = msg

    def get_range(self, angle_deg):
        """
        Return the LiDAR range measurement at a specified angle.

        Parameters
        ----------
        angle_deg : float
            Desired angle in degrees relative to the robot.

        Returns
        -------
        Distance measurement in metres.
        """

        # If no scan has been received yet,
        # return infinity.
        if self.latest_scan is None:
            return float('inf')

        # Convert requested angle to radians.
        angle_rad = math.radians(angle_deg)

        # Convert angle into an index within the scan array.
        idx = int(
            (angle_rad - self.latest_scan.angle_min)
            / self.latest_scan.angle_increment
        )

        # Prevent index from exceeding array bounds.
        idx = max(
            0,
            min(idx, len(self.latest_scan.ranges) - 1)
        )

        r = self.latest_scan.ranges[idx]

        # Replace invalid measurements with
        # the sensor's maximum range.
        if not math.isfinite(r):
            return self.latest_scan.range_max

        return r

    def compute_wall_angle(self, side_dist, front_side_dist):
        """
        Estimate the orientation of the wall using two LiDAR
        measurements:

            side beam (90°)
            front-side beam (45°)

        The estimated angle is used to keep the robot aligned
        with the wall rather than simply maintaining distance.
        """

        theta = math.radians(self.side * 45)

        return math.atan2(
            front_side_dist * math.sin(theta)
            - side_dist * self.side,

            front_side_dist * math.cos(theta)
        )

    def timer_callback(self):
        """
        Main wall-following controller.

        Runs periodically according to the configured timer.
        """

        # Wait until a scan has been received.
        if self.latest_scan is None:
            return

        scan = self.latest_scan
        ranges = scan.ranges

        # -----------------------------------------------------
        # Find the closest obstacle
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
        # Obtain wall measurements
        # -----------------------------------------------------

        # Distance directly to the side wall.
        side_dist = self.get_range(self.side * 90)

        # Distance to the wall at 45°.
        front_side_dist = self.get_range(self.side * 45)

        # -----------------------------------------------------
        # Safety behaviour
        # -----------------------------------------------------

        # If an obstacle is closer than the configured
        # safety distance, temporarily ignore wall following.
        if closest_range < self.safe_distance:

            msg.linear.x = self.forward_speed

            # Turn away from the obstacle.
            if closest_angle == 0:
                msg.angular.z = -1.5 * self.side
            else:
                msg.angular.z = -1.5 * closest_angle

        # -----------------------------------------------------
        # Normal wall following
        # -----------------------------------------------------

        else:

            # Difference between actual and desired
            # wall distance.
            dist_error = (
                side_dist -
                self.target_distance
            ) * self.side

            # Difference between robot heading
            # and wall orientation.
            ang_error = self.compute_wall_angle(
                side_dist,
                front_side_dist
            )

            # If no nearby wall exists at 45°,
            # ignore the angle estimate.
            if front_side_dist > 2:
                ang_error = 0

            # Constant forward motion.
            msg.linear.x = self.forward_speed

            # Steering controller.
            #
            # Angular velocity consists of:
            #
            # - distance correction
            # - heading correction
            msg.angular.z = (
                self.kd * dist_error +
                self.ktheta * ang_error
            )

        # Publish velocity command.
        self.cmd_pub.publish(msg)


def main(args=None):
    """
    Main program entry point.
    """

    # Initialise ROS2.
    rclpy.init(args=args)

    # Create wall-following node.
    node = WallHugger()

    try:
        # Keep controller running.
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    # ---------------------------------------------------------
    # Stop the robot before shutting down
    # ---------------------------------------------------------

    stop = Twist()

    # Zero velocity command.
    node.cmd_pub.publish(stop)

    # Clean up resources.
    node.destroy_node()

    rclpy.shutdown()


# Program entry point.
if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry                  # Robot odometry
from irobot_create_msgs.msg import WheelVels, WheelTicks
import csv
import time
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist               # Commanded robot velocities

# ---------------------------------------------------------
# QoS configuration
# ---------------------------------------------------------

# BEST_EFFORT delivery is sufficient for high-frequency
# sensor data where occasional packet loss is acceptable.
qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=10
)


class VelocityLogger(Node):
    """
    ROS2 node that compares robot motion information obtained
    from several different sources:

        • Commanded velocities (cmd_vel_des)
        • Odometry estimates
        • Wheel angular velocities
        • Wheel encoder ticks

    For each source, the node calculates:

        - Linear velocity
        - Angular velocity
        - Linear acceleration
        - Angular acceleration

    All values are recorded into a CSV file for later analysis.
    """

    def __init__(self):

        # Initialise the ROS2 node.
        super().__init__('velocity_logger')

        # ---------------------------------------------------------
        # Subscriptions
        # ---------------------------------------------------------

        # Estimated robot velocity from odometry.
        self.create_subscription(
            Odometry,
            '/robot_2/odom',
            self.odom_cb,
            qos
        )

        # Measured wheel angular velocities.
        self.create_subscription(
            WheelVels,
            '/robot_2/wheel_vels',
            self.wheelvel_cb,
            qos
        )

        # Raw encoder tick counts.
        self.create_subscription(
            WheelTicks,
            '/robot_2/wheel_ticks',
            self.tick_cb,
            qos
        )

        # Desired velocity commands sent to the robot.
        self.create_subscription(
            Twist,
            '/robot_2/cmd_vel_des',
            self.cmdvel_cb,
            qos
        )

        # ---------------------------------------------------------
        # Previous measurements
        # ---------------------------------------------------------

        # Previous encoder tick message.
        self.prev_tick = None
        self.prev_tick_time = None
        self.prev_tick_v = None
        self.prev_tick_w = None

        # Most recent commanded velocity.
        self.latest_cmdvel = None

        # Previous wheel velocity estimates.
        self.prev_wheel_v = None
        self.prev_wheel_w = None
        self.prev_wheel_time = None

        # Previous odometry estimates.
        self.prev_odom_v = None
        self.prev_odom_w = None
        self.prev_odom_time = None

        # Storage for logged data.
        self.log = []

        # ---------------------------------------------------------
        # Robot parameters
        # ---------------------------------------------------------

        # Encoder conversion factor
        # (radians travelled per encoder tick).
        self.k = 0.012349028

        # Wheel radius (m).
        self.r = 0.03575

        # Distance between wheels (m).
        self.L = 0.233

        # ---------------------------------------------------------
        # CSV output
        # ---------------------------------------------------------

        self.file = open(
            'robot_velocity_log.csv',
            'w',
            newline=''
        )

        self.writer = csv.writer(self.file)

        # Write column headings.
        self.writer.writerow([
            "time",

            "cmd_v",
            "cmd_w",

            "odom_v",
            "odom_w",
            "odom_a",
            "odom_alpha",

            "wheel_v",
            "wheel_w",
            "wheel_a",
            "wheel_alpha",

            "tick_v",
            "tick_w",
            "tick_a",
            "tick_alpha"
        ])

    # ---------------------------------------------------------
    # Commanded velocity callback
    # ---------------------------------------------------------

    def cmdvel_cb(self, msg):
        """
        Store the latest commanded robot velocity.
        """

        self.latest_cmdvel = msg

        self.latest_cmdvel_time = self.get_clock().now()

        self.try_log()

    # ---------------------------------------------------------
    # Odometry callback
    # ---------------------------------------------------------

    def odom_cb(self, msg):
        """
        Compute velocity and acceleration from odometry.
        """

        # Convert ROS timestamp into seconds.
        t = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        # Extract linear and angular velocity.
        v = msg.twist.twist.linear.x
        w = msg.twist.twist.angular.z

        # Initialise previous values.
        if self.prev_odom_time is None:

            self.prev_odom_time = t
            self.prev_odom_v = v
            self.prev_odom_w = w
            return

        # Time between measurements.
        dt = t - self.prev_odom_time

        # Numerical differentiation.
        a = (v - self.prev_odom_v) / dt
        alpha = (w - self.prev_odom_w) / dt

        # Store calculated values.
        self.odom_data = (
            v,
            w,
            a,
            alpha,
            t
        )

        # Update previous measurements.
        self.prev_odom_time = t
        self.prev_odom_v = v
        self.prev_odom_w = w

        self.try_log()

    # ---------------------------------------------------------
    # Wheel velocity callback
    # ---------------------------------------------------------

    def wheelvel_cb(self, msg):
        """
        Compute robot velocity from wheel angular velocities.
        """

        t = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        r = self.r
        L = self.L

        # Convert wheel angular velocity
        # into linear wheel speed.
        vL = r * msg.velocity_left
        vR = r * msg.velocity_right

        # Differential-drive kinematics.
        v = (vR + vL) / 2.0
        w = (vR - vL) / L

        if self.prev_wheel_time is None:

            self.prev_wheel_time = t
            self.prev_wheel_v = v
            self.prev_wheel_w = w
            return

        dt = t - self.prev_wheel_time

        # Numerical differentiation.
        a = (v - self.prev_wheel_v) / dt
        alpha = (w - self.prev_wheel_w) / dt

        self.wheelvel_data = (
            v,
            w,
            a,
            alpha,
            t
        )

        self.prev_wheel_time = t
        self.prev_wheel_v = v
        self.prev_wheel_w = w

        self.try_log()

    # ---------------------------------------------------------
    # Wheel encoder callback
    # ---------------------------------------------------------

    def tick_cb(self, msg):
        """
        Compute velocity and acceleration from encoder ticks.
        """

        t = (
            msg.header.stamp.sec +
            msg.header.stamp.nanosec * 1e-9
        )

        # Wait until two measurements exist.
        if self.prev_tick is None:

            self.prev_tick = msg
            self.prev_tick_time = t
            return

        dt = t - self.prev_tick_time

        # Change in encoder counts.
        dL = (
            msg.ticks_left -
            self.prev_tick.ticks_left
        )

        dR = (
            msg.ticks_right -
            self.prev_tick.ticks_right
        )

        k = self.k
        r = self.r
        L = self.L

        # Convert encoder ticks into wheel speeds.
        vL = r * k * dL / dt
        vR = r * k * dR / dt

        # Differential-drive equations.
        v = (vR + vL) / 2.0
        w = (vR - vL) / L

        # Calculate acceleration.
        if self.prev_tick_v is None:

            a = 0.0
            alpha = 0.0

        else:

            a = (v - self.prev_tick_v) / dt

            alpha = (
                w -
                self.prev_tick_w
            ) / dt

        # Update previous measurements.
        self.prev_tick_v = v
        self.prev_tick_w = w

        self.prev_tick = msg
        self.prev_tick_time = t

        self.tick_data = (
            v,
            w,
            a,
            alpha,
            t
        )

        self.try_log()

    # ---------------------------------------------------------
    # Logging routine
    # ---------------------------------------------------------

    def try_log(self):
        """
        Write one row to the CSV file whenever all data
        sources have been updated.
        """

        # Wait until all three measurement methods
        # have produced data.
        if (
            not hasattr(self, "odom_data") or
            not hasattr(self, "wheelvel_data") or
            not hasattr(self, "tick_data")
        ):
            return

        now = self.get_clock().now()

        # Determine whether the latest command
        # is still considered valid.
        if self.latest_cmdvel is None:

            cmd_v = 0.0
            cmd_w = 0.0

        elif (
            now -
            self.latest_cmdvel_time
        ).nanoseconds / 1e9 > 0.5:

            # Treat old commands as zero.
            cmd_v = 0.0
            cmd_w = 0.0

        else:

            cmd_v = self.latest_cmdvel.linear.x
            cmd_w = self.latest_cmdvel.angular.z

        # Use odometry timestamp as the log time.
        t = self.odom_data[4]

        # Assemble one complete CSV row.
        row = [

            t,

            cmd_v,
            cmd_w,

            *self.odom_data[:4],

            *self.wheelvel_data[:4],

            *self.tick_data[:4]
        ]

        # Write row immediately.
        self.writer.writerow(row)

        self.file.flush()

        print("Velocity and acceleration calculated")


def main(args=None):
    """
    Main program entry point.
    """

    # Initialise ROS2.
    rclpy.init(args=args)

    # Create logging node.
    node = VelocityLogger()

    try:
        # Keep the node running.
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    # Clean up resources.
    node.destroy_node()

    rclpy.shutdown()


# Program entry point.
if __name__ == '__main__':
    main()

#!/usr/bin/env python3

# Import required libraries
import json                     # Save scan data in JSON format
from pathlib import Path        # Platform-independent file paths

import rclpy                    # ROS2 Python client library
from rclpy.node import Node     # Base class for ROS2 nodes
from sensor_msgs.msg import LaserScan  # Message type for LiDAR scans

import math                     # Mathematical functions (isinf, isnan)


class ScanLogger(Node):
    """
    ROS2 node that subscribes to a LaserScan topic and logs each scan
    to a JSON Lines (.jsonl) file.

    Before saving, the scan is preprocessed by:
        - Replacing invalid values (NaN or Infinity) with 4.0 m
        - Clipping all distances to a maximum of 4.0 m
        - Averaging adjacent beams to reduce the scan resolution
          by half (e.g., 720 beams -> 360 beams)

    The resulting scans are intended for later processing by
    the RGB image generation and YOLO pipeline.
    """

    def __init__(self):
        # Initialize the ROS2 node with the name "scan_logger"
        super().__init__('scan_logger')

        # ---------------------------------------------------------
        # Create directory for storing scan logs
        # ---------------------------------------------------------

        # Create ~/scan_logs if it doesn't already exist
        log_dir = Path.home() / 'scan_logs'
        log_dir.mkdir(exist_ok=True)

        # Output file where scans will be appended
        self.output_file = log_dir / 'scan.jsonl'

        # ---------------------------------------------------------
        # Subscribe to the LiDAR topic
        # ---------------------------------------------------------

        self.subscription = self.create_subscription(
            LaserScan,            # Message type
            '/robot_2/scan',      # Topic to subscribe to
            self.scan_callback,   # Function called for every scan
            10,                   # Queue size
        )

        # Display where data is being written
        self.get_logger().info(
            f'Writing scans to: {self.output_file}'
        )

    def scan_callback(self, msg: LaserScan):
        """
        Called automatically every time a LaserScan message
        arrives on /robot_2/scan.
        """

        # ---------------------------------------------------------
        # Step 1: Clean the raw range measurements
        # ---------------------------------------------------------

        # Replace invalid measurements with 4.0 m.
        #
        # Some LiDAR sensors report:
        #   inf  -> no obstacle detected
        #   NaN  -> invalid measurement
        #
        # Both are replaced with the maximum distance used
        # by the later image-generation pipeline.
        #
        # Any values larger than 4 m are also clipped to 4 m.
        processed_ranges = [
            4.0 if math.isinf(r) or math.isnan(r)
            else min(r, 4.0)
            for r in msg.ranges
        ]

        # ---------------------------------------------------------
        # Step 2: Reduce scan resolution
        # ---------------------------------------------------------

        # If the scan contains an even number of beams
        # (for example 720), average every adjacent pair.
        #
        # Example:
        #
        # Beam 0 + Beam 1 -> New Beam 0
        # Beam 2 + Beam 3 -> New Beam 1
        #
        # This halves the number of beams while smoothing
        # measurement noise.
        if len(processed_ranges) % 2 == 0:

            processed_ranges = [
                (processed_ranges[i] + processed_ranges[i + 1]) / 2.0
                for i in range(0, len(processed_ranges), 2)
            ]

            # Since every two beams become one,
            # the angular spacing doubles.
            angle_increment = msg.angle_increment * 2.0

        else:
            # If the scan already has an odd number of beams,
            # leave it unchanged.
            angle_increment = msg.angle_increment

        # ---------------------------------------------------------
        # Step 3: Build JSON record
        # ---------------------------------------------------------

        # Store both metadata and processed scan values.
        #
        # Each dictionary represents one complete LiDAR scan.
        data = {

            # ROS timestamp
            'timestamp_sec': msg.header.stamp.sec,
            'timestamp_nanosec': msg.header.stamp.nanosec,

            # Coordinate frame of the sensor
            'frame_id': msg.header.frame_id,

            # Angular information
            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,

            # Updated after beam averaging
            'angle_increment': angle_increment,

            # Timing information
            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,

            # Valid distance limits
            'range_min': msg.range_min,

            # Maximum distance after clipping
            'range_max': 4.0,

            # Processed scan that will later be converted
            # into binary and RGB images.
            'ranges': processed_ranges,

            # Original intensity values from the LiDAR.
            # These are preserved even though they are not
            # currently used in the image pipeline.
            'intensities': list(msg.intensities)
        }

        # ---------------------------------------------------------
        # Step 4: Append scan to the JSONL file
        # ---------------------------------------------------------

        # JSON Lines format stores one JSON object per line,
        # making it easy to stream large datasets.
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

        # ---------------------------------------------------------
        # Step 5: Display status information
        # ---------------------------------------------------------

        self.get_logger().info(
            f"Logged scan: {len(processed_ranges)} beams "
            f"(angle_increment={angle_increment:.6f})"
        )


def main(args=None):
    """
    Main entry point for the ROS2 node.
    """

    # Initialize ROS2
    rclpy.init(args=args)

    # Create the ScanLogger node
    node = ScanLogger()

    try:
        # Keep the node running and continuously listen
        # for incoming LaserScan messages.
        rclpy.spin(node)

    # Allow graceful shutdown using Ctrl+C
    except KeyboardInterrupt:
        pass

    # Clean up resources before exiting
    node.destroy_node()

    # Shut down the ROS2 client library
    rclpy.shutdown()


# Program entry point
if __name__ == '__main__':
    main()

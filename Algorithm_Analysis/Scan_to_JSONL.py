#!/usr/bin/env python3

import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import math


class ScanLogger(Node):
    def __init__(self):
        super().__init__('scan_logger')

        # Directory + filename
        log_dir = Path.home() / 'scan_logs'
        log_dir.mkdir(exist_ok=True)

        self.output_file = log_dir / 'scan.jsonl'

        self.subscription = self.create_subscription(
            LaserScan,
            '/robot_2/scan',
            self.scan_callback,
            10,
        )

        self.get_logger().info(f'Writing scans to: {self.output_file}')

    def scan_callback(self, msg: LaserScan):

        # 4 m
        processed_ranges = [
            4.0 if math.isinf(r) or math.isnan(r)
            else min(r, 4.0)
            for r in msg.ranges
        ]

        # averaging adjacent pairs
        if len(processed_ranges) % 2 == 0:
            processed_ranges = [
                (processed_ranges[i] + processed_ranges[i + 1]) / 2.0
                for i in range(0, len(processed_ranges), 2)
            ]

            angle_increment = msg.angle_increment * 2.0
        else:
            angle_increment = msg.angle_increment

        data = {
            'timestamp_sec': msg.header.stamp.sec,
            'timestamp_nanosec': msg.header.stamp.nanosec,
            'frame_id': msg.header.frame_id,

            'angle_min': msg.angle_min,
            'angle_max': msg.angle_max,
            'angle_increment': angle_increment,

            'time_increment': msg.time_increment,
            'scan_time': msg.scan_time,

            'range_min': msg.range_min,
            'range_max': 4.0,

            # Processed scan used by your YOLO pipeline
            'ranges': processed_ranges,

            'intensities': list(msg.intensities)
        }

        with open(self.output_file, 'a') as f:
            f.write(json.dumps(data) + '\n')

        self.get_logger().info(
            f"Logged scan: {len(processed_ranges)} beams "
            f"(angle_increment={angle_increment:.6f})"
        )


def main(args=None):
    rclpy.init(args=args)

    node = ScanLogger()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

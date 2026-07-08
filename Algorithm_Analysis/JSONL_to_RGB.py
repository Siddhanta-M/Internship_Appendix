#!/usr/bin/env python3

import json
import numpy as np
import cv2
from pathlib import Path
from collections import deque
import time

class SimpleRGBBuffer:
    def __init__(self, buffer_size=3):
        self.buffer_size = buffer_size
        self.binary_arrays = deque(maxlen=buffer_size)

    def add_frame(self, ranges):
        """
        Convert one scan into a 64x360 binary image.
        """

        ranges = np.array(ranges)

        binary_array = np.zeros((64, 360), dtype=np.uint8)

        # Logger now stores 360 beams
        angle_indices = np.arange(360)

        # 0-4m -> 0-63 bins
        distance_indices = np.clip(
            ranges * (63.0 / 4.0),
            0,
            63
        ).astype(np.uint8)

        binary_array[distance_indices, angle_indices] = 255

        self.binary_arrays.append(binary_array)

    def get_rgb_image(self):
        if len(self.binary_arrays) < self.buffer_size:
            return None

        rgb = np.zeros((64, 384, 3), dtype=np.uint8)

        rgb[:, :360, 0] = self.binary_arrays[0]  # oldest
        rgb[:, :360, 1] = self.binary_arrays[1]  # middle
        rgb[:, :360, 2] = self.binary_arrays[2]  # newest

        return rgb


def main():

    jsonl_file = Path.home() / "scan_logs" / "scan.jsonl"

    output_dir = Path.home() / "scan_logs" / "rgb_output"
    output_dir.mkdir(exist_ok=True)

    current_image = output_dir / "current_rgb.png"

    buffer = SimpleRGBBuffer(buffer_size=3)

    frame_counter = 0
    image_counter = 0

    print(f"Watching: {jsonl_file}")
    print(f"Writing live image: {current_image}")

    with open(jsonl_file, "r") as f:

        # Start reading only new scans
        f.seek(0, 2)

        while True:

            line = f.readline()

            if not line:
                time.sleep(0.05)
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "ranges" not in data:
                continue

            ranges = data["ranges"]

            if len(ranges) != 360:
                print(
                    f"Skipping frame {frame_counter}: "
                    f"expected 360 beams, got {len(ranges)}"
                )
                continue

            buffer.add_frame(ranges)

            if frame_counter >= 2:

                rgb = buffer.get_rgb_image()

                if rgb is not None:


                    cv2.imwrite(str(current_image), rgb)

                    if image_counter % 25 == 0:
                        archive_file = (
                                output_dir /
                                f"rgb_{image_counter:05d}.png"
                        )

                    cv2.imwrite(str(archive_file), rgb)

                    image_counter += 1

                    if image_counter % 10 == 0:
                        print(
                            f"Frames: {frame_counter} "
                            f" Images: {image_counter}"
                        )

            frame_counter += 1


if __name__ == "__main__":
    main()

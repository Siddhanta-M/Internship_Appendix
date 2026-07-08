#!/usr/bin/env python3

# Import required libraries
import json                    # Read JSON formatted scan data
import numpy as np             # Numerical array operations
import cv2                     # Save images using OpenCV
from pathlib import Path       # Platform-independent file paths
from collections import deque  # Fixed-length buffer
import time                    # Sleep while waiting for new data


class SimpleRGBBuffer:
    """
    Stores the most recent laser scans and converts them into
    a 3-channel RGB image.

    Each scan is converted into a 64x360 binary image.
    The three newest scans are placed into the RGB channels:

        Red   = oldest scan
        Green = middle scan
        Blue  = newest scan

    This allows motion between consecutive scans to be encoded
    as color differences.
    """

    def __init__(self, buffer_size=3):
        # Number of scans to remember
        self.buffer_size = buffer_size

        # deque automatically removes the oldest frame when full
        self.binary_arrays = deque(maxlen=buffer_size)

    def add_frame(self, ranges):
        """
        Convert one LiDAR scan into a 64x360 binary image.

        Input:
            ranges = list of 360 distance measurements

        Output:
            64 rows  -> distance bins (0 to 4 meters)
            360 cols -> beam angles
        """

        # Convert list into NumPy array for faster processing
        ranges = np.array(ranges)

        # Create an empty binary image
        # Height = 64 distance bins
        # Width  = 360 beam angles
        binary_array = np.zeros((64, 360), dtype=np.uint8)

        # Beam index represents the column number
        # Beam 0 -> column 0
        # Beam 359 -> column 359
        angle_indices = np.arange(360)

        # Convert measured distance into one of 64 vertical bins.
        #
        # 0 m  -> row 0
        # 4 m  -> row 63
        #
        # Any values outside this range are clipped.
        distance_indices = np.clip(
            ranges * (63.0 / 4.0),
            0,
            63
        ).astype(np.uint8)

        # Mark each detected point as white (255)
        binary_array[distance_indices, angle_indices] = 255

        # Store this frame in the rolling buffer
        self.binary_arrays.append(binary_array)

    def get_rgb_image(self):
        """
        Combine the last three scans into one RGB image.

        Returns None until three frames have been collected.
        """

        # Need three frames before RGB image can be built
        if len(self.binary_arrays) < self.buffer_size:
            return None

        # Create RGB image.
        #
        # Width is 384 instead of 360.
        # Only the first 360 columns are used.
        rgb = np.zeros((64, 384, 3), dtype=np.uint8)

        # Store scans in different color channels
        rgb[:, :360, 0] = self.binary_arrays[0]  # Red = oldest
        rgb[:, :360, 1] = self.binary_arrays[1]  # Green = middle
        rgb[:, :360, 2] = self.binary_arrays[2]  # Blue = newest

        return rgb


def main():

    # Location of continuously growing JSON log file
    jsonl_file = Path.home() / "scan_logs" / "scan.jsonl"

    # Directory where RGB images will be written
    output_dir = Path.home() / "scan_logs" / "rgb_output"
    output_dir.mkdir(exist_ok=True)

    # Image that is continuously overwritten with the newest frame
    current_image = output_dir / "current_rgb.png"

    # Create rolling buffer of three scans
    buffer = SimpleRGBBuffer(buffer_size=3)

    # Count incoming scans
    frame_counter = 0

    # Count saved RGB images
    image_counter = 0

    print(f"Watching: {jsonl_file}")
    print(f"Writing live image: {current_image}")

    # Open log file
    with open(jsonl_file, "r") as f:

        # Move to the end of the file.
        #
        # This ignores old scans and waits only for newly
        # appended data (similar to "tail -f").
        f.seek(0, 2)

        while True:

            # Read one new line
            line = f.readline()

            # If no new data has arrived, wait briefly
            if not line:
                time.sleep(0.05)
                continue

            # Attempt to decode JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Ignore incomplete or corrupted lines
                continue

            # Skip entries that don't contain scan data
            if "ranges" not in data:
                continue

            ranges = data["ranges"]

            # Verify correct number of laser beams
            if len(ranges) != 360:
                print(
                    f"Skipping frame {frame_counter}: "
                    f"expected 360 beams, got {len(ranges)}"
                )
                continue

            # Convert scan into binary image and add to buffer
            buffer.add_frame(ranges)

            # Need at least three scans before RGB image exists
            if frame_counter >= 2:

                rgb = buffer.get_rgb_image()

                if rgb is not None:

                    # Continuously overwrite latest image
                    cv2.imwrite(str(current_image), rgb)

                    # Every 25th image, create a filename
                    # for permanent archiving.
                    if image_counter % 25 == 0:
                        archive_file = (
                            output_dir /
                            f"rgb_{image_counter:05d}.png"
                        )

                    # Save archive image.
                    #
                    # NOTE:
                    # Because archive_file is only updated every
                    # 25 images, this line repeatedly overwrites
                    # the same archive image between updates.
                    # If the intention is to save every frame,
                    # archive_file should be created every loop.
                    cv2.imwrite(str(archive_file), rgb)

                    image_counter += 1

                    # Print progress every 10 saved images
                    if image_counter % 10 == 0:
                        print(
                            f"Frames: {frame_counter} "
                            f" Images: {image_counter}"
                        )

            # Count processed scans
            frame_counter += 1


# Program entry point
if __name__ == "__main__":
    main()

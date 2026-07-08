#!/home/robot/yolo/venv/bin/python

# Import required libraries
import os                         # File operations
import math                       # Mathematical functions
import cv2                        # Image processing
from ultralytics import YOLO      # YOLO object detection model
import json                       # JSON file handling

# ROS2 libraries
import rclpy
from rclpy.node import Node

# Robot position messages
from nav_msgs.msg import Odometry

# Convert quaternion orientation into Euler angles
from tf_transformations import euler_from_quaternion

# ROS2 communication quality settings
from rclpy.qos import QoSProfile, ReliabilityPolicy


# ---------------------------------------------------------
# QoS configuration
# ---------------------------------------------------------

# BEST_EFFORT is suitable for odometry data because:
# - low latency is preferred
# - occasional dropped messages are acceptable
qos = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=10
)



class LidarDetector(Node):
    """
    ROS2 node that performs object detection on RGB-encoded
    LiDAR images using a trained YOLO model.

    The node performs the following tasks:

        1. Load a trained YOLO LiDAR model.
        2. Read the latest RGB LiDAR image.
        3. Detect objects using YOLO.
        4. Convert detections from image coordinates into
           robot-relative coordinates.
        5. Transform detections into global coordinates.
        6. Save observations and robot pose to JSONL.
        7. Save an annotated detection image.

    This node connects the perception system with a
    world-coordinate object representation.
    """


    def __init__(self):

        # Initialise ROS2 node.
        super().__init__("lidar_detector")


        # ---------------------------------------------------------
        # Load trained YOLO model
        # ---------------------------------------------------------

        # Load previously trained weights.
        #
        # The model has been trained using RGB representations
        # of LiDAR scans rather than normal camera images.
        self.model = YOLO(
            "/home/robot/2d-lidar-identification/training_outputs/"
            "120_DO_simple_Fused/webots_model_assets_simpel_fused_fulltrain_120/"
            "yolov8n_lidar.pt"
        )

        self.get_logger().info(
            "Model loaded successfully"
        )


        # ---------------------------------------------------------
        # Robot pose storage
        # ---------------------------------------------------------

        # Used to store the initial robot position.
        #
        # All detections are expressed relative to this
        # starting pose before being transformed globally.
        self.initialized = False

        self.x = None
        self.y = None
        self.theta = None

        self.x0 = None
        self.y0 = None
        self.theta0 = None



        # ---------------------------------------------------------
        # Odometry subscription
        # ---------------------------------------------------------

        # Receive robot position and orientation.
        self.create_subscription(
            Odometry,
            "/robot_2/odom",
            self.odom_callback,
            qos
        )


        # ---------------------------------------------------------
        # Detection timer
        # ---------------------------------------------------------

        # Run object detection every 0.5 seconds.
        #
        # Equivalent to 2 Hz detection frequency.
        self.timer = self.create_timer(
            0.5,
            self.run_detection
        )



    def yaw_from_quaternion(self, q):
        """
        Convert quaternion orientation into yaw angle.

        ROS stores orientation as a quaternion:
            (x, y, z, w)

        For a ground robot, only yaw (rotation around the
        vertical axis) is required.
        """

        _, _, yaw = euler_from_quaternion(
            [
                q.x,
                q.y,
                q.z,
                q.w
            ]
        )

        return yaw



    def odom_callback(self, msg):
        """
        Update current robot position and orientation.
        """

        # Current robot position.
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y


        # Current robot orientation.
        q = msg.pose.pose.orientation

        self.theta = self.yaw_from_quaternion(q)


        # Store initial robot pose only once.
        #
        # This creates the coordinate origin used for
        # relative detection mapping.
        if not self.initialized:

            self.x0 = self.x
            self.y0 = self.y
            self.theta0 = self.theta

            self.initialized = True



    def run_detection(self):
        """
        Main object detection pipeline.

        Converts detected objects from:

            image coordinates
                |
                v
            robot coordinates
                |
                v
            global coordinates

        and stores the resulting observations.
        """


        # Wait until initial robot pose exists.
        if not self.initialized:
            return



        # ---------------------------------------------------------
        # Load latest RGB LiDAR image
        # ---------------------------------------------------------

        rgb_image = cv2.imread(
            "/home/robot/scan_logs/rgb_output/current_rgb.png"
        )


        if rgb_image is None:

            self.get_logger().warn(
                "Waiting for image..."
            )

            return



        # OpenCV loads images as BGR.
        #
        # YOLO expects RGB ordering.
        rgb_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_BGR2RGB
        )



        # ---------------------------------------------------------
        # YOLO inference
        # ---------------------------------------------------------

        results = self.model(
            rgb_image,

            # Matches RGB LiDAR image dimensions.
            imgsz=[64, 384],

            # Run on CPU.
            device="cpu",

            # Disable half precision.
            half=False,

            # Disable verbose output.
            verbose=False,
        )



        detections = []


        # ---------------------------------------------------------
        # Process every detected object
        # ---------------------------------------------------------

        for box in results[0].boxes:


            # YOLO bounding box centre.
            #
            # cx = horizontal image position
            # cy = vertical image position
            cx, cy, _, _ = box.xywh[0].tolist()



            # -----------------------------------------------------
            # Convert image coordinates to LiDAR coordinates
            # -----------------------------------------------------

            # Image height represents distance.
            #
            # The LiDAR image maps:
            #
            # row 0   -> 0 m
            # row 64  -> 4 m
            #
            distance = (
                4 * cy
            ) / 64


            # Image centre corresponds to robot forward direction.
            #
            # Convert pixel angle into radians.
            angle = math.radians(
                cx - 180
            )


            # Convert polar coordinates into robot coordinates.
            #
            # xr = forward distance
            # yr = sideways distance
            xr = (
                distance *
                math.cos(angle)
            )

            yr = (
                distance *
                math.sin(angle)
            )



            # -----------------------------------------------------
            # Transform robot coordinates into global coordinates
            # -----------------------------------------------------

            # Calculate robot heading relative to start position.
            #
            # atan2(sin(),cos()) keeps angle within [-pi,pi].
            robot_theta = math.atan2(
                math.sin(self.theta - self.theta0),
                math.cos(self.theta - self.theta0)
            )



            # Rotate and translate detection position.
            x_global = (
                (self.x - self.x0)
                +
                xr * math.cos(robot_theta)
                -
                yr * math.sin(robot_theta)
            )


            y_global = (
                (self.y - self.y0)
                +
                xr * math.sin(robot_theta)
                +
                yr * math.cos(robot_theta)
            )



            # Store detection information.
            detections.append(
                {
                    "class_name":
                        results[0].names[
                            int(box.cls)
                        ],

                    "confidence":
                        float(box.conf),

                    "x":
                        x_global,

                    "y":
                        y_global
                }
            )



        # ---------------------------------------------------------
        # Save observation
        # ---------------------------------------------------------

        observation = {
            "robot_pose": None,
            "detections": detections
        }


        # Store robot pose relative to starting position.
        observation["robot_pose"] = {

            "x":
                self.x - self.x0,

            "y":
                self.y - self.y0,

            "theta":
                math.atan2(
                    math.sin(
                        self.theta - self.theta0
                    ),
                    math.cos(
                        self.theta - self.theta0
                    )
                )
        }



        # Append observation to JSON Lines file.
        #
        # Each line contains one complete observation.
        with open(
            "/home/robot/scan_logs/observations.jsonl",
            "a"
        ) as f:

            f.write(
                json.dumps(observation)
                +
                "\n"
            )



        # ---------------------------------------------------------
        # Save annotated detection image
        # ---------------------------------------------------------

        # Draw bounding boxes and labels.
        annotated = results[0].plot()



        # Temporary file prevents other programs from reading
        # partially written images.
        temp_file = (
            "/home/robot/scan_logs/"
            "latest_detection.tmp.jpg"
        )

        final_file = (
            "/home/robot/scan_logs/"
            "latest_detection.jpg"
        )


        cv2.imwrite(
            temp_file,
            annotated
        )


        # Atomic replacement.
        os.replace(
            temp_file,
            final_file
        )


        self.get_logger().info(
            f"{len(detections)} detections"
        )



def main(args=None):

    # Initialise ROS2.
    rclpy.init(args=args)


    # Create detector node.
    node = LidarDetector()


    try:

        # Keep node running.
        rclpy.spin(node)


    except KeyboardInterrupt:
        pass


    finally:

        # Clean shutdown.
        node.destroy_node()
        rclpy.shutdown()



# Program entry point.
if __name__ == "__main__":
    main()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
# from message_filters import Subscriber, ApproximateTimeSynchronizer
from ultralytics import YOLO
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point # PoseStamped를 다른 프레임으로 변환하기 위해 필요(호출되진 않지만, tf2_ros.Buffer의 transform 메서드에서 내부적으로 사용됨)
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time
from geometry_msgs.msg import Point
import threading
import time

MODEL_PATH = "/home/kim/test_ws/src/my_tb4_ex/resource/yolo_models/AMR_yolov8s_lr0.001.pt"
CONFIDENCE = 0.5

class TargetPosePublisher(Node):
    def __init__(self):
        super().__init__('target_pose_publisher')
        self.info('Initializing TargetPosePublisher node...')
        
        sim_time = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time])

        self.model = YOLO(MODEL_PATH)
        self.target_pose_publisher = self.create_publisher(PoseStamped, 'target_pose', 10)
        self.obstacle_pose_publisher = self.create_publisher(PoseStamped, 'obstacle_pose', 10)
        self.annotated_image_publisher = self.create_publisher(Image, 'tb4/annotated_image', 10)
        self.rgb_sub = self.create_subscription(CompressedImage, 'oakd/rgb/image_raw/compressed', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, 'oakd/stereo/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, 'oakd/rgb/camera_info', self.camera_info_callback, 10)
        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.rgb_img = None
        self.depth_img = None
        self.depth_frame_id = None

        # for PID control
        self.target_error_pub = self.create_publisher(Point, 'target_error', 10)

        thread = threading.Thread(target=self.main_process)
        self.event = threading.Event()
        thread.start()

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        if self.fx is not None and self.fy is not None and self.cx is not None and self.cy is not None:
            # fx=221.76500407999384, fy=221.76500407999382, cx=160.0, cy=120.0
            self.info(f'Camera intrinsic parameters received: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}')
            self.event.set()
            self.camera_info_sub.destroy()
    
    def rgb_callback(self, msg):
        self.rgb_img = self.bridge.compressed_imgmsg_to_cv2(msg)
    
    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame_id = msg.header.frame_id

    def main_process(self):
        self.info('Received synchronized RGB and depth images.')
        self.info('Waiting for camera intrinsic parameters...')
        self.event.wait()
        self.info('Starting main processing loop.')

        while rclpy.ok():
            if self.rgb_img is None or self.depth_img is None:
                self.warn('RGB or depth image not received yet, skipping this iteration.')
                time.sleep(0.1)
                continue

            results = self.model(self.rgb_img, conf=CONFIDENCE, verbose=False)
            annotated = results[0].plot()
            self.annotated_image_publisher.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))
            
            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    if int(box.cls[0]) == 0:
                        x_center = box.xywh[0][0]
                        y_center = box.xywh[0][1]
                        depth = self.depth_img[int(y_center), int(x_center)]
                        if depth == 0:
                            self.warn('Depth value is zero, skipping this detection.')
                            continue
                        x = (x_center - self.cx) * depth / self.fx
                        y = (y_center - self.cy) * depth / self.fy
                        z = depth
                        detected_pose = PoseStamped()
                        # detected_pose.header.stamp = self.get_clock().now().to_msg()
                        detected_pose.header.stamp = Time().to_msg()
                        detected_pose.header.frame_id = self.depth_frame_id
                        detected_pose.pose.position.x = float(x)
                        detected_pose.pose.position.y = float(y)
                        detected_pose.pose.position.z = float(z)

                        try:
                            detected_map_pose = self.tf_buffer.transform(
                                detected_pose,
                                'map',
                                timeout=Duration(seconds=1.0))
                        except Exception as e:
                            self.warn(f'TF transform failed: {e}')
                            continue
                        # self.info(f'Target pose in map frame: (x: {detected_map_pose.pose.position.x:.2f}, y: {detected_map_pose.pose.position.y:.2f}, z: {detected_map_pose.pose.position.z:.2f})')
                        
                        target_pose = PoseStamped()
                        target_pose.header = detected_map_pose.header
                        target_pose.pose.position.x = detected_map_pose.pose.position.x
                        target_pose.pose.position.y = detected_map_pose.pose.position.y
                        target_pose.pose.position.z = 0.0

                        self.target_pose_publisher.publish(target_pose)

                        # for PID control
                        target_error = Point()
                        target_error.x = float(x_center - self.cx)
                        target_error.z = float(z - 0.7)
                        self.info(f'Publishing target error: x_error={target_error.x:.2f}, z_error={target_error.z:.2f}')
                        self.target_error_pub.publish(target_error)
            time.sleep(0.1)

    def info(self, message):
        self.get_logger().info(message)
    
    def warn(self, message):
        self.get_logger().warn(message)

def main():
    rclpy.init()
    node = TargetPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
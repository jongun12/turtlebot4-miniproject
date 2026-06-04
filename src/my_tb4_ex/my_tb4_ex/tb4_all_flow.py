import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point # PoseStamped를 다른 프레임으로 변환하기 위해 필요(호출되진 않지만, tf2_ros.Buffer의 transform 메서드에서 내부적으로 사용됨)
from tf2_ros import Buffer, TransformListener
from rclpy.duration import Duration
from rclpy.parameter import Parameter
from rclpy.time import Time
from geometry_msgs.msg import Point
import threading
import time
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from my_tb4_ex.pid_controller import PID

MODEL_PATH = "/home/kim/test_ws/src/my_tb4_ex/resource/yolo_models/AMR_yolov8s_lr0.001.pt"
CONFIDENCE = 0.5
FIRST_GO_TO_POSE = (-3.0, -2.0, -90.0) # (x, y, yaw)
POSE_BEFORE_DOCK = (-0.5, 0.0, 0.0)

class TargetPosePublisher(Node):
    def __init__(self):
        super().__init__('target_pose_publisher')
        self.info('Initializing TargetPosePublisher node...')
        
        sim_time = Parameter('use_sim_time', Parameter.Type.BOOL, True)
        self.set_parameters([sim_time])

        self.tb4_navigator = TurtleBot4Navigator()
        initial_pose = self.tb4_navigator.getPoseStamped([0.0, 0.0], 0.0)
        self.tb4_navigator.setInitialPose(initial_pose)
        self.info(f'초기 위치 설정 중...')
        time.sleep(5.0)
        self.tb4_navigator.waitUntilNav2Active()

        if self.tb4_navigator.getDockedStatus():
            self.info('현재 도킹 상태 -> 언도킹 시도')
            self.tb4_navigator.undock()
        else:
            self.warn('초기 위치가 일치하지 않을 수 있습니다.')

        self.model = YOLO(MODEL_PATH)
        self.annotated_image_publisher = self.create_publisher(Image, 'tb4/annotated_image', 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)
        self.rgb_sub = self.create_subscription(CompressedImage, 'oakd/rgb/image_raw/compressed', self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, 'oakd/stereo/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, 'oakd/rgb/camera_info', self.camera_info_callback, 10)
        self.trigger_sub = self.create_subscription(Point, 'car_position', self.trigger_callback, 10)
        self.target_pose_publisher = self.create_publisher(PoseStamped, 'target_pose', 10)
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

        self.target_pose = None
        self.target_error = None

        self.counter = 0

        # for PID control
        self.target_error_pub = self.create_publisher(Point, 'target_error', 10)
        self.z_pid = PID(Kp=1.0, Ki=0.0, Kd=0.0)
        self.x_pid = PID(Kp=0.05, Ki=0.0, Kd=0.0)
        self.x_pid.max_error = 10.0
        self.x_pid.min_error = -10.0
        self.z_pid.min_error = 0.0

        # main_process
        # thread = threading.Thread(target=self.main_process)
        # self.event = threading.Event()
        # self.start_event = threading.Event()
        # self.end_event = threading.Event()
        # thread.start()

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
        if self.rgb_img is None or self.depth_img is None:
            self.warn('RGB or depth image not received yet, skipping this callback.')
            return
        
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
                    
                    self.target_pose = PoseStamped()
                    self.target_pose.header = detected_map_pose.header
                    self.target_pose.pose.position.x = detected_map_pose.pose.position.x
                    self.target_pose.pose.position.y = detected_map_pose.pose.position.y
                    self.target_pose.pose.position.z = 0.0

                    # for PID control
                    self.target_error = Point()
                    self.target_error.x = float(x_center - self.cx)
                    self.target_error.z = float(z - 0.7)
                    self.info(f'Publishing target error: x_error={self.target_error.x:.2f}, z_error={self.target_error.z:.2f}')
                    self.target_error_pub.publish(self.target_error)
        else:
            self.info('No target detected in the RGB image.')
            self.target_pose = PoseStamped()
            self.target_pose.pose.position.z = -1.0

        self.target_pose_publisher.publish(self.target_pose)

    
    def depth_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_frame_id = msg.header.frame_id
        results = self.model(self.rgb_img, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()
        self.annotated_image_publisher.publish(self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8'))
    
    def trigger_callback(self, msg):
        if msg.z != -1.0 and self.start_event.is_set() == False:
            self.counter += 1
            if self.counter >= 5:
                self.info('Received trigger to start processing.')
                self.start_event.set()
                self.counter = 0
            return
        else:
            self.counter = 0
        
        if self.start_event.is_set() == True and msg.z == -1.0:
            self.counter += 1
            if self.counter >= 5:
                self.info('Target lost for 5 consecutive triggers, stopping processing.')
                self.end_event.set()
                self.counter = 0
            return
        else:
            self.counter = 0

    def main_process(self):
        # self.start_event.wait()
        self.info('Waiting for camera intrinsic parameters...')
        # self.event.wait()
        self.info('Starting main processing loop.')

        goal_pose = self.tb4_navigator.getPoseStamped([FIRST_GO_TO_POSE[0], FIRST_GO_TO_POSE[1]], FIRST_GO_TO_POSE[2])
        self.tb4_navigator.goToPose(goal_pose)

        while not self.tb4_navigator.isTaskComplete():
            feedback = self.tb4_navigator.getFeedback()
            if feedback:
                remaining = feedback.distance_remaining
                self.tb4_navigator.get_logger().info(f'남은 거리: {remaining:.2f}m')
            time.sleep(0.1)

        result = self.tb4_navigator.getResult()
        if result == 1:
            self.tb4_navigator.get_logger().info('목표 위치 도달 성공')
        elif result == 2:
            self.tb4_navigator.get_logger().warn('이동이 취소되었습니다.')
        elif result == 3:
            self.tb4_navigator.get_logger().error(f'이동 실패')
        else:
            self.tb4_navigator.get_logger().warn('알 수 없는 결과 코드')
        ###########################################################################################################################
        # start_time = self.get_clock().now()
        # while self.target_pose.pose.position.z == -1.0 or self.target_pose is None: # Check if target is detected
        #     if (self.get_clock().now() - start_time).sec > 10.0:
        #         self.warn('No target detected within 10 seconds, go back to initial position.')
        #     self.info('No detection yet, Rotating in place to search for target...')
        #     cmd_vel = self.get_cmd_vel(0.0, 1.0)
        #     self.cmd_vel_publisher.publish(cmd_vel)
        #     time.sleep(0.1)

        # self.info(f'Target detected at (x: {self.target_pose.pose.position.x:.2f}, y: {self.target_pose.pose.position.y:.2f}, z: {self.target_pose.pose.position.z:.2f})')
        # self.tb4_navigator.goToPose(self.target_pose)
        # while not self.tb4_navigator.isTaskComplete():
        #     feedback = self.tb4_navigator.getFeedback()
        #     if feedback:
        #         remaining = feedback.distance_remaining
        #         self.tb4_navigator.get_logger().info(f'남은 거리: {remaining:.2f}m')
        #     time.sleep(0.1)
        # result = self.tb4_navigator.getResult()
        # if result == 1:
        #     self.tb4_navigator.get_logger().info('목표 위치 도달 성공')
        # elif result == 2:
        #     self.tb4_navigator.get_logger().warn('이동이 취소되었습니다.')
        # elif result == 3:
        #     self.tb4_navigator.get_logger().error(f'이동 실패')
        # else:
        #     self.tb4_navigator.get_logger().warn('알 수 없는 결과 코드')
        ###########################################################################################################################
        # while not self.end_event.is_set():
        #     z_control = self.z_pid.update(self.target_error.z)
        #     x_control = self.x_pid.update(self.target_error.x)
        #     cmd_vel = self.get_cmd_vel(z_control, -x_control)
        #     self.cmd_vel_publisher.publish(cmd_vel)
        #     time.sleep(0.1)
        ###########################################################################################################################
        # self.info('Moving to pose before docking...')
        # goal_pose = self.tb4_navigator.getPoseStamped([POSE_BEFORE_DOCK[0], POSE_BEFORE_DOCK[1]], POSE_BEFORE_DOCK[2])
        # self.tb4_navigator.goToPose(goal_pose)

        # while not self.tb4_navigator.isTaskComplete():
        #     feedback = self.tb4_navigator.getFeedback()
        #     if feedback:
        #         remaining = feedback.distance_remaining
        #         self.tb4_navigator.get_logger().info(f'남은 거리: {remaining:.2f}m')
        #     time.sleep(0.1)

        # result = self.tb4_navigator.getResult()
        # if result == 1:
        #     self.tb4_navigator.get_logger().info('목표 위치 도달 성공')
        # elif result == 2:
        #     self.tb4_navigator.get_logger().warn('이동이 취소되었습니다.')
        # elif result == 3:
        #     self.tb4_navigator.get_logger().error(f'이동 실패')
        # else:
        #     self.tb4_navigator.get_logger().warn('알 수 없는 결과 코드')
        ###########################################################################################################################
        # self.info('Attempting to dock...')
        # self.tb4_navigator.dock()
        ###########################################################################################################################

    def get_cmd_vel(self, x, z):
        cmd_vel = Twist()
        cmd_vel.linear.x = x
        cmd_vel.angular.z = z
        return cmd_vel


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
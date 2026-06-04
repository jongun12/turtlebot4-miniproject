import cv2
from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

MODEL_PATH = "/home/kim/test_ws/src/turtlebot4_sing/resource/yolo_models/20260601_v2_yolov8n_lr0.001.pt"
CAMERA_INDEX = 4
CONFIDENCE = 0.25

class YoloCamPublisher(Node):
    def __init__(self):
        super().__init__("yolo_cam_publisher")
        self.detection_img_publisher = self.create_publisher(Image, "yolo_cam/detection_image", 10)
        self.pose_publisher = self.create_publisher(Point, "car_position", 10)
        self.bridge = CvBridge()

        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.publish_detection_image)

        self.model = YOLO(MODEL_PATH)

        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def publish_detection_image(self):
        if not self.cap.isOpened():
            self.get_logger().error("Cannot open camera")
            return
        
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().error("Cannot read camera frame")
            return
        
        results = self.model.predict(frame, conf=CONFIDENCE, verbose=False)
        annotated = results[0].plot()
        point_msg = Point()  # Initialize point_msg outside the loop
        point_msg.z = -1.0
        if results[0].boxes is not None:
            for box in results[0].boxes:
                if box.cls[0] == 0: # Assuming class 0 is the car
                    x_center = int(box.xywh[0][0])
                    y_center = int(box.xywh[0][1])
                    point_msg = Point()
                    point_msg.x = float(x_center)
                    point_msg.y = float(y_center)
                    point_msg.z = 0.0
        self.pose_publisher.publish(point_msg)

        img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        self.detection_img_publisher.publish(img_msg)

def main():
    rclpy.init()
    yolo_cam_publisher = YoloCamPublisher()
    try:
        rclpy.spin(yolo_cam_publisher)
    except KeyboardInterrupt:
        pass
    yolo_cam_publisher.cap.release()
    yolo_cam_publisher.destroy_node()
    rclpy.shutdown()
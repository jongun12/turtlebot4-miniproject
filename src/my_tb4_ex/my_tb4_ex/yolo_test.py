import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO

class YoloTestNode(Node):
    def __init__(self):
        super().__init__('yolo_test_node')
        self.get_logger().info('Initializing YOLO Test Node...')
        self.model = YOLO('yolov8n.pt')
        self.image_pub = self.create_publisher(Image, 'tb4/yolo_test/annotated_image', 10)
        self.image_sub = self.create_subscription(
            Image,
            'oakd/rgb/preview/image_raw',
            self.image_callback,
            10
        )
        self.bridge = CvBridge()

    
    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model(cv_image)
        annotated = results[0].plot()
        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        self.image_pub.publish(annotated_msg)
    
def main():
    rclpy.init()
    yolo_test_node = YoloTestNode()
    try:
        rclpy.spin(yolo_test_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_test_node.destroy_node()
        rclpy.shutdown()
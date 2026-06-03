import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from my_tb4_ex.pid_controller import PID

class FollowPIDNode(Node):
    def __init__(self):
        super().__init__('follow_pid_node')
        self.get_logger().info('Initializing FollowPIDNode...')
        self.z_pid = PID(Kp=1.0, Ki=0.0, Kd=0.0)
        self.x_pid = PID(Kp=0.05, Ki=0.0, Kd=0.0)
        self.x_pid.max_error = 10.0
        self.x_pid.min_error = -10.0
        self.z_pid.min_error = 0.0
        self.target_error_sub = self.create_subscription(Point, 'target_error', self.error_callback, 10)
        self.cmd_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

    def error_callback(self, msg):
        z_error = msg.z
        x_error = msg.x

        z_control = self.z_pid.update(z_error)
        x_control = self.x_pid.update(x_error)

        cmd_msg = Twist()
        cmd_msg.linear.x = z_control
        cmd_msg.angular.z = -x_control
        self.cmd_publisher.publish(cmd_msg)

def main():
    rclpy.init()
    follow_pid_node = FollowPIDNode()
    try:
        rclpy.spin(follow_pid_node)
    except KeyboardInterrupt:
        pass
    finally:
        follow_pid_node.destroy_node()
        rclpy.shutdown()
import rclpy
from rclpy.node import Node
from nav2_simple_commander.robot_navigator import TaskResult
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator
from builtin_interfaces.msg import Time
import time

def main():
    rclpy.init()

    tb4_navigator = TurtleBot4Navigator()

    initial_pose = tb4_navigator.getPoseStamped([0.0, 0.0], 0.0)
    tb4_navigator.setInitialPose(initial_pose)
    tb4_navigator.get_logger().info(f'초기 위치 설정 중...')
    time.sleep(5.0)
    tb4_navigator.waitUntilNav2Active()

    if tb4_navigator.getDockedStatus():
        tb4_navigator.get_logger().info('현재 도킹 상태 -> 언도킹 시도')
        tb4_navigator.undock()
    else:
        tb4_navigator.get_logger().info('언도킹 상태에서 시작함')
    
    goal_pose = tb4_navigator.getPoseStamped([-3.0, -2.0], -90.0)
    tb4_navigator.goToPose(goal_pose)

    while not tb4_navigator.isTaskComplete():
        feedback = tb4_navigator.getFeedback()
        if feedback:
            remaining = feedback.distance_remaining
            tb4_navigator.get_logger().info(f'남은 거리: {remaining:.2f}m')
    
    result = tb4_navigator.getResult()
    if result == TaskResult.SUCCEEDED:
        tb4_navigator.get_logger().info('목표 위치 도달 성공')
    elif result == TaskResult.CANCELED:
        tb4_navigator.get_logger().warn('이동이 취소되었습니다.')
    elif result == TaskResult.FAILED:
        tb4_navigator.get_logger().error(f'이동 실패')
    else:
        tb4_navigator.get_logger().warn('알 수 없는 결과 코드')
    
    tb4_navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

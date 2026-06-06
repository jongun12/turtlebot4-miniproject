"""Simple TurtleBot4 coverage navigation example."""

import math
import time

import rclpy
from nav2_simple_commander.robot_navigator import TaskResult
from nav_msgs.msg import OccupancyGrid
from rescue_interfaces.srv import SetCoverageMode
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from turtlebot4_navigation.turtlebot4_navigator import TurtleBot4Navigator


class SimpleCoverageNavigator(TurtleBot4Navigator):
    """Pick free map points and send them to TurtleBot4Navigator."""

    VALID_MODES = ['idle', 'explore', 'coverage', 'paused']

    def __init__(self):
        super().__init__()

        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('initial_mode', 'idle')
        self.declare_parameter('planning_period_sec', 1.0)
        self.declare_parameter('coverage_spacing_m', 0.65)
        self.declare_parameter('visited_radius_m', 0.40)
        self.declare_parameter('goal_clearance_m', 0.25)
        self.declare_parameter('min_goal_distance_m', 0.45)
        self.declare_parameter('free_threshold', 25)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('frontier_bonus_m', 1.25)

        self.robot_frame = self.get_parameter('robot_frame').value
        self.mode = self.get_parameter('initial_mode').value
        self.planning_period_sec = self.get_parameter('planning_period_sec').value
        self.coverage_spacing_m = self.get_parameter('coverage_spacing_m').value
        self.visited_radius_m = self.get_parameter('visited_radius_m').value
        self.goal_clearance_m = self.get_parameter('goal_clearance_m').value
        self.min_goal_distance_m = self.get_parameter('min_goal_distance_m').value
        self.free_threshold = self.get_parameter('free_threshold').value
        self.occupied_threshold = self.get_parameter('occupied_threshold').value
        self.frontier_bonus_m = self.get_parameter('frontier_bonus_m').value

        if self.mode not in self.VALID_MODES:
            self.mode = 'idle'

        self.map = None
        self.goal_active = False
        self.current_goal = None
        self.visited_points = []
        self.failed_points = []

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_topic = self.get_parameter('map_topic').value
        self.create_subscription(OccupancyGrid, map_topic, self.save_map, 10)
        self.create_service(SetCoverageMode, 'coverage/set_mode', self.set_mode)

        self.get_logger().info('coverage node ready')
        self.get_logger().info('set mode to coverage or explore to start')

    def save_map(self, msg):
        self.map = msg

    def set_mode(self, request, response):
        if request.mode not in self.VALID_MODES:
            response.accepted = False
            response.message = 'mode must be one of: ' + str(self.VALID_MODES)
            return response

        self.mode = request.mode
        response.accepted = True
        response.message = 'coverage mode changed to ' + self.mode
        return response

    def tick(self):
        if self.map is None:
            self.get_logger().info('waiting for map')
            return

        robot_xy = self.get_robot_xy()
        if robot_xy is None:
            self.get_logger().info('waiting for robot TF')
            return

        self.remember(self.visited_points, robot_xy)

        if self.mode in ['idle', 'paused']:
            return

        if self.goal_active:
            self.check_goal_result()
            return

        next_goal = self.find_next_goal(robot_xy)
        if next_goal is None:
            self.get_logger().info('coverage complete: no safe unvisited goal')
            self.mode = 'idle'
            return

        self.send_goal(next_goal, robot_xy)

    def check_goal_result(self):
        if not self.isTaskComplete():
            return

        result = self.getResult()
        goal_xy = [
            self.current_goal.pose.position.x,
            self.current_goal.pose.position.y,
        ]

        if result == TaskResult.SUCCEEDED:
            self.get_logger().info('goal reached')
            self.remember(self.visited_points, goal_xy)
        else:
            self.get_logger().warn('goal failed')
            self.remember(self.failed_points, goal_xy)

        self.goal_active = False
        self.current_goal = None

    def send_goal(self, goal_xy, robot_xy):
        dx = goal_xy[0] - robot_xy[0]
        dy = goal_xy[1] - robot_xy[1]
        yaw_deg = math.degrees(math.atan2(dy, dx))

        pose = self.getPoseStamped([goal_xy[0], goal_xy[1]], yaw_deg)
        self.current_goal = pose
        self.goal_active = True

        self.get_logger().info(
            'go to x={:.2f}, y={:.2f}, yaw={:.1f}'.format(
                goal_xy[0], goal_xy[1], yaw_deg
            )
        )
        self.goToPose(pose)

    def find_next_goal(self, robot_xy):
        grid = self.map
        resolution = grid.info.resolution
        step = max(1, round(self.coverage_spacing_m / resolution))
        clearance = max(1, math.ceil(self.goal_clearance_m / resolution))

        best_goal = None
        best_score = None

        for map_y in range(step // 2, grid.info.height, step):
            for map_x in range(step // 2, grid.info.width, step):
                if not self.is_free(map_x, map_y):
                    continue
                if not self.has_clearance(map_x, map_y, clearance):
                    continue

                goal_xy = self.map_to_world(map_x, map_y)
                distance = self.distance(robot_xy, goal_xy)

                if distance < self.min_goal_distance_m:
                    continue
                if self.is_near(self.visited_points, goal_xy, self.visited_radius_m):
                    continue
                if self.is_near(self.failed_points, goal_xy, self.visited_radius_m):
                    continue

                score = distance
                if self.is_frontier(map_x, map_y):
                    score = score - self.frontier_bonus_m
                    if self.mode == 'explore':
                        score = score - self.frontier_bonus_m

                if best_score is None or score < best_score:
                    best_score = score
                    best_goal = goal_xy

        return best_goal

    def get_robot_xy(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                'map',
                self.robot_frame,
                Time(),
            )
        except TransformException as error:
            self.get_logger().warn(str(error))
            return None

        translation = transform.transform.translation
        return [translation.x, translation.y]

    def is_free(self, map_x, map_y):
        value = self.map.data[self.index(map_x, map_y)]
        return 0 <= value <= self.free_threshold

    def has_clearance(self, map_x, map_y, radius):
        min_x = max(0, map_x - radius)
        max_x = min(self.map.info.width - 1, map_x + radius)
        min_y = max(0, map_y - radius)
        max_y = min(self.map.info.height - 1, map_y + radius)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                value = self.map.data[self.index(x, y)]
                if value >= self.occupied_threshold:
                    return False
        return True

    def is_frontier(self, map_x, map_y):
        for y in range(max(0, map_y - 1), min(self.map.info.height, map_y + 2)):
            for x in range(max(0, map_x - 1), min(self.map.info.width, map_x + 2)):
                if self.map.data[self.index(x, y)] == -1:
                    return True
        return False

    def map_to_world(self, map_x, map_y):
        origin = self.map.info.origin.position
        resolution = self.map.info.resolution
        world_x = origin.x + (map_x + 0.5) * resolution
        world_y = origin.y + (map_y + 0.5) * resolution
        return [world_x, world_y]

    def index(self, map_x, map_y):
        return map_y * self.map.info.width + map_x

    def remember(self, point_list, point):
        if not self.is_near(point_list, point, self.visited_radius_m * 0.5):
            point_list.append(point)

    def is_near(self, point_list, point, radius):
        for saved_point in point_list:
            if self.distance(saved_point, point) <= radius:
                return True
        return False

    def distance(self, point_a, point_b):
        dx = point_a[0] - point_b[0]
        dy = point_a[1] - point_b[1]
        return math.sqrt(dx * dx + dy * dy)


def main(args=None):
    rclpy.init(args=args)
    navigator = SimpleCoverageNavigator()

    navigator.waitUntilNav2Active()

    try:
        while rclpy.ok():
            rclpy.spin_once(navigator, timeout_sec=0.1)
            navigator.tick()
            time.sleep(navigator.planning_period_sec)
    except KeyboardInterrupt:
        pass
    finally:
        navigator.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

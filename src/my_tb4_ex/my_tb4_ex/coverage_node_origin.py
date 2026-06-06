"""Online coverage planner node for TurtleBot navigation."""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point, PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from rescue_interfaces.msg import CoverageStatus
from rescue_interfaces.srv import SetCoverageMode
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class CoverageCandidate:
    """A sampled map cell that can become a Nav2 goal."""

    map_x: int
    map_y: int
    world_x: float
    world_y: float
    distance: float
    is_frontier: bool


class CoveragePlannerNode(Node):
    """Generate coverage goals from a live SLAM map and send them to Nav2."""

    ACTIVE_MODES = {'explore', 'coverage'}
    VALID_MODES = {'idle', 'explore', 'coverage', 'paused', 'revisit'}

    def __init__(self):
        """Create publishers, subscribers, services, TF, and Nav2 action client."""
        super().__init__('coverage_planner')

        self.declare_parameter('map_topic', 'map')
        self.declare_parameter('status_topic', 'coverage/status')
        self.declare_parameter('path_topic', 'coverage/path')
        self.declare_parameter('marker_topic', 'coverage/markers')
        self.declare_parameter('set_mode_service', 'coverage/set_mode')
        self.declare_parameter('nav_to_pose_action', 'navigate_to_pose')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', 'base_link')
        self.declare_parameter('planning_period_sec', 1.0)
        self.declare_parameter('initial_mode', 'idle')
        self.declare_parameter('coverage_spacing_m', 0.65)
        self.declare_parameter('visited_radius_m', 0.40)
        self.declare_parameter('goal_clearance_m', 0.25)
        self.declare_parameter('min_goal_distance_m', 0.45)
        self.declare_parameter('free_threshold', 25)
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('prefer_frontiers', True)
        self.declare_parameter('frontier_weight_m', 1.25)
        self.declare_parameter('max_marker_points', 160)
        self.declare_parameter('max_candidates', 5000)

        self.map_frame = str(self.get_parameter('map_frame').value)
        self.robot_frame = str(self.get_parameter('robot_frame').value)
        self.mode = str(self.get_parameter('initial_mode').value)
        if self.mode not in self.VALID_MODES:
            self.get_logger().warn(
                f'Unknown initial_mode "{self.mode}", falling back to idle.'
            )
            self.mode = 'idle'

        self.coverage_spacing_m = float(
            self.get_parameter('coverage_spacing_m').value
        )
        self.visited_radius_m = float(
            self.get_parameter('visited_radius_m').value
        )
        self.goal_clearance_m = float(
            self.get_parameter('goal_clearance_m').value
        )
        self.min_goal_distance_m = float(
            self.get_parameter('min_goal_distance_m').value
        )
        self.free_threshold = int(self.get_parameter('free_threshold').value)
        self.occupied_threshold = int(
            self.get_parameter('occupied_threshold').value
        )
        self.prefer_frontiers = bool(
            self.get_parameter('prefer_frontiers').value
        )
        self.frontier_weight_m = float(
            self.get_parameter('frontier_weight_m').value
        )
        self.max_marker_points = int(self.get_parameter('max_marker_points').value)
        self.max_candidates = int(self.get_parameter('max_candidates').value)

        self.last_map: Optional[OccupancyGrid] = None
        self.current_goal = PoseStamped()
        self.current_goal.header.frame_id = self.map_frame
        self.goal_in_progress = False
        self.goal_handle = None

        self.visited_points: List[Tuple[float, float]] = []
        self.failed_points: List[Tuple[float, float]] = []
        self.last_candidates: List[CoverageCandidate] = []
        self.last_state = 'starting'
        self.last_message = 'Waiting for timer.'

        self.tf_buffer = Buffer(cache_time=Duration(seconds=20.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        map_topic = str(self.get_parameter('map_topic').value)
        status_topic = str(self.get_parameter('status_topic').value)
        path_topic = str(self.get_parameter('path_topic').value)
        marker_topic = str(self.get_parameter('marker_topic').value)
        set_mode_service = str(self.get_parameter('set_mode_service').value)
        nav_action_name = str(self.get_parameter('nav_to_pose_action').value)
        planning_period = float(self.get_parameter('planning_period_sec').value)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            map_topic,
            self._on_map,
            10,
        )
        self.status_pub = self.create_publisher(CoverageStatus, status_topic, 10)
        self.path_pub = self.create_publisher(Path, path_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.mode_srv = self.create_service(
            SetCoverageMode,
            set_mode_service,
            self._set_mode,
        )
        self.nav_client = ActionClient(self, NavigateToPose, nav_action_name)
        self.timer = self.create_timer(planning_period, self._tick)

        self.get_logger().info(
            'coverage_planner ready. Set mode to coverage/explore to send goals.'
        )

    def _on_map(self, msg: OccupancyGrid) -> None:
        """Store the newest live SLAM map."""
        self.last_map = msg

    def _set_mode(self, request, response):
        """Switch planner mode through a stable team service."""
        if request.mode not in self.VALID_MODES:
            response.accepted = False
            response.message = (
                f'Invalid mode "{request.mode}". '
                f'Use one of {sorted(self.VALID_MODES)}.'
            )
            return response

        self.mode = request.mode
        response.accepted = True
        response.message = f'Coverage mode changed to {self.mode}.'
        if request.pause_after_current:
            response.message += ' Current goal will finish before new planning.'
        return response

    def _tick(self) -> None:
        """Run one planning cycle and publish debug outputs."""
        self._update_planner()
        self._publish_status()
        self._publish_path()
        self._publish_markers()

    def _update_planner(self) -> None:
        """Update map coverage state and send a new Nav2 goal when appropriate."""
        if self.last_map is None:
            self._set_state('waiting_for_map', 'Waiting for SLAM map.')
            return

        robot_xy = self._lookup_robot_xy()
        if robot_xy is None:
            self._set_state(
                'waiting_for_tf',
                f'Waiting for TF {self.map_frame}->{self.robot_frame}.',
            )
            return

        self._remember_point(self.visited_points, robot_xy)

        if self.mode in {'idle', 'paused'}:
            self._set_state(self.mode, f'Planner mode is {self.mode}.')
            return

        if self.mode == 'revisit':
            self._set_state(
                'revisit_waiting',
                'Revisit mode is reserved for DB-provided survivor goals.',
            )
            return

        self.last_candidates = self._build_candidates(robot_xy)
        if self.goal_in_progress:
            self._set_state('navigating', 'Waiting for current Nav2 goal result.')
            return

        if not self._nav2_ready():
            self._set_state('waiting_for_nav2', 'Waiting for NavigateToPose server.')
            return

        candidate = self._select_candidate()
        if candidate is None:
            self._set_state('complete', 'No unvisited safe coverage candidate left.')
            return

        goal_pose = self._make_goal_pose(candidate, robot_xy)
        self._send_nav_goal(goal_pose)

    def _lookup_robot_xy(self) -> Optional[Tuple[float, float]]:
        """Return robot position in the map frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                Time(),
            )
        except TransformException as exc:
            self.last_message = str(exc)
            return None

        translation = transform.transform.translation
        return translation.x, translation.y

    def _build_candidates(
        self,
        robot_xy: Tuple[float, float],
    ) -> List[CoverageCandidate]:
        """Sample safe free-space cells from the current occupancy grid."""
        grid = self.last_map
        if grid is None or grid.info.resolution <= 0.0:
            return []

        width = grid.info.width
        height = grid.info.height
        resolution = grid.info.resolution
        step = max(1, int(round(self.coverage_spacing_m / resolution)))
        clearance_cells = max(1, int(math.ceil(self.goal_clearance_m / resolution)))

        candidates: List[CoverageCandidate] = []
        for map_y in range(step // 2, height, step):
            for map_x in range(step // 2, width, step):
                if len(candidates) >= self.max_candidates:
                    return candidates
                if not self._is_free(grid, map_x, map_y):
                    continue
                if not self._has_clearance(grid, map_x, map_y, clearance_cells):
                    continue

                world_x, world_y = self._map_to_world(grid, map_x, map_y)
                distance = self._distance(robot_xy, (world_x, world_y))
                if distance < self.min_goal_distance_m:
                    continue
                if self._near_any(self.visited_points, world_x, world_y,
                                  self.visited_radius_m):
                    continue
                if self._near_any(self.failed_points, world_x, world_y,
                                  self.visited_radius_m):
                    continue

                candidates.append(
                    CoverageCandidate(
                        map_x=map_x,
                        map_y=map_y,
                        world_x=world_x,
                        world_y=world_y,
                        distance=distance,
                        is_frontier=self._is_frontier(grid, map_x, map_y),
                    )
                )

        return candidates

    def _select_candidate(self) -> Optional[CoverageCandidate]:
        """Pick the best next coverage candidate."""
        if not self.last_candidates:
            return None

        def score(candidate: CoverageCandidate) -> float:
            frontier_bonus = self.frontier_weight_m if candidate.is_frontier else 0.0
            if not self.prefer_frontiers:
                frontier_bonus = 0.0
            if self.mode == 'explore':
                frontier_bonus *= 1.5
            return candidate.distance - frontier_bonus

        return min(self.last_candidates, key=score)

    def _send_nav_goal(self, pose: PoseStamped) -> None:
        """Send the selected goal to Nav2."""
        goal = NavigateToPose.Goal()
        goal.pose = pose

        self.current_goal = pose
        self.goal_in_progress = True
        self._set_state('sending_goal', 'Sending coverage goal to Nav2.')

        future = self.nav_client.send_goal_async(goal)
        future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        """Handle Nav2 goal acceptance."""
        try:
            self.goal_handle = future.result()
        except Exception as exc:  # pragma: no cover - defensive callback guard
            self.goal_in_progress = False
            self._remember_pose(self.failed_points, self.current_goal)
            self._set_state('goal_error', f'Goal send failed: {exc}')
            return

        if not self.goal_handle.accepted:
            self.goal_in_progress = False
            self._remember_pose(self.failed_points, self.current_goal)
            self._set_state('goal_rejected', 'Nav2 rejected coverage goal.')
            return

        result_future = self.goal_handle.get_result_async()
        result_future.add_done_callback(self._on_goal_result)
        self._set_state('navigating', 'Nav2 accepted coverage goal.')

    def _on_goal_result(self, future) -> None:
        """Handle Nav2 goal completion."""
        try:
            result = future.result()
        except Exception as exc:  # pragma: no cover - defensive callback guard
            self.goal_in_progress = False
            self._remember_pose(self.failed_points, self.current_goal)
            self._set_state('goal_error', f'Goal result failed: {exc}')
            return

        self.goal_in_progress = False
        self.goal_handle = None

        if result.status == GoalStatus.STATUS_SUCCEEDED:
            self._remember_pose(self.visited_points, self.current_goal)
            self._set_state('goal_succeeded', 'Coverage goal reached.')
        else:
            self._remember_pose(self.failed_points, self.current_goal)
            self._set_state('goal_failed', f'Nav2 goal ended with status {result.status}.')

    def _nav2_ready(self) -> bool:
        """Return whether the NavigateToPose action server is ready."""
        if self.nav_client.server_is_ready():
            return True
        return self.nav_client.wait_for_server(timeout_sec=0.1)

    def _make_goal_pose(
        self,
        candidate: CoverageCandidate,
        robot_xy: Tuple[float, float],
    ) -> PoseStamped:
        """Create a map-frame Nav2 goal pose for a candidate."""
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = candidate.world_x
        pose.pose.position.y = candidate.world_y

        yaw = math.atan2(candidate.world_y - robot_xy[1],
                         candidate.world_x - robot_xy[0])
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        return pose

    def _publish_status(self) -> None:
        """Publish current planner status."""
        total_goals = len(self.visited_points) + len(self.last_candidates)
        visited_goals = len(self.visited_points)

        status = CoverageStatus()
        status.header.stamp = self.get_clock().now().to_msg()
        status.header.frame_id = self.map_frame
        status.mode = self.mode
        status.state = self.last_state
        status.total_goals = total_goals
        status.visited_goals = visited_goals
        status.coverage_ratio = (
            float(visited_goals) / float(total_goals)
            if total_goals > 0 else 0.0
        )
        status.current_goal = self.current_goal
        status.message = self.last_message
        self.status_pub.publish(status)

    def _publish_path(self) -> None:
        """Publish the active goal as a simple debug path."""
        path = Path()
        path.header.stamp = self.get_clock().now().to_msg()
        path.header.frame_id = self.map_frame
        if self.goal_in_progress:
            path.poses.append(self.current_goal)
        self.path_pub.publish(path)

    def _publish_markers(self) -> None:
        """Publish RViz markers for candidates, visited points, and current goal."""
        markers = MarkerArray()
        markers.markers.append(self._delete_all_marker())
        markers.markers.append(
            self._points_marker(
                marker_id=1,
                ns='coverage_candidates',
                points=[
                    (candidate.world_x, candidate.world_y)
                    for candidate in self.last_candidates
                    if not candidate.is_frontier
                ][:self.max_marker_points],
                rgb=(0.1, 0.4, 1.0),
                scale=0.06,
            )
        )
        markers.markers.append(
            self._points_marker(
                marker_id=2,
                ns='coverage_frontiers',
                points=[
                    (candidate.world_x, candidate.world_y)
                    for candidate in self.last_candidates
                    if candidate.is_frontier
                ][:self.max_marker_points],
                rgb=(0.1, 0.9, 0.25),
                scale=0.08,
            )
        )
        markers.markers.append(
            self._points_marker(
                marker_id=3,
                ns='coverage_visited',
                points=self.visited_points[-self.max_marker_points:],
                rgb=(0.65, 0.65, 0.65),
                scale=0.05,
            )
        )
        if self.goal_in_progress:
            markers.markers.append(self._current_goal_marker())
        self.marker_pub.publish(markers)

    def _delete_all_marker(self) -> Marker:
        """Return a marker that clears previous visualization output."""
        marker = Marker()
        marker.action = Marker.DELETEALL
        return marker

    def _points_marker(
        self,
        marker_id: int,
        ns: str,
        points: List[Tuple[float, float]],
        rgb: Tuple[float, float, float],
        scale: float,
    ) -> Marker:
        """Create a POINTS marker."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = ns
        marker.id = marker_id
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.scale.x = scale
        marker.scale.y = scale
        marker.color.r = rgb[0]
        marker.color.g = rgb[1]
        marker.color.b = rgb[2]
        marker.color.a = 0.9
        marker.points = [self._point(x, y) for x, y in points]
        return marker

    def _current_goal_marker(self) -> Marker:
        """Create a SPHERE marker for the active Nav2 goal."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.map_frame
        marker.ns = 'coverage_current_goal'
        marker.id = 4
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose = self.current_goal.pose
        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.08
        marker.color.r = 1.0
        marker.color.g = 0.25
        marker.color.b = 0.15
        marker.color.a = 0.95
        return marker

    def _point(self, x: float, y: float) -> Point:
        """Create a geometry point in the map plane."""
        point = Point()
        point.x = x
        point.y = y
        return point

    def _set_state(self, state: str, message: str) -> None:
        """Store the latest planner state and message."""
        self.last_state = state
        self.last_message = message

    def _remember_pose(self, store: List[Tuple[float, float]], pose: PoseStamped) -> None:
        """Remember a pose as visited or failed."""
        self._remember_point(store, (pose.pose.position.x, pose.pose.position.y))

    def _remember_point(
        self,
        store: List[Tuple[float, float]],
        point: Tuple[float, float],
    ) -> None:
        """Append a point unless a nearby one is already stored."""
        if not self._near_any(store, point[0], point[1], self.visited_radius_m * 0.5):
            store.append(point)

    def _is_free(self, grid: OccupancyGrid, map_x: int, map_y: int) -> bool:
        """Return whether the cell is known free space."""
        value = grid.data[self._index(grid, map_x, map_y)]
        return value >= 0 and value <= self.free_threshold

    def _has_clearance(
        self,
        grid: OccupancyGrid,
        map_x: int,
        map_y: int,
        radius_cells: int,
    ) -> bool:
        """Return whether no occupied cell is near the candidate."""
        min_x = max(0, map_x - radius_cells)
        max_x = min(grid.info.width - 1, map_x + radius_cells)
        min_y = max(0, map_y - radius_cells)
        max_y = min(grid.info.height - 1, map_y + radius_cells)

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                value = grid.data[self._index(grid, x, y)]
                if value >= self.occupied_threshold:
                    return False
        return True

    def _is_frontier(self, grid: OccupancyGrid, map_x: int, map_y: int) -> bool:
        """Return whether a free cell touches unknown space."""
        for y in range(max(0, map_y - 1), min(grid.info.height, map_y + 2)):
            for x in range(max(0, map_x - 1), min(grid.info.width, map_x + 2)):
                if grid.data[self._index(grid, x, y)] == -1:
                    return True
        return False

    def _map_to_world(
        self,
        grid: OccupancyGrid,
        map_x: int,
        map_y: int,
    ) -> Tuple[float, float]:
        """Convert occupancy-grid cell coordinates to map-frame coordinates."""
        origin = grid.info.origin.position
        resolution = grid.info.resolution
        return (
            origin.x + (float(map_x) + 0.5) * resolution,
            origin.y + (float(map_y) + 0.5) * resolution,
        )

    def _index(self, grid: OccupancyGrid, map_x: int, map_y: int) -> int:
        """Return flattened occupancy grid index."""
        return map_y * grid.info.width + map_x

    def _near_any(
        self,
        points: List[Tuple[float, float]],
        x: float,
        y: float,
        radius: float,
    ) -> bool:
        """Return whether a point is close to any stored point."""
        radius_sq = radius * radius
        return any((px - x) ** 2 + (py - y) ** 2 <= radius_sq
                   for px, py in points)

    def _distance(
        self,
        point_a: Tuple[float, float],
        point_b: Tuple[float, float],
    ) -> float:
        """Return planar Euclidean distance."""
        return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def main(args=None):
    """Run the coverage planner node."""
    rclpy.init(args=args)
    node = CoveragePlannerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np


def _install_ros_stubs():
    try:
        import rclpy  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    rclpy = types.ModuleType("rclpy")
    rclpy.time = types.SimpleNamespace(Time=object)
    sys.modules["rclpy"] = rclpy

    duration = types.ModuleType("rclpy.duration")
    duration.Duration = object
    sys.modules["rclpy.duration"] = duration

    node = types.ModuleType("rclpy.node")
    node.Node = object
    sys.modules["rclpy.node"] = node

    qos = types.ModuleType("rclpy.qos")
    qos.DurabilityPolicy = types.SimpleNamespace(TRANSIENT_LOCAL=1)
    qos.QoSProfile = object
    qos.ReliabilityPolicy = types.SimpleNamespace(RELIABLE=1)
    sys.modules["rclpy.qos"] = qos

    tf2_ros = types.ModuleType("tf2_ros")
    tf2_ros.Buffer = object
    tf2_ros.TransformListener = object
    sys.modules["tf2_ros"] = tf2_ros

    for package, names in {
        "geometry_msgs.msg": ["Point", "PointStamped", "PoseStamped", "PoseWithCovarianceStamped"],
        "nav_msgs.msg": ["OccupancyGrid", "Path"],
        "sensor_msgs.msg": ["CameraInfo", "CompressedImage", "Image", "PointCloud2", "PointField"],
        "std_msgs.msg": ["Float32MultiArray", "String"],
        "trajectory_msgs.msg": ["JointTrajectoryPoint"],
        "visualization_msgs.msg": ["Marker", "MarkerArray"],
    }.items():
        module = types.ModuleType(package)
        for name in names:
            setattr(module, name, type(name, (), {}))
        sys.modules[package] = module

    viz = sys.modules["visualization_msgs.msg"]

    class Marker:
        ADD = 0
        DELETE = 2
        LINE_STRIP = 4
        CUBE = 1
        SPHERE = 2
        CYLINDER = 3
        TEXT_VIEW_FACING = 9

        def __init__(self):
            self.header = types.SimpleNamespace(frame_id="", stamp=None)
            self.ns = ""
            self.id = 0
            self.type = 0
            self.action = 0
            self.pose = types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=types.SimpleNamespace(w=0.0),
            )
            self.scale = types.SimpleNamespace(x=0.0, y=0.0, z=0.0)
            self.color = types.SimpleNamespace(r=0.0, g=0.0, b=0.0, a=0.0)
            self.points = []
            self.text = ""

    class MarkerArray:
        def __init__(self):
            self.markers = []

    viz.Marker = Marker
    viz.MarkerArray = MarkerArray

    sensor_msgs_py = types.ModuleType("sensor_msgs_py")
    point_cloud2 = types.ModuleType("sensor_msgs_py.point_cloud2")
    point_cloud2.read_points = lambda *args, **kwargs: []
    sensor_msgs_py.point_cloud2 = point_cloud2
    sys.modules["sensor_msgs_py"] = sensor_msgs_py
    sys.modules["sensor_msgs_py.point_cloud2"] = point_cloud2

    car_models = types.ModuleType("pros_car_py.car_models")
    car_models.DeviceDataTypeEnum = types.SimpleNamespace(
        car_C_rear_wheel="car_C_rear_wheel",
        car_C_front_wheel="car_C_front_wheel",
        robot_arm="robot_arm",
    )
    sys.modules["pros_car_py.car_models"] = car_models

    cv_bridge = types.ModuleType("cv_bridge")
    cv_bridge.CvBridge = object
    sys.modules["cv_bridge"] = cv_bridge

    ultralytics = types.ModuleType("ultralytics")
    ultralytics.YOLO = object
    sys.modules["ultralytics"] = ultralytics

    ament_index_python = types.ModuleType("ament_index_python")
    ament_packages = types.ModuleType("ament_index_python.packages")
    ament_packages.get_package_share_directory = lambda package: "/tmp"
    sys.modules["ament_index_python"] = ament_index_python
    sys.modules["ament_index_python.packages"] = ament_packages

    sys.modules["torch"] = types.ModuleType("torch")

    if "cv2" not in sys.modules:
        cv2 = types.ModuleType("cv2")
        cv2.CC_STAT_LEFT = 0
        cv2.CC_STAT_TOP = 1
        cv2.CC_STAT_WIDTH = 2
        cv2.CC_STAT_HEIGHT = 3
        cv2.CC_STAT_AREA = 4

        def connected_components_with_stats(image, connectivity=8):
            mask = np.asarray(image).astype(bool)
            height, width = mask.shape
            labels = np.zeros((height, width), dtype=np.int32)
            stats = [[0, 0, 0, 0, 0]]
            centroids = [[0.0, 0.0]]
            label = 0
            for y in range(height):
                for x in range(width):
                    if not mask[y, x] or labels[y, x] != 0:
                        continue
                    label += 1
                    stack = [(x, y)]
                    labels[y, x] = label
                    xs = []
                    ys = []
                    while stack:
                        cx, cy = stack.pop()
                        xs.append(cx)
                        ys.append(cy)
                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dx == 0 and dy == 0:
                                    continue
                                nx = cx + dx
                                ny = cy + dy
                                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                                    continue
                                if mask[ny, nx] and labels[ny, nx] == 0:
                                    labels[ny, nx] = label
                                    stack.append((nx, ny))
                    stats.append([
                        min(xs),
                        min(ys),
                        max(xs) - min(xs) + 1,
                        max(ys) - min(ys) + 1,
                        len(xs),
                    ])
                    centroids.append([sum(xs) / len(xs), sum(ys) / len(ys)])
            return label + 1, labels, np.asarray(stats, dtype=np.int32), np.asarray(centroids, dtype=np.float32)

        cv2.connectedComponentsWithStats = connected_components_with_stats
        cv2.dilate = lambda image, kernel, iterations=1: image
        sys.modules["cv2"] = cv2


_install_ros_stubs()

from pros_car_py.task_mission_controller import (
    BridgeBearMemory,
    BridgeVisionAnalyzer,
    MissionState,
    Task1MissionController,
    normalize_angle,
)


class DummyController:
    pass


class DummyLogger:
    def warn(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass


class DummyPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class DummyClock:
    def __init__(self):
        self.now_value = types.SimpleNamespace(
            nanoseconds=1_000_000_000,
            to_msg=lambda: types.SimpleNamespace(sec=1, nanosec=0),
        )

    def now(self):
        return self.now_value


def _bridge_node():
    node = DummyController()
    node.bridge_landmark = {
        "valid": True,
        "confidence": 0.8,
        "last_seen_time": types.SimpleNamespace(nanoseconds=1_000_000_000),
        "entry_map_x": 0.0,
        "entry_map_y": 0.0,
        "forward_axis": (1.0, 0.0),
        "left_side_map_points": [(0.15, 0.28, 0.0), (0.6, 0.34, 0.0), (0.9, 0.34, 0.0), (1.2, 0.34, 0.0)],
        "right_side_map_points": [(0.15, -0.28, 0.0), (0.6, -0.34, 0.0), (0.9, -0.34, 0.0), (1.2, -0.34, 0.0)],
        "centerline_map_points": [(0.2, 0.0, 0.0), (0.8, 0.0, 0.0), (1.2, 0.0, 0.0)],
        "left_observation_count": 2,
        "right_observation_count": 2,
    }
    node.bridge_side_inflation_radius = 0.18
    node.robot_footprint_radius = 0.22
    node.bridge_entry_opening_keep_clear = 0.30
    node.bridge_entry_gate_clear_radius = 0.45
    node.bridge_entry_gate_clear_width = 0.65
    node.bridge_side_obstacle_start_after_entry = 0.35
    node.bridge_center_corridor_clear_width = 0.45
    node.bridge_side_commit_min_points = 4
    node.bridge_side_commit_min_observations = 2
    node.bridge_side_obstacle_max_entry_distance = 0.0
    node.bridge_side_line_width = 0.08
    node.bridge_side_line_max_point_gap = 0.45
    node.bridge_side_line_min_length = 0.40
    node.bridge_side_line_outlier_distance = 0.25
    node.bridge_side_line_smoothing_enabled = False
    node.bridge_side_line_fit_enabled = True
    node.bridge_geometry_max_jump = 0.35
    node.bridge_geometry_parallel_angle_tolerance = math.radians(25.0)
    node.bridge_geometry_entry_must_lie_between_sides = True
    node.bridge_geometry_pre_entry_must_be_before_gate = True
    node.task2_allow_ramp_fallback_entry = True
    node.task2_ramp_entry_min_confidence = 0.55
    node.task2_ramp_entry_center_tolerance_pixels = 35.0
    node.task2_ramp_entry_min_bottom_y_ratio = 0.70
    node.task2_ramp_entry_min_vertical_coverage = 0.45
    node.task2_ramp_entry_max_side_view_score = 0.45
    node.task2_ramp_entry_confirm_frames = 3
    node.task2_ramp_entry_confirm_count = 0
    node.task2_side_view_recovery_enabled = True
    node.task2_side_view_min_ramp_confidence = 0.45
    node.task2_side_view_max_score = 0.55
    node.task2_bridge_approach_hard_tolerance = 100.0
    node.task2_bridge_approach_soft_tolerance = 45.0
    node.task2_turn_allow_frontal_bridge_ascent = True
    node.task2_turn_frontal_bridge_min_frontalness = 0.82
    node.task2_turn_frontal_bridge_min_confidence = 0.80
    node.task2_turn_frontal_bridge_min_bottom_y_ratio = 0.95
    node.task2_turn_frontal_bridge_center_tolerance_pixels = 65.0
    node.task2_turn_frontal_bridge_bear_tolerance_pixels = 70.0
    node.task2_turn_frontal_bridge_require_target = True
    node.task2_target_bridge_min_overlap_ratio = 0.12
    node.task2_target_bridge_min_lower_overlap_ratio = 0.20
    node.target_timeout = 1.0
    node.align_pixel_tolerance = 80.0
    node.task2_top_use_tf_z = True
    node.task2_top_z_threshold = 0.18
    node.task2_top_min_ascent_seconds = 7.0
    node.task2_top_visual_confidence_threshold = 0.55
    node.task2_ascent_timeout = 14.0
    node.task2_ascent_max_extra_seconds = 4.0
    node.task2_ascent_forward_speed_scale = 1.25
    node.task2_ascent_bear_pid_enabled = True
    node.task2_ascent_bear_pid_center_tolerance_pixels = 35.0
    node.task2_ascent_bear_pid_kp = 2.0
    node.task2_ascent_bear_pid_ki = 0.0
    node.task2_ascent_bear_pid_kd = 0.10
    node.task2_ascent_bear_pid_integral_limit = 200.0
    node.task2_ascent_bear_pid_max_turn = 340.0
    node.task2_ascent_bear_pid_rotate_only_pixels = 150.0
    node.task2_ascent_bear_pid_forward_scale = 0.75
    node.task2_ascent_stop_on_bridge_loss_seconds = 0.35
    node.task2_ascent_bear_pid_integral = 0.0
    node.task2_ascent_bear_pid_last_error = None
    node.task2_ascent_bear_pid_last_time = None
    node.virtual_obstacle_shape = "line"
    node.virtual_obstacle_line_length = 0.35
    node.virtual_obstacle_line_width = 0.08
    node.virtual_obstacle_display_as_cylinder = False
    node.augmented_map_min_line_cells = 1
    node.augmented_map_debug_bridge_lines = True
    node.augmented_map_bridge_line_debug_width_cells = 2
    node.bridge_landmark_min_width = 0.25
    node.bridge_landmark_max_width = 2.5
    node.fitted_bridge_side_lines = {"left": [], "right": []}
    node.last_accepted_bridge_side_lines = {"left": [], "right": []}
    node.bridge_geometry_quality_accepted = False
    node.bridge_geometry_quality_reason = ""
    node.virtual_obstacles = []
    node.map_msg = None
    node.map_free_threshold = 25
    node.bridge_landmark_min_observations = 1
    node.task2_bridge_tracking_expire = 999.0
    node._elapsed_seconds = lambda start_time: 0.0
    for name in (
        "_bridge_landmark_is_usable",
        "_bridge_entry_xy",
        "_near_bridge_entry_opening",
        "_bridge_entry_coordinates",
        "_inside_bridge_center_corridor",
        "_bridge_side_has_enough_observations",
        "_committed_bridge_side_point",
        "_bridge_side_blocks_world_point",
        "_action_safety_result",
        "_action_safety_check",
        "_world_point_safety_check",
        "_virtual_obstacle_at_world_point",
        "_virtual_obstacle_contains_point",
        "_bridge_side_blocker_at_world_point",
        "_static_map_safety_at_world_point",
        "_blocker_side_from_point",
        "_virtual_obstacle_is_suppressed",
        "_virtual_obstacle_inside_active_bridge_gate",
        "_suppress_virtual_obstacle_for_bridge_entry",
        "_bridge_pre_entry_xy",
        "_fit_bridge_side_lines",
        "_bridge_side_lines_jump_too_far",
        "_fit_one_bridge_side_line",
        "_sort_points_along_bridge_axis",
        "_reject_bridge_side_outliers",
        "_smooth_polyline",
        "_split_large_line_gaps",
        "_polyline_length",
        "_bridge_side_line_geometry_is_valid",
        "_entry_between_side_lines",
        "_pre_entry_before_gate",
        "_rasterize_polyline_obstacle",
        "_rasterize_line_obstacle",
        "_mark_augmented_line_width",
        "_point_to_segment_distance",
        "_committed_bridge_side_point_count",
        "_bridge_ramp_is_usable",
        "_bridge_side_view_likely",
        "_task2_entry_source",
        "_task2_entry_delta",
        "_task2_bridge_entry_confirmed",
        "_task2_bridge_pre_entry_confirmed",
        "_task2_bridge_entry_delta",
        "_task2_bridge_pre_entry_delta",
        "_task2_remember_bridge_delta",
        "_target_surface_candidate",
        "_task2_bridge_confidence_for_ascent",
        "_task2_bridge_bear_delta_for_ascent",
        "_task2_bridge_bear_turn_action",
        "_task2_bridge_bear_ascent_action",
        "_reset_task2_ascent_bear_pid",
        "_task2_frontal_bridge_base_ready_for_ascent",
        "_task2_frontal_bridge_ready_for_ascent",
        "_task2_ascend_bridge",
        "_start_task2_ascent_settle",
        "_publish_ascent_or_action",
        "_publish_ascent_bear_pid",
        "_publish_ascent_forward",
        "_bridge_top_confidence",
        "_bridge_top_visual_score",
        "_make_marker",
        "_line_marker",
        "_text_marker",
    ):
        setattr(node, name, getattr(Task1MissionController, name).__get__(node))
    node.task2_entry_ignore_virtual_obstacle_inside_gate = True
    node.task2_entry_virtual_obstacle_gate_margin = 0.35
    node.task2_bridge_entry_center_tolerance = 35.0
    node._bridge_observation_is_fresh = lambda bridge: bridge is not None
    node._task2_bridge_visible = lambda allow_cached=False: None
    node._segmentation_image_width = lambda: 640.0
    node._target_visible = lambda: False
    node._time_after = lambda start_time, seconds: types.SimpleNamespace(
        nanoseconds=start_time.nanoseconds + int(seconds * 1e9)
    )
    node._log_event = lambda *args, **kwargs: None
    node._cleanup_expired_virtual_obstacles = lambda: None
    return node


def _map_node():
    node = _bridge_node()
    info = types.SimpleNamespace(
        resolution=0.1,
        width=30,
        height=20,
        origin=types.SimpleNamespace(
            position=types.SimpleNamespace(x=-0.5, y=-1.0)
        ),
    )
    node.map_msg = types.SimpleNamespace(
        header=types.SimpleNamespace(frame_id="map"),
        info=info,
        data=[0] * (info.width * info.height),
    )
    node.map_free_threshold = 25
    node.virtual_obstacles = []
    node.augmented_map_pub = DummyPublisher()
    node.augmented_map_publish_count = 0
    node.last_augmented_map_obstacle_cell_count = 0
    node.augmented_map_visible_reason = ""
    node._mark_augmented_disk = Task1MissionController._mark_augmented_disk.__get__(
        node
    )
    return node


def test_normalize_angle_wrap_adds_short_delta():
    previous = math.radians(170.0)
    current = math.radians(-170.0)
    delta = normalize_angle(current - previous)
    assert abs(math.degrees(abs(delta)) - 20.0) <= 0.5


def test_lower_frame_bbox_gate_accepts_lower_target():
    node = DummyController()
    node._bbox_visible = lambda: True
    node.yolo_bbox = {
        "center_x": 320.0,
        "center_y": 380.0,
        "y2": 455.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "area_ratio": 0.02,
    }
    node.task1_target_min_bbox_area_ratio = 0.004
    node.task1_target_center_min_ratio = 0.15
    node.task1_target_center_max_ratio = 0.85
    node.task1_target_lower_min_center_y_ratio = 0.52
    node.task1_target_lower_min_bottom_y_ratio = 0.68

    valid, _ = Task1MissionController._target_bbox_in_lower_camera(node)

    assert valid


def test_lower_frame_bbox_gate_rejects_upper_target():
    node = DummyController()
    node._bbox_visible = lambda: True
    node.yolo_bbox = {
        "center_x": 320.0,
        "center_y": 120.0,
        "y2": 170.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "area_ratio": 0.02,
    }
    node.task1_target_min_bbox_area_ratio = 0.004
    node.task1_target_center_min_ratio = 0.15
    node.task1_target_center_max_ratio = 0.85
    node.task1_target_lower_min_center_y_ratio = 0.52
    node.task1_target_lower_min_bottom_y_ratio = 0.68

    valid, reason = Task1MissionController._target_bbox_in_lower_camera(node)

    assert not valid
    assert "lower camera" in reason


def test_road_corridor_centered_wide_road_is_valid():
    node = DummyController()
    node.task2_road_explore_min_area_ratio = 0.015
    node.task2_road_explore_min_bottom_coverage = 0.08
    node.task2_road_explore_min_width_ratio = 0.20
    road = {
        "found": True,
        "area_ratio": 0.08,
        "bottom_coverage": 0.30,
        "bottom_left_x": 200.0,
        "bottom_right_x": 440.0,
        "bottom_center_x": 320.0,
        "mid_left_x": 230.0,
        "mid_right_x": 410.0,
        "mid_center_x": 320.0,
    }
    node._segmentation_segment = lambda label: road if label == "road" else None
    node._segmentation_image_width = lambda: 640.0

    corridor = Task1MissionController._task2_road_corridor(node)

    assert corridor["valid"]
    assert abs(corridor["heading_error"]) < 1.0
    assert corridor["width_ratio"] > 0.20


def test_grab_gate_accepts_close_central_lower_bbox():
    node = DummyController()
    node._target_visible = lambda: True
    node._bbox_visible = lambda: True
    node.bear_context = "task1_ground_bear"
    node.yolo_target = {"distance": 0.35}
    node.yolo_bbox = {
        "center_x": 320.0,
        "center_y": 360.0,
        "y2": 450.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "area_ratio": 0.03,
    }
    node.grab_distance = 0.40
    node.grab_bbox_center_min_x_ratio = 0.38
    node.grab_bbox_center_max_x_ratio = 0.62
    node.grab_bbox_min_center_y_ratio = 0.55
    node.grab_bbox_min_bottom_y_ratio = 0.72
    node.grab_bbox_max_bottom_y_ratio = 1.0
    node.grab_bbox_min_area_ratio = 0.008

    valid, _ = Task1MissionController._target_ready_for_grab(node)

    assert valid


def test_action_projection_marks_clockwise_as_right_front():
    node = DummyController()
    node.virtual_obstacle_front_distance = 0.45
    node.virtual_obstacle_diagonal_distance = 0.42

    angle, distance = Task1MissionController._action_projection(
        node, "CLOCKWISE_ROTATION_SLOW"
    )

    assert abs(math.degrees(angle) - (-45.0)) <= 0.5
    assert abs(distance - 0.42) <= 1e-6


def test_missing_bridge_frame_retains_last_geometry():
    node = DummyController()
    now = types.SimpleNamespace(nanoseconds=2_000_000_000)
    previous_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.segmentation_smoothing_alpha = 0.45
    node.segmentation_last_seen = {"bridge": previous_time}
    node.segmentation_info = {
        "bridge": {
            "found": True,
            "raw_found": True,
            "usable": True,
            "delta_x": 80.0,
            "area_ratio": 0.1,
            "bottom_coverage": 0.2,
            "center_x": 400.0,
            "bottom_center_x": 410.0,
            "bottom_left_x": 300.0,
            "bottom_right_x": 520.0,
        }
    }
    missing = {
        "found": False,
        "delta_x": 0.0,
        "area_ratio": 0.0,
        "bottom_coverage": 0.0,
        "center_x": 0.0,
        "bottom_center_x": 0.0,
        "bottom_left_x": 0.0,
        "bottom_right_x": 0.0,
    }

    retained = Task1MissionController._filtered_segment_update(
        node, "bridge", missing, now
    )

    assert not retained["raw_found"]
    assert retained["predicted_or_cached"]
    assert retained["center_x"] == 400.0
    assert retained["bottom_left_x"] == 300.0


def test_turn_sign_auto_flips_after_repeated_wrong_way_pulses():
    node = DummyController()
    node.task2_turn_wrong_way_pixel_epsilon = 8.0
    node.task2_turn_wrong_way_limit = 2
    node.task2_turn_auto_flip_enabled = True
    node.task2_turn_sign_confirmed = False
    node.task2_turn_direction_sign = 1.0
    node.task2_turn_wrong_way_count = 0
    node.task2_turn_command_action = "CLOCKWISE_ROTATION_SLOW"
    node.task2_turn_error_before_pulse = 100.0
    node.task2_turn_last_observed_error = None
    node.get_logger = lambda: DummyLogger()

    Task1MissionController._task2_evaluate_turn_pulse(node, 120.0)
    node.task2_turn_command_action = "CLOCKWISE_ROTATION_SLOW"
    node.task2_turn_error_before_pulse = 100.0
    Task1MissionController._task2_evaluate_turn_pulse(node, 120.0)

    assert node.task2_turn_direction_sign == -1.0
    assert node.task2_turn_wrong_way_count == 0


def test_bridge_entry_gate_samples_remain_free():
    node = _bridge_node()

    assert not Task1MissionController._bridge_side_blocks_world_point(node, 0.05, 0.0)
    assert not Task1MissionController._bridge_side_blocks_world_point(node, 0.15, 0.25)


def test_bridge_center_corridor_remains_free():
    node = _bridge_node()

    assert not Task1MissionController._bridge_side_blocks_world_point(node, 0.9, 0.0)
    assert not Task1MissionController._bridge_side_blocks_world_point(node, 1.1, 0.18)


def test_bridge_side_points_after_entry_are_obstacles():
    node = _bridge_node()

    assert Task1MissionController._bridge_side_blocks_world_point(node, 0.9, 0.34)
    assert Task1MissionController._bridge_side_blocks_world_point(node, 1.2, -0.34)


def test_augmented_map_marks_side_cells_but_not_entry_gate():
    node = _map_node()

    Task1MissionController._publish_augmented_map(node)

    msg = node.augmented_map_pub.messages[-1]
    info = msg.info

    def cell_value(x, y):
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        return msg.data[gy * info.width + gx]

    assert cell_value(0.0, 0.0) == 0
    assert cell_value(0.9, 0.0) == 0
    assert cell_value(0.9, 0.34) == 100
    assert node.last_augmented_map_obstacle_cell_count > 0


def test_action_toward_entry_is_not_blocked_by_side_obstacles():
    node = _bridge_node()
    node.pose = (0.0, 0.0, 0.0)
    node.virtual_obstacle_front_distance = 0.30
    node.virtual_obstacle_diagonal_distance = 0.30
    node.motion_safety_sample_count = 4
    node._virtual_obstacle_blocks_world = lambda _x, _y: False
    node._action_projection = lambda action: Task1MissionController._action_projection(
        node, action
    )
    node._action_is_rotation = lambda action: Task1MissionController._action_is_rotation(
        node, action
    )
    node._robot_footprint_is_clear_at = (
        lambda x, y, yaw=None: Task1MissionController._robot_footprint_is_clear_at(
            node, x, y, yaw
        )
    )
    node._bridge_side_blocks_world_point = (
        lambda x, y: Task1MissionController._bridge_side_blocks_world_point(node, x, y)
    )
    node.map_msg = None
    node.map_free_threshold = 25

    assert Task1MissionController._action_swept_path_is_clear(node, "FORWARD_SLOW")


def test_top_bear_search_does_not_drive_forward_when_disabled():
    node = DummyController()
    node.next_state_after_grab = None
    node.task2_ascent_completed = True
    node.bridge_top_confirmed = True
    node.task2_phase_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.task2_top_search_timeout_seconds = 60.0
    node.task2_top_bear_search_budget_seconds = 60.0
    node.task2_top_search_rotate_seconds = 8.0
    node.task2_top_search_turn_direction_switch_seconds = 4.0
    node.task2_top_search_allow_forward = False
    node.task2_top_search_max_forward_seconds = 0.0
    node.task2_top_bear_allow_cached_memory = False
    node.task2_top_bear_require_visible_bbox = True
    node.task2_relax_bridge_surface_after_top = True
    node.bridge_bear_memory = BridgeBearMemory()
    node.get_clock = lambda: DummyClock()
    node._target_visible = lambda: False
    node._update_bridge_bear_memory = lambda: None
    node._elapsed_seconds = lambda start_time: 5.0
    actions = []
    node._publish_action = actions.append

    Task1MissionController._task2_search_bridge_bear(node)

    assert actions
    assert "FORWARD_SLOW" not in actions


def test_descent_is_blocked_when_bear_not_secured():
    node = DummyController()
    node.task2_require_bear_secured_before_descent = True
    node.bear_secured = False
    node._publish_action = lambda action: setattr(node, "last_action", action)
    node.get_logger = lambda: DummyLogger()
    node._set_state = lambda state, reason="": setattr(node, "next_state", state)

    Task1MissionController._task2_descend_bridge(node)

    assert node.last_action == "STOP"
    assert node.next_state.value == "task2_search_bridge_bear"


def test_verified_grab_sets_bear_secured_and_shared_task_flag():
    node = DummyController()
    node.verify_grab_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.verify_grab_seen_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.verify_grab_seconds = 0.5
    node.next_state_after_grab = types.SimpleNamespace(value="task2_descent")
    node.bear_context = "task2_bridge_bear"
    node.task1_observe_completed = False
    node.bear_secured = False
    node.grab_retry_count = 0
    node._publish_action = lambda action: None
    node.get_clock = lambda: DummyClock()
    node._elapsed_seconds = lambda start_time: 1.0
    node._grab_bbox_indicates_bear_held = lambda: (True, "ok")
    node._reset_verify_grab = lambda: None
    node._set_state = lambda state: setattr(node, "next_state", state)
    node.get_logger = lambda: DummyLogger()
    node._log_event = lambda *args, **kwargs: None

    Task1MissionController._verify_grab(node)

    assert node.bear_secured
    assert node.task1_observe_completed


def test_unknown_forward_block_recovery_does_not_stop_forever():
    node = DummyController()
    node.task2_entry_blocked_start_time = None
    node.task2_entry_blocked_count = 0
    node.task2_entry_recovery_phase = None
    node.task2_entry_recovery_phase_start_time = None
    node.task2_entry_blocked_last_reason = ""
    node.task2_entry_unknown_block_stop_seconds = 0.2
    node.task2_entry_unknown_block_backup_seconds = 0.4
    node.task2_entry_unknown_block_turn_seconds = 0.5
    node.task2_entry_unknown_block_max_retries = 3
    node.task2_bridge_last_delta_sign = 1.0
    node.get_clock = lambda: DummyClock()
    node._elapsed_seconds = lambda start_time: 1.0
    node._task2_bridge_entry_delta = lambda bridge=None: None
    node._task2_bridge_rough_target_delta = lambda bridge=None: 20.0
    node._set_state = lambda state, reason="": setattr(node, "next_state", state)
    node._log_event = lambda *args, **kwargs: None

    result = Task1MissionController._task2_entry_blocked_recovery_action(
        node,
        {
            "blocker_type": "unknown",
            "blocker_side": "unknown",
            "blocker_distance": 0.2,
            "reason": "test unknown block",
        },
        None,
    )

    assert result == "BACKWARD_SLOW"


def test_virtual_obstacle_inside_bridge_gate_is_suppressed():
    node = _bridge_node()
    node.pose = (0.0, 0.0, 0.0)
    node.get_clock = lambda: DummyClock()
    obstacle = {
        "x": 0.55,
        "y": 0.05,
        "radius": 0.35,
        "suppressed_until": None,
        "suppression_reason": "",
        "created_state": "task2_approach_bridge_entry",
        "created_action": "FORWARD_SLOW",
    }
    node.virtual_obstacles = [obstacle]

    blocker = Task1MissionController._virtual_obstacle_at_world_point(
        node, 0.55, 0.05, bridge={"entry_confirmed": 1.0}
    )

    assert blocker is None
    assert obstacle["suppression_reason"]


def test_virtual_obstacle_outside_gate_still_blocks():
    node = _bridge_node()
    node.pose = (0.0, 0.0, 0.0)
    obstacle = {
        "x": 2.0,
        "y": 0.8,
        "radius": 0.4,
        "suppressed_until": None,
        "suppression_reason": "",
        "created_state": "explore_map",
        "created_action": "FORWARD_SLOW",
    }
    node.virtual_obstacles = [obstacle]

    blocker = Task1MissionController._virtual_obstacle_at_world_point(
        node, 2.0, 0.8, bridge=None
    )

    assert blocker is obstacle


def test_static_map_block_reports_static_map_type():
    node = _map_node()
    info = node.map_msg.info
    gx = int((1.6 - info.origin.position.x) / info.resolution)
    gy = int((0.8 - info.origin.position.y) / info.resolution)
    node.map_msg.data[gy * info.width + gx] = 100
    node.pose = (0.0, 0.0, 0.0)

    result = Task1MissionController._static_map_safety_at_world_point(node, 1.6, 0.8)

    assert result["blocked"]
    assert result["blocker_type"] == "static_map"


def test_bridge_side_safety_reports_left_and_right():
    node = _bridge_node()
    node.pose = (0.0, 0.0, 0.0)

    left = Task1MissionController._world_point_safety_check(node, 0.9, 0.34)
    right = Task1MissionController._world_point_safety_check(node, 0.9, -0.34)

    assert left["blocker_type"] == "bridge_left_side"
    assert left["blocker_side"] == "left"
    assert right["blocker_type"] == "bridge_right_side"
    assert right["blocker_side"] == "right"


def test_rough_bridge_target_is_not_confirmed_entry():
    node = DummyController()
    node._task2_bridge_visible = lambda allow_cached=False: None
    node._bridge_observation_is_fresh = lambda bridge: True
    bridge = {
        "raw_found": True,
        "entry_confirmed": 0.0,
        "entry_from_road_connection": 0.0,
        "entry_confidence": 0.0,
        "entry_u": 0.0,
        "target_u": 350.0,
    }

    assert not Task1MissionController._task2_bridge_entry_confirmed(node, bridge)


def test_bridge_side_lines_rasterize_thin_cells_not_blobs():
    node = _map_node()

    Task1MissionController._publish_augmented_map(node)

    assert node.bridge_side_line_cells > 0
    assert node.virtual_obstacle_cells == 0
    assert node.last_augmented_map_obstacle_cell_count == node.bridge_side_line_cells
    assert node.bridge_side_line_cells < 80


def test_stuck_virtual_obstacle_defaults_to_line_cells():
    node = _map_node()
    node.bridge_landmark["left_side_map_points"] = []
    node.bridge_landmark["right_side_map_points"] = []
    node.virtual_obstacles = [
        {
            "x": 1.2,
            "y": 0.6,
            "x0": 1.0,
            "y0": 0.6,
            "x1": 1.4,
            "y1": 0.6,
            "shape": "line",
            "line_width": 0.08,
            "radius": 0.25,
            "created_time": types.SimpleNamespace(nanoseconds=1_000_000_000),
            "suppressed_until": None,
        }
    ]

    Task1MissionController._publish_augmented_map(node)

    assert node.virtual_obstacle_cells > 0
    assert node.bridge_side_line_cells == 0


def test_disk_virtual_obstacle_supported_when_explicitly_configured():
    node = _map_node()
    node.bridge_landmark["left_side_map_points"] = []
    node.bridge_landmark["right_side_map_points"] = []
    node.virtual_obstacles = [
        {
            "x": 1.2,
            "y": 0.6,
            "shape": "disk",
            "radius": 0.25,
            "created_time": types.SimpleNamespace(nanoseconds=1_000_000_000),
            "suppressed_until": None,
        }
    ]

    Task1MissionController._publish_augmented_map(node)

    assert node.virtual_obstacle_cells > 0


def test_too_short_bridge_side_lines_are_rejected():
    node = _bridge_node()
    node.bridge_landmark["left_side_map_points"] = [(0.6, 0.34, 0.0), (0.7, 0.34, 0.0)]
    node.bridge_landmark["right_side_map_points"] = [(0.6, -0.34, 0.0), (0.7, -0.34, 0.0)]

    Task1MissionController._fit_bridge_side_lines(node)

    assert not node.bridge_geometry_quality_accepted


def test_nonparallel_bridge_side_lines_are_rejected():
    node = _bridge_node()
    node.bridge_landmark["left_side_map_points"] = [(0.6, 0.34, 0.0), (0.9, 0.34, 0.0), (1.2, 0.34, 0.0), (1.5, 0.34, 0.0)]
    node.bridge_landmark["right_side_map_points"] = [(0.6, -0.34, 0.0), (0.9, -0.10, 0.0), (1.2, 0.10, 0.0), (1.5, 0.35, 0.0)]

    Task1MissionController._fit_bridge_side_lines(node)

    assert not node.bridge_geometry_quality_accepted


def test_wrong_position_bridge_side_lines_are_rejected_before_map_marking():
    node = _map_node()
    node.last_accepted_bridge_side_lines = {
        "left": [(0.6, 0.34, 0.0), (1.2, 0.34, 0.0)],
        "right": [(0.6, -0.34, 0.0), (1.2, -0.34, 0.0)],
    }
    node.bridge_landmark["left_side_map_points"] = [
        (1.6, 0.34, 0.0),
        (1.9, 0.34, 0.0),
        (2.2, 0.34, 0.0),
        (2.5, 0.34, 0.0),
    ]
    node.bridge_landmark["right_side_map_points"] = [
        (1.6, -0.34, 0.0),
        (1.9, -0.34, 0.0),
        (2.2, -0.34, 0.0),
        (2.5, -0.34, 0.0),
    ]

    Task1MissionController._publish_augmented_map(node)

    assert not node.bridge_geometry_quality_accepted
    assert node.bridge_side_line_cells == 0
    assert "jumped too far" in node.bridge_geometry_quality_reason


def test_marker_array_distinguishes_side_lines_and_virtual_obstacles():
    node = _bridge_node()
    node.pose = (0.0, 0.0, 0.0)
    node.map_frame = "map"
    node.get_clock = lambda: DummyClock()
    node.bridge_marker_pub = DummyPublisher()
    node.bridge_marker_publish_count = 0
    node.virtual_obstacles = [
        {
            "x": 1.2,
            "y": 0.6,
            "x0": 1.0,
            "y0": 0.6,
            "x1": 1.4,
            "y1": 0.6,
            "shape": "line",
            "line_width": 0.08,
            "radius": 0.25,
            "created_time": types.SimpleNamespace(nanoseconds=1_000_000_000),
            "suppressed_until": None,
        }
    ]

    Task1MissionController._publish_bridge_markers(node)
    markers = node.bridge_marker_pub.messages[-1].markers

    side = [m for m in markers if m.ns == "bridge_left_side"][0]
    virtual = [m for m in markers if m.ns == "stuck_virtual_obstacles"][0]
    assert side.type == side.LINE_STRIP
    assert virtual.type == virtual.LINE_STRIP


def _yolo_node_without_init():
    path = Path(
        "/home/kiwi/workspace/pros/ros2_yolo_integration/src/yolo_example_pkg/yolo_example_pkg/object_detect.py"
    )
    spec = importlib.util.spec_from_file_location("object_detect_for_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return object.__new__(module.YoloDetectionNode)


def test_target_surface_info_marks_side_contact_with_bridge_mask():
    node = _yolo_node_without_init()
    node.target_surface_pub = DummyPublisher()
    bridge = np.zeros((480, 640), dtype=bool)
    bridge[180:250, 342:348] = True
    target = {
        "x1": 300,
        "y1": 180,
        "x2": 340,
        "y2": 250,
        "center_x": 320,
        "center_y": 215,
    }

    node.publish_target_surface_info(target, bridge)
    data = node.target_surface_pub.messages[-1].data

    assert data[2] == 0.0
    assert data[10] == 1.0
    assert data[12] > 0.0


def test_bridge_ramp_valid_when_centered_lower_and_continuous():
    node = _yolo_node_without_init()
    bridge = np.zeros((480, 640), dtype=bool)
    for y in range(150, 470):
        half_width = int(35 + (y - 150) * 0.25)
        bridge[y, 320 - half_width : 320 + half_width] = True
    road = np.zeros_like(bridge)

    quality = node._compute_bridge_ramp_quality(bridge, road)

    assert quality["ramp_valid"]
    assert quality["lower_present"]
    assert quality["continuous_lower_to_upper"]
    assert quality["side_view_score"] <= 0.55


def test_bridge_ramp_invalid_when_side_view_mask_only():
    node = _yolo_node_without_init()
    bridge = np.zeros((480, 640), dtype=bool)
    for y in range(210, 470):
        center = 520 + int((y - 210) * 0.25)
        bridge[y, center : min(639, center + 45)] = True
    road = np.zeros_like(bridge)

    quality = node._compute_bridge_ramp_quality(bridge, road)

    assert not quality["ramp_valid"]
    assert quality["side_view_score"] >= 0.55 or not quality["centered"]


def test_road_bridge_contact_confirms_entry_source():
    node = _bridge_node()
    bridge = {
        "entry_confirmed": 1.0,
        "entry_from_road_connection": 1.0,
        "entry_confidence": 0.8,
        "entry_u": 320.0,
        "entry_gate_center_u": 320.0,
        "pre_entry_confidence": 0.7,
        "pre_entry_u": 318.0,
        "raw_found": True,
    }

    source = Task1MissionController._task2_entry_source(node, bridge)

    assert source == "road_contact"


def test_missing_contact_can_use_confirmed_ramp_fallback():
    node = _bridge_node()
    bridge = {
        "entry_confirmed": 0.0,
        "ramp_valid": 1.0,
        "ramp_confidence": 0.80,
        "ramp_lower_present": 1.0,
        "ramp_continuous": 1.0,
        "side_view_score": 0.20,
        "vertical_coverage_score": 0.80,
        "bottom_y_ratio": 0.82,
        "bottom_center_x": 320.0,
        "mid_center_x": 320.0,
        "center_x": 320.0,
        "raw_found": True,
    }

    source = "none"
    for _ in range(node.task2_ramp_entry_confirm_frames):
        source = Task1MissionController._task2_entry_source(node, bridge)

    assert source == "ramp_fallback"


def test_target_surface_candidate_accepts_bridge_side_contact():
    node = _bridge_node()
    node.target_surface_stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.target_surface_info = {
        "target_found": True,
        "bridge_found": True,
        "bbox_bridge_overlap_ratio": 0.0,
        "bbox_lower_half_bridge_overlap_ratio": 0.0,
        "target_center_on_bridge": False,
        "target_bottom_center_on_bridge": False,
        "target_side_bridge_contact": True,
        "target_side_bridge_contact_ratio": 0.06,
    }

    valid, reason = Task1MissionController._target_surface_candidate(node)

    assert valid
    assert "side" in reason


def test_frontal_bridge_gate_allows_ascent_without_road_contact_when_bear_centered():
    node = _bridge_node()
    node._target_visible = lambda: True
    node.yolo_target = {"delta_x": 12.0}
    node.target_surface_stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.target_surface_info = {
        "target_found": True,
        "bridge_found": True,
        "target_side_bridge_contact": True,
        "target_side_bridge_contact_ratio": 0.08,
        "bbox_bridge_overlap_ratio": 0.0,
        "bbox_lower_half_bridge_overlap_ratio": 0.0,
    }
    bridge = {
        "raw_found": True,
        "frontalness": 0.90,
        "ramp_confidence": 0.88,
        "bottom_y_ratio": 0.98,
        "bottom_center_x": 320.0,
        "mid_center_x": 320.0,
    }

    ready, reason = Task1MissionController._task2_frontal_bridge_ready_for_ascent(
        node, bridge, delta_x=8.0
    )

    assert ready
    assert "bear centered" in reason


def test_frontal_bridge_gate_rejects_off_center_bridge_bear():
    node = _bridge_node()
    node._target_visible = lambda: True
    node.yolo_target = {"delta_x": 120.0}
    node.target_surface_stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.target_surface_info = {
        "target_found": True,
        "bridge_found": True,
        "target_side_bridge_contact": True,
        "target_side_bridge_contact_ratio": 0.08,
        "bbox_bridge_overlap_ratio": 0.0,
        "bbox_lower_half_bridge_overlap_ratio": 0.0,
    }
    bridge = {
        "raw_found": True,
        "frontalness": 0.90,
        "ramp_confidence": 0.88,
        "bottom_y_ratio": 0.98,
        "bottom_center_x": 320.0,
        "mid_center_x": 320.0,
    }

    ready, reason = Task1MissionController._task2_frontal_bridge_ready_for_ascent(
        node, bridge, delta_x=8.0
    )

    assert not ready
    assert "not centered" in reason


def test_bridge_bottom_center_alone_cannot_confirm_entry():
    node = _bridge_node()
    bridge = {
        "entry_confirmed": 0.0,
        "ramp_valid": 0.0,
        "bottom_center_x": 320.0,
        "mid_center_x": 0.0,
        "raw_found": True,
    }

    source = Task1MissionController._task2_entry_source(node, bridge)

    assert source == "none"


def test_side_view_bridge_mask_is_likely_recovery_condition():
    node = _bridge_node()
    bridge = {
        "raw_found": True,
        "ramp_valid": 0.0,
        "ramp_confidence": 0.20,
        "ramp_continuous": 0.0,
        "side_view_score": 0.80,
        "bottom_center_x": 520.0,
        "mid_center_x": 560.0,
    }

    assert BridgeVisionAnalyzer.side_view_likely(node, bridge)


def test_top_confidence_false_at_min_time_without_z_or_visual_signal():
    node = _bridge_node()
    node.pose_z = 0.0
    node.start_pose_z = 0.0
    node.task2_ascent_start_z = 0.0
    node.task2_phase_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node._elapsed_seconds = lambda start_time: 8.0
    node._task2_bridge_visible = lambda allow_cached=False: {
        "ramp_confidence": 0.80,
        "bottom_y_ratio": 0.90,
        "side_view_score": 0.20,
    }

    ok, _, _ = Task1MissionController._bridge_top_confidence(node)

    assert not ok


def test_ascent_bear_pid_action_selected_when_bridge_bear_off_center():
    node = _bridge_node()
    node._target_visible = lambda: True
    node.yolo_target = {"delta_x": 90.0}
    node.target_surface_stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.target_surface_info = {
        "target_found": True,
        "bridge_found": True,
        "target_side_bridge_contact": True,
        "target_side_bridge_contact_ratio": 0.08,
        "bbox_bridge_overlap_ratio": 0.0,
        "bbox_lower_half_bridge_overlap_ratio": 0.0,
    }

    action = Task1MissionController._task2_bridge_bear_ascent_action(node)

    assert action == "ASCEND_BEAR_PID"


def test_ascent_bear_pid_turns_right_for_positive_bear_error():
    node = _bridge_node()
    node._target_visible = lambda: True
    node.yolo_target = {"delta_x": 90.0}
    node.target_surface_stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.target_surface_info = {
        "target_found": True,
        "bridge_found": True,
        "target_side_bridge_contact": True,
        "target_side_bridge_contact_ratio": 0.08,
        "bbox_bridge_overlap_ratio": 0.0,
        "bbox_lower_half_bridge_overlap_ratio": 0.0,
    }
    node.rear_pub = DummyPublisher()
    node.front_pub = DummyPublisher()
    node.get_clock = lambda: DummyClock()
    node._apply_stuck_recovery = lambda action: action
    node.last_logged_action = None
    node.last_action_log_time = None
    node._log_event = lambda *args, **kwargs: None

    Task1MissionController._publish_ascent_bear_pid(node)

    rear = node.rear_pub.messages[-1].data
    front = node.front_pub.messages[-1].data
    assert rear[0] > rear[1]
    assert front[0] > front[1]


def test_ascent_does_not_complete_from_top_confidence_while_bridge_mask_visible():
    node = _bridge_node()
    node.task2_phase_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.task2_ascent_stop_start_time = None
    node.task2_ascent_lost_bridge_start_time = None
    node.task2_ascent_action = "ASCEND_FORWARD"
    node.task2_ascent_centering_enabled = False
    node.task2_use_bridge_top_pose = False
    node.pose_z = 0.30
    node.start_pose_z = 0.0
    node.task2_ascent_start_z = 0.0
    node.bridge_top_confirm_count = 0
    node.bridge_top_confirmed = False
    node._update_bridge_bear_memory = lambda: None
    node._task2_bridge_visible = lambda allow_cached=False: {
        "raw_found": True,
        "ramp_confidence": 0.8,
        "bottom_y_ratio": 0.95,
        "side_view_score": 0.1,
        "bottom_center_x": 320.0,
        "mid_center_x": 320.0,
    }
    actions = []
    node._publish_ascent_or_action = actions.append

    Task1MissionController._task2_ascend_bridge(node)

    assert actions == ["ASCEND_FORWARD"]
    assert node.task2_ascent_stop_start_time is None


def test_ascent_completes_when_bridge_mask_lost_for_grace_period():
    node = _bridge_node()
    node.task2_phase_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.task2_ascent_stop_start_time = None
    node.task2_ascent_lost_bridge_start_time = types.SimpleNamespace(
        nanoseconds=1_000_000_000
    )
    node.pose_z = 0.0
    node.start_pose_z = 0.0
    node.task2_ascent_start_z = 0.0
    node.bridge_top_confirmed = False
    node._update_bridge_bear_memory = lambda: None
    node._task2_bridge_visible = lambda allow_cached=False: None
    node._elapsed_seconds = lambda start_time: 0.4
    actions = []
    node._publish_action = actions.append
    node._start_task2_ascent_settle = lambda: setattr(node, "settled", True)

    Task1MissionController._task2_ascend_bridge(node)

    assert actions == ["STOP"]
    assert node.settled


def test_top_confidence_true_when_z_threshold_passed():
    node = _bridge_node()
    node.pose_z = 0.25
    node.start_pose_z = 0.0
    node.task2_ascent_start_z = 0.0
    node.task2_phase_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node._elapsed_seconds = lambda start_time: 2.0

    ok, score, reason = Task1MissionController._bridge_top_confidence(node)

    assert ok
    assert score == 1.0
    assert "z threshold" in reason


def test_bridge_bear_memory_recent_and_expires():
    node = _bridge_node()
    memory = BridgeBearMemory()
    node.yolo_bbox = {"confidence": 0.8, "center_x": 320.0}
    node.yolo_target = {"distance": 0.5}
    node.target_surface_info = {"bbox_bridge_overlap_ratio": 0.2}
    node.get_clock = lambda: DummyClock()
    node.state = MissionState.TASK2_SEARCH_BRIDGE_BEAR
    memory.update(node, 0.8)

    assert memory.recent(node, ttl_seconds=12.0, min_confidence=0.45)
    node._elapsed_seconds = lambda start_time: 13.0
    memory.clear_if_expired(node, ttl_seconds=12.0)
    assert not memory.valid

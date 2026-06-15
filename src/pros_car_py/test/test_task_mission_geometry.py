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
    BridgeBearAnchor,
    BridgeVisionAnalyzer,
    MissionState,
    Task1MissionController,
    normalize_angle,
)
from pros_car_py.simple_task_mission_controller import (
    BearAnchorController,
    MissionModeConfig,
    SimpleMissionState,
    SimpleTaskMissionController,
)
from pros_car_py.visual_corridor_controller import (
    VisualCorridorController,
    corridor_from_msg,
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
    node.state = MissionState.TASK2_SEARCH_BRIDGE
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
    node.task2_turn_centered_no_entry_timeout = 1.2
    node.task2_turn_allow_near_ramp_fallback = True
    node.task2_turn_near_ramp_min_confidence = 0.48
    node.task2_turn_near_ramp_min_vertical_coverage = 0.38
    node.task2_turn_no_entry_forward_pulse_seconds = 0.4
    node.task2_turn_centered_no_entry_start_time = None
    node.task2_turn_centered_frames = 0
    node.task2_top_use_tf_z = True
    node.task2_top_z_threshold = 0.18
    node.task2_top_min_ascent_seconds = 7.0
    node.task2_top_visual_confidence_threshold = 0.55
    node.task2_top_confirm_frames = 5
    node.task2_top_use_bear_depth = True
    node.task2_top_bear_depth_threshold = 0.37
    node.task2_top_bear_depth_confirm_frames = 3
    node.task2_top_bear_depth_max_center_error_pixels = 80.0
    node.task2_top_require_stable_stop_before_approach = True
    node.task2_top_platform_settle_seconds = 0.8
    node.task2_top_bear_depth_confirm_count = 0
    node.task2_use_bear_anchor_for_ascent = True
    node.task2_bear_anchor_min_confidence = 0.45
    node.task2_bear_anchor_max_age_seconds = 0.6
    node.task2_bear_anchor_pre_ascent_min_center_y_ratio = 0.08
    node.task2_bear_anchor_pre_ascent_max_center_y_ratio = 0.45
    node.task2_bear_anchor_pre_ascent_min_center_x_ratio = 0.35
    node.task2_bear_anchor_pre_ascent_max_center_x_ratio = 0.65
    node.task2_bear_anchor_ascent_min_center_x_ratio = 0.30
    node.task2_bear_anchor_ascent_max_center_x_ratio = 0.70
    node.task2_bear_anchor_center_tolerance_pixels = 45.0
    node.task2_bear_anchor_require_near_bridge_before_top = True
    node.task2_bear_anchor_memory_ttl_seconds = 10.0
    node.task2_bear_anchor_min_y_progress_ratio = 0.12
    node.task2_bear_anchor_depth_progress_min_m = 0.20
    node.task2_bear_anchor_progress_confirm_frames = 3
    node.task2_bear_anchor_first_y_ratio = None
    node.task2_bear_anchor_last_y_ratio = None
    node.task2_bear_anchor_first_depth = None
    node.task2_bear_anchor_best_depth = None
    node.task2_bear_anchor_last_seen_time = None
    node.task2_bear_anchor_progress_start_time = None
    node.task2_bear_anchor_progress_confirm_count = 0
    node.task2_bear_anchor_last_info = None
    node.task2_bear_anchor_last_memory_log_time = None
    node.task2_bridge_bear_approach_max_depth = 0.60
    node.task2_bridge_bear_observe_depth = 0.37
    node.task2_bridge_bear_grab_depth = 0.37
    node.task2_bridge_bear_reacquire_if_depth_above = 0.85
    node.bridge_top_confirmed = False
    node.task2_top_confirmed_time = None
    node.task2_ascent_timeout = 14.0
    node.task2_ascent_max_extra_seconds = 4.0
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
    node.yolo_target = None
    node.yolo_bbox = None
    node.yolo_target_stamp = None
    node.yolo_bbox_stamp = None
    node.target_surface_info = None
    node.target_surface_stamp = None
    node.target_timeout = 1.5
    node.bridge_bear_memory = BridgeBearMemory()
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
        "_task2_entry_source_detailed",
        "_task2_near_ramp_fallback_usable",
        "_task2_entry_delta",
        "_task2_bridge_entry_confirmed",
        "_task2_bridge_pre_entry_confirmed",
        "_task2_bridge_entry_delta",
        "_task2_bridge_pre_entry_delta",
        "_task2_turn_centered_no_entry_recovery_action",
        "_reset_bridge_turn_controller",
        "_task2_remember_bridge_delta",
        "_bridge_top_confidence",
        "_bridge_top_visual_score",
        "_bear_anchor_info",
        "_update_bear_anchor_tracking",
        "_bear_anchor_vertical_progress",
        "_bear_depth_top_platform_ok",
        "_bear_anchor_log_fields",
        "_bridge_ramp_log_fields",
        "_task2_top_recently_confirmed",
        "_reset_task2_bear_anchor_tracking",
        "_make_marker",
        "_line_marker",
        "_text_marker",
        "_target_surface_candidate",
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


def test_invalid_bridge_entry_is_not_smoothed_from_current_frame():
    node = DummyController()
    node.segmentation_smoothing_alpha = 0.45
    node.segmentation_last_seen = {"bridge": None}
    node.segmentation_info = None
    node._sanitize_bridge_entry_fields = (
        Task1MissionController._sanitize_bridge_entry_fields.__get__(node)
    )
    now = types.SimpleNamespace(nanoseconds=1_000_000_000)
    segment = {
        "found": True,
        "delta_x": 0.0,
        "area_ratio": 0.1,
        "bottom_coverage": 0.2,
        "entry_u": 320.0,
        "entry_v": 410.0,
        "entry_confidence": 0.8,
        "entry_depth": 0.5,
        "entry_from_road_connection": 1.0,
        "entry_confirmed": 0.0,
        "pre_entry_u": 318.0,
        "pre_entry_v": 450.0,
        "pre_entry_confidence": 0.7,
        "pre_entry_depth": 0.6,
        "target_u": 350.0,
        "target_v": 300.0,
        "target_confidence": 0.45,
    }

    result = Task1MissionController._filtered_segment_update(node, "bridge", segment, now)

    assert result["invalid_entry_suppressed"]
    assert result["entry_u"] == 0.0
    assert result["entry_depth"] == 0.0
    assert result["pre_entry_u"] == 0.0
    assert result["target_u"] == 350.0
    assert result["target_confidence"] == 0.45


def test_previous_valid_entry_does_not_leak_into_invalid_current_entry():
    node = DummyController()
    node.segmentation_smoothing_alpha = 0.45
    node.segmentation_last_seen = {"bridge": types.SimpleNamespace(nanoseconds=1)}
    node._sanitize_bridge_entry_fields = (
        Task1MissionController._sanitize_bridge_entry_fields.__get__(node)
    )
    node.segmentation_info = {
        "bridge": {
            "found": True,
            "raw_found": True,
            "delta_x": 0.0,
            "area_ratio": 0.1,
            "bottom_coverage": 0.2,
            "entry_u": 320.0,
            "entry_v": 410.0,
            "entry_confidence": 0.8,
            "entry_depth": 0.5,
            "entry_from_road_connection": 1.0,
            "entry_confirmed": 1.0,
            "pre_entry_u": 318.0,
            "pre_entry_v": 450.0,
            "pre_entry_confidence": 0.7,
            "pre_entry_depth": 0.6,
            "target_u": 320.0,
            "target_v": 300.0,
            "target_confidence": 0.45,
        }
    }
    now = types.SimpleNamespace(nanoseconds=2_000_000_000)
    segment = {
        "found": True,
        "delta_x": 0.0,
        "area_ratio": 0.1,
        "bottom_coverage": 0.2,
        "entry_u": 1e-20,
        "entry_v": 1e-20,
        "entry_confidence": 0.0,
        "entry_depth": 1e-20,
        "entry_from_road_connection": 0.0,
        "entry_confirmed": 0.0,
        "pre_entry_u": 1e-20,
        "pre_entry_v": 1e-20,
        "pre_entry_confidence": 0.0,
        "pre_entry_depth": 1e-20,
        "target_u": 360.0,
        "target_v": 305.0,
        "target_confidence": 0.5,
    }

    result = Task1MissionController._filtered_segment_update(node, "bridge", segment, now)

    assert result["entry_u"] == 0.0
    assert result["entry_v"] == 0.0
    assert result["pre_entry_depth"] == 0.0
    assert result["target_u"] > 320.0


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


def test_near_ramp_fallback_can_produce_entry_source_without_contact():
    node = _bridge_node()
    bridge = {
        "entry_confirmed": 0.0,
        "ramp_valid": 0.0,
        "ramp_confidence": 0.52,
        "ramp_lower_present": 1.0,
        "ramp_continuous": 1.0,
        "side_view_score": 0.20,
        "vertical_coverage_score": 0.55,
        "bottom_y_ratio": 0.78,
        "bottom_center_x": 320.0,
        "mid_center_x": 318.0,
        "center_x": 320.0,
        "raw_found": True,
    }

    source = "none"
    for _ in range(node.task2_ramp_entry_confirm_frames):
        detail = Task1MissionController._task2_entry_source_detailed(node, bridge)
        source = detail["source"] if detail["accepted"] else "none"

    assert source == "near_ramp_fallback"


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


def test_centered_bridge_with_no_entry_does_not_stop_forever():
    node = _bridge_node()
    node.get_clock = lambda: DummyClock()
    actions = []
    node._set_state = lambda state, reason="": setattr(node, "next_state", state)
    node._task2_road_explore_action = lambda: "LEFT_FRONT"
    bridge = {
        "raw_found": True,
        "ramp_valid": 0.0,
        "ramp_confidence": 0.50,
        "ramp_lower_present": 1.0,
        "ramp_continuous": 1.0,
        "side_view_score": 0.20,
        "vertical_coverage_score": 0.50,
        "bottom_center_x": 320.0,
        "mid_center_x": 320.0,
    }
    entry_detail = {
        "source": "none",
        "accepted": False,
        "reason": "test no entry",
    }

    action = Task1MissionController._task2_turn_centered_no_entry_recovery_action(
        node, bridge, entry_detail
    )
    actions.append(action)

    assert actions[-1] != "STOP"


def test_side_view_centered_no_entry_transitions_to_recovery():
    node = _bridge_node()
    node.get_clock = lambda: DummyClock()
    node._set_state = lambda state, reason="": setattr(node, "next_state", state)
    bridge = {
        "raw_found": True,
        "ramp_valid": 0.0,
        "ramp_confidence": 0.20,
        "ramp_continuous": 0.0,
        "side_view_score": 0.85,
        "bottom_center_x": 540.0,
        "mid_center_x": 560.0,
    }
    entry_detail = {"source": "none", "accepted": False, "reason": "side view"}

    action = Task1MissionController._task2_turn_centered_no_entry_recovery_action(
        node, bridge, entry_detail
    )

    assert action is None
    assert node.next_state == MissionState.TASK2_SIDE_VIEW_RECOVERY


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


def _set_anchor_bbox(node, center_x=320.0, center_y=120.0, depth=0.55, confidence=0.80):
    stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.yolo_target = {"found": True, "distance": depth, "delta_x": center_x - 320.0}
    node.yolo_bbox = {
        "found": True,
        "center_x": center_x,
        "center_y": center_y,
        "width": 80.0,
        "height": 80.0,
        "x1": center_x - 40.0,
        "y1": center_y - 40.0,
        "x2": center_x + 40.0,
        "y2": center_y + 40.0,
        "image_width": 640.0,
        "image_height": 480.0,
        "confidence": confidence,
        "distance": depth,
        "area": 6400.0,
        "area_ratio": 6400.0 / (640.0 * 480.0),
    }
    node.yolo_target_stamp = stamp
    node.yolo_bbox_stamp = stamp
    node.get_clock = lambda: DummyClock()


def test_bear_anchor_valid_before_ascent_when_upper_central_and_fresh():
    node = _bridge_node()
    _set_anchor_bbox(node, center_x=320.0, center_y=120.0, depth=0.55)

    info = BridgeBearAnchor.info(node)

    assert info["visible"]
    assert BridgeBearAnchor.valid_for_pre_ascent(node, info)


def test_bear_anchor_invalid_before_ascent_when_too_low_or_off_center():
    node = _bridge_node()
    _set_anchor_bbox(node, center_x=520.0, center_y=380.0, depth=0.55)

    info = BridgeBearAnchor.info(node)

    assert info["visible"]
    assert not BridgeBearAnchor.valid_for_pre_ascent(node, info)


def test_ascent_bear_anchor_horizontal_error_selects_arc_correction():
    node = _bridge_node()
    _set_anchor_bbox(node, center_x=380.0, center_y=180.0, depth=0.50)

    info = BridgeBearAnchor.info(node)
    action = BridgeBearAnchor.action(node, "ASCEND_FORWARD", phase="ascent", info=info)

    assert action == "RIGHT_FRONT"


def test_bear_vertical_progress_positive_when_bbox_moves_downward():
    node = _bridge_node()
    _set_anchor_bbox(node, center_x=320.0, center_y=120.0, depth=0.55)
    BridgeBearAnchor.info(node)
    _set_anchor_bbox(node, center_x=320.0, center_y=210.0, depth=0.48)

    info = BridgeBearAnchor.info(node)

    assert info["vertical_progress"] > 0.0


def test_bear_depth_confirms_top_platform_after_confirm_frames():
    node = _bridge_node()
    node.task2_top_use_tf_z = False
    _set_anchor_bbox(node, center_x=320.0, center_y=300.0, depth=0.35)

    ok = False
    for _ in range(node.task2_top_bear_depth_confirm_frames):
        ok, reason = Task1MissionController._bear_depth_top_platform_ok(node)

    assert ok
    assert "confirms bridge top" in reason


def test_top_confidence_can_be_confirmed_by_bear_depth():
    node = _bridge_node()
    node.pose_z = 0.0
    node.start_pose_z = 0.0
    node.task2_ascent_start_z = 0.0
    node.task2_top_use_tf_z = False
    node.task2_phase_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    phase_start = node.task2_phase_start_time
    node._elapsed_seconds = lambda start_time: 4.0 if start_time is phase_start else 0.0
    node._task2_bridge_visible = lambda allow_cached=False: {
        "ramp_confidence": 0.70,
        "bottom_y_ratio": 0.80,
        "side_view_score": 0.20,
    }
    _set_anchor_bbox(node, center_x=320.0, center_y=300.0, depth=0.35)

    ok = False
    for _ in range(node.task2_top_bear_depth_confirm_frames):
        ok, score, reason = Task1MissionController._bridge_top_confidence(node)

    assert ok
    assert score == 1.0
    assert "bear depth" in reason


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


def _corridor_node():
    node = _yolo_node_without_init()
    node.drivable_corridor_bottom_y_ratio = 0.88
    node.drivable_corridor_lower_y_ratio = 0.70
    node.drivable_corridor_center_y_ratio = 0.50
    node.drivable_corridor_center_band_height_ratio = 0.06
    node.drivable_corridor_min_bottom_width_ratio = 0.12
    node.drivable_corridor_min_center_width_ratio = 0.08
    node.drivable_corridor_min_continuous_score = 0.55
    node.drivable_corridor_center_tolerance_pixels = 45.0
    node.drivable_corridor_max_slope_pixels = 140.0
    return node


def test_drivable_corridor_valid_for_centered_road_connected_to_bottom():
    node = _corridor_node()
    road = np.zeros((480, 640), dtype=bool)
    road[235:480, 210:430] = True
    bridge = np.zeros_like(road)

    corridor = node._compute_drivable_corridor(road, bridge)

    assert corridor["corridor_valid"]
    assert corridor["bottom_connected"]
    assert corridor["centerline_reached"]
    assert corridor["drivable_type"] == 1.0
    assert abs(corridor["corridor_error_x_pixels"]) <= 2.0


def test_drivable_corridor_classifies_bridge_only_as_type_two():
    node = _corridor_node()
    road = np.zeros((480, 640), dtype=bool)
    bridge = np.zeros_like(road)
    bridge[230:480, 230:410] = True

    corridor = node._compute_drivable_corridor(road, bridge)

    assert corridor["corridor_valid"]
    assert corridor["drivable_type"] == 2.0


def test_drivable_corridor_rejects_upper_mask_not_connected_to_lower_frame():
    node = _corridor_node()
    road = np.zeros((480, 640), dtype=bool)
    road[60:190, 210:430] = True
    bridge = np.zeros_like(road)

    corridor = node._compute_drivable_corridor(road, bridge)

    assert not corridor["corridor_valid"]
    assert not corridor["bottom_connected"]
    assert corridor["reason_code"] == 2.0


def test_drivable_corridor_rejects_disconnected_bottom_and_center_masks():
    node = _corridor_node()
    road = np.zeros((480, 640), dtype=bool)
    road[420:480, 210:430] = True
    road[220:260, 210:430] = True
    bridge = np.zeros_like(road)

    corridor = node._compute_drivable_corridor(road, bridge)

    assert not corridor["corridor_valid"]
    assert corridor["reason_code"] in (3.0, 5.0)


def test_drivable_corridor_classifies_connected_road_bridge_as_mixed():
    node = _corridor_node()
    road = np.zeros((480, 640), dtype=bool)
    bridge = np.zeros_like(road)
    road[330:480, 220:420] = True
    bridge[230:360, 240:400] = True

    corridor = node._compute_drivable_corridor(road, bridge)

    assert corridor["corridor_valid"]
    assert corridor["drivable_type"] == 3.0
    assert corridor["bridge_ratio_in_corridor"] > 0.20


def test_drivable_corridor_rejects_side_sliver_as_side_view():
    node = _corridor_node()
    road = np.zeros((480, 640), dtype=bool)
    road[230:480, 520:560] = True
    bridge = np.zeros_like(road)

    corridor = node._compute_drivable_corridor(road, bridge)

    assert not corridor["corridor_valid"]
    assert corridor["side_view_likely"]


def test_corridor_from_msg_parses_requested_layout():
    data = [
        1.0,
        1.0,
        1.0,
        0.0,
        0.75,
        0.35,
        0.20,
        320.0,
        325.0,
        330.0,
        10.0,
        8.0,
        0.7,
        0.3,
        3.0,
        0.0,
        640.0,
        480.0,
        0.0,
    ]

    parsed = corridor_from_msg(data)

    assert parsed["valid"]
    assert parsed["bottom_connected"]
    assert parsed["drivable_type"] == 3
    assert parsed["error_x"] == 10.0


def test_visual_corridor_controller_allows_forward_when_centered():
    controller = VisualCorridorController(action_hold_seconds=0.0)
    info = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": 8.0,
        "drivable_type": 1,
        "side_view_likely": False,
        "bottom_width_ratio": 0.30,
        "center_width_ratio": 0.20,
    }

    result = controller.update(info, 0.1, desired_forward="FORWARD_SLOW", mode="road")

    assert result["allow_forward"]
    assert result["action"] == "FORWARD_SLOW"


def test_visual_corridor_controller_steers_instead_of_stopping_when_off_center():
    controller = VisualCorridorController(action_hold_seconds=0.0)
    info = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": 80.0,
        "drivable_type": 1,
        "side_view_likely": False,
        "bottom_width_ratio": 0.30,
        "center_width_ratio": 0.20,
    }

    result = controller.update(info, 0.1, desired_forward="FORWARD_SLOW", mode="road")

    assert result["action"] == "RIGHT_FRONT"
    assert result["pid_output"] > 0.0


def test_visual_corridor_controller_returns_left_correction_when_corridor_left():
    controller = VisualCorridorController(action_hold_seconds=0.0)
    info = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": -80.0,
        "drivable_type": 1,
        "side_view_likely": False,
        "bottom_width_ratio": 0.30,
        "center_width_ratio": 0.20,
    }

    result = controller.update(info, 0.1, desired_forward="FORWARD_SLOW", mode="road")

    assert result["action"] == "LEFT_FRONT"
    assert result["pid_output"] < 0.0


def test_visual_corridor_controller_no_mask_rotates_not_stop_forever():
    controller = VisualCorridorController(action_hold_seconds=0.0)

    result = controller.update({}, 0.1, desired_forward="FORWARD_SLOW", mode="road")

    assert result["action"] == "CLOCKWISE_ROTATION_SLOW"


def _simple_visual_gate_node(corridor):
    node = object.__new__(SimpleTaskMissionController)
    node.corridor_info = corridor
    node.corridor_stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.control_period_seconds = 0.1
    node.corridor_timeout_seconds = 0.7
    node.return_direct_fallback = True
    node.last_visual_override = None
    node.last_desired_action = None
    node.last_visual_gate_mode = ""
    node.last_visual_gate_reason = ""
    node.last_corridor_stale_warn_time = None
    node.visual_controller = VisualCorridorController(action_hold_seconds=0.0)
    node.bear_anchor_controller = BearAnchorController()
    node.bridge_corridor_min_bottom_width_ratio = 0.90
    node.bridge_corridor_min_center_width_ratio = 0.75
    node.bridge_corridor_min_continuous_score = 0.80
    node.bridge_corridor_center_tolerance_pixels = 35.0
    node.corridor_arc_tolerance_pixels = 120.0
    node.bridge_align_allow_forward = False
    node.bridge_align_allow_gentle_arc = True
    node.bear_anchor_enabled = True
    node.bear_anchor_center_tolerance_pixels = 35.0
    node.bear_anchor_arc_tolerance_pixels = 120.0
    node.bear_anchor_max_age_seconds = 0.6
    node.bear_anchor_min_confidence = 0.40
    node.bear_anchor_pre_ascent_min_y_ratio = 0.05
    node.bear_anchor_pre_ascent_max_y_ratio = 0.55
    node.bear_anchor_ascent_min_y_ratio = 0.05
    node.bear_anchor_ascent_max_y_ratio = 0.90
    node.top_bear_depth_threshold = 0.40
    node._stamp_age = lambda stamp: 0.0
    node._log_event = lambda *args, **kwargs: None
    node._search_action = lambda: "CLOCKWISE_ROTATION_SLOW"
    node._warn_stale_corridor = lambda reason: None
    node._fresh_target_visible = lambda: False
    node._fresh_bbox_visible = lambda: False
    return node


def test_simple_visual_gate_allows_ascent_forward_only_on_bridge_corridor():
    corridor = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": 0.0,
        "drivable_type": 2,
        "side_view_likely": False,
        "bottom_width_ratio": 0.95,
        "center_width_ratio": 0.80,
    }
    node = _simple_visual_gate_node(corridor)

    action = SimpleTaskMissionController._visual_safe_action(
        node, "ASCEND_FORWARD", "ascent"
    )

    assert action == "ASCEND_FORWARD"


def test_simple_visual_gate_allows_forward_slow_on_valid_road_corridor():
    corridor = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": 0.0,
        "drivable_type": 1,
        "side_view_likely": False,
        "bottom_width_ratio": 0.30,
        "center_width_ratio": 0.20,
    }
    node = _simple_visual_gate_node(corridor)

    action = SimpleTaskMissionController._visual_safe_action(
        node, "FORWARD_SLOW", "road"
    )

    assert action == "FORWARD_SLOW"


def test_simple_visual_gate_overrides_forward_slow_on_invalid_corridor():
    corridor = {
        "valid": False,
        "bottom_connected": True,
        "centerline_reached": False,
        "continuous_score": 0.2,
        "error_x": -70.0,
        "drivable_type": 1,
        "side_view_likely": False,
        "bottom_width_ratio": 0.30,
        "center_width_ratio": 0.0,
    }
    node = _simple_visual_gate_node(corridor)

    action = SimpleTaskMissionController._visual_safe_action(
        node, "FORWARD_SLOW", "road"
    )

    assert action == "COUNTERCLOCKWISE_ROTATION_SLOW"


def test_simple_visual_gate_stale_corridor_searches_without_forward():
    node = _simple_visual_gate_node(None)
    node.corridor_info = None

    action = SimpleTaskMissionController._visual_safe_action(
        node, "FORWARD_SLOW", "road"
    )

    assert action == "CLOCKWISE_ROTATION_SLOW"


def test_simple_visual_gate_rejects_forward_on_side_view_corridor():
    corridor = {
        "valid": False,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": 150.0,
        "drivable_type": 2,
        "side_view_likely": True,
        "bottom_width_ratio": 0.08,
        "center_width_ratio": 0.05,
    }
    node = _simple_visual_gate_node(corridor)

    action = SimpleTaskMissionController._visual_safe_action(
        node, "ASCEND_FORWARD", "ascent"
    )

    assert action in ("CLOCKWISE_ROTATION_SLOW", "COUNTERCLOCKWISE_ROTATION_SLOW")


def test_simple_top_platform_can_be_confirmed_by_bear_depth():
    node = object.__new__(SimpleTaskMissionController)
    node.pose_z = 0.0
    node.ascent_start_z = 0.0
    node.top_z_threshold = 0.18
    node.top_bear_depth_threshold = 0.40
    node.top_confirm_frames = 3
    node.top_confirm_count = 0
    node.top_confirmed = False
    node.top_confirm_reason = "not evaluated"
    node.yolo_target = {"distance": 0.35}
    node._fresh_target_visible = lambda: True
    node._log_event = lambda *args, **kwargs: None

    confirmed = False
    for _ in range(node.top_confirm_frames):
        confirmed = SimpleTaskMissionController._top_platform_confirmed(node)

    assert confirmed


def test_simple_search_drivable_does_not_stop_forever_when_no_corridor():
    node = object.__new__(SimpleTaskMissionController)
    node.mission_config = MissionModeConfig("bridge_first_shared_bear", 2)
    node.current_task = 2
    node._bridge_corridor_visible = lambda: False
    node._road_corridor_visible = lambda: False
    node._fresh_target_visible = lambda: False
    node._target_on_bridge = lambda: False
    node._search_action = lambda: "CLOCKWISE_ROTATION_SLOW"
    actions = []
    node._publish_action = actions.append

    SimpleTaskMissionController._state_search_drivable(node)

    assert actions[-1] != "STOP"


def test_simple_align_bridge_entry_does_not_stop_forever_when_not_ready():
    node = object.__new__(SimpleTaskMissionController)
    node.align_bridge_confirm_count = 0
    node.bridge_corridor_confirm_frames = 6
    node.bear_anchor_center_tolerance_pixels = 35.0
    node.bridge_entry_timeout_seconds = 45.0
    node._strict_bridge_corridor_ready = lambda: False
    node._bear_anchor_info = lambda: {
        "valid_pre_ascent": False,
        "error_x": 0.0,
    }
    node._bear_anchor_action = lambda base, phase: None
    node._state_elapsed = lambda: 1.0
    node._visual_safe_action = lambda desired, mode: "CLOCKWISE_ROTATION_SLOW"
    actions = []
    node._publish_action = actions.append

    SimpleTaskMissionController._state_align_bridge_entry(node)

    assert actions[-1] != "STOP"


def test_simple_stuck_recovery_creates_no_virtual_obstacles():
    node = object.__new__(SimpleTaskMissionController)
    node.stuck_detection_enabled = True
    node.pose = (0.0, 0.0, 0.0)
    node.stuck_reference_pose = (0.0, 0.0, 0.0)
    node.stuck_reference_time = types.SimpleNamespace(nanoseconds=0)
    node.stuck_recovery_step = None
    node.stuck_recovery_until = None
    node.stuck_check_seconds = 1.0
    node.stuck_min_translation = 0.04
    node.stuck_min_yaw_change = 0.08
    node.stuck_stop_seconds = 0.25
    node.stuck_back_seconds = 0.45
    node.stuck_rotate_seconds = 0.65
    node.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=2_000_000_000)
    )
    node._elapsed_seconds = lambda start_time: 2.0
    node._log_event = lambda *args, **kwargs: None

    action = SimpleTaskMissionController._apply_stuck_recovery(node, "FORWARD_SLOW")

    assert action == "STOP"
    assert not hasattr(node, "virtual_obstacles")


def test_simple_ascent_never_transitions_directly_to_approach_bear():
    node = object.__new__(SimpleTaskMissionController)
    node.ascent_start_time = types.SimpleNamespace(nanoseconds=0)
    node.ascent_start_z = 0.0
    node.pose_z = 0.2
    node.top_settle_start_time = types.SimpleNamespace(nanoseconds=0)
    node.top_settle_seconds = 0.1
    node.ascent_min_seconds = 8.0
    node.ascent_max_seconds = 16.0
    node.ascent_bear_depth_confirm_count = 0
    node._state_elapsed = lambda: 9.0
    node._elapsed_seconds = lambda start_time: 0.2
    node._top_platform_confirmed = lambda: True
    node._bear_depth_stop_ready = lambda: False
    node._fresh_target_visible = lambda: False
    node._publish_action = lambda action: None
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_ascend_bridge(node)

    assert node.state == SimpleMissionState.SEARCH_BEAR_ON_TOP


def test_simple_drop_after_task1_advances_to_task2_search():
    node = object.__new__(SimpleTaskMissionController)
    node.mission_config = MissionModeConfig("task1_then_task2", 1)
    node.current_task = 1
    node.task1_complete = False
    node.task2_complete = False
    node.bear_secured = True
    node._publish_action = lambda action: None
    node.drop_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.drop_step_index = 0
    node.drop_step_sent = False
    node.drop_step_deadline = None
    node.arm_lift_positions = [90.0, 90.0, 20.0]
    node.arm_drop_positions = [65.0, 190.0, 20.0]
    node.arm_release_positions = [65.0, 190.0, 90.0]
    node.arm_drop_retract_positions = [180.0, 0.0, 90.0]
    node._run_arm_sequence = lambda sequence, name: True
    node._reset_drop_sequence = lambda: None
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_drop_bear(node)

    assert node.task1_complete
    assert node.current_task == 2
    assert node.state == SimpleMissionState.SEARCH_DRIVABLE


def test_shared_bear_mode_starts_with_task_two():
    config = MissionModeConfig("bridge_first_shared_bear", 1)

    assert config.initial_task() == 2
    assert config.bridge_first_shared_bear


def test_shared_bear_drop_marks_both_tasks_complete():
    node = object.__new__(SimpleTaskMissionController)
    node.mission_config = MissionModeConfig("bridge_first_shared_bear", 2)
    node.current_task = 2
    node.task1_complete = False
    node.task2_complete = False
    node.bear_secured = True
    node._publish_action = lambda action: None
    node.drop_start_time = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.drop_step_index = 0
    node.drop_step_sent = False
    node.drop_step_deadline = None
    node.arm_lift_positions = [90.0, 90.0, 20.0]
    node.arm_drop_positions = [65.0, 190.0, 20.0]
    node.arm_release_positions = [65.0, 190.0, 90.0]
    node.arm_drop_retract_positions = [180.0, 0.0, 90.0]
    node._run_arm_sequence = lambda sequence, name: True
    node._reset_drop_sequence = lambda: None
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_drop_bear(node)

    assert node.task1_complete
    assert node.task2_complete
    assert node.state == SimpleMissionState.DONE


def test_align_bridge_transitions_to_approach_only_after_confirm_frames():
    node = object.__new__(SimpleTaskMissionController)
    node.align_bridge_confirm_count = 5
    node.bridge_corridor_confirm_frames = 6
    node.bear_anchor_center_tolerance_pixels = 35.0
    node.bridge_entry_timeout_seconds = 45.0
    node._strict_bridge_corridor_ready = lambda: True
    node._bear_anchor_info = lambda: {"valid_pre_ascent": False, "error_x": 0.0}
    node._state_elapsed = lambda: 1.0
    node._log_event = lambda *args, **kwargs: None
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_align_bridge_entry(node)

    assert node.state == SimpleMissionState.APPROACH_BRIDGE_ENTRY


def test_approach_bridge_entry_transitions_to_ascent_after_strict_stability():
    node = object.__new__(SimpleTaskMissionController)
    node.bridge_entry_approach_start_time = types.SimpleNamespace(nanoseconds=0)
    node.approach_bridge_confirm_count = 5
    node.bridge_entry_approach_confirm_frames = 6
    node.bridge_entry_approach_min_seconds = 1.0
    node.bridge_entry_approach_max_seconds = 5.0
    node._strict_bridge_corridor_ready = lambda: True
    node._elapsed_seconds = lambda start_time: 1.5
    node._log_event = lambda *args, **kwargs: None
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_approach_bridge_entry(node)

    assert node.state == SimpleMissionState.ASCEND_BRIDGE


def test_loose_road_corridor_does_not_satisfy_strict_bridge_corridor():
    corridor = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.9,
        "error_x": 0.0,
        "drivable_type": 1,
        "side_view_likely": False,
        "bottom_width_ratio": 0.95,
        "center_width_ratio": 0.80,
    }
    node = _simple_visual_gate_node(corridor)

    assert not SimpleTaskMissionController._strict_bridge_corridor_ready(node)


def _bear_anchor_node(
    center_x=320.0,
    center_y=160.0,
    confidence=0.8,
    distance=0.8,
    age=0.0,
):
    node = object.__new__(SimpleTaskMissionController)
    node.bear_anchor_enabled = True
    node.bear_anchor_max_age_seconds = 0.6
    node.bear_anchor_min_confidence = 0.40
    node.bear_anchor_pre_ascent_min_y_ratio = 0.05
    node.bear_anchor_pre_ascent_max_y_ratio = 0.55
    node.bear_anchor_ascent_min_y_ratio = 0.05
    node.bear_anchor_ascent_max_y_ratio = 0.90
    node.target_timeout_seconds = 1.0
    node.yolo_target = {"found": True, "distance": distance, "delta_x": center_x - 320.0}
    node.yolo_bbox = {
        "found": True,
        "center_x": center_x,
        "center_y": center_y,
        "image_width": 640.0,
        "image_height": 480.0,
        "confidence": confidence,
        "distance": distance,
        "area": 3600.0,
        "area_ratio": 3600.0 / (640.0 * 480.0),
    }
    stamp = types.SimpleNamespace(nanoseconds=1_000_000_000)
    node.yolo_target_stamp = stamp
    node.yolo_bbox_stamp = stamp
    node.target_surface_info = None
    node.target_surface_stamp = None
    node._stamp_age = lambda _stamp: age
    node._fresh_target_visible = lambda: True
    node._fresh_bbox_visible = lambda: True
    return node


def test_bear_anchor_valid_when_fresh_confident_and_in_vertical_band():
    node = _bear_anchor_node(center_x=320.0, center_y=160.0, confidence=0.8)

    anchor = SimpleTaskMissionController._bear_anchor_info(node)

    assert anchor["valid_pre_ascent"]
    assert anchor["valid_ascent"]


def test_bear_anchor_invalid_when_stale_low_confidence_or_too_low():
    stale = _bear_anchor_node(age=1.0)
    low_conf = _bear_anchor_node(confidence=0.2)
    too_low = _bear_anchor_node(center_y=470.0)

    assert not SimpleTaskMissionController._bear_anchor_info(stale)["valid_pre_ascent"]
    assert not SimpleTaskMissionController._bear_anchor_info(low_conf)["valid_pre_ascent"]
    assert not SimpleTaskMissionController._bear_anchor_info(too_low)["valid_pre_ascent"]


def test_bear_anchor_action_returns_right_and_left_corrections():
    controller = BearAnchorController(action_hold_seconds=0.0) if False else BearAnchorController()
    right = {"visible": True, "error_x": 70.0}
    left = {"visible": True, "error_x": -70.0}

    assert controller.action(right, "ASCEND_FORWARD", 0.1) == "RIGHT_FRONT"
    assert controller.action(left, "ASCEND_FORWARD", 0.1) == "LEFT_FRONT"


def test_ascent_uses_bear_anchor_before_corridor_fallback():
    node = object.__new__(SimpleTaskMissionController)
    node.ascent_start_time = types.SimpleNamespace(nanoseconds=0)
    node.ascent_start_z = 0.0
    node.pose_z = 0.0
    node.top_settle_start_time = None
    node.top_confirm_count = 0
    node.ascent_bear_depth_confirm_count = 0
    node.top_confirmed = False
    node.top_confirm_reason = ""
    node.ascent_min_seconds = 8.0
    node.ascent_max_seconds = 16.0
    node._state_elapsed = lambda: 1.0
    node._bear_depth_stop_ready = lambda: False
    node._top_platform_confirmed = lambda: False
    node._bear_anchor_action = lambda base, phase: "RIGHT_FRONT"
    node._visual_safe_action = lambda action, mode: action
    actions = []
    node._publish_action = actions.append

    SimpleTaskMissionController._state_ascend_bridge(node)

    assert actions[-1] == "RIGHT_FRONT"


def test_bear_depth_stop_transitions_ascent_to_observe_bear():
    node = object.__new__(SimpleTaskMissionController)
    node.ascent_start_time = types.SimpleNamespace(nanoseconds=0)
    node.ascent_start_z = 0.0
    node.pose_z = 0.0
    node.top_settle_start_time = None
    node.top_confirm_count = 0
    node.ascent_bear_depth_confirm_count = 3
    node.top_confirmed = False
    node.top_confirm_reason = ""
    node._bear_depth_stop_ready = lambda: True
    node._publish_action = lambda action: None
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_ascend_bridge(node)

    assert node.state == SimpleMissionState.OBSERVE_BEAR


def test_approach_grab_does_not_drive_forward_under_bridge_depth_stop():
    node = object.__new__(SimpleTaskMissionController)
    node.current_task = 2
    node.yolo_target = {"distance": 0.35, "delta_x": 0.0}
    node.yolo_bbox = {"found": True}
    node.grab_align_pixel_tolerance = 28.0
    node.grab_distance = 0.30
    node.top_bear_depth_threshold = 0.40
    node._fresh_target_visible = lambda: True
    node._publish_action = lambda action: setattr(node, "published_action", action)
    node._set_state = lambda state, reason="": setattr(node, "state", state)

    SimpleTaskMissionController._state_approach_grab(node)

    assert node.state == SimpleMissionState.SECURE_BEAR
    assert not hasattr(node, "published_action")


def test_stuck_recovery_does_not_activate_for_pure_rotation():
    node = object.__new__(SimpleTaskMissionController)
    node.stuck_detection_enabled = True
    node.pose = (0.0, 0.0, 0.0)
    node.stuck_reference_pose = (0.0, 0.0, 0.0)
    node.stuck_reference_time = types.SimpleNamespace(nanoseconds=0)
    node.stuck_recovery_step = None
    node.stuck_recovery_until = None
    node.get_clock = lambda: types.SimpleNamespace(
        now=lambda: types.SimpleNamespace(nanoseconds=2_000_000_000)
    )

    action = SimpleTaskMissionController._apply_stuck_recovery(
        node, "CLOCKWISE_ROTATION_SLOW"
    )

    assert action == "CLOCKWISE_ROTATION_SLOW"


def test_search_action_uses_bridge_aware_mode_for_task_two():
    corridor = {
        "valid": True,
        "bottom_connected": True,
        "centerline_reached": True,
        "continuous_score": 0.8,
        "error_x": 80.0,
        "drivable_type": 2,
        "side_view_likely": False,
        "bottom_width_ratio": 0.40,
        "center_width_ratio": 0.30,
    }
    node = _simple_visual_gate_node(corridor)
    node.current_task = 2

    action = SimpleTaskMissionController._search_action(node)

    assert action == "RIGHT_FRONT"

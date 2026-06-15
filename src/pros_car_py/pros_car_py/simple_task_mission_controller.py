import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, String
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from pros_car_py.car_models import DeviceDataTypeEnum
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS
from pros_car_py.visual_corridor_controller import (
    VisualCorridorController,
    corridor_from_msg,
)


class SimpleMissionState(str, Enum):
    INIT = "init"
    SEARCH_DRIVABLE = "search_drivable"
    FOLLOW_ROAD_TO_BRIDGE = "follow_road_to_bridge"
    ALIGN_BRIDGE_ENTRY = "align_bridge_entry"
    APPROACH_BRIDGE_ENTRY = "approach_bridge_entry"
    ASCEND_BRIDGE = "ascend_bridge"
    SEARCH_BEAR_ON_TOP = "search_bear_on_top"
    OBSERVE_BEAR = "observe_bear"
    APPROACH_GRAB = "approach_grab"
    SECURE_BEAR = "secure_bear"
    VERIFY_GRAB = "verify_grab"
    RETURN_START = "return_start"
    DROP_BEAR = "drop_bear"
    DONE = "done"


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def distance_2d(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def normalized_depth_meters(distance):
    distance = float(distance)
    if distance > 10.0:
        return distance / 100.0
    return distance


@dataclass
class MissionModeConfig:
    mission_mode: str
    start_task: int

    @property
    def bridge_first_shared_bear(self):
        return self.mission_mode == "bridge_first_shared_bear"

    def initial_task(self):
        return 2 if self.bridge_first_shared_bear else int(self.start_task)


class BearAnchorController:
    """Small PID-style helper for keeping the bridge bear centered."""

    FORWARD_ACTIONS = VisualCorridorController.FORWARD_ACTIONS

    def __init__(
        self,
        kp=0.018,
        ki=0.0,
        kd=0.004,
        center_tolerance_pixels=35.0,
        arc_tolerance_pixels=120.0,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.center_tolerance_pixels = float(center_tolerance_pixels)
        self.arc_tolerance_pixels = float(arc_tolerance_pixels)
        self.integral = 0.0
        self.previous_error = None

    def reset(self):
        self.integral = 0.0
        self.previous_error = None

    def action(self, anchor, base_forward_action, dt):
        if not anchor.get("visible", False):
            self.reset()
            return None
        error = float(anchor.get("error_x", 0.0))
        dt = max(0.001, float(dt))
        self.integral = max(-90.0, min(90.0, self.integral + error * dt))
        derivative = 0.0
        if self.previous_error is not None:
            derivative = (error - self.previous_error) / dt
        self.previous_error = error
        _ = self.kp * error + self.ki * self.integral + self.kd * derivative

        if abs(error) <= self.center_tolerance_pixels:
            return base_forward_action
        if abs(error) <= self.arc_tolerance_pixels:
            return "RIGHT_FRONT" if error > 0.0 else "LEFT_FRONT"
        return "CLOCKWISE_ROTATION_SLOW" if error > 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"


class SimpleTaskMissionController(Node):
    """Corridor-first mission controller for task 1 and task 2.

    This node intentionally does not copy the older bridge-side point cloud,
    augmented map, virtual obstacle, or marker pipeline. It treats the YOLO
    drivable corridor topic as the authority for forward-like movement.
    """

    def __init__(self):
        super().__init__("simple_task_mission_controller")

        self._declare_parameters()
        self._load_parameters()

        self.state = SimpleMissionState.INIT
        self.state_start_time = self.get_clock().now()
        self.mission_start_time = self.state_start_time
        self.mission_config = MissionModeConfig(self.mission_mode, self.start_task)
        self.current_task = self.mission_config.initial_task()
        self.task1_complete = False
        self.task2_complete = False
        self.pose = None
        self.pose_z = 0.0
        self.start_pose = None
        self.start_pose_z = 0.0
        self.map_msg = None
        self.latest_path = None
        self.yolo_target = None
        self.yolo_target_stamp = None
        self.yolo_bbox = None
        self.yolo_bbox_stamp = None
        self.target_surface_info = None
        self.target_surface_stamp = None
        self.segmentation_info = None
        self.segmentation_stamp = None
        self.corridor_info = None
        self.corridor_stamp = None
        self.last_action = None
        self.last_action_log_time = None
        self.last_desired_action = None
        self.last_visual_gate_mode = ""
        self.last_visual_gate_reason = ""
        self.last_visual_override = None
        self.last_corridor_stale_warn_time = None
        self.bear_secured = False
        self.grab_retry_count = 0
        self.observe_start_time = None
        self.grab_start_time = None
        self.grab_step_index = 0
        self.grab_step_sent = False
        self.grab_step_deadline = None
        self.drop_start_time = None
        self.drop_step_index = 0
        self.drop_step_sent = False
        self.drop_step_deadline = None
        self.verify_start_time = None
        self.verify_seen_start_time = None
        self.pre_grab_bbox = None
        self.ascent_start_time = None
        self.ascent_start_z = 0.0
        self.top_settle_start_time = None
        self.top_confirm_count = 0
        self.top_confirmed = False
        self.top_confirm_reason = "not evaluated"
        self.align_bridge_confirm_count = 0
        self.approach_bridge_confirm_count = 0
        self.ascent_bear_depth_confirm_count = 0
        self.bridge_entry_approach_start_time = None
        self.arm_home_sent = False
        self.stuck_reference_pose = None
        self.stuck_reference_time = self.get_clock().now()
        self.stuck_recovery_step = None
        self.stuck_recovery_until = None

        self.visual_controller = VisualCorridorController(
            kp=self.corridor_kp,
            ki=self.corridor_ki,
            kd=self.corridor_kd,
            max_error=self.corridor_max_error_pixels,
            forward_deadband=self.corridor_forward_tolerance_pixels,
            rotate_deadband=self.corridor_arc_tolerance_pixels,
            action_hold_seconds=self.corridor_action_hold_seconds,
            min_continuous_score=self.corridor_min_continuous_score,
            min_bottom_width_ratio=self.corridor_min_bottom_width_ratio,
            min_center_width_ratio=self.corridor_min_center_width_ratio,
        )
        self.bear_anchor_controller = BearAnchorController(
            kp=self.bear_anchor_kp,
            ki=self.bear_anchor_ki,
            kd=self.bear_anchor_kd,
            center_tolerance_pixels=self.bear_anchor_center_tolerance_pixels,
            arc_tolerance_pixels=self.bear_anchor_arc_tolerance_pixels,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rear_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10
        )
        self.front_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10
        )
        self.arm_pub = self.create_publisher(
            JointTrajectoryPoint, DeviceDataTypeEnum.robot_arm, 10
        )
        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.target_label_pub = self.create_publisher(String, "/target_label", 10)
        self.state_pub = self.create_publisher(String, "/task1/state", 10)
        self.debug_pub = self.create_publisher(String, "/simple_task_mission/debug", 10)

        self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self._amcl_pose_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, "/yolo/target_info", self._target_callback, 10
        )
        self.create_subscription(
            Float32MultiArray, "/yolo/target_bbox", self._bbox_callback, 10
        )
        self.create_subscription(
            Float32MultiArray,
            "/yolo/target_surface_info",
            self._target_surface_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/yolo/segmentation_info",
            self._segmentation_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/yolo/drivable_corridor_info",
            self._corridor_callback,
            10,
        )
        self.create_subscription(OccupancyGrid, "/map", self._map_callback, 10)
        self.create_subscription(Path, "/received_global_plan", self._path_callback, 10)

        self.control_timer = self.create_timer(
            self.control_period_seconds, self._control_loop
        )
        self._setup_file_logging()
        self._log_event("bridge_obstacle_logic_disabled")
        self.get_logger().info(
            "Simple task mission ready. It will use /yolo/drivable_corridor_info "
            "to gate road and bridge movement."
        )

    def _declare_parameters(self):
        self.declare_parameter("target_label", "bear")
        self.declare_parameter("mission_mode", "bridge_first_shared_bear")
        self.declare_parameter("start_task", 2)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("use_tf_pose_fallback", True)
        self.declare_parameter("control_period_seconds", 0.10)
        self.declare_parameter("mission_log_directory", "/tmp/simple_task_mission_logs")
        self.declare_parameter("mission_log_to_file", True)
        self.declare_parameter("corridor_timeout_seconds", 0.60)
        self.declare_parameter("corridor_min_continuous_score", 0.55)
        self.declare_parameter("corridor_min_bottom_width_ratio", 0.12)
        self.declare_parameter("corridor_min_center_width_ratio", 0.08)
        self.declare_parameter("corridor_forward_tolerance_pixels", 45.0)
        self.declare_parameter("corridor_arc_tolerance_pixels", 120.0)
        self.declare_parameter("corridor_action_hold_seconds", 0.2)
        self.declare_parameter("corridor_max_error_pixels", 180.0)
        self.declare_parameter("corridor_kp", 0.018)
        self.declare_parameter("corridor_ki", 0.0)
        self.declare_parameter("corridor_kd", 0.004)
        self.declare_parameter("bridge_corridor_min_bottom_width_ratio", 0.90)
        self.declare_parameter("bridge_corridor_min_center_width_ratio", 0.75)
        self.declare_parameter("bridge_corridor_min_continuous_score", 0.80)
        self.declare_parameter("bridge_corridor_center_tolerance_pixels", 35.0)
        self.declare_parameter("bridge_corridor_confirm_frames", 6)
        self.declare_parameter("bridge_entry_approach_confirm_frames", 6)
        self.declare_parameter("bridge_entry_approach_min_seconds", 1.0)
        self.declare_parameter("bridge_entry_approach_max_seconds", 5.0)
        self.declare_parameter("bridge_align_allow_gentle_arc", True)
        self.declare_parameter("bridge_align_allow_forward", False)
        self.declare_parameter("bridge_entry_timeout_seconds", 45.0)
        self.declare_parameter("ascent_forward_speed_scale", 1.25)
        self.declare_parameter("ascent_min_seconds", 8.0)
        self.declare_parameter("ascent_max_seconds", 16.0)
        self.declare_parameter("top_z_threshold_m", 0.18)
        self.declare_parameter("top_bear_depth_threshold_m", 0.40)
        self.declare_parameter("top_bear_depth_confirm_frames", 3)
        self.declare_parameter("top_bear_depth_center_tolerance_pixels", 90.0)
        self.declare_parameter("top_settle_seconds", 0.8)
        self.declare_parameter("bear_anchor_enabled", True)
        self.declare_parameter("bear_anchor_kp", 0.018)
        self.declare_parameter("bear_anchor_ki", 0.0)
        self.declare_parameter("bear_anchor_kd", 0.004)
        self.declare_parameter("bear_anchor_center_tolerance_pixels", 35.0)
        self.declare_parameter("bear_anchor_arc_tolerance_pixels", 120.0)
        self.declare_parameter("bear_anchor_max_age_seconds", 0.6)
        self.declare_parameter("bear_anchor_min_confidence", 0.40)
        self.declare_parameter("bear_anchor_pre_ascent_min_y_ratio", 0.05)
        self.declare_parameter("bear_anchor_pre_ascent_max_y_ratio", 0.55)
        self.declare_parameter("bear_anchor_ascent_min_y_ratio", 0.05)
        self.declare_parameter("bear_anchor_ascent_max_y_ratio", 0.90)
        self.declare_parameter("stuck_detection_enabled", True)
        self.declare_parameter("stuck_check_seconds", 2.5)
        self.declare_parameter("stuck_min_translation_m", 0.04)
        self.declare_parameter("stuck_min_yaw_change_rad", 0.08)
        self.declare_parameter("stuck_stop_seconds", 0.25)
        self.declare_parameter("stuck_back_seconds", 0.45)
        self.declare_parameter("stuck_rotate_seconds", 0.65)
        self.declare_parameter("observe_distance", 1.0)
        self.declare_parameter("observe_seconds", 5.0)
        self.declare_parameter("grab_distance", 0.40)
        self.declare_parameter("align_pixel_tolerance", 65.0)
        self.declare_parameter("grab_align_pixel_tolerance", 28.0)
        self.declare_parameter("target_timeout_seconds", 1.0)
        self.declare_parameter("bbox_timeout_seconds", 1.0)
        self.declare_parameter("return_goal_tolerance", 0.42)
        self.declare_parameter("return_heading_tolerance", 0.25)
        self.declare_parameter("return_direct_fallback", True)
        self.declare_parameter("verify_grab_seconds", 0.8)
        self.declare_parameter("verify_grab_timeout_seconds", 4.0)
        self.declare_parameter("verify_grab_retry_limit", 2)
        self.declare_parameter("verify_grab_max_depth", 0.75)
        self.declare_parameter("verify_grab_min_bbox_area_ratio", 0.012)
        self.declare_parameter("verify_grab_lift_pixel_delta", 25.0)
        self.declare_parameter("arm_positions_in_degrees", True)
        self.declare_parameter("arm_home_positions", [180.0, 0.0, 90.0])
        self.declare_parameter("arm_open_positions", [180.0, 0.0, 90.0])
        self.declare_parameter("arm_pregrasp_positions", [90.0, 120.0, 90.0])
        self.declare_parameter("arm_reach_positions", [65.0, 190.0, 90.0])
        self.declare_parameter("arm_close_positions", [65.0, 190.0, 20.0])
        self.declare_parameter("arm_lift_positions", [90.0, 90.0, 20.0])
        self.declare_parameter("arm_carry_positions", [180.0, 0.0, 20.0])
        self.declare_parameter("arm_drop_positions", [65.0, 190.0, 20.0])
        self.declare_parameter("arm_release_positions", [65.0, 190.0, 90.0])
        self.declare_parameter("arm_drop_retract_positions", [180.0, 0.0, 90.0])

    def _load_parameters(self):
        self.target_label = self._string_param("target_label")
        self.mission_mode = self._string_param("mission_mode")
        self.start_task = self._integer_param("start_task")
        self.map_frame = self._string_param("map_frame")
        self.base_frame = self._string_param("base_frame")
        self.use_tf_pose_fallback = self._bool_param("use_tf_pose_fallback")
        self.control_period_seconds = self._double_param("control_period_seconds")
        self.mission_log_directory = self._string_param("mission_log_directory")
        self.mission_log_to_file = self._bool_param("mission_log_to_file")
        self.corridor_timeout_seconds = self._double_param("corridor_timeout_seconds")
        self.corridor_min_continuous_score = self._double_param(
            "corridor_min_continuous_score"
        )
        self.corridor_min_bottom_width_ratio = self._double_param(
            "corridor_min_bottom_width_ratio"
        )
        self.corridor_min_center_width_ratio = self._double_param(
            "corridor_min_center_width_ratio"
        )
        self.corridor_forward_tolerance_pixels = self._double_param(
            "corridor_forward_tolerance_pixels"
        )
        self.corridor_arc_tolerance_pixels = self._double_param(
            "corridor_arc_tolerance_pixels"
        )
        self.corridor_action_hold_seconds = self._double_param(
            "corridor_action_hold_seconds"
        )
        self.corridor_max_error_pixels = self._double_param("corridor_max_error_pixels")
        self.corridor_kp = self._double_param("corridor_kp")
        self.corridor_ki = self._double_param("corridor_ki")
        self.corridor_kd = self._double_param("corridor_kd")
        self.bridge_corridor_min_bottom_width_ratio = self._double_param(
            "bridge_corridor_min_bottom_width_ratio"
        )
        self.bridge_corridor_min_center_width_ratio = self._double_param(
            "bridge_corridor_min_center_width_ratio"
        )
        self.bridge_corridor_min_continuous_score = self._double_param(
            "bridge_corridor_min_continuous_score"
        )
        self.bridge_corridor_center_tolerance_pixels = self._double_param(
            "bridge_corridor_center_tolerance_pixels"
        )
        self.bridge_corridor_confirm_frames = self._integer_param(
            "bridge_corridor_confirm_frames"
        )
        self.bridge_entry_approach_confirm_frames = self._integer_param(
            "bridge_entry_approach_confirm_frames"
        )
        self.bridge_entry_approach_min_seconds = self._double_param(
            "bridge_entry_approach_min_seconds"
        )
        self.bridge_entry_approach_max_seconds = self._double_param(
            "bridge_entry_approach_max_seconds"
        )
        self.bridge_align_allow_gentle_arc = self._bool_param("bridge_align_allow_gentle_arc")
        self.bridge_align_allow_forward = self._bool_param("bridge_align_allow_forward")
        self.bridge_entry_timeout_seconds = self._double_param(
            "bridge_entry_timeout_seconds"
        )
        self.ascent_forward_speed_scale = self._double_param(
            "ascent_forward_speed_scale"
        )
        self.ascent_min_seconds = self._double_param("ascent_min_seconds")
        self.ascent_max_seconds = self._double_param("ascent_max_seconds")
        self.top_z_threshold = self._double_param("top_z_threshold_m")
        self.top_bear_depth_threshold = self._double_param(
            "top_bear_depth_threshold_m"
        )
        self.top_confirm_frames = self._integer_param("top_bear_depth_confirm_frames")
        self.top_bear_depth_center_tolerance_pixels = self._double_param(
            "top_bear_depth_center_tolerance_pixels"
        )
        self.top_settle_seconds = self._double_param("top_settle_seconds")
        self.bear_anchor_enabled = self._bool_param("bear_anchor_enabled")
        self.bear_anchor_kp = self._double_param("bear_anchor_kp")
        self.bear_anchor_ki = self._double_param("bear_anchor_ki")
        self.bear_anchor_kd = self._double_param("bear_anchor_kd")
        self.bear_anchor_center_tolerance_pixels = self._double_param(
            "bear_anchor_center_tolerance_pixels"
        )
        self.bear_anchor_arc_tolerance_pixels = self._double_param(
            "bear_anchor_arc_tolerance_pixels"
        )
        self.bear_anchor_max_age_seconds = self._double_param(
            "bear_anchor_max_age_seconds"
        )
        self.bear_anchor_min_confidence = self._double_param(
            "bear_anchor_min_confidence"
        )
        self.bear_anchor_pre_ascent_min_y_ratio = self._double_param(
            "bear_anchor_pre_ascent_min_y_ratio"
        )
        self.bear_anchor_pre_ascent_max_y_ratio = self._double_param(
            "bear_anchor_pre_ascent_max_y_ratio"
        )
        self.bear_anchor_ascent_min_y_ratio = self._double_param(
            "bear_anchor_ascent_min_y_ratio"
        )
        self.bear_anchor_ascent_max_y_ratio = self._double_param(
            "bear_anchor_ascent_max_y_ratio"
        )
        self.stuck_detection_enabled = self._bool_param("stuck_detection_enabled")
        self.stuck_check_seconds = self._double_param("stuck_check_seconds")
        self.stuck_min_translation = self._double_param("stuck_min_translation_m")
        self.stuck_min_yaw_change = self._double_param("stuck_min_yaw_change_rad")
        self.stuck_stop_seconds = self._double_param("stuck_stop_seconds")
        self.stuck_back_seconds = self._double_param("stuck_back_seconds")
        self.stuck_rotate_seconds = self._double_param("stuck_rotate_seconds")
        self.observe_distance = self._double_param("observe_distance")
        self.observe_seconds = self._double_param("observe_seconds")
        self.grab_distance = self._double_param("grab_distance")
        self.align_pixel_tolerance = self._double_param("align_pixel_tolerance")
        self.grab_align_pixel_tolerance = self._double_param("grab_align_pixel_tolerance")
        self.target_timeout_seconds = self._double_param("target_timeout_seconds")
        self.bbox_timeout_seconds = self._double_param("bbox_timeout_seconds")
        self.return_goal_tolerance = self._double_param("return_goal_tolerance")
        self.return_heading_tolerance = self._double_param("return_heading_tolerance")
        self.return_direct_fallback = self._bool_param("return_direct_fallback")
        self.verify_grab_seconds = self._double_param("verify_grab_seconds")
        self.verify_grab_timeout_seconds = self._double_param(
            "verify_grab_timeout_seconds"
        )
        self.verify_grab_retry_limit = self._integer_param("verify_grab_retry_limit")
        self.verify_grab_max_depth = self._double_param("verify_grab_max_depth")
        self.verify_grab_min_bbox_area_ratio = self._double_param(
            "verify_grab_min_bbox_area_ratio"
        )
        self.verify_grab_lift_pixel_delta = self._double_param(
            "verify_grab_lift_pixel_delta"
        )
        self.arm_positions_in_degrees = self._bool_param("arm_positions_in_degrees")
        self.arm_home_positions = self._double_array_param("arm_home_positions")
        self.arm_open_positions = self._double_array_param("arm_open_positions")
        self.arm_pregrasp_positions = self._double_array_param("arm_pregrasp_positions")
        self.arm_reach_positions = self._double_array_param("arm_reach_positions")
        self.arm_close_positions = self._double_array_param("arm_close_positions")
        self.arm_lift_positions = self._double_array_param("arm_lift_positions")
        self.arm_carry_positions = self._double_array_param("arm_carry_positions")
        self.arm_drop_positions = self._double_array_param("arm_drop_positions")
        self.arm_release_positions = self._double_array_param("arm_release_positions")
        self.arm_drop_retract_positions = self._double_array_param(
            "arm_drop_retract_positions"
        )

    def _string_param(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _bool_param(self, name):
        return self.get_parameter(name).get_parameter_value().bool_value

    def _double_param(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def _integer_param(self, name):
        return self.get_parameter(name).get_parameter_value().integer_value

    def _double_array_param(self, name):
        values = self.get_parameter(name).get_parameter_value().double_array_value
        return [float(value) for value in values]

    def _setup_file_logging(self):
        self.log_file = None
        self.jsonl_file = None
        self.mission_id = datetime.now().strftime("%Y%m%d_%H%M%S_simple")
        if not self.mission_log_to_file:
            return
        os.makedirs(self.mission_log_directory, exist_ok=True)
        base = os.path.join(self.mission_log_directory, self.mission_id)
        self.log_file = open(base + ".log", "a", encoding="utf-8")
        self.jsonl_file = open(base + ".jsonl", "a", encoding="utf-8")

    def _amcl_pose_callback(self, msg):
        pose = msg.pose.pose
        self.pose = (
            float(pose.position.x),
            float(pose.position.y),
            yaw_from_quaternion(pose.orientation),
        )
        self.pose_z = float(pose.position.z)

    def _target_callback(self, msg):
        data = list(msg.data)
        if len(data) < 3:
            return
        self.yolo_target = {
            "found": data[0] >= 0.5,
            "distance": normalized_depth_meters(data[1]),
            "delta_x": float(data[2]),
        }
        self.yolo_target_stamp = self.get_clock().now()

    def _bbox_callback(self, msg):
        data = list(msg.data)
        if len(data) < 13 or data[0] < 0.5:
            self.yolo_bbox = None
            self.yolo_bbox_stamp = self.get_clock().now()
            return
        area = max(0.0, float(data[3]) * float(data[4]))
        image_area = max(1.0, float(data[9]) * float(data[10]))
        self.yolo_bbox = {
            "found": True,
            "center_x": float(data[1]),
            "center_y": float(data[2]),
            "width": float(data[3]),
            "height": float(data[4]),
            "x1": float(data[5]),
            "y1": float(data[6]),
            "x2": float(data[7]),
            "y2": float(data[8]),
            "image_width": float(data[9]),
            "image_height": float(data[10]),
            "confidence": float(data[11]),
            "distance": normalized_depth_meters(data[12]),
            "area": area,
            "area_ratio": float(area / image_area),
        }
        self.yolo_bbox_stamp = self.get_clock().now()

    def _target_surface_callback(self, msg):
        data = list(msg.data)
        if len(data) < 8:
            return
        self.target_surface_info = {
            "target_found": data[0] >= 0.5,
            "bridge_found": data[1] >= 0.5,
            "bbox_bridge_overlap_ratio": float(data[2]),
            "target_center_on_bridge": data[4] >= 0.5,
            "target_bottom_center_on_bridge": data[5] >= 0.5,
            "image_width": float(data[6]),
            "image_height": float(data[7]),
        }
        self.target_surface_stamp = self.get_clock().now()

    def _segmentation_callback(self, msg):
        self.segmentation_info = list(msg.data)
        self.segmentation_stamp = self.get_clock().now()

    def _corridor_callback(self, msg):
        self.corridor_info = corridor_from_msg(msg.data)
        self.corridor_stamp = self.get_clock().now()

    def _map_callback(self, msg):
        self.map_msg = msg

    def _path_callback(self, msg):
        self.latest_path = msg

    def _control_loop(self):
        self._publish_target_label()
        self._publish_state()
        self._update_tf_pose()
        if not self.arm_home_sent:
            self._publish_arm_positions(self.arm_home_positions)
            self.arm_home_sent = True
            self._log_event("arm_home", action="home")

        if self.state == SimpleMissionState.INIT:
            self._state_init()
        elif self.state == SimpleMissionState.SEARCH_DRIVABLE:
            self._state_search_drivable()
        elif self.state == SimpleMissionState.FOLLOW_ROAD_TO_BRIDGE:
            self._state_follow_road()
        elif self.state == SimpleMissionState.ALIGN_BRIDGE_ENTRY:
            self._state_align_bridge_entry()
        elif self.state == SimpleMissionState.APPROACH_BRIDGE_ENTRY:
            self._state_approach_bridge_entry()
        elif self.state == SimpleMissionState.ASCEND_BRIDGE:
            self._state_ascend_bridge()
        elif self.state == SimpleMissionState.SEARCH_BEAR_ON_TOP:
            self._state_search_bear_on_top()
        elif self.state == SimpleMissionState.OBSERVE_BEAR:
            self._state_observe_bear()
        elif self.state == SimpleMissionState.APPROACH_GRAB:
            self._state_approach_grab()
        elif self.state == SimpleMissionState.SECURE_BEAR:
            self._state_secure_bear()
        elif self.state == SimpleMissionState.VERIFY_GRAB:
            self._state_verify_grab()
        elif self.state == SimpleMissionState.RETURN_START:
            self._state_return_start()
        elif self.state == SimpleMissionState.DROP_BEAR:
            self._state_drop_bear()
        elif self.state == SimpleMissionState.DONE:
            self._publish_action("STOP")
        self._publish_debug()

    def _state_init(self):
        self._publish_action("STOP")
        if self.pose is None:
            self._log_event("waiting_pose")
            return
        self.start_pose = self.pose
        self.start_pose_z = self.pose_z
        self._log_event("captured_start_pose", pose=self.start_pose)
        self._set_state(SimpleMissionState.SEARCH_DRIVABLE, "start pose captured")

    def _state_search_drivable(self):
        if self.current_task == 2 and self._bridge_corridor_visible():
            self._set_state(SimpleMissionState.ALIGN_BRIDGE_ENTRY, "bridge corridor visible")
            return
        if self._road_corridor_visible():
            self._set_state(SimpleMissionState.FOLLOW_ROAD_TO_BRIDGE, "road corridor visible")
            return
        if (
            not self.mission_config.bridge_first_shared_bear
            and self.current_task == 1
            and self._fresh_target_visible()
            and not self._target_on_bridge()
        ):
            self._set_state(SimpleMissionState.OBSERVE_BEAR, "ground bear visible")
            return
        self._publish_action(self._search_action())

    def _state_follow_road(self):
        if self.current_task == 2 and self._bridge_corridor_visible():
            self._set_state(SimpleMissionState.ALIGN_BRIDGE_ENTRY, "bridge reached")
            return
        if (
            not self.mission_config.bridge_first_shared_bear
            and self.current_task == 1
            and self._fresh_target_visible()
            and not self._target_on_bridge()
        ):
            self._set_state(SimpleMissionState.OBSERVE_BEAR, "ground bear visible")
            return
        action = self._visual_safe_action("FORWARD_SLOW", "road")
        self._publish_action(action)

    def _state_align_bridge_entry(self):
        anchor = self._bear_anchor_info()
        centered_by_corridor = self._strict_bridge_corridor_ready()
        centered_by_anchor = (
            anchor["valid_pre_ascent"]
            and abs(anchor["error_x"]) <= self.bear_anchor_center_tolerance_pixels
        )
        if centered_by_corridor or centered_by_anchor:
            self.align_bridge_confirm_count += 1
            self._log_event(
                "bridge_align_confirming",
                count=self.align_bridge_confirm_count,
                by_corridor=centered_by_corridor,
                by_anchor=centered_by_anchor,
            )
        else:
            self.align_bridge_confirm_count = 0
        if self.align_bridge_confirm_count >= self.bridge_corridor_confirm_frames:
            self._set_state(
                SimpleMissionState.APPROACH_BRIDGE_ENTRY,
                "bridge entry alignment confirmed",
            )
            return
        if self._state_elapsed() >= self.bridge_entry_timeout_seconds:
            self._log_event("bridge_entry_timeout", level="warn")
            self.align_bridge_confirm_count = 0
            self._set_state(SimpleMissionState.FOLLOW_ROAD_TO_BRIDGE, "entry timeout")
            return
        action = self._bear_anchor_action("FORWARD_SLOW", "pre_ascent")
        if action is None:
            action = self._visual_safe_action("FORWARD_SLOW", "bridge_align")
        elif action in VisualCorridorController.FORWARD_ACTIONS:
            action = self._visual_safe_action(action, "bridge_align")
        self._publish_action(action)

    def _state_approach_bridge_entry(self):
        if self.bridge_entry_approach_start_time is None:
            self.bridge_entry_approach_start_time = self.get_clock().now()
            self.approach_bridge_confirm_count = 0

        ready = self._strict_bridge_corridor_ready()
        if ready:
            self.approach_bridge_confirm_count += 1
            self._log_event(
                "bridge_entry_approach_confirming",
                count=self.approach_bridge_confirm_count,
            )
        else:
            self.approach_bridge_confirm_count = 0

        elapsed = self._elapsed_seconds(self.bridge_entry_approach_start_time)
        if (
            self.approach_bridge_confirm_count >= self.bridge_entry_approach_confirm_frames
            and elapsed >= self.bridge_entry_approach_min_seconds
        ):
            self._set_state(SimpleMissionState.ASCEND_BRIDGE, "bridge entry approach confirmed")
            return
        if elapsed >= self.bridge_entry_approach_max_seconds:
            self._set_state(SimpleMissionState.ASCEND_BRIDGE, "bridge entry approach time reached")
            return

        action = self._bear_anchor_action("FORWARD_SLOW", "pre_ascent")
        if action is None:
            action = self._visual_safe_action("FORWARD_SLOW", "bridge_entry")
        elif action in VisualCorridorController.FORWARD_ACTIONS:
            action = self._visual_safe_action(action, "bridge_entry")
        self._publish_action(action)

    def _state_ascend_bridge(self):
        if self.ascent_start_time is None:
            self.ascent_start_time = self.get_clock().now()
            self.ascent_start_z = self.pose_z
            self.top_settle_start_time = None
            self.top_confirm_count = 0
            self.ascent_bear_depth_confirm_count = 0
            self.top_confirmed = False
            self.top_confirm_reason = "not evaluated"
            self._log_event("ascent_start")

        if self._bear_depth_stop_ready():
            self._publish_action("STOP")
            self.top_confirmed = True
            self.top_confirm_reason = "bear depth <= 0.40m"
            self._set_state(SimpleMissionState.OBSERVE_BEAR, "bear depth stop on bridge top")
            return

        if (
            self._state_elapsed() >= self.ascent_min_seconds
            and self._top_platform_confirmed()
        ):
            self._publish_action("STOP")
            if self.top_settle_start_time is None:
                self.top_settle_start_time = self.get_clock().now()
                self._log_event("top_settle_start", reason=self.top_confirm_reason)
                return
            if self._elapsed_seconds(self.top_settle_start_time) >= self.top_settle_seconds:
                if self._fresh_target_visible():
                    self._set_state(SimpleMissionState.OBSERVE_BEAR, "top confirmed with bear visible")
                else:
                    self._set_state(SimpleMissionState.SEARCH_BEAR_ON_TOP, "top confirmed")
            return
        if self._state_elapsed() >= self.ascent_max_seconds:
            self._publish_action("STOP")
            self.top_confirmed = True
            self.top_confirm_reason = "ascent max time reached"
            self._set_state(SimpleMissionState.SEARCH_BEAR_ON_TOP, "ascent timeout")
            return
        action = self._bear_anchor_action("ASCEND_FORWARD", "ascent")
        if action is None:
            action = "ASCEND_FORWARD"
        if action in VisualCorridorController.FORWARD_ACTIONS:
            action = self._visual_safe_action(action, "ascent")
        self._publish_action(action)

    def _state_search_bear_on_top(self):
        if self.top_confirmed and self._fresh_target_visible():
            if normalized_depth_meters(self.yolo_target["distance"]) <= self.top_bear_depth_threshold:
                self._set_state(SimpleMissionState.OBSERVE_BEAR, "top bear within depth stop")
            else:
                self._set_state(SimpleMissionState.OBSERVE_BEAR, "bear visible on top")
            return
        self._publish_action("CLOCKWISE_ROTATION_SLOW")

    def _state_observe_bear(self):
        self._publish_action("STOP")
        if not self._fresh_target_visible():
            self.observe_start_time = None
            self._set_state(SimpleMissionState.SEARCH_BEAR_ON_TOP, "lost target during observe")
            return
        if self.observe_start_time is None:
            self.observe_start_time = self.get_clock().now()
        if self._elapsed_seconds(self.observe_start_time) >= self.observe_seconds:
            self._set_state(SimpleMissionState.APPROACH_GRAB, "observe complete")

    def _state_approach_grab(self):
        if not self._fresh_target_visible():
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
            return
        distance = self.yolo_target["distance"]
        delta_x = self.yolo_target["delta_x"]
        if abs(delta_x) > self.grab_align_pixel_tolerance:
            mode = "ascent" if self.current_task == 2 else "road"
            desired = "RIGHT_FRONT" if delta_x > 0.0 else "LEFT_FRONT"
            self._publish_action(self._visual_safe_action(desired, mode))
            return
        bridge_depth_stop = self.current_task == 2 and 0.0 < distance <= self.top_bear_depth_threshold
        if bridge_depth_stop or (0.0 < distance <= self.grab_distance):
            self.pre_grab_bbox = dict(self.yolo_bbox) if self.yolo_bbox else None
            self._set_state(SimpleMissionState.SECURE_BEAR, "grab range reached")
            return
        mode = "ascent" if self.current_task == 2 else "road"
        self._publish_action(self._visual_safe_action("FORWARD_SLOW", mode))

    def _state_secure_bear(self):
        self._publish_action("STOP")
        if self.grab_start_time is None:
            self.grab_start_time = self.get_clock().now()
            self.grab_step_index = 0
            self.grab_step_sent = False
            self.grab_step_deadline = None
        sequence = [
            (self.arm_open_positions, 0.35),
            (self.arm_pregrasp_positions, 0.50),
            (self.arm_reach_positions, 0.85),
            (self.arm_close_positions, 1.10),
            (self.arm_lift_positions, 0.80),
            (self.arm_carry_positions, 0.50),
        ]
        if self._run_arm_sequence(sequence, "grab"):
            self._reset_grab_sequence()
            self._set_state(SimpleMissionState.VERIFY_GRAB, "grab sequence complete")

    def _state_verify_grab(self):
        self._publish_action("STOP")
        if self.verify_start_time is None:
            self.verify_start_time = self.get_clock().now()
            self.verify_seen_start_time = None
            self._publish_arm_positions(self.arm_carry_positions)
        valid, reason = self._grab_bbox_indicates_bear_held()
        if valid:
            if self.verify_seen_start_time is None:
                self.verify_seen_start_time = self.get_clock().now()
                return
            if self._elapsed_seconds(self.verify_seen_start_time) >= self.verify_grab_seconds:
                self.bear_secured = True
                self.grab_retry_count = 0
                self._reset_verify_sequence()
                self._set_state(SimpleMissionState.RETURN_START, "bear verified")
                return
        else:
            self.verify_seen_start_time = None

        if self._elapsed_seconds(self.verify_start_time) >= self.verify_grab_timeout_seconds:
            self._log_event("verify_grab_failed", level="warn", reason=reason)
            self._reset_verify_sequence()
            self._reset_grab_sequence()
            self._publish_arm_positions(self.arm_open_positions)
            if self.grab_retry_count < self.verify_grab_retry_limit:
                self.grab_retry_count += 1
                self._set_state(SimpleMissionState.APPROACH_GRAB, "retry grab")
            else:
                self.grab_retry_count = 0
                self._set_state(SimpleMissionState.SEARCH_BEAR_ON_TOP, "reacquire bear")

    def _state_return_start(self):
        if self.start_pose is None or self.pose is None:
            self._publish_action("STOP")
            return
        distance = distance_2d(self.pose, self.start_pose)
        heading_error = normalize_angle(self.start_pose[2] - self.pose[2])
        if distance <= self.return_goal_tolerance and abs(heading_error) <= self.return_heading_tolerance:
            self._publish_action("STOP")
            self._set_state(SimpleMissionState.DROP_BEAR, "returned to start")
            return
        self._publish_goal_pose(self.start_pose)
        action = self._action_toward_pose(self.start_pose)
        self._publish_action(self._visual_safe_action(action, "return"))

    def _state_drop_bear(self):
        self._publish_action("STOP")
        if self.drop_start_time is None:
            self.drop_start_time = self.get_clock().now()
            self.drop_step_index = 0
            self.drop_step_sent = False
            self.drop_step_deadline = None
        sequence = [
            (self.arm_lift_positions, 0.35),
            (self.arm_drop_positions, 0.80),
            (self.arm_release_positions, 1.00),
            (self.arm_drop_retract_positions, 0.60),
        ]
        if self._run_arm_sequence(sequence, "drop"):
            self.bear_secured = False
            self._reset_drop_sequence()
            if self.mission_config.bridge_first_shared_bear:
                self.task1_complete = True
                self.task2_complete = True
                self._set_state(SimpleMissionState.DONE, "shared bridge bear delivered")
            elif self.current_task == 1:
                self.task1_complete = True
                self.current_task = 2
                self._set_state(
                    SimpleMissionState.SEARCH_DRIVABLE,
                    "task 1 complete; starting task 2",
                )
            else:
                self.task2_complete = True
                self._set_state(SimpleMissionState.DONE, "task 2 bear dropped")

    def _visual_safe_action(self, desired_action, mode):
        if desired_action not in VisualCorridorController.FORWARD_ACTIONS:
            return desired_action
        self.last_desired_action = desired_action
        self.last_visual_gate_mode = mode
        if mode == "bear_approach" and self._fresh_target_visible():
            if normalized_depth_meters(self.yolo_target["distance"]) <= self.top_bear_depth_threshold:
                self.last_visual_gate_reason = "bear already within 0.40m stop depth"
                return "STOP"
        info = self._fresh_corridor()
        if info is None:
            if mode == "return" and self.return_direct_fallback:
                self.last_visual_gate_reason = "corridor unavailable; explicit return fallback enabled"
                return desired_action
            action = self._search_action()
            reason = "corridor unavailable or stale; using slow search only"
            self._warn_stale_corridor(reason)
            self._log_visual_override(desired_action, action, mode, info, reason)
            self.last_visual_gate_reason = reason
            return action

        if mode == "bridge_align":
            action = self._bridge_align_correction_action(info)
            reason = "bridge alignment is rotation/arc only"
            if action != desired_action:
                self._log_visual_override(desired_action, action, mode, info, reason)
            self.last_visual_gate_reason = reason
            return action

        if mode == "bridge_entry":
            if not self._strict_bridge_corridor_ready():
                action = self._bridge_align_correction_action(info)
                reason = "strict bridge corridor not ready for entry forward"
                self._log_visual_override(desired_action, action, mode, info, reason)
                self.last_visual_gate_reason = reason
                return action

        if mode == "ascent":
            anchor = self._bear_anchor_info()
            anchor_centered = (
                anchor["valid_ascent"]
                and abs(anchor["error_x"]) <= self.bear_anchor_center_tolerance_pixels
            )
            if desired_action == "ASCEND_FORWARD" and not (
                self._strict_bridge_corridor_ready() or anchor_centered
            ):
                action = self._bridge_align_correction_action(info)
                reason = "ascent forward blocked until strict bridge or centered bear anchor"
                self._log_visual_override(desired_action, action, mode, info, reason)
                self.last_visual_gate_reason = reason
                return action
            if desired_action in ("LEFT_FRONT", "RIGHT_FRONT") and not (
                self._strict_bridge_corridor_ready() or anchor["valid_ascent"]
            ):
                action = self._bridge_align_correction_action(info)
                reason = "ascent arc blocked until bridge or bear anchor is valid"
                self._log_visual_override(desired_action, action, mode, info, reason)
                self.last_visual_gate_reason = reason
                return action

        if mode == "road" and int(info.get("drivable_type", 0)) not in (1, 3):
            action = self._bridge_align_correction_action(info)
            reason = "road mode requires road or mixed corridor"
            self._log_visual_override(desired_action, action, mode, info, reason)
            self.last_visual_gate_reason = reason
            return action

        result = self.visual_controller.update(
            info,
            self.control_period_seconds,
            desired_forward=desired_action,
            mode=mode,
        )
        action = result["action"]
        self.last_visual_gate_reason = result["reason"]
        if action != desired_action:
            self._log_visual_override(desired_action, action, mode, info, result["reason"])
        return action

    def _bridge_align_correction_action(self, info):
        error = float((info or {}).get("error_x", 0.0))
        if abs(error) <= self.bridge_corridor_center_tolerance_pixels:
            if self.bridge_align_allow_forward:
                return "FORWARD_SLOW"
            if self.bridge_align_allow_gentle_arc:
                return "RIGHT_FRONT" if error >= 0.0 else "LEFT_FRONT"
        if abs(error) <= self.corridor_arc_tolerance_pixels and self.bridge_align_allow_gentle_arc:
            return "RIGHT_FRONT" if error > 0.0 else "LEFT_FRONT"
        return "CLOCKWISE_ROTATION_SLOW" if error >= 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"

    def _log_visual_override(self, desired_action, chosen_action, mode, info, reason):
        info = info or {}
        key = (
            mode,
            desired_action,
            chosen_action,
            reason,
            info.get("valid", False),
            info.get("error_x", 0.0),
        )
        if key == self.last_visual_override:
            return
        self.last_visual_override = key
        self._log_event(
            "visual_gate_override",
            desired_action=desired_action,
            chosen_action=chosen_action,
            mode=mode,
            corridor_valid=info.get("valid", False),
            bottom_connected=info.get("bottom_connected", False),
            centerline_reached=info.get("centerline_reached", False),
            continuous_score=info.get("continuous_score", 0.0),
            bottom_width_ratio=info.get("bottom_width_ratio", 0.0),
            center_width_ratio=info.get("center_width_ratio", 0.0),
            error_x=info.get("error_x", 0.0),
            slope=info.get("slope", 0.0),
            drivable_type=info.get("drivable_type", 0),
            side_view_likely=info.get("side_view_likely", False),
            reason=reason,
        )

    def _warn_stale_corridor(self, reason):
        now = self.get_clock().now()
        if (
            self.last_corridor_stale_warn_time is not None
            and self._elapsed_seconds(self.last_corridor_stale_warn_time) < 1.0
        ):
            return
        self.last_corridor_stale_warn_time = now
        self.get_logger().warn(reason)

    def _target_alignment_action(self, delta_x, desired_forward):
        if abs(delta_x) <= self.align_pixel_tolerance:
            mode = "bear_approach" if getattr(self, "current_task", 1) == 2 else "road"
            return self._visual_safe_action(desired_forward, mode)
        desired = "RIGHT_FRONT" if delta_x > 0.0 else "LEFT_FRONT"
        mode = "bear_approach" if getattr(self, "current_task", 1) == 2 else "road"
        return self._visual_safe_action(desired, mode)

    def _strict_bridge_corridor_ready(self, log_rejection=True):
        info = self._fresh_corridor()
        ready = bool(
            info
            and info["drivable_type"] in (2, 3)
            and info["bottom_connected"]
            and info["centerline_reached"]
            and info["continuous_score"] >= self.bridge_corridor_min_continuous_score
            and info["bottom_width_ratio"] >= self.bridge_corridor_min_bottom_width_ratio
            and info["center_width_ratio"] >= self.bridge_corridor_min_center_width_ratio
            and abs(info["error_x"]) <= self.bridge_corridor_center_tolerance_pixels
            and not info["side_view_likely"]
        )
        if not ready and log_rejection:
            info = info or {}
            self._log_event(
                "strict_bridge_corridor_rejected",
                corridor_valid=info.get("valid", False),
                continuous_score=info.get("continuous_score", 0.0),
                bottom_width_ratio=info.get("bottom_width_ratio", 0.0),
                center_width_ratio=info.get("center_width_ratio", 0.0),
                error_x=info.get("error_x", 0.0),
                drivable_type=info.get("drivable_type", 0),
                side_view_likely=info.get("side_view_likely", False),
            )
        return ready

    def _target_surface_available(self):
        return (
            self.target_surface_info is not None
            and self._stamp_age(self.target_surface_stamp) <= self.target_timeout_seconds
        )

    def _bear_anchor_info(self):
        empty = {
            "visible": False,
            "fresh": False,
            "confidence": 0.0,
            "distance": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "center_x_ratio": 0.0,
            "center_y_ratio": 0.0,
            "error_x": 0.0,
            "depth": 0.0,
            "valid_pre_ascent": False,
            "valid_ascent": False,
            "reason": "no fresh bear bbox",
        }
        if not self.bear_anchor_enabled:
            empty["reason"] = "bear anchor disabled"
            return empty
        if not (self._fresh_target_visible() and self._fresh_bbox_visible()):
            return empty
        bbox = self.yolo_bbox
        width = max(1.0, bbox["image_width"])
        height = max(1.0, bbox["image_height"])
        distance = normalized_depth_meters(
            self.yolo_target.get("distance", bbox.get("distance", 0.0))
        )
        center_x_ratio = bbox["center_x"] / width
        center_y_ratio = bbox["center_y"] / height
        confidence = float(bbox.get("confidence", 0.0))
        fresh = (
            self._stamp_age(self.yolo_target_stamp) <= self.bear_anchor_max_age_seconds
            and self._stamp_age(self.yolo_bbox_stamp) <= self.bear_anchor_max_age_seconds
        )
        surface_ok = True
        if self._target_surface_available():
            surface_ok = self._target_on_bridge()
        common_ok = fresh and confidence >= self.bear_anchor_min_confidence and distance > 0.0
        valid_pre_ascent = (
            common_ok
            and surface_ok
            and self.bear_anchor_pre_ascent_min_y_ratio
            <= center_y_ratio
            <= self.bear_anchor_pre_ascent_max_y_ratio
        )
        valid_ascent = (
            common_ok
            and self.bear_anchor_ascent_min_y_ratio
            <= center_y_ratio
            <= self.bear_anchor_ascent_max_y_ratio
        )
        reason = "valid"
        if not common_ok:
            reason = "stale, low confidence, or invalid depth"
        elif not surface_ok:
            reason = "target surface not on bridge"
        elif not (valid_pre_ascent or valid_ascent):
            reason = "bbox vertical position outside anchor band"
        return {
            "visible": True,
            "fresh": fresh,
            "confidence": confidence,
            "distance": distance,
            "center_x": bbox["center_x"],
            "center_y": bbox["center_y"],
            "center_x_ratio": center_x_ratio,
            "center_y_ratio": center_y_ratio,
            "error_x": bbox["center_x"] - width * 0.5,
            "depth": distance,
            "valid_pre_ascent": bool(valid_pre_ascent),
            "valid_ascent": bool(valid_ascent),
            "reason": reason,
        }

    def _bear_anchor_action(self, base_forward_action, phase):
        anchor = self._bear_anchor_info()
        valid = anchor["valid_ascent"] if phase == "ascent" else anchor["valid_pre_ascent"]
        if not valid:
            if anchor["visible"]:
                self._log_event("bear_anchor_rejected", phase=phase, reason=anchor["reason"])
            return None
        action = self.bear_anchor_controller.action(
            anchor,
            base_forward_action,
            self.control_period_seconds,
        )
        self._log_event(
            "bear_anchor_used",
            phase=phase,
            action=action,
            error_x=anchor["error_x"],
            depth=anchor["depth"],
        )
        return action

    def _bear_depth_stop_ready(self):
        if not (self._fresh_target_visible() and self._fresh_bbox_visible()):
            self.ascent_bear_depth_confirm_count = 0
            return False
        depth = normalized_depth_meters(self.yolo_target["distance"])
        delta_x = abs(float(self.yolo_target.get("delta_x", 9999.0)))
        ready_now = (
            depth > 0.0
            and depth <= self.top_bear_depth_threshold
            and delta_x <= self.top_bear_depth_center_tolerance_pixels
        )
        if ready_now:
            self.ascent_bear_depth_confirm_count += 1
            self._log_event(
                "bear_depth_stop_confirming",
                depth=depth,
                count=self.ascent_bear_depth_confirm_count,
            )
        else:
            self.ascent_bear_depth_confirm_count = 0
        if self.ascent_bear_depth_confirm_count >= self.top_confirm_frames:
            self._log_event("bear_depth_stop_confirmed", depth=depth)
            return True
        return False

    def _search_action(self):
        info = self._fresh_corridor()
        if info is not None:
            if self.current_task == 2:
                if int(info.get("drivable_type", 0)) in (2, 3):
                    return self._bridge_align_correction_action(info)
                if self._road_corridor_visible():
                    return self._visual_safe_action("FORWARD_SLOW", "road")
                return self._bridge_align_correction_action(info)
            result = self.visual_controller.update(
                info,
                self.control_period_seconds,
                desired_forward="FORWARD_SLOW",
                mode="return" if self.state == SimpleMissionState.RETURN_START else "road",
            )
            return result["action"]
        return self._segmentation_search_action()

    def _segmentation_search_action(self):
        data = self.segmentation_info or []
        if len(data) >= 8 and self._stamp_age(self.segmentation_stamp) <= self.corridor_timeout_seconds:
            road_found = data[0] >= 0.5
            bridge_found = data[4] >= 0.5
            if self.current_task == 2 and bridge_found:
                delta_x = float(data[5])
            elif road_found:
                delta_x = float(data[1])
            elif bridge_found:
                delta_x = float(data[5])
            else:
                delta_x = 0.0
            if road_found or bridge_found:
                return (
                    "CLOCKWISE_ROTATION_SLOW"
                    if delta_x >= 0.0
                    else "COUNTERCLOCKWISE_ROTATION_SLOW"
                )
        return "CLOCKWISE_ROTATION_SLOW"

    def _bridge_corridor_visible(self):
        info = self._fresh_corridor()
        return bool(
            info
            and info["drivable_type"] in (2, 3)
            and info["bottom_connected"]
            and info["centerline_reached"]
            and info["continuous_score"] >= self.corridor_min_continuous_score
            and info["bottom_width_ratio"] >= self.corridor_min_bottom_width_ratio
            and info["center_width_ratio"] >= self.corridor_min_center_width_ratio
            and not info["side_view_likely"]
        )

    def _road_corridor_visible(self):
        info = self._fresh_corridor()
        return bool(
            info
            and info["drivable_type"] in (1, 3)
            and info["bottom_connected"]
            and info["centerline_reached"]
            and info["continuous_score"] >= self.corridor_min_continuous_score
            and info["bottom_width_ratio"] >= self.corridor_min_bottom_width_ratio
            and info["center_width_ratio"] >= self.corridor_min_center_width_ratio
            and not info["side_view_likely"]
        )

    def _bridge_entry_ready(self):
        return self._strict_bridge_corridor_ready()

    def _top_platform_confirmed(self):
        confirmed = False
        reason = "not confirmed"
        z_gain = self.pose_z - self.ascent_start_z
        if z_gain >= self.top_z_threshold:
            confirmed = True
            reason = "z threshold reached"
        elif self._fresh_target_visible():
            depth = self.yolo_target["distance"]
            if 0.0 < depth <= self.top_bear_depth_threshold:
                confirmed = True
                reason = "bear depth confirms top"
        if confirmed:
            self.top_confirm_count += 1
        else:
            self.top_confirm_count = 0
        self.top_confirmed = self.top_confirm_count >= self.top_confirm_frames
        self.top_confirm_reason = reason
        self._log_event(
            "top_check",
            z_gain=z_gain,
            top_confirm_count=self.top_confirm_count,
            reason=reason,
        )
        return self.top_confirmed

    def _grab_bbox_indicates_bear_held(self):
        if not self._fresh_bbox_visible():
            return False, "no fresh bbox"
        bbox = self.yolo_bbox
        if 0.0 < bbox["distance"] <= self.verify_grab_max_depth:
            close = True
        else:
            close = False
        large = bbox["area_ratio"] >= self.verify_grab_min_bbox_area_ratio
        moved_up = True
        if self.pre_grab_bbox is not None:
            moved_up = (
                bbox["center_y"]
                <= self.pre_grab_bbox["center_y"] - self.verify_grab_lift_pixel_delta
            ) or bbox["area"] >= self.pre_grab_bbox["area"] * 1.08
        if (close or large) and moved_up:
            return True, "bbox matches held bear"
        return False, "bbox not close/large/lifted enough"

    def _run_arm_sequence(self, sequence, name):
        if self.grab_start_time is None and name == "grab":
            self.grab_start_time = self.get_clock().now()
        index_attr = "grab_step_index" if name == "grab" else "drop_step_index"
        sent_attr = "grab_step_sent" if name == "grab" else "drop_step_sent"
        deadline_attr = "grab_step_deadline" if name == "grab" else "drop_step_deadline"
        index = getattr(self, index_attr)
        if index >= len(sequence):
            return True
        positions, hold_seconds = sequence[index]
        now = self.get_clock().now()
        if not getattr(self, sent_attr):
            self._publish_arm_positions(positions)
            setattr(self, deadline_attr, now.nanoseconds + int(hold_seconds * 1e9))
            setattr(self, sent_attr, True)
            self._log_event(
                f"arm_{name}_step",
                step=index + 1,
                total=len(sequence),
                positions=positions,
            )
            return False
        deadline = getattr(self, deadline_attr)
        if deadline is not None and now.nanoseconds >= deadline:
            setattr(self, index_attr, index + 1)
            setattr(self, sent_attr, False)
            setattr(self, deadline_attr, None)
        return False

    def _action_toward_pose(self, target_pose):
        dx = target_pose[0] - self.pose[0]
        dy = target_pose[1] - self.pose[1]
        target_yaw = math.atan2(dy, dx)
        yaw_error = normalize_angle(target_yaw - self.pose[2])
        if distance_2d(self.pose, target_pose) <= self.return_goal_tolerance:
            yaw_error = normalize_angle(target_pose[2] - self.pose[2])
            if abs(yaw_error) <= self.return_heading_tolerance:
                return "STOP"
        if abs(yaw_error) > 0.45:
            return "CLOCKWISE_ROTATION_SLOW" if yaw_error < 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"
        if abs(yaw_error) > 0.16:
            return "RIGHT_FRONT" if yaw_error < 0.0 else "LEFT_FRONT"
        return "FORWARD_SLOW"

    def _fresh_target_visible(self):
        if not self.yolo_target or not self.yolo_target.get("found", False):
            return False
        return self._stamp_age(self.yolo_target_stamp) <= self.target_timeout_seconds

    def _fresh_bbox_visible(self):
        if not self.yolo_bbox or not self.yolo_bbox.get("found", False):
            return False
        return self._stamp_age(self.yolo_bbox_stamp) <= self.bbox_timeout_seconds

    def _fresh_corridor(self):
        if self.corridor_info is None:
            return None
        if self._stamp_age(self.corridor_stamp) > self.corridor_timeout_seconds:
            return None
        return self.corridor_info

    def _target_on_bridge(self):
        if not self.target_surface_info:
            return False
        if self._stamp_age(self.target_surface_stamp) > self.target_timeout_seconds:
            return False
        return bool(
            self.target_surface_info.get("target_center_on_bridge", False)
            or self.target_surface_info.get("target_bottom_center_on_bridge", False)
            or self.target_surface_info.get("bbox_bridge_overlap_ratio", 0.0) >= 0.20
        )

    def _update_tf_pose(self):
        if self.pose is not None or not self.use_tf_pose_fallback:
            return
        try:
            tf = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time()
            )
        except Exception:
            return
        t = tf.transform.translation
        r = tf.transform.rotation
        self.pose = (float(t.x), float(t.y), yaw_from_quaternion(r))
        self.pose_z = float(t.z)

    def _publish_action(self, action_key):
        action_key = self._apply_stuck_recovery(action_key)
        if action_key == "ASCEND_FORWARD":
            velocities = self._ascend_forward_velocities()
        else:
            velocities = ACTION_MAPPINGS.get(action_key, ACTION_MAPPINGS["STOP"])
        rear_msg = Float32MultiArray()
        rear_msg.data = [float(velocities[0]), float(velocities[1])]
        self.rear_pub.publish(rear_msg)
        front_msg = Float32MultiArray()
        front_msg.data = [float(velocities[2]), float(velocities[3])]
        self.front_pub.publish(front_msg)
        if action_key != self.last_action:
            self.last_action = action_key
            self._log_event("action", action=action_key)

    def _apply_stuck_recovery(self, action_key):
        if not self.stuck_detection_enabled or self.pose is None:
            return action_key
        now = self.get_clock().now()
        if self.stuck_recovery_step is not None:
            if now.nanoseconds < self.stuck_recovery_until:
                return self.stuck_recovery_step
            if self.stuck_recovery_step == "STOP":
                self.stuck_recovery_step = "BACKWARD_SLOW"
                self.stuck_recovery_until = now.nanoseconds + int(self.stuck_back_seconds * 1e9)
                return self.stuck_recovery_step
            if self.stuck_recovery_step == "BACKWARD_SLOW":
                self.stuck_recovery_step = "CLOCKWISE_ROTATION_SLOW"
                self.stuck_recovery_until = now.nanoseconds + int(self.stuck_rotate_seconds * 1e9)
                return self.stuck_recovery_step
            self.stuck_recovery_step = None
            self.stuck_recovery_until = None
            self.stuck_reference_pose = self.pose
            self.stuck_reference_time = now
            return action_key

        if action_key not in VisualCorridorController.FORWARD_ACTIONS:
            self.stuck_reference_pose = self.pose
            self.stuck_reference_time = now
            return action_key

        if self.stuck_reference_pose is None:
            self.stuck_reference_pose = self.pose
            self.stuck_reference_time = now
            return action_key

        elapsed = self._elapsed_seconds(self.stuck_reference_time)
        moved = distance_2d(self.pose, self.stuck_reference_pose)
        turned = abs(normalize_angle(self.pose[2] - self.stuck_reference_pose[2]))
        if moved >= self.stuck_min_translation or turned >= self.stuck_min_yaw_change:
            self.stuck_reference_pose = self.pose
            self.stuck_reference_time = now
            return action_key
        if elapsed >= self.stuck_check_seconds:
            self.stuck_recovery_step = "STOP"
            self.stuck_recovery_until = now.nanoseconds + int(self.stuck_stop_seconds * 1e9)
            self._log_event(
                "stuck_recovery",
                level="warn",
                recovery="stop_back_rotate",
                moved=moved,
                turned=turned,
            )
            return self.stuck_recovery_step
        return action_key

    def _ascend_forward_velocities(self):
        slow = ACTION_MAPPINGS.get("FORWARD_SLOW", [0.0, 0.0, 0.0, 0.0])
        fast = ACTION_MAPPINGS.get("FORWARD", slow)
        scaled = [float(value) * self.ascent_forward_speed_scale for value in slow]
        limited = []
        for value, max_value in zip(scaled, fast):
            sign = 1.0 if value >= 0.0 else -1.0
            limited.append(sign * min(abs(value), abs(float(max_value))))
        return limited

    def _publish_arm_positions(self, positions):
        msg = JointTrajectoryPoint()
        if self.arm_positions_in_degrees:
            msg.positions = [math.radians(float(value)) for value in positions]
        else:
            msg.positions = [float(value) for value in positions]
        msg.velocities = [0.0] * len(msg.positions)
        self.arm_pub.publish(msg)

    def _publish_goal_pose(self, pose):
        msg = PoseStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(pose[0])
        msg.pose.position.y = float(pose[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(float(pose[2]) / 2.0)
        msg.pose.orientation.w = math.cos(float(pose[2]) / 2.0)
        self.goal_pub.publish(msg)

    def _publish_target_label(self):
        msg = String()
        msg.data = self.target_label
        self.target_label_pub.publish(msg)

    def _publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)

    def _publish_debug(self):
        msg = String()
        info = self._fresh_corridor() or {}
        anchor = self._bear_anchor_info()
        bbox_center_x_ratio = 0.0
        bbox_center_y_ratio = 0.0
        if self.yolo_bbox and self.yolo_bbox["image_width"] > 0.0 and self.yolo_bbox["image_height"] > 0.0:
            bbox_center_x_ratio = self.yolo_bbox["center_x"] / self.yolo_bbox["image_width"]
            bbox_center_y_ratio = self.yolo_bbox["center_y"] / self.yolo_bbox["image_height"]
        fields = {
            "mission_id": self.mission_id,
            "mission_mode": self.mission_mode,
            "state": self.state.value,
            "current_task": self.current_task,
            "pose": self.pose,
            "action": self.last_action,
            "desired_action": self.last_desired_action,
            "visual_gate_mode": self.last_visual_gate_mode,
            "visual_gate_reason": self.last_visual_gate_reason,
            "target_visible": self._fresh_target_visible(),
            "target_distance": self.yolo_target["distance"] if self.yolo_target else 0.0,
            "target_delta_x": self.yolo_target["delta_x"] if self.yolo_target else 0.0,
            "target_on_bridge": self._target_on_bridge(),
            "bbox_center_x_ratio": bbox_center_x_ratio,
            "bbox_center_y_ratio": bbox_center_y_ratio,
            "corridor_valid": info.get("valid", False),
            "bottom_connected": info.get("bottom_connected", False),
            "centerline_reached": info.get("centerline_reached", False),
            "continuous_score": info.get("continuous_score", 0.0),
            "bottom_width_ratio": info.get("bottom_width_ratio", 0.0),
            "center_width_ratio": info.get("center_width_ratio", 0.0),
            "error_x": info.get("error_x", 0.0),
            "slope": info.get("slope", 0.0),
            "drivable_type": info.get("drivable_type", 0),
            "side_view_likely": info.get("side_view_likely", False),
            "strict_bridge_corridor_ready": self._strict_bridge_corridor_ready(
                log_rejection=False
            ),
            "bear_anchor_visible": anchor["visible"],
            "bear_anchor_valid_pre_ascent": anchor["valid_pre_ascent"],
            "bear_anchor_valid_ascent": anchor["valid_ascent"],
            "bear_anchor_error_x": anchor["error_x"],
            "bear_anchor_depth": anchor["depth"],
            "bear_anchor_center_x_ratio": anchor["center_x_ratio"],
            "bear_anchor_center_y_ratio": anchor["center_y_ratio"],
            "bear_depth_stop_ready": (
                self.ascent_bear_depth_confirm_count >= self.top_confirm_frames
            ),
            "bear_depth_confirm_count": self.ascent_bear_depth_confirm_count,
            "top_confirmed": self.top_confirmed,
            "top_confirm_reason": self.top_confirm_reason,
            "align_bridge_confirm_count": self.align_bridge_confirm_count,
            "approach_bridge_confirm_count": self.approach_bridge_confirm_count,
            "mission_elapsed": self._elapsed_seconds(self.mission_start_time),
            "bear_secured": self.bear_secured,
            "task1_complete": self.task1_complete,
            "task2_complete": self.task2_complete,
        }
        msg.data = json.dumps(fields, sort_keys=True)
        self.debug_pub.publish(msg)
        self._write_jsonl(fields)

    def _set_state(self, state, reason=""):
        if self.state == state:
            return
        old_state = self.state
        self.state = state
        self.state_start_time = self.get_clock().now()
        self.visual_controller.reset()
        self.bear_anchor_controller.reset()
        self.align_bridge_confirm_count = 0
        self.approach_bridge_confirm_count = 0
        self.ascent_bear_depth_confirm_count = 0
        if state != SimpleMissionState.APPROACH_BRIDGE_ENTRY:
            self.bridge_entry_approach_start_time = None
        self._log_event("state_transition", old_state=old_state.value, state=state.value, reason=reason)

    def _log_event(self, event, level="info", **fields):
        record = {
            "mission_id": self.mission_id,
            "event": event,
            "state": self.state.value,
            "current_task": self.current_task,
            "time_ns": self.get_clock().now().nanoseconds,
        }
        record.update(fields)
        text = " ".join(f"{key}={value}" for key, value in record.items())
        if level == "warn":
            self.get_logger().warn(text)
        elif level == "error":
            self.get_logger().error(text)
        else:
            self.get_logger().info(text)
        if self.log_file:
            self.log_file.write(text + "\n")
            self.log_file.flush()
        self._write_jsonl(record)

    def _write_jsonl(self, record):
        if self.jsonl_file:
            self.jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")
            self.jsonl_file.flush()

    def _state_elapsed(self):
        return self._elapsed_seconds(self.state_start_time)

    def _stamp_age(self, stamp):
        if stamp is None:
            return 1e9
        return self._elapsed_seconds(stamp)

    def _elapsed_seconds(self, start_time):
        return (self.get_clock().now().nanoseconds - start_time.nanoseconds) / 1e9

    def _reset_grab_sequence(self):
        self.grab_start_time = None
        self.grab_step_index = 0
        self.grab_step_sent = False
        self.grab_step_deadline = None

    def _reset_drop_sequence(self):
        self.drop_start_time = None
        self.drop_step_index = 0
        self.drop_step_sent = False
        self.drop_step_deadline = None

    def _reset_verify_sequence(self):
        self.verify_start_time = None
        self.verify_seen_start_time = None

    def destroy_node(self):
        for handle in (self.log_file, self.jsonl_file):
            if handle:
                handle.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleTaskMissionController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

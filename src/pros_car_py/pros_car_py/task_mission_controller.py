import ast
import json
import math
import os
import uuid
from datetime import datetime
from enum import Enum

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import Point, PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectoryPoint
from visualization_msgs.msg import Marker, MarkerArray

from pros_car_py.car_models import DeviceDataTypeEnum
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS

try:
    from sensor_msgs_py import point_cloud2
except Exception:
    point_cloud2 = None


class MissionState(str, Enum):
    EXPLORE_MAP = "explore_map"
    GO_TO_BEAR_AREA = "go_to_bear_area"
    SEARCH_BEAR = "search_bear"
    APPROACH_BEAR = "approach_bear"
    OBSERVE_BEAR = "observe_bear"
    APPROACH_GRAB = "approach_grab"
    SECURE_BEAR = "secure_bear"
    VERIFY_GRAB = "verify_grab"
    RETURN_START = "return_start"
    DROP_BEAR = "drop_bear"
    TASK2_NAVIGATE_BRIDGE = "task2_navigate_bridge"
    TASK2_SEARCH_BRIDGE = "task2_search_bridge"
    TASK2_EXPLORE_FOR_BRIDGE = "task2_explore_for_bridge"
    TASK2_TURN_TO_BRIDGE = "task2_turn_to_bridge"
    TASK2_SIDE_VIEW_RECOVERY = "task2_side_view_recovery"
    TASK2_APPROACH_BRIDGE_ENTRY = "task2_approach_bridge_entry"
    TASK2_FINAL_ALIGN_BRIDGE = "task2_final_align_bridge"
    TASK2_ASCEND_BRIDGE = "task2_ascent"
    TASK2_SEARCH_BRIDGE_BEAR = "task2_search_bridge_bear"
    TASK2_DESCEND_BRIDGE = "task2_descent"
    DONE = "done"


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def distance_2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


class MissionLogger:
    """Small adapter so pure controllers can emit consistent mission events."""

    def __init__(self, node):
        self.node = node

    def event(self, level, event, **fields):
        self.node._log_event(level, event, **fields)


class StuckMonitor:
    """Pure checks layered on top of the existing ROS motion monitor state."""

    @staticmethod
    def low_progress(node, timeout):
        if node.motion_monitor_start_time is None:
            return False
        if node._elapsed_seconds(node.motion_monitor_start_time) < timeout:
            return False
        if node.motion_monitor_action is None:
            return False
        return not node._motion_progressed(node.motion_monitor_action)


class BridgeVisionAnalyzer:
    """Single source of truth for Task 2 image-space bridge entry decisions."""

    @staticmethod
    def ramp_is_usable(node, bridge):
        if bridge is None:
            return False
        if not node.task2_allow_ramp_fallback_entry:
            return False
        if float(bridge.get("ramp_valid", 0.0)) < 0.5:
            return False
        if float(bridge.get("ramp_confidence", 0.0)) < node.task2_ramp_entry_min_confidence:
            return False
        if float(bridge.get("ramp_lower_present", 0.0)) < 0.5:
            return False
        if float(bridge.get("ramp_continuous", 0.0)) < 0.5:
            return False
        if float(bridge.get("side_view_score", 1.0)) > node.task2_ramp_entry_max_side_view_score:
            return False
        if float(bridge.get("vertical_coverage_score", 0.0)) < node.task2_ramp_entry_min_vertical_coverage:
            return False
        if float(bridge.get("bottom_y_ratio", 0.0)) < node.task2_ramp_entry_min_bottom_y_ratio:
            return False
        delta = BridgeVisionAnalyzer.ramp_delta(node, bridge)
        return delta is not None and abs(delta) <= node.task2_ramp_entry_center_tolerance_pixels

    @staticmethod
    def side_view_likely(node, bridge):
        if bridge is None or not node.task2_side_view_recovery_enabled:
            return False
        if not bridge.get("raw_found", bridge.get("found", False)):
            return False
        ramp_valid = float(bridge.get("ramp_valid", 0.0)) >= 0.5
        ramp_conf = float(bridge.get("ramp_confidence", 0.0))
        side_score = float(bridge.get("side_view_score", 0.0))
        continuous = float(bridge.get("ramp_continuous", 0.0)) >= 0.5
        delta = BridgeVisionAnalyzer.ramp_delta(node, bridge)
        not_centered = delta is not None and abs(delta) > node.task2_bridge_approach_hard_tolerance
        return (
            not ramp_valid
            and (
                side_score >= node.task2_side_view_max_score
                or ramp_conf < node.task2_side_view_min_ramp_confidence
                or not continuous
                or not_centered
            )
        )

    @staticmethod
    def entry_source(node, bridge):
        if bridge is None or not node._bridge_observation_is_fresh(bridge):
            return "none"
        if node._task2_bridge_entry_confirmed(bridge):
            return "road_contact"
        if BridgeVisionAnalyzer.ramp_is_usable(node, bridge):
            return "ramp_fallback"
        return "none"

    @staticmethod
    def ramp_delta(node, bridge):
        if bridge is None:
            return None
        image_width = node._segmentation_image_width()
        if image_width <= 0.0:
            return None
        centers = [
            float(bridge.get("bottom_center_x", 0.0)),
            float(bridge.get("mid_center_x", 0.0)),
            float(bridge.get("center_x", 0.0)),
        ]
        centers = [value for value in centers if value > 0.0]
        if not centers:
            return None
        target = 0.55 * centers[0] + 0.45 * centers[min(1, len(centers) - 1)]
        delta = target - image_width * 0.5
        node._task2_remember_bridge_delta(delta)
        return delta


class AscentController:
    """Image-space ascent action selection with stronger forward climb."""

    @staticmethod
    def action(node, bridge):
        bear_action = None
        if hasattr(node, "_task2_bridge_bear_ascent_action"):
            bear_action = node._task2_bridge_bear_ascent_action()
        if bear_action is not None:
            return bear_action
        if not node.task2_ascent_centering_enabled or bridge is None:
            return node.task2_ascent_action
        delta = BridgeVisionAnalyzer.ramp_delta(node, bridge)
        if delta is None:
            return node.task2_ascent_action
        if abs(delta) > node.task2_bridge_approach_hard_tolerance:
            return node._turn_action_from_error(delta)
        if abs(delta) > node.task2_bridge_approach_soft_tolerance:
            return "RIGHT_FRONT" if delta > 0.0 else "LEFT_FRONT"
        return node.task2_ascent_action


class BridgeBearMemory:
    """Recent bridge-top bear candidate cache."""

    def __init__(self):
        self.valid = False
        self.last_seen_time = None
        self.bbox = None
        self.target_info = None
        self.surface_info = None
        self.seen_state = None
        self.confidence = 0.0

    def update(self, node, confidence):
        self.valid = True
        self.last_seen_time = node.get_clock().now()
        self.bbox = dict(node.yolo_bbox) if node.yolo_bbox is not None else None
        self.target_info = dict(node.yolo_target) if node.yolo_target is not None else None
        self.surface_info = (
            dict(node.target_surface_info)
            if node.target_surface_info is not None
            else None
        )
        self.seen_state = node.state.value if hasattr(node.state, "value") else str(node.state)
        self.confidence = float(confidence)

    def recent(self, node, ttl_seconds, min_confidence):
        if not self.valid or self.last_seen_time is None:
            return False
        if self.confidence < min_confidence:
            return False
        return node._elapsed_seconds(self.last_seen_time) <= ttl_seconds

    def clear_if_expired(self, node, ttl_seconds):
        if self.valid and self.last_seen_time is not None:
            if node._elapsed_seconds(self.last_seen_time) > ttl_seconds:
                self.valid = False


class Task1MissionController(Node):
    """
    Task 1 automation:
    1. Navigate near the bear area.
    2. Locate the bear with YOLO and approach within N units.
    3. Stay stationary for 5+ seconds.
    4. Secure the bear with a configurable arm/gripper sequence.
    5. Return to the start pose.
    """

    def __init__(self):
        super().__init__("task_mission_controller")

        self.declare_parameter("start_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("bear_search_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("use_current_pose_as_start", True)
        self.declare_parameter("auto_set_initial_pose", False)
        self.declare_parameter("initial_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("initial_pose_publish_count", 10)
        self.declare_parameter("initial_pose_publish_period_seconds", 0.5)
        self.declare_parameter("initial_pose_position_covariance", 0.25)
        self.declare_parameter("initial_pose_yaw_covariance", 0.25)
        self.declare_parameter("target_label", "bear")
        self.declare_parameter("auto_explore", True)
        self.declare_parameter("map_topic", "/map")
        self.declare_parameter("use_tf_pose_fallback", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("base_frame", "base_footprint")
        self.declare_parameter("run_task2_after_task1", True)
        self.declare_parameter("mission_mode", "bridge_first_shared_bear")
        self.declare_parameter("task2_use_bridge_entry_pose", False)
        self.declare_parameter("task2_bridge_entry_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("task2_use_bridge_top_pose", False)
        self.declare_parameter("task2_bridge_top_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("task2_use_bridge_exit_pose", False)
        self.declare_parameter("task2_bridge_exit_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("task2_ascent_min_seconds", 8.0)
        self.declare_parameter("task2_ascent_timeout_seconds", 20.0)
        self.declare_parameter("task2_ascent_action", "ASCEND_FORWARD")
        self.declare_parameter("task2_ascent_forward_speed_scale", 1.25)
        self.declare_parameter("task2_ascent_require_top_confidence", True)
        self.declare_parameter("task2_ascent_continue_if_not_top", True)
        self.declare_parameter("task2_ascent_max_extra_seconds", 4.0)
        self.declare_parameter("task2_ascent_centering_enabled", True)
        self.declare_parameter("task2_ascent_stop_on_bridge_loss_seconds", 10.0) #*
        self.declare_parameter("task2_ascent_bear_pid_enabled", True)
        self.declare_parameter("task2_ascent_bear_pid_center_tolerance_pixels", 105.0)
        self.declare_parameter("task2_ascent_bear_pid_kp", 1.5)
        self.declare_parameter("task2_ascent_bear_pid_ki", 0.0)
        self.declare_parameter("task2_ascent_bear_pid_kd", 0.05)
        self.declare_parameter("task2_ascent_bear_pid_integral_limit", 200.0)
        self.declare_parameter("task2_ascent_bear_pid_max_turn", 340.0)
        self.declare_parameter("task2_ascent_bear_pid_rotate_only_pixels", 150.0)
        self.declare_parameter("task2_ascent_bear_pid_forward_scale", 0.75)
        self.declare_parameter("task2_descent_min_seconds", 6.0)
        self.declare_parameter("task2_descent_timeout_seconds", 10.0)
        self.declare_parameter("task2_bridge_search_timeout_seconds", 120.0)
        self.declare_parameter("task2_bridge_confirm_seconds", 0.4)
        self.declare_parameter("task2_bridge_min_area_ratio", 0.008)
        self.declare_parameter("task2_bridge_entry_bottom_coverage", 0.04)
        self.declare_parameter("task2_bridge_center_tolerance", 90.0)
        self.declare_parameter("task2_bridge_candidate_min_bottom_y_ratio", 0.55) #*
        self.declare_parameter("task2_bridge_entry_center_tolerance", 35.0)
        self.declare_parameter("task2_bridge_entry_lateral_tolerance", 45.0)
        self.declare_parameter("task2_bridge_entry_progress_epsilon", 0.03)
        self.declare_parameter("task2_bridge_require_connected_entry", False)
        self.declare_parameter("task2_bridge_min_connection_score", 0.002)
        self.declare_parameter("task2_bridge_max_connection_gap_y_ratio", 0.08)
        self.declare_parameter("task2_bridge_connection_center_tolerance", 75.0)
        self.declare_parameter("task2_bridge_road_center_tolerance", 65.0)
        self.declare_parameter("task2_bridge_central_connection_min_score", 0.20)
        self.declare_parameter("task2_bridge_entry_phase_timeout_seconds", 2.4)
        self.declare_parameter("task2_bridge_entry_backup_seconds", 0.7)
        self.declare_parameter("task2_bridge_entry_shift_seconds", 0.8)
        self.declare_parameter("task2_bridge_entry_rotate_seconds", 0.5)
        self.declare_parameter("task2_bridge_entry_commit_seconds", 0.8)
        self.declare_parameter("task2_bridge_entry_max_recovery_cycles", 4)
        self.declare_parameter("task2_bridge_detect_min_area_ratio", 0.006)
        self.declare_parameter("task2_bridge_turn_tolerance", 70.0)
        self.declare_parameter("task2_bridge_turn_confirm_seconds", 2.0) #*
        self.declare_parameter("task2_bridge_approach_soft_tolerance", 45.0)
        self.declare_parameter("task2_bridge_approach_hard_tolerance", 100.0)
        self.declare_parameter("task2_bridge_entry_bottom_coverage_close", 0.12)
        self.declare_parameter("task2_bridge_entry_bottom_y_ratio_close", 0.82)
        self.declare_parameter("task2_bridge_entry_area_ratio_close", 0.12)
        self.declare_parameter("task2_bridge_final_align_tolerance", 25.0)
        self.declare_parameter("task2_bridge_final_confirm_seconds", 0.6)
        self.declare_parameter("task2_bridge_entry_roi_min_y_ratio", 0.55)
        self.declare_parameter("task2_bridge_entry_close_bottom_y_ratio", 0.82)
        self.declare_parameter("task2_bridge_entry_close_bottom_coverage", 0.16)
        self.declare_parameter("task2_bridge_entry_close_area_ratio", 0.14)
        self.declare_parameter("task2_bridge_entry_final_tolerance", 22.0)
        self.declare_parameter("task2_bridge_entry_confirm_seconds", 0.7)
        self.declare_parameter("task2_bridge_orbit_enabled", True)
        self.declare_parameter("task2_bridge_orbit_side_keep_pixels", 130.0)
        self.declare_parameter("task2_bridge_orbit_forward_seconds", 1.2)
        self.declare_parameter("task2_bridge_orbit_turn_seconds", 2.0) #*
        self.declare_parameter("task2_bridge_orbit_max_cycles", 5)
        self.declare_parameter("task2_bridge_lost_grace_seconds", 0.6)
        self.declare_parameter("task2_bridge_tracking_loss_grace_seconds", 1.2)
        self.declare_parameter("task2_bridge_tracking_expire_seconds", 3.0)
        self.declare_parameter("task2_bridge_fresh_required_for_transition", True)
        self.declare_parameter("task2_bridge_approach_timeout_seconds", 25.0)
        self.declare_parameter("task2_bridge_no_entry_progress_timeout_seconds", 4.0)
        self.declare_parameter("task2_bridge_progress_min_bottom_y_delta", 0.03)
        self.declare_parameter("task2_bridge_progress_min_coverage_delta", 0.025)
        self.declare_parameter("task2_road_search_spin_seconds", 1.5)
        self.declare_parameter("task2_road_search_drive_seconds", 6.0)
        self.declare_parameter("task2_scan_complete_radians", 6.0)
        self.declare_parameter("task2_scan_min_seconds", 3.0)
        self.declare_parameter("task2_scan_timeout_seconds", 15.0)
        self.declare_parameter("task2_bridge_detection_confirm_seconds", 0.4) #*
        self.declare_parameter("task2_road_explore_min_area_ratio", 0.015)
        self.declare_parameter("task2_road_explore_min_bottom_coverage", 0.08)
        self.declare_parameter("task2_road_explore_min_width_ratio", 0.20)
        self.declare_parameter("task2_road_explore_center_tolerance", 45.0)
        self.declare_parameter("task2_road_explore_hard_tolerance", 105.0)
        self.declare_parameter("task2_road_explore_segment_seconds", 4.0)
        self.declare_parameter("task2_road_reacquire_timeout_seconds", 5.0)
        self.declare_parameter("task2_explore_goal_min_translation", 0.8)
        self.declare_parameter("task2_explore_max_segment_seconds", 8.0)
        self.declare_parameter("use_segmentation_drivable_guard", False)
        self.declare_parameter("segmentation_timeout_seconds", 1.5)
        self.declare_parameter("segmentation_missing_grace_seconds", 0.8)
        self.declare_parameter("segmentation_smoothing_alpha", 0.45)
        self.declare_parameter("drivable_center_tolerance",100.0)
        self.declare_parameter("drivable_soft_turn_tolerance", 45.0)
        self.declare_parameter("drivable_bridge_soft_turn_tolerance", 45.0)
        self.declare_parameter("drivable_turn_hysteresis_pixels", 15.0)
        self.declare_parameter("drivable_action_hold_seconds", 0.2)
        self.declare_parameter("drivable_anchor_bottom_coverage", 0.06)
        self.declare_parameter("drivable_anchor_min_area_ratio", 0.01)
        self.declare_parameter("drivable_anchor_grace_seconds", 0.35)
        self.declare_parameter("drivable_min_bottom_coverage", 0.04)
        self.declare_parameter("observe_distance", 0.6)
        self.declare_parameter("observe_seconds", 5.0)
        self.declare_parameter("observe_distance_margin", 0.6)
        self.declare_parameter("observe_target_loss_grace_seconds", 2.0)
        self.declare_parameter("grab_distance", 0.40)
        self.declare_parameter("grab_align_pixel_tolerance", 20.0)
        self.declare_parameter("grab_confirm_seconds", 0.6)
        self.declare_parameter("grab_lost_target_secure_margin", 0.25)
        self.declare_parameter("grab_approach_timeout_seconds", 10.0)
        self.declare_parameter("stationary_tolerance", 0.05)
        self.declare_parameter("goal_tolerance", 0.45)
        self.declare_parameter("path_goal_tolerance", 0.75)
        self.declare_parameter("goal_republish_period_seconds", 1.0)
        self.declare_parameter("navigation_no_plan_timeout_seconds", 2.0)
        self.declare_parameter("return_direct_fallback", True)
        self.declare_parameter("lookahead_distance", 0.55)
        self.declare_parameter("angle_tolerance_deg", 20.0)
        self.declare_parameter("align_pixel_tolerance", 80.0)
        self.declare_parameter("target_timeout_seconds", 1.5)
        self.declare_parameter("task1_require_target_on_road", False)
        self.declare_parameter("task1_allow_target_without_road_mask", False)
        self.declare_parameter("task1_road_target_min_area_ratio", 0.005)
        self.declare_parameter("task1_road_target_x_tolerance_pixels", 180.0)
        self.declare_parameter("task1_road_target_max_width_ratio", 0.58)
        self.declare_parameter("task1_road_target_y_margin_ratio", 0.08)
        self.declare_parameter("task1_road_target_log_period_seconds", 1.0)
        self.declare_parameter("task1_target_lower_min_center_y_ratio", 0.52)
        self.declare_parameter("task1_target_lower_min_bottom_y_ratio", 0.68)
        self.declare_parameter("task1_target_min_bbox_area_ratio", 0.004)
        self.declare_parameter("task1_target_center_min_ratio", 0.15)
        self.declare_parameter("task1_target_center_max_ratio", 0.85)
        self.declare_parameter("grab_bbox_center_min_x_ratio", 0.38)
        self.declare_parameter("grab_bbox_center_max_x_ratio", 0.62)
        self.declare_parameter("grab_bbox_min_center_y_ratio", 0.55)
        self.declare_parameter("grab_bbox_min_bottom_y_ratio", 0.72)
        self.declare_parameter("grab_bbox_max_bottom_y_ratio", 1.0)
        self.declare_parameter("grab_bbox_min_area_ratio", 0.00008) #*
        self.declare_parameter("grab_gate_loss_grace_seconds", 0.4)
        self.declare_parameter("task2_target_bridge_min_overlap_ratio", 0.12)
        self.declare_parameter("task2_target_bridge_min_lower_overlap_ratio", 0.20)
        self.declare_parameter("task2_target_bridge_confirm_seconds", 0.5)
        self.declare_parameter("task2_target_bridge_loss_grace_seconds", 0.7)
        self.declare_parameter("task2_bridge_corridor_min_width_ratio", 0.20) #*
        self.declare_parameter("task2_bridge_corridor_side_margin_pixels", 30.0)
        self.declare_parameter("task2_bridge_corridor_center_tolerance", 25.0)
        self.declare_parameter("task2_bridge_corridor_hard_tolerance", 75.0)
        self.declare_parameter("task2_bridge_corridor_loss_grace_seconds", 0.5)
        self.declare_parameter("task2_turn_visual_deadband_pixels", 25.0)
        self.declare_parameter("task2_turn_visual_coarse_pixels", 90.0)
        self.declare_parameter("task2_turn_map_yaw_tolerance_deg", 8.0)
        self.declare_parameter("task2_turn_command_pulse_seconds", 0.18)
        self.declare_parameter("task2_turn_settle_seconds", 0.08)
        self.declare_parameter("task2_turn_max_cached_control_seconds", 0.8)
        self.declare_parameter("task2_turn_center_confirm_frames", 4)
        self.declare_parameter("task2_turn_min_state_seconds", 0.8)
        self.declare_parameter("task2_turn_wrong_way_pixel_epsilon", 8.0)
        self.declare_parameter("task2_turn_wrong_way_limit", 2)
        self.declare_parameter("task2_turn_direction_sign", 1.0)
        self.declare_parameter("task2_turn_auto_flip_enabled", True)
        self.declare_parameter("task2_turn_allow_frontal_bridge_ascent", True)
        self.declare_parameter("task2_turn_frontal_bridge_min_frontalness", 0.82)
        self.declare_parameter("task2_turn_frontal_bridge_min_confidence", 0.80)
        self.declare_parameter("task2_turn_frontal_bridge_min_bottom_y_ratio", 0.95)
        self.declare_parameter("task2_turn_frontal_bridge_center_tolerance_pixels", 65.0)
        self.declare_parameter("task2_turn_frontal_bridge_bear_tolerance_pixels", 70.0)
        self.declare_parameter("task2_turn_frontal_bridge_require_target", True)
        self.declare_parameter("task2_approach_min_seconds", 1.0)
        self.declare_parameter("task2_entry_close_confirm_frames", 5)
        self.declare_parameter("task2_entry_close_max_range_m", 0.90)
        self.declare_parameter("task2_entry_close_min_connection_score", 0.01)
        self.declare_parameter("task2_entry_close_min_frontalness", 0.55)
        self.declare_parameter("task2_entry_close_max_centerline_slope_pixels", 55.0)
        self.declare_parameter("task2_final_align_loss_timeout_seconds", 1.5)
        self.declare_parameter("task2_final_align_close_hysteresis_seconds", 0.8)
        self.declare_parameter("task2_final_align_confirm_frames", 6)
        self.declare_parameter("bridge_map_projection_enabled", True)
        self.declare_parameter("bridge_map_tf_timeout_seconds", 0.20)
        self.declare_parameter("bridge_map_max_tf_age_seconds", 0.50)
        self.declare_parameter("bridge_landmark_min_observations", 3)
        self.declare_parameter("bridge_landmark_max_point_std_m", 0.30)
        self.declare_parameter("bridge_landmark_min_width_m", 0.25)
        self.declare_parameter("bridge_landmark_max_width_m", 2.50)
        self.declare_parameter("bridge_side_inflation_radius_m", 0.18)
        self.declare_parameter("bridge_entry_opening_keep_clear_m", 0.30)
        self.declare_parameter("bridge_entry_gate_clear_radius_m", 0.45)
        self.declare_parameter("bridge_entry_gate_clear_width_m", 0.65)
        self.declare_parameter("bridge_side_obstacle_start_after_entry_m", 0.35)
        self.declare_parameter("bridge_center_corridor_clear_width_m", 0.45)
        self.declare_parameter("bridge_side_commit_min_points", 4)
        self.declare_parameter("bridge_side_commit_min_observations", 2)
        self.declare_parameter("bridge_side_obstacle_max_entry_distance_m", 0.0)
        self.declare_parameter("bridge_side_line_width_m", 0.08)
        self.declare_parameter("bridge_side_line_max_point_gap_m", 0.35)
        self.declare_parameter("bridge_side_line_min_length_m", 0.40)
        self.declare_parameter("bridge_side_line_outlier_distance_m", 0.25)
        self.declare_parameter("bridge_side_line_smoothing_enabled", True)
        self.declare_parameter("bridge_side_line_fit_enabled", True)
        self.declare_parameter("bridge_geometry_max_jump_m", 0.35)
        self.declare_parameter("bridge_geometry_parallel_angle_tolerance_deg", 25.0)
        self.declare_parameter("bridge_geometry_entry_must_lie_between_sides", True)
        self.declare_parameter("bridge_geometry_pre_entry_must_be_before_gate", True)
        self.declare_parameter("robot_footprint_radius_m", 0.22)
        self.declare_parameter("motion_safety_sample_count", 5)
        self.declare_parameter("augmented_map_enabled", True)
        self.declare_parameter("augmented_map_publish_rate_hz", 2.0)
        self.declare_parameter("bridge_markers_publish_rate_hz", 5.0)
        self.declare_parameter("augmented_map_min_line_cells", 1)
        self.declare_parameter("augmented_map_debug_bridge_lines", True)
        self.declare_parameter("augmented_map_bridge_line_debug_width_cells", 2)
        self.declare_parameter("mission_log_to_file", True)
        self.declare_parameter("mission_log_directory", "/tmp/task_mission_logs")
        self.declare_parameter("mission_log_jsonl", True)
        self.declare_parameter("mission_log_text", True)
        self.declare_parameter("mission_log_flush_every_event", True)
        self.declare_parameter("mission_time_limit_seconds", 600.0)
        self.declare_parameter("task2_target_total_budget_seconds", 360.0)
        self.declare_parameter("task2_search_budget_seconds", 45.0)
        self.declare_parameter("task2_entry_budget_seconds", 90.0)
        self.declare_parameter("task2_ascent_budget_seconds", 20.0)
        self.declare_parameter("task2_top_bear_search_budget_seconds", 60.0)
        self.declare_parameter("task2_descent_budget_seconds", 12.0)
        self.declare_parameter("task_return_budget_seconds", 90.0)
        self.declare_parameter("task2_top_search_rotate_seconds", 8.0)
        self.declare_parameter("task2_top_search_reverse_recenter_seconds", 0.6)
        self.declare_parameter("task2_top_search_max_forward_seconds", 0.0)
        self.declare_parameter("task2_top_search_allow_forward", False)
        self.declare_parameter("task2_top_search_timeout_seconds", 60.0)
        self.declare_parameter("task2_ascent_stop_settle_seconds", 0.8)
        self.declare_parameter("task2_require_bear_secured_before_descent", True)
        self.declare_parameter("task2_entry_unknown_block_stop_seconds", 0.2)
        self.declare_parameter("task2_entry_unknown_block_backup_seconds", 0.4)
        self.declare_parameter("task2_entry_unknown_block_turn_seconds", 0.5)
        self.declare_parameter("task2_entry_unknown_block_max_retries", 3)
        self.declare_parameter("task2_entry_ignore_virtual_obstacle_inside_gate", True)
        self.declare_parameter("task2_entry_virtual_obstacle_gate_margin_m", 0.35)
        self.declare_parameter("task2_entry_allow_cautious_forward_without_map_sides", True)
        self.declare_parameter("task2_side_view_recovery_enabled", True)
        self.declare_parameter("task2_side_view_max_turn_seconds", 2.5)
        self.declare_parameter("task2_side_view_backup_seconds", 0.5)
        self.declare_parameter("task2_side_view_road_follow_seconds", 2.0)
        self.declare_parameter("task2_side_view_stuck_timeout_seconds", 1.2)
        self.declare_parameter("task2_side_view_min_ramp_confidence", 0.45)
        self.declare_parameter("task2_side_view_max_score", 0.55)
        self.declare_parameter("task2_allow_ramp_fallback_entry", True)
        self.declare_parameter("task2_ramp_entry_min_confidence", 0.55)
        self.declare_parameter("task2_ramp_entry_confirm_frames", 6)
        self.declare_parameter("task2_ramp_entry_center_tolerance_pixels", 35.0)
        self.declare_parameter("task2_ramp_entry_min_bottom_y_ratio", 0.70)
        self.declare_parameter("task2_ramp_entry_min_vertical_coverage", 0.45)
        self.declare_parameter("task2_ramp_entry_max_side_view_score", 0.45)
        self.declare_parameter("task2_top_use_tf_z", True)
        self.declare_parameter("task2_top_z_threshold_m", 0.18)
        self.declare_parameter("task2_top_min_ascent_seconds", 7.0)
        self.declare_parameter("task2_top_confirm_frames", 5)
        self.declare_parameter("task2_top_visual_confidence_threshold", 0.55)
        self.declare_parameter("task2_bridge_bear_memory_ttl_seconds", 12.0)
        self.declare_parameter("task2_bridge_bear_memory_min_confidence", 0.45)
        self.declare_parameter("task2_relax_bridge_surface_after_top", True)
        self.declare_parameter("task2_top_bear_require_visible_bbox", True)
        self.declare_parameter("task2_top_bear_allow_cached_memory", True)
        self.declare_parameter("task2_top_search_turn_direction_switch_seconds", 4.0)
        self.declare_parameter("task2_top_search_use_cached_bear_direction", True)
        self.declare_parameter("task2_top_search_allow_short_recenter", True)
        self.declare_parameter("task2_top_search_short_recenter_seconds", 0.3)
        self.declare_parameter("control_period_seconds", 0.1)
        self.declare_parameter("exploration_grid_spacing", 1.2)
        self.declare_parameter("exploration_clearance", 0.35)
        self.declare_parameter("exploration_min_goal_distance", 0.8)
        self.declare_parameter("exploration_goal_timeout_seconds", 25.0)
        self.declare_parameter("exploration_scan_seconds", 2.0)
        self.declare_parameter("map_free_threshold", 25)
        self.declare_parameter("stuck_recovery_enabled", True)
        self.declare_parameter("stuck_timeout_seconds", 1.5)
        self.declare_parameter("stuck_min_translation", 0.05)
        self.declare_parameter("stuck_min_yaw_change", 0.10)
        self.declare_parameter("stuck_recovery_stop_seconds", 0.25)
        self.declare_parameter("stuck_recovery_back_seconds", 0.8)
        self.declare_parameter("stuck_recovery_turn_seconds", 0.7)
        self.declare_parameter("stuck_recovery_shift_seconds", 0.5)
        self.declare_parameter("stuck_recovery_cooldown_seconds", 1.0)
        self.declare_parameter("virtual_obstacle_enabled", False)
        self.declare_parameter("virtual_obstacle_front_distance", 0.45)
        self.declare_parameter("virtual_obstacle_diagonal_distance", 0.42)
        self.declare_parameter("virtual_obstacle_radius", 0.35)
        self.declare_parameter("virtual_obstacle_merge_distance", 0.30)
        self.declare_parameter("virtual_obstacle_ttl_seconds", 180.0)
        self.declare_parameter("virtual_obstacle_max_count", 100)
        self.declare_parameter("virtual_obstacle_shape", "line")
        self.declare_parameter("virtual_obstacle_line_length_m", 0.35)
        self.declare_parameter("virtual_obstacle_line_width_m", 0.08)
        self.declare_parameter("virtual_obstacle_display_as_cylinder", False)
        self.declare_parameter("enable_grab_sequence", True)
        self.declare_parameter("grab_wait_seconds", 3.0)
        self.declare_parameter("verify_grab_enabled", True)
        self.declare_parameter("verify_grab_seconds", 0.8)
        self.declare_parameter("verify_grab_timeout_seconds", 4.0)
        self.declare_parameter("verify_grab_bbox_timeout_seconds", 1.5)
        self.declare_parameter("verify_grab_retry_limit", 2)
        self.declare_parameter("verify_grab_min_bbox_area_ratio", 0.012)
        self.declare_parameter("verify_grab_min_center_x_ratio", 0.25)
        self.declare_parameter("verify_grab_max_center_x_ratio", 0.75)
        self.declare_parameter("verify_grab_min_center_y_ratio", 0.20)
        self.declare_parameter("verify_grab_max_center_y_ratio", 0.95)
        self.declare_parameter("verify_grab_max_depth", 0.75)
        self.declare_parameter("verify_grab_require_lift_evidence", True)
        self.declare_parameter("verify_grab_lift_pixel_delta", 25.0)
        self.declare_parameter("verify_grab_min_area_growth", 1.10)
        self.declare_parameter("return_hold_verify_enabled", True)
        self.declare_parameter("return_hold_verify_loss_grace_seconds", 1.0)
        self.declare_parameter("return_hold_verify_log_period_seconds", 1.0)
        self.declare_parameter("enable_drop_sequence", True)
        self.declare_parameter("drop_wait_seconds", 1.5)
        self.declare_parameter("startup_arm_stow_enabled", True)
        self.declare_parameter("startup_arm_stow_publish_count", 10)
        self.declare_parameter("startup_arm_stow_publish_period_seconds", 0.2)
        self.declare_parameter("arm_positions_in_degrees", True)
        self.declare_parameter(
            "arm_open_positions",
            [180.0, 0.0, 90.0],
        )
        self.declare_parameter(
            "arm_pregrasp_positions",
            [90.0, 120.0, 90.0],
        )
        self.declare_parameter(
            "arm_reach_positions",
            [65.0, 190.0, 90.0],
        )
        self.declare_parameter(
            "arm_close_positions",
            [65.0, 190.0, 20.0],
        )
        self.declare_parameter(
            "arm_lift_positions",
            [90.0, 90.0, 20.0],
        )
        self.declare_parameter(
            "arm_carry_positions",
            [180.0, 0.0, 20.0],
        )
        self.declare_parameter(
            "arm_drop_positions",
            [65.0, 190.0, 20.0],
        )
        self.declare_parameter(
            "arm_release_positions",
            [65.0, 190.0, 90.0],
        )
        self.declare_parameter(
            "arm_drop_retract_positions",
            [180.0, 0.0, 90.0],
        )
        self.declare_parameter(
            "startup_arm_stow_positions",
            [180.0, 0.0, 90.0],
        )

        self.start_pose = self._pose_param("start_pose")
        self.bear_search_pose = self._pose_param("bear_search_pose")
        self.auto_set_initial_pose = self._bool_param("auto_set_initial_pose")
        self.initial_pose = self._pose_param("initial_pose")
        self.initial_pose_publish_count = self._integer_param(
            "initial_pose_publish_count"
        )
        self.initial_pose_publish_period = self._double_param(
            "initial_pose_publish_period_seconds"
        )
        self.initial_pose_position_covariance = self._double_param(
            "initial_pose_position_covariance"
        )
        self.initial_pose_yaw_covariance = self._double_param(
            "initial_pose_yaw_covariance"
        )
        self.use_current_pose_as_start = (
            self.get_parameter("use_current_pose_as_start")
            .get_parameter_value()
            .bool_value
        )
        self.start_pose_locked = not self.use_current_pose_as_start
        self.target_label = (
            self.get_parameter("target_label").get_parameter_value().string_value
        )
        self.auto_explore = self._bool_param("auto_explore")
        self.map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.use_tf_pose_fallback = self._bool_param("use_tf_pose_fallback")
        self.map_frame = self.get_parameter("map_frame").get_parameter_value().string_value
        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.run_task2_after_task1 = self._bool_param("run_task2_after_task1")
        self.mission_mode = self._string_param("mission_mode")
        if self.mission_mode not in (
            "bridge_first_shared_bear",
            "legacy_task1_then_task2",
        ):
            self.get_logger().warn(
                f"Unsupported mission_mode '{self.mission_mode}', using bridge_first_shared_bear."
            )
            self.mission_mode = "bridge_first_shared_bear"
        self.task2_use_bridge_entry_pose = self._bool_param(
            "task2_use_bridge_entry_pose"
        )
        self.task2_bridge_entry_pose = self._pose_param("task2_bridge_entry_pose")
        self.task2_use_bridge_top_pose = self._bool_param("task2_use_bridge_top_pose")
        self.task2_bridge_top_pose = self._pose_param("task2_bridge_top_pose")
        self.task2_use_bridge_exit_pose = self._bool_param("task2_use_bridge_exit_pose")
        self.task2_bridge_exit_pose = self._pose_param("task2_bridge_exit_pose")
        self.task2_ascent_min_seconds = self._double_param("task2_ascent_min_seconds")
        self.task2_ascent_timeout = self._double_param("task2_ascent_timeout_seconds")
        self.task2_ascent_action = self._string_param("task2_ascent_action")
        self.task2_ascent_forward_speed_scale = self._double_param(
            "task2_ascent_forward_speed_scale"
        )
        self.task2_ascent_require_top_confidence = self._bool_param(
            "task2_ascent_require_top_confidence"
        )
        self.task2_ascent_continue_if_not_top = self._bool_param(
            "task2_ascent_continue_if_not_top"
        )
        self.task2_ascent_max_extra_seconds = self._double_param(
            "task2_ascent_max_extra_seconds"
        )
        self.task2_ascent_centering_enabled = self._bool_param(
            "task2_ascent_centering_enabled"
        )
        self.task2_ascent_stop_on_bridge_loss_seconds = self._double_param(
            "task2_ascent_stop_on_bridge_loss_seconds"
        )
        self.task2_ascent_bear_pid_enabled = self._bool_param(
            "task2_ascent_bear_pid_enabled"
        )
        self.task2_ascent_bear_pid_center_tolerance_pixels = self._double_param(
            "task2_ascent_bear_pid_center_tolerance_pixels"
        )
        self.task2_ascent_bear_pid_kp = self._double_param("task2_ascent_bear_pid_kp")
        self.task2_ascent_bear_pid_ki = self._double_param("task2_ascent_bear_pid_ki")
        self.task2_ascent_bear_pid_kd = self._double_param("task2_ascent_bear_pid_kd")
        self.task2_ascent_bear_pid_integral_limit = self._double_param(
            "task2_ascent_bear_pid_integral_limit"
        )
        self.task2_ascent_bear_pid_max_turn = self._double_param(
            "task2_ascent_bear_pid_max_turn"
        )
        self.task2_ascent_bear_pid_rotate_only_pixels = self._double_param(
            "task2_ascent_bear_pid_rotate_only_pixels"
        )
        self.task2_ascent_bear_pid_forward_scale = self._double_param(
            "task2_ascent_bear_pid_forward_scale"
        )
        self.task2_descent_min_seconds = self._double_param("task2_descent_min_seconds")
        self.task2_descent_timeout = self._double_param("task2_descent_timeout_seconds")
        self.task2_bridge_search_timeout = self._double_param(
            "task2_bridge_search_timeout_seconds"
        )
        self.task2_bridge_confirm_seconds = self._double_param(
            "task2_bridge_confirm_seconds"
        )
        self.task2_bridge_min_area_ratio = self._double_param(
            "task2_bridge_min_area_ratio"
        )
        self.task2_bridge_entry_bottom_coverage = self._double_param(
            "task2_bridge_entry_bottom_coverage"
        )
        self.task2_bridge_center_tolerance = self._double_param(
            "task2_bridge_center_tolerance"
        )
        self.task2_bridge_candidate_min_bottom_y_ratio = self._double_param(
            "task2_bridge_candidate_min_bottom_y_ratio"
        )
        self.task2_bridge_entry_center_tolerance = self._double_param(
            "task2_bridge_entry_center_tolerance"
        )
        self.task2_bridge_entry_lateral_tolerance = self._double_param(
            "task2_bridge_entry_lateral_tolerance"
        )
        self.task2_bridge_entry_progress_epsilon = self._double_param(
            "task2_bridge_entry_progress_epsilon"
        )
        self.task2_bridge_require_connected_entry = self._bool_param(
            "task2_bridge_require_connected_entry"
        )
        self.task2_bridge_min_connection_score = self._double_param(
            "task2_bridge_min_connection_score"
        )
        self.task2_bridge_max_connection_gap_y_ratio = self._double_param(
            "task2_bridge_max_connection_gap_y_ratio"
        )
        self.task2_bridge_connection_center_tolerance = self._double_param(
            "task2_bridge_connection_center_tolerance"
        )
        self.task2_bridge_road_center_tolerance = self._double_param(
            "task2_bridge_road_center_tolerance"
        )
        self.task2_bridge_central_connection_min_score = self._double_param(
            "task2_bridge_central_connection_min_score"
        )
        self.task2_bridge_entry_phase_timeout = self._double_param(
            "task2_bridge_entry_phase_timeout_seconds"
        )
        self.task2_bridge_entry_backup_seconds = self._double_param(
            "task2_bridge_entry_backup_seconds"
        )
        self.task2_bridge_entry_shift_seconds = self._double_param(
            "task2_bridge_entry_shift_seconds"
        )
        self.task2_bridge_entry_rotate_seconds = self._double_param(
            "task2_bridge_entry_rotate_seconds"
        )
        self.task2_bridge_entry_commit_seconds = self._double_param(
            "task2_bridge_entry_commit_seconds"
        )
        self.task2_bridge_entry_max_recovery_cycles = self._integer_param(
            "task2_bridge_entry_max_recovery_cycles"
        )
        self.task2_bridge_detect_min_area_ratio = self._double_param(
            "task2_bridge_detect_min_area_ratio"
        )
        self.task2_bridge_turn_tolerance = self._double_param(
            "task2_bridge_turn_tolerance"
        )
        self.task2_bridge_turn_confirm_seconds = self._double_param(
            "task2_bridge_turn_confirm_seconds"
        )
        self.task2_bridge_approach_soft_tolerance = self._double_param(
            "task2_bridge_approach_soft_tolerance"
        )
        self.task2_bridge_approach_hard_tolerance = self._double_param(
            "task2_bridge_approach_hard_tolerance"
        )
        self.task2_bridge_entry_bottom_coverage_close = self._double_param(
            "task2_bridge_entry_bottom_coverage_close"
        )
        self.task2_bridge_entry_bottom_y_ratio_close = self._double_param(
            "task2_bridge_entry_bottom_y_ratio_close"
        )
        self.task2_bridge_entry_area_ratio_close = self._double_param(
            "task2_bridge_entry_area_ratio_close"
        )
        self.task2_bridge_final_align_tolerance = self._double_param(
            "task2_bridge_final_align_tolerance"
        )
        self.task2_bridge_final_confirm_seconds = self._double_param(
            "task2_bridge_final_confirm_seconds"
        )
        self.task2_bridge_entry_roi_min_y_ratio = self._double_param(
            "task2_bridge_entry_roi_min_y_ratio"
        )
        self.task2_bridge_entry_close_bottom_y_ratio = self._double_param(
            "task2_bridge_entry_close_bottom_y_ratio"
        )
        self.task2_bridge_entry_close_bottom_coverage = self._double_param(
            "task2_bridge_entry_close_bottom_coverage"
        )
        self.task2_bridge_entry_close_area_ratio = self._double_param(
            "task2_bridge_entry_close_area_ratio"
        )
        self.task2_bridge_entry_final_tolerance = self._double_param(
            "task2_bridge_entry_final_tolerance"
        )
        self.task2_bridge_entry_confirm_seconds = self._double_param(
            "task2_bridge_entry_confirm_seconds"
        )
        self.task2_bridge_orbit_enabled = self._bool_param(
            "task2_bridge_orbit_enabled"
        )
        self.task2_bridge_orbit_side_keep_pixels = self._double_param(
            "task2_bridge_orbit_side_keep_pixels"
        )
        self.task2_bridge_orbit_forward_seconds = self._double_param(
            "task2_bridge_orbit_forward_seconds"
        )
        self.task2_bridge_orbit_turn_seconds = self._double_param(
            "task2_bridge_orbit_turn_seconds"
        )
        self.task2_bridge_orbit_max_cycles = self._integer_param(
            "task2_bridge_orbit_max_cycles"
        )
        self.task2_bridge_lost_grace_seconds = self._double_param(
            "task2_bridge_lost_grace_seconds"
        )
        self.task2_bridge_tracking_loss_grace = self._double_param(
            "task2_bridge_tracking_loss_grace_seconds"
        )
        self.task2_bridge_tracking_expire = self._double_param(
            "task2_bridge_tracking_expire_seconds"
        )
        self.task2_bridge_fresh_required_for_transition = self._bool_param(
            "task2_bridge_fresh_required_for_transition"
        )
        self.task2_bridge_approach_timeout_seconds = self._double_param(
            "task2_bridge_approach_timeout_seconds"
        )
        self.task2_bridge_no_entry_progress_timeout_seconds = self._double_param(
            "task2_bridge_no_entry_progress_timeout_seconds"
        )
        self.task2_bridge_progress_min_bottom_y_delta = self._double_param(
            "task2_bridge_progress_min_bottom_y_delta"
        )
        self.task2_bridge_progress_min_coverage_delta = self._double_param(
            "task2_bridge_progress_min_coverage_delta"
        )
        self.task2_road_search_spin_seconds = self._double_param(
            "task2_road_search_spin_seconds"
        )
        self.task2_road_search_drive_seconds = self._double_param(
            "task2_road_search_drive_seconds"
        )
        self.task2_scan_complete_radians = self._double_param(
            "task2_scan_complete_radians"
        )
        self.task2_scan_min_seconds = self._double_param("task2_scan_min_seconds")
        self.task2_scan_timeout_seconds = self._double_param(
            "task2_scan_timeout_seconds"
        )
        self.task2_bridge_detection_confirm_seconds = self._double_param(
            "task2_bridge_detection_confirm_seconds"
        )
        self.task2_road_explore_min_area_ratio = self._double_param(
            "task2_road_explore_min_area_ratio"
        )
        self.task2_road_explore_min_bottom_coverage = self._double_param(
            "task2_road_explore_min_bottom_coverage"
        )
        self.task2_road_explore_min_width_ratio = self._double_param(
            "task2_road_explore_min_width_ratio"
        )
        self.task2_road_explore_center_tolerance = self._double_param(
            "task2_road_explore_center_tolerance"
        )
        self.task2_road_explore_hard_tolerance = self._double_param(
            "task2_road_explore_hard_tolerance"
        )
        self.task2_road_explore_segment_seconds = self._double_param(
            "task2_road_explore_segment_seconds"
        )
        self.task2_road_reacquire_timeout_seconds = self._double_param(
            "task2_road_reacquire_timeout_seconds"
        )
        self.task2_explore_goal_min_translation = self._double_param(
            "task2_explore_goal_min_translation"
        )
        self.task2_explore_max_segment_seconds = self._double_param(
            "task2_explore_max_segment_seconds"
        )
        self.use_segmentation_drivable_guard = self._bool_param(
            "use_segmentation_drivable_guard"
        )
        self.segmentation_timeout = self._double_param("segmentation_timeout_seconds")
        self.segmentation_missing_grace = self._double_param(
            "segmentation_missing_grace_seconds"
        )
        self.segmentation_smoothing_alpha = self._double_param(
            "segmentation_smoothing_alpha"
        )
        self.drivable_center_tolerance = self._double_param("drivable_center_tolerance")
        self.drivable_soft_turn_tolerance = self._double_param(
            "drivable_soft_turn_tolerance"
        )
        self.drivable_bridge_soft_turn_tolerance = self._double_param(
            "drivable_bridge_soft_turn_tolerance"
        )
        self.drivable_turn_hysteresis_pixels = self._double_param(
            "drivable_turn_hysteresis_pixels"
        )
        self.drivable_action_hold_seconds = self._double_param(
            "drivable_action_hold_seconds"
        )
        self.drivable_anchor_bottom_coverage = self._double_param(
            "drivable_anchor_bottom_coverage"
        )
        self.drivable_anchor_min_area_ratio = self._double_param(
            "drivable_anchor_min_area_ratio"
        )
        self.drivable_anchor_grace_seconds = self._double_param(
            "drivable_anchor_grace_seconds"
        )
        self.drivable_min_bottom_coverage = self._double_param(
            "drivable_min_bottom_coverage"
        )
        self.observe_distance = self._double_param("observe_distance")
        self.observe_seconds = self._double_param("observe_seconds")
        self.observe_distance_margin = self._double_param("observe_distance_margin")
        self.observe_target_loss_grace = self._double_param(
            "observe_target_loss_grace_seconds"
        )
        self.grab_distance = self._double_param("grab_distance")
        self.grab_align_pixel_tolerance = self._double_param("grab_align_pixel_tolerance")
        self.grab_confirm_seconds = self._double_param("grab_confirm_seconds")
        self.grab_lost_target_secure_margin = self._double_param(
            "grab_lost_target_secure_margin"
        )
        self.grab_approach_timeout = self._double_param("grab_approach_timeout_seconds")
        self.stationary_tolerance = self._double_param("stationary_tolerance")
        self.goal_tolerance = self._double_param("goal_tolerance")
        self.path_goal_tolerance = self._double_param("path_goal_tolerance")
        self.goal_republish_period = self._double_param("goal_republish_period_seconds")
        self.navigation_no_plan_timeout = self._double_param(
            "navigation_no_plan_timeout_seconds"
        )
        self.return_direct_fallback = self._bool_param("return_direct_fallback")
        self.lookahead_distance = self._double_param("lookahead_distance")
        self.angle_tolerance = math.radians(self._double_param("angle_tolerance_deg"))
        self.align_pixel_tolerance = self._double_param("align_pixel_tolerance")
        self.target_timeout = self._double_param("target_timeout_seconds")
        self.task1_require_target_on_road = self._bool_param(
            "task1_require_target_on_road"
        )
        self.task1_allow_target_without_road_mask = self._bool_param(
            "task1_allow_target_without_road_mask"
        )
        self.task1_road_target_min_area_ratio = self._double_param(
            "task1_road_target_min_area_ratio"
        )
        self.task1_road_target_x_tolerance_pixels = self._double_param(
            "task1_road_target_x_tolerance_pixels"
        )
        self.task1_road_target_max_width_ratio = self._double_param(
            "task1_road_target_max_width_ratio"
        )
        self.task1_road_target_y_margin_ratio = self._double_param(
            "task1_road_target_y_margin_ratio"
        )
        self.task1_road_target_log_period = self._double_param(
            "task1_road_target_log_period_seconds"
        )
        self.task1_target_lower_min_center_y_ratio = self._double_param(
            "task1_target_lower_min_center_y_ratio"
        )
        self.task1_target_lower_min_bottom_y_ratio = self._double_param(
            "task1_target_lower_min_bottom_y_ratio"
        )
        self.task1_target_min_bbox_area_ratio = self._double_param(
            "task1_target_min_bbox_area_ratio"
        )
        self.task1_target_center_min_ratio = self._double_param(
            "task1_target_center_min_ratio"
        )
        self.task1_target_center_max_ratio = self._double_param(
            "task1_target_center_max_ratio"
        )
        self.grab_bbox_center_min_x_ratio = self._double_param(
            "grab_bbox_center_min_x_ratio"
        )
        self.grab_bbox_center_max_x_ratio = self._double_param(
            "grab_bbox_center_max_x_ratio"
        )
        self.grab_bbox_min_center_y_ratio = self._double_param(
            "grab_bbox_min_center_y_ratio"
        )
        self.grab_bbox_min_bottom_y_ratio = self._double_param(
            "grab_bbox_min_bottom_y_ratio"
        )
        self.grab_bbox_max_bottom_y_ratio = self._double_param(
            "grab_bbox_max_bottom_y_ratio"
        )
        self.grab_bbox_min_area_ratio = self._double_param("grab_bbox_min_area_ratio")
        self.grab_gate_loss_grace_seconds = self._double_param(
            "grab_gate_loss_grace_seconds"
        )
        self.task2_target_bridge_min_overlap_ratio = self._double_param(
            "task2_target_bridge_min_overlap_ratio"
        )
        self.task2_target_bridge_min_lower_overlap_ratio = self._double_param(
            "task2_target_bridge_min_lower_overlap_ratio"
        )
        self.task2_target_bridge_confirm_seconds = self._double_param(
            "task2_target_bridge_confirm_seconds"
        )
        self.task2_target_bridge_loss_grace_seconds = self._double_param(
            "task2_target_bridge_loss_grace_seconds"
        )
        self.task2_bridge_corridor_min_width_ratio = self._double_param(
            "task2_bridge_corridor_min_width_ratio"
        )
        self.task2_bridge_corridor_side_margin_pixels = self._double_param(
            "task2_bridge_corridor_side_margin_pixels"
        )
        self.task2_bridge_corridor_center_tolerance = self._double_param(
            "task2_bridge_corridor_center_tolerance"
        )
        self.task2_bridge_corridor_hard_tolerance = self._double_param(
            "task2_bridge_corridor_hard_tolerance"
        )
        self.task2_bridge_corridor_loss_grace_seconds = self._double_param(
            "task2_bridge_corridor_loss_grace_seconds"
        )
        self.task2_turn_visual_deadband_pixels = self._double_param(
            "task2_turn_visual_deadband_pixels"
        )
        self.task2_turn_visual_coarse_pixels = self._double_param(
            "task2_turn_visual_coarse_pixels"
        )
        self.task2_turn_map_yaw_tolerance = math.radians(
            self._double_param("task2_turn_map_yaw_tolerance_deg")
        )
        self.task2_turn_command_pulse_seconds = self._double_param(
            "task2_turn_command_pulse_seconds"
        )
        self.task2_turn_settle_seconds = self._double_param(
            "task2_turn_settle_seconds"
        )
        self.task2_turn_max_cached_control_seconds = self._double_param(
            "task2_turn_max_cached_control_seconds"
        )
        self.task2_turn_center_confirm_frames = self._integer_param(
            "task2_turn_center_confirm_frames"
        )
        self.task2_turn_min_state_seconds = self._double_param(
            "task2_turn_min_state_seconds"
        )
        self.task2_turn_wrong_way_pixel_epsilon = self._double_param(
            "task2_turn_wrong_way_pixel_epsilon"
        )
        self.task2_turn_wrong_way_limit = self._integer_param(
            "task2_turn_wrong_way_limit"
        )
        self.task2_turn_direction_sign = self._double_param("task2_turn_direction_sign")
        self.task2_turn_auto_flip_enabled = self._bool_param(
            "task2_turn_auto_flip_enabled"
        )
        self.task2_turn_allow_frontal_bridge_ascent = self._bool_param(
            "task2_turn_allow_frontal_bridge_ascent"
        )
        self.task2_turn_frontal_bridge_min_frontalness = self._double_param(
            "task2_turn_frontal_bridge_min_frontalness"
        )
        self.task2_turn_frontal_bridge_min_confidence = self._double_param(
            "task2_turn_frontal_bridge_min_confidence"
        )
        self.task2_turn_frontal_bridge_min_bottom_y_ratio = self._double_param(
            "task2_turn_frontal_bridge_min_bottom_y_ratio"
        )
        self.task2_turn_frontal_bridge_center_tolerance_pixels = self._double_param(
            "task2_turn_frontal_bridge_center_tolerance_pixels"
        )
        self.task2_turn_frontal_bridge_bear_tolerance_pixels = self._double_param(
            "task2_turn_frontal_bridge_bear_tolerance_pixels"
        )
        self.task2_turn_frontal_bridge_require_target = self._bool_param(
            "task2_turn_frontal_bridge_require_target"
        )
        self.task2_approach_min_seconds = self._double_param(
            "task2_approach_min_seconds"
        )
        self.task2_entry_close_confirm_frames = self._integer_param(
            "task2_entry_close_confirm_frames"
        )
        self.task2_entry_close_max_range_m = self._double_param(
            "task2_entry_close_max_range_m"
        )
        self.task2_entry_close_min_connection_score = self._double_param(
            "task2_entry_close_min_connection_score"
        )
        self.task2_entry_close_min_frontalness = self._double_param(
            "task2_entry_close_min_frontalness"
        )
        self.task2_entry_close_max_centerline_slope_pixels = self._double_param(
            "task2_entry_close_max_centerline_slope_pixels"
        )
        self.task2_final_align_loss_timeout = self._double_param(
            "task2_final_align_loss_timeout_seconds"
        )
        self.task2_final_align_close_hysteresis = self._double_param(
            "task2_final_align_close_hysteresis_seconds"
        )
        self.task2_final_align_confirm_frames = self._integer_param(
            "task2_final_align_confirm_frames"
        )
        self.bridge_map_projection_enabled = self._bool_param(
            "bridge_map_projection_enabled"
        )
        self.bridge_map_tf_timeout = self._double_param("bridge_map_tf_timeout_seconds")
        self.bridge_map_max_tf_age = self._double_param("bridge_map_max_tf_age_seconds")
        self.bridge_landmark_min_observations = self._integer_param(
            "bridge_landmark_min_observations"
        )
        self.bridge_landmark_max_point_std = self._double_param(
            "bridge_landmark_max_point_std_m"
        )
        self.bridge_landmark_min_width = self._double_param(
            "bridge_landmark_min_width_m"
        )
        self.bridge_landmark_max_width = self._double_param(
            "bridge_landmark_max_width_m"
        )
        self.bridge_side_inflation_radius = self._double_param(
            "bridge_side_inflation_radius_m"
        )
        self.bridge_entry_opening_keep_clear = self._double_param(
            "bridge_entry_opening_keep_clear_m"
        )
        self.bridge_entry_gate_clear_radius = self._double_param(
            "bridge_entry_gate_clear_radius_m"
        )
        self.bridge_entry_gate_clear_width = self._double_param(
            "bridge_entry_gate_clear_width_m"
        )
        self.bridge_side_obstacle_start_after_entry = self._double_param(
            "bridge_side_obstacle_start_after_entry_m"
        )
        self.bridge_center_corridor_clear_width = self._double_param(
            "bridge_center_corridor_clear_width_m"
        )
        self.bridge_side_commit_min_points = self._integer_param(
            "bridge_side_commit_min_points"
        )
        self.bridge_side_commit_min_observations = self._integer_param(
            "bridge_side_commit_min_observations"
        )
        self.bridge_side_obstacle_max_entry_distance = self._double_param(
            "bridge_side_obstacle_max_entry_distance_m"
        )
        self.bridge_side_line_width = self._double_param("bridge_side_line_width_m")
        self.bridge_side_line_max_point_gap = self._double_param(
            "bridge_side_line_max_point_gap_m"
        )
        self.bridge_side_line_min_length = self._double_param(
            "bridge_side_line_min_length_m"
        )
        self.bridge_side_line_outlier_distance = self._double_param(
            "bridge_side_line_outlier_distance_m"
        )
        self.bridge_side_line_smoothing_enabled = self._bool_param(
            "bridge_side_line_smoothing_enabled"
        )
        self.bridge_side_line_fit_enabled = self._bool_param(
            "bridge_side_line_fit_enabled"
        )
        self.bridge_geometry_max_jump = self._double_param(
            "bridge_geometry_max_jump_m"
        )
        self.bridge_geometry_parallel_angle_tolerance = math.radians(
            self._double_param("bridge_geometry_parallel_angle_tolerance_deg")
        )
        self.bridge_geometry_entry_must_lie_between_sides = self._bool_param(
            "bridge_geometry_entry_must_lie_between_sides"
        )
        self.bridge_geometry_pre_entry_must_be_before_gate = self._bool_param(
            "bridge_geometry_pre_entry_must_be_before_gate"
        )
        self.robot_footprint_radius = self._double_param("robot_footprint_radius_m")
        self.motion_safety_sample_count = self._integer_param(
            "motion_safety_sample_count"
        )
        self.augmented_map_enabled = self._bool_param("augmented_map_enabled")
        self.augmented_map_publish_rate = self._double_param(
            "augmented_map_publish_rate_hz"
        )
        self.bridge_markers_publish_rate = self._double_param(
            "bridge_markers_publish_rate_hz"
        )
        self.augmented_map_min_line_cells = self._integer_param(
            "augmented_map_min_line_cells"
        )
        self.augmented_map_debug_bridge_lines = self._bool_param(
            "augmented_map_debug_bridge_lines"
        )
        self.augmented_map_bridge_line_debug_width_cells = self._integer_param(
            "augmented_map_bridge_line_debug_width_cells"
        )
        self.mission_log_to_file = self._bool_param("mission_log_to_file")
        self.mission_log_directory = self._string_param("mission_log_directory")
        self.mission_log_jsonl = self._bool_param("mission_log_jsonl")
        self.mission_log_text = self._bool_param("mission_log_text")
        self.mission_log_flush_every_event = self._bool_param(
            "mission_log_flush_every_event"
        )
        self.mission_time_limit_seconds = self._double_param(
            "mission_time_limit_seconds"
        )
        self.task2_target_total_budget_seconds = self._double_param(
            "task2_target_total_budget_seconds"
        )
        self.task2_search_budget_seconds = self._double_param(
            "task2_search_budget_seconds"
        )
        self.task2_entry_budget_seconds = self._double_param(
            "task2_entry_budget_seconds"
        )
        self.task2_ascent_budget_seconds = self._double_param(
            "task2_ascent_budget_seconds"
        )
        self.task2_top_bear_search_budget_seconds = self._double_param(
            "task2_top_bear_search_budget_seconds"
        )
        self.task2_descent_budget_seconds = self._double_param(
            "task2_descent_budget_seconds"
        )
        self.task_return_budget_seconds = self._double_param(
            "task_return_budget_seconds"
        )
        self.task2_top_search_rotate_seconds = self._double_param(
            "task2_top_search_rotate_seconds"
        )
        self.task2_top_search_reverse_recenter_seconds = self._double_param(
            "task2_top_search_reverse_recenter_seconds"
        )
        self.task2_top_search_max_forward_seconds = self._double_param(
            "task2_top_search_max_forward_seconds"
        )
        self.task2_top_search_allow_forward = self._bool_param(
            "task2_top_search_allow_forward"
        )
        self.task2_top_search_timeout_seconds = self._double_param(
            "task2_top_search_timeout_seconds"
        )
        self.task2_ascent_stop_settle_seconds = self._double_param(
            "task2_ascent_stop_settle_seconds"
        )
        self.task2_require_bear_secured_before_descent = self._bool_param(
            "task2_require_bear_secured_before_descent"
        )
        self.task2_entry_unknown_block_stop_seconds = self._double_param(
            "task2_entry_unknown_block_stop_seconds"
        )
        self.task2_entry_unknown_block_backup_seconds = self._double_param(
            "task2_entry_unknown_block_backup_seconds"
        )
        self.task2_entry_unknown_block_turn_seconds = self._double_param(
            "task2_entry_unknown_block_turn_seconds"
        )
        self.task2_entry_unknown_block_max_retries = self._integer_param(
            "task2_entry_unknown_block_max_retries"
        )
        self.task2_entry_ignore_virtual_obstacle_inside_gate = self._bool_param(
            "task2_entry_ignore_virtual_obstacle_inside_gate"
        )
        self.task2_entry_virtual_obstacle_gate_margin = self._double_param(
            "task2_entry_virtual_obstacle_gate_margin_m"
        )
        self.task2_entry_allow_cautious_forward_without_map_sides = self._bool_param(
            "task2_entry_allow_cautious_forward_without_map_sides"
        )
        self.task2_side_view_recovery_enabled = self._bool_param(
            "task2_side_view_recovery_enabled"
        )
        self.task2_side_view_max_turn_seconds = self._double_param(
            "task2_side_view_max_turn_seconds"
        )
        self.task2_side_view_backup_seconds = self._double_param(
            "task2_side_view_backup_seconds"
        )
        self.task2_side_view_road_follow_seconds = self._double_param(
            "task2_side_view_road_follow_seconds"
        )
        self.task2_side_view_stuck_timeout_seconds = self._double_param(
            "task2_side_view_stuck_timeout_seconds"
        )
        self.task2_side_view_min_ramp_confidence = self._double_param(
            "task2_side_view_min_ramp_confidence"
        )
        self.task2_side_view_max_score = self._double_param(
            "task2_side_view_max_score"
        )
        self.task2_allow_ramp_fallback_entry = self._bool_param(
            "task2_allow_ramp_fallback_entry"
        )
        self.task2_ramp_entry_min_confidence = self._double_param(
            "task2_ramp_entry_min_confidence"
        )
        self.task2_ramp_entry_confirm_frames = self._integer_param(
            "task2_ramp_entry_confirm_frames"
        )
        self.task2_ramp_entry_center_tolerance_pixels = self._double_param(
            "task2_ramp_entry_center_tolerance_pixels"
        )
        self.task2_ramp_entry_min_bottom_y_ratio = self._double_param(
            "task2_ramp_entry_min_bottom_y_ratio"
        )
        self.task2_ramp_entry_min_vertical_coverage = self._double_param(
            "task2_ramp_entry_min_vertical_coverage"
        )
        self.task2_ramp_entry_max_side_view_score = self._double_param(
            "task2_ramp_entry_max_side_view_score"
        )
        self.task2_top_use_tf_z = self._bool_param("task2_top_use_tf_z")
        self.task2_top_z_threshold = self._double_param("task2_top_z_threshold_m")
        self.task2_top_min_ascent_seconds = self._double_param(
            "task2_top_min_ascent_seconds"
        )
        self.task2_top_confirm_frames = self._integer_param(
            "task2_top_confirm_frames"
        )
        self.task2_top_visual_confidence_threshold = self._double_param(
            "task2_top_visual_confidence_threshold"
        )
        self.task2_bridge_bear_memory_ttl_seconds = self._double_param(
            "task2_bridge_bear_memory_ttl_seconds"
        )
        self.task2_bridge_bear_memory_min_confidence = self._double_param(
            "task2_bridge_bear_memory_min_confidence"
        )
        self.task2_relax_bridge_surface_after_top = self._bool_param(
            "task2_relax_bridge_surface_after_top"
        )
        self.task2_top_bear_require_visible_bbox = self._bool_param(
            "task2_top_bear_require_visible_bbox"
        )
        self.task2_top_bear_allow_cached_memory = self._bool_param(
            "task2_top_bear_allow_cached_memory"
        )
        self.task2_top_search_turn_direction_switch_seconds = self._double_param(
            "task2_top_search_turn_direction_switch_seconds"
        )
        self.task2_top_search_use_cached_bear_direction = self._bool_param(
            "task2_top_search_use_cached_bear_direction"
        )
        self.task2_top_search_allow_short_recenter = self._bool_param(
            "task2_top_search_allow_short_recenter"
        )
        self.task2_top_search_short_recenter_seconds = self._double_param(
            "task2_top_search_short_recenter_seconds"
        )
        self.exploration_grid_spacing = self._double_param("exploration_grid_spacing")
        self.exploration_clearance = self._double_param("exploration_clearance")
        self.exploration_min_goal_distance = self._double_param(
            "exploration_min_goal_distance"
        )
        self.exploration_goal_timeout = self._double_param(
            "exploration_goal_timeout_seconds"
        )
        self.exploration_scan_seconds = self._double_param("exploration_scan_seconds")
        self.map_free_threshold = self._integer_param("map_free_threshold")
        self.stuck_recovery_enabled = self._bool_param("stuck_recovery_enabled")
        self.stuck_timeout = self._double_param("stuck_timeout_seconds")
        self.stuck_min_translation = self._double_param("stuck_min_translation")
        self.stuck_min_yaw_change = self._double_param("stuck_min_yaw_change")
        self.stuck_recovery_stop_seconds = self._double_param(
            "stuck_recovery_stop_seconds"
        )
        self.stuck_recovery_back_seconds = self._double_param(
            "stuck_recovery_back_seconds"
        )
        self.stuck_recovery_turn_seconds = self._double_param(
            "stuck_recovery_turn_seconds"
        )
        self.stuck_recovery_shift_seconds = self._double_param(
            "stuck_recovery_shift_seconds"
        )
        self.stuck_recovery_cooldown = self._double_param(
            "stuck_recovery_cooldown_seconds"
        )
        self.virtual_obstacle_enabled = self._bool_param("virtual_obstacle_enabled")
        self.virtual_obstacle_front_distance = self._double_param(
            "virtual_obstacle_front_distance"
        )
        self.virtual_obstacle_diagonal_distance = self._double_param(
            "virtual_obstacle_diagonal_distance"
        )
        self.virtual_obstacle_radius = self._double_param("virtual_obstacle_radius")
        self.virtual_obstacle_merge_distance = self._double_param(
            "virtual_obstacle_merge_distance"
        )
        self.virtual_obstacle_ttl_seconds = self._double_param(
            "virtual_obstacle_ttl_seconds"
        )
        self.virtual_obstacle_max_count = self._integer_param(
            "virtual_obstacle_max_count"
        )
        self.virtual_obstacle_shape = self._string_param("virtual_obstacle_shape")
        self.virtual_obstacle_line_length = self._double_param(
            "virtual_obstacle_line_length_m"
        )
        self.virtual_obstacle_line_width = self._double_param(
            "virtual_obstacle_line_width_m"
        )
        self.virtual_obstacle_display_as_cylinder = self._bool_param(
            "virtual_obstacle_display_as_cylinder"
        )
        self.enable_grab_sequence = (
            self.get_parameter("enable_grab_sequence")
            .get_parameter_value()
            .bool_value
        )
        self.grab_wait_seconds = self._double_param("grab_wait_seconds")
        self.verify_grab_enabled = self._bool_param("verify_grab_enabled")
        self.verify_grab_seconds = self._double_param("verify_grab_seconds")
        self.verify_grab_timeout = self._double_param("verify_grab_timeout_seconds")
        self.verify_grab_bbox_timeout = self._double_param(
            "verify_grab_bbox_timeout_seconds"
        )
        self.verify_grab_retry_limit = self._integer_param("verify_grab_retry_limit")
        self.verify_grab_min_bbox_area_ratio = self._double_param(
            "verify_grab_min_bbox_area_ratio"
        )
        self.verify_grab_min_center_x_ratio = self._double_param(
            "verify_grab_min_center_x_ratio"
        )
        self.verify_grab_max_center_x_ratio = self._double_param(
            "verify_grab_max_center_x_ratio"
        )
        self.verify_grab_min_center_y_ratio = self._double_param(
            "verify_grab_min_center_y_ratio"
        )
        self.verify_grab_max_center_y_ratio = self._double_param(
            "verify_grab_max_center_y_ratio"
        )
        self.verify_grab_max_depth = self._double_param("verify_grab_max_depth")
        self.verify_grab_require_lift_evidence = self._bool_param(
            "verify_grab_require_lift_evidence"
        )
        self.verify_grab_lift_pixel_delta = self._double_param(
            "verify_grab_lift_pixel_delta"
        )
        self.verify_grab_min_area_growth = self._double_param(
            "verify_grab_min_area_growth"
        )
        self.return_hold_verify_enabled = self._bool_param("return_hold_verify_enabled")
        self.return_hold_verify_loss_grace = self._double_param(
            "return_hold_verify_loss_grace_seconds"
        )
        self.return_hold_verify_log_period = self._double_param(
            "return_hold_verify_log_period_seconds"
        )
        self.enable_drop_sequence = self._bool_param("enable_drop_sequence")
        self.drop_wait_seconds = self._double_param("drop_wait_seconds")
        self.startup_arm_stow_enabled = self._bool_param("startup_arm_stow_enabled")
        self.startup_arm_stow_publish_count = self._integer_param(
            "startup_arm_stow_publish_count"
        )
        self.startup_arm_stow_publish_period = self._double_param(
            "startup_arm_stow_publish_period_seconds"
        )
        self.arm_positions_in_degrees = self._bool_param("arm_positions_in_degrees")
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
        self.startup_arm_stow_positions = self._double_array_param(
            "startup_arm_stow_positions"
        )

        self.pose = None
        self.pose_z = 0.0
        self.start_pose_z = None
        self.max_pose_z_seen = 0.0
        self.task2_ascent_start_z = None
        self.map_msg = None
        self.map_revision = 0
        self.exploration_candidates = []
        self.exploration_candidates_revision = -1
        self.visited_exploration_goals = set()
        self.exploration_scan_until_time = None
        self.latest_path = None
        self.path_index = 0
        self.current_goal = None
        self.current_goal_name = None
        self.navigation_goal_start_time = None
        self.last_goal_publish_time = None
        self.last_no_plan_log_time = None
        self.return_direct_fallback_announced = False
        self.yolo_target = None
        self.yolo_target_stamp = None
        self.yolo_bbox = None
        self.yolo_bbox_stamp = None
        self.target_surface_info = None
        self.target_surface_stamp = None
        self.segmentation_info = None
        self.segmentation_info_stamp = None
        self.segmentation_connection = None
        self.segmentation_last_seen = {"road": None, "bridge": None}
        self.last_drivable_action = None
        self.last_drivable_action_time = None
        self.last_drivable_delta_sign = 1.0
        self.last_drivable_anchor_time = None
        self.task1_observe_completed = False
        self.task1_recovery_completed = False
        self.task2_ascent_completed = False
        self.task2_descent_completed = False
        self.task2_recovery_completed = False
        self.bear_secured = False
        self.bear_context = None
        self.current_task = (
            2 if self.mission_mode == "bridge_first_shared_bear" else 1
        )
        self.next_state_after_grab = (
            MissionState.TASK2_DESCEND_BRIDGE
            if self.mission_mode == "bridge_first_shared_bear"
            else MissionState.RETURN_START
        )
        self.task2_phase_start_time = None
        self.task2_bridge_confirm_start_time = None
        self.task2_bridge_detection_start_time = None
        self.task2_search_cycle_start_time = None
        self.task2_scan_previous_yaw = None
        self.task2_scan_accumulated_yaw = 0.0
        self.task2_scan_start_pose = None
        self.task2_scan_direction = 1.0
        self.task2_last_scan_log_time = None
        self.task2_explore_segment_start_time = None
        self.task2_explore_segment_start_pose = None
        self.task2_road_reacquire_start_time = None
        self.task2_road_explore_visited = set()
        self.last_road_explore_log_time = None
        self.task2_bridge_entry_phase = None
        self.task2_bridge_entry_phase_start_time = None
        self.task2_bridge_entry_recovery_cycles = 0
        self.task2_bridge_entry_shift_direction = 1.0
        self.task2_bridge_entry_best_score = 0.0
        self.task2_bridge_last_seen_time = None
        self.task2_bridge_best_entry_score = 0.0
        self.task2_bridge_best_entry_time = None
        self.task2_bridge_best_bottom_y = 0.0
        self.task2_bridge_best_bottom_coverage = 0.0
        self.task2_bridge_orbit_cycle_count = 0
        self.task2_bridge_orbit_phase = None
        self.task2_bridge_orbit_phase_start_time = None
        self.task2_bridge_last_delta_sign = 1.0
        self.task2_bridge_corridor_last_seen_time = None
        self.bridge_landmark = self._new_bridge_landmark()
        self.bridge_edge_observations = []
        self.bridge_entry_observations = []
        self.bridge_pre_entry_observations = []
        self.last_bridge_debug_log_time = None
        self.last_foxglove_debug_publish_time = None
        self.last_bridge_marker_publish_time = None
        self.last_augmented_map_publish_time = None
        self.bridge_marker_publish_count = 0
        self.augmented_map_publish_count = 0
        self.last_augmented_map_obstacle_cell_count = 0
        self.bridge_edge_points_received = False
        self.bridge_boundary_points_received = False
        self.bridge_entry_point_received = False
        self.bridge_tf_success = False
        self.bridge_last_tf_error = ""
        self.bridge_source_frame = ""
        self.bridge_point_counts = {"left": 0, "right": 0, "center": 0, "entry": 0}
        self.bridge_left_observation_count = 0
        self.bridge_right_observation_count = 0
        self.bridge_center_observation_count = 0
        self.bridge_marker_visible_reason = "not published yet"
        self.augmented_map_visible_reason = "not published yet"
        self.fitted_bridge_side_lines = {"left": [], "right": []}
        self.last_accepted_bridge_side_lines = {"left": [], "right": []}
        self.bridge_side_line_cells = 0
        self.virtual_obstacle_cells = 0
        self.entry_gate_clear_cells = 0
        self.center_corridor_clear_cells = 0
        self.bridge_geometry_quality_reason = "not evaluated yet"
        self.bridge_geometry_quality_accepted = False
        self.last_bridge_geometry_quality_publish_time = None
        self.task2_turn_state_start_time = None
        self.task2_turn_pulse_start_time = None
        self.task2_turn_settle_start_time = None
        self.task2_turn_error_before_pulse = None
        self.task2_turn_command_action = None
        self.task2_turn_command_direction = 1.0
        self.task2_turn_wrong_way_count = 0
        self.task2_turn_centered_frames = 0
        self.task2_turn_last_observed_error = None
        self.task2_turn_sign_confirmed = False
        self.task2_entry_close_confirm_count = 0
        self.task2_entry_close_last_reason = ""
        self.task2_entry_blocked_count = 0
        self.task2_entry_blocked_start_time = None
        self.task2_entry_blocked_last_reason = ""
        self.task2_entry_recovery_phase = None
        self.task2_entry_recovery_phase_start_time = None
        self.task2_final_align_confirm_count = 0
        self.task2_final_align_loss_start_time = None
        self.task2_final_align_close_loss_start_time = None
        self.task2_ascent_stop_start_time = None
        self.task2_ascent_lost_bridge_start_time = None
        self.task2_ascent_bear_pid_integral = 0.0
        self.task2_ascent_bear_pid_last_error = None
        self.task2_ascent_bear_pid_last_time = None
        self.task2_ramp_entry_confirm_count = 0
        self.task2_side_view_recovery_start_time = None
        self.task2_side_view_recovery_direction = 1.0
        self.bridge_top_confirmed = False
        self.bridge_top_confirm_count = 0
        self.bridge_top_confidence_reason = "not evaluated"
        self.bridge_top_visual_confidence = 0.0
        self.bridge_bear_memory = BridgeBearMemory()
        self.mission_logger = MissionLogger(self)
        self.target_surface_confirm_start_time = None
        self.target_surface_last_valid_time = None
        self.target_surface_last_log_time = None
        self.observe_start_time = None
        self.observe_start_pose = None
        self.last_observed_target_time = None
        self.last_observed_target_distance = 0.0
        self.grab_approach_start_time = None
        self.grab_approach_open_sent = False
        self.grab_ready_start_time = None
        self.last_grab_target_time = None
        self.last_grab_target_distance = 0.0
        self.last_grab_target_delta_x = 0.0
        self.grab_start_time = None
        self.grab_step_index = 0
        self.grab_step_sent = False
        self.grab_step_deadline = None
        self.pre_grab_bbox = None
        self.verify_grab_start_time = None
        self.verify_grab_seen_start_time = None
        self.verify_grab_last_log_time = None
        self.grab_retry_count = 0
        self.return_hold_loss_start_time = None
        self.return_hold_last_log_time = None
        self.drop_start_time = None
        self.drop_step_index = 0
        self.drop_step_sent = False
        self.drop_step_deadline = None
        self.startup_arm_stow_publish_sent = 0
        self.last_startup_arm_stow_publish_time = None
        self.last_pose_wait_log_time = None
        self.last_task1_road_target_log_time = None
        self.last_lower_bbox_log_time = None
        self.last_grab_gate_log_time = None
        self.last_explore_wait_log_time = None
        self.last_initial_pose_publish_time = None
        self.initial_pose_publish_sent = 0
        self.motion_monitor_action = None
        self.motion_monitor_pose = None
        self.motion_monitor_start_time = None
        self.last_stuck_recovery_time = None
        self.stuck_recovery_phase = None
        self.stuck_recovery_phase_start_time = None
        self.stuck_recovery_last_action = None
        self.stuck_recovery_turn_direction = 1.0
        self.stuck_recovery_shift_direction = 1.0
        self.virtual_obstacles = []
        self.last_virtual_obstacle_log_time = None
        self.mission_start_time = self.get_clock().now()
        self.state_start_time = self.mission_start_time
        self.mission_summary_logged = False
        self.last_logged_action = None
        self.last_action_log_time = None
        self.mission_run_id = ""
        self.mission_log_path = None
        self.mission_jsonl_path = None
        self.mission_log_file = None
        self.mission_jsonl_file = None
        self._init_mission_logging()

        if self.mission_mode == "bridge_first_shared_bear":
            self.state = MissionState.TASK2_SEARCH_BRIDGE
        else:
            self.state = (
                MissionState.EXPLORE_MAP
                if self.auto_explore
                else MissionState.GO_TO_BEAR_AREA
            )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.goal_pub = self.create_publisher(PoseStamped, "/goal_pose", 10)
        self.initial_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )
        self.rear_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_rear_wheel, 10
        )
        self.front_pub = self.create_publisher(
            Float32MultiArray, DeviceDataTypeEnum.car_C_front_wheel, 10
        )
        self.arm_pub = self.create_publisher(
            JointTrajectoryPoint, DeviceDataTypeEnum.robot_arm, 10
        )
        self.target_label_pub = self.create_publisher(String, "/target_label", 10)
        self.state_pub = self.create_publisher(String, "/task1/state", 10)
        transient_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.bridge_marker_pub = self.create_publisher(
            MarkerArray,
            "/task_mission/bridge_semantic_markers",
            transient_qos,
        )
        # Foxglove note: /task_mission/augmented_map is an overlay copy, not
        # the original /map. Add both topics manually with Fixed Frame = map.
        self.augmented_map_pub = self.create_publisher(
            OccupancyGrid,
            "/task_mission/augmented_map",
            transient_qos,
        )
        self.bridge_debug_pub = self.create_publisher(
            String, "/task_mission/bridge_debug", 10
        )
        self.foxglove_debug_pub = self.create_publisher(
            String, "/task_mission/foxglove_debug", 10
        )
        self.bridge_geometry_quality_pub = self.create_publisher(
            String, "/task_mission/bridge_geometry_quality", 10
        )

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self._amcl_pose_callback,
            10,
        )
        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            OccupancyGrid, self.map_topic, self._map_callback, map_qos
        )
        self.create_subscription(Path, "/received_global_plan", self._path_callback, 10)
        self.create_subscription(
            Float32MultiArray,
            "/yolo/target_info",
            self._target_info_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/yolo/target_bbox",
            self._target_bbox_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/yolo/target_surface_info",
            self._target_surface_info_callback,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            "/yolo/segmentation_info",
            self._segmentation_info_callback,
            10,
        )
        self.create_subscription(
            PointCloud2,
            "/yolo/bridge_edge_points",
            self._bridge_edge_points_callback,
            10,
        )
        self.create_subscription(
            PointCloud2,
            "/yolo/bridge_boundary_points",
            self._bridge_boundary_points_callback,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/yolo/bridge_entry_point",
            self._bridge_entry_point_callback,
            10,
        )
        self.create_subscription(
            PointStamped,
            "/yolo/bridge_pre_entry_point",
            self._bridge_pre_entry_point_callback,
            10,
        )

        control_period = self._double_param("control_period_seconds")
        self.timer = self.create_timer(control_period, self._control_loop)

        self._publish_target_label()
        self._publish_state()
        self.get_logger().info(
            "Foxglove setup: fixed frame=map, add /map, "
            "/task_mission/augmented_map, /task_mission/bridge_semantic_markers, "
            "/yolo/bridge_edge_points, /yolo/bridge_entry_point."
        )
        self.get_logger().info(
            "Foxglove instructions: 1. Fixed Frame = map 2. Add /map as "
            "OccupancyGrid 3. Add /task_mission/augmented_map as OccupancyGrid "
            "overlay 4. Add /task_mission/bridge_semantic_markers as MarkerArray "
            "5. Add /yolo/bridge_edge_points as PointCloud2 6. Add "
            "/yolo/bridge_entry_point as PointStamped 7. Add /task_mission/"
            "bridge_debug and /task_mission/foxglove_debug as Raw Messages."
        )
        self.get_logger().info(
            'Debug commands: ros2 topic list | grep -E "bridge|augmented|foxglove|geometry"'
        )
        self.get_logger().info(
            "Debug commands: ros2 topic echo /yolo/bridge_geometry_debug ; "
            "ros2 topic echo /task_mission/foxglove_debug"
        )
        self._log_event("info", "startup")
        if self.mission_mode == "bridge_first_shared_bear":
            self.get_logger().info(
                "Task mission controller ready in bridge_first_shared_bear mode."
            )
        else:
            self.get_logger().info(
                "Task mission controller ready in legacy Task 1 then Task 2 mode. "
                "It will explore the map until a bear is found."
                if self.auto_explore
                else "Task mission controller ready. Set start_pose and "
                "bear_search_pose before the final run."
            )

    def _double_param(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def _integer_param(self, name):
        return self.get_parameter(name).get_parameter_value().integer_value

    def _bool_param(self, name):
        return self.get_parameter(name).get_parameter_value().bool_value

    def _string_param(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _double_array_param(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return [float(item) for item in value]

    def _pose_param(self, name):
        values = self._double_array_param(name)
        if len(values) != 3:
            raise ValueError(f"{name} must be [x, y, yaw_rad]")
        return values

    def _init_mission_logging(self):
        if not self.mission_log_to_file:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.mission_run_id = f"{timestamp}_{uuid.uuid4().hex[:6]}"
        try:
            os.makedirs(self.mission_log_directory, exist_ok=True)
            if self.mission_log_text:
                self.mission_log_path = os.path.join(
                    self.mission_log_directory,
                    f"mission_{self.mission_run_id}.log",
                )
                self.mission_log_file = open(self.mission_log_path, "a", encoding="utf-8")
            if self.mission_log_jsonl:
                self.mission_jsonl_path = os.path.join(
                    self.mission_log_directory,
                    f"mission_{self.mission_run_id}.jsonl",
                )
                self.mission_jsonl_file = open(
                    self.mission_jsonl_path, "a", encoding="utf-8"
                )
        except OSError as exc:
            self.mission_log_to_file = False
            self.get_logger().warn(f"Mission file logging disabled: {exc}")

    def _log_event(self, level, event, **fields):
        record = self._mission_log_record(event, **fields)
        level = (level or "info").lower()
        message = self._format_log_record(record)
        logger = self.get_logger()
        if level == "warn" or level == "warning":
            logger.warn(message)
        elif level == "error":
            logger.error(message)
        elif level == "debug":
            logger.debug(message)
        else:
            logger.info(message)

        if not self.mission_log_to_file:
            return
        try:
            if self.mission_log_file is not None:
                self.mission_log_file.write(message + "\n")
                if self.mission_log_flush_every_event:
                    self.mission_log_file.flush()
            if self.mission_jsonl_file is not None:
                self.mission_jsonl_file.write(
                    json.dumps(self._json_safe(record), sort_keys=True) + "\n"
                )
                if self.mission_log_flush_every_event:
                    self.mission_jsonl_file.flush()
        except OSError as exc:
            self.mission_log_to_file = False
            logger.warn(f"Mission file logging failed and was disabled: {exc}")

    def _mission_log_record(self, event, **fields):
        now = self.get_clock().now()
        pose = self.pose if getattr(self, "pose", None) is not None else (None, None, None)
        bridge = getattr(self, "bridge_landmark", {})
        target = getattr(self, "yolo_target", None) or {}
        target_surface = getattr(self, "target_surface_info", None) or {}
        connection = self._bridge_connection() if hasattr(self, "segmentation_connection") else None
        record = {
            "timestamp": now.nanoseconds / 1e9,
            "run_id": getattr(self, "mission_run_id", ""),
            "event": event,
            "state": getattr(getattr(self, "state", None), "value", str(getattr(self, "state", ""))),
            "action": fields.get("action"),
            "current_task": getattr(self, "current_task", None),
            "pose_x": pose[0],
            "pose_y": pose[1],
            "pose_yaw": pose[2],
            "bridge_raw": self._current_bridge_raw_visible(),
            "bridge_fresh_age": self._bridge_fresh_age() if hasattr(self, "bridge_landmark") else None,
            "bridge_cached": bridge.get("source_is_cached"),
            "bridge_entry_pixel_x": bridge.get("entry_pixel_x"),
            "bridge_entry_pixel_y": bridge.get("entry_pixel_y"),
            "bridge_entry_depth": bridge.get("entry_depth"),
            "bridge_entry_map_x": bridge.get("entry_map_x"),
            "bridge_entry_map_y": bridge.get("entry_map_y"),
            "road_bridge_connection_score": (
                None if connection is None else connection.get("score")
            ),
            "bridge_frontalness": self._current_bridge_value("frontalness"),
            "bridge_corridor_width": self._current_bridge_corridor_width(),
            "target_visible": bool(target.get("found", False)),
            "target_distance": target.get("distance"),
            "target_on_bridge": bool(
                target_surface.get("target_bottom_center_on_bridge", False)
                or target_surface.get("target_center_on_bridge", False)
                or target_surface.get("target_side_bridge_contact", False)
            ),
            "virtual_obstacle_count": len(getattr(self, "virtual_obstacles", [])),
            "augmented_map_obstacle_cell_count": getattr(
                self, "last_augmented_map_obstacle_cell_count", 0
            ),
            "state_transition_reason": fields.get("reason", ""),
        }
        record.update(fields)
        return record

    def _format_log_record(self, record):
        details = " ".join(
            f"{key}={value}"
            for key, value in record.items()
            if key not in ("timestamp", "run_id") and value is not None
        )
        return f"[mission {record.get('run_id', '')}] {details}"

    def _json_safe(self, value):
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._json_safe(v) for v in value]
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "nanoseconds"):
            return value.nanoseconds / 1e9
        return value

    def _log_state_transition(self, old_state, new_state, reason=""):
        self._log_event(
            "info",
            "state_transition",
            old_state=getattr(old_state, "value", str(old_state)),
            new_state=getattr(new_state, "value", str(new_state)),
            reason=reason,
        )

    def _log_action(self, action_key, reason="", **fields):
        self._log_event("info", "action", action=action_key, reason=reason, **fields)

    def _log_bridge_debug(self, **fields):
        self._log_event("info", "bridge_debug", **fields)

    def _log_mission_summary(self):
        if self.mission_summary_logged:
            return
        self.mission_summary_logged = True
        self._log_event(
            "info",
            "mission_summary",
            task1_observe_completed=self.task1_observe_completed,
            task1_recovery_completed=self.task1_recovery_completed,
            task2_ascent_completed=self.task2_ascent_completed,
            task2_descent_completed=self.task2_descent_completed,
            task2_recovery_completed=self.task2_recovery_completed,
            bear_secured=self.bear_secured,
            elapsed_seconds=self._elapsed_seconds(self.mission_start_time),
            log_path=self.mission_log_path,
            jsonl_path=self.mission_jsonl_path,
        )
        for handle_name in ("mission_log_file", "mission_jsonl_file"):
            handle = getattr(self, handle_name, None)
            if handle is not None:
                try:
                    handle.flush()
                    handle.close()
                except OSError:
                    pass
                setattr(self, handle_name, None)

    def _current_bridge_raw_visible(self):
        bridge = None
        if getattr(self, "segmentation_info", None) is not None:
            bridge = self.segmentation_info.get("bridge")
        return bool(bridge and bridge.get("raw_found", bridge.get("found", False)))

    def _current_bridge_value(self, key):
        if getattr(self, "segmentation_info", None) is None:
            return None
        bridge = self.segmentation_info.get("bridge")
        if bridge is None:
            return None
        return bridge.get(key)

    def _current_bridge_corridor_width(self):
        if getattr(self, "segmentation_info", None) is None:
            return None
        bridge = self.segmentation_info.get("bridge")
        if bridge is None:
            return None
        image_width = float(self.segmentation_info.get("image_width", 0.0))
        if image_width <= 0.0:
            return None
        width_ratio = float(bridge.get("bottom_width_ratio", 0.0))
        return width_ratio * image_width

    def _amcl_pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self._set_current_pose(
            (position.x, position.y, yaw_from_quaternion(orientation)),
            source="/amcl_pose",
            z=float(position.z),
        )

    def _set_current_pose(self, pose, source, z=None):
        self.pose = pose
        if z is not None:
            self.pose_z = float(z)
            self.max_pose_z_seen = max(self.max_pose_z_seen, self.pose_z)
        if not self.start_pose_locked:
            self.start_pose = [self.pose[0], self.pose[1], self.pose[2]]
            self.start_pose_z = self.pose_z
            self.start_pose_locked = True
            self.get_logger().info(
                f"Captured current {source} as start_pose: "
                f"x={self.start_pose[0]:.2f}, y={self.start_pose[1]:.2f}, "
                f"yaw={self.start_pose[2]:.2f}"
            )

    def _update_pose_from_tf(self):
        if not self.use_tf_pose_fallback:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.base_frame,
                rclpy.time.Time(),
            )
        except Exception:
            return

        translation = transform.transform.translation
        rotation = transform.transform.rotation
        self._set_current_pose(
            (translation.x, translation.y, yaw_from_quaternion(rotation)),
            source=f"TF {self.map_frame}->{self.base_frame}",
            z=float(translation.z),
        )

    def _map_callback(self, msg):
        self.map_msg = msg
        self.map_revision += 1

    def _path_callback(self, msg):
        if not msg.poses:
            return
        if self.current_goal is None:
            self._set_latest_path(msg)
            return

        end = msg.poses[-1].pose.position
        if distance_2d((end.x, end.y), self.current_goal[:2]) <= self.path_goal_tolerance:
            self._set_latest_path(msg)

    def _set_latest_path(self, msg):
        self.latest_path = msg
        self.path_index = 0

    def _target_info_callback(self, msg):
        if len(msg.data) < 3:
            return
        self.yolo_target = {
            "found": msg.data[0] >= 0.5,
            "distance": msg.data[1],
            "delta_x": msg.data[2],
        }
        self.yolo_target_stamp = self.get_clock().now()

    def _target_bbox_callback(self, msg):
        if len(msg.data) < 13:
            return

        width = msg.data[3]
        height = msg.data[4]
        image_width = msg.data[9]
        image_height = msg.data[10]
        area = max(0.0, width) * max(0.0, height)
        image_area = max(1.0, image_width * image_height)

        self.yolo_bbox = {
            "found": msg.data[0] >= 0.5,
            "center_x": msg.data[1],
            "center_y": msg.data[2],
            "width": width,
            "height": height,
            "x1": msg.data[5],
            "y1": msg.data[6],
            "x2": msg.data[7],
            "y2": msg.data[8],
            "image_width": image_width,
            "image_height": image_height,
            "confidence": msg.data[11],
            "distance": msg.data[12],
            "area": area,
            "area_ratio": area / image_area,
        }
        self.yolo_bbox_stamp = self.get_clock().now()

    def _target_surface_info_callback(self, msg):
        if len(msg.data) < 8:
            return
        self.target_surface_info = {
            "target_found": msg.data[0] >= 0.5,
            "bridge_found": msg.data[1] >= 0.5,
            "bbox_bridge_overlap_ratio": msg.data[2],
            "bbox_lower_half_bridge_overlap_ratio": msg.data[3],
            "target_center_on_bridge": msg.data[4] >= 0.5,
            "target_bottom_center_on_bridge": msg.data[5] >= 0.5,
            "image_width": msg.data[6],
            "image_height": msg.data[7],
            "target_left_side_bridge_contact": (
                len(msg.data) >= 13 and msg.data[8] >= 0.5
            ),
            "target_right_side_bridge_contact": (
                len(msg.data) >= 13 and msg.data[9] >= 0.5
            ),
            "target_side_bridge_contact": (
                len(msg.data) >= 13 and msg.data[10] >= 0.5
            ),
            "target_side_bridge_contact_ratio": (
                msg.data[11] if len(msg.data) >= 13 else 0.0
            ),
            "target_side_bridge_contact_pixels": (
                msg.data[12] if len(msg.data) >= 13 else 0.0
            ),
        }
        self.target_surface_stamp = self.get_clock().now()

    def _segmentation_info_callback(self, msg):
        if len(msg.data) < 10:
            return
        now = self.get_clock().now()
        road = self._filtered_segment_update(
            "road",
            {
                "found": msg.data[0] >= 0.5,
                "delta_x": msg.data[1],
                "area_ratio": msg.data[2],
                "bottom_coverage": msg.data[3],
                "center_x": msg.data[10] if len(msg.data) >= 22 else 0.0,
                "center_y_ratio": msg.data[11] if len(msg.data) >= 22 else 0.0,
                "top_y_ratio": msg.data[12] if len(msg.data) >= 22 else 0.0,
                "bottom_y_ratio": msg.data[13] if len(msg.data) >= 22 else 0.0,
                "bottom_center_x": msg.data[22] if len(msg.data) >= 28 else 0.0,
                "top_center_x": msg.data[23] if len(msg.data) >= 28 else 0.0,
                "bottom_left_x": msg.data[28] if len(msg.data) >= 38 else 0.0,
                "bottom_right_x": msg.data[29] if len(msg.data) >= 38 else 0.0,
                "mid_left_x": msg.data[30] if len(msg.data) >= 38 else 0.0,
                "mid_right_x": msg.data[31] if len(msg.data) >= 38 else 0.0,
                "bottom_width_ratio": msg.data[32] if len(msg.data) >= 38 else 0.0,
            },
            now,
        )
        bridge = self._filtered_segment_update(
            "bridge",
            {
                "found": msg.data[4] >= 0.5,
                "delta_x": msg.data[5],
                "area_ratio": msg.data[6],
                "bottom_coverage": msg.data[7],
                "center_x": msg.data[14] if len(msg.data) >= 22 else 0.0,
                "center_y_ratio": msg.data[15] if len(msg.data) >= 22 else 0.0,
                "top_y_ratio": msg.data[16] if len(msg.data) >= 22 else 0.0,
                "bottom_y_ratio": msg.data[17] if len(msg.data) >= 22 else 0.0,
                "bottom_center_x": msg.data[24] if len(msg.data) >= 28 else 0.0,
                "mid_center_x": msg.data[25] if len(msg.data) >= 28 else 0.0,
                "bottom_left_x": msg.data[33] if len(msg.data) >= 38 else 0.0,
                "bottom_right_x": msg.data[34] if len(msg.data) >= 38 else 0.0,
                "mid_left_x": msg.data[35] if len(msg.data) >= 38 else 0.0,
                "mid_right_x": msg.data[36] if len(msg.data) >= 38 else 0.0,
                "bottom_width_ratio": msg.data[37] if len(msg.data) >= 38 else 0.0,
                "entry_u": msg.data[38] if len(msg.data) >= 45 else 0.0,
                "entry_v": msg.data[39] if len(msg.data) >= 45 else 0.0,
                "entry_confidence": msg.data[40] if len(msg.data) >= 45 else 0.0,
                "entry_from_road_connection": msg.data[41] if len(msg.data) >= 45 else 0.0,
                "centerline_slope_pixels": msg.data[42] if len(msg.data) >= 45 else 0.0,
                "frontalness": msg.data[43] if len(msg.data) >= 45 else 0.0,
                "entry_depth": msg.data[44] if len(msg.data) >= 45 else 0.0,
                "target_u": msg.data[45] if len(msg.data) >= 54 else 0.0,
                "target_v": msg.data[46] if len(msg.data) >= 54 else 0.0,
                "target_confidence": msg.data[47] if len(msg.data) >= 54 else 0.0,
                "entry_confirmed": msg.data[48] if len(msg.data) >= 54 else 0.0,
                "entry_gate_left_u": msg.data[49] if len(msg.data) >= 54 else 0.0,
                "entry_gate_right_u": msg.data[50] if len(msg.data) >= 54 else 0.0,
                "entry_gate_center_u": msg.data[51] if len(msg.data) >= 54 else 0.0,
                "entry_gate_v": msg.data[52] if len(msg.data) >= 54 else 0.0,
                "entry_gate_width_pixels": msg.data[53] if len(msg.data) >= 54 else 0.0,
                "pre_entry_u": msg.data[54] if len(msg.data) >= 59 else 0.0,
                "pre_entry_v": msg.data[55] if len(msg.data) >= 59 else 0.0,
                "pre_entry_confidence": msg.data[56] if len(msg.data) >= 59 else 0.0,
                "pre_entry_depth": msg.data[57] if len(msg.data) >= 59 else 0.0,
                "entry_gate_confirmed": msg.data[58] if len(msg.data) >= 59 else 0.0,
                "ramp_valid": msg.data[59] if len(msg.data) >= 66 else 0.0,
                "ramp_confidence": msg.data[60] if len(msg.data) >= 66 else 0.0,
                "ramp_lower_present": msg.data[61] if len(msg.data) >= 66 else 0.0,
                "ramp_continuous": msg.data[62] if len(msg.data) >= 66 else 0.0,
                "side_view_score": msg.data[63] if len(msg.data) >= 66 else 1.0,
                "vertical_coverage_score": msg.data[64] if len(msg.data) >= 66 else 0.0,
                "ramp_reason_code": msg.data[65] if len(msg.data) >= 66 else 0.0,
            },
            now,
        )
        self.segmentation_info = {
            "road": road,
            "bridge": bridge,
            "image_width": msg.data[8],
            "image_height": msg.data[9],
        }
        if len(msg.data) >= 22:
            self.segmentation_connection = {
                "connected": msg.data[18] >= 0.5,
                "score": msg.data[19],
                "delta_x": msg.data[20],
                "gap_y_ratio": msg.data[21],
                "central_score": msg.data[26] if len(msg.data) >= 28 else msg.data[19],
                "central_delta_x": msg.data[27] if len(msg.data) >= 28 else msg.data[20],
            }
        else:
            self.segmentation_connection = None
        self.segmentation_info_stamp = now

    def _bridge_entry_point_callback(self, msg):
        self.bridge_entry_point_received = True
        self.bridge_source_frame = msg.header.frame_id
        self.bridge_point_counts["entry"] += 1
        if not self.bridge_map_projection_enabled:
            return
        point = self._transform_point_to_map(msg.point, msg.header.frame_id, msg.header.stamp)
        if point is None:
            return
        self.bridge_entry_observations.append(point)
        self.bridge_entry_observations = self.bridge_entry_observations[-12:]
        x = sum(p[0] for p in self.bridge_entry_observations) / len(
            self.bridge_entry_observations
        )
        y = sum(p[1] for p in self.bridge_entry_observations) / len(
            self.bridge_entry_observations
        )
        self.bridge_landmark["entry_map_x"] = x
        self.bridge_landmark["entry_map_y"] = y
        self.bridge_landmark["valid"] = True
        self.bridge_landmark["last_seen_time"] = self.get_clock().now()
        self.bridge_landmark["confidence"] = min(
            1.0, max(self.bridge_landmark.get("confidence", 0.0), 0.35)
        )

    def _bridge_pre_entry_point_callback(self, msg):
        self.bridge_source_frame = msg.header.frame_id
        if not self.bridge_map_projection_enabled:
            return
        point = self._transform_point_to_map(msg.point, msg.header.frame_id, msg.header.stamp)
        if point is None:
            return
        self.bridge_pre_entry_observations.append(point)
        self.bridge_pre_entry_observations = self.bridge_pre_entry_observations[-12:]
        x = sum(p[0] for p in self.bridge_pre_entry_observations) / len(
            self.bridge_pre_entry_observations
        )
        y = sum(p[1] for p in self.bridge_pre_entry_observations) / len(
            self.bridge_pre_entry_observations
        )
        self.bridge_landmark["pre_entry_map_x"] = x
        self.bridge_landmark["pre_entry_map_y"] = y
        self.bridge_landmark["valid"] = True
        self.bridge_landmark["last_seen_time"] = self.get_clock().now()
        self.bridge_landmark["confidence"] = min(
            1.0, max(self.bridge_landmark.get("confidence", 0.0), 0.40)
        )

    def _bridge_edge_points_callback(self, msg):
        self.bridge_edge_points_received = True
        self.bridge_source_frame = msg.header.frame_id
        if (
            not self.bridge_map_projection_enabled
            or point_cloud2 is None
            or not msg.header.frame_id
        ):
            return
        left = []
        right = []
        center = []
        try:
            points = point_cloud2.read_points(
                msg, field_names=("x", "y", "z", "side"), skip_nans=True
            )
            for x, y, z, side in points:
                side_id = int(round(float(side)))
                if side_id == 0:
                    self.bridge_point_counts["left"] += 1
                elif side_id == 1:
                    self.bridge_point_counts["right"] += 1
                else:
                    self.bridge_point_counts["center"] += 1
                transformed = self._transform_xyz_to_map(
                    (float(x), float(y), float(z)), msg.header.frame_id, msg.header.stamp
                )
                if transformed is None:
                    continue
                if side_id == 0:
                    left.append(transformed)
                elif side_id == 1:
                    right.append(transformed)
                else:
                    center.append(transformed)
        except Exception as exc:
            self.get_logger().warn(f"Could not read bridge edge point cloud: {exc}")
            return

        if not left and not right and not center:
            return
        self._update_bridge_side_landmark(left, right, center)

    def _bridge_boundary_points_callback(self, msg):
        self.bridge_boundary_points_received = True
        self.bridge_source_frame = msg.header.frame_id
        if (
            not self.bridge_map_projection_enabled
            or point_cloud2 is None
            or not msg.header.frame_id
        ):
            return
        left = []
        right = []
        center = []
        entry_points = []
        try:
            points = point_cloud2.read_points(
                msg, field_names=("x", "y", "z", "side", "role"), skip_nans=True
            )
            for x, y, z, _side, role in points:
                role_id = int(round(float(role)))
                transformed = self._transform_xyz_to_map(
                    (float(x), float(y), float(z)), msg.header.frame_id, msg.header.stamp
                )
                if role_id == 0:
                    self.bridge_point_counts["left"] += 1
                    if transformed is not None:
                        left.append(transformed)
                elif role_id == 1:
                    self.bridge_point_counts["right"] += 1
                    if transformed is not None:
                        right.append(transformed)
                elif role_id == 2:
                    self.bridge_point_counts["entry"] += 1
                    if transformed is not None:
                        entry_points.append(transformed)
                elif role_id == 3:
                    self.bridge_point_counts["center"] += 1
                    if transformed is not None:
                        center.append(transformed)
        except Exception as exc:
            self.get_logger().warn(f"Could not read bridge boundary point cloud: {exc}")
            self.bridge_last_tf_error = str(exc)
            return

        for point in entry_points:
            self.bridge_entry_observations.append(point)
        self.bridge_entry_observations = self.bridge_entry_observations[-12:]
        if entry_points:
            x = sum(p[0] for p in self.bridge_entry_observations) / len(
                self.bridge_entry_observations
            )
            y = sum(p[1] for p in self.bridge_entry_observations) / len(
                self.bridge_entry_observations
            )
            self.bridge_landmark["entry_map_x"] = x
            self.bridge_landmark["entry_map_y"] = y
        if left or right or center:
            self._update_bridge_side_landmark(left, right, center)

    def _transform_point_to_map(self, point, source_frame, stamp):
        return self._transform_xyz_to_map(
            (float(point.x), float(point.y), float(point.z)), source_frame, stamp
        )

    def _transform_xyz_to_map(self, xyz, source_frame, stamp):
        if not source_frame:
            self.bridge_tf_success = False
            self.bridge_last_tf_error = "missing source frame"
            return None
        if self._ros_stamp_age(stamp) > self.bridge_map_max_tf_age:
            self.bridge_tf_success = False
            self.bridge_last_tf_error = "bridge point stamp too old"
            return None
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                source_frame,
                rclpy.time.Time.from_msg(stamp),
                timeout=Duration(nanoseconds=int(self.bridge_map_tf_timeout * 1e9)),
            )
        except Exception:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.map_frame,
                    source_frame,
                    rclpy.time.Time(),
                    timeout=Duration(nanoseconds=int(self.bridge_map_tf_timeout * 1e9)),
                )
            except Exception as exc:
                self.bridge_tf_success = False
                self.bridge_last_tf_error = str(exc)
                return None
        self.bridge_tf_success = True
        self.bridge_last_tf_error = ""
        return self._apply_transform_xyz(xyz, transform)

    def _ros_stamp_age(self, stamp):
        if stamp is None:
            return 999.0
        now = self.get_clock().now().nanoseconds / 1e9
        msg_time = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return abs(now - msg_time)

    def _apply_transform_xyz(self, xyz, transform):
        q = transform.transform.rotation
        tx = transform.transform.translation.x
        ty = transform.transform.translation.y
        tz = transform.transform.translation.z
        x, y, z = xyz
        # Quaternion rotation, expanded to avoid adding another runtime dependency.
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        ix = qw * x + qy * z - qz * y
        iy = qw * y + qz * x - qx * z
        iz = qw * z + qx * y - qy * x
        iw = -qx * x - qy * y - qz * z
        rx = ix * qw + iw * -qx + iy * -qz - iz * -qy
        ry = iy * qw + iw * -qy + iz * -qx - ix * -qz
        rz = iz * qw + iw * -qz + ix * -qy - iy * -qx
        return (rx + tx, ry + ty, rz + tz)

    def _update_bridge_side_landmark(self, left, right, center):
        landmark = self.bridge_landmark
        if left:
            self.bridge_left_observation_count += 1
            landmark["left_side_map_points"] = self._merge_points(
                landmark["left_side_map_points"], left
            )
        if right:
            self.bridge_right_observation_count += 1
            landmark["right_side_map_points"] = self._merge_points(
                landmark["right_side_map_points"], right
            )
        if center:
            self.bridge_center_observation_count += 1
            landmark["centerline_map_points"] = self._merge_points(
                landmark["centerline_map_points"], center
            )
        landmark["left_observation_count"] = self.bridge_left_observation_count
        landmark["right_observation_count"] = self.bridge_right_observation_count
        landmark["center_observation_count"] = self.bridge_center_observation_count
        landmark["valid"] = True
        landmark["last_seen_time"] = self.get_clock().now()
        landmark["confidence"] = min(1.0, max(landmark.get("confidence", 0.0), 0.45))
        self._update_bridge_forward_axis()

        if landmark["left_side_map_points"] and landmark["right_side_map_points"]:
            widths = []
            for lp in landmark["left_side_map_points"]:
                nearest = min(
                    distance_2d(lp[:2], rp[:2])
                    for rp in landmark["right_side_map_points"]
                )
                widths.append(nearest)
            if widths:
                median_width = sorted(widths)[len(widths) // 2]
                if (
                    self.bridge_landmark_min_width
                    <= median_width
                    <= self.bridge_landmark_max_width
                ):
                    landmark["confidence"] = min(1.0, landmark["confidence"] + 0.2)

    def _update_bridge_forward_axis(self):
        entry = self._bridge_entry_xy()
        if entry is None:
            return
        center_points = self.bridge_landmark.get("centerline_map_points", [])
        if not center_points:
            return
        farthest = max(center_points, key=lambda p: distance_2d(entry, p[:2]))
        dx = float(farthest[0]) - entry[0]
        dy = float(farthest[1]) - entry[1]
        length = math.hypot(dx, dy)
        if length < 0.10:
            return
        self.bridge_landmark["forward_axis"] = (dx / length, dy / length)
        self.bridge_landmark["estimated_bridge_heading"] = math.atan2(dy, dx)

    def _merge_points(self, existing, new_points, max_count=40):
        merged = list(existing)
        for point in new_points:
            if any(distance_2d(point[:2], old[:2]) < 0.08 for old in merged):
                continue
            merged.append(point)
        return merged[-max_count:]

    def _filtered_segment_update(self, label, segment, now):
        prev = None if self.segmentation_info is None else self.segmentation_info.get(label)
        alpha = min(1.0, max(0.0, self.segmentation_smoothing_alpha))
        found = bool(segment["found"])

        if found:
            self.segmentation_last_seen[label] = now
        elif prev is not None:
            retained = dict(prev)
            retained["raw_found"] = False
            retained["found"] = False
            retained["usable"] = False
            retained["predicted_or_cached"] = True
            retained["last_update_time"] = now
            return retained

        if prev is None:
            return {
                "found": found,
                "raw_found": found,
                "usable": found,
                "predicted_or_cached": False,
                "last_valid_time": now if found else None,
                "last_update_time": now,
                "delta_x": float(segment["delta_x"]),
                "area_ratio": float(segment["area_ratio"]),
                "bottom_coverage": float(segment["bottom_coverage"]),
                "center_x": float(segment.get("center_x", 0.0)),
                "center_y_ratio": float(segment.get("center_y_ratio", 0.0)),
                "top_y_ratio": float(segment.get("top_y_ratio", 0.0)),
                "bottom_y_ratio": float(segment.get("bottom_y_ratio", 0.0)),
                "bottom_center_x": float(segment.get("bottom_center_x", 0.0)),
                "top_center_x": float(segment.get("top_center_x", 0.0)),
                "mid_center_x": float(segment.get("mid_center_x", 0.0)),
                "bottom_left_x": float(segment.get("bottom_left_x", 0.0)),
                "bottom_right_x": float(segment.get("bottom_right_x", 0.0)),
                "mid_left_x": float(segment.get("mid_left_x", 0.0)),
                "mid_right_x": float(segment.get("mid_right_x", 0.0)),
                "bottom_width_ratio": float(segment.get("bottom_width_ratio", 0.0)),
                "entry_u": float(segment.get("entry_u", 0.0)),
                "entry_v": float(segment.get("entry_v", 0.0)),
                "entry_confidence": float(segment.get("entry_confidence", 0.0)),
                "entry_from_road_connection": float(
                    segment.get("entry_from_road_connection", 0.0)
                ),
                "centerline_slope_pixels": float(
                    segment.get("centerline_slope_pixels", 0.0)
                ),
                "frontalness": float(segment.get("frontalness", 0.0)),
                "entry_depth": float(segment.get("entry_depth", 0.0)),
                "target_u": float(segment.get("target_u", 0.0)),
                "target_v": float(segment.get("target_v", 0.0)),
                "target_confidence": float(segment.get("target_confidence", 0.0)),
                "entry_confirmed": float(segment.get("entry_confirmed", 0.0)),
                "entry_gate_left_u": float(segment.get("entry_gate_left_u", 0.0)),
                "entry_gate_right_u": float(segment.get("entry_gate_right_u", 0.0)),
                "entry_gate_center_u": float(segment.get("entry_gate_center_u", 0.0)),
                "entry_gate_v": float(segment.get("entry_gate_v", 0.0)),
                "entry_gate_width_pixels": float(segment.get("entry_gate_width_pixels", 0.0)),
                "pre_entry_u": float(segment.get("pre_entry_u", 0.0)),
                "pre_entry_v": float(segment.get("pre_entry_v", 0.0)),
                "pre_entry_confidence": float(segment.get("pre_entry_confidence", 0.0)),
                "pre_entry_depth": float(segment.get("pre_entry_depth", 0.0)),
                "entry_gate_confirmed": float(segment.get("entry_gate_confirmed", 0.0)),
                "ramp_valid": float(segment.get("ramp_valid", 0.0)),
                "ramp_confidence": float(segment.get("ramp_confidence", 0.0)),
                "ramp_lower_present": float(segment.get("ramp_lower_present", 0.0)),
                "ramp_continuous": float(segment.get("ramp_continuous", 0.0)),
                "side_view_score": float(segment.get("side_view_score", 1.0)),
                "vertical_coverage_score": float(segment.get("vertical_coverage_score", 0.0)),
                "ramp_reason_code": float(segment.get("ramp_reason_code", 0.0)),
            }

        # Keep filtered values continuous to avoid steering jitter from segmentation flicker.
        delta_x = (1.0 - alpha) * float(prev["delta_x"]) + alpha * float(segment["delta_x"])
        area_ratio = (1.0 - alpha) * float(prev["area_ratio"]) + alpha * float(segment["area_ratio"])
        bottom_coverage = (1.0 - alpha) * float(prev["bottom_coverage"]) + alpha * float(
            segment["bottom_coverage"]
        )
        center_x = (1.0 - alpha) * float(prev.get("center_x", 0.0)) + alpha * float(
            segment.get("center_x", 0.0)
        )
        center_y_ratio = (1.0 - alpha) * float(
            prev.get("center_y_ratio", 0.0)
        ) + alpha * float(segment.get("center_y_ratio", 0.0))
        top_y_ratio = (1.0 - alpha) * float(prev.get("top_y_ratio", 0.0)) + alpha * float(
            segment.get("top_y_ratio", 0.0)
        )
        bottom_y_ratio = (1.0 - alpha) * float(
            prev.get("bottom_y_ratio", 0.0)
        ) + alpha * float(segment.get("bottom_y_ratio", 0.0))
        bottom_center_x = (1.0 - alpha) * float(
            prev.get("bottom_center_x", 0.0)
        ) + alpha * float(segment.get("bottom_center_x", 0.0))
        top_center_x = (1.0 - alpha) * float(prev.get("top_center_x", 0.0)) + alpha * float(
            segment.get("top_center_x", 0.0)
        )
        mid_center_x = (1.0 - alpha) * float(prev.get("mid_center_x", 0.0)) + alpha * float(
            segment.get("mid_center_x", 0.0)
        )
        bottom_left_x = (1.0 - alpha) * float(
            prev.get("bottom_left_x", 0.0)
        ) + alpha * float(segment.get("bottom_left_x", 0.0))
        bottom_right_x = (1.0 - alpha) * float(
            prev.get("bottom_right_x", 0.0)
        ) + alpha * float(segment.get("bottom_right_x", 0.0))
        mid_left_x = (1.0 - alpha) * float(prev.get("mid_left_x", 0.0)) + alpha * float(
            segment.get("mid_left_x", 0.0)
        )
        mid_right_x = (1.0 - alpha) * float(prev.get("mid_right_x", 0.0)) + alpha * float(
            segment.get("mid_right_x", 0.0)
        )
        bottom_width_ratio = (1.0 - alpha) * float(
            prev.get("bottom_width_ratio", 0.0)
        ) + alpha * float(segment.get("bottom_width_ratio", 0.0))
        entry_u = (1.0 - alpha) * float(prev.get("entry_u", 0.0)) + alpha * float(
            segment.get("entry_u", 0.0)
        )
        entry_v = (1.0 - alpha) * float(prev.get("entry_v", 0.0)) + alpha * float(
            segment.get("entry_v", 0.0)
        )
        entry_confidence = (1.0 - alpha) * float(
            prev.get("entry_confidence", 0.0)
        ) + alpha * float(segment.get("entry_confidence", 0.0))
        entry_from_road_connection = (1.0 - alpha) * float(
            prev.get("entry_from_road_connection", 0.0)
        ) + alpha * float(segment.get("entry_from_road_connection", 0.0))
        centerline_slope_pixels = (1.0 - alpha) * float(
            prev.get("centerline_slope_pixels", 0.0)
        ) + alpha * float(segment.get("centerline_slope_pixels", 0.0))
        frontalness = (1.0 - alpha) * float(
            prev.get("frontalness", 0.0)
        ) + alpha * float(segment.get("frontalness", 0.0))
        entry_depth = (1.0 - alpha) * float(
            prev.get("entry_depth", 0.0)
        ) + alpha * float(segment.get("entry_depth", 0.0))
        target_u = (1.0 - alpha) * float(prev.get("target_u", 0.0)) + alpha * float(
            segment.get("target_u", 0.0)
        )
        target_v = (1.0 - alpha) * float(prev.get("target_v", 0.0)) + alpha * float(
            segment.get("target_v", 0.0)
        )
        target_confidence = (1.0 - alpha) * float(
            prev.get("target_confidence", 0.0)
        ) + alpha * float(segment.get("target_confidence", 0.0))
        entry_gate_left_u = (1.0 - alpha) * float(
            prev.get("entry_gate_left_u", 0.0)
        ) + alpha * float(segment.get("entry_gate_left_u", 0.0))
        entry_gate_right_u = (1.0 - alpha) * float(
            prev.get("entry_gate_right_u", 0.0)
        ) + alpha * float(segment.get("entry_gate_right_u", 0.0))
        entry_gate_center_u = (1.0 - alpha) * float(
            prev.get("entry_gate_center_u", 0.0)
        ) + alpha * float(segment.get("entry_gate_center_u", 0.0))
        entry_gate_v = (1.0 - alpha) * float(
            prev.get("entry_gate_v", 0.0)
        ) + alpha * float(segment.get("entry_gate_v", 0.0))
        entry_gate_width_pixels = (1.0 - alpha) * float(
            prev.get("entry_gate_width_pixels", 0.0)
        ) + alpha * float(segment.get("entry_gate_width_pixels", 0.0))
        pre_entry_u = (1.0 - alpha) * float(
            prev.get("pre_entry_u", 0.0)
        ) + alpha * float(segment.get("pre_entry_u", 0.0))
        pre_entry_v = (1.0 - alpha) * float(
            prev.get("pre_entry_v", 0.0)
        ) + alpha * float(segment.get("pre_entry_v", 0.0))
        pre_entry_confidence = (1.0 - alpha) * float(
            prev.get("pre_entry_confidence", 0.0)
        ) + alpha * float(segment.get("pre_entry_confidence", 0.0))
        pre_entry_depth = (1.0 - alpha) * float(
            prev.get("pre_entry_depth", 0.0)
        ) + alpha * float(segment.get("pre_entry_depth", 0.0))
        ramp_confidence = (1.0 - alpha) * float(
            prev.get("ramp_confidence", 0.0)
        ) + alpha * float(segment.get("ramp_confidence", 0.0))
        side_view_score = (1.0 - alpha) * float(
            prev.get("side_view_score", 1.0)
        ) + alpha * float(segment.get("side_view_score", 1.0))
        vertical_coverage_score = (1.0 - alpha) * float(
            prev.get("vertical_coverage_score", 0.0)
        ) + alpha * float(segment.get("vertical_coverage_score", 0.0))
        return {
            "found": found,
            "raw_found": found,
            "usable": found,
            "predicted_or_cached": False,
            "last_valid_time": now,
            "last_update_time": now,
            "delta_x": delta_x,
            "area_ratio": area_ratio,
            "bottom_coverage": bottom_coverage,
            "center_x": center_x,
            "center_y_ratio": center_y_ratio,
            "top_y_ratio": top_y_ratio,
            "bottom_y_ratio": bottom_y_ratio,
            "bottom_center_x": bottom_center_x,
            "top_center_x": top_center_x,
            "mid_center_x": mid_center_x,
            "bottom_left_x": bottom_left_x,
            "bottom_right_x": bottom_right_x,
            "mid_left_x": mid_left_x,
            "mid_right_x": mid_right_x,
            "bottom_width_ratio": bottom_width_ratio,
            "entry_u": entry_u,
            "entry_v": entry_v,
            "entry_confidence": entry_confidence,
            "entry_from_road_connection": entry_from_road_connection,
            "centerline_slope_pixels": centerline_slope_pixels,
            "frontalness": frontalness,
            "entry_depth": entry_depth,
            "target_u": target_u,
            "target_v": target_v,
            "target_confidence": target_confidence,
            "entry_confirmed": float(segment.get("entry_confirmed", 0.0)),
            "entry_gate_left_u": entry_gate_left_u,
            "entry_gate_right_u": entry_gate_right_u,
            "entry_gate_center_u": entry_gate_center_u,
            "entry_gate_v": entry_gate_v,
            "entry_gate_width_pixels": entry_gate_width_pixels,
            "pre_entry_u": pre_entry_u,
            "pre_entry_v": pre_entry_v,
            "pre_entry_confidence": pre_entry_confidence,
            "pre_entry_depth": pre_entry_depth,
            "entry_gate_confirmed": float(segment.get("entry_gate_confirmed", 0.0)),
            "ramp_valid": float(segment.get("ramp_valid", 0.0)),
            "ramp_confidence": ramp_confidence,
            "ramp_lower_present": float(segment.get("ramp_lower_present", 0.0)),
            "ramp_continuous": float(segment.get("ramp_continuous", 0.0)),
            "side_view_score": side_view_score,
            "vertical_coverage_score": vertical_coverage_score,
            "ramp_reason_code": float(segment.get("ramp_reason_code", 0.0)),
        }

    def _control_loop(self):
        self._publish_startup_arm_stow_if_needed()
        self._publish_target_label()
        self._publish_state()
        self._publish_initial_pose_if_needed()
        self._update_pose_from_tf()
        self._publish_bridge_debug_outputs()

        if self.pose is None:
            self._publish_action("STOP")
            self._log_waiting_for_pose()
            return

        if self._should_interrupt_search_for_target():
            self._publish_action("STOP")
            self._clear_navigation()
            self.exploration_scan_until_time = None
            self.bear_context = "task1_ground_bear"
            self._set_state(MissionState.APPROACH_BEAR)
            return

        if self.state == MissionState.EXPLORE_MAP:
            self._explore_map()
        elif self.state == MissionState.GO_TO_BEAR_AREA:
            self._navigate_state(
                goal_name="bear_search_pose",
                goal=self.bear_search_pose,
                next_state=MissionState.SEARCH_BEAR,
            )
        elif self.state == MissionState.SEARCH_BEAR:
            self._search_bear()
        elif self.state == MissionState.APPROACH_BEAR:
            self._approach_bear()
        elif self.state == MissionState.OBSERVE_BEAR:
            self._observe_bear()
        elif self.state == MissionState.APPROACH_GRAB:
            self._approach_for_grab()
        elif self.state == MissionState.SECURE_BEAR:
            self._secure_bear()
        elif self.state == MissionState.VERIFY_GRAB:
            self._verify_grab()
        elif self.state == MissionState.RETURN_START:
            self._navigate_state(
                goal_name="start_pose",
                goal=self.start_pose,
                next_state=MissionState.DROP_BEAR,
            )
        elif self.state == MissionState.DROP_BEAR:
            self._drop_bear()
        elif self.state == MissionState.TASK2_NAVIGATE_BRIDGE:
            self._task2_navigate_bridge()
        elif self.state == MissionState.TASK2_SEARCH_BRIDGE:
            self._task2_search_bridge()
        elif self.state == MissionState.TASK2_EXPLORE_FOR_BRIDGE:
            self._task2_explore_for_bridge()
        elif self.state == MissionState.TASK2_TURN_TO_BRIDGE:
            self._task2_turn_to_bridge()
        elif self.state == MissionState.TASK2_SIDE_VIEW_RECOVERY:
            self._task2_side_view_recovery()
        elif self.state == MissionState.TASK2_APPROACH_BRIDGE_ENTRY:
            self._task2_approach_bridge_entry()
        elif self.state == MissionState.TASK2_FINAL_ALIGN_BRIDGE:
            self._task2_final_align_bridge()
        elif self.state == MissionState.TASK2_ASCEND_BRIDGE:
            self._task2_ascend_bridge()
        elif self.state == MissionState.TASK2_SEARCH_BRIDGE_BEAR:
            self._task2_search_bridge_bear()
        elif self.state == MissionState.TASK2_DESCEND_BRIDGE:
            self._task2_descend_bridge()
        elif self.state == MissionState.DONE:
            self._publish_action("STOP")

    def _navigate_state(self, goal_name, goal, next_state):
        if self.state == MissionState.RETURN_START and not self._check_return_hold():
            return

        if self.current_goal_name != goal_name:
            self._start_navigation(goal_name, goal)

        if self._goal_reached(goal):
            self._publish_action("STOP")
            self._clear_navigation()
            self._set_state(next_state)
            return

        self._republish_goal_if_needed(goal)
        action = self._action_from_plan()
        if self._should_use_direct_return_fallback(action):
            action = self._action_toward_pose(goal)
            self._announce_direct_return_fallback()
        elif action == "STOP":
            self._log_waiting_for_plan(goal_name)

        self._publish_action(action)

    def _start_navigation(self, goal_name, goal):
        self.current_goal_name = goal_name
        self.current_goal = goal
        self.path_index = 0
        self.latest_path = None
        self.navigation_goal_start_time = self.get_clock().now()
        self.last_goal_publish_time = self.navigation_goal_start_time
        self.last_no_plan_log_time = None
        self.return_direct_fallback_announced = False
        self._publish_goal(goal)
        self.get_logger().info(
            f"Published {goal_name}: x={goal[0]:.2f}, y={goal[1]:.2f}, "
            f"yaw={goal[2]:.2f}"
        )

    def _clear_navigation(self):
        self.current_goal_name = None
        self.current_goal = None
        self.latest_path = None
        self.path_index = 0
        self.navigation_goal_start_time = None
        self.last_goal_publish_time = None
        self.last_no_plan_log_time = None
        self.return_direct_fallback_announced = False

    def _goal_reached(self, goal):
        return distance_2d(self.pose[:2], goal[:2]) <= self.goal_tolerance

    def _republish_goal_if_needed(self, goal):
        if self.goal_republish_period <= 0.0:
            return
        now = self.get_clock().now()
        if (
            self.last_goal_publish_time is not None
            and self._elapsed_seconds(self.last_goal_publish_time)
            < self.goal_republish_period
        ):
            return
        self.last_goal_publish_time = now
        self._publish_goal(goal)

    def _should_use_direct_return_fallback(self, planned_action):
        if not self.return_direct_fallback:
            return False
        if self.state != MissionState.RETURN_START:
            return False
        if planned_action != "STOP":
            return False
        if self.navigation_goal_start_time is None:
            return False
        return self._elapsed_seconds(self.navigation_goal_start_time) >= (
            self.navigation_no_plan_timeout
        )

    def _announce_direct_return_fallback(self):
        if self.return_direct_fallback_announced:
            return
        self.return_direct_fallback_announced = True
        self.get_logger().warn(
            "No usable /received_global_plan for start_pose; "
            "using slow direct return fallback."
        )

    def _log_waiting_for_plan(self, goal_name):
        now = self.get_clock().now()
        if (
            self.last_no_plan_log_time is not None
            and self._elapsed_seconds(self.last_no_plan_log_time) < 3.0
        ):
            return
        self.last_no_plan_log_time = now
        self.get_logger().warn(
            f"Waiting for /received_global_plan for {goal_name}; "
            "republishing /goal_pose."
        )

    def _should_interrupt_search_for_target(self):
        search_states = (
            MissionState.EXPLORE_MAP,
            MissionState.GO_TO_BEAR_AREA,
            MissionState.SEARCH_BEAR,
        )
        return self.state in search_states and self._task1_target_visible()

    def _explore_map(self):
        if self.exploration_scan_until_time is not None:
            if self._elapsed_seconds(self.exploration_scan_until_time) < 0.0:
                self._publish_action("CLOCKWISE_ROTATION_SLOW")
                return
            self.exploration_scan_until_time = None

        if self.current_goal_name == "explore_goal" and self.current_goal is not None:
            if self._goal_reached(self.current_goal):
                self._mark_exploration_goal_visited(self.current_goal)
                self._publish_action("STOP")
                self._clear_navigation()
                self._start_exploration_scan()
                return

            if self._navigation_goal_timed_out():
                self.get_logger().warn("Exploration goal timed out; trying another map point.")
                self._mark_exploration_goal_visited(self.current_goal)
                self._publish_action("STOP")
                self._clear_navigation()
            else:
                self._publish_action(self._action_from_plan())
                return

        next_goal = self._select_next_exploration_goal()
        if next_goal is None:
            self._log_waiting_for_exploration_goal()
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
            return

        self._start_navigation("explore_goal", next_goal)
        self._publish_action(self._action_from_plan())

    def _start_exploration_scan(self):
        if self.exploration_scan_seconds <= 0.0:
            return
        now = self.get_clock().now()
        self.exploration_scan_until_time = self._time_after(now, self.exploration_scan_seconds)

    def _navigation_goal_timed_out(self):
        if self.navigation_goal_start_time is None:
            return False
        return self._elapsed_seconds(self.navigation_goal_start_time) >= self.exploration_goal_timeout

    def _select_next_exploration_goal(self):
        self._refresh_exploration_candidates()
        if not self.exploration_candidates or self.pose is None:
            return None

        options = self._unvisited_exploration_options()
        if not options:
            self.visited_exploration_goals.clear()
            options = self._unvisited_exploration_options()

        if not options:
            return None

        options.sort(key=lambda goal: distance_2d(self.pose[:2], goal[:2]))
        goal = options[0]
        goal[2] = math.atan2(goal[1] - self.pose[1], goal[0] - self.pose[0])
        return goal

    def _unvisited_exploration_options(self):
        options = []
        for goal in self.exploration_candidates:
            if self._exploration_goal_key(goal) in self.visited_exploration_goals:
                continue
            if distance_2d(self.pose[:2], goal[:2]) < self.exploration_min_goal_distance:
                continue
            options.append(list(goal))
        return options

    def _refresh_exploration_candidates(self):
        if self.map_msg is None:
            return
        if self.exploration_candidates_revision == self.map_revision:
            return

        info = self.map_msg.info
        resolution = info.resolution
        if resolution <= 0.0:
            return

        width = info.width
        height = info.height
        stride = max(1, int(self.exploration_grid_spacing / resolution))
        clearance_cells = max(0, int(self.exploration_clearance / resolution))
        candidates = []

        reverse_row = False
        for grid_y in range(clearance_cells, height - clearance_cells, stride):
            grid_x_values = range(clearance_cells, width - clearance_cells, stride)
            if reverse_row:
                grid_x_values = reversed(list(grid_x_values))
            reverse_row = not reverse_row

            for grid_x in grid_x_values:
                if not self._map_cell_has_clearance(grid_x, grid_y, clearance_cells):
                    continue
                candidates.append(self._map_to_world_pose(grid_x, grid_y))

        self.exploration_candidates = candidates
        self.exploration_candidates_revision = self.map_revision
        self.get_logger().info(
            f"Loaded {len(candidates)} exploration goals from {self.map_topic}."
        )

    def _map_cell_has_clearance(self, grid_x, grid_y, clearance_cells):
        for check_y in range(grid_y - clearance_cells, grid_y + clearance_cells + 1):
            for check_x in range(grid_x - clearance_cells, grid_x + clearance_cells + 1):
                if not self._map_cell_is_free(check_x, check_y):
                    return False
        return True

    def _map_cell_is_free(self, grid_x, grid_y):
        info = self.map_msg.info
        if grid_x < 0 or grid_y < 0 or grid_x >= info.width or grid_y >= info.height:
            return False
        value = self.map_msg.data[grid_y * info.width + grid_x]
        if not (0 <= value <= self.map_free_threshold):
            return False
        x = info.origin.position.x + (grid_x + 0.5) * info.resolution
        y = info.origin.position.y + (grid_y + 0.5) * info.resolution
        return not self._virtual_obstacle_blocks_world(x, y)

    def _map_to_world_pose(self, grid_x, grid_y):
        info = self.map_msg.info
        origin = info.origin.position
        x = origin.x + (grid_x + 0.5) * info.resolution
        y = origin.y + (grid_y + 0.5) * info.resolution
        return [x, y, 0.0]

    def _exploration_goal_key(self, goal):
        scale = max(self.exploration_grid_spacing, 0.01)
        return (round(goal[0] / scale), round(goal[1] / scale))

    def _mark_exploration_goal_visited(self, goal):
        self.visited_exploration_goals.add(self._exploration_goal_key(goal))

    def _action_from_plan(self):
        if self.latest_path is None or not self.latest_path.poses:
            return "STOP"

        last_index = len(self.latest_path.poses) - 1
        self.path_index = min(max(self.path_index, 0), last_index)

        while self.path_index < last_index:
            point = self.latest_path.poses[self.path_index].pose.position
            if distance_2d(self.pose[:2], (point.x, point.y)) >= self.lookahead_distance:
                break
            self.path_index += 1

        target = self.latest_path.poses[self.path_index].pose.position
        target_yaw = math.atan2(target.y - self.pose[1], target.x - self.pose[0])
        diff = normalize_angle(target_yaw - self.pose[2])

        if abs(diff) <= self.angle_tolerance:
            return "FORWARD"
        if diff > 0.0:
            return "COUNTERCLOCKWISE_ROTATION"
        return "CLOCKWISE_ROTATION"

    def _action_toward_pose(self, goal):
        distance = distance_2d(self.pose[:2], goal[:2])
        if distance <= self.goal_tolerance:
            return "STOP"

        target_yaw = math.atan2(goal[1] - self.pose[1], goal[0] - self.pose[0])
        diff = normalize_angle(target_yaw - self.pose[2])

        if abs(diff) <= self.angle_tolerance:
            return "FORWARD_SLOW"
        if diff > 0.0:
            return "COUNTERCLOCKWISE_ROTATION_SLOW"
        return "CLOCKWISE_ROTATION_SLOW"

    def _search_bear(self):
        if self._task1_target_visible():
            self._publish_action("STOP")
            self.bear_context = "task1_ground_bear"
            self._set_state(MissionState.APPROACH_BEAR)
            return
        self._publish_action("CLOCKWISE_ROTATION_SLOW")

    def _approach_bear(self):
        if not self._mission_target_visible():
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
            return

        delta_x = self.yolo_target["delta_x"]
        distance = self.yolo_target["distance"]

        if abs(delta_x) > self.align_pixel_tolerance:
            if delta_x > 0.0:
                self._publish_action("CLOCKWISE_ROTATION_SLOW")
            else:
                self._publish_action("COUNTERCLOCKWISE_ROTATION_SLOW")
            return

        lower_ok, lower_reason = self._target_bbox_in_lower_camera()
        if distance > 0.0 and distance <= self.observe_distance and lower_ok:
            self._publish_action("STOP")
            self.observe_start_time = self.get_clock().now()
            self.observe_start_pose = self.pose
            self.last_observed_target_time = self.observe_start_time
            self.last_observed_target_distance = distance
            self._set_state(MissionState.OBSERVE_BEAR)
            return
        if distance > 0.0 and distance <= self.observe_distance and not lower_ok:
            self._publish_action("STOP")
            self._log_lower_bbox_gate("observe", lower_reason)
            return

        self._publish_action("FORWARD_SLOW")

    def _observe_bear(self):
        self._publish_action("STOP")

        lower_ok, lower_reason = self._target_bbox_in_lower_camera()
        if self._mission_target_visible() and lower_ok:
            self.last_observed_target_time = self.get_clock().now()
            self.last_observed_target_distance = self.yolo_target["distance"]
        elif (
            self.last_observed_target_time is None
            or self._elapsed_seconds(self.last_observed_target_time)
            > self.observe_target_loss_grace
        ):
            if not lower_ok:
                self._log_lower_bbox_gate("observe", lower_reason)
            self._reset_observation()
            self._set_state(MissionState.APPROACH_BEAR)
            return

        distance = self.last_observed_target_distance
        if (
            distance > 0.0
            and distance > self.observe_distance + self.observe_distance_margin
        ):
            self._reset_observation()
            self._set_state(MissionState.APPROACH_BEAR)
            return

        if (
            self.observe_start_pose is not None
            and distance_2d(self.pose[:2], self.observe_start_pose[:2])
            > self.stationary_tolerance
        ):
            self.observe_start_time = self.get_clock().now()
            self.observe_start_pose = self.pose
            return

        elapsed = self._elapsed_seconds(self.observe_start_time)
        if elapsed >= self.observe_seconds:
            self.task1_observe_completed = True
            self._log_event(
                "info",
                "task1_observe_completed",
                task1_observe_completed=True,
                bear_context=self.bear_context,
            )
            self.grab_approach_start_time = None
            self.grab_approach_open_sent = False
            self._set_state(MissionState.APPROACH_GRAB)

    def _reset_observation(self):
        self.observe_start_time = None
        self.observe_start_pose = None
        self.last_observed_target_time = None
        self.last_observed_target_distance = 0.0

    def _approach_for_grab(self):
        if self.grab_approach_start_time is None:
            self.grab_approach_start_time = self.get_clock().now()
            self.grab_approach_open_sent = False
            self.grab_ready_start_time = None
            self.last_grab_target_time = None
            self.last_grab_target_distance = 0.0
            self.last_grab_target_delta_x = 0.0

        if self.enable_grab_sequence and not self.grab_approach_open_sent:
            self._publish_arm_positions(self.arm_open_positions)
            self.grab_approach_open_sent = True

        if self._elapsed_seconds(self.grab_approach_start_time) >= self.grab_approach_timeout:
            self._publish_action("STOP")
            self.get_logger().warn("Grab approach timed out before grab gate; reacquiring bear.")
            self._reset_grab_attempt_state()
            self._set_state(MissionState.APPROACH_BEAR)
            return

        if not self._mission_target_visible():
            self.grab_ready_start_time = None
            self._publish_action("STOP")
            return

        delta_x = self.yolo_target["delta_x"]
        distance = self.yolo_target["distance"]
        self.last_grab_target_time = self.get_clock().now()
        self.last_grab_target_distance = distance
        self.last_grab_target_delta_x = delta_x

        if abs(delta_x) > self.grab_align_pixel_tolerance:
            self.grab_ready_start_time = None
            if delta_x > 0.0:
                self._publish_action("CLOCKWISE_ROTATION_SLOW")
            else:
                self._publish_action("COUNTERCLOCKWISE_ROTATION_SLOW")
            return

        ready, ready_reason = self._target_ready_for_grab(require_bridge=True)
        if ready:
            self._publish_action("STOP")
            if self.grab_ready_start_time is None:
                self.grab_ready_start_time = self.get_clock().now()
                return
            if self._elapsed_seconds(self.grab_ready_start_time) >= self.grab_confirm_seconds:
                self.pre_grab_bbox = self._copy_current_bbox()
                self._set_state(MissionState.SECURE_BEAR)
            return
        if distance > 0.0 and distance <= self.grab_distance:
            self.grab_ready_start_time = None
            self._publish_action("STOP")
            self._log_grab_gate_wait(ready_reason)
            return

        self.grab_ready_start_time = None
        self._publish_action("FORWARD_SLOW")

    def _last_grab_target_was_close(self):
        if self.last_grab_target_time is None:
            return False
        if self._elapsed_seconds(self.last_grab_target_time) > 1.5:
            return False
        return (
            0.0 < self.last_grab_target_distance
            <= self.grab_distance + self.grab_lost_target_secure_margin
            and abs(self.last_grab_target_delta_x) <= self.grab_align_pixel_tolerance * 1.5
        )

    def _secure_bear(self):
        self._publish_action("STOP")

        if self.grab_start_time is None:
            self.bear_secured = False
            ready, reason = self._target_ready_for_grab(require_bridge=True)
            if not ready:
                self.get_logger().warn(
                    f"Pre-grasp gate rejected before arm motion: {reason}; realigning."
                )
                self._reset_grab_attempt_state()
                self._set_state(MissionState.APPROACH_GRAB)
                return
            self.grab_start_time = self.get_clock().now()
            self.grab_step_index = 0
            self.grab_step_sent = False
            self.grab_step_deadline = None

        if not self.enable_grab_sequence:
            if self._elapsed_seconds(self.grab_start_time) >= self.grab_wait_seconds:
                self._finish_grab_sequence()
            return

        sequence = [
            (self.arm_open_positions, 0.4),
            (self.arm_pregrasp_positions, 0.5),
            (self.arm_reach_positions, 0.8),
            (self.arm_close_positions, 1.1),
            (self.arm_lift_positions, 0.8),
            (self.arm_carry_positions, 0.5),
        ]

        if self.grab_step_index >= len(sequence):
            self._finish_grab_sequence()
            return

        positions, hold_seconds = sequence[self.grab_step_index]
        now = self.get_clock().now()

        if not self.grab_step_sent:
            self._publish_arm_positions(positions)
            self.get_logger().info(
                f"Arm grab step {self.grab_step_index + 1}/{len(sequence)}: "
                f"{[round(value, 2) for value in positions]}"
                + (" deg" if self.arm_positions_in_degrees else " rad")
            )
            self.grab_step_deadline = now.nanoseconds + int(hold_seconds * 1e9)
            self.grab_step_sent = True
            return

        if self.grab_step_deadline is not None and now.nanoseconds >= self.grab_step_deadline:
            self.grab_step_index += 1
            self.grab_step_sent = False
            self.grab_step_deadline = None

    def _drop_bear(self):
        self._publish_action("STOP")

        if self.drop_start_time is None:
            self.drop_start_time = self.get_clock().now()
            self.drop_step_index = 0
            self.drop_step_sent = False
            self.drop_step_deadline = None
            self.get_logger().info("Dropping bear at start pose.")

        if not self.enable_drop_sequence:
            if self._elapsed_seconds(self.drop_start_time) >= self.drop_wait_seconds:
                self._finish_drop_sequence()
            return

        sequence = [
            (self.arm_lift_positions, 0.4),
            (self.arm_drop_positions, 0.8),
            (self.arm_release_positions, 1.0),
            (self.arm_drop_retract_positions, 0.6),
        ]

        if self.drop_step_index >= len(sequence):
            self._finish_drop_sequence()
            return

        positions, hold_seconds = sequence[self.drop_step_index]
        now = self.get_clock().now()

        if not self.drop_step_sent:
            self._publish_arm_positions(positions)
            self.get_logger().info(
                f"Arm drop step {self.drop_step_index + 1}/{len(sequence)}: "
                f"{[round(value, 2) for value in positions]}"
                + (" deg" if self.arm_positions_in_degrees else " rad")
            )
            self.drop_step_deadline = now.nanoseconds + int(hold_seconds * 1e9)
            self.drop_step_sent = True
            return

        if self.drop_step_deadline is not None and now.nanoseconds >= self.drop_step_deadline:
            self.drop_step_index += 1
            self.drop_step_sent = False
            self.drop_step_deadline = None

    def _finish_drop_sequence(self):
        self._reset_drop_state()
        self._reset_grab_attempt_state()
        self.bear_secured = False
        if self.mission_mode == "bridge_first_shared_bear":
            self.task1_recovery_completed = True
            self.task2_recovery_completed = True
            self._log_event(
                "info",
                "shared_bridge_bear_dropped_at_start",
                task1_recovery_completed=self.task1_recovery_completed,
                task2_recovery_completed=self.task2_recovery_completed,
            )
            self.get_logger().info(
                "Shared bridge-bear mission complete: "
                f"task1_observe={self.task1_observe_completed}, "
                f"task1_recovery={self.task1_recovery_completed}, "
                f"task2_ascent={self.task2_ascent_completed}, "
                f"task2_descent={self.task2_descent_completed}, "
                f"task2_recovery={self.task2_recovery_completed}."
            )
            self._set_state(MissionState.DONE)
            return

        if self.current_task == 1:
            self.task1_recovery_completed = True
            self._log_event(
                "info",
                "task1_recovery_completed",
                task1_recovery_completed=True,
            )

        if self.current_task == 1 and self.run_task2_after_task1:
            self._start_task2()
            return
        if self.current_task == 2:
            self.task2_recovery_completed = True
            self._log_event(
                "info",
                "task2_recovery_completed",
                task2_recovery_completed=True,
            )
        self._set_state(MissionState.DONE)

    def _reset_drop_state(self):
        self.drop_start_time = None
        self.drop_step_index = 0
        self.drop_step_sent = False
        self.drop_step_deadline = None

    def _start_task2(self):
        self.current_task = 2
        self.bear_context = None
        self.bear_secured = False
        self.next_state_after_grab = MissionState.TASK2_DESCEND_BRIDGE
        self.task2_phase_start_time = None
        self.task2_bridge_confirm_start_time = None
        self.task2_search_cycle_start_time = None
        self.task2_ramp_entry_confirm_count = 0
        self.task2_side_view_recovery_start_time = None
        self.task2_ascent_start_z = None
        self.bridge_top_confirmed = False
        self.bridge_top_confirm_count = 0
        self.bridge_bear_memory = BridgeBearMemory()
        self._clear_navigation()
        self.get_logger().info("Starting Task 2 after Task 1 completion.")
        self._set_state(MissionState.TASK2_SEARCH_BRIDGE)

    def _task2_navigate_bridge(self):
        if self.task2_use_bridge_entry_pose:
            self._navigate_state(
                goal_name="task2_bridge_entry_pose",
                goal=self.task2_bridge_entry_pose,
                next_state=MissionState.TASK2_FINAL_ALIGN_BRIDGE,
            )
            return
        self._set_state(MissionState.TASK2_SEARCH_BRIDGE)

    def _task2_search_bridge_by_segmentation(self):
        self._task2_search_bridge()

    def _task2_search_bridge(self):
        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()
            self._reset_task2_scan()
            self.get_logger().info("Task 2: bounded 360 bridge scan started.")

        bridge = self._task2_bridge_visible()
        if self._task2_confirmed_bridge_candidate(bridge):
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: confirmed bridge entry candidate; turning toward bridge."
            )
            self._set_state(MissionState.TASK2_TURN_TO_BRIDGE)
            return

        if self._bridge_landmark_is_usable():
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: using cached bridge landmark instead of repeating full scan."
            )
            self._set_state(MissionState.TASK2_TURN_TO_BRIDGE)
            return

        scan_complete, timed_out = self._task2_update_scan_progress()
        if scan_complete or timed_out or self._budget_exceeded(self.task2_search_budget_seconds):
            self._publish_action("STOP")
            reason = "timeout" if timed_out else "budget" if self._budget_exceeded(self.task2_search_budget_seconds) else "full scan"
            self.get_logger().warn(
                f"Task 2: {reason} completed without confirmed bridge entry; "
                "switching to road-guided exploration."
            )
            self._set_state(MissionState.TASK2_EXPLORE_FOR_BRIDGE)
            return

        self._publish_action(self._task2_scan_action())

    def _task2_explore_for_bridge(self):
        if self.task2_explore_segment_start_time is None:
            self.task2_explore_segment_start_time = self.get_clock().now()
            self.task2_explore_segment_start_pose = tuple(self.pose)
            self.task2_road_reacquire_start_time = None
            self._task2_reset_bridge_confirm()
            self.get_logger().info(
                "Task 2 bridge exploration: following road corridor for one segment."
            )

        bridge = self._task2_bridge_visible()
        if self._task2_confirmed_bridge_candidate(bridge):
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: bridge detected during road exploration; turning toward entry."
            )
            self._reset_task2_exploration_segment()
            self._set_state(MissionState.TASK2_TURN_TO_BRIDGE)
            return

        if self._bridge_landmark_is_usable():
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: bridge landmark available during road exploration; homing to it."
            )
            self._reset_task2_exploration_segment()
            self._set_state(MissionState.TASK2_TURN_TO_BRIDGE)
            return

        elapsed = self._elapsed_seconds(self.task2_explore_segment_start_time)
        translation = 0.0
        if self.task2_explore_segment_start_pose is not None:
            translation = distance_2d(
                self.pose[:2], self.task2_explore_segment_start_pose[:2]
            )

        if (
            translation >= self.task2_explore_goal_min_translation
            or elapsed >= self.task2_road_explore_segment_seconds
            or elapsed >= self.task2_explore_max_segment_seconds
            or self._budget_exceeded(self.task2_target_total_budget_seconds)
        ):
            self._publish_action("STOP")
            if self.task2_explore_segment_start_pose is not None:
                self.task2_road_explore_visited.add(
                    self._task2_explore_pose_key(self.task2_explore_segment_start_pose)
                )
            self.get_logger().info(
                "Task 2 road exploration segment complete: "
                f"translation={translation:.2f}m, elapsed={elapsed:.1f}s; rescanning."
            )
            self._reset_task2_exploration_segment()
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        self._publish_action(self._task2_road_explore_action())

    def _task2_confirmed_bridge_candidate(self, bridge):
        if (
            bridge is None
            or not self._bridge_observation_is_fresh(bridge)
            or not self._task2_bridge_has_entry_candidate(bridge)
        ):
            self.task2_bridge_detection_start_time = None
            return False

        if (
            not self._task2_bridge_entry_confirmed(bridge)
            and not self._bridge_ramp_is_usable(bridge)
        ):
            self.task2_bridge_detection_start_time = None
            return False

        if self.task2_bridge_detection_start_time is None:
            self.task2_bridge_detection_start_time = self.get_clock().now()
            return False

        return (
            self._elapsed_seconds(self.task2_bridge_detection_start_time)
            >= self.task2_bridge_detection_confirm_seconds
        )

    def _bridge_ramp_is_usable(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        return BridgeVisionAnalyzer.ramp_is_usable(self, bridge)

    def _bridge_side_view_likely(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        return BridgeVisionAnalyzer.side_view_likely(self, bridge)

    def _task2_entry_source(self, bridge=None, update_ramp_confirm=True):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        source = BridgeVisionAnalyzer.entry_source(self, bridge)
        if source == "road_contact":
            delta = self._task2_bridge_entry_delta(bridge)
            if (
                self._task2_bridge_pre_entry_confirmed(bridge)
                and delta is not None
                and abs(delta) <= self.task2_bridge_approach_hard_tolerance
            ):
                self.task2_ramp_entry_confirm_count = 0
                return source
            source = "ramp_fallback" if self._bridge_ramp_is_usable(bridge) else "none"
        if source == "ramp_fallback":
            if update_ramp_confirm:
                self.task2_ramp_entry_confirm_count += 1
            if (
                self.task2_ramp_entry_confirm_count
                >= self.task2_ramp_entry_confirm_frames
            ):
                return source
            return "none"
        if update_ramp_confirm:
            self.task2_ramp_entry_confirm_count = 0
        return "none"

    def _task2_entry_delta(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        source = self._task2_entry_source(bridge, update_ramp_confirm=False)
        if source == "road_contact":
            return self._task2_bridge_pre_entry_delta(bridge) or self._task2_bridge_entry_delta(bridge)
        if source == "ramp_fallback":
            return BridgeVisionAnalyzer.ramp_delta(self, bridge)
        return BridgeVisionAnalyzer.ramp_delta(self, bridge)

    def _task2_update_scan_progress(self):
        if self.pose is None:
            return False, False

        current_yaw = self.pose[2]
        if self.task2_scan_previous_yaw is None:
            self.task2_scan_previous_yaw = current_yaw
            self.task2_scan_start_pose = tuple(self.pose)
            return False, False

        delta_yaw = normalize_angle(current_yaw - self.task2_scan_previous_yaw)
        self.task2_scan_accumulated_yaw += abs(delta_yaw)
        self.task2_scan_previous_yaw = current_yaw

        now = self.get_clock().now()
        if (
            self.task2_last_scan_log_time is None
            or self._elapsed_seconds(self.task2_last_scan_log_time) >= 1.0
        ):
            self.task2_last_scan_log_time = now
            self.get_logger().info(
                "Task 2 bridge scan rotation: "
                f"{math.degrees(self.task2_scan_accumulated_yaw):.0f} deg."
            )

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        scan_complete = (
            elapsed >= self.task2_scan_min_seconds
            and self.task2_scan_accumulated_yaw >= self.task2_scan_complete_radians
        )
        timed_out = elapsed >= self.task2_scan_timeout_seconds
        return scan_complete, timed_out

    def _task2_scan_action(self):
        return (
            "CLOCKWISE_ROTATION_SLOW"
            if self.task2_scan_direction >= 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _reset_task2_scan(self):
        self.task2_scan_previous_yaw = None
        self.task2_scan_accumulated_yaw = 0.0
        self.task2_scan_start_pose = None
        self.task2_last_scan_log_time = None
        self.task2_bridge_detection_start_time = None

    def _reset_task2_exploration_segment(self):
        self.task2_explore_segment_start_time = None
        self.task2_explore_segment_start_pose = None
        self.task2_road_reacquire_start_time = None

    def _task2_explore_pose_key(self, pose):
        scale = max(0.25, self.task2_explore_goal_min_translation)
        return (round(pose[0] / scale), round(pose[1] / scale))

    def _task2_road_corridor(self):
        road = self._segmentation_segment("road")
        image_width = self._segmentation_image_width()
        if road is None or image_width <= 0.0:
            return {"valid": False, "reason": "no fresh road segmentation"}

        bottom_left = float(road.get("bottom_left_x", 0.0))
        bottom_right = float(road.get("bottom_right_x", 0.0))
        bottom_center = float(road.get("bottom_center_x", 0.0))
        mid_left = float(road.get("mid_left_x", 0.0))
        mid_right = float(road.get("mid_right_x", 0.0))
        mid_center = float(road.get("mid_center_x", 0.0))

        if bottom_center <= 0.0 and bottom_left > 0.0 and bottom_right > bottom_left:
            bottom_center = (bottom_left + bottom_right) * 0.5
        if mid_center <= 0.0 and mid_left > 0.0 and mid_right > mid_left:
            mid_center = (mid_left + mid_right) * 0.5
        if mid_center <= 0.0:
            mid_center = float(road.get("top_center_x", 0.0)) or bottom_center

        width_pixels = max(0.0, bottom_right - bottom_left)
        width_ratio = float(road.get("bottom_width_ratio", 0.0))
        if width_ratio <= 0.0 and width_pixels > 0.0:
            width_ratio = width_pixels / image_width

        area_ratio = float(road.get("area_ratio", 0.0))
        bottom_coverage = float(road.get("bottom_coverage", 0.0))
        heading_error = 0.65 * (bottom_center - image_width * 0.5) + 0.35 * (
            mid_center - image_width * 0.5
        )
        curvature_error = mid_center - bottom_center

        valid = True
        reason = "ok"
        if not road.get("found", False):
            valid = False
            reason = "road not found"
        elif area_ratio < self.task2_road_explore_min_area_ratio:
            valid = False
            reason = f"area_ratio {area_ratio:.3f} below threshold"
        elif bottom_coverage < self.task2_road_explore_min_bottom_coverage:
            valid = False
            reason = f"bottom_coverage {bottom_coverage:.3f} below threshold"
        elif bottom_center <= 0.0:
            valid = False
            reason = "bottom center unavailable"
        elif width_ratio < self.task2_road_explore_min_width_ratio:
            valid = False
            reason = f"width_ratio {width_ratio:.2f} below threshold"

        return {
            "valid": valid,
            "reason": reason,
            "bottom_left_x": bottom_left,
            "bottom_right_x": bottom_right,
            "bottom_center_x": bottom_center,
            "top_or_mid_center_x": mid_center,
            "width_pixels": width_pixels,
            "width_ratio": width_ratio,
            "heading_error": heading_error,
            "curvature_error": curvature_error,
            "bottom_coverage": bottom_coverage,
            "area_ratio": area_ratio,
        }

    def _task2_road_explore_action(self):
        corridor = self._task2_road_corridor()
        now = self.get_clock().now()
        if (
            self.last_road_explore_log_time is None
            or self._elapsed_seconds(self.last_road_explore_log_time) >= 1.0
        ):
            self.last_road_explore_log_time = now
            if corridor["valid"]:
                self.get_logger().info(
                    "Task 2 road corridor: "
                    f"error={corridor['heading_error']:.0f}px, "
                    f"width={corridor['width_ratio']:.2f}, "
                    f"coverage={corridor['bottom_coverage']:.2f}."
                )
            else:
                self.get_logger().info(
                    f"Task 2 road corridor invalid: {corridor['reason']}."
                )

        if not corridor["valid"]:
            if self.task2_road_reacquire_start_time is None:
                self.task2_road_reacquire_start_time = now
            if (
                self._elapsed_seconds(self.task2_road_reacquire_start_time)
                >= self.task2_road_reacquire_timeout_seconds
            ):
                return "STOP"
            return self._task2_scan_action()

        self.task2_road_reacquire_start_time = None
        error = corridor["heading_error"]
        if abs(error) > self.task2_road_explore_hard_tolerance:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if error > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )
        if abs(error) > self.task2_road_explore_center_tolerance:
            return "RIGHT_FRONT" if error > 0.0 else "LEFT_FRONT"
        return self._avoid_virtual_obstacle_for_action("FORWARD_SLOW")

    def _task2_bridge_confidence_for_ascent(self, bridge):
        landmark = getattr(self, "bridge_landmark", {}) or {}
        values = [
            float(landmark.get("confidence", 0.0)),
            float(bridge.get("entry_confidence", 0.0)) if bridge else 0.0,
            float(bridge.get("ramp_confidence", 0.0)) if bridge else 0.0,
            float(bridge.get("target_confidence", 0.0)) if bridge else 0.0,
        ]
        return max(values)

    def _task2_bridge_bear_delta_for_ascent(self, require_surface=True):
        if not self._target_visible():
            return None, "bridge bear target is not visible"
        if require_surface:
            surface_valid, surface_reason = self._target_surface_candidate()
            if not surface_valid:
                return None, surface_reason

        target = self.yolo_target or {}
        if "delta_x" in target:
            return float(target.get("delta_x", 0.0)), "bridge bear target is visible"

        bbox = self.yolo_bbox or {}
        image_width = float(
            bbox.get("image_width", 0.0)
            or target.get("image_width", 0.0)
            or self._segmentation_image_width()
        )
        center_x = float(bbox.get("center_x", 0.0) or target.get("center_x", 0.0))
        if image_width > 0.0 and center_x > 0.0:
            return center_x - image_width * 0.5, "bridge bear bbox is visible"
        return None, "bridge bear has no image-center measurement"

    def _task2_bridge_bear_turn_action(self, require_surface=True):
        delta, reason = self._task2_bridge_bear_delta_for_ascent(
            require_surface=require_surface
        )
        if delta is None:
            return None, reason
        if abs(delta) <= self.task2_turn_frontal_bridge_bear_tolerance_pixels:
            return "STOP", "bridge bear centered"
        return (
            "CLOCKWISE_ROTATION_SLOW"
            if delta > 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        ), reason

    def _task2_bridge_bear_ascent_action(self):
        if not self.task2_ascent_bear_pid_enabled:
            return None
        delta, _ = self._task2_bridge_bear_delta_for_ascent(require_surface=True)
        if delta is None:
            self._reset_task2_ascent_bear_pid()
            return None
        if abs(delta) <= self.task2_ascent_bear_pid_center_tolerance_pixels:
            self._reset_task2_ascent_bear_pid()
            return None
        return "ASCEND_BEAR_PID"

    def _reset_task2_ascent_bear_pid(self):
        self.task2_ascent_bear_pid_integral = 0.0
        self.task2_ascent_bear_pid_last_error = None
        self.task2_ascent_bear_pid_last_time = None

    def _task2_frontal_bridge_base_ready_for_ascent(self, bridge, delta_x=None):
        if not self.task2_turn_allow_frontal_bridge_ascent:
            return False, "frontal bridge ascent bypass is disabled"
        if bridge is None or not self._bridge_observation_is_fresh(bridge):
            return False, "bridge observation is not fresh"
        if not bridge.get("raw_found", bridge.get("found", False)):
            return False, "bridge mask is cached"

        frontalness = float(bridge.get("frontalness", 0.0))
        if frontalness < self.task2_turn_frontal_bridge_min_frontalness:
            return False, f"frontalness too low ({frontalness:.2f})"

        confidence = self._task2_bridge_confidence_for_ascent(bridge)
        if confidence < self.task2_turn_frontal_bridge_min_confidence:
            return False, f"bridge confidence too low ({confidence:.2f})"

        bottom_y = float(bridge.get("bottom_y_ratio", 0.0))
        if bottom_y < self.task2_turn_frontal_bridge_min_bottom_y_ratio:
            return False, f"bridge bottom y too high ({bottom_y:.2f})"

        if delta_x is None:
            delta_x = BridgeVisionAnalyzer.ramp_delta(self, bridge)
        if delta_x is None:
            delta_x = self._task2_bridge_rough_target_delta(bridge)
        if delta_x is None:
            return False, "no bridge center measurement"
        if abs(delta_x) > self.task2_turn_frontal_bridge_center_tolerance_pixels:
            return False, f"bridge center error too large ({delta_x:.0f}px)"
        return True, "frontal bridge fills lower frame and is centered"

    def _task2_frontal_bridge_ready_for_ascent(self, bridge, delta_x=None):
        ready, reason = self._task2_frontal_bridge_base_ready_for_ascent(
            bridge, delta_x=delta_x
        )
        if not ready:
            return False, reason

        if not self.task2_turn_frontal_bridge_require_target:
            return True, reason

        bear_delta, bear_reason = self._task2_bridge_bear_delta_for_ascent(
            require_surface=True
        )
        if bear_delta is None:
            return False, bear_reason
        if abs(bear_delta) > self.task2_turn_frontal_bridge_bear_tolerance_pixels:
            return False, f"bridge bear is not centered ({bear_delta:.0f}px)"
        return True, f"{reason}; bridge bear centered"

    def _task2_turn_to_bridge(self):
        if self.task2_turn_state_start_time is None:
            self.task2_turn_state_start_time = self.get_clock().now()
            self.task2_turn_centered_frames = 0

        bridge = self._task2_bridge_visible(allow_cached=True)
        fresh = self._bridge_observation_is_fresh(bridge)
        elapsed = self._elapsed_seconds(self.task2_turn_state_start_time)
        if bridge is not None and self._bridge_side_view_likely(bridge):
            if (
                elapsed >= self.task2_side_view_max_turn_seconds
                or StuckMonitor.low_progress(self, self.task2_side_view_stuck_timeout_seconds)
            ):
                self.get_logger().warn(
                    "Task 2: bridge side-view likely during turn; recovering on road."
                )
                self._reset_bridge_turn_controller()
                self._set_state(MissionState.TASK2_SIDE_VIEW_RECOVERY)
                return

        delta_x = self._task2_entry_delta(bridge) if bridge is not None else None
        if delta_x is None and bridge is not None:
            delta_x = self._task2_bridge_rough_target_delta(bridge)

        if fresh and delta_x is not None:
            self._task2_evaluate_turn_pulse(delta_x)
            frontal_base_ready, _ = self._task2_frontal_bridge_base_ready_for_ascent(
                bridge, delta_x=delta_x
            )
            if frontal_base_ready:
                bear_action, _ = self._task2_bridge_bear_turn_action(
                    require_surface=True
                )
                if bear_action is not None and bear_action != "STOP":
                    self.task2_turn_centered_frames = 0
                    self._publish_action(bear_action)
                    return

            if abs(delta_x) <= self.task2_turn_visual_deadband_pixels:
                self.task2_turn_centered_frames += 1
                self._publish_action("STOP")
                if (
                    self.task2_turn_centered_frames
                    >= self.task2_turn_center_confirm_frames
                    and self._elapsed_seconds(self.task2_turn_state_start_time)
                    >= self.task2_turn_min_state_seconds
                ):
                    entry_source = self._task2_entry_source(bridge)
                    if entry_source != "none":
                        self.get_logger().info(
                            "Task 2: bridge turn centered with fresh frames; approaching entry."
                        )
                        self._reset_bridge_turn_controller()
                        self._set_state(MissionState.TASK2_APPROACH_BRIDGE_ENTRY)
                    else:
                        frontal_ready, frontal_reason = (
                            self._task2_frontal_bridge_ready_for_ascent(
                                bridge, delta_x=delta_x
                            )
                        )
                        if frontal_ready:
                            self.get_logger().info(
                                "Task 2: bridge is frontal, low in frame, and bear-centered; "
                                "ascending without road-contact entry."
                            )
                            self._log_event(
                                "info",
                                "task2_frontal_bridge_ascent_bypass",
                                reason=frontal_reason,
                            )
                            self._reset_bridge_turn_controller()
                            self._set_state(MissionState.TASK2_ASCEND_BRIDGE)
                return
            self.task2_turn_centered_frames = 0
            action = self._task2_turn_pulse_action(delta_x)
            self._publish_action(action)
            return

        if bridge is not None and delta_x is not None:
            if (
                self.task2_turn_state_start_time is not None
                and self._elapsed_seconds(self.task2_turn_state_start_time)
                <= self.task2_turn_max_cached_control_seconds
            ):
                self._publish_action(self._turn_action_from_error(delta_x))
                return
            self._publish_action("STOP")
            return

        bearing_error = self._bridge_landmark_bearing_error()
        if bearing_error is not None:
            if isinstance(bearing_error, float) and abs(bearing_error) <= self.task2_turn_map_yaw_tolerance:
                self._publish_action("STOP")
                return
            if abs(bearing_error) == 1.0:
                self._publish_action(
                    "CLOCKWISE_ROTATION_SLOW"
                    if bearing_error > 0.0
                    else "COUNTERCLOCKWISE_ROTATION_SLOW"
                )
            else:
                self._publish_action(
                    "COUNTERCLOCKWISE_ROTATION_SLOW"
                    if bearing_error > 0.0
                    else "CLOCKWISE_ROTATION_SLOW"
                )
            return

        if (
            self.task2_turn_state_start_time is not None
            and self._elapsed_seconds(self.task2_turn_state_start_time)
            < self.task2_bridge_tracking_expire
        ):
            self._publish_action("STOP")
            return

        self.get_logger().warn(
            "Task 2: bridge track expired during turn; returning to bounded search."
        )
        self._reset_bridge_turn_controller()
        self._set_state(MissionState.TASK2_SEARCH_BRIDGE)

    def _task2_side_view_recovery(self):
        if self.task2_side_view_recovery_start_time is None:
            self.task2_side_view_recovery_start_time = self.get_clock().now()
            bridge = self._task2_bridge_visible(allow_cached=True)
            delta = BridgeVisionAnalyzer.ramp_delta(self, bridge)
            self.task2_side_view_recovery_direction = 1.0 if (delta or 0.0) >= 0.0 else -1.0
            self.get_logger().warn(
                "Task 2 side-view recovery started: backing up, following road, then rescanning."
            )

        elapsed = self._elapsed_seconds(self.task2_side_view_recovery_start_time)
        if elapsed < self.task2_side_view_backup_seconds:
            self._publish_action("BACKWARD_SLOW")
            return

        road_follow_end = (
            self.task2_side_view_backup_seconds
            + self.task2_side_view_road_follow_seconds
        )
        if elapsed < road_follow_end:
            road_action = self._task2_road_explore_action()
            if road_action != "STOP":
                self._publish_action(road_action)
                return
            self._publish_action(
                "CLOCKWISE_ROTATION_SLOW"
                if self.task2_side_view_recovery_direction > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )
            return

        self.task2_side_view_recovery_start_time = None
        self._reset_task2_bridge_runtime()
        self._set_state(
            MissionState.TASK2_SEARCH_BRIDGE,
            reason="side-view recovery complete",
        )

    def _turn_action_from_error(self, delta_x):
        turn_value = delta_x * self.task2_turn_direction_sign
        return (
            "CLOCKWISE_ROTATION_SLOW"
            if turn_value > 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _task2_turn_pulse_action(self, delta_x):
        now = self.get_clock().now()
        if self.task2_turn_pulse_start_time is not None:
            if (
                self._elapsed_seconds(self.task2_turn_pulse_start_time)
                < self.task2_turn_command_pulse_seconds
            ):
                return self.task2_turn_command_action or self._turn_action_from_error(delta_x)
            self.task2_turn_pulse_start_time = None
            self.task2_turn_settle_start_time = now
            return "STOP"

        if self.task2_turn_settle_start_time is not None:
            if (
                self._elapsed_seconds(self.task2_turn_settle_start_time)
                < self.task2_turn_settle_seconds
            ):
                return "STOP"
            self.task2_turn_settle_start_time = None

        self.task2_turn_error_before_pulse = delta_x
        self.task2_turn_command_action = self._turn_action_from_error(delta_x)
        self.task2_turn_command_direction = 1.0 if delta_x > 0.0 else -1.0
        self.task2_turn_pulse_start_time = now
        return self.task2_turn_command_action

    def _task2_evaluate_turn_pulse(self, current_error):
        previous = self.task2_turn_error_before_pulse
        if previous is None or self.task2_turn_command_action is None:
            self.task2_turn_last_observed_error = current_error
            return

        same_sign = previous * current_error > 0.0
        worse = abs(current_error) > abs(previous) + self.task2_turn_wrong_way_pixel_epsilon
        if same_sign and worse:
            self.task2_turn_wrong_way_count += 1
            self.get_logger().warn(
                "Task 2 turn pulse increased bridge error: "
                f"before={previous:.0f}px, after={current_error:.0f}px, "
                f"wrong_way={self.task2_turn_wrong_way_count}/"
                f"{self.task2_turn_wrong_way_limit}."
            )
            if (
                self.task2_turn_auto_flip_enabled
                and not self.task2_turn_sign_confirmed
                and self.task2_turn_wrong_way_count >= self.task2_turn_wrong_way_limit
            ):
                self.task2_turn_direction_sign *= -1.0
                self.task2_turn_wrong_way_count = 0
                self.get_logger().warn(
                    "Task 2 auto-flipped turn direction sign to "
                    f"{self.task2_turn_direction_sign:+.0f}."
                )
        elif abs(current_error) < abs(previous) - self.task2_turn_wrong_way_pixel_epsilon:
            self.task2_turn_wrong_way_count = 0
            self.task2_turn_sign_confirmed = True

        self.task2_turn_error_before_pulse = None
        self.task2_turn_command_action = None
        self.task2_turn_last_observed_error = current_error

    def _reset_bridge_turn_controller(self):
        self.task2_turn_state_start_time = None
        self.task2_turn_pulse_start_time = None
        self.task2_turn_settle_start_time = None
        self.task2_turn_error_before_pulse = None
        self.task2_turn_command_action = None
        self.task2_turn_centered_frames = 0

    def _task2_approach_bridge_entry(self):
        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()

        if self._budget_exceeded(self.task2_entry_budget_seconds):
            self._publish_action("STOP")
            self.get_logger().warn(
                "Task 2: bridge entry budget exceeded; restarting bounded search."
            )
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        bridge = self._task2_bridge_visible(allow_cached=True)
        if bridge is None:
            action = self._task2_entry_homing_action(None)
            if action != "STOP" or self._bridge_landmark_is_usable():
                self._publish_action(action)
                return
            self._publish_action("STOP")
            self.get_logger().warn("Task 2: bridge lost during approach; searching again.")
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        if self._bridge_side_view_likely(bridge):
            self._publish_action("STOP")
            self.get_logger().warn(
                "Task 2: bridge approach sees side-view ramp; entering recovery."
            )
            self._set_state(MissionState.TASK2_SIDE_VIEW_RECOVERY)
            return

        if not self._task2_bridge_has_entry_candidate(bridge):
            self.get_logger().info(
                "Task 2: bridge visible but lower entry is not usable; homing cautiously."
            )
            self._publish_action(self._task2_entry_homing_action(bridge))
            return

        if self._task2_bridge_entry_close(bridge):
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: bridge entry close and centered; final alignment."
            )
            self._set_state(MissionState.TASK2_FINAL_ALIGN_BRIDGE)
            return

        entry_score = self._task2_bridge_entry_score(bridge)
        self._task2_update_bridge_entry_progress(bridge, entry_score)

        if (
            self._elapsed_seconds(self.task2_phase_start_time)
            >= self.task2_bridge_approach_timeout_seconds
        ):
            self.get_logger().warn(
                "Task 2: bridge approach timeout; restarting bounded search."
            )
            self._publish_action("STOP")
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        if (
            self.task2_bridge_best_entry_time is not None
            and self._elapsed_seconds(self.task2_bridge_best_entry_time)
            >= self.task2_bridge_no_entry_progress_timeout_seconds
        ):
            self.get_logger().warn("Task 2: no entry progress; using landmark homing.")
            self._publish_action(self._task2_entry_homing_action(bridge))
            return

        self._publish_action(self._task2_entry_homing_action(bridge))

    def _task2_entry_homing_action(self, bridge=None):
        if self._budget_exceeded(self.task2_entry_budget_seconds):
            return "STOP"

        fresh = self._bridge_observation_is_fresh(bridge)
        if bridge is not None and fresh:
            entry_confirmed = self._task2_bridge_entry_confirmed(bridge)
            pre_entry_confirmed = self._task2_bridge_pre_entry_confirmed(bridge)
            delta_x = None
            homing_target = "rough_bridge_target"
            if pre_entry_confirmed:
                delta_x = self._task2_bridge_pre_entry_delta(bridge)
                homing_target = "pre_entry_staging"
            if delta_x is None and entry_confirmed:
                delta_x = self._task2_bridge_entry_delta(bridge)
                homing_target = "entry_gate"
            if delta_x is None:
                delta_x = self._task2_bridge_rough_target_delta(bridge)
            if delta_x is None:
                return self._task2_entry_blocked_recovery_action(
                    self._action_safety_result(
                        clear=False,
                        blocker_type="unknown",
                        reason="no bridge target delta",
                    ),
                    bridge,
                )

            if abs(delta_x) > self.task2_bridge_approach_hard_tolerance:
                self._reset_task2_entry_blocked_recovery()
                return self._turn_action_from_error(delta_x)
            if abs(delta_x) > self.task2_bridge_entry_center_tolerance:
                self._reset_task2_entry_blocked_recovery()
                return "RIGHT_SHIFT" if delta_x > 0.0 else "LEFT_SHIFT"

            if homing_target == "pre_entry_staging":
                safety = self._action_safety_check(
                    "FORWARD_SLOW", context="task2_bridge_entry", bridge=bridge
                )
                if safety["clear"] or safety["blocker_type"] in (
                    "bridge_entry_gate",
                    "bridge_center_corridor",
                ):
                    self._reset_task2_entry_blocked_recovery()
                    return "FORWARD_SLOW"
                return self._task2_entry_blocked_recovery_action(safety, bridge)

            if not entry_confirmed:
                if self._bridge_ramp_is_usable(bridge):
                    safety = self._action_safety_check(
                        "FORWARD_SLOW", context="task2_bridge_entry", bridge=bridge
                    )
                    if safety["clear"] or safety["blocker_type"] in (
                        "bridge_entry_gate",
                        "bridge_center_corridor",
                    ):
                        self._reset_task2_entry_blocked_recovery()
                        return "FORWARD_SLOW"
                    return self._task2_entry_blocked_recovery_action(safety, bridge)
                road_action = self._task2_road_explore_action()
                if road_action in ("FORWARD", "FORWARD_SLOW"):
                    return "RIGHT_FRONT" if self.task2_bridge_last_delta_sign >= 0.0 else "LEFT_FRONT"
                if road_action == "STOP":
                    return self._turn_action_from_error(
                        delta_x if abs(delta_x) > 4.0 else self.task2_bridge_last_delta_sign
                    )
                return road_action

            safety = self._action_safety_check(
                "FORWARD_SLOW", context="task2_bridge_entry", bridge=bridge
            )
            if safety["clear"] or safety["blocker_type"] in (
                "bridge_entry_gate",
                "bridge_center_corridor",
            ):
                self._reset_task2_entry_blocked_recovery()
                return "FORWARD_SLOW"

            blocker_type = safety["blocker_type"]
            blocked_side = safety["blocker_side"]
            self._log_event(
                "warn",
                "bridge_entry_forward_blocked",
                reason=safety["reason"],
                blocker_type=blocker_type,
                blocker_side=blocked_side,
                blocker_distance=safety["blocker_distance"],
                blocker_x=safety["blocker_x"],
                blocker_y=safety["blocker_y"],
            )
            if blocker_type == "bridge_left_side" or blocked_side in ("left", "front_left"):
                self._log_event("warn", "bridge_side_blocks_entry", **safety)
                return "RIGHT_FRONT"
            if blocker_type == "bridge_right_side" or blocked_side in ("right", "front_right"):
                self._log_event("warn", "bridge_side_blocks_entry", **safety)
                return "LEFT_FRONT"
            if blocker_type == "virtual_obstacle":
                self._log_event("warn", "virtual_obstacle_blocks_entry", **safety)
            if blocker_type == "static_map":
                self._log_event("warn", "static_map_blocks_entry", **safety)
            return self._task2_entry_blocked_recovery_action(safety, bridge)

        bearing_error = self._bridge_landmark_bearing_error()
        pre_entry_bearing = self._bridge_pre_entry_bearing_error()
        if pre_entry_bearing is not None:
            if abs(pre_entry_bearing) <= self.task2_turn_map_yaw_tolerance:
                safety = self._action_safety_check("FORWARD_SLOW", context="task2_bridge_entry", bridge=bridge)
                if safety["clear"]:
                    return "FORWARD_SLOW"
                return self._task2_entry_blocked_recovery_action(safety, bridge)
            return self._turn_action_from_bearing_error(pre_entry_bearing)
        if bearing_error is not None:
            return self._turn_action_from_bearing_error(bearing_error)
        return self._task2_entry_blocked_recovery_action(
            self._action_safety_result(
                clear=False,
                blocker_type="unknown",
                reason="no fresh bridge and no usable landmark bearing",
            ),
            bridge,
        )

    def _task2_entry_blocked_recovery_action(self, safety_result, bridge=None):
        now = self.get_clock().now()
        reason = safety_result.get("reason", "blocked")
        if self.task2_entry_blocked_start_time is None:
            self.task2_entry_blocked_start_time = now
            self.task2_entry_recovery_phase = "stop"
            self.task2_entry_recovery_phase_start_time = now
            self.task2_entry_blocked_count += 1
            self.task2_entry_blocked_last_reason = reason

        if self.task2_entry_blocked_count > self.task2_entry_unknown_block_max_retries:
            self._reset_task2_entry_blocked_recovery()
            self._set_state(
                MissionState.TASK2_SEARCH_BRIDGE,
                reason="bridge entry blocked recovery retries exceeded",
            )
            return "STOP"

        phase = self.task2_entry_recovery_phase or "stop"
        elapsed = self._elapsed_seconds(self.task2_entry_recovery_phase_start_time)
        if phase == "stop":
            action = "STOP"
            if elapsed >= self.task2_entry_unknown_block_stop_seconds:
                self.task2_entry_recovery_phase = "backup"
                self.task2_entry_recovery_phase_start_time = now
                action = "BACKWARD_SLOW"
        elif phase == "backup":
            action = "BACKWARD_SLOW"
            if elapsed >= self.task2_entry_unknown_block_backup_seconds:
                self.task2_entry_recovery_phase = "turn"
                self.task2_entry_recovery_phase_start_time = now
                action = self._task2_entry_recovery_turn_action(bridge)
        else:
            action = self._task2_entry_recovery_turn_action(bridge)
            if elapsed >= self.task2_entry_unknown_block_turn_seconds:
                self.task2_entry_recovery_phase = None
                self.task2_entry_recovery_phase_start_time = None
                self.task2_entry_blocked_start_time = None

        self._log_event(
            "warn",
            "bridge_entry_blocked_recovery",
            blocker_type=safety_result.get("blocker_type", "unknown"),
            blocker_side=safety_result.get("blocker_side", "unknown"),
            blocker_distance=safety_result.get("blocker_distance", 0.0),
            recovery_phase=phase,
            retry_count=self.task2_entry_blocked_count,
            chosen_action=action,
            reason=reason,
        )
        return action

    def _task2_entry_recovery_turn_action(self, bridge=None):
        delta = self._task2_bridge_entry_delta(bridge)
        if delta is None:
            delta = self._task2_bridge_rough_target_delta(bridge)
        if delta is None:
            delta = self.task2_bridge_last_delta_sign
        return (
            "CLOCKWISE_ROTATION_SLOW"
            if delta > 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _reset_task2_entry_blocked_recovery(self):
        self.task2_entry_blocked_count = 0
        self.task2_entry_blocked_start_time = None
        self.task2_entry_recovery_phase = None
        self.task2_entry_recovery_phase_start_time = None
        self.task2_entry_blocked_last_reason = ""

    def _bridge_forward_blocked_side(self):
        if self.pose is None:
            return "unknown"
        projection = self._action_projection("FORWARD_SLOW")
        if projection is None:
            return "unknown"
        _, distance = projection
        sample_x = self.pose[0] + math.cos(self.pose[2]) * distance
        sample_y = self.pose[1] + math.sin(self.pose[2]) * distance
        nearest = None
        nearest_side = "unknown"
        for key, side in (
            ("left_side_map_points", "left"),
            ("right_side_map_points", "right"),
        ):
            if not self._bridge_side_has_enough_observations(key):
                continue
            for point in self.bridge_landmark.get(key, []):
                if not self._committed_bridge_side_point(point, key):
                    continue
                d = distance_2d((sample_x, sample_y), point[:2])
                if nearest is None or d < nearest:
                    nearest = d
                    nearest_side = side
        return nearest_side

    def _task2_final_align_bridge(self):
        bridge = self._task2_bridge_visible(allow_cached=True)
        self._update_bridge_bear_memory()
        if bridge is None:
            if self.task2_final_align_loss_start_time is None:
                self.task2_final_align_loss_start_time = self.get_clock().now()
            bearing_error = self._bridge_landmark_bearing_error()
            if (
                bearing_error is not None
                and self._elapsed_seconds(self.task2_final_align_loss_start_time)
                <= self.task2_final_align_loss_timeout
            ):
                self._publish_action(self._turn_action_from_bearing_error(bearing_error))
                return
            if self._elapsed_seconds(self.task2_final_align_loss_start_time) <= self.task2_final_align_loss_timeout:
                self._publish_action("STOP")
                return
            self._publish_action("STOP")
            self.get_logger().warn(
                "Task 2: bridge lost during final alignment timeout; approaching again."
            )
            self._set_state(MissionState.TASK2_APPROACH_BRIDGE_ENTRY)
            return
        self.task2_final_align_loss_start_time = None

        close = self._task2_bridge_entry_close(bridge)
        if not close:
            self._publish_action("STOP")
            if self.task2_final_align_close_loss_start_time is None:
                self.task2_final_align_close_loss_start_time = self.get_clock().now()
            if (
                self._elapsed_seconds(self.task2_final_align_close_loss_start_time)
                >= self.task2_final_align_close_hysteresis
            ):
                self.get_logger().info(
                    "Task 2: final align close criteria lost; approaching again "
                    f"({self.task2_entry_close_last_reason})."
                )
                self._set_state(MissionState.TASK2_APPROACH_BRIDGE_ENTRY)
            return
        self.task2_final_align_close_loss_start_time = None

        if not self._bridge_observation_is_fresh(bridge):
            self._publish_action("STOP")
            return

        entry_source = self._task2_entry_source(bridge)
        if entry_source == "none":
            self._task2_reset_bridge_confirm()
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: final alignment waiting for road-contact or ramp-fallback entry."
            )
            return

        delta_x = (
            self._task2_bridge_entry_delta(bridge)
            if entry_source == "road_contact"
            else BridgeVisionAnalyzer.ramp_delta(self, bridge)
        )
        if delta_x is None:
            self._task2_reset_bridge_confirm()
            self._publish_action("STOP")
            return

        if abs(delta_x) <= self.task2_bridge_entry_final_tolerance:
            self.task2_final_align_confirm_count += 1
            self._publish_action("STOP")
            if self.task2_final_align_confirm_count >= self.task2_final_align_confirm_frames:
                self._publish_action("STOP")
                self.get_logger().info(
                    "Task 2: final entry alignment confirmed; starting ascent."
                )
                self._set_state(MissionState.TASK2_ASCEND_BRIDGE)
            return

        self.task2_final_align_confirm_count = 0
        self._publish_action(self._turn_action_from_error(delta_x))

    def _turn_action_from_bearing_error(self, bearing_error):
        if bearing_error is None:
            return "STOP"
        if abs(bearing_error) == 1.0:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if bearing_error > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )
        if abs(bearing_error) <= self.task2_turn_map_yaw_tolerance:
            return "STOP"
        return (
            "COUNTERCLOCKWISE_ROTATION_SLOW"
            if bearing_error > 0.0
            else "CLOCKWISE_ROTATION_SLOW"
        )

    def _task2_ascend_bridge(self):
        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()
            self.task2_ascent_stop_start_time = None
            self.task2_ascent_lost_bridge_start_time = None
            self.task2_ascent_start_z = self.pose_z
            self.bridge_top_confirm_count = 0
            self.bridge_top_confirmed = False
            self.get_logger().info(
                "Task 2 ascent: climbing until the bridge mask leaves the camera frame."
            )

        if self.task2_ascent_stop_start_time is not None:
            self._publish_action("STOP")
            if (
                self._elapsed_seconds(self.task2_ascent_stop_start_time)
                >= self.task2_ascent_stop_settle_seconds
            ):
                self.task2_ascent_completed = True
                self.bridge_top_confirmed = True
                self.task2_phase_start_time = None
                self.task2_ascent_stop_start_time = None
                self._log_event("info", "task2_ascent_completed")
                self._set_state(
                    MissionState.TASK2_SEARCH_BRIDGE_BEAR,
                    reason="ascent stop-settle complete",
            )
            return

        self._update_bridge_bear_memory()

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        bridge = self._task2_bridge_visible(allow_cached=False)
        bridge_mask_visible = bridge is not None and self._bridge_observation_is_fresh(bridge)

        top_ok, top_score, top_reason = self._bridge_top_confidence()
        self.bridge_top_visual_confidence = top_score
        self.bridge_top_confidence_reason = (
            "bridge mask visible; ascent continues"
            if bridge_mask_visible
            else f"bridge mask lost; {top_reason}"
        )

        if not bridge_mask_visible:
            self._reset_task2_ascent_bear_pid()
            self._publish_action("STOP")
            if self.task2_ascent_lost_bridge_start_time is None:
                self.task2_ascent_lost_bridge_start_time = self.get_clock().now()
                self._log_event(
                    "info",
                    "task2_ascent_bridge_mask_lost",
                    elapsed=elapsed,
                    top_confidence=top_ok,
                    top_score=top_score,
                    reason=top_reason,
                )
                return
            if (
                self._elapsed_seconds(self.task2_ascent_lost_bridge_start_time)
                >= self.task2_ascent_stop_on_bridge_loss_seconds
            ):
                self._log_event(
                    "info",
                    "task2_ascent_bridge_mask_loss_gate",
                    top_confidence=top_ok,
                    top_score=top_score,
                    reason=top_reason,
                    elapsed=elapsed,
                    pose_z=self.pose_z,
                    start_pose_z=self.start_pose_z or 0.0,
                )
                self.bridge_top_confirmed = True
                self._start_task2_ascent_settle()
                return
            return

        self.task2_ascent_lost_bridge_start_time = None

        self._publish_ascent_or_action(AscentController.action(self, bridge))

    def _start_task2_ascent_settle(self):
        self._publish_action("STOP")
        if self.task2_ascent_stop_start_time is None:
            self.task2_ascent_stop_start_time = self.get_clock().now()
            self.get_logger().info(
                "Task 2 ascent complete; stopping before bridge-top bear search."
            )

    def _publish_ascent_or_action(self, action_key):
        if action_key == "ASCEND_FORWARD":
            self._publish_ascent_forward()
        elif action_key == "ASCEND_BEAR_PID":
            self._publish_ascent_bear_pid()
        else:
            self._publish_action(action_key)

    def _publish_ascent_bear_pid(self):
        recovery_action = self._apply_stuck_recovery("FORWARD")
        if recovery_action != "FORWARD":
            self._reset_task2_ascent_bear_pid()
            self._publish_action(recovery_action)
            return

        delta, reason = self._task2_bridge_bear_delta_for_ascent(require_surface=True)
        if delta is None:
            self._reset_task2_ascent_bear_pid()
            self._publish_ascent_forward()
            return

        now = self.get_clock().now()
        now_sec = now.nanoseconds / 1e9
        if self.task2_ascent_bear_pid_last_time is None:
            dt = 0.0
            derivative = 0.0
        else:
            dt = max(1e-3, now_sec - self.task2_ascent_bear_pid_last_time)
            derivative = (
                float(delta) - float(self.task2_ascent_bear_pid_last_error)
            ) / dt
        if dt > 0.0:
            self.task2_ascent_bear_pid_integral += float(delta) * dt
            integral_limit = abs(self.task2_ascent_bear_pid_integral_limit)
            self.task2_ascent_bear_pid_integral = max(
                -integral_limit,
                min(integral_limit, self.task2_ascent_bear_pid_integral),
            )
        self.task2_ascent_bear_pid_last_error = float(delta)
        self.task2_ascent_bear_pid_last_time = now_sec

        slow = ACTION_MAPPINGS.get("FORWARD_SLOW", [0.0, 0.0, 0.0, 0.0])
        fast = ACTION_MAPPINGS.get("FORWARD", slow)
        max_wheel_speed = max(abs(float(value)) for value in fast + slow) or 1.0
        base_speed = min(
            abs(float(slow[0]))
            * self.task2_ascent_forward_speed_scale
            * self.task2_ascent_bear_pid_forward_scale,
            max_wheel_speed,
        )
        if abs(delta) >= self.task2_ascent_bear_pid_rotate_only_pixels:
            base_speed = 0.0

        max_turn = min(abs(self.task2_ascent_bear_pid_max_turn), max_wheel_speed)
        pid_turn = (
            self.task2_ascent_bear_pid_kp * float(delta)
            + self.task2_ascent_bear_pid_ki * self.task2_ascent_bear_pid_integral
            + self.task2_ascent_bear_pid_kd * derivative
        )
        turn_speed = max(
            -max_turn,
            min(max_turn, pid_turn),
        )
        velocities = [
            base_speed + turn_speed,
            base_speed - turn_speed,
            base_speed + turn_speed,
            base_speed - turn_speed,
        ]
        velocities = [
            max(-max_wheel_speed, min(max_wheel_speed, float(value)))
            for value in velocities
        ]

        rear_msg = Float32MultiArray()
        rear_msg.data = [velocities[0], velocities[1]]
        self.rear_pub.publish(rear_msg)

        front_msg = Float32MultiArray()
        front_msg.data = [velocities[2], velocities[3]]
        self.front_pub.publish(front_msg)

        if (
            self.last_logged_action != "ASCEND_BEAR_PID"
            or self.last_action_log_time is None
            or self._elapsed_seconds(self.last_action_log_time) >= 0.5
        ):
            self.last_logged_action = "ASCEND_BEAR_PID"
            self.last_action_log_time = now
            self._log_event(
                "info",
                "ascent_bear_pid_action",
                action="ASCEND_BEAR_PID",
                bear_delta_x=delta,
                turn_speed=turn_speed,
                base_speed=base_speed,
                pid_integral=self.task2_ascent_bear_pid_integral,
                pid_derivative=derivative,
                reason=reason,
            )

    def _publish_ascent_forward(self):
        recovery_action = self._apply_stuck_recovery("FORWARD")
        if recovery_action != "FORWARD":
            self._publish_action(recovery_action)
            return
        slow = ACTION_MAPPINGS.get("FORWARD_SLOW", [0.0, 0.0, 0.0, 0.0])
        fast = ACTION_MAPPINGS.get("FORWARD", slow)
        velocities = [
            max(min(float(s) * self.task2_ascent_forward_speed_scale, float(f)), -abs(float(f)))
            for s, f in zip(slow, fast)
        ]
        rear_msg = Float32MultiArray()
        rear_msg.data = [float(velocities[0]), float(velocities[1])]
        self.rear_pub.publish(rear_msg)

        front_msg = Float32MultiArray()
        front_msg.data = [float(velocities[2]), float(velocities[3])]
        self.front_pub.publish(front_msg)
        now = self.get_clock().now()
        if (
            self.last_logged_action != "ASCEND_FORWARD"
            or self.last_action_log_time is None
            or self._elapsed_seconds(self.last_action_log_time) >= 1.0
        ):
            self.last_logged_action = "ASCEND_FORWARD"
            self.last_action_log_time = now
            self._log_event(
                "info",
                "ascent_action",
                action="ASCEND_FORWARD",
                scale=self.task2_ascent_forward_speed_scale,
                pose_z=self.pose_z,
                top_confidence=self.bridge_top_confirmed,
                top_confirm_frames=self.bridge_top_confirm_count,
            )

    def _bridge_top_confidence(self):
        elapsed = (
            self._elapsed_seconds(self.task2_phase_start_time)
            if self.task2_phase_start_time is not None
            else 0.0
        )
        start_z = (
            self.task2_ascent_start_z
            if self.task2_ascent_start_z is not None
            else self.start_pose_z
        )
        z_delta = self.pose_z - float(start_z or 0.0)
        if self.task2_top_use_tf_z and z_delta >= self.task2_top_z_threshold:
            return True, 1.0, f"z threshold reached ({z_delta:.2f}m)"

        visual_score = self._bridge_top_visual_score()
        if (
            elapsed >= self.task2_top_min_ascent_seconds
            and visual_score >= self.task2_top_visual_confidence_threshold
        ):
            return True, visual_score, "visual top confidence reached"

        if elapsed >= self.task2_ascent_timeout + max(0.0, self.task2_ascent_max_extra_seconds):
            return True, max(visual_score, 0.55), "max ascent time reached"

        return False, visual_score, f"top confidence waiting (z_delta={z_delta:.2f}m)"

    def _bridge_top_visual_score(self):
        bridge = self._task2_bridge_visible(allow_cached=False)
        score = 0.0
        if bridge is None:
            score = 0.60
        else:
            ramp_conf = float(bridge.get("ramp_confidence", 0.0))
            bottom_y = float(bridge.get("bottom_y_ratio", 0.0))
            side_score = float(bridge.get("side_view_score", 1.0))
            if bottom_y < 0.58:
                score += 0.35
            if ramp_conf < 0.35:
                score += 0.20
            if side_score < 0.45:
                score += 0.15
        if self._target_visible():
            score += 0.30
        return min(1.0, score)

    def _task2_search_bridge_bear(self):
        self.next_state_after_grab = MissionState.TASK2_DESCEND_BRIDGE
        if not self.task2_ascent_completed or not self.bridge_top_confirmed:
            self._publish_action("STOP")
            self.get_logger().warn(
                "Task 2 bridge-bear search requested before ascent/top confirmation; "
                "returning to ascent."
            )
            self._set_state(MissionState.TASK2_ASCEND_BRIDGE)
            return
        self._update_bridge_bear_memory()
        target_ready = self._target_visible() and (
            not self.task2_top_bear_require_visible_bbox or self._bbox_visible()
        )
        if target_ready:
            if (
                self.task2_relax_bridge_surface_after_top
                or self._target_on_bridge_surface()
            ):
                self._publish_action("STOP")
                self.bear_context = "task2_bridge_bear"
                self.task2_phase_start_time = None
                self._log_event(
                    "info",
                    "bridge_bear_accepted",
                    reason="top confirmed and target visible",
                    bridge_bear_memory_valid=self.bridge_bear_memory.valid,
                )
                self._set_state(MissionState.APPROACH_BEAR)
                return
            surface_candidate, _ = self._target_surface_candidate()
            if surface_candidate:
                self._publish_action("STOP")
                return
            self._log_target_surface_wait("target rejected because it is not on bridge")
        elif self._target_visible():
            self._log_target_surface_wait("target visible but bbox is not fresh")

        if (
            self.task2_top_bear_allow_cached_memory
            and self.bridge_bear_memory.recent(
                self,
                self.task2_bridge_bear_memory_ttl_seconds,
                self.task2_bridge_bear_memory_min_confidence,
            )
        ):
            action = self._cached_bridge_bear_search_action()
            self._publish_action(action)
            return

        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        if elapsed >= min(
            self.task2_top_search_timeout_seconds,
            self.task2_top_bear_search_budget_seconds
            if self.task2_top_bear_search_budget_seconds > 0.0
            else self.task2_top_search_timeout_seconds,
        ):
            self._publish_action("STOP")
            self.get_logger().warn(
                "Task 2: bear not found on bridge top; restarting rotation search."
            )
            self.task2_phase_start_time = self.get_clock().now()
            return

        rotate_window = max(0.5, self.task2_top_search_rotate_seconds)
        rotate_window = max(0.5, self.task2_top_search_turn_direction_switch_seconds)
        phase = elapsed % (rotate_window * 2.0)
        if phase < rotate_window:
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
        elif phase < rotate_window * 2.0:
            self._publish_action("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif (
            self.task2_top_search_allow_forward
            and elapsed < self.task2_top_search_max_forward_seconds
        ):
            self._publish_action("FORWARD_SLOW")
        else:
            self._publish_action("STOP")

    def _cached_bridge_bear_search_action(self):
        if not self.task2_top_search_use_cached_bear_direction:
            return "CLOCKWISE_ROTATION_SLOW"
        bbox = self.bridge_bear_memory.bbox or {}
        image_width = float(bbox.get("image_width", 0.0))
        center_x = float(bbox.get("center_x", 0.0))
        if image_width <= 0.0 or center_x <= 0.0:
            return "CLOCKWISE_ROTATION_SLOW"
        delta = center_x - image_width * 0.5
        if abs(delta) <= self.task2_bridge_entry_center_tolerance:
            elapsed = (
                self._elapsed_seconds(self.task2_phase_start_time)
                if self.task2_phase_start_time is not None
                else 999.0
            )
            if (
                self.task2_top_search_allow_short_recenter
                and elapsed <= self.task2_top_search_short_recenter_seconds
            ):
                return "FORWARD_SLOW"
            return "STOP"
        return self._turn_action_from_error(delta)

    def _update_bridge_bear_memory(self):
        if self.state not in (
            MissionState.TASK2_FINAL_ALIGN_BRIDGE,
            MissionState.TASK2_ASCEND_BRIDGE,
            MissionState.TASK2_SEARCH_BRIDGE_BEAR,
        ):
            return
        self.bridge_bear_memory.clear_if_expired(
            self, self.task2_bridge_bear_memory_ttl_seconds
        )
        if not self._target_visible():
            return
        if self.task2_top_bear_require_visible_bbox and not self._bbox_visible():
            return
        surface_candidate, _ = self._target_surface_candidate()
        confidence = 0.0
        if self.yolo_bbox is not None:
            confidence = max(confidence, float(self.yolo_bbox.get("confidence", 0.0)))
        if self.target_surface_info is not None:
            confidence = max(
                confidence,
                float(self.target_surface_info.get("bbox_bridge_overlap_ratio", 0.0)),
                float(self.target_surface_info.get("bbox_lower_half_bridge_overlap_ratio", 0.0)),
                float(self.target_surface_info.get("target_side_bridge_contact_ratio", 0.0)),
            )
        if surface_candidate or self.bridge_top_confirmed or confidence >= self.task2_bridge_bear_memory_min_confidence:
            self.bridge_bear_memory.update(self, confidence)
            self._log_event(
                "info",
                "bridge_bear_memory_update",
                confidence=confidence,
                seen_state=self.bridge_bear_memory.seen_state,
            )

    def _task2_descend_bridge(self):
        if (
            self.task2_require_bear_secured_before_descent
            and not self.bear_secured
        ):
            self._publish_action("STOP")
            self.get_logger().error(
                "Task 2 descent blocked because bear_secured is False; "
                "returning to bridge-top bear search."
            )
            self._set_state(
                MissionState.TASK2_SEARCH_BRIDGE_BEAR,
                reason="bear not secured before descent",
            )
            return

        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()
            self._clear_navigation()
            self.get_logger().info("Task 2 descent: timed bridge descent.")

        if self.task2_use_bridge_exit_pose:
            if self._goal_reached(self.task2_bridge_exit_pose):
                self._publish_action("STOP")
                self._clear_navigation()
                self.task2_descent_completed = True
                self.task2_phase_start_time = None
                self.next_state_after_grab = MissionState.RETURN_START
                self._set_state(MissionState.RETURN_START)
                return
            self._navigate_state(
                goal_name="task2_bridge_exit_pose",
                goal=self.task2_bridge_exit_pose,
                next_state=MissionState.RETURN_START,
            )
            return

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        descent_limit = min(
            self.task2_descent_min_seconds,
            self.task2_descent_budget_seconds
            if self.task2_descent_budget_seconds > 0.0
            else self.task2_descent_min_seconds,
        )
        if elapsed >= descent_limit or self._budget_exceeded(self.task2_descent_budget_seconds):
            self._publish_action("STOP")
            self.task2_descent_completed = True
            self.task2_phase_start_time = None
            self.next_state_after_grab = MissionState.RETURN_START
            self._set_state(MissionState.RETURN_START)
            return

        bridge = self._task2_bridge_visible()
        if bridge is None:
            self._publish_action("FORWARD_SLOW")
            return

        corridor_action = self._task2_bridge_corridor_action(
            bridge=bridge, mode="middle", centered_action="FORWARD_SLOW"
        )
        if corridor_action is not None:
            self._publish_action(corridor_action)
            return

        self._publish_action(
            self._task2_bridge_center_action(
                bridge=bridge,
                preferred_key="mid_center_x",
                hard_tolerance=self.task2_bridge_approach_hard_tolerance,
                soft_tolerance=self.task2_bridge_approach_soft_tolerance,
                centered_action="FORWARD_SLOW",
                allow_arc=True,
            )
        )

    def _finish_grab_sequence(self):
        if self.verify_grab_enabled:
            self._reset_verify_grab()
            self._set_state(MissionState.VERIFY_GRAB)
        else:
            self.bear_secured = True
            self._log_event("info", "bear_secured_without_verify")
            self._set_state(self.next_state_after_grab)

    def _verify_grab(self):
        self._publish_action("STOP")

        if self.verify_grab_start_time is None:
            self.verify_grab_start_time = self.get_clock().now()
            self.verify_grab_seen_start_time = None
            self.verify_grab_last_log_time = None
            if self.enable_grab_sequence:
                self._publish_arm_positions(self.arm_carry_positions)
            self.get_logger().info("Verifying bear is secured using YOLO bbox.")

        valid, reason = self._grab_bbox_indicates_bear_held()
        if valid:
            if self.verify_grab_seen_start_time is None:
                self.verify_grab_seen_start_time = self.get_clock().now()
                self.get_logger().info("Bear hold verification candidate detected.")
                return
            if self._elapsed_seconds(self.verify_grab_seen_start_time) >= self.verify_grab_seconds:
                self.get_logger().info(
                    f"Bear hold verified; next state -> {self.next_state_after_grab.value}."
                )
                self.bear_secured = True
                if self.bear_context == "task2_bridge_bear":
                    self.task1_observe_completed = True
                    self._log_event(
                        "info",
                        "shared_bridge_bear_verified",
                        task1_observe_completed=self.task1_observe_completed,
                        bear_secured=self.bear_secured,
                    )
                else:
                    self._log_event("info", "bear_secured", bear_secured=True)
                self.grab_retry_count = 0
                self._reset_verify_grab()
                self._set_state(self.next_state_after_grab)
            return

        self.verify_grab_seen_start_time = None
        self._log_grab_verification_wait(reason)
        if self._elapsed_seconds(self.verify_grab_start_time) >= self.verify_grab_timeout:
            self._handle_grab_verification_failure(reason)

    def _grab_bbox_indicates_bear_held(self, require_lift_evidence=True):
        if not self._bbox_visible():
            return False, "no fresh bear bbox"

        bbox = self.yolo_bbox
        image_width = bbox["image_width"]
        image_height = bbox["image_height"]
        if image_width <= 0.0 or image_height <= 0.0:
            return False, "invalid image size in bbox"

        center_x_ratio = bbox["center_x"] / image_width
        center_y_ratio = bbox["center_y"] / image_height
        if not (
            self.verify_grab_min_center_x_ratio
            <= center_x_ratio
            <= self.verify_grab_max_center_x_ratio
        ):
            return False, f"bbox x ratio {center_x_ratio:.2f} outside gripper ROI"
        if not (
            self.verify_grab_min_center_y_ratio
            <= center_y_ratio
            <= self.verify_grab_max_center_y_ratio
        ):
            return False, f"bbox y ratio {center_y_ratio:.2f} outside gripper ROI"

        close_enough = 0.0 < bbox["distance"] <= self.verify_grab_max_depth
        large_enough = bbox["area_ratio"] >= self.verify_grab_min_bbox_area_ratio
        if not (close_enough or large_enough):
            return (
                False,
                "bbox is neither close nor large enough "
                f"(depth={bbox['distance']:.2f}, area={bbox['area_ratio']:.3f})",
            )

        if (
            require_lift_evidence
            and self.verify_grab_require_lift_evidence
            and self.pre_grab_bbox is not None
        ):
            moved_up = (
                bbox["center_y"]
                <= self.pre_grab_bbox["center_y"] - self.verify_grab_lift_pixel_delta
            )
            grew = bbox["area"] >= (
                self.pre_grab_bbox["area"] * self.verify_grab_min_area_growth
            )
            if not (moved_up or grew):
                return False, "bbox did not move upward or grow after lift"

        return True, "bbox matches held-bear criteria"

    def _check_return_hold(self):
        if self.current_task == 2:
            self._reset_return_hold_monitor()
            return True

        if not self.return_hold_verify_enabled:
            self._reset_return_hold_monitor()
            return True

        valid, reason = self._grab_bbox_indicates_bear_held(
            require_lift_evidence=False
        )
        if valid:
            self._reset_return_hold_monitor()
            return True

        now = self.get_clock().now()
        if self.return_hold_loss_start_time is None:
            self.return_hold_loss_start_time = now
            self.return_hold_last_log_time = now
            self.get_logger().warn(
                f"Return hold check lost bear candidate ({reason}); waiting for confirmation."
            )
            return True

        if self._elapsed_seconds(self.return_hold_loss_start_time) < self.return_hold_verify_loss_grace:
            if (
                self.return_hold_last_log_time is None
                or self._elapsed_seconds(self.return_hold_last_log_time)
                >= self.return_hold_verify_log_period
            ):
                self.return_hold_last_log_time = now
                self.get_logger().info(
                    f"Return hold check still waiting ({reason})."
                )
            return True

        self._handle_return_hold_failure(reason)
        return False

    def _handle_return_hold_failure(self, reason):
        self._publish_action("STOP")
        self._clear_navigation()
        self._reset_return_hold_monitor()
        self._reset_verify_grab()
        self._reset_grab_attempt_state()
        self._reset_drop_state()
        self.next_state_after_grab = MissionState.RETURN_START
        if self.enable_grab_sequence:
            self._publish_arm_positions(self.arm_open_positions)
        self.get_logger().warn(
            "Bear appears lost during return_start "
            f"({reason}); reacquiring before another grab attempt."
        )
        self._set_state(
            MissionState.APPROACH_GRAB
            if self._mission_target_visible()
            else MissionState.APPROACH_BEAR
        )

    def _handle_grab_verification_failure(self, reason):
        self.bear_secured = False
        self._reset_verify_grab()
        self._reset_grab_attempt_state()
        if self.enable_grab_sequence:
            self._publish_arm_positions(self.arm_open_positions)

        if self.grab_retry_count < self.verify_grab_retry_limit:
            self.grab_retry_count += 1
            self.get_logger().warn(
                "Bear hold verification failed "
                f"({reason}); retry {self.grab_retry_count}/{self.verify_grab_retry_limit}."
            )
            self._set_state(
                MissionState.APPROACH_GRAB
                if self._mission_target_visible()
                else MissionState.APPROACH_BEAR
            )
            return

        self.get_logger().error(
            f"Bear hold verification failed ({reason}) after all retries; "
            "searching for the bear again instead of returning without it."
        )
        self.grab_retry_count = 0
        self._set_state(MissionState.APPROACH_BEAR)

    def _log_grab_verification_wait(self, reason):
        now = self.get_clock().now()
        if (
            self.verify_grab_last_log_time is not None
            and self._elapsed_seconds(self.verify_grab_last_log_time) < 1.0
        ):
            return
        self.verify_grab_last_log_time = now
        self.get_logger().info(f"Waiting for bear hold verification: {reason}.")

    def _reset_verify_grab(self):
        self.verify_grab_start_time = None
        self.verify_grab_seen_start_time = None
        self.verify_grab_last_log_time = None

    def _reset_return_hold_monitor(self):
        self.return_hold_loss_start_time = None
        self.return_hold_last_log_time = None

    def _reset_grab_attempt_state(self):
        self.grab_approach_start_time = None
        self.grab_approach_open_sent = False
        self.grab_ready_start_time = None
        self.last_grab_target_time = None
        self.last_grab_target_distance = 0.0
        self.last_grab_target_delta_x = 0.0
        self.grab_start_time = None
        self.grab_step_index = 0
        self.grab_step_sent = False
        self.grab_step_deadline = None
        self.pre_grab_bbox = None

    def _segmentation_follow_action(self, prefer_bridge=False):
        segment = self._preferred_drivable_segment(prefer_bridge=prefer_bridge)
        action = self._drivable_follow_action_from_segment(
            segment,
            prefer_bridge=prefer_bridge,
            forward_action="FORWARD_SLOW",
        )
        return self._stabilize_drivable_action(action)

    def _task2_bridge_visible(self, allow_cached=True):
        bridge = self._segmentation_segment("bridge", allow_cached=allow_cached)
        if bridge is None or not bridge.get("usable", bridge.get("found", False)):
            return None
        if float(bridge.get("area_ratio", 0.0)) < self.task2_bridge_detect_min_area_ratio:
            return None
        if bridge.get("raw_found", bridge.get("found", False)):
            self.task2_bridge_last_seen_time = self.get_clock().now()
            self._update_bridge_landmark(bridge)
        return bridge

    def _bridge_observation_is_fresh(self, bridge):
        return bridge is not None and bool(bridge.get("raw_found", bridge.get("found", False)))

    def _new_bridge_landmark(self):
        return {
            "valid": False,
            "confidence": 0.0,
            "observation_count": 0,
            "last_seen_time": None,
            "last_fresh_time": None,
            "last_robot_pose": None,
            "image_center_error": 0.0,
            "entry_pixel_x": 0.0,
            "entry_pixel_y": 0.0,
            "entry_depth": 0.0,
            "entry_map_x": None,
            "entry_map_y": None,
            "pre_entry_map_x": None,
            "pre_entry_map_y": None,
            "estimated_bridge_heading": None,
            "forward_axis": None,
            "left_side_map_points": [],
            "right_side_map_points": [],
            "centerline_map_points": [],
            "left_observation_count": 0,
            "right_observation_count": 0,
            "center_observation_count": 0,
            "position_variance": 999.0,
            "source_is_cached": False,
        }

    def _reset_bridge_landmark(self):
        self.bridge_landmark = self._new_bridge_landmark()
        self.bridge_edge_observations = []
        self.bridge_entry_observations = []
        self.bridge_pre_entry_observations = []
        self.fitted_bridge_side_lines = {"left": [], "right": []}
        self.last_accepted_bridge_side_lines = {"left": [], "right": []}
        self.bridge_left_observation_count = 0
        self.bridge_right_observation_count = 0
        self.bridge_center_observation_count = 0

    def _update_bridge_landmark(self, bridge):
        if bridge is None:
            return
        now = self.get_clock().now()
        landmark = self.bridge_landmark
        landmark["valid"] = True
        landmark["observation_count"] += 1
        landmark["last_seen_time"] = now
        if self._bridge_observation_is_fresh(bridge):
            landmark["last_fresh_time"] = now
        landmark["last_robot_pose"] = tuple(self.pose) if self.pose is not None else None
        landmark["image_center_error"] = self._task2_bridge_delta("entry_u", bridge)
        landmark["entry_pixel_x"] = float(
            bridge.get("entry_u", 0.0) or bridge.get("bottom_center_x", 0.0)
        )
        landmark["entry_pixel_y"] = float(bridge.get("entry_v", 0.0))
        landmark["entry_depth"] = float(bridge.get("entry_depth", 0.0))
        landmark["source_is_cached"] = bool(bridge.get("predicted_or_cached", False))
        landmark["confidence"] = min(
            1.0,
            0.2
            + 0.1 * landmark["observation_count"]
            + 0.35 * float(bridge.get("frontalness", 0.0))
            + 0.25 * float(bridge.get("entry_confidence", 0.0)),
        )

    def _bridge_landmark_is_usable(self):
        if not self.bridge_landmark.get("valid", False):
            return False
        last_seen = self.bridge_landmark.get("last_seen_time")
        if last_seen is None:
            return False
        if self._elapsed_seconds(last_seen) > self.task2_bridge_tracking_expire:
            return False
        return self.bridge_landmark.get("confidence", 0.0) > 0.15

    def _bridge_landmark_bearing_error(self):
        if self.pose is None or not self._bridge_landmark_is_usable():
            return None
        x = self.bridge_landmark.get("entry_map_x")
        y = self.bridge_landmark.get("entry_map_y")
        if x is None or y is None:
            robot_pose = self.bridge_landmark.get("last_robot_pose")
            image_error = float(self.bridge_landmark.get("image_center_error", 0.0))
            if robot_pose is None or abs(image_error) < 1.0:
                return None
            return 1.0 if image_error > 0.0 else -1.0
        target_yaw = math.atan2(y - self.pose[1], x - self.pose[0])
        return normalize_angle(target_yaw - self.pose[2])

    def _task2_bridge_corridor(self, mode="lower", bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible()
        image_width = self._segmentation_image_width()
        if bridge is None or image_width <= 0.0:
            return {"valid": False, "reason": "no fresh bridge segmentation"}

        bottom_left = float(bridge.get("bottom_left_x", 0.0))
        bottom_right = float(bridge.get("bottom_right_x", 0.0))
        mid_left = float(bridge.get("mid_left_x", 0.0))
        mid_right = float(bridge.get("mid_right_x", 0.0))

        if mode == "middle" and mid_left > 0.0 and mid_right > mid_left:
            left_edge = 0.6 * mid_left + 0.4 * bottom_left if bottom_left > 0.0 else mid_left
            right_edge = (
                0.6 * mid_right + 0.4 * bottom_right
                if bottom_right > bottom_left
                else mid_right
            )
        else:
            left_edge = bottom_left
            right_edge = bottom_right

        if right_edge <= left_edge:
            center = self._segment_center_x(bridge, "bottom_center_x")
            width_ratio = float(bridge.get("bottom_width_ratio", 0.0))
            if center <= 0.0 or width_ratio <= 0.0:
                return {"valid": False, "reason": "bridge corridor edges unavailable"}
            half_width = width_ratio * image_width * 0.5
            left_edge = center - half_width
            right_edge = center + half_width

        width_pixels = max(0.0, right_edge - left_edge)
        width_ratio = width_pixels / image_width
        center_x = (left_edge + right_edge) * 0.5
        image_center = image_width * 0.5
        center_error = center_x - image_center
        left_clearance = image_center - left_edge
        right_clearance = right_edge - image_center
        valid = width_ratio >= self.task2_bridge_corridor_min_width_ratio
        reason = "ok" if valid else f"bridge corridor width {width_ratio:.2f} too narrow"
        if valid:
            self.task2_bridge_corridor_last_seen_time = self.get_clock().now()
        return {
            "valid": valid,
            "reason": reason,
            "left_edge_x": left_edge,
            "right_edge_x": right_edge,
            "center_x": center_x,
            "width_pixels": width_pixels,
            "width_ratio": width_ratio,
            "center_error": center_error,
            "left_clearance": left_clearance,
            "right_clearance": right_clearance,
        }

    def _task2_bridge_corridor_action(self, bridge=None, mode="lower", centered_action="FORWARD_SLOW"):
        corridor = self._task2_bridge_corridor(mode=mode, bridge=bridge)
        if not corridor["valid"]:
            if (
                self.task2_bridge_corridor_last_seen_time is not None
                and self._elapsed_seconds(self.task2_bridge_corridor_last_seen_time)
                <= self.task2_bridge_corridor_loss_grace_seconds
            ):
                return "STOP"
            return None

        margin = self.task2_bridge_corridor_side_margin_pixels
        if corridor["left_clearance"] < margin:
            return "RIGHT_SHIFT"
        if corridor["right_clearance"] < margin:
            return "LEFT_SHIFT"

        error = corridor["center_error"]
        if abs(error) > self.task2_bridge_corridor_hard_tolerance:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if error > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )
        if abs(error) > self.task2_bridge_corridor_center_tolerance:
            return "RIGHT_FRONT" if error > 0.0 else "LEFT_FRONT"
        return self._avoid_virtual_obstacle_for_action(centered_action)

    def _task2_bridge_delta(self, preferred_key="bottom_center_x", bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible()
        if bridge is None:
            return None

        if preferred_key in ("final", "bottom_center_x"):
            keys = ("bottom_center_x", "mid_center_x", "center_x")
        elif preferred_key == "mid_center_x":
            keys = ("mid_center_x", "bottom_center_x", "center_x")
        elif preferred_key == "center_x":
            keys = ("center_x", "bottom_center_x", "mid_center_x")
        else:
            keys = (preferred_key, "bottom_center_x", "mid_center_x", "center_x")

        image_width = self._segmentation_image_width()
        for key in keys:
            center_x = float(bridge.get(key, 0.0))
            if center_x > 0.0 and image_width > 0.0:
                delta = center_x - (image_width / 2.0)
                self._task2_remember_bridge_delta(delta)
                return delta

        if "delta_x" in bridge:
            delta = float(bridge.get("delta_x", 0.0))
            self._task2_remember_bridge_delta(delta)
            return delta
        return None

    def _task2_bridge_rough_target_delta(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=True)
        if bridge is None:
            return None
        image_width = self._segmentation_image_width()
        for key in ("target_u", "bottom_center_x", "center_x"):
            value = float(bridge.get(key, 0.0))
            if value > 0.0 and image_width > 0.0:
                delta = value - image_width * 0.5
                self._task2_remember_bridge_delta(delta)
                return delta
        return self._task2_bridge_delta("bottom_center_x", bridge)

    def _task2_bridge_entry_delta(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        if bridge is None or not self._task2_bridge_entry_confirmed(bridge):
            return None
        image_width = self._segmentation_image_width()
        for key in ("entry_gate_center_u", "entry_u"):
            value = float(bridge.get(key, 0.0))
            if value > 0.0 and image_width > 0.0:
                delta = value - image_width * 0.5
                self._task2_remember_bridge_delta(delta)
                return delta
        return None

    def _task2_bridge_pre_entry_delta(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        if bridge is None or not self._task2_bridge_pre_entry_confirmed(bridge):
            return None
        image_width = self._segmentation_image_width()
        value = float(bridge.get("pre_entry_u", 0.0))
        if value > 0.0 and image_width > 0.0:
            delta = value - image_width * 0.5
            self._task2_remember_bridge_delta(delta)
            return delta
        return None

    def _task2_bridge_entry_confirmed(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        if bridge is None:
            return False
        if not self._bridge_observation_is_fresh(bridge):
            return False
        if float(bridge.get("entry_confirmed", 0.0)) >= 0.5:
            return True
        return (
            float(bridge.get("entry_from_road_connection", 0.0)) >= 0.5
            and float(bridge.get("entry_confidence", 0.0)) >= 0.5
            and float(bridge.get("entry_u", 0.0)) > 0.0
        )

    def _task2_bridge_pre_entry_confirmed(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        if bridge is None or not self._bridge_observation_is_fresh(bridge):
            return False
        return (
            float(bridge.get("pre_entry_confidence", 0.0)) > 0.0
            and float(bridge.get("pre_entry_u", 0.0)) > 0.0
        )

    def _bridge_pre_entry_bearing_error(self):
        if self.pose is None:
            return None
        pre_entry = self._bridge_pre_entry_xy()
        if pre_entry is None:
            return None
        target_yaw = math.atan2(pre_entry[1] - self.pose[1], pre_entry[0] - self.pose[0])
        return normalize_angle(target_yaw - self.pose[2])

    def _task2_bridge_center_action(
        self,
        bridge,
        preferred_key,
        hard_tolerance,
        soft_tolerance,
        centered_action,
        allow_arc=True,
    ):
        delta_x = self._task2_bridge_delta(preferred_key, bridge)
        if delta_x is None:
            return "STOP"

        if abs(delta_x) > hard_tolerance:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if delta_x > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )

        if allow_arc and soft_tolerance is not None and abs(delta_x) > soft_tolerance:
            return "RIGHT_FRONT" if delta_x > 0.0 else "LEFT_FRONT"

        return centered_action

    def _task2_bridge_entry_close(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible(allow_cached=False)
        if bridge is None:
            self.task2_entry_close_confirm_count = 0
            return False
        valid, reason = self._task2_bridge_entry_close_detailed(bridge)
        self.task2_entry_close_last_reason = reason
        if valid:
            self.task2_entry_close_confirm_count += 1
        else:
            self.task2_entry_close_confirm_count = 0
        return self.task2_entry_close_confirm_count >= self.task2_entry_close_confirm_frames

    def _task2_bridge_entry_close_detailed(self, bridge):
        if not self._bridge_observation_is_fresh(bridge):
            return False, "bridge observation is cached"

        if (
            self.task2_phase_start_time is not None
            and self._elapsed_seconds(self.task2_phase_start_time)
            < self.task2_approach_min_seconds
        ):
            return False, "approach minimum time has not elapsed"

        entry_source = self._task2_entry_source(bridge)
        if entry_source == "none":
            return False, "bridge entry is neither road-contact nor confirmed ramp fallback"

        pre_entry = self._bridge_pre_entry_xy()
        if entry_source == "road_contact" and pre_entry is not None and self.pose is not None:
            if distance_2d(self.pose[:2], pre_entry) > max(self.goal_tolerance, 0.55):
                return False, "robot has not reached pre-entry staging area"

        bottom_delta = (
            self._task2_bridge_entry_delta(bridge)
            if entry_source == "road_contact"
            else BridgeVisionAnalyzer.ramp_delta(self, bridge)
        )
        if (
            bottom_delta is None
            or abs(bottom_delta)
            > (
                self.task2_bridge_approach_soft_tolerance
                if entry_source == "road_contact"
                else self.task2_ramp_entry_center_tolerance_pixels
            )
        ):
            return False, f"entry not centered enough ({bottom_delta})"

        if not self._task2_bridge_has_entry_candidate(bridge):
            return False, "no valid entry candidate"

        connection = self._bridge_connection()
        from_contact = float(bridge.get("entry_from_road_connection", 0.0)) >= 0.5
        if entry_source == "road_contact" and not from_contact:
            return False, "confirmed entry is not marked as road-bridge contact"
        if entry_source == "road_contact" and connection is not None:
            score = float(connection.get("score", 0.0))
            if connection.get("connected", False) and score < self.task2_entry_close_min_connection_score:
                return False, "road-bridge connection score too low"

        frontalness = float(bridge.get("frontalness", 0.0))
        if frontalness < self.task2_entry_close_min_frontalness:
            return False, f"frontalness too low ({frontalness:.2f})"

        slope = abs(float(bridge.get("centerline_slope_pixels", 999.0)))
        if slope > self.task2_entry_close_max_centerline_slope_pixels:
            return False, f"centerline slope too large ({slope:.0f}px)"

        corridor = self._task2_bridge_corridor(mode="lower", bridge=bridge)
        if not corridor["valid"]:
            return False, corridor.get("reason", "bridge corridor invalid")
        if not (
            self.task2_bridge_corridor_min_width_ratio
            <= corridor["width_ratio"]
            <= 0.95
        ):
            return False, f"bridge corridor width ratio implausible ({corridor['width_ratio']:.2f})"

        entry_depth = float(bridge.get("entry_depth", 0.0))
        if entry_source == "road_contact" and not (0.0 < entry_depth <= self.task2_entry_close_max_range_m):
            return False, f"entry depth not close/valid ({entry_depth:.2f}m)"

        return True, f"entry close criteria satisfied via {entry_source}"

    def _task2_bridge_has_entry_candidate(self, bridge=None):
        if bridge is None:
            bridge = self._task2_bridge_visible()
        if bridge is None:
            return False
        if float(bridge.get("area_ratio", 0.0)) < self.task2_bridge_detect_min_area_ratio:
            return False
        entry_low_enough = (
            float(bridge.get("bottom_y_ratio", 0.0))
            >= self.task2_bridge_entry_roi_min_y_ratio
            or float(bridge.get("bottom_coverage", 0.0)) > 0.04
        )
        has_center = (
            float(bridge.get("bottom_center_x", 0.0)) > 0.0
            or float(bridge.get("center_x", 0.0)) > 0.0
        )
        return entry_low_enough and has_center

    def _task2_bridge_entry_score(self, bridge):
        bottom_y = float(bridge.get("bottom_y_ratio", 0.0))
        bottom_coverage = float(bridge.get("bottom_coverage", 0.0))
        area_ratio = float(bridge.get("area_ratio", 0.0))
        delta = self._task2_bridge_delta("bottom_center_x", bridge)
        if delta is None:
            delta = 999.0
        image_width = max(1.0, self._segmentation_image_width())
        center_score = max(0.0, 1.0 - abs(delta) / (image_width * 0.5))
        return (
            bottom_y * 2.0
            + bottom_coverage * 4.0
            + area_ratio * 3.0
            + center_score * 1.5
        )

    def _task2_update_bridge_entry_progress(self, bridge, entry_score):
        now = self.get_clock().now()
        bottom_y = float(bridge.get("bottom_y_ratio", 0.0))
        bottom_coverage = float(bridge.get("bottom_coverage", 0.0))
        score_improved = entry_score > self.task2_bridge_best_entry_score + 0.03
        y_improved = (
            bottom_y
            > self.task2_bridge_best_bottom_y
            + self.task2_bridge_progress_min_bottom_y_delta
        )
        coverage_improved = (
            bottom_coverage
            > self.task2_bridge_best_bottom_coverage
            + self.task2_bridge_progress_min_coverage_delta
        )

        if self.task2_bridge_best_entry_time is None:
            self.task2_bridge_best_entry_score = entry_score
            self.task2_bridge_best_bottom_y = bottom_y
            self.task2_bridge_best_bottom_coverage = bottom_coverage
            self.task2_bridge_best_entry_time = now
            self.get_logger().info(f"Task 2: entry score initialized: {entry_score:.2f}.")
            return

        if score_improved or y_improved or coverage_improved:
            previous = self.task2_bridge_best_entry_score
            self.task2_bridge_best_entry_score = max(
                self.task2_bridge_best_entry_score, entry_score
            )
            self.task2_bridge_best_bottom_y = max(
                self.task2_bridge_best_bottom_y, bottom_y
            )
            self.task2_bridge_best_bottom_coverage = max(
                self.task2_bridge_best_bottom_coverage, bottom_coverage
            )
            self.task2_bridge_best_entry_time = now
            self._task2_reset_orbit()
            self.get_logger().info(
                f"Task 2: bridge entry score improved: {previous:.2f} -> {entry_score:.2f}."
            )

    def _task2_bridge_recently_seen(self):
        return (
            self.task2_bridge_last_seen_time is not None
            and self._elapsed_seconds(self.task2_bridge_last_seen_time)
            <= self.task2_bridge_lost_grace_seconds
        )

    def _task2_bridge_orbit_search_action(self, bridge):
        if not self.task2_bridge_orbit_enabled:
            return self._task2_bridge_center_action(
                bridge=bridge,
                preferred_key="center_x",
                hard_tolerance=self.task2_bridge_orbit_side_keep_pixels,
                soft_tolerance=None,
                centered_action="CLOCKWISE_ROTATION_SLOW",
                allow_arc=False,
            )

        delta = self._task2_bridge_delta("center_x", bridge)
        if delta is None:
            delta = self.task2_bridge_last_delta_sign
        self._task2_remember_bridge_delta(delta)

        if self.task2_bridge_orbit_cycle_count >= self.task2_bridge_orbit_max_cycles:
            self.get_logger().warn(
                "Task 2: orbit max cycles reached; restarting bridge search."
            )
            self._task2_reset_orbit()
            return None

        if self.task2_bridge_orbit_phase is None:
            self.task2_bridge_orbit_phase = "arc"
            self.task2_bridge_orbit_phase_start_time = self.get_clock().now()
            self.get_logger().info("Task 2: orbit search started.")

        phase_elapsed = self._elapsed_seconds(self.task2_bridge_orbit_phase_start_time)
        if self.task2_bridge_orbit_phase == "arc":
            if phase_elapsed >= self.task2_bridge_orbit_forward_seconds:
                self.task2_bridge_orbit_phase = "turn"
                self.task2_bridge_orbit_phase_start_time = self.get_clock().now()
                phase_elapsed = 0.0
            elif abs(delta) > min(
                self.task2_bridge_orbit_side_keep_pixels,
                self.task2_bridge_approach_hard_tolerance,
            ):
                return (
                    "CLOCKWISE_ROTATION_SLOW"
                    if delta > 0.0
                    else "COUNTERCLOCKWISE_ROTATION_SLOW"
                )
            else:
                return "RIGHT_FRONT" if delta > 0.0 else "LEFT_FRONT"

        if self.task2_bridge_orbit_phase == "turn":
            if phase_elapsed >= self.task2_bridge_orbit_turn_seconds:
                self.task2_bridge_orbit_cycle_count += 1
                self.get_logger().info(
                    f"Task 2: orbit cycle {self.task2_bridge_orbit_cycle_count}/"
                    f"{self.task2_bridge_orbit_max_cycles}."
                )
                self.task2_bridge_orbit_phase = "arc"
                self.task2_bridge_orbit_phase_start_time = self.get_clock().now()
                phase_elapsed = 0.0
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if delta > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )

        return "CLOCKWISE_ROTATION_SLOW"

    def _task2_remember_bridge_delta(self, delta):
        if abs(delta) > 4.0:
            self.task2_bridge_last_delta_sign = 1.0 if delta > 0.0 else -1.0

    def _task2_reset_orbit(self):
        self.task2_bridge_orbit_cycle_count = 0
        self.task2_bridge_orbit_phase = None
        self.task2_bridge_orbit_phase_start_time = None

    def _reset_task2_bridge_runtime(self):
        self.task2_bridge_last_seen_time = None
        self.task2_bridge_best_entry_score = 0.0
        self.task2_bridge_best_entry_time = None
        self.task2_bridge_best_bottom_y = 0.0
        self.task2_bridge_best_bottom_coverage = 0.0
        self.task2_bridge_last_delta_sign = 1.0
        self.task2_bridge_confirm_start_time = None
        self.task2_bridge_detection_start_time = None
        self.task2_ramp_entry_confirm_count = 0
        self.task2_side_view_recovery_start_time = None
        self._task2_reset_orbit()

    def _task2_reset_bridge_confirm(self):
        self.task2_bridge_confirm_start_time = None
        self.task2_bridge_detection_start_time = None

    def _bridge_entry_candidate(self, bridge):
        if bridge is None:
            return False
        near_field = (
            bridge["bottom_coverage"] >= self.task2_bridge_entry_bottom_coverage * 0.35
            or bridge.get("bottom_y_ratio", 0.0)
            >= self.task2_bridge_candidate_min_bottom_y_ratio
            or bridge.get("bottom_center_x", 0.0) > 0.0
        )
        return (
            bridge["area_ratio"] >= self.task2_bridge_min_area_ratio
            and near_field
            or bridge["bottom_coverage"] >= self.task2_bridge_entry_bottom_coverage * 0.5
        )

    def _bridge_entry_ready(self, bridge):
        if bridge is None:
            return False
        bridge_delta = self._segment_delta_x(bridge, "bottom_center_x")
        return (
            abs(bridge_delta) <= self.task2_bridge_entry_center_tolerance
            and bridge["area_ratio"] >= self.task2_bridge_min_area_ratio
            and (
                bridge["bottom_coverage"] >= self.task2_bridge_entry_bottom_coverage
                or bridge.get("bottom_y_ratio", 0.0) >= 0.55
            )
        )

    def _bridge_entry_action(self, bridge):
        if bridge is None:
            self._reset_bridge_entry_alignment()
            return "CLOCKWISE_ROTATION_SLOW"

        bridge_delta = self._segment_delta_x(bridge, "bottom_center_x")
        if abs(bridge_delta) > self.task2_bridge_entry_center_tolerance:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if bridge_delta > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )

        if bridge["bottom_coverage"] < self.task2_bridge_entry_bottom_coverage:
            return "FORWARD_SLOW"

        return "STOP"

    def _bounded_bridge_entry_action(self, bridge):
        if self.task2_bridge_entry_phase is None:
            self._start_bridge_entry_phase("align")

        if self.task2_bridge_entry_phase == "align":
            self._refresh_bridge_entry_progress(bridge)

        phase_elapsed = self._elapsed_seconds(self.task2_bridge_entry_phase_start_time)

        if self.task2_bridge_entry_phase == "backup":
            if phase_elapsed < self.task2_bridge_entry_backup_seconds:
                return "BACKWARD_SLOW"
            self._start_bridge_entry_phase("shift")
            return "RIGHT_SHIFT" if self.task2_bridge_entry_shift_direction > 0.0 else "LEFT_SHIFT"

        if self.task2_bridge_entry_phase == "shift":
            if phase_elapsed < self.task2_bridge_entry_shift_seconds:
                return "RIGHT_SHIFT" if self.task2_bridge_entry_shift_direction > 0.0 else "LEFT_SHIFT"
            self._start_bridge_entry_phase("rotate")
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if self.task2_bridge_entry_shift_direction < 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )

        if self.task2_bridge_entry_phase == "rotate":
            if phase_elapsed < self.task2_bridge_entry_rotate_seconds:
                return (
                    "CLOCKWISE_ROTATION_SLOW"
                    if self.task2_bridge_entry_shift_direction < 0.0
                    else "COUNTERCLOCKWISE_ROTATION_SLOW"
                )
            self._start_bridge_entry_phase("align")

        if self.task2_bridge_entry_phase == "commit":
            if phase_elapsed < self.task2_bridge_entry_commit_seconds:
                return self._bridge_forward_alignment_action(bridge)
            self._start_bridge_entry_phase("align")

        action = self._bridge_connection_alignment_action(bridge)
        if self._bridge_alignment_can_commit(bridge, action):
            self._start_bridge_entry_phase("commit")
            return self._bridge_forward_alignment_action(bridge)

        if self._elapsed_seconds(self.task2_bridge_entry_phase_start_time) >= self.task2_bridge_entry_phase_timeout:
            self._start_bridge_entry_recovery(bridge)
            return "BACKWARD_SLOW"

        return action

    def _bridge_alignment_can_commit(self, bridge, action):
        if action not in ("FORWARD_SLOW", "RIGHT_FRONT", "LEFT_FRONT"):
            return False
        connection = self._bridge_connection()
        if connection is None:
            return False
        return (
            connection["connected"]
            and connection["score"] >= self.task2_bridge_min_connection_score * 0.7
            and connection.get("central_score", 0.0)
            >= self.task2_bridge_central_connection_min_score * 0.7
        )

    def _bridge_forward_alignment_action(self, bridge):
        return self._drivable_follow_action_from_segment(
            bridge,
            prefer_bridge=True,
            forward_action="FORWARD_SLOW",
            hard_turn_tolerance=self.task2_bridge_center_tolerance,
        )

    def _start_bridge_entry_phase(self, phase):
        self.task2_bridge_entry_phase = phase
        self.task2_bridge_entry_phase_start_time = self.get_clock().now()

    def _start_bridge_entry_recovery(self, bridge):
        if (
            self.task2_bridge_entry_recovery_cycles
            >= self.task2_bridge_entry_max_recovery_cycles
        ):
            self.task2_bridge_entry_recovery_cycles = 0
            self.task2_bridge_entry_shift_direction *= -1.0
        else:
            self.task2_bridge_entry_recovery_cycles += 1
            self.task2_bridge_entry_shift_direction = self._bridge_entry_shift_direction(
                bridge
            )
        self._start_bridge_entry_phase("backup")

    def _reset_bridge_entry_alignment(self):
        self.task2_bridge_entry_phase = None
        self.task2_bridge_entry_phase_start_time = None
        self.task2_bridge_entry_recovery_cycles = 0
        self.task2_bridge_entry_best_score = 0.0

    def _refresh_bridge_entry_progress(self, bridge):
        score = self._bridge_entry_progress_score(bridge)
        if score <= self.task2_bridge_entry_best_score + self.task2_bridge_entry_progress_epsilon:
            return
        self.task2_bridge_entry_best_score = score
        self.task2_bridge_entry_phase_start_time = self.get_clock().now()

    def _bridge_entry_progress_score(self, bridge):
        connection = self._bridge_connection()
        bridge_delta = abs(self._segment_delta_x(bridge, "bottom_center_x"))
        image_width = max(1.0, self._segmentation_image_width())
        center_score = max(0.0, 1.0 - bridge_delta / (image_width * 0.5))
        score = (
            float(bridge.get("bottom_coverage", 0.0)) * 4.0
            + float(bridge.get("area_ratio", 0.0)) * 8.0
            + center_score
        )
        if connection is not None:
            score += float(connection.get("score", 0.0)) * 40.0
            score += float(connection.get("central_score", 0.0)) * 2.0
            score -= abs(
                float(connection.get("central_delta_x", connection.get("delta_x", 0.0)))
            ) / image_width
        return score

    def _bridge_entry_shift_direction(self, bridge):
        road = self._segmentation_segment("road")
        if road is not None:
            road_center_x = self._segment_center_x(road, "bottom_center_x")
            bridge_center_x = self._segment_center_x(bridge, "bottom_center_x")
            if road_center_x > 0.0 and bridge_center_x > 0.0:
                center_delta = bridge_center_x - road_center_x
                if abs(center_delta) > 4.0:
                    return 1.0 if center_delta > 0.0 else -1.0

        bridge_delta = float(bridge.get("delta_x", 0.0))
        if abs(bridge_delta) > 4.0:
            return 1.0 if bridge_delta > 0.0 else -1.0
        self.task2_bridge_entry_shift_direction *= -1.0
        return self.task2_bridge_entry_shift_direction

    def _bridge_road_connection_ready(self):
        if not self.task2_bridge_require_connected_entry:
            return True

        road = self._segmentation_segment("road")
        bridge = self._segmentation_segment("bridge")
        connection = self._bridge_connection()
        if road is None or bridge is None or connection is None:
            return False

        road_center_x = self._segment_center_x(road, "bottom_center_x")
        bridge_center_x = self._segment_center_x(bridge, "bottom_center_x")
        if road_center_x <= 0.0 or bridge_center_x <= 0.0:
            return False

        centers_aligned = (
            abs(bridge_center_x - road_center_x)
            <= self.task2_bridge_entry_lateral_tolerance
        )
        connection_delta_x = float(
            connection.get("central_delta_x", connection["delta_x"])
        )
        connection_centered = (
            abs(connection_delta_x)
            <= min(
                self.task2_bridge_connection_center_tolerance,
                self.task2_bridge_entry_lateral_tolerance,
            )
        )
        connection_valid = (
            connection["connected"]
            and connection["score"] >= self.task2_bridge_min_connection_score
            and connection.get("central_score", 0.0)
            >= self.task2_bridge_central_connection_min_score
            and connection["gap_y_ratio"]
            <= self.task2_bridge_max_connection_gap_y_ratio
        )
        return connection_valid and centers_aligned and connection_centered

    def _bridge_connection_alignment_action(self, bridge):
        road = self._segmentation_segment("road")
        connection = self._bridge_connection()

        bridge_delta_x = self._segment_delta_x(bridge, "bottom_center_x")
        if abs(bridge_delta_x) > self.task2_bridge_entry_center_tolerance:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if bridge_delta_x > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )

        if road is not None:
            road_center_x = self._segment_center_x(road, "bottom_center_x")
            bridge_center_x = self._segment_center_x(bridge, "bottom_center_x")
            if road_center_x > 0.0 and bridge_center_x > 0.0:
                center_delta = bridge_center_x - road_center_x
                if abs(center_delta) > self.task2_bridge_entry_lateral_tolerance:
                    return "RIGHT_SHIFT" if center_delta > 0.0 else "LEFT_SHIFT"

        if connection is not None and connection["connected"]:
            connection_delta_x = float(
                connection.get("central_delta_x", connection["delta_x"])
            )
            if abs(connection_delta_x) > self.task2_bridge_entry_lateral_tolerance:
                return "RIGHT_SHIFT" if connection_delta_x > 0.0 else "LEFT_SHIFT"

        # Near the bridge, avoid arcing into the side fence. Rotate or shift first,
        # then commit straight when the entry corridor is centered.
        if abs(bridge_delta_x) > self.drivable_bridge_soft_turn_tolerance:
            return (
                "CLOCKWISE_ROTATION_SLOW"
                if bridge_delta_x > 0.0
                else "COUNTERCLOCKWISE_ROTATION_SLOW"
            )
        return "FORWARD_SLOW"

    def _segment_center_x(self, segment, preferred_key):
        value = float(segment.get(preferred_key, 0.0))
        if value > 0.0:
            return value
        return float(segment.get("center_x", 0.0))

    def _segment_delta_x(self, segment, preferred_key):
        center_x = self._segment_center_x(segment, preferred_key)
        image_width = self._segmentation_image_width()
        if center_x > 0.0 and image_width > 0.0:
            return center_x - (image_width / 2.0)
        return float(segment.get("delta_x", 0.0))

    def _segmentation_image_width(self):
        if self.segmentation_info is None:
            return 0.0
        return float(self.segmentation_info.get("image_width", 0.0))

    def _road_search_action(self):
        road = self._segmentation_segment("road")
        if road is not None:
            return self._segmentation_follow_action(prefer_bridge=False)

        if self.task2_search_cycle_start_time is None:
            self.task2_search_cycle_start_time = self.get_clock().now()

        cycle_seconds = (
            self.task2_road_search_spin_seconds
            + self.task2_road_search_drive_seconds
        )
        if cycle_seconds > 0.0:
            elapsed = self._elapsed_seconds(self.task2_search_cycle_start_time)
            phase = elapsed % cycle_seconds
            if phase < self.task2_road_search_spin_seconds:
                return "CLOCKWISE_ROTATION_SLOW"

        return "CLOCKWISE_ROTATION_SLOW"

    def _segmentation_guard_action(self, action_key):
        return action_key

    def _state_uses_drivable_guard(self):
        return self.state in (
            MissionState.EXPLORE_MAP,
            MissionState.GO_TO_BEAR_AREA,
            MissionState.APPROACH_BEAR,
            MissionState.APPROACH_GRAB,
            MissionState.RETURN_START,
            MissionState.TASK2_NAVIGATE_BRIDGE,
            MissionState.TASK2_ASCEND_BRIDGE,
            MissionState.TASK2_SEARCH_BRIDGE_BEAR,
            MissionState.TASK2_DESCEND_BRIDGE,
        )

    def _preferred_drivable_segment(self, prefer_bridge=False):
        road = self._segmentation_segment("road")
        bridge = self._segmentation_segment("bridge")
        if prefer_bridge:
            return bridge or road
        return road or bridge

    def _bridge_connection(self):
        if self.segmentation_connection is None or self.segmentation_info_stamp is None:
            return None
        if self._elapsed_seconds(self.segmentation_info_stamp) > self.segmentation_timeout:
            return None
        return self.segmentation_connection

    def _segmentation_segment(self, label, allow_cached=True):
        if self.segmentation_info is None or self.segmentation_info_stamp is None:
            return None
        if self._elapsed_seconds(self.segmentation_info_stamp) > self.segmentation_timeout:
            return None
        segment = self.segmentation_info.get(label)
        if segment is None:
            return None
        if segment.get("raw_found", segment.get("found", False)):
            segment["usable"] = True
            segment["predicted_or_cached"] = False
            return segment
        if not allow_cached:
            return None
        last_seen = self.segmentation_last_seen.get(label)
        grace = (
            self.task2_bridge_tracking_loss_grace
            if label == "bridge"
            else self.segmentation_missing_grace
        )
        if last_seen is not None and self._elapsed_seconds(last_seen) <= grace:
            cached = dict(segment)
            cached["found"] = True
            cached["usable"] = True
            cached["predicted_or_cached"] = True
            return cached
        return None

    def _drivable_follow_action_from_segment(
        self,
        segment,
        prefer_bridge=False,
        forward_action="FORWARD_SLOW",
        hard_turn_tolerance=None,
    ):
        if segment is None:
            return self._reacquire_drivable_action(None)

        delta_x = self._segment_delta_x(segment, "bottom_center_x")
        abs_delta_x = abs(delta_x)
        self._remember_drivable_direction(delta_x)

        # If road/bridge mask is not anchored in the lower frame, do not drive forward.
        # Reorient first until drivable area returns to the near-field region.
        if self._needs_lower_frame_realign(segment):
            return self._reacquire_drivable_action(segment)

        hard_tolerance = (
            self.drivable_center_tolerance
            if hard_turn_tolerance is None
            else hard_turn_tolerance
        )
        soft_tolerance = (
            self.drivable_bridge_soft_turn_tolerance
            if prefer_bridge
            else self.drivable_soft_turn_tolerance
        )
        soft_hysteresis = max(0.0, soft_tolerance - self.drivable_turn_hysteresis_pixels)

        if abs_delta_x > hard_tolerance:
            return "CLOCKWISE_ROTATION_SLOW" if delta_x > 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"

        if abs_delta_x > soft_tolerance:
            return "RIGHT_FRONT" if delta_x > 0.0 else "LEFT_FRONT"

        if (
            abs_delta_x > soft_hysteresis
            and self.last_drivable_action in ("RIGHT_FRONT", "LEFT_FRONT")
        ):
            return self.last_drivable_action

        if (
            segment["bottom_coverage"] < self.drivable_min_bottom_coverage
            and segment["area_ratio"] < 0.05
        ):
            return self._reacquire_drivable_action(segment)
        return forward_action

    def _needs_lower_frame_realign(self, segment):
        if segment is None:
            return True

        area_ratio = float(segment["area_ratio"])
        bottom_coverage = float(segment["bottom_coverage"])
        has_visible_surface = area_ratio >= self.drivable_anchor_min_area_ratio
        bottom_aligned = bottom_coverage >= self.drivable_anchor_bottom_coverage
        now = self.get_clock().now()

        if bottom_aligned:
            self.last_drivable_anchor_time = now
            return False

        if not has_visible_surface:
            return False

        if self.last_drivable_anchor_time is not None:
            if self._elapsed_seconds(self.last_drivable_anchor_time) <= self.drivable_anchor_grace_seconds:
                return False
        return True

    def _remember_drivable_direction(self, delta_x):
        threshold = max(4.0, self.drivable_turn_hysteresis_pixels * 0.5)
        if abs(delta_x) < threshold:
            return
        self.last_drivable_delta_sign = 1.0 if delta_x > 0.0 else -1.0

    def _reacquire_drivable_action(self, segment):
        if segment is not None:
            delta_x = float(segment["delta_x"])
            self._remember_drivable_direction(delta_x)
            if abs(delta_x) > self.drivable_turn_hysteresis_pixels:
                return "CLOCKWISE_ROTATION_SLOW" if delta_x > 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"

        return (
            "CLOCKWISE_ROTATION_SLOW"
            if self.last_drivable_delta_sign >= 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _stabilize_drivable_action(self, action_key):
        steering_actions = {
            "FORWARD_SLOW",
            "BACKWARD_SLOW",
            "RIGHT_FRONT",
            "LEFT_FRONT",
            "RIGHT_SHIFT",
            "LEFT_SHIFT",
            "CLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "STOP",
        }
        if action_key not in steering_actions:
            self.last_drivable_action = None
            self.last_drivable_action_time = None
            return action_key

        now = self.get_clock().now()
        if self.last_drivable_action is None or self.last_drivable_action_time is None:
            self.last_drivable_action = action_key
            self.last_drivable_action_time = now
            return action_key

        if action_key != self.last_drivable_action:
            if self._elapsed_seconds(self.last_drivable_action_time) < self.drivable_action_hold_seconds:
                return self.last_drivable_action
            self.last_drivable_action = action_key
            self.last_drivable_action_time = now
            return action_key

        self.last_drivable_action_time = now
        return action_key

    def _target_bbox_in_lower_camera(self):
        if not self._bbox_visible():
            return False, "no fresh target bbox"

        bbox = self.yolo_bbox
        image_width = max(1.0, float(bbox.get("image_width", 0.0)))
        image_height = max(1.0, float(bbox.get("image_height", 0.0)))
        center_x_ratio = float(bbox.get("center_x", 0.0)) / image_width
        center_y_ratio = float(bbox.get("center_y", 0.0)) / image_height
        bottom_y_ratio = float(bbox.get("y2", 0.0)) / image_height
        area_ratio = float(bbox.get("area_ratio", 0.0))

        if area_ratio < self.task1_target_min_bbox_area_ratio:
            return False, f"bbox area ratio {area_ratio:.3f} too small"
        if not (
            self.task1_target_center_min_ratio
            <= center_x_ratio
            <= self.task1_target_center_max_ratio
        ):
            return False, f"bbox x ratio {center_x_ratio:.2f} outside visible gate"
        if (
            center_y_ratio < self.task1_target_lower_min_center_y_ratio
            and bottom_y_ratio < self.task1_target_lower_min_bottom_y_ratio
        ):
            return (
                False,
                "bbox is not in lower camera region "
                f"(center_y={center_y_ratio:.2f}, bottom_y={bottom_y_ratio:.2f})",
            )
        return True, "bbox is in lower camera region"

    def _target_ready_for_grab(self, require_bridge=False):
        if not self._target_visible():
            return False, "no fresh target info"
        if not self._bbox_visible():
            return False, "no fresh target bbox"

        distance = float(self.yolo_target.get("distance", 0.0))
        if not (0.0 < distance <= self.grab_distance):
            return False, f"target distance {distance:.2f} outside grab range"

        bbox = self.yolo_bbox
        image_width = max(1.0, float(bbox.get("image_width", 0.0)))
        image_height = max(1.0, float(bbox.get("image_height", 0.0)))
        center_x_ratio = float(bbox.get("center_x", 0.0)) / image_width
        center_y_ratio = float(bbox.get("center_y", 0.0)) / image_height
        bottom_y_ratio = float(bbox.get("y2", 0.0)) / image_height
        area_ratio = float(bbox.get("area_ratio", 0.0))

        if not (
            self.grab_bbox_center_min_x_ratio
            <= center_x_ratio
            <= self.grab_bbox_center_max_x_ratio
        ):
            return False, f"bbox x ratio {center_x_ratio:.2f} outside gripper ROI"
        if (
            center_y_ratio < self.grab_bbox_min_center_y_ratio
            and bottom_y_ratio < self.grab_bbox_min_bottom_y_ratio
        ):
            return False, (
                "bbox is too high for gripper "
                f"(center_y={center_y_ratio:.2f}, bottom_y={bottom_y_ratio:.2f})"
            )
        if bottom_y_ratio > self.grab_bbox_max_bottom_y_ratio:
            return False, f"bbox bottom_y {bottom_y_ratio:.2f} outside gripper ROI"
        if area_ratio < self.grab_bbox_min_area_ratio:
            return False, f"bbox area ratio {area_ratio:.3f} too small for grab"
        if (
            require_bridge
            and self.bear_context == "task2_bridge_bear"
            and not (
                self.task2_relax_bridge_surface_after_top
                and self.bridge_top_confirmed
            )
            and not self._target_on_bridge_surface(allow_grace=True)
        ):
            return False, "target is not confirmed on bridge surface"
        return True, "target satisfies grab gate"

    def _target_surface_candidate(self):
        if self.target_surface_info is None or self.target_surface_stamp is None:
            return False, "no target surface info"
        if self._elapsed_seconds(self.target_surface_stamp) > self.target_timeout:
            return False, "target surface info is stale"

        info = self.target_surface_info
        if not info.get("target_found", False):
            return False, "target not found in surface info"
        if not info.get("bridge_found", False):
            return False, "bridge mask not found in surface info"

        overlap = float(info.get("bbox_bridge_overlap_ratio", 0.0))
        lower_overlap = float(info.get("bbox_lower_half_bridge_overlap_ratio", 0.0))
        bottom_on_bridge = bool(info.get("target_bottom_center_on_bridge", False))
        center_on_bridge = bool(info.get("target_center_on_bridge", False))
        side_contact = bool(info.get("target_side_bridge_contact", False))
        side_ratio = float(info.get("target_side_bridge_contact_ratio", 0.0))
        valid = (
            lower_overlap >= self.task2_target_bridge_min_lower_overlap_ratio
            or overlap >= self.task2_target_bridge_min_overlap_ratio
            or bottom_on_bridge
            or center_on_bridge
            or side_contact
        )
        if not valid:
            return (
                False,
                "bridge overlap too low "
                f"(overlap={overlap:.2f}, lower={lower_overlap:.2f}, "
                f"side={side_ratio:.2f})",
            )
        if side_contact and not (bottom_on_bridge or center_on_bridge):
            return True, "target touches bridge surface at bbox side"
        return True, "target overlaps bridge surface"

    def _target_on_bridge_surface(self, allow_grace=False):
        valid, reason = self._target_surface_candidate()
        now = self.get_clock().now()
        if valid:
            self.target_surface_last_valid_time = now
            if self.target_surface_confirm_start_time is None:
                self.target_surface_confirm_start_time = now
                self._log_target_surface_wait(
                    "target/bridge overlap candidate detected"
                )
                return False
            return (
                self._elapsed_seconds(self.target_surface_confirm_start_time)
                >= self.task2_target_bridge_confirm_seconds
            )

        self.target_surface_confirm_start_time = None
        if (
            allow_grace
            and self.target_surface_last_valid_time is not None
            and self._elapsed_seconds(self.target_surface_last_valid_time)
            <= self.task2_target_bridge_loss_grace_seconds
        ):
            return True
        self._log_target_surface_wait(reason)
        return False

    def _log_lower_bbox_gate(self, context, reason):
        now = self.get_clock().now()
        if (
            self.last_lower_bbox_log_time is not None
            and self._elapsed_seconds(self.last_lower_bbox_log_time) < 1.0
        ):
            return
        self.last_lower_bbox_log_time = now
        self.get_logger().info(f"Lower-camera gate waiting for {context}: {reason}.")

    def _log_grab_gate_wait(self, reason):
        now = self.get_clock().now()
        if (
            self.last_grab_gate_log_time is not None
            and self._elapsed_seconds(self.last_grab_gate_log_time) < 1.0
        ):
            return
        self.last_grab_gate_log_time = now
        self.get_logger().info(f"Waiting for grab gate: {reason}.")

    def _log_target_surface_wait(self, reason):
        now = self.get_clock().now()
        if (
            self.target_surface_last_log_time is not None
            and self._elapsed_seconds(self.target_surface_last_log_time) < 1.0
        ):
            return
        self.target_surface_last_log_time = now
        if self.target_surface_info is not None:
            self.get_logger().info(
                "Bridge-bear gate waiting: "
                f"{reason}; overlap="
                f"{self.target_surface_info.get('bbox_bridge_overlap_ratio', 0.0):.2f}, "
                f"lower="
                f"{self.target_surface_info.get('bbox_lower_half_bridge_overlap_ratio', 0.0):.2f}, "
                f"side="
                f"{self.target_surface_info.get('target_side_bridge_contact_ratio', 0.0):.2f}."
            )
        else:
            self.get_logger().info(f"Bridge-bear gate waiting: {reason}.")

    def _bbox_visible(self):
        if self.yolo_bbox is None or not self.yolo_bbox["found"]:
            return False
        return self._elapsed_seconds(self.yolo_bbox_stamp) <= self.verify_grab_bbox_timeout

    def _copy_current_bbox(self):
        if not self._bbox_visible():
            return None
        return dict(self.yolo_bbox)

    def _target_visible(self):
        if self.yolo_target is None or not self.yolo_target["found"]:
            return False
        return self._elapsed_seconds(self.yolo_target_stamp) <= self.target_timeout

    def _mission_target_visible(self):
        if self.current_task == 1:
            return self._task1_target_visible()
        if self.bear_context == "task2_bridge_bear" and self.state in (
            MissionState.APPROACH_BEAR,
            MissionState.OBSERVE_BEAR,
            MissionState.APPROACH_GRAB,
        ):
            if (
                self.task2_relax_bridge_surface_after_top
                and self.bridge_top_confirmed
            ):
                return self._target_visible()
            return self._target_visible() and self._target_on_bridge_surface(allow_grace=True)
        return self._target_visible()

    def _task1_target_visible(self):
        if not self._target_visible():
            return False
        if not self.task1_require_target_on_road:
            return True

        valid, reason = self._task1_target_bbox_on_road()
        if valid:
            return True

        self._log_task1_road_target_rejected(reason)
        return False

    def _task1_target_bbox_on_road(self):
        if self.yolo_bbox is None or not self.yolo_bbox["found"]:
            return False, "no fresh target bbox for road check"
        if self._elapsed_seconds(self.yolo_bbox_stamp) > self.target_timeout:
            return False, "target bbox is stale for road check"

        road = self._segmentation_segment("road")
        if road is None or not road.get("found", False):
            if self.task1_allow_target_without_road_mask:
                return True, "road mask unavailable but fallback is enabled"
            return False, "road mask unavailable"
        if float(road.get("area_ratio", 0.0)) < self.task1_road_target_min_area_ratio:
            return False, "road mask too small"

        bbox = self.yolo_bbox
        image_width = max(1.0, float(bbox.get("image_width", 0.0)))
        image_height = max(1.0, float(bbox.get("image_height", 0.0)))
        foot_x = float(bbox.get("center_x", 0.0))
        foot_y_ratio = min(1.0, max(0.0, float(bbox.get("y2", 0.0)) / image_height))

        road_center_x = self._segment_center_x(road, "bottom_center_x")
        if road_center_x <= 0.0:
            return False, "road center unavailable"

        road_half_width = self._task1_estimated_road_half_width(road, image_width)
        road_x_ok = abs(foot_x - road_center_x) <= road_half_width

        margin = max(0.0, self.task1_road_target_y_margin_ratio)
        road_top = float(road.get("top_y_ratio", 0.0))
        road_bottom = float(road.get("bottom_y_ratio", 0.0))
        if road_bottom <= 0.0:
            road_bottom = max(float(road.get("center_y_ratio", 0.0)), road_top)
        road_y_ok = (road_top - margin) <= foot_y_ratio <= (road_bottom + margin)

        if not road_x_ok:
            return (
                False,
                "bear bbox lower center is outside road mask x band "
                f"(dx={foot_x - road_center_x:.0f}px, allowed={road_half_width:.0f}px)",
            )
        if not road_y_ok:
            return (
                False,
                "bear bbox foot is outside road mask y band "
                f"(foot_y={foot_y_ratio:.2f}, road={road_top:.2f}-{road_bottom:.2f})",
            )

        bridge_reason = self._task1_bridge_mask_rejects_bbox(
            bbox=bbox,
            road=road,
            road_center_x=road_center_x,
            road_half_width=road_half_width,
            foot_x=foot_x,
            foot_y_ratio=foot_y_ratio,
        )
        if bridge_reason is not None:
            return False, bridge_reason

        return True, "bear bbox appears on road mask"

    def _task1_estimated_road_half_width(self, road, image_width):
        bottom_coverage = max(0.0, float(road.get("bottom_coverage", 0.0)))
        max_width_ratio = min(
            0.75, max(0.20, self.task1_road_target_max_width_ratio)
        )
        coverage_width = image_width * min(
            max_width_ratio, max(0.18, bottom_coverage * 2.4)
        )
        return max(self.task1_road_target_x_tolerance_pixels, coverage_width)

    def _task1_bridge_mask_rejects_bbox(
        self,
        bbox,
        road,
        road_center_x,
        road_half_width,
        foot_x,
        foot_y_ratio,
    ):
        bridge = self._segmentation_segment("bridge")
        if bridge is None or not bridge.get("found", False):
            return None
        if float(bridge.get("area_ratio", 0.0)) < self.task2_bridge_detect_min_area_ratio:
            return None

        bridge_center_x = self._segment_center_x(bridge, "bottom_center_x")
        if bridge_center_x <= 0.0:
            bridge_center_x = self._segment_center_x(bridge, "mid_center_x")
        if bridge_center_x <= 0.0:
            return None

        margin = max(0.0, self.task1_road_target_y_margin_ratio)
        bridge_top = float(bridge.get("top_y_ratio", 0.0))
        bridge_bottom = float(bridge.get("bottom_y_ratio", 0.0))
        bridge_y_contains = (bridge_top - margin) <= foot_y_ratio <= (
            bridge_bottom + margin
        )
        if not bridge_y_contains:
            return None

        bridge_dx = abs(foot_x - bridge_center_x)
        road_dx = abs(foot_x - road_center_x)
        if bridge_dx + 25.0 < road_dx:
            return "bear bbox aligns more strongly with bridge mask than road mask"

        road_center_y = float(road.get("center_y_ratio", 0.0))
        if (
            road_center_y > 0.0
            and foot_y_ratio < road_center_y - margin
            and bridge_dx <= max(self.task1_road_target_x_tolerance_pixels, road_half_width)
        ):
            return "bear bbox appears above road center inside bridge mask"

        return None

    def _log_task1_road_target_rejected(self, reason):
        now = self.get_clock().now()
        if (
            self.last_task1_road_target_log_time is not None
            and self._elapsed_seconds(self.last_task1_road_target_log_time)
            < self.task1_road_target_log_period
        ):
            return
        self.last_task1_road_target_log_time = now
        self.get_logger().info(
            f"Task 1 ignoring detected bear because it is not on the road mask: {reason}."
        )

    def _elapsed_seconds(self, start_time):
        if start_time is None:
            return 0.0
        now = self.get_clock().now()
        return (now.nanoseconds - start_time.nanoseconds) / 1e9

    def _remaining_mission_time(self):
        if self.mission_time_limit_seconds <= 0.0:
            return float("inf")
        return max(
            0.0,
            self.mission_time_limit_seconds - self._elapsed_seconds(self.mission_start_time),
        )

    def _state_elapsed(self):
        return self._elapsed_seconds(self.state_start_time)

    def _budget_exceeded(self, budget):
        if budget <= 0.0:
            return False
        return self._state_elapsed() >= min(budget, self._remaining_mission_time())

    def _time_after(self, start_time, seconds):
        return start_time + Duration(nanoseconds=int(seconds * 1e9))

    def _log_waiting_for_pose(self):
        now = self.get_clock().now()
        if (
            self.last_pose_wait_log_time is None
            or self._elapsed_seconds(self.last_pose_wait_log_time) >= 5.0
        ):
            self.last_pose_wait_log_time = now
            self.get_logger().warn(
                "Waiting for /amcl_pose. Start pros_app localization_unity, "
                "confirm /map and /scan exist, then set the robot initial pose if needed."
            )

    def _log_waiting_for_exploration_goal(self):
        now = self.get_clock().now()
        if (
            self.last_explore_wait_log_time is None
            or self._elapsed_seconds(self.last_explore_wait_log_time) >= 5.0
        ):
            self.last_explore_wait_log_time = now
            if self.map_msg is None:
                self.get_logger().warn(
                    f"Waiting for {self.map_topic}; rotating in place while scanning."
                )
            else:
                self.get_logger().warn(
                    "No safe free-space exploration goal found on the map; "
                    "rotating in place while scanning."
                )

    def _publish_initial_pose_if_needed(self):
        if not self.auto_set_initial_pose or self.pose is not None:
            return
        if self.initial_pose_publish_sent >= self.initial_pose_publish_count:
            return

        now = self.get_clock().now()
        if (
            self.last_initial_pose_publish_time is not None
            and self._elapsed_seconds(self.last_initial_pose_publish_time)
            < self.initial_pose_publish_period
        ):
            return

        self.last_initial_pose_publish_time = now
        self.initial_pose_publish_sent += 1
        self._publish_initial_pose(self.initial_pose)

    def _publish_initial_pose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.position.x = float(pose[0])
        msg.pose.pose.position.y = float(pose[1])
        msg.pose.pose.position.z = 0.0
        msg.pose.pose.orientation.z = math.sin(pose[2] / 2.0)
        msg.pose.pose.orientation.w = math.cos(pose[2] / 2.0)
        msg.pose.covariance[0] = self.initial_pose_position_covariance
        msg.pose.covariance[7] = self.initial_pose_position_covariance
        msg.pose.covariance[35] = self.initial_pose_yaw_covariance
        self.initial_pose_pub.publish(msg)
        self.get_logger().info(
            "Published /initialpose: "
            f"x={pose[0]:.2f}, y={pose[1]:.2f}, yaw={pose[2]:.2f} "
            f"({self.initial_pose_publish_sent}/{self.initial_pose_publish_count})"
        )

    def _publish_startup_arm_stow_if_needed(self):
        if not self.startup_arm_stow_enabled:
            return
        if self.startup_arm_stow_publish_sent >= self.startup_arm_stow_publish_count:
            return

        if self.state in (
            MissionState.APPROACH_GRAB,
            MissionState.SECURE_BEAR,
            MissionState.VERIFY_GRAB,
            MissionState.RETURN_START,
            MissionState.DROP_BEAR,
        ):
            self.startup_arm_stow_publish_sent = self.startup_arm_stow_publish_count
            return

        if (
            self.last_startup_arm_stow_publish_time is not None
            and self._elapsed_seconds(self.last_startup_arm_stow_publish_time)
            < self.startup_arm_stow_publish_period
        ):
            return

        self.last_startup_arm_stow_publish_time = self.get_clock().now()
        self.startup_arm_stow_publish_sent += 1
        self._publish_arm_positions(self.startup_arm_stow_positions)
        if self.startup_arm_stow_publish_sent == 1:
            self.get_logger().info(
                "Moving arm to startup stow position to keep camera view clear."
            )

    def _cleanup_expired_virtual_obstacles(self):
        if not self.virtual_obstacle_enabled or not self.virtual_obstacles:
            return
        now = self.get_clock().now()
        ttl_ns = int(self.virtual_obstacle_ttl_seconds * 1e9)
        self.virtual_obstacles = [
            obstacle
            for obstacle in self.virtual_obstacles
            if now.nanoseconds - obstacle["created_time"].nanoseconds <= ttl_ns
        ]

    def _mark_virtual_obstacle_from_action(self, action_key):
        if not self.virtual_obstacle_enabled or self.pose is None:
            return

        projection = self._action_projection(action_key)
        if projection is None:
            return

        angle_offset, distance = projection
        obstacle_yaw = self.pose[2] + angle_offset
        x = self.pose[0] + math.cos(obstacle_yaw) * distance
        y = self.pose[1] + math.sin(obstacle_yaw) * distance
        line_yaw = obstacle_yaw + math.pi * 0.5
        half = max(0.05, self.virtual_obstacle_line_length * 0.5)
        x0 = x - math.cos(line_yaw) * half
        y0 = y - math.sin(line_yaw) * half
        x1 = x + math.cos(line_yaw) * half
        y1 = y + math.sin(line_yaw) * half
        now = self.get_clock().now()

        for obstacle in self.virtual_obstacles:
            if distance_2d((x, y), (obstacle["x"], obstacle["y"])) <= self.virtual_obstacle_merge_distance:
                obstacle["x"] = (obstacle["x"] + x) * 0.5
                obstacle["y"] = (obstacle["y"] + y) * 0.5
                obstacle["x0"] = (obstacle.get("x0", x0) + x0) * 0.5
                obstacle["y0"] = (obstacle.get("y0", y0) + y0) * 0.5
                obstacle["x1"] = (obstacle.get("x1", x1) + x1) * 0.5
                obstacle["y1"] = (obstacle.get("y1", y1) + y1) * 0.5
                obstacle["hit_count"] += 1
                obstacle["created_time"] = now
                obstacle["source_action"] = action_key
                return

        self.virtual_obstacles.append(
            {
                "shape": self.virtual_obstacle_shape,
                "x": x,
                "y": y,
                "x0": x0,
                "y0": y0,
                "x1": x1,
                "y1": y1,
                "line_width": self.virtual_obstacle_line_width,
                "radius": self.virtual_obstacle_radius,
                "source_action": action_key,
                "created_action": action_key,
                "created_state": self.state.value,
                "created_time": now,
                "hit_count": 1,
                "suppressed_until": None,
                "suppression_reason": "",
            }
        )
        if len(self.virtual_obstacles) > self.virtual_obstacle_max_count:
            self.virtual_obstacles = self.virtual_obstacles[-self.virtual_obstacle_max_count :]
        self.get_logger().warn(
            "Added virtual obstacle from stuck event: "
            f"x={x:.2f}, y={y:.2f}, shape={self.virtual_obstacle_shape}, "
            f"action={action_key}."
        )

    def _action_projection(self, action_key):
        diagonal = self.virtual_obstacle_diagonal_distance
        front = self.virtual_obstacle_front_distance
        if action_key in ("FORWARD", "FORWARD_SLOW"):
            return 0.0, front
        if action_key == "RIGHT_FRONT":
            return -math.pi / 4.0, diagonal
        if action_key == "LEFT_FRONT":
            return math.pi / 4.0, diagonal
        if action_key in ("CLOCKWISE_ROTATION", "CLOCKWISE_ROTATION_SLOW", "CLOCKWISE_ROTATION_MEDIAN"):
            return -math.pi / 4.0, diagonal
        if action_key in (
            "COUNTERCLOCKWISE_ROTATION",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_MEDIAN",
        ):
            return math.pi / 4.0, diagonal
        if action_key == "RIGHT_SHIFT":
            return -math.pi / 2.0, diagonal
        if action_key == "LEFT_SHIFT":
            return math.pi / 2.0, diagonal
        if action_key in ("BACKWARD", "BACKWARD_SLOW"):
            return math.pi, front
        return None

    def _virtual_obstacle_blocks_world(self, x, y):
        self._cleanup_expired_virtual_obstacles()
        for obstacle in self.virtual_obstacles:
            if self._virtual_obstacle_is_suppressed(obstacle):
                continue
            if self._virtual_obstacle_contains_point(obstacle, x, y):
                return True
        return False

    def _virtual_obstacle_contains_point(self, obstacle, x, y):
        if obstacle.get("shape", self.virtual_obstacle_shape) == "disk":
            return distance_2d((x, y), (obstacle["x"], obstacle["y"])) <= obstacle["radius"]
        width = float(obstacle.get("line_width", self.virtual_obstacle_line_width))
        radius = max(width * 0.5, self.robot_footprint_radius * 0.5)
        p0 = (float(obstacle.get("x0", obstacle["x"])), float(obstacle.get("y0", obstacle["y"])))
        p1 = (float(obstacle.get("x1", obstacle["x"])), float(obstacle.get("y1", obstacle["y"])))
        return self._point_to_segment_distance((x, y), p0, p1) <= radius

    def _virtual_obstacle_is_suppressed(self, obstacle):
        until = obstacle.get("suppressed_until")
        if until is None:
            return False
        if hasattr(until, "nanoseconds"):
            return self.get_clock().now().nanoseconds <= until.nanoseconds
        return False

    def _virtual_obstacle_inside_active_bridge_gate(self, obstacle, bridge=None):
        x = float(obstacle.get("x", 0.0))
        y = float(obstacle.get("y", 0.0))
        if self._near_bridge_entry_opening(x, y):
            return True
        if self._inside_bridge_center_corridor(x, y):
            return True
        if obstacle.get("created_state") == MissionState.TASK2_APPROACH_BRIDGE_ENTRY.value:
            return True
        if bridge is not None and self.pose is not None:
            delta = self._task2_bridge_entry_delta(bridge)
            if delta is None:
                delta = self._task2_bridge_rough_target_delta(bridge)
            if delta is not None and abs(delta) <= self.task2_bridge_entry_center_tolerance:
                target_yaw = math.atan2(y - self.pose[1], x - self.pose[0])
                if abs(normalize_angle(target_yaw - self.pose[2])) <= math.radians(25.0):
                    return True
        entry = self._bridge_entry_xy()
        if entry is not None:
            return (
                distance_2d((x, y), entry)
                <= self.bridge_entry_gate_clear_radius
                + self.task2_entry_virtual_obstacle_gate_margin
            )
        return False

    def _suppress_virtual_obstacle_for_bridge_entry(self, obstacle):
        if obstacle.get("suppression_reason"):
            return
        obstacle["suppressed_until"] = self._time_after(self.get_clock().now(), 6.0)
        obstacle["suppression_reason"] = "inside active bridge entry gate"
        self._log_event(
            "warn",
            "virtual_obstacle_suppressed_for_bridge_entry",
            obstacle_x=obstacle.get("x"),
            obstacle_y=obstacle.get("y"),
            created_state=obstacle.get("created_state"),
            created_action=obstacle.get("created_action"),
        )

    def _bridge_side_blocks_world_point(self, x, y, radius=None):
        if radius is None:
            radius = self.bridge_side_line_width * 0.5 + self.robot_footprint_radius
        if self._near_bridge_entry_opening(x, y):
            return False
        if self._inside_bridge_center_corridor(x, y):
            return False
        if not self.bridge_geometry_quality_accepted:
            self._fit_bridge_side_lines()
        return self._bridge_side_blocker_at_world_point(x, y, radius=radius) is not None

    def _near_bridge_entry_opening(self, x, y):
        entry = self._bridge_entry_xy()
        if entry is None:
            return False
        if distance_2d((x, y), entry) <= max(
            self.bridge_entry_opening_keep_clear,
            self.bridge_entry_gate_clear_radius,
        ):
            return True
        axis = self.bridge_landmark.get("forward_axis")
        if axis is None:
            return False
        forward, lateral = self._bridge_entry_coordinates(x, y)
        gate_half_width = max(0.0, self.bridge_entry_gate_clear_width * 0.5)
        return (
            -self.bridge_entry_gate_clear_radius
            <= forward
            <= self.bridge_side_obstacle_start_after_entry
            and abs(lateral) <= gate_half_width + self.robot_footprint_radius
        )

    def _bridge_entry_xy(self):
        entry_x = self.bridge_landmark.get("entry_map_x")
        entry_y = self.bridge_landmark.get("entry_map_y")
        if entry_x is None or entry_y is None:
            return None
        return (float(entry_x), float(entry_y))

    def _bridge_pre_entry_xy(self):
        x = self.bridge_landmark.get("pre_entry_map_x")
        y = self.bridge_landmark.get("pre_entry_map_y")
        if x is None or y is None:
            return None
        return (float(x), float(y))

    def _bridge_entry_coordinates(self, x, y):
        entry = self._bridge_entry_xy()
        axis = self.bridge_landmark.get("forward_axis")
        if entry is None or axis is None:
            return None, None
        dx = float(x) - entry[0]
        dy = float(y) - entry[1]
        ax, ay = axis
        forward = dx * ax + dy * ay
        lateral = -dx * ay + dy * ax
        return forward, lateral

    def _inside_bridge_center_corridor(self, x, y):
        half_width = max(0.0, self.bridge_center_corridor_clear_width * 0.5)
        if half_width <= 0.0:
            return False
        forward, lateral = self._bridge_entry_coordinates(x, y)
        if forward is not None and lateral is not None and forward >= 0.0:
            return abs(lateral) <= half_width
        centerline = self.bridge_landmark.get("centerline_map_points", [])
        if not centerline:
            return False
        return min(distance_2d((x, y), p[:2]) for p in centerline) <= half_width

    def _bridge_side_has_enough_observations(self, key):
        points = self.bridge_landmark.get(key, [])
        if len(points) < self.bridge_side_commit_min_points:
            return False
        count_key = (
            "left_observation_count"
            if key == "left_side_map_points"
            else "right_observation_count"
        )
        return (
            self.bridge_landmark.get(count_key, 0)
            >= self.bridge_side_commit_min_observations
        )

    def _committed_bridge_side_point(self, point, key):
        if not self._bridge_side_has_enough_observations(key):
            return False
        x, y = float(point[0]), float(point[1])
        if self._near_bridge_entry_opening(x, y):
            return False
        if self._inside_bridge_center_corridor(x, y):
            return False
        forward, _ = self._bridge_entry_coordinates(x, y)
        if forward is not None:
            if forward < self.bridge_side_obstacle_start_after_entry:
                return False
            if (
                self.bridge_side_obstacle_max_entry_distance > 0.0
                and forward > self.bridge_side_obstacle_max_entry_distance
            ):
                return False
        else:
            entry = self._bridge_entry_xy()
            if (
                entry is not None
                and distance_2d((x, y), entry)
                < self.bridge_side_obstacle_start_after_entry
            ):
                return False
        return True

    def _robot_footprint_is_clear_at(self, x, y, yaw=None):
        if self._virtual_obstacle_blocks_world(x, y):
            return False
        if self._bridge_side_blocks_world_point(x, y):
            return False
        if self.map_msg is not None:
            info = self.map_msg.info
            resolution = info.resolution
            if resolution > 0.0:
                gx = int((x - info.origin.position.x) / resolution)
                gy = int((y - info.origin.position.y) / resolution)
                if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
                    return False
                value = self.map_msg.data[gy * info.width + gx]
                if value > self.map_free_threshold:
                    return False
        return True

    def _action_safety_result(
        self,
        clear=True,
        blocker_type="none",
        blocker_side="unknown",
        blocker_distance=0.0,
        blocker_x=0.0,
        blocker_y=0.0,
        reason="clear",
    ):
        return {
            "clear": bool(clear),
            "blocked": not bool(clear),
            "blocker_type": blocker_type,
            "blocker_side": blocker_side,
            "blocker_distance": float(blocker_distance),
            "blocker_x": float(blocker_x),
            "blocker_y": float(blocker_y),
            "reason": reason,
        }

    def _action_safety_check(self, action_key, context="default", bridge=None):
        if self.pose is None:
            return self._action_safety_result()
        if context == "task2_bridge_entry":
            self._fit_bridge_side_lines()
        projection = self._action_projection(action_key)
        if projection is None or self._action_is_rotation(action_key):
            return self._action_safety_result()

        angle_offset, distance = projection
        steps = max(2, int(self.motion_safety_sample_count))
        yaw = self.pose[2] + angle_offset
        for index in range(1, steps + 1):
            fraction = index / steps
            x = self.pose[0] + math.cos(yaw) * distance * fraction
            y = self.pose[1] + math.sin(yaw) * distance * fraction
            result = self._world_point_safety_check(
                x, y, context=context, bridge=bridge
            )
            if not result["clear"]:
                result["blocker_distance"] = distance * fraction
                return result
        return self._action_safety_result()

    def _world_point_safety_check(self, x, y, context="default", bridge=None):
        if self._near_bridge_entry_opening(x, y):
            return self._action_safety_result(
                blocker_type="bridge_entry_gate",
                blocker_x=x,
                blocker_y=y,
                reason="bridge entry gate is explicitly free",
            )
        if self._inside_bridge_center_corridor(x, y):
            return self._action_safety_result(
                blocker_type="bridge_center_corridor",
                blocker_x=x,
                blocker_y=y,
                reason="bridge center corridor is explicitly free",
            )

        virtual = self._virtual_obstacle_at_world_point(x, y, bridge=bridge)
        if virtual is not None:
            return self._action_safety_result(
                clear=False,
                blocker_type="virtual_obstacle",
                blocker_side=self._blocker_side_from_point(virtual["x"], virtual["y"]),
                blocker_x=virtual["x"],
                blocker_y=virtual["y"],
                reason=virtual.get("suppression_reason") or "virtual obstacle",
            )

        bridge_block = self._bridge_side_blocker_at_world_point(x, y)
        if bridge_block is not None:
            key, point = bridge_block
            return self._action_safety_result(
                clear=False,
                blocker_type=(
                    "bridge_left_side"
                    if key == "left_side_map_points"
                    else "bridge_right_side"
                ),
                blocker_side=(
                    "left" if key == "left_side_map_points" else "right"
                ),
                blocker_x=point[0],
                blocker_y=point[1],
                reason="committed bridge side obstacle",
            )

        if self.map_msg is not None:
            map_result = self._static_map_safety_at_world_point(x, y)
            if map_result is not None:
                return map_result

        return self._action_safety_result()

    def _virtual_obstacle_at_world_point(self, x, y, bridge=None):
        self._cleanup_expired_virtual_obstacles()
        for obstacle in self.virtual_obstacles:
            if self._virtual_obstacle_is_suppressed(obstacle):
                continue
            if not self._virtual_obstacle_contains_point(obstacle, x, y):
                continue
            if (
                self.task2_entry_ignore_virtual_obstacle_inside_gate
                and self._virtual_obstacle_inside_active_bridge_gate(obstacle, bridge)
            ):
                self._suppress_virtual_obstacle_for_bridge_entry(obstacle)
                continue
            return obstacle
        return None

    def _bridge_side_blocker_at_world_point(self, x, y, radius=None):
        if radius is None:
            radius = self.bridge_side_line_width * 0.5 + self.robot_footprint_radius
        if not self.bridge_geometry_quality_accepted:
            self._fit_bridge_side_lines()
        if not self.bridge_geometry_quality_accepted:
            return None
        for side, key in (("left", "left_side_map_points"), ("right", "right_side_map_points")):
            points = self.fitted_bridge_side_lines.get(side, [])
            for p0, p1 in zip(points, points[1:]):
                if self._point_to_segment_distance((x, y), p0[:2], p1[:2]) <= radius:
                    return key, p0
        return None

    def _static_map_safety_at_world_point(self, x, y):
        info = self.map_msg.info
        resolution = info.resolution
        if resolution <= 0.0:
            return None
        gx = int((x - info.origin.position.x) / resolution)
        gy = int((y - info.origin.position.y) / resolution)
        if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
            return self._action_safety_result(
                clear=False,
                blocker_type="unknown",
                blocker_side=self._blocker_side_from_point(x, y),
                blocker_x=x,
                blocker_y=y,
                reason="outside static map bounds",
            )
        value = self.map_msg.data[gy * info.width + gx]
        if value > self.map_free_threshold:
            return self._action_safety_result(
                clear=False,
                blocker_type="static_map",
                blocker_side=self._blocker_side_from_point(x, y),
                blocker_x=x,
                blocker_y=y,
                reason=f"static map cell occupied ({value})",
            )
        return None

    def _blocker_side_from_point(self, x, y):
        if self.pose is None:
            return "unknown"
        dx = float(x) - self.pose[0]
        dy = float(y) - self.pose[1]
        forward = math.cos(self.pose[2]) * dx + math.sin(self.pose[2]) * dy
        lateral = -math.sin(self.pose[2]) * dx + math.cos(self.pose[2]) * dy
        if forward < -0.05:
            return "behind"
        if abs(lateral) < 0.10:
            return "front"
        return "front_left" if lateral > 0.0 else "front_right"

    def _action_swept_path_is_clear(self, action_key):
        return self._action_safety_check(action_key)["clear"]

    def _avoid_virtual_obstacle_for_action(self, action_key):
        if self.pose is None:
            return action_key
        if not self._action_swept_path_is_clear(action_key):
            now = self.get_clock().now()
            if (
                self.last_virtual_obstacle_log_time is None
                or self._elapsed_seconds(self.last_virtual_obstacle_log_time) >= 1.0
            ):
                self.last_virtual_obstacle_log_time = now
                self.get_logger().warn(
                    f"Movement blocked by semantic swept-path safety: {action_key}."
                )
            if action_key in ("FORWARD", "FORWARD_SLOW", "RIGHT_FRONT", "LEFT_FRONT"):
                return "STOP"
            return action_key
        if self._action_is_rotation(action_key):
            return action_key
        if not self.virtual_obstacle_enabled:
            return action_key
        projection = self._action_projection(action_key)
        if projection is None:
            return action_key

        angle_offset, distance = projection
        predicted_yaw = self.pose[2] + angle_offset
        x = self.pose[0] + math.cos(predicted_yaw) * distance
        y = self.pose[1] + math.sin(predicted_yaw) * distance
        if not self._virtual_obstacle_blocks_world(x, y):
            return action_key

        now = self.get_clock().now()
        if (
            self.last_virtual_obstacle_log_time is None
            or self._elapsed_seconds(self.last_virtual_obstacle_log_time) >= 1.0
        ):
            self.last_virtual_obstacle_log_time = now
            self.get_logger().warn(
                "Movement blocked by virtual obstacle: "
                f"action={action_key}, predicted=({x:.2f},{y:.2f})."
            )

        if action_key in ("RIGHT_FRONT", "RIGHT_SHIFT", "CLOCKWISE_ROTATION_SLOW"):
            return "COUNTERCLOCKWISE_ROTATION_SLOW"
        if action_key in ("LEFT_FRONT", "LEFT_SHIFT", "COUNTERCLOCKWISE_ROTATION_SLOW"):
            return "CLOCKWISE_ROTATION_SLOW"
        if action_key in ("FORWARD", "FORWARD_SLOW"):
            return (
                "COUNTERCLOCKWISE_ROTATION_SLOW"
                if self.last_drivable_delta_sign >= 0.0
                else "CLOCKWISE_ROTATION_SLOW"
            )
        return "STOP"

    def _publish_bridge_debug_outputs(self):
        now = self.get_clock().now()
        if (
            self.bridge_markers_publish_rate > 0.0
            and (
                self.last_bridge_marker_publish_time is None
                or self._elapsed_seconds(self.last_bridge_marker_publish_time)
                >= 1.0 / self.bridge_markers_publish_rate
            )
        ):
            self.last_bridge_marker_publish_time = now
            self._publish_bridge_markers()

        if (
            self.augmented_map_enabled
            and self.augmented_map_publish_rate > 0.0
            and (
                self.last_augmented_map_publish_time is None
                or self._elapsed_seconds(self.last_augmented_map_publish_time)
                >= 1.0 / self.augmented_map_publish_rate
            )
        ):
            self.last_augmented_map_publish_time = now
            self._publish_augmented_map()

        if (
            self.last_bridge_debug_log_time is None
            or self._elapsed_seconds(self.last_bridge_debug_log_time) >= 1.0
        ):
            self.last_bridge_debug_log_time = now
            text = (
                f"bridge_valid={self.bridge_landmark.get('valid', False)} "
                f"conf={self.bridge_landmark.get('confidence', 0.0):.2f} "
                f"obs={self.bridge_landmark.get('observation_count', 0)} "
                f"fresh_age={self._bridge_fresh_age():.2f}s "
                f"entry=({self.bridge_landmark.get('entry_map_x')},"
                f"{self.bridge_landmark.get('entry_map_y')}) "
                f"turn_sign={self.task2_turn_direction_sign:+.0f} "
                f"wrong_way={self.task2_turn_wrong_way_count} "
                f"close_reason={self.task2_entry_close_last_reason} "
                f"pose_z={self.pose_z:.3f} "
                f"top_confidence={self.bridge_top_confirmed} "
                f"top_frames={self.bridge_top_confirm_count} "
                f"bear_memory={self.bridge_bear_memory.valid} "
                f"virtual_obstacle_shape={self.virtual_obstacle_shape} "
                f"bridge_side_line_cells={self.bridge_side_line_cells}"
            )
            msg = String()
            msg.data = text
            self.bridge_debug_pub.publish(msg)
            self._log_bridge_debug(
                bridge_valid=self.bridge_landmark.get("valid", False),
                bridge_confidence=self.bridge_landmark.get("confidence", 0.0),
                committed_side_points=self._committed_bridge_side_point_count(),
                augmented_map_obstacle_cells=self.last_augmented_map_obstacle_cell_count,
            )

        if (
            self.last_foxglove_debug_publish_time is None
            or self._elapsed_seconds(self.last_foxglove_debug_publish_time) >= 1.0
        ):
            self.last_foxglove_debug_publish_time = now
            self._publish_foxglove_debug()

        if (
            self.last_bridge_geometry_quality_publish_time is None
            or self._elapsed_seconds(self.last_bridge_geometry_quality_publish_time) >= 1.0
        ):
            self.last_bridge_geometry_quality_publish_time = now
            self._publish_bridge_geometry_quality()

    def _publish_foxglove_debug(self):
        committed = self._committed_bridge_side_point_count()
        if self.bridge_marker_publish_count <= 0:
            marker_reason = "marker topic not published yet"
        elif committed <= 0 and self._bridge_entry_xy() is None:
            marker_reason = "no transformed bridge geometry committed"
        elif not self.bridge_tf_success and (self.bridge_edge_points_received or self.bridge_boundary_points_received):
            marker_reason = f"TF failed: {self.bridge_last_tf_error}"
        else:
            marker_reason = self.bridge_marker_visible_reason

        bridge = self._task2_bridge_visible(allow_cached=True)
        entry_source = self._task2_entry_source(bridge, update_ramp_confirm=False)
        ramp_valid = bool(bridge and float(bridge.get("ramp_valid", 0.0)) >= 0.5)
        ramp_confidence = float(bridge.get("ramp_confidence", 0.0)) if bridge else 0.0
        side_view_score = float(bridge.get("side_view_score", 0.0)) if bridge else 0.0
        ramp_continuous = bool(bridge and float(bridge.get("ramp_continuous", 0.0)) >= 0.5)
        msg = String()
        msg.data = (
            "node_alive=True "
            f"map_received={self.map_msg is not None} "
            f"edge_points_received={self.bridge_edge_points_received} "
            f"boundary_points_received={self.bridge_boundary_points_received} "
            f"entry_point_received={self.bridge_entry_point_received} "
            f"tf_success={self.bridge_tf_success} "
            f"source_frame={self.bridge_source_frame} "
            f"left_points={self.bridge_point_counts['left']} "
            f"right_points={self.bridge_point_counts['right']} "
            f"center_points={self.bridge_point_counts['center']} "
            f"entry_points={self.bridge_point_counts['entry']} "
            f"committed_side_points={committed} "
            f"markers_published={self.bridge_marker_publish_count > 0} "
            f"augmented_map_published={self.augmented_map_publish_count > 0} "
            f"augmented_map_obstacle_cells={self.last_augmented_map_obstacle_cell_count} "
            f"fitted_left_line_points={len(self.fitted_bridge_side_lines.get('left', []))} "
            f"fitted_right_line_points={len(self.fitted_bridge_side_lines.get('right', []))} "
            f"bridge_side_line_cells={self.bridge_side_line_cells} "
            f"virtual_obstacle_cells={self.virtual_obstacle_cells} "
            f"virtual_obstacle_shape={self.virtual_obstacle_shape} "
            f"entry_gate_clear_cells={self.entry_gate_clear_cells} "
            f"center_corridor_clear_cells={self.center_corridor_clear_cells} "
            f"ramp_valid={ramp_valid} "
            f"ramp_confidence={ramp_confidence:.2f} "
            f"side_view_score={side_view_score:.2f} "
            f"ramp_continuous={ramp_continuous} "
            f"entry_source={entry_source} "
            f"pose_z={self.pose_z:.3f} "
            f"start_pose_z={(self.start_pose_z or 0.0):.3f} "
            f"top_confidence={self.bridge_top_confirmed} "
            f"top_confirm_frames={self.bridge_top_confirm_count} "
            f"top_reason={self.bridge_top_confidence_reason} "
            f"bridge_bear_memory_valid={self.bridge_bear_memory.valid} "
            f"marker_reason={marker_reason} "
            f"augmented_map_reason={self.augmented_map_visible_reason}"
        )
        self.foxglove_debug_pub.publish(msg)

    def _publish_bridge_geometry_quality(self):
        self._fit_bridge_side_lines()
        left = self.fitted_bridge_side_lines.get("left", [])
        right = self.fitted_bridge_side_lines.get("right", [])
        width = self._estimated_fitted_bridge_width(left, right)
        heading = self.bridge_landmark.get("estimated_bridge_heading")
        bridge = self._task2_bridge_visible(allow_cached=True)
        entry_source = self._task2_entry_source(bridge, update_ramp_confirm=False)
        msg = String()
        msg.data = (
            f"projection_ok={self.bridge_edge_points_received or self.bridge_boundary_points_received} "
            f"tf_ok={self.bridge_tf_success} "
            f"entry_source={entry_source} "
            f"ramp_valid={bool(bridge and float(bridge.get('ramp_valid', 0.0)) >= 0.5)} "
            f"ramp_confidence={float(bridge.get('ramp_confidence', 0.0)) if bridge else 0.0:.2f} "
            f"side_view_score={float(bridge.get('side_view_score', 0.0)) if bridge else 0.0:.2f} "
            f"top_confidence={self.bridge_top_confirmed} "
            f"left_point_count_raw={self.bridge_point_counts['left']} "
            f"right_point_count_raw={self.bridge_point_counts['right']} "
            f"left_point_count_after_filter={len(left)} "
            f"right_point_count_after_filter={len(right)} "
            f"fitted_width_m={width:.3f} "
            f"fitted_length_m={max(self._polyline_length(left), self._polyline_length(right)):.3f} "
            f"fitted_heading_deg={math.degrees(heading) if heading is not None else 0.0:.1f} "
            f"distance_to_entry={self._distance_to_bridge_entry():.3f} "
            f"accepted={self.bridge_geometry_quality_accepted} "
            f"reason_if_rejected={self.bridge_geometry_quality_reason}"
        )
        self.bridge_geometry_quality_pub.publish(msg)

    def _estimated_fitted_bridge_width(self, left, right):
        if not left or not right:
            return 0.0
        widths = [min(distance_2d(lp[:2], rp[:2]) for rp in right) for lp in left]
        return sorted(widths)[len(widths) // 2] if widths else 0.0

    def _distance_to_bridge_entry(self):
        if self.pose is None:
            return 0.0
        entry = self._bridge_entry_xy()
        if entry is None:
            return 0.0
        return distance_2d(self.pose[:2], entry)

    def _committed_bridge_side_point_count(self):
        count = 0
        for key in ("left_side_map_points", "right_side_map_points"):
            if not self._bridge_side_has_enough_observations(key):
                continue
            count += sum(
                1
                for point in self.bridge_landmark.get(key, [])
                if self._committed_bridge_side_point(point, key)
            )
        return count

    def _bridge_fresh_age(self):
        fresh = self.bridge_landmark.get("last_fresh_time")
        if fresh is None:
            return 999.0
        return self._elapsed_seconds(fresh)

    def _make_marker(self, marker_id, marker_type, ns, color, scale=0.05):
        marker = Marker()
        marker.header.frame_id = self.map_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = ns
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]
        return marker

    def _publish_bridge_markers(self):
        self._fit_bridge_side_lines()
        marker_array = MarkerArray()
        left_line = self.fitted_bridge_side_lines.get("left", [])
        right_line = self.fitted_bridge_side_lines.get("right", [])
        marker_array.markers.extend(
            [
                self._line_marker(1, "bridge_left_side", left_line, (1.0, 0.0, 0.0, 1.0)),
                self._line_marker(2, "bridge_right_side", right_line, (1.0, 0.0, 0.0, 1.0)),
                self._line_marker(3, "bridge_centerline", self.bridge_landmark.get("centerline_map_points", []), (0.0, 0.2, 1.0, 1.0)),
            ]
        )
        if left_line:
            marker_array.markers.append(self._text_marker(101, "bridge_labels", "left side obstacle", left_line[0], (1.0, 0.0, 0.0, 1.0)))
        if right_line:
            marker_array.markers.append(self._text_marker(102, "bridge_labels", "right side obstacle", right_line[0], (1.0, 0.0, 0.0, 1.0)))
        entry_x = self.bridge_landmark.get("entry_map_x")
        entry_y = self.bridge_landmark.get("entry_map_y")
        if (
            entry_x is None
            and entry_y is None
            and self._committed_bridge_side_point_count() <= 0
        ):
            marker = self._make_marker(
                0,
                Marker.TEXT_VIEW_FACING,
                "bridge_diagnostic",
                (1.0, 1.0, 1.0, 0.9),
                0.25,
            )
            marker.pose.position.x = self.pose[0] if self.pose is not None else 0.0
            marker.pose.position.y = self.pose[1] if self.pose is not None else 0.0
            marker.pose.position.z = 0.4
            marker.text = "no bridge geometry yet"
            marker_array.markers.append(marker)
        if entry_x is not None and entry_y is not None:
            entry = self._make_marker(4, Marker.CUBE, "bridge_entry_gate", (0.0, 1.0, 0.0, 0.65), 0.18)
            entry.pose.position.x = float(entry_x)
            entry.pose.position.y = float(entry_y)
            entry.pose.position.z = 0.05
            entry.scale.x = max(0.18, self.bridge_entry_gate_clear_width)
            entry.scale.y = 0.08
            entry.scale.z = 0.04
            marker_array.markers.append(entry)
            marker_array.markers.append(
                self._text_marker(
                    103,
                    "bridge_labels",
                    "entry gate free",
                    (entry_x, entry_y, 0.2),
                    (0.0, 1.0, 0.0, 1.0),
                )
            )
            if self.pose is not None:
                robot_line = self._line_marker(
                    5,
                    "robot_to_bridge_entry",
                    [(self.pose[0], self.pose[1], 0.0), (entry_x, entry_y, 0.0)],
                    (0.0, 0.8, 1.0, 0.8),
                )
                marker_array.markers.append(robot_line)
        pre_entry = self._bridge_pre_entry_xy()
        if pre_entry is not None:
            marker = self._make_marker(
                6,
                Marker.SPHERE,
                "bridge_pre_entry_staging",
                (0.0, 0.9, 1.0, 1.0),
                0.16,
            )
            marker.pose.position.x = pre_entry[0]
            marker.pose.position.y = pre_entry[1]
            marker.pose.position.z = 0.08
            marker_array.markers.append(marker)
            marker_array.markers.append(
                self._text_marker(
                    104,
                    "bridge_labels",
                    "pre-entry staging",
                    (pre_entry[0], pre_entry[1], 0.25),
                    (0.0, 0.9, 1.0, 1.0),
                )
            )
        if not self.bridge_geometry_quality_accepted and (left_line or right_line):
            marker_array.markers.append(
                self._text_marker(
                    105,
                    "bridge_labels",
                    "rejected geometry",
                    (self.pose[0], self.pose[1], 0.6) if self.pose is not None else (0.0, 0.0, 0.6),
                    (1.0, 1.0, 0.0, 1.0),
                )
            )

        marker_id = 20
        for obstacle in self.virtual_obstacles:
            if (
                obstacle.get("shape", self.virtual_obstacle_shape) == "disk"
                or self.virtual_obstacle_display_as_cylinder
            ):
                marker = self._make_marker(
                    marker_id, Marker.CYLINDER, "stuck_virtual_obstacles", (1.0, 0.35, 0.0, 0.7), obstacle["radius"] * 2.0
                )
                marker.pose.position.x = obstacle["x"]
                marker.pose.position.y = obstacle["y"]
                marker.pose.position.z = 0.05
                marker.scale.z = 0.1
            else:
                marker = self._line_marker(
                    marker_id,
                    "stuck_virtual_obstacles",
                    [
                        (obstacle.get("x0", obstacle["x"]), obstacle.get("y0", obstacle["y"]), 0.08),
                        (obstacle.get("x1", obstacle["x"]), obstacle.get("y1", obstacle["y"]), 0.08),
                    ],
                    (1.0, 0.35, 0.0, 0.9),
                )
                marker.scale.x = max(0.03, float(obstacle.get("line_width", self.virtual_obstacle_line_width)))
            marker_array.markers.append(marker)
            marker_id += 1

        self.bridge_marker_pub.publish(marker_array)
        self.bridge_marker_publish_count += 1
        self.bridge_marker_visible_reason = (
            "published"
            if marker_array.markers
            else "no marker geometry available"
        )

    def _line_marker(self, marker_id, ns, points, color):
        marker = self._make_marker(marker_id, Marker.LINE_STRIP, ns, color, 0.06)
        marker.points = []
        for point in points:
            p = Point()
            p.x = float(point[0])
            p.y = float(point[1])
            p.z = float(point[2]) if len(point) > 2 else 0.02
            marker.points.append(p)
        if len(marker.points) < 2:
            marker.action = Marker.DELETE
        return marker

    def _text_marker(self, marker_id, ns, text, point, color):
        marker = self._make_marker(marker_id, Marker.TEXT_VIEW_FACING, ns, color, 0.22)
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = float(point[2]) if len(point) > 2 else 0.25
        marker.text = text
        return marker

    def _publish_augmented_map(self):
        if self.map_msg is None:
            self.augmented_map_visible_reason = "no /map received"
            return
        msg = OccupancyGrid()
        msg.header = self.map_msg.header
        msg.info = self.map_msg.info
        data = list(self.map_msg.data)
        obstacle_cells = 0
        self.bridge_side_line_cells = 0
        self.virtual_obstacle_cells = 0
        self.entry_gate_clear_cells = 0
        self.center_corridor_clear_cells = 0
        self._fit_bridge_side_lines()
        if self.bridge_geometry_quality_accepted:
            bridge_line_width = self.bridge_side_line_width
            if self.augmented_map_debug_bridge_lines and self.map_msg.info.resolution > 0.0:
                bridge_line_width = max(
                    bridge_line_width,
                    self.map_msg.info.resolution
                    * max(1, self.augmented_map_bridge_line_debug_width_cells),
                )
            for side in ("left", "right"):
                cells = self._rasterize_polyline_obstacle(
                    data,
                    self.fitted_bridge_side_lines.get(side, []),
                    bridge_line_width,
                )
                self.bridge_side_line_cells += cells
                obstacle_cells += cells
            if (
                self.bridge_side_line_cells == 0
                and any(self.fitted_bridge_side_lines.get(side, []) for side in ("left", "right"))
            ):
                self.get_logger().error(
                    "bridge markers exist but augmented_map line rasterization produced zero cells"
                )
        for obstacle in self.virtual_obstacles:
            if obstacle.get("shape", self.virtual_obstacle_shape) == "disk":
                cells = self._mark_augmented_disk(
                    data, obstacle["x"], obstacle["y"], obstacle["radius"], skip_entry=False
                )
            else:
                cells = self._rasterize_line_obstacle(
                    data,
                    (obstacle.get("x0", obstacle["x"]), obstacle.get("y0", obstacle["y"]), 0.0),
                    (obstacle.get("x1", obstacle["x"]), obstacle.get("y1", obstacle["y"]), 0.0),
                    float(obstacle.get("line_width", self.virtual_obstacle_line_width)),
                )
            self.virtual_obstacle_cells += cells
            obstacle_cells += cells
        msg.data = data
        self.augmented_map_pub.publish(msg)
        self.augmented_map_publish_count += 1
        self.last_augmented_map_obstacle_cell_count = obstacle_cells
        self.augmented_map_visible_reason = (
            "published" if obstacle_cells > 0 else "published with no added obstacle cells"
        )

    def _fit_bridge_side_lines(self):
        left = self._fit_one_bridge_side_line("left_side_map_points")
        right = self._fit_one_bridge_side_line("right_side_map_points")
        self.fitted_bridge_side_lines = {"left": left, "right": right}
        accepted, reason = self._bridge_side_line_geometry_is_valid(left, right)
        if accepted and self._bridge_side_lines_jump_too_far(left, right):
            accepted = False
            reason = "fitted side lines jumped too far from last accepted geometry"
        self.bridge_geometry_quality_accepted = accepted
        self.bridge_geometry_quality_reason = reason
        if accepted:
            self.last_accepted_bridge_side_lines = {
                "left": list(left),
                "right": list(right),
            }
        return self.fitted_bridge_side_lines

    def _bridge_side_lines_jump_too_far(self, left, right):
        if self.bridge_geometry_max_jump <= 0.0:
            return False
        previous_left = self.last_accepted_bridge_side_lines.get("left", [])
        previous_right = self.last_accepted_bridge_side_lines.get("right", [])
        if len(previous_left) < 2 or len(previous_right) < 2:
            return False
        for current, previous in ((left, previous_left), (right, previous_right)):
            if len(current) < 2 or len(previous) < 2:
                return False
            endpoint_jump = max(
                distance_2d(current[0][:2], previous[0][:2]),
                distance_2d(current[-1][:2], previous[-1][:2]),
            )
            if endpoint_jump > self.bridge_geometry_max_jump:
                return True
        return False

    def _fit_one_bridge_side_line(self, key):
        if not self._bridge_side_has_enough_observations(key):
            return []
        raw = [
            p
            for p in self.bridge_landmark.get(key, [])
            if self._committed_bridge_side_point(p, key)
        ]
        if len(raw) < 2:
            return []
        points = self._reject_bridge_side_outliers(raw)
        points = self._sort_points_along_bridge_axis(points)
        if self.bridge_side_line_smoothing_enabled and len(points) >= 3:
            points = self._smooth_polyline(points)
        if self.bridge_side_line_fit_enabled:
            points = self._split_large_line_gaps(points)
        if self._polyline_length(points) < self.bridge_side_line_min_length:
            return []
        return points

    def _sort_points_along_bridge_axis(self, points):
        axis = self.bridge_landmark.get("forward_axis")
        entry = self._bridge_entry_xy() or (0.0, 0.0)
        if axis is None:
            return sorted(points, key=lambda p: distance_2d(entry, p[:2]))
        ax, ay = axis
        return sorted(
            points,
            key=lambda p: (float(p[0]) - entry[0]) * ax + (float(p[1]) - entry[1]) * ay,
        )

    def _reject_bridge_side_outliers(self, points):
        if len(points) < 3:
            return list(points)
        axis = self.bridge_landmark.get("forward_axis")
        entry = self._bridge_entry_xy()
        if axis is None or entry is None:
            return list(points)
        ax, ay = axis
        lateral_values = [
            -((float(p[0]) - entry[0]) * ay) + (float(p[1]) - entry[1]) * ax
            for p in points
        ]
        median_lateral = sorted(lateral_values)[len(lateral_values) // 2]
        return [
            p
            for p, lateral in zip(points, lateral_values)
            if abs(lateral - median_lateral) <= self.bridge_side_line_outlier_distance
        ]

    def _smooth_polyline(self, points):
        smoothed = [points[0]]
        for index in range(1, len(points) - 1):
            prev_p = points[index - 1]
            cur_p = points[index]
            next_p = points[index + 1]
            smoothed.append(
                (
                    (prev_p[0] + cur_p[0] + next_p[0]) / 3.0,
                    (prev_p[1] + cur_p[1] + next_p[1]) / 3.0,
                    (prev_p[2] + cur_p[2] + next_p[2]) / 3.0 if len(cur_p) > 2 else 0.0,
                )
            )
        smoothed.append(points[-1])
        return smoothed

    def _split_large_line_gaps(self, points):
        if len(points) < 2:
            return points
        filtered = [points[0]]
        for point in points[1:]:
            if distance_2d(filtered[-1][:2], point[:2]) <= self.bridge_side_line_max_point_gap:
                filtered.append(point)
        return filtered

    def _polyline_length(self, points):
        return sum(distance_2d(a[:2], b[:2]) for a, b in zip(points, points[1:]))

    def _bridge_side_line_geometry_is_valid(self, left, right):
        if len(left) < 2 or len(right) < 2:
            return False, "not enough fitted side-line points"
        left_length = self._polyline_length(left)
        right_length = self._polyline_length(right)
        if left_length < self.bridge_side_line_min_length or right_length < self.bridge_side_line_min_length:
            return False, "side line length too short"
        left_heading = math.atan2(left[-1][1] - left[0][1], left[-1][0] - left[0][0])
        right_heading = math.atan2(right[-1][1] - right[0][1], right[-1][0] - right[0][0])
        if abs(normalize_angle(left_heading - right_heading)) > self.bridge_geometry_parallel_angle_tolerance:
            return False, "side lines are not parallel enough"
        widths = [
            min(distance_2d(lp[:2], rp[:2]) for rp in right)
            for lp in left
        ]
        if not widths:
            return False, "could not estimate bridge width"
        median_width = sorted(widths)[len(widths) // 2]
        if not (self.bridge_landmark_min_width <= median_width <= self.bridge_landmark_max_width):
            return False, f"fitted bridge width {median_width:.2f}m out of range"
        if self.bridge_geometry_entry_must_lie_between_sides and not self._entry_between_side_lines(left, right):
            return False, "entry gate is outside fitted side corridor"
        if self.bridge_geometry_pre_entry_must_be_before_gate and not self._pre_entry_before_gate():
            return False, "pre-entry is not before bridge gate"
        return True, "accepted fitted bridge side lines"

    def _entry_between_side_lines(self, left, right):
        entry = self._bridge_entry_xy()
        axis = self.bridge_landmark.get("forward_axis")
        if entry is None or axis is None:
            return True
        ax, ay = axis
        left_lateral = sorted(
            -((p[0] - entry[0]) * ay) + (p[1] - entry[1]) * ax for p in left
        )[len(left) // 2]
        right_lateral = sorted(
            -((p[0] - entry[0]) * ay) + (p[1] - entry[1]) * ax for p in right
        )[len(right) // 2]
        return min(left_lateral, right_lateral) <= 0.0 <= max(left_lateral, right_lateral)

    def _pre_entry_before_gate(self):
        entry = self._bridge_entry_xy()
        pre_entry = self._bridge_pre_entry_xy()
        axis = self.bridge_landmark.get("forward_axis")
        if entry is None or pre_entry is None or axis is None:
            return True
        dx = pre_entry[0] - entry[0]
        dy = pre_entry[1] - entry[1]
        return dx * axis[0] + dy * axis[1] <= self.bridge_entry_gate_clear_radius

    def _rasterize_polyline_obstacle(self, data, points, width_m):
        count = 0
        for p0, p1 in zip(points, points[1:]):
            count += self._rasterize_line_obstacle(data, p0, p1, width_m)
        return count

    def _rasterize_line_obstacle(self, data, p0, p1, width_m):
        return self._mark_augmented_line_width(
            data, p0[0], p0[1], p1[0], p1[1], width_m
        )

    def _mark_augmented_line_width(self, data, x0, y0, x1, y1, width_m):
        if self.map_msg is None or self.map_msg.info.resolution <= 0.0:
            return 0
        length = distance_2d((x0, y0), (x1, y1))
        resolution = self.map_msg.info.resolution
        steps = max(1, int(length / max(resolution * 0.5, 0.01)))
        count = 0
        radius = max(width_m * 0.5, resolution * 0.5)
        min_radius = max(1, self.augmented_map_min_line_cells) * resolution
        radius = max(radius, min_radius)
        for index in range(steps + 1):
            t = index / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            count += self._mark_augmented_disk(
                data, x, y, radius, skip_entry=True
            )
        return count

    def _point_to_segment_distance(self, point, p0, p1):
        px, py = point
        x0, y0 = p0
        x1, y1 = p1
        dx = x1 - x0
        dy = y1 - y0
        denom = dx * dx + dy * dy
        if denom <= 1e-9:
            return distance_2d(point, p0)
        t = max(0.0, min(1.0, ((px - x0) * dx + (py - y0) * dy) / denom))
        return distance_2d(point, (x0 + dx * t, y0 + dy * t))

    def _mark_augmented_disk(self, data, x, y, radius, skip_entry=False):
        info = self.map_msg.info
        if info.resolution <= 0.0:
            return 0
        center_x = int((x - info.origin.position.x) / info.resolution)
        center_y = int((y - info.origin.position.y) / info.resolution)
        cells = max(1, int(radius / info.resolution))
        count = 0
        for gy in range(center_y - cells, center_y + cells + 1):
            for gx in range(center_x - cells, center_x + cells + 1):
                if gx < 0 or gy < 0 or gx >= info.width or gy >= info.height:
                    continue
                wx = info.origin.position.x + (gx + 0.5) * info.resolution
                wy = info.origin.position.y + (gy + 0.5) * info.resolution
                if skip_entry and self._near_bridge_entry_opening(wx, wy):
                    continue
                if skip_entry and self._inside_bridge_center_corridor(wx, wy):
                    continue
                if distance_2d((wx, wy), (x, y)) > radius:
                    continue
                index = gy * info.width + gx
                if data[index] <= self.map_free_threshold or data[index] < 0:
                    data[index] = 100
                    count += 1
        return count

    def _apply_stuck_recovery(self, action_key):
        if not self.stuck_recovery_enabled or self.pose is None:
            self._reset_motion_monitor()
            return action_key

        if self.stuck_recovery_phase is not None:
            return self._stuck_recovery_action()

        if self._is_deliberate_scan_rotation(action_key):
            self._reset_motion_monitor()
            return action_key

        if not self._stuck_monitor_enabled_for_action(action_key):
            self._reset_motion_monitor()
            return action_key

        if (
            self.last_stuck_recovery_time is not None
            and self._elapsed_seconds(self.last_stuck_recovery_time)
            < self.stuck_recovery_cooldown
        ):
            self._start_motion_monitor(action_key)
            return action_key

        if self.motion_monitor_action != action_key or self.motion_monitor_pose is None:
            self._start_motion_monitor(action_key)
            return action_key

        if self._motion_progressed(action_key):
            self._start_motion_monitor(action_key)
            return action_key

        if self._elapsed_seconds(self.motion_monitor_start_time) < self.stuck_timeout:
            return action_key

        self._start_stuck_recovery(action_key)
        return "STOP"

    def _is_deliberate_scan_rotation(self, action_key):
        if action_key not in (
            "CLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
        ):
            return False
        if self.current_goal_name is not None:
            return False
        return self.state in (
            MissionState.EXPLORE_MAP,
            MissionState.SEARCH_BEAR,
            MissionState.TASK2_SEARCH_BRIDGE,
        )

    def _stuck_monitor_enabled_for_action(self, action_key):
        return action_key in {
            "FORWARD",
            "FORWARD_SLOW",
            "BACKWARD",
            "BACKWARD_SLOW",
            "LEFT_FRONT",
            "RIGHT_FRONT",
            "LEFT_SHIFT",
            "RIGHT_SHIFT",
            "CLOCKWISE_ROTATION",
            "CLOCKWISE_ROTATION_SLOW",
            "CLOCKWISE_ROTATION_MEDIAN",
            "COUNTERCLOCKWISE_ROTATION",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_MEDIAN",
        }

    def _start_motion_monitor(self, action_key):
        self.motion_monitor_action = action_key
        self.motion_monitor_pose = tuple(self.pose)
        self.motion_monitor_start_time = self.get_clock().now()

    def _motion_progressed(self, action_key):
        if self.motion_monitor_pose is None or self.pose is None:
            return True

        translation = distance_2d(self.pose[:2], self.motion_monitor_pose[:2])
        yaw_change = abs(normalize_angle(self.pose[2] - self.motion_monitor_pose[2]))
        if self._action_is_rotation(action_key):
            return yaw_change >= self.stuck_min_yaw_change
        return translation >= self.stuck_min_translation

    def _action_is_rotation(self, action_key):
        return action_key in {
            "CLOCKWISE_ROTATION",
            "CLOCKWISE_ROTATION_SLOW",
            "CLOCKWISE_ROTATION_MEDIAN",
            "COUNTERCLOCKWISE_ROTATION",
            "COUNTERCLOCKWISE_ROTATION_SLOW",
            "COUNTERCLOCKWISE_ROTATION_MEDIAN",
        }

    def _start_stuck_recovery(self, action_key):
        self.stuck_recovery_last_action = action_key
        self.stuck_recovery_turn_direction = self._recovery_turn_direction(action_key)
        self.stuck_recovery_shift_direction *= -1.0
        self.stuck_recovery_phase = "stop"
        self.stuck_recovery_phase_start_time = self.get_clock().now()
        self._mark_virtual_obstacle_from_action(action_key)
        self._reset_motion_monitor()
        if self.current_goal_name is not None:
            self._clear_navigation()
        self.get_logger().warn(
            f"Possible stuck while executing {action_key}; backing up and realigning."
        )

    def _recovery_turn_direction(self, action_key):
        if action_key.startswith("CLOCKWISE"):
            return -1.0
        if action_key.startswith("COUNTERCLOCKWISE"):
            return 1.0
        if action_key in ("RIGHT_FRONT", "RIGHT_SHIFT"):
            return -1.0
        if action_key in ("LEFT_FRONT", "LEFT_SHIFT"):
            return 1.0
        return -1.0 if self.last_drivable_delta_sign > 0.0 else 1.0

    def _stuck_recovery_action(self):
        elapsed = self._elapsed_seconds(self.stuck_recovery_phase_start_time)

        if self.stuck_recovery_phase == "stop":
            if elapsed < self.stuck_recovery_stop_seconds:
                return "STOP"
            self._advance_stuck_recovery_phase("back")
            return "BACKWARD_SLOW"

        if self.stuck_recovery_phase == "back":
            if elapsed < self.stuck_recovery_back_seconds:
                return "BACKWARD_SLOW"
            self._advance_stuck_recovery_phase("turn")
            return self._stuck_recovery_turn_action()

        if self.stuck_recovery_phase == "turn":
            if elapsed < self.stuck_recovery_turn_seconds:
                return self._stuck_recovery_turn_action()
            self._advance_stuck_recovery_phase("shift")
            return self._stuck_recovery_shift_action()

        if self.stuck_recovery_phase == "shift":
            if elapsed < self.stuck_recovery_shift_seconds:
                return self._stuck_recovery_shift_action()
            self._reset_stuck_recovery()
            return "STOP"

        self._reset_stuck_recovery()
        return "STOP"

    def _advance_stuck_recovery_phase(self, phase):
        self.stuck_recovery_phase = phase
        self.stuck_recovery_phase_start_time = self.get_clock().now()

    def _stuck_recovery_turn_action(self):
        return (
            "CLOCKWISE_ROTATION_SLOW"
            if self.stuck_recovery_turn_direction > 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _stuck_recovery_shift_action(self):
        return "RIGHT_SHIFT" if self.stuck_recovery_shift_direction > 0.0 else "LEFT_SHIFT"

    def _reset_motion_monitor(self):
        self.motion_monitor_action = None
        self.motion_monitor_pose = None
        self.motion_monitor_start_time = None

    def _reset_stuck_recovery(self):
        self.last_stuck_recovery_time = self.get_clock().now()
        self.stuck_recovery_phase = None
        self.stuck_recovery_phase_start_time = None
        self.stuck_recovery_last_action = None
        self._reset_motion_monitor()

    def _publish_goal(self, goal):
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(goal[0])
        msg.pose.position.y = float(goal[1])
        msg.pose.position.z = 0.0
        msg.pose.orientation.z = math.sin(goal[2] / 2.0)
        msg.pose.orientation.w = math.cos(goal[2] / 2.0)
        self.goal_pub.publish(msg)

    def _publish_action(self, action_key):
        action_key = self._segmentation_guard_action(action_key)
        action_key = self._avoid_virtual_obstacle_for_action(action_key)
        action_key = self._apply_stuck_recovery(action_key)
        velocities = ACTION_MAPPINGS.get(action_key, ACTION_MAPPINGS["STOP"])

        rear_msg = Float32MultiArray()
        rear_msg.data = [float(velocities[0]), float(velocities[1])]
        self.rear_pub.publish(rear_msg)

        front_msg = Float32MultiArray()
        front_msg.data = [float(velocities[2]), float(velocities[3])]
        self.front_pub.publish(front_msg)
        now = self.get_clock().now()
        if (
            self.last_logged_action != action_key
            or self.last_action_log_time is None
            or self._elapsed_seconds(self.last_action_log_time) >= 1.0
        ):
            self.last_logged_action = action_key
            self.last_action_log_time = now
            self._log_action(action_key)

    def _publish_arm_positions(self, positions):
        msg = JointTrajectoryPoint()
        if self.arm_positions_in_degrees:
            msg.positions = [math.radians(float(value)) for value in positions]
        else:
            msg.positions = [float(value) for value in positions]
        msg.velocities = [0.0] * len(msg.positions)
        self.arm_pub.publish(msg)

    def _publish_target_label(self):
        msg = String()
        msg.data = self.target_label
        self.target_label_pub.publish(msg)

    def _publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.state_pub.publish(msg)

    def _set_state(self, state, reason=""):
        if self.state == state:
            return
        old_state = self.state
        self.state = state
        self.state_start_time = self.get_clock().now()
        self.last_drivable_action = None
        self.last_drivable_action_time = None
        self.last_drivable_anchor_time = None
        self._reset_motion_monitor()
        self.stuck_recovery_phase = None
        self.stuck_recovery_phase_start_time = None
        if state != MissionState.RETURN_START:
            self._reset_return_hold_monitor()
        if state in (
            MissionState.TASK2_SEARCH_BRIDGE,
            MissionState.TASK2_EXPLORE_FOR_BRIDGE,
            MissionState.TASK2_TURN_TO_BRIDGE,
            MissionState.TASK2_SIDE_VIEW_RECOVERY,
            MissionState.TASK2_APPROACH_BRIDGE_ENTRY,
            MissionState.TASK2_FINAL_ALIGN_BRIDGE,
            MissionState.TASK2_ASCEND_BRIDGE,
            MissionState.TASK2_SEARCH_BRIDGE_BEAR,
            MissionState.TASK2_DESCEND_BRIDGE,
        ):
            self.task2_phase_start_time = None
            self.task2_bridge_confirm_start_time = None
            self.task2_search_cycle_start_time = None
            self._reset_task2_bridge_runtime()
        if state == MissionState.TASK2_SEARCH_BRIDGE:
            self._reset_task2_exploration_segment()
            self._reset_bridge_turn_controller()
        if state == MissionState.TASK2_EXPLORE_FOR_BRIDGE:
            self._reset_task2_scan()
        if state == MissionState.TASK2_TURN_TO_BRIDGE:
            self._reset_bridge_turn_controller()
        if state == MissionState.TASK2_APPROACH_BRIDGE_ENTRY:
            self._reset_task2_entry_blocked_recovery()
            self.task2_entry_close_confirm_count = 0
            self.task2_entry_close_last_reason = ""
            self.task2_final_align_confirm_count = 0
            self.task2_final_align_loss_start_time = None
            self.task2_final_align_close_loss_start_time = None
        if state == MissionState.TASK2_FINAL_ALIGN_BRIDGE:
            self._reset_task2_entry_blocked_recovery()
            self.task2_final_align_confirm_count = 0
            self.task2_final_align_loss_start_time = None
            self.task2_final_align_close_loss_start_time = None
        if state != MissionState.TASK2_ASCEND_BRIDGE:
            self.task2_ascent_stop_start_time = None
        self._publish_state()
        self._log_state_transition(old_state, self.state, reason=reason)
        self.get_logger().info(f"Mission state -> {self.state.value}")
        if self.state == MissionState.DONE:
            self._log_mission_summary()


def main(args=None):
    rclpy.init(args=args)
    node = Task1MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_action("STOP")
        node._log_mission_summary()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

import ast
import math
from enum import Enum

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Float32MultiArray, String
from trajectory_msgs.msg import JointTrajectoryPoint

from pros_car_py.car_models import DeviceDataTypeEnum
from pros_car_py.ros_communicator_config import ACTION_MAPPINGS


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
    TASK2_TURN_TO_BRIDGE = "task2_turn_to_bridge"
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
        self.declare_parameter("task2_use_bridge_entry_pose", False)
        self.declare_parameter("task2_bridge_entry_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("task2_use_bridge_top_pose", False)
        self.declare_parameter("task2_bridge_top_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("task2_use_bridge_exit_pose", False)
        self.declare_parameter("task2_bridge_exit_pose", [0.0, 0.0, 0.0])
        self.declare_parameter("task2_ascent_min_seconds", 6.0)
        self.declare_parameter("task2_ascent_timeout_seconds", 10.0)
        self.declare_parameter("task2_descent_min_seconds", 6.0)
        self.declare_parameter("task2_descent_timeout_seconds", 10.0)
        self.declare_parameter("task2_bridge_search_timeout_seconds", 120.0)
        self.declare_parameter("task2_bridge_confirm_seconds", 0.4)
        self.declare_parameter("task2_bridge_min_area_ratio", 0.008)
        self.declare_parameter("task2_bridge_entry_bottom_coverage", 0.04)
        self.declare_parameter("task2_bridge_center_tolerance", 90.0)
        self.declare_parameter("task2_bridge_candidate_min_bottom_y_ratio", 0.48)
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
        self.declare_parameter("task2_bridge_turn_confirm_seconds", 0.3)
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
        self.declare_parameter("task2_bridge_orbit_turn_seconds", 0.6)
        self.declare_parameter("task2_bridge_orbit_max_cycles", 5)
        self.declare_parameter("task2_bridge_lost_grace_seconds", 0.6)
        self.declare_parameter("task2_bridge_approach_timeout_seconds", 25.0)
        self.declare_parameter("task2_bridge_no_entry_progress_timeout_seconds", 4.0)
        self.declare_parameter("task2_bridge_progress_min_bottom_y_delta", 0.03)
        self.declare_parameter("task2_bridge_progress_min_coverage_delta", 0.025)
        self.declare_parameter("task2_road_search_spin_seconds", 1.5)
        self.declare_parameter("task2_road_search_drive_seconds", 6.0)
        self.declare_parameter("use_segmentation_drivable_guard", False)
        self.declare_parameter("segmentation_timeout_seconds", 1.5)
        self.declare_parameter("segmentation_missing_grace_seconds", 0.8)
        self.declare_parameter("segmentation_smoothing_alpha", 0.45)
        self.declare_parameter("drivable_center_tolerance", 110.0)
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
        self.declare_parameter("task1_require_target_on_road", True)
        self.declare_parameter("task1_allow_target_without_road_mask", False)
        self.declare_parameter("task1_road_target_min_area_ratio", 0.005)
        self.declare_parameter("task1_road_target_x_tolerance_pixels", 180.0)
        self.declare_parameter("task1_road_target_max_width_ratio", 0.58)
        self.declare_parameter("task1_road_target_y_margin_ratio", 0.08)
        self.declare_parameter("task1_road_target_log_period_seconds", 1.0)
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
        self.segmentation_info = None
        self.segmentation_info_stamp = None
        self.segmentation_connection = None
        self.segmentation_last_seen = {"road": None, "bridge": None}
        self.last_drivable_action = None
        self.last_drivable_action_time = None
        self.last_drivable_delta_sign = 1.0
        self.last_drivable_anchor_time = None
        self.current_task = 1
        self.next_state_after_grab = MissionState.RETURN_START
        self.task2_phase_start_time = None
        self.task2_bridge_confirm_start_time = None
        self.task2_search_cycle_start_time = None
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

        self.state = MissionState.EXPLORE_MAP if self.auto_explore else MissionState.GO_TO_BEAR_AREA
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
            "/yolo/segmentation_info",
            self._segmentation_info_callback,
            10,
        )

        control_period = self._double_param("control_period_seconds")
        self.timer = self.create_timer(control_period, self._control_loop)

        self._publish_target_label()
        self._publish_state()
        self.get_logger().info(
            "Task 1 controller ready. It will explore the map until a bear is found."
            if self.auto_explore
            else "Task 1 controller ready. Set start_pose and bear_search_pose before "
            "the final run."
        )

    def _double_param(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def _integer_param(self, name):
        return self.get_parameter(name).get_parameter_value().integer_value

    def _bool_param(self, name):
        return self.get_parameter(name).get_parameter_value().bool_value

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

    def _amcl_pose_callback(self, msg):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        self._set_current_pose(
            (position.x, position.y, yaw_from_quaternion(orientation)),
            source="/amcl_pose",
        )

    def _set_current_pose(self, pose, source):
        self.pose = pose
        if not self.start_pose_locked:
            self.start_pose = [self.pose[0], self.pose[1], self.pose[2]]
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

    def _filtered_segment_update(self, label, segment, now):
        prev = None if self.segmentation_info is None else self.segmentation_info.get(label)
        alpha = min(1.0, max(0.0, self.segmentation_smoothing_alpha))
        found = bool(segment["found"])

        if found:
            self.segmentation_last_seen[label] = now

        if prev is None:
            return {
                "found": found,
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
        return {
            "found": found,
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
        }

    def _control_loop(self):
        self._publish_startup_arm_stow_if_needed()
        self._publish_target_label()
        self._publish_state()
        self._publish_initial_pose_if_needed()
        self._update_pose_from_tf()

        if self.pose is None:
            self._publish_action("STOP")
            self._log_waiting_for_pose()
            return

        if self._should_interrupt_search_for_target():
            self._publish_action("STOP")
            self._clear_navigation()
            self.exploration_scan_until_time = None
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
        elif self.state == MissionState.TASK2_TURN_TO_BRIDGE:
            self._task2_turn_to_bridge()
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
        return 0 <= value <= self.map_free_threshold

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

        if distance > 0.0 and distance <= self.observe_distance:
            self._publish_action("STOP")
            self.observe_start_time = self.get_clock().now()
            self.observe_start_pose = self.pose
            self.last_observed_target_time = self.observe_start_time
            self.last_observed_target_distance = distance
            self._set_state(MissionState.OBSERVE_BEAR)
            return

        self._publish_action("FORWARD_SLOW")

    def _observe_bear(self):
        self._publish_action("STOP")

        if self._mission_target_visible():
            self.last_observed_target_time = self.get_clock().now()
            self.last_observed_target_distance = self.yolo_target["distance"]
        elif (
            self.last_observed_target_time is None
            or self._elapsed_seconds(self.last_observed_target_time)
            > self.observe_target_loss_grace
        ):
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
            self.get_logger().warn("Grab approach timed out; attempting arm recovery now.")
            self.pre_grab_bbox = self._copy_current_bbox()
            self._set_state(MissionState.SECURE_BEAR)
            return

        if not self._mission_target_visible():
            if self._last_grab_target_was_close():
                self._publish_action("STOP")
                self.pre_grab_bbox = self._copy_current_bbox()
                self._set_state(MissionState.SECURE_BEAR)
            else:
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

        if distance > 0.0 and distance <= self.grab_distance:
            self._publish_action("STOP")
            if self.grab_ready_start_time is None:
                self.grab_ready_start_time = self.get_clock().now()
                return
            if self._elapsed_seconds(self.grab_ready_start_time) >= self.grab_confirm_seconds:
                self.pre_grab_bbox = self._copy_current_bbox()
                self._set_state(MissionState.SECURE_BEAR)
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
        if self.current_task == 1 and self.run_task2_after_task1:
            self._start_task2()
            return
        self._set_state(MissionState.DONE)

    def _reset_drop_state(self):
        self.drop_start_time = None
        self.drop_step_index = 0
        self.drop_step_sent = False
        self.drop_step_deadline = None

    def _start_task2(self):
        self.current_task = 2
        self.next_state_after_grab = MissionState.TASK2_DESCEND_BRIDGE
        self.task2_phase_start_time = None
        self.task2_bridge_confirm_start_time = None
        self.task2_search_cycle_start_time = None
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
            self._task2_reset_bridge_confirm()
            self.get_logger().info("Task 2: searching for bridge mask.")

        bridge = self._task2_bridge_visible()
        if bridge is None:
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
            return

        if self._task2_bridge_has_entry_candidate(bridge):
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: bridge entry candidate detected, turning toward bridge."
            )
            self._set_state(MissionState.TASK2_TURN_TO_BRIDGE)
            return

        self.get_logger().info(
            "Task 2: bridge visible but entry candidate not visible; "
            "orbiting for better viewpoint."
        )
        action = self._task2_bridge_orbit_search_action(bridge)
        if action is None:
            self._publish_action("STOP")
            self._reset_task2_bridge_runtime()
            return
        self._publish_action(action)

    def _task2_turn_to_bridge(self):
        bridge = self._task2_bridge_visible()
        if bridge is None:
            self._publish_action("STOP")
            self.get_logger().warn("Task 2: bridge lost while turning; searching again.")
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        if not self._task2_bridge_has_entry_candidate(bridge):
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: bridge visible but entry candidate disappeared; orbit search."
            )
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        delta_x = self._task2_bridge_delta("bottom_center_x", bridge)
        if delta_x is None:
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
            self._task2_reset_bridge_confirm()
            return

        if abs(delta_x) <= self.task2_bridge_turn_tolerance:
            if self.task2_bridge_confirm_start_time is None:
                self.task2_bridge_confirm_start_time = self.get_clock().now()
                self._publish_action("STOP")
                return
            if (
                self._elapsed_seconds(self.task2_bridge_confirm_start_time)
                >= self.task2_bridge_turn_confirm_seconds
            ):
                self._publish_action("STOP")
                self.get_logger().info(
                    "Task 2: bridge centered enough; approaching bridge entry."
                )
                self._set_state(MissionState.TASK2_APPROACH_BRIDGE_ENTRY)
                return
            self._publish_action("STOP")
            return

        self._task2_reset_bridge_confirm()
        self._publish_action(
            "CLOCKWISE_ROTATION_SLOW"
            if delta_x > 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _task2_approach_bridge_entry(self):
        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()

        bridge = self._task2_bridge_visible()
        if bridge is None:
            if self._task2_bridge_recently_seen():
                self._publish_action("STOP")
                return
            self._publish_action("STOP")
            self.get_logger().warn("Task 2: bridge lost during approach; searching again.")
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        if not self._task2_bridge_has_entry_candidate(bridge):
            self.get_logger().info(
                "Task 2: bridge visible but lower entry is not usable; orbiting."
            )
            action = self._task2_bridge_orbit_search_action(bridge)
            if action is None:
                self._publish_action("STOP")
                self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
                return
            self._publish_action(action)
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
                "Task 2: bridge approach timeout; orbiting for a better entry view."
            )
            action = self._task2_bridge_orbit_search_action(bridge)
            if action is None:
                self._publish_action("STOP")
                self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
                return
            self._publish_action(action)
            return

        if (
            self.task2_bridge_best_entry_time is not None
            and self._elapsed_seconds(self.task2_bridge_best_entry_time)
            >= self.task2_bridge_no_entry_progress_timeout_seconds
        ):
            self.get_logger().warn("Task 2: no entry progress; running orbit recovery.")
            action = self._task2_bridge_orbit_search_action(bridge)
            if action is None:
                self._publish_action("STOP")
                self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
                return
            self._publish_action(action)
            return

        self._publish_action(
            self._task2_bridge_center_action(
                bridge=bridge,
                preferred_key="bottom_center_x",
                hard_tolerance=self.task2_bridge_approach_hard_tolerance,
                soft_tolerance=self.task2_bridge_entry_center_tolerance,
                centered_action="FORWARD_SLOW",
                allow_arc=True,
            )
        )

    def _task2_final_align_bridge(self):
        bridge = self._task2_bridge_visible()
        if bridge is None:
            if self._task2_bridge_recently_seen():
                self._publish_action("STOP")
                return
            self._publish_action("STOP")
            self.get_logger().warn(
                "Task 2: bridge lost during final alignment; searching again."
            )
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE)
            return

        if not self._task2_bridge_entry_close(bridge):
            self._publish_action("STOP")
            self.get_logger().info(
                "Task 2: bridge entry no longer close; approaching again."
            )
            self._set_state(MissionState.TASK2_APPROACH_BRIDGE_ENTRY)
            return

        delta_x = self._task2_bridge_delta("bottom_center_x", bridge)
        if delta_x is None:
            self._task2_reset_bridge_confirm()
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
            return

        if abs(delta_x) <= self.task2_bridge_entry_final_tolerance:
            if self.task2_bridge_confirm_start_time is None:
                self.task2_bridge_confirm_start_time = self.get_clock().now()
                self._publish_action("STOP")
                return
            if (
                self._elapsed_seconds(self.task2_bridge_confirm_start_time)
                >= self.task2_bridge_entry_confirm_seconds
            ):
                self._publish_action("STOP")
                self.get_logger().info(
                    "Task 2: final entry alignment confirmed; starting ascent."
                )
                self._set_state(MissionState.TASK2_ASCEND_BRIDGE)
                return
            self._publish_action("STOP")
            return

        self._task2_reset_bridge_confirm()
        self._publish_action(
            "CLOCKWISE_ROTATION_SLOW"
            if delta_x > 0.0
            else "COUNTERCLOCKWISE_ROTATION_SLOW"
        )

    def _task2_ascend_bridge(self):
        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()
            self.get_logger().info(
                "Task 2 ascent: timed bridge climb with visual correction."
            )

        if self.task2_use_bridge_top_pose:
            self._navigate_state(
                goal_name="task2_bridge_top_pose",
                goal=self.task2_bridge_top_pose,
                next_state=MissionState.TASK2_SEARCH_BRIDGE_BEAR,
            )
            return

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        if elapsed >= self.task2_ascent_min_seconds:
            self._publish_action("STOP")
            self.task2_phase_start_time = None
            self._set_state(MissionState.TASK2_SEARCH_BRIDGE_BEAR)
            return

        bridge = self._task2_bridge_visible()
        if bridge is None:
            self._publish_action("FORWARD_SLOW")
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

    def _task2_search_bridge_bear(self):
        self.next_state_after_grab = MissionState.TASK2_DESCEND_BRIDGE
        if self._target_visible():
            self._publish_action("STOP")
            self.task2_phase_start_time = None
            self._set_state(MissionState.APPROACH_BEAR)
            return

        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        if elapsed < 2.0:
            self._publish_action("CLOCKWISE_ROTATION_SLOW")
        elif elapsed < 4.0:
            self._publish_action("COUNTERCLOCKWISE_ROTATION_SLOW")
        elif elapsed < 6.0:
            self._publish_action("FORWARD_SLOW")
        else:
            self.task2_phase_start_time = None
            self._publish_action("STOP")

    def _task2_descend_bridge(self):
        if self.task2_phase_start_time is None:
            self.task2_phase_start_time = self.get_clock().now()
            self._clear_navigation()
            self.get_logger().info("Task 2 descent: timed bridge descent.")

        if self.task2_use_bridge_exit_pose:
            self._navigate_state(
                goal_name="task2_bridge_exit_pose",
                goal=self.task2_bridge_exit_pose,
                next_state=MissionState.RETURN_START,
            )
            return

        elapsed = self._elapsed_seconds(self.task2_phase_start_time)
        if elapsed >= self.task2_descent_min_seconds:
            self._publish_action("STOP")
            self.task2_phase_start_time = None
            self.next_state_after_grab = MissionState.RETURN_START
            self._set_state(MissionState.RETURN_START)
            return

        bridge = self._task2_bridge_visible()
        if bridge is None:
            self._publish_action("FORWARD_SLOW")
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
            f"({reason}); restarting from secure_bear."
        )
        self._set_state(MissionState.SECURE_BEAR)

    def _handle_grab_verification_failure(self, reason):
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

    def _task2_bridge_visible(self):
        bridge = self._segmentation_segment("bridge")
        if bridge is None or not bridge.get("found", False):
            return None
        if float(bridge.get("area_ratio", 0.0)) < self.task2_bridge_detect_min_area_ratio:
            return None
        self.task2_bridge_last_seen_time = self.get_clock().now()
        return bridge

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
            bridge = self._task2_bridge_visible()
        if bridge is None:
            return False

        bottom_delta = self._task2_bridge_delta("bottom_center_x", bridge)
        if (
            bottom_delta is None
            or abs(bottom_delta) > self.task2_bridge_approach_hard_tolerance
        ):
            return False

        close_votes = 0
        if (
            float(bridge.get("bottom_y_ratio", 0.0))
            >= self.task2_bridge_entry_close_bottom_y_ratio
        ):
            close_votes += 1
        if (
            float(bridge.get("bottom_coverage", 0.0))
            >= self.task2_bridge_entry_close_bottom_coverage
        ):
            close_votes += 1
        if (
            float(bridge.get("area_ratio", 0.0))
            >= self.task2_bridge_entry_close_area_ratio
        ):
            close_votes += 1
        return close_votes >= 2

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
        self._task2_reset_orbit()

    def _task2_reset_bridge_confirm(self):
        self.task2_bridge_confirm_start_time = None

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

    def _segmentation_segment(self, label):
        if self.segmentation_info is None or self.segmentation_info_stamp is None:
            return None
        if self._elapsed_seconds(self.segmentation_info_stamp) > self.segmentation_timeout:
            return None
        segment = self.segmentation_info.get(label)
        if segment is None:
            return None
        if segment["found"]:
            return segment
        last_seen = self.segmentation_last_seen.get(label)
        if last_seen is not None and self._elapsed_seconds(last_seen) <= self.segmentation_missing_grace:
            return segment
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
        action_key = self._apply_stuck_recovery(action_key)
        velocities = ACTION_MAPPINGS.get(action_key, ACTION_MAPPINGS["STOP"])

        rear_msg = Float32MultiArray()
        rear_msg.data = [float(velocities[0]), float(velocities[1])]
        self.rear_pub.publish(rear_msg)

        front_msg = Float32MultiArray()
        front_msg.data = [float(velocities[2]), float(velocities[3])]
        self.front_pub.publish(front_msg)

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

    def _set_state(self, state):
        if self.state == state:
            return
        self.state = state
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
            MissionState.TASK2_TURN_TO_BRIDGE,
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
        self._publish_state()
        self.get_logger().info(f"Mission state -> {self.state.value}")


def main(args=None):
    rclpy.init(args=args)
    node = Task1MissionController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_action("STOP")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

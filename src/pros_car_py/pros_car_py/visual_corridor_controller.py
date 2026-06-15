class VisualCorridorController:
    """Small PID-like visual corridor follower with action hold hysteresis.

    Positive error means the drivable corridor is to the right of image center.
    """

    FORWARD_ACTIONS = {
        "FORWARD",
        "FORWARD_SLOW",
        "ASCEND_FORWARD",
        "LEFT_FRONT",
        "RIGHT_FRONT",
    }

    def __init__(
        self,
        kp=0.018,
        ki=0.0,
        kd=0.004,
        max_error=160.0,
        forward_deadband=45.0,
        rotate_deadband=120.0,
        action_hold_seconds=0.2,
        min_continuous_score=0.55,
        min_bottom_width_ratio=0.12,
        min_center_width_ratio=0.08,
    ):
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.max_error = max(1.0, float(max_error))
        self.forward_deadband = max(0.0, float(forward_deadband))
        self.rotate_deadband = max(self.forward_deadband, float(rotate_deadband))
        self.action_hold_seconds = max(0.0, float(action_hold_seconds))
        self.min_continuous_score = max(0.0, float(min_continuous_score))
        self.min_bottom_width_ratio = max(0.0, float(min_bottom_width_ratio))
        self.min_center_width_ratio = max(0.0, float(min_center_width_ratio))
        self.integral = 0.0
        self.previous_error = None
        self.smoothed_error = None
        self.held_action = None
        self.hold_remaining = 0.0

    def reset(self):
        self.integral = 0.0
        self.previous_error = None
        self.smoothed_error = None
        self.held_action = None
        self.hold_remaining = 0.0

    def update(self, corridor_info, dt, desired_forward="FORWARD_SLOW", mode="road"):
        dt = max(0.001, float(dt))
        info = corridor_info or {}
        error = self._error(info)
        pid = self._pid(error, dt) if self._has_any_mask(info) else 0.0

        action, allow_forward, reason = self._choose_action(
            info, error, desired_forward, mode
        )
        action = self._hold_action(action, dt)
        return {
            "action": action,
            "allow_forward": allow_forward,
            "pid_output": pid,
            "reason": reason,
        }

    def _choose_action(self, info, error, desired_forward, mode):
        valid = bool(info.get("valid", False))
        bottom_connected = bool(info.get("bottom_connected", False))
        centerline_reached = bool(info.get("centerline_reached", False))
        side_view = bool(info.get("side_view_likely", False))
        score = float(info.get("continuous_score", 0.0))
        bottom_width = float(info.get("bottom_width_ratio", 0.0))
        center_width = float(info.get("center_width_ratio", 0.0))
        drivable_type = int(float(info.get("drivable_type", 0)))
        desired_forward = desired_forward or "FORWARD_SLOW"

        bridge_mode = mode in ("bridge_entry", "ascent")
        bridge_ok = drivable_type in (2, 3) if bridge_mode else drivable_type in (1, 3)
        if mode == "return":
            bridge_ok = drivable_type in (1, 2, 3)

        ready = (
            valid
            and bottom_connected
            and centerline_reached
            and score >= self.min_continuous_score
            and bottom_width >= self.min_bottom_width_ratio
            and center_width >= self.min_center_width_ratio
            and not side_view
            and bridge_ok
        )
        abs_error = abs(error)
        if ready and abs_error <= self.forward_deadband:
            if mode == "ascent":
                desired_forward = "ASCEND_FORWARD"
            return desired_forward, True, "corridor centered and continuous"
        if ready and abs_error <= self.rotate_deadband:
            return (
                "RIGHT_FRONT" if error > 0.0 else "LEFT_FRONT",
                True,
                "corridor valid but needs arc correction",
            )
        if side_view:
            self._reset_pid_memory()
            return self._rotation_from_error(error), False, "side-view corridor rejected"
        if self._has_bottom_mask(info):
            return self._rotation_from_error(error), False, "bottom mask seen but corridor not continuous"
        if self._has_any_mask(info):
            return self._rotation_from_error(error), False, "mask seen away from lower frame"

        self._reset_pid_memory()
        return "CLOCKWISE_ROTATION_SLOW", False, "no drivable corridor visible"

    def _pid(self, error, dt):
        clipped = max(-self.max_error, min(self.max_error, float(error)))
        if self.smoothed_error is None:
            self.smoothed_error = clipped
        else:
            self.smoothed_error = 0.65 * self.smoothed_error + 0.35 * clipped
        self.integral += self.smoothed_error * dt
        limit = self.max_error * 0.5
        self.integral = max(-limit, min(limit, self.integral))
        derivative = 0.0
        if self.previous_error is not None:
            derivative = (self.smoothed_error - self.previous_error) / dt
        self.previous_error = self.smoothed_error
        return (
            self.kp * self.smoothed_error
            + self.ki * self.integral
            + self.kd * derivative
        )

    def _hold_action(self, action, dt):
        if self.held_action is not None and self.hold_remaining > 0.0:
            if action != self.held_action:
                self.hold_remaining = max(0.0, self.hold_remaining - dt)
                return self.held_action
        if action != self.held_action:
            self.held_action = action
            self.hold_remaining = self.action_hold_seconds
        else:
            self.hold_remaining = max(0.0, self.hold_remaining - dt)
        return action

    def _rotation_from_error(self, error):
        return "CLOCKWISE_ROTATION_SLOW" if error >= 0.0 else "COUNTERCLOCKWISE_ROTATION_SLOW"

    def _error(self, info):
        if "error_x" in info:
            return float(info.get("error_x", 0.0))
        return float(info.get("corridor_error_x_pixels", 0.0))

    def _has_bottom_mask(self, info):
        return bool(info.get("bottom_connected", False)) or float(
            info.get("bottom_width_ratio", 0.0)
        ) > 0.02

    def _has_any_mask(self, info):
        return self._has_bottom_mask(info) or float(
            info.get("center_width_ratio", 0.0)
        ) > 0.02 or bool(info.get("valid", False))

    def _reset_pid_memory(self):
        self.integral = 0.0
        self.previous_error = None
        self.smoothed_error = None


def corridor_from_msg(data):
    values = list(data or [])
    if len(values) < 19:
        return {"valid": False, "reason": "missing /yolo/drivable_corridor_info"}
    return {
        "valid": values[0] >= 0.5,
        "bottom_connected": values[1] >= 0.5,
        "centerline_reached": values[2] >= 0.5,
        "upper_mid_reached": values[3] >= 0.5,
        "continuous_score": float(values[4]),
        "bottom_width_ratio": float(values[5]),
        "center_width_ratio": float(values[6]),
        "center_x_bottom": float(values[7]),
        "center_x_mid": float(values[8]),
        "center_x_centerline": float(values[9]),
        "error_x": float(values[10]),
        "slope": float(values[11]),
        "road_ratio_in_corridor": float(values[12]),
        "bridge_ratio_in_corridor": float(values[13]),
        "drivable_type": int(round(float(values[14]))),
        "side_view_likely": values[15] >= 0.5,
        "image_width": float(values[16]),
        "image_height": float(values[17]),
        "reason_code": float(values[18]),
    }

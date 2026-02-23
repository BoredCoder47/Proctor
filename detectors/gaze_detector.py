import cv2


class GazeDetector:
    def __init__(self, h_dev=0.15, v_down=0.17, v_up=0.25, debug=False):
        """
        h_dev:
            horizontal deviation threshold

        v_down:
            downward deviation (more sensitive)

        v_up:
            upward deviation (less sensitive)
        """
        self.h_dev = h_dev
        self.v_down = v_down
        self.v_up = v_up
        self.debug = debug

        self.center_x = None
        self.center_y = None

    def is_looking_away(
        self,
        left_center,
        right_center,
        frame_width,
        frame_height,
        frame=None
    ):
        if (
            left_center is None
            or right_center is None
            or frame_width <= 0
            or frame_height <= 0
        ):
            return False, {}

        # midpoint
        mid_x = (left_center[0] + right_center[0]) / 2.0
        mid_y = (left_center[1] + right_center[1]) / 2.0

        nx = mid_x / frame_width
        ny = mid_y / frame_height

        # calibration
        if self.center_x is None:
            self.center_x = nx
            self.center_y = ny

        dx = abs(nx - self.center_x)
        dy = ny - self.center_y

        horizontal_out = dx > self.h_dev
        down_out = dy > self.v_down
        up_out = dy < -self.v_up

        looking_away = horizontal_out or down_out or up_out

        if self.debug and frame is not None:
            cv2.circle(frame, (int(mid_x), int(mid_y)), 4, (255, 0, 0), -1)
            text = f"dx={dx:.2f} dy={dy:.2f}"
            cv2.putText(frame, text, (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        debug = {
            "dx": round(dx, 3),
            "dy": round(dy, 3),
            "h_out": horizontal_out,
            "down_out": down_out,
            "up_out": up_out
        }

        return looking_away, debug

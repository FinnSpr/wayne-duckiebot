#!/usr/bin/env python3
"""
DebuggingNode: ROS node for live debugging of the self-driving pipeline.

Subscribes to:
  - /{vehicle}/camera_node/image/compressed       (CompressedImage)
  - /{vehicle}/left_wheel_encoder_driver_node/tick   (WheelEncoderStamped)
  - /{vehicle}/right_wheel_encoder_driver_node/tick  (WheelEncoderStamped)

Runs a Flask web server (default :5000) showing:
  - Live visualisations (camera, unwarped, lane overlay, masks, heatmap)
  - State override dropdown
  - Current velocities (v, ω)
  - Average process time of the last 5 calls (excluding visualisation)
"""

import base64
import os
import time
import traceback
from collections import deque
from threading import Lock, Thread
from typing import Optional

import config
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelEncoderStamped
from flask import Flask, Response, request
from sensor_msgs.msg import CompressedImage

from image_utils import BEVConfig, load_calibrations
from planning import State
from process import SelfDrivingPipeline

# ---------------------------------------------------------------------------
#  HTML template (single-page app served by Flask)
# ---------------------------------------------------------------------------
_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duckiebot Debug</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #1a1a2e; color: #e0e0e0; font-family: 'Segoe UI', system-ui, sans-serif; padding: 12px; }
  h2 { font-size: 15px; font-weight: 600; margin-bottom: 8px; }
  .bar { display: flex; flex-wrap: wrap; align-items: center; gap: 16px; margin-bottom: 10px; }
  .bar label { font-size: 13px; }
  select, button { padding: 4px 8px; border-radius: 4px; border: 1px solid #555; background: #2a2a4a; color: #ddd; font-size: 13px; }
  .stat { font-size: 13px; background: #16213e; padding: 4px 10px; border-radius: 4px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 10px; }
  .card { background: #16213e; border-radius: 6px; overflow: hidden; text-align: center; }
  .card img { width: 100%; display: block; }
  .card .lbl { padding: 4px 0; font-size: 12px; color: #aaa; }
  .wait { grid-column: 1 / -1; text-align: center; padding: 60px; color: #555; font-size: 16px; }
</style>
</head>
<body>
<div class="bar">
  <label>State override:
    <select id="state-sel" onchange="setState(this.value)">
      <option value="">None</option>
      {% for s in states %}<option value="{{ s }}" {% if s == override %}selected{% endif %}>{{ s }}</option>{% endfor %}
    </select>
  </label>
  <button onclick="toggleProcess()" id="proc-btn">{{ 'Pause' if proc else 'Resume' }}</button>
  <span class="stat" id="vel">v: {{ vel }}  ω: {{ omega }}</span>
  <span class="stat" id="time">Avg (last {{ n }}): {{ avg_ms }} ms</span>
</div>
<div class="grid" id="grid">
  {% if panels %}
  {% for p in panels %}
  <div class="card">
    <img src="/vis/{{ p }}" id="img-{{ p }}">
    <div class="lbl">{{ p }}</div>
  </div>
  {% endfor %}
  {% else %}
  <div class="wait">Waiting for frames&hellip;</div>
  {% endif %}
</div>
<script>
  // refresh images every 500 ms
  setInterval(function() {
    var imgs = document.querySelectorAll('.card img');
    for (var i = 0; i < imgs.length; i++) {
      imgs[i].src = imgs[i].src.replace(/\?.*/, '') + '?t=' + Date.now();
    }
  }, 500);

  // refresh stats every second
  setInterval(function() {
    fetch('/stats')
      .then(r => r.json())
      .then(d => {
        document.getElementById('vel').textContent = 'v: ' + d.v + '  ω: ' + d.omega;
        document.getElementById('time').textContent = 'Avg (last ' + d.n + '): ' + d.avg_ms + ' ms';
      });
  }, 1000);

  function setState(val) {
    fetch('/state', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({state: val}) });
  }

  function toggleProcess() {
    fetch('/process/toggle', { method: 'POST' })
      .then(r => r.json())
      .then(d => {
        document.getElementById('proc-btn').textContent = d.processing ? 'Pause' : 'Resume';
      });
  }
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
#  Debugging node
# ---------------------------------------------------------------------------


class DebuggingNode(DTROS):
    """ROS node that wraps the SelfDrivingPipeline and exposes a Flask debug UI."""

    def __init__(self, node_name: str = "debugging_node", port: int = 5000):
        super().__init__(node_name=node_name, node_type=NodeType.GENERIC)

        # Force pipeline to build visualisations
        config.PUBLISH_VISUALIZATIONS = True

        self._port = port
        self._vehicle_name = os.environ["VEHICLE_NAME"]
        self._bridge = CvBridge()

        # ------------------------------------------------------------------
        # Calibration & pipeline
        # ------------------------------------------------------------------
        self._K, self._D, self._P, self._H = load_calibrations(
            config.INTRINSIC_CALIBRATION_FILE, config.EXTRINSIC_CALIBRATION_FILE
        )
        self._bev_cfg = BEVConfig(
            bev_size=config.BEV_SIZE, bev_resolution=config.BEV_RESOLUTION
        )

        self._pipeline = SelfDrivingPipeline(
            self._K, self._D, self._P, self._H, self._bev_cfg
        )
        self._pipeline.planner.set_intersection_decisions(config.INTERSECTION_DECISIONS)

        self._real_update_state = self._pipeline.planner.update_state
        self._real_get_visualizations = self._pipeline.get_visualizations

        # ------------------------------------------------------------------
        # Shared state
        # ------------------------------------------------------------------
        self._image: Optional[np.ndarray] = None
        self._left_encoder: int = 0
        self._right_encoder: int = 0
        self._data_lock = Lock()

        self._process_times: deque = deque(maxlen=5)

        self._latest_color_vis: dict = {}
        self._latest_bw_vis: dict = {}
        self._latest_velocities = (0.0, 0.0)
        self._latest_state: State = State.DRIVE
        self._state_override: Optional[State] = None
        self._results_lock = Lock()
        self._process_lock = Lock()

        self._last_process_time = rospy.Time(0)
        self._process_interval = rospy.Duration(1.0 / config.HZ)

        self._process_enabled = True

        # ------------------------------------------------------------------
        # ROS subscribers
        # ------------------------------------------------------------------
        rospy.Subscriber(
            f"/{self._vehicle_name}/camera_node/image/compressed",
            CompressedImage,
            self._cb_camera,
        )
        rospy.Subscriber(
            f"/{self._vehicle_name}/left_wheel_encoder_driver_node/tick",
            WheelEncoderStamped,
            self._cb_left_encoder,
        )
        rospy.Subscriber(
            f"/{self._vehicle_name}/right_wheel_encoder_driver_node/tick",
            WheelEncoderStamped,
            self._cb_right_encoder,
        )

        rospy.loginfo(
            f"DebuggingNode listening on "
            f"/{self._vehicle_name}/camera_node/image/compressed "
            f"at {config.HZ} Hz"
        )

        # ------------------------------------------------------------------
        # Flask
        # ------------------------------------------------------------------
        self._app = Flask(__name__)
        self._setup_routes()

    # ======================================================================
    #  ROS callbacks
    # ======================================================================

    def _cb_camera(self, msg: CompressedImage) -> None:
        with self._data_lock:
            self._image = self._bridge.compressed_imgmsg_to_cv2(msg)

        if not self._process_enabled:
            return

        now = rospy.Time.now()
        if (now - self._last_process_time) < self._process_interval:
            return
        self._last_process_time = now

        self._process_latest_frame()

    def _cb_left_encoder(self, msg: WheelEncoderStamped) -> None:
        with self._data_lock:
            self._left_encoder = msg.data

    def _cb_right_encoder(self, msg: WheelEncoderStamped) -> None:
        with self._data_lock:
            self._right_encoder = msg.data

    # ======================================================================
    #  Frame processing
    # ======================================================================

    def _process_latest_frame(self) -> None:
        with self._data_lock:
            image = self._image
            left_enc = self._left_encoder
            right_enc = self._right_encoder

        if image is None:
            return

        if not self._process_lock.acquire(blocking=False):
            return

        try:
            self._do_process(image, left_enc, right_enc)
        except Exception:
            rospy.logerr(f"Pipeline processing failed:\n{traceback.format_exc()}")
        finally:
            self._process_lock.release()

    def _do_process(self, image: np.ndarray, left_enc: int, right_enc: int) -> None:
        # --- state override ---
        if self._state_override is not None:
            self._pipeline.planner.update_state = lambda: None
            self._pipeline.planner.state = self._state_override
        else:
            self._pipeline.planner.update_state = self._real_update_state

        # --- monkey-patch get_visualizations for timing ---
        timing = {"t1": None}

        def patched_get_vis(img, waypoints):
            timing["t1"] = time.time()
            return self._real_get_visualizations(img, waypoints)

        self._pipeline.get_visualizations = patched_get_vis

        try:
            t0 = time.time()
            vel_left, vel_right, color_vis, bw_vis = self._pipeline.process(
                image, left_enc, right_enc
            )

            if timing["t1"] is not None:
                self._process_times.append(timing["t1"] - t0)

            with self._results_lock:
                self._latest_velocities = (vel_left, vel_right)
                self._latest_color_vis = color_vis
                self._latest_bw_vis = bw_vis
                self._latest_state = self._pipeline.planner.state

        finally:
            self._pipeline.get_visualizations = self._real_get_visualizations

    # ======================================================================
    #  Flask routes
    # ======================================================================

    def _setup_routes(self) -> None:
        app = self._app

        @app.route("/")
        def index():
            with self._results_lock:
                v, omega = self._latest_velocities
                state = self._latest_state
                color_vis = dict(self._latest_color_vis)
                bw_vis = dict(self._latest_bw_vis)
                times = list(self._process_times)
                override = self._state_override

            # Build ordered panel list
            panels = []
            for key in ("image", "unwarped_image", "visualization"):
                if key in color_vis and color_vis[key] is not None:
                    panels.append(key)
            if "heatmap" in color_vis and color_vis["heatmap"] is not None:
                panels.append("heatmap")
            for key in ("edge_mask", "white_lane_mask", "yellow_lane_mask"):
                if key in bw_vis and bw_vis[key] is not None:
                    panels.append(key)

            avg_ms = f"{np.mean(times) * 1000:.1f}" if times else "--"
            n = len(times)

            from flask import render_template_string

            return render_template_string(
                _PAGE,
                states=[s.name for s in State],
                override=override.name if override else "",
                vel=f"{v:.3f}",
                omega=f"{omega:.3f}",
                avg_ms=avg_ms,
                n=n,
                panels=panels,
                proc=self._process_enabled,
            )

        @app.route("/vis/<name>")
        def vis(name: str):
            with self._results_lock:
                color_vis = dict(self._latest_color_vis)
                bw_vis = dict(self._latest_bw_vis)

            img = color_vis.get(name)
            if img is None:
                img = bw_vis.get(name)
            if img is None:
                return ("", 404)

            # Convert to JPEG
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            _, jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return Response(
                jpeg.tobytes(), mimetype="image/jpeg"
            )

        @app.route("/stats")
        def stats():
            with self._results_lock:
                v, omega = self._latest_velocities
                times = list(self._process_times)

            avg_ms = f"{np.mean(times) * 1000:.1f}" if times else "--"

            from flask import jsonify

            return jsonify(
                v=f"{v:.3f}",
                omega=f"{omega:.3f}",
                avg_ms=avg_ms,
                n=len(times),
            )

        @app.route("/state", methods=["POST"])
        def set_state():
            data = request.get_json(silent=True) or {}
            name = data.get("state", "")
            if not name:
                self._state_override = None
            else:
                try:
                    self._state_override = State[name]
                except KeyError:
                    pass
            rospy.loginfo(f"State override: {self._state_override}")
            return ("", 204)

        @app.route("/process/toggle", methods=["POST"])
        def toggle_process():
            self._process_enabled = not self._process_enabled
            from flask import jsonify

            return jsonify(processing=self._process_enabled)

    # ======================================================================
    #  Lifecycle
    # ======================================================================

    def run(self) -> None:
        """Start Flask in a daemon thread, ROS spin in main (handles Ctrl+C)."""
        def _run_flask():
            self._app.run(
                host="0.0.0.0", port=self._port, debug=False, threaded=True
            )

        flask_thread = Thread(target=_run_flask, daemon=True)
        flask_thread.start()

        rospy.loginfo(
            f"Flask debug UI starting on http://0.0.0.0:{self._port}"
        )
        rospy.spin()


if __name__ == "__main__":
    node = DebuggingNode(node_name="debugging_node")
    node.run()

#!/usr/bin/env python3
"""
rgano_navigator.py — Rgano-powered autonomous navigation for Freenove Tank Bot

THE SAME FORMULA EVERYWHERE
============================
In memory space:  edge_score = 1 - consensus(memory, neighbors)
In image space:   edge_score = 1 - consensus(pixel, neighbors)
In physical space: edge_score = 1 - consensus(sensor_reading, expected)

The robot navigates by finding where visual consensus breaks down.
Edges in the camera frame = boundaries = obstacles or paths.

NAVIGATION REGIMES (QX cascade, same pattern as revision detection):
  1. PathRegime    — find open corridors (low-edge regions)
  2. ObstacleRegime — detect barriers (high-edge vertical bands)
  3. DropoffRegime  — detect floor edges (horizontal edge lines = cliffs)
  4. TargetRegime   — find objects of interest (edge-bounded regions)
  5. ConsensusRegime — fuse all signals for final steering decision

For CMU Swartz Center + NFL Draft Week Pitch Competition
April 2026, Pittsburgh

"Rgano detects edges — whether those edges are between draft prospects,
file revisions, or walls the robot needs to avoid."
"""

from __future__ import annotations

import numpy as np
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum


# ════════════════════════════════════════════════════════════════════
# CORE: Step-to-consensus edge detection (from rgano/core/edges.py)
# ════════════════════════════════════════════════════════════════════

def neighbor_mean_field(image: np.ndarray, lam: float = 0.35) -> np.ndarray:
    """Compute mean of 4-connected neighbors (relaxation step)."""
    padded = np.pad(image, ((1, 1), (1, 1), (0, 0)), mode='edge')
    neighbors = (
        padded[:-2, 1:-1] +  # up
        padded[2:, 1:-1] +   # down
        padded[1:-1, :-2] +  # left
        padded[1:-1, 2:]     # right
    ) / 4.0
    return (1 - lam) * image + lam * neighbors


def step_to_consensus(
    image: np.ndarray,
    max_steps: int = 32,
    eps: float = 0.03,
    lam: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process-derived edge detection via neighbor consensus relaxation.

    Returns:
        t_hit: (H, W) int — step when each pixel reached consensus
        edges: (H, W) float — normalized edge strength [0, 1]
    """
    if image.ndim == 2:
        image = image[..., np.newaxis]

    h, w = image.shape[:2]
    S = image.astype(np.float32) / 255.0 if image.max() > 1.0 else image.astype(np.float32)
    t_hit = np.full((h, w), max_steps, dtype=np.int32)
    converged = np.zeros((h, w), dtype=bool)

    for t in range(1, max_steps + 1):
        S_new = neighbor_mean_field(S, lam)
        diff = np.abs(S_new - S).max(axis=-1)

        just_converged = (~converged) & (diff < eps)
        t_hit[just_converged] = t
        converged |= just_converged

        S = S_new

        if converged.all():
            break

    # Normalize: late convergence = edge, early = interior
    edges = t_hit.astype(np.float32) / max_steps
    return t_hit, edges


# ════════════════════════════════════════════════════════════════════
# NAVIGATION REGIMES
# ════════════════════════════════════════════════════════════════════

class SteerCommand(Enum):
    FORWARD = "forward"
    LEFT = "left"
    RIGHT = "right"
    REVERSE = "reverse"
    STOP = "stop"
    SEEK = "seek"  # slow rotation looking for path


@dataclass
class NavSignal:
    """Output of a single navigation regime."""
    name: str
    command: SteerCommand
    confidence: float  # 0-1
    speed: float       # 0-1 suggested speed fraction
    details: Dict[str, float] = field(default_factory=dict)


def path_regime(edge_map: np.ndarray) -> NavSignal:
    """
    Regime 1: Find open corridors.

    Low-edge regions in the center = clear path ahead.
    Analogous to StrongEdgeRegime: if path is clear, don't need further analysis.
    """
    h, w = edge_map.shape
    # Look at the center-bottom of the frame (where the robot is heading)
    roi = edge_map[h // 2:, w // 4: 3 * w // 4]
    avg_edge = roi.mean()

    # Low edge density = clear path
    clearness = 1.0 - avg_edge

    if clearness > 0.7:
        return NavSignal("path", SteerCommand.FORWARD, clearness, speed=0.8)
    elif clearness > 0.5:
        return NavSignal("path", SteerCommand.FORWARD, clearness, speed=0.4)
    else:
        return NavSignal("path", SteerCommand.STOP, clearness, speed=0.0,
                         details={"avg_edge": avg_edge})


def obstacle_regime(edge_map: np.ndarray) -> NavSignal:
    """
    Regime 2: Detect obstacles via vertical edge bands.

    High-edge vertical bands = walls/objects.
    Steer away from the side with more edges.
    Analogous to SemanticEdgeRegime: different kind of boundary.
    """
    h, w = edge_map.shape
    center_strip = edge_map[h // 3: 2 * h // 3, :]

    left_density = center_strip[:, :w // 2].mean()
    right_density = center_strip[:, w // 2:].mean()

    # If both sides dense, reverse
    if left_density > 0.5 and right_density > 0.5:
        return NavSignal("obstacle", SteerCommand.REVERSE, 0.8, speed=0.5,
                         details={"left": left_density, "right": right_density})

    # Steer away from the denser side
    if left_density > right_density + 0.1:
        return NavSignal("obstacle", SteerCommand.RIGHT,
                         left_density, speed=0.5,
                         details={"left": left_density, "right": right_density})
    elif right_density > left_density + 0.1:
        return NavSignal("obstacle", SteerCommand.LEFT,
                         right_density, speed=0.5,
                         details={"left": left_density, "right": right_density})

    return NavSignal("obstacle", SteerCommand.FORWARD, 0.3, speed=0.4)


def dropoff_regime(edge_map: np.ndarray) -> NavSignal:
    """
    Regime 3: Detect floor edges (table edges, stairs, cliffs).

    Strong horizontal edge line in the bottom third = dropoff.
    Analogous to CorrelationEdgeRegime: subtle but critical signal.
    """
    h, w = edge_map.shape
    floor_strip = edge_map[3 * h // 4:, :]

    # Horizontal edge: look for a consistent band across the width
    row_means = floor_strip.mean(axis=1)
    max_row_edge = row_means.max()

    if max_row_edge > 0.6:
        return NavSignal("dropoff", SteerCommand.REVERSE, max_row_edge, speed=0.6,
                         details={"max_horizontal_edge": max_row_edge})

    return NavSignal("dropoff", SteerCommand.FORWARD, 0.1, speed=0.0)


def target_regime(edge_map: np.ndarray) -> NavSignal:
    """
    Regime 4: Find objects of interest (edge-bounded regions).

    A cluster of edges forming a bounded shape = something worth investigating.
    The robot steers toward edge-bounded regions in the center.
    """
    h, w = edge_map.shape

    # Find the column with the highest edge density (potential object)
    col_density = edge_map[h // 4: 3 * h // 4, :].mean(axis=0)
    peak_col = int(np.argmax(col_density))
    peak_val = col_density[peak_col]

    if peak_val < 0.4:
        return NavSignal("target", SteerCommand.FORWARD, 0.1, speed=0.0)

    # Steer toward the peak
    center = w // 2
    offset = (peak_col - center) / center  # -1 to 1

    if offset < -0.2:
        cmd = SteerCommand.LEFT
    elif offset > 0.2:
        cmd = SteerCommand.RIGHT
    else:
        cmd = SteerCommand.FORWARD

    return NavSignal("target", cmd, peak_val, speed=0.3,
                     details={"peak_col": peak_col, "offset": offset, "peak_val": peak_val})


def consensus_regime(signals: List[NavSignal], ultrasonic_cm: float) -> NavSignal:
    """
    Regime 5 (meta): Fuse all signals for final decision.

    Same pattern as ConsensusRegime in revision detection:
    count how many signals agree, weight by confidence.

    Also fuses ultrasonic distance as a hard override.
    """
    # Hard override: ultrasonic too close = emergency stop/reverse
    if ultrasonic_cm < 15:
        return NavSignal("consensus", SteerCommand.REVERSE, 1.0, speed=0.7,
                         details={"ultrasonic_cm": ultrasonic_cm, "reason": "too_close"})
    if ultrasonic_cm < 30:
        # Reduce speed but trust visual signals
        speed_cap = 0.4
    else:
        speed_cap = 1.0

    # Count votes by command
    votes: Dict[SteerCommand, float] = {}
    for sig in signals:
        if sig.confidence > 0.2:  # ignore low-confidence signals
            votes[sig.command] = votes.get(sig.command, 0.0) + sig.confidence

    if not votes:
        return NavSignal("consensus", SteerCommand.SEEK, 0.3, speed=0.2)

    # Winner takes all (weighted by confidence)
    best_cmd = max(votes, key=votes.get)
    best_confidence = votes[best_cmd] / sum(votes.values())

    # Compute speed from the strongest signal for this command
    best_speed = max(
        (s.speed for s in signals if s.command == best_cmd and s.confidence > 0.2),
        default=0.3,
    )

    return NavSignal(
        "consensus", best_cmd,
        best_confidence,
        speed=min(best_speed, speed_cap),
        details={"votes": {k.value: round(v, 2) for k, v in votes.items()},
                 "ultrasonic_cm": ultrasonic_cm},
    )


# ════════════════════════════════════════════════════════════════════
# NAVIGATOR: Ties it all together
# ════════════════════════════════════════════════════════════════════

class RganoNavigator:
    """
    Rgano-powered autonomous navigation for the Freenove Tank Bot.

    Usage:
        from car import Car
        from camera import Camera

        car = Car()
        cam = Camera()
        cam.start_stream()

        nav = RganoNavigator(car, cam)
        nav.start()   # begins autonomous navigation loop
        # ...
        nav.stop()
    """

    def __init__(self, car, cam, ultrasonic=None,
                 max_steps=24, eps=0.04, lam=0.35,
                 base_speed=2000, frame_interval=0.15):
        self.car = car
        self.cam = cam
        self.sonic = ultrasonic or (car.sonic if hasattr(car, 'sonic') else None)
        self.max_steps = max_steps
        self.eps = eps
        self.lam = lam
        self.base_speed = base_speed
        self.frame_interval = frame_interval

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_edge_map: Optional[np.ndarray] = None
        self._last_signals: List[NavSignal] = []
        self._last_decision: Optional[NavSignal] = None

    def start(self):
        """Start the autonomous navigation loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._nav_loop, daemon=True)
        self._thread.start()
        print("[RganoNav] started — edge_score = 1 - consensus(pixel, neighbors)")

    def stop(self):
        """Stop navigation and halt motors."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        self.car.motor.setMotorModel(0, 0)
        print("[RganoNav] stopped")

    @property
    def edge_map(self) -> Optional[np.ndarray]:
        return self._last_edge_map

    @property
    def last_decision(self) -> Optional[NavSignal]:
        return self._last_decision

    def _nav_loop(self):
        """Main navigation loop: grab frame → detect edges → run cascade → steer."""
        while self._running:
            try:
                # 1. Grab frame
                frame_bytes = self.cam.get_frame()
                if frame_bytes is None:
                    time.sleep(0.05)
                    continue

                # Decode JPEG to numpy
                frame = self._decode_frame(frame_bytes)
                if frame is None:
                    continue

                # 2. Downscale for speed (edge detection on 80x60 is fine for nav)
                small = self._downscale(frame, 80, 60)

                # 3. Step-to-consensus edge detection (THE FORMULA)
                _, edge_map = step_to_consensus(
                    small, self.max_steps, self.eps, self.lam
                )
                self._last_edge_map = edge_map

                # 4. Get ultrasonic distance
                dist = self.sonic.get_distance() if self.sonic else 999.0

                # 5. Run navigation regime cascade
                signals = [
                    path_regime(edge_map),
                    obstacle_regime(edge_map),
                    dropoff_regime(edge_map),
                    target_regime(edge_map),
                ]
                self._last_signals = signals

                # 6. Consensus fusion
                decision = consensus_regime(signals, dist)
                self._last_decision = decision

                # 7. Execute steering command
                self._execute(decision)

                time.sleep(self.frame_interval)

            except Exception as e:
                print(f"[RganoNav] error: {e}")
                self.car.motor.setMotorModel(0, 0)
                time.sleep(0.5)

    def _execute(self, decision: NavSignal):
        """Convert a NavSignal into motor commands."""
        speed = int(decision.speed * self.base_speed)

        if decision.command == SteerCommand.FORWARD:
            self.car.motor.setMotorModel(speed, speed)
        elif decision.command == SteerCommand.LEFT:
            self.car.motor.setMotorModel(-speed, speed)
        elif decision.command == SteerCommand.RIGHT:
            self.car.motor.setMotorModel(speed, -speed)
        elif decision.command == SteerCommand.REVERSE:
            self.car.motor.setMotorModel(-speed, -speed)
        elif decision.command == SteerCommand.SEEK:
            # Slow rotation to find a path
            self.car.motor.setMotorModel(-800, 800)
        else:
            self.car.motor.setMotorModel(0, 0)

    @staticmethod
    def _decode_frame(frame_bytes: bytes) -> Optional[np.ndarray]:
        """Decode JPEG bytes to numpy array."""
        try:
            import cv2
            arr = np.frombuffer(frame_bytes, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except ImportError:
            # Fallback: PIL
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(frame_bytes))
                return np.array(img)
            except Exception:
                return None

    @staticmethod
    def _downscale(frame: np.ndarray, w: int, h: int) -> np.ndarray:
        """Downscale a frame for fast edge detection."""
        try:
            import cv2
            return cv2.resize(frame, (w, h))
        except ImportError:
            from PIL import Image
            img = Image.fromarray(frame)
            return np.array(img.resize((w, h)))


# ════════════════════════════════════════════════════════════════════
# MEMORY INTEGRATION: Feed navigation data to always-on-memory
# ════════════════════════════════════════════════════════════════════

class NavigationMemory:
    """
    Periodically ingests navigation decisions into the always-on-memory
    system for consolidation. The robot builds a topological memory of
    its environment using the same regime cascade.

    The edge maps become memories. Consolidation finds patterns:
    "this corner always has high obstacle edges on the left."
    """

    def __init__(self, memory_url: str = "http://localhost:8888"):
        self.url = memory_url
        self.buffer: List[dict] = []

    def record(self, edge_map: np.ndarray, decision: NavSignal, signals: List[NavSignal]):
        """Buffer a navigation snapshot."""
        self.buffer.append({
            "edge_mean": float(edge_map.mean()),
            "edge_std": float(edge_map.std()),
            "decision": decision.command.value,
            "confidence": decision.confidence,
            "speed": decision.speed,
            "regime_scores": {s.name: s.confidence for s in signals},
            "timestamp": time.time(),
        })

        # Flush every 20 snapshots
        if len(self.buffer) >= 20:
            self.flush()

    def flush(self):
        """Send buffered navigation data to always-on-memory."""
        if not self.buffer:
            return

        import json
        summary = self._summarize(self.buffer)
        payload = {
            "text": json.dumps(summary),
            "source": "rgano_navigator",
            "collection": "navigation",
        }

        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.url}/ingest",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[NavMemory] flush failed: {e}")

        self.buffer.clear()

    @staticmethod
    def _summarize(buffer: List[dict]) -> dict:
        """Compress a batch of navigation snapshots."""
        commands = [b["decision"] for b in buffer]
        return {
            "type": "navigation_batch",
            "count": len(buffer),
            "dominant_command": max(set(commands), key=commands.count),
            "avg_edge_density": np.mean([b["edge_mean"] for b in buffer]),
            "avg_confidence": np.mean([b["confidence"] for b in buffer]),
            "command_distribution": {
                cmd: commands.count(cmd) / len(commands)
                for cmd in set(commands)
            },
            "time_start": buffer[0]["timestamp"],
            "time_end": buffer[-1]["timestamp"],
        }


# ════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════

def main():
    """Run the Rgano navigator on the Freenove Tank Bot."""
    from car import Car
    from camera import Camera

    print("=" * 60)
    print("  RGANO NAVIGATOR")
    print("  edge_score = 1 - consensus(pixel, neighbors)")
    print("  Same formula. Pixels or prospects.")
    print("=" * 60)

    car = Car()
    cam = Camera(stream_size=(320, 240))
    cam.start_stream()

    nav = RganoNavigator(car, cam)

    # Optional: connect to always-on-memory for spatial learning
    # nav_mem = NavigationMemory("http://localhost:8888")

    try:
        nav.start()
        print("Press Ctrl+C to stop")
        while True:
            time.sleep(1)
            d = nav.last_decision
            if d:
                print(f"  [{d.name}] {d.command.value} "
                      f"(confidence={d.confidence:.2f}, speed={d.speed:.2f}) "
                      f"{d.details}")
    except KeyboardInterrupt:
        nav.stop()
        cam.close()
        car.close()
        print("\nLanded.")


if __name__ == "__main__":
    main()

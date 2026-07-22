from __future__ import annotations

from dataclasses import dataclass
import math


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class IdealKinematicBase:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    z: float = 0.40

    def reset(self, pose: list[float] | tuple[float, ...] | None = None) -> None:
        values = list(pose or [0.0, 0.0, 0.0])
        if len(values) < 3:
            raise ValueError("ideal base reset pose requires x, y, yaw")
        self.x = float(values[0])
        self.y = float(values[1])
        self.yaw = wrap_to_pi(float(values[2]))

    def integrate(self, *, vx_body: float, vy_body: float, wz: float, dt: float) -> None:
        dt = max(0.0, float(dt))
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        self.x += (cos_yaw * float(vx_body) - sin_yaw * float(vy_body)) * dt
        self.y += (sin_yaw * float(vx_body) + cos_yaw * float(vy_body)) * dt
        self.yaw = wrap_to_pi(self.yaw + float(wz) * dt)

    def pose(self) -> list[float]:
        return [self.x, self.y, self.yaw]

    def world_velocity(self, *, vx_body: float, vy_body: float, wz: float) -> list[float]:
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        return [
            cos_yaw * float(vx_body) - sin_yaw * float(vy_body),
            sin_yaw * float(vx_body) + cos_yaw * float(vy_body),
            0.0,
            0.0,
            0.0,
            float(wz),
        ]

    def telemetry(self, *, vx_body: float, vy_body: float, wz: float) -> dict[str, list[float] | float]:
        velocity = self.world_velocity(vx_body=vx_body, vy_body=vy_body, wz=wz)
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "heading": self.yaw,
            "linear_velocity": velocity[:3],
            "angular_velocity": velocity[3:],
        }

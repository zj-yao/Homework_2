from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable


Point = tuple[float, float]


@dataclass(frozen=True)
class LineSegment:
    """A finite counting line in image coordinates."""

    start: Point
    end: Point

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError("Counting line start and end must be different points")


def _as_point(point: Point) -> Point:
    return float(point[0]), float(point[1])


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def signed_side(line: LineSegment, point: Point, eps: float = 1e-9) -> int:
    """Return -1/0/1 for which side of the directed line a point is on."""

    value = _cross(_as_point(line.start), _as_point(line.end), _as_point(point))
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def _on_segment(a: Point, b: Point, c: Point, eps: float = 1e-9) -> bool:
    return (
        min(a[0], b[0]) - eps <= c[0] <= max(a[0], b[0]) + eps
        and min(a[1], b[1]) - eps <= c[1] <= max(a[1], b[1]) + eps
        and abs(_cross(a, b, c)) <= eps
    )


def _orientation(a: Point, b: Point, c: Point, eps: float = 1e-9) -> int:
    value = _cross(a, b, c)
    if value > eps:
        return 1
    if value < -eps:
        return -1
    return 0


def segments_intersect(a: Point, b: Point, c: Point, d: Point, eps: float = 1e-9) -> bool:
    """Return True when two finite line segments intersect."""

    a = _as_point(a)
    b = _as_point(b)
    c = _as_point(c)
    d = _as_point(d)

    o1 = _orientation(a, b, c, eps)
    o2 = _orientation(a, b, d, eps)
    o3 = _orientation(c, d, a, eps)
    o4 = _orientation(c, d, b, eps)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a, b, c, eps):
        return True
    if o2 == 0 and _on_segment(a, b, d, eps):
        return True
    if o3 == 0 and _on_segment(c, d, a, eps):
        return True
    if o4 == 0 and _on_segment(c, d, b, eps):
        return True
    return False


def segment_crosses_line(line: LineSegment, previous: Point, current: Point) -> bool:
    """Return True when a center trajectory crosses the finite counting line."""

    previous_side = signed_side(line, previous)
    current_side = signed_side(line, current)
    if previous_side == 0 or current_side == 0 or previous_side == current_side:
        return False
    return segments_intersect(previous, current, line.start, line.end)


@dataclass
class LineCrossingCounter:
    """Count each stable tracking ID at most once when it crosses a line."""

    line: LineSegment
    last_centers: dict[Hashable, Point] = field(default_factory=dict)
    last_sides: dict[Hashable, int] = field(default_factory=dict)
    crossed_ids: set[Hashable] = field(default_factory=set)

    @property
    def total(self) -> int:
        return len(self.crossed_ids)

    def update(self, track_id: Hashable | None, center: Point) -> bool:
        """Update one track center and return True only for a new crossing."""

        if track_id is None:
            return False

        center = _as_point(center)
        current_side = signed_side(self.line, center)
        previous_center = self.last_centers.get(track_id)
        previous_side = self.last_sides.get(track_id)
        crossed_now = False

        if (
            previous_center is not None
            and previous_side is not None
            and current_side != 0
            and current_side != previous_side
            and track_id not in self.crossed_ids
            and segments_intersect(previous_center, center, self.line.start, self.line.end)
        ):
            self.crossed_ids.add(track_id)
            crossed_now = True

        self.last_centers[track_id] = center
        if current_side != 0:
            self.last_sides[track_id] = current_side
        return crossed_now

    def reset(self) -> None:
        self.last_centers.clear()
        self.last_sides.clear()
        self.crossed_ids.clear()

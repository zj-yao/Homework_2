from src.detection_tracking.line_counter import (
    LineCrossingCounter,
    LineSegment,
    segment_crosses_line,
    signed_side,
)


def test_line_counter_counts_track_once_when_center_crosses_line():
    counter = LineCrossingCounter(LineSegment((5.0, 0.0), (5.0, 10.0)))

    assert counter.update(track_id=17, center=(2.0, 5.0)) is False
    assert counter.update(track_id=17, center=(8.0, 5.0)) is True

    assert counter.total == 1
    assert counter.crossed_ids == {17}


def test_line_counter_does_not_double_count_same_tracking_id():
    counter = LineCrossingCounter(LineSegment((5.0, 0.0), (5.0, 10.0)))

    counter.update(track_id=3, center=(2.0, 4.0))
    assert counter.update(track_id=3, center=(8.0, 4.0)) is True
    assert counter.update(track_id=3, center=(2.0, 4.0)) is False

    assert counter.total == 1


def test_signed_side_changes_across_line():
    line = LineSegment((0.0, 0.0), (0.0, 10.0))

    left_side = signed_side(line, (-1.0, 5.0))
    right_side = signed_side(line, (1.0, 5.0))

    assert left_side != 0
    assert right_side != 0
    assert left_side == -right_side


def test_segment_crossing_requires_intersection_with_finite_counting_line():
    line = LineSegment((5.0, 0.0), (5.0, 10.0))

    assert segment_crosses_line(line, previous=(2.0, 5.0), current=(8.0, 5.0))
    assert not segment_crosses_line(line, previous=(2.0, 15.0), current=(8.0, 15.0))

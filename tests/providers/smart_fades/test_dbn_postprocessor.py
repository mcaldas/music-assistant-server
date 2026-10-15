"""Tests for the pure numpy DBN postprocessor."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from music_assistant.providers.smart_fades.dbn_postprocessor import (
    DBNDownBeatTracker,
    decode_one_level,
    one_level_tempo_range,
)

FPS = 50


def test_state_space_sizes() -> None:
    """Test that the bar state space has correct dimensions."""
    positions, intervals = DBNDownBeatTracker._build_bar_state_space(
        num_beats=4, min_interval=14, max_interval=55
    )
    # Each interval i contributes i states, repeated for each beat
    expected_states = sum(range(14, 56)) * 4
    assert len(positions) == expected_states
    assert len(intervals) == expected_states
    # Positions for a 4-beat bar range from [0, 4)
    assert positions.min() >= 0.0
    assert positions.max() < 4.0


def test_state_space_small() -> None:
    """Test state space with small interval range for easy verification."""
    positions, _intervals = DBNDownBeatTracker._build_bar_state_space(
        num_beats=2, min_interval=3, max_interval=4
    )
    # interval=3: 3 states, interval=4: 4 states, x 2 beats = 14 states
    assert len(positions) == 14
    # Beat 0 positions: [0/3, 1/3, 2/3, 0/4, 1/4, 2/4, 3/4]
    # Beat 1 positions: [1+0/3, 1+1/3, 1+2/3, 1+0/4, 1+1/4, 1+2/4, 1+3/4]
    assert positions[0] == pytest.approx(0.0)
    assert positions[7] == pytest.approx(1.0)


def test_transition_model_structure() -> None:
    """Test that the transition model is a valid sparse matrix."""
    positions, intervals = DBNDownBeatTracker._build_bar_state_space(
        num_beats=4, min_interval=14, max_interval=55
    )
    tm_states, tm_pointers, _tm_log_probs = DBNDownBeatTracker._build_transition_model(
        positions, intervals, num_beats=4, transition_lambda=100
    )
    num_states = len(positions)
    # pointers has num_states + 1 entries (CSR format)
    assert len(tm_pointers) == num_states + 1
    # Every state must have at least one predecessor
    for s in range(num_states):
        assert tm_pointers[s + 1] > tm_pointers[s]
    # All source state indices must be valid
    assert tm_states.max() < num_states


def test_dbn_tracker_constant_tempo() -> None:
    """Test that the DBN produces regular beats for a constant-tempo signal."""
    fps = 50
    bpm = 120.0
    duration = 10.0  # seconds
    num_frames = int(duration * fps)

    # Synthesize activations: strong peaks at expected beat positions
    interval = 60.0 / bpm * fps  # frames per beat
    beat_act = np.full(num_frames, 0.05)
    downbeat_act = np.full(num_frames, 0.02)
    for i in range(int(duration * bpm / 60)):
        frame = int(i * interval)
        if frame < num_frames:
            beat_act[frame] = 0.95
            if i % 4 == 0:
                downbeat_act[frame] = 0.95

    combined = np.column_stack(
        [
            np.maximum(beat_act - downbeat_act, 1e-5),
            downbeat_act,
        ]
    )

    tracker = DBNDownBeatTracker(beats_per_bar=[4], min_bpm=55, max_bpm=215, fps=fps)
    result, _num_beats = tracker(combined)

    # Should detect ~20 beats in 10s at 120 BPM
    beat_times = result[:, 0]
    assert 18 <= len(beat_times) <= 22

    # Inter-beat intervals should be close to 0.5s
    ibis = np.diff(beat_times)
    assert np.all(np.abs(ibis - 0.5) < 0.06)

    # Should have downbeats (beat_position == 1)
    downbeat_mask = result[:, 1] == 1
    assert downbeat_mask.sum() >= 2


def test_dbn_tracker_fills_intro_gaps() -> None:
    """Test that the DBN fills in regular beats even when the intro has no peaks."""
    fps = 50
    bpm = 120.0
    duration = 10.0
    num_frames = int(duration * fps)

    interval = 60.0 / bpm * fps
    beat_act = np.full(num_frames, 0.05)
    downbeat_act = np.full(num_frames, 0.02)

    # Only place peaks after 5s (second half)
    for i in range(int(5.0 * bpm / 60), int(duration * bpm / 60)):
        frame = int(i * interval)
        if frame < num_frames:
            beat_act[frame] = 0.95
            if i % 4 == 0:
                downbeat_act[frame] = 0.95

    combined = np.column_stack(
        [
            np.maximum(beat_act - downbeat_act, 1e-5),
            downbeat_act,
        ]
    )

    tracker = DBNDownBeatTracker(beats_per_bar=[4], min_bpm=55, max_bpm=215, fps=fps, threshold=0.0)
    result, _num_beats = tracker(combined)

    beat_times = result[:, 0]
    # DBN should still find beats in the first 5s via tempo continuity
    early_beats = beat_times[beat_times < 5.0]
    assert len(early_beats) >= 5, f"Expected beats in intro region, got {len(early_beats)}"


def test_winning_meter_is_returned() -> None:
    """A 4/4 activation pattern reports beats_per_bar=4 alongside the beats."""
    fps = 50
    bpm = 120.0
    duration = 20.0
    num_frames = int(duration * fps)

    interval = 60.0 / bpm * fps
    beat_act = np.full(num_frames, 0.05)
    downbeat_act = np.full(num_frames, 0.02)
    for i in range(int(duration * bpm / 60)):
        frame = int(i * interval)
        if frame < num_frames:
            beat_act[frame] = 0.95
            if i % 4 == 0:
                downbeat_act[frame] = 0.95

    combined = np.column_stack(
        [
            np.maximum(beat_act - downbeat_act, 1e-5),
            downbeat_act,
        ]
    )

    tracker = DBNDownBeatTracker(beats_per_bar=[3, 4], min_bpm=55, max_bpm=215, fps=fps)
    result, num_beats = tracker(combined)

    assert num_beats == 4
    assert len(result) > 0


def test_dbn_tracker_output_format() -> None:
    """Test that output matches the madmom interface: (M, 2) with [time, beat_pos]."""
    fps = 50
    num_frames = 500  # 10s
    rng = np.random.default_rng(42)
    combined = rng.uniform(0.01, 0.1, (num_frames, 2)).astype(np.float64)
    # Place a few clear peaks
    for i in range(0, num_frames, 25):  # 120 BPM
        combined[i, 0] = 0.9
        if (i // 25) % 4 == 0:
            combined[i, 1] = 0.9

    tracker = DBNDownBeatTracker(beats_per_bar=[4], min_bpm=55, max_bpm=215, fps=fps)
    result, _num_beats = tracker(combined)

    assert result.ndim == 2
    assert result.shape[1] == 2
    # Column 0: times in seconds, should be positive and sorted
    assert np.all(result[:, 0] >= 0)
    assert np.all(np.diff(result[:, 0]) > 0)
    # Column 1: beat positions, integers 1..num_beats
    assert np.all(result[:, 1] >= 1)
    assert np.all(result[:, 1] <= 4)


def _beat_times(*stretches: tuple[int, int], start: float = 0.5, fps: int = FPS) -> np.ndarray:
    """
    Return beat times in seconds for stretches of (interval in frames, number of intervals).

    :param stretches: The stretches, in order.
    :param start: Time of the first beat.
    :param fps: Frames per second the intervals are counted in.
    """
    frames = [0]
    for interval, count in stretches:
        last = frames[-1]
        frames.extend(last + interval * (i + 1) for i in range(count))
    return start + np.array(frames) / fps


# the second decode's range for a track at 100 BPM (30 frames a beat): 100 / 1.2 to 100 * 1.2
RANGE_AT_100_BPM = (83.3333, 120.0)


def test_a_steady_grid_in_the_band_is_kept() -> None:
    """A grid at one tempo inside the band needs no second decode."""
    assert one_level_tempo_range(_beat_times((30, 200)), FPS) is None


def test_a_grid_too_short_to_have_a_tempo_is_kept() -> None:
    """No beat or a single beat has no interval to judge."""
    assert one_level_tempo_range(np.array([]), FPS) is None
    assert one_level_tempo_range(np.array([1.0]), FPS) is None


def test_a_grid_whose_beats_are_not_in_order_is_kept() -> None:
    """Beats at one time or going backwards have no tempo to judge."""
    assert one_level_tempo_range(np.array([1.0, 1.0]), FPS) is None
    assert one_level_tempo_range(np.array([2.0, 1.4, 0.8]), FPS) is None


@pytest.mark.parametrize(
    ("interval", "folded_tempo"),
    [
        (20, 75.0),  # 150 BPM is counted at half
        (17, 3000 / 17 / 2),  # 176 BPM too
        (10, 75.0),  # 300 BPM at a quarter
        (50, 120.0),  # 60 BPM is counted at double
        (100, 120.0),  # 30 BPM at four times
    ],
)
def test_a_tempo_outside_the_band_is_folded_into_it(interval: int, folded_tempo: float) -> None:
    """A steady grid outside the band is decoded again around its tempo's octave in the band."""
    tempo_range = one_level_tempo_range(_beat_times((interval, 120)), FPS)

    assert tempo_range == pytest.approx((folded_tempo / 1.2, folded_tempo * 1.2))


@pytest.mark.parametrize(
    ("interval", "kept"),
    [
        (45, True),  # 66.7 BPM, the slowest interval in the band
        (46, False),  # 65.2 BPM, just under it
        (23, True),  # 130.4 BPM, the fastest interval in the band
        (22, False),  # 136.4 BPM: the band ends below twice its low edge
    ],
)
def test_the_band_holds_one_octave(interval: int, kept: bool) -> None:
    """At 50 frames a second the band holds beat intervals of 23 to 45 frames."""
    tempo_range = one_level_tempo_range(_beat_times((interval, 120)), FPS)

    assert (tempo_range is None) is kept
    if tempo_range is not None:
        # folded into 66 to 132 BPM: 65.2 comes back as 130.4, 136.4 as 68.2
        assert 66.0 <= tempo_range[0] * 1.2 < 132.0


def test_the_band_starts_at_its_low_edge_and_ends_before_twice_that() -> None:
    """66 BPM is in the band and 132 BPM is not."""
    # no whole-frame interval is 66 or 132 BPM at 50 frames a second; at 110 an interval of
    # 100 frames is 66 BPM and one of 50 frames is 132
    assert one_level_tempo_range(_beat_times((100, 120), fps=110), 110) is None
    assert one_level_tempo_range(_beat_times((50, 120), fps=110), 110) == pytest.approx(
        (66.0 / 1.2, 66.0 * 1.2)
    )


@pytest.mark.parametrize("off_interval", [15, 60])
def test_a_run_an_octave_off_asks_for_a_second_decode(off_interval: int) -> None:
    """A stretch at double or half the track's tempo is a run."""
    beats = _beat_times((30, 100), (off_interval, 8), (30, 100))

    assert one_level_tempo_range(beats, FPS) == pytest.approx(RANGE_AT_100_BPM, abs=1e-3)


@pytest.mark.parametrize("off_interval", [45, 20, 36, 25])
def test_a_stretch_at_another_ratio_is_left_alone(off_interval: int) -> None:
    """Two thirds, one and a half times or a fifth off the tempo may be the music's own."""
    beats = _beat_times((30, 100), (off_interval, 40), (30, 100))

    assert one_level_tempo_range(beats, FPS) is None


def test_a_run_shorter_than_eight_beats_is_left_alone() -> None:
    """A fill of a few beats is not a change of level."""
    beats = _beat_times((30, 100), (15, 7), (30, 100))

    assert one_level_tempo_range(beats, FPS) is None


def test_a_run_is_counted_in_beats_in_a_row() -> None:
    """Beats an octave off that are not next to each other do not add up to a run."""
    stretches = [(30, 20), (15, 4)] * 6
    beats = _beat_times(*stretches, (30, 20))

    assert one_level_tempo_range(beats, FPS) is None


@pytest.mark.parametrize(
    ("main", "off_interval", "fps", "kept"),
    [
        (82, 35, 100, False),  # 14.6 % under half the main interval: an octave off
        (64, 27, 100, True),  # 15.6 % under half: not
        (82, 47, 100, False),  # 14.6 % over half
        (64, 37, 100, True),  # 15.6 % over half
        (41, 70, 50, False),  # 14.6 % under double the main interval
        (32, 54, 50, True),  # 15.6 % under double
        (41, 94, 50, False),  # 14.6 % over double
        (32, 74, 50, True),  # 15.6 % over double
        # on the line itself, which counts: 0.15 * 20 and 0.15 * 60 are exactly 3.0 and 9.0
        (40, 17, 50, False),  # exactly 15 % under half
        (40, 23, 50, False),  # exactly 15 % over half
        (30, 51, 50, False),  # exactly 15 % under double
        (26, 15, 50, True),  # 15.4 % over half
        # half of an odd interval is not a whole frame: 13.5 for 27, 16.5 for 33
        (27, 15, 50, False),  # not rounded down to 13
        (27, 16, 50, True),  # nor up to 14
        (33, 14, 50, True),  # not rounded down to 16
    ],
)
def test_an_octave_run_is_within_the_deviation_of_half_or_double(
    main: int, off_interval: int, fps: int, kept: bool
) -> None:
    """Only intervals close to half or double the main one make a run."""
    beats = _beat_times((main, 100), (off_interval, 8), (main, 100), fps=fps)

    assert (one_level_tempo_range(beats, fps) is None) is kept


@pytest.mark.parametrize(
    ("stretches", "folded_tempo"),
    [
        # 50 beats at 100 BPM last 30 s and 60 beats at 150 BPM last 24 s: counted in
        # beats the track would be at 150 BPM, outside the band
        ([(30, 50), (20, 60)], None),
        # 20.4 s at 176 BPM and 20 s at 150 BPM: with the longer interval given more than
        # its time the track would be folded from 150 BPM, not from 176
        ([(17, 60), (20, 50)], 3000 / 17 / 2),
    ],
)
def test_the_main_interval_is_the_one_most_of_the_time_has(
    stretches: list[tuple[int, int]], folded_tempo: float | None
) -> None:
    """The main interval has the largest share of the track's time, not the most beats."""
    tempo_range = one_level_tempo_range(_beat_times(*stretches), FPS)

    if folded_tempo is None:
        assert tempo_range is None
    else:
        assert tempo_range == pytest.approx((folded_tempo / 1.2, folded_tempo * 1.2))


class _ScriptedTracker:
    """A tracker stand-in that answers with given grids and records what it was asked."""

    fps = FPS

    def __init__(
        self, first: np.ndarray, second: np.ndarray | None = None, num_beats: int = 4
    ) -> None:
        """
        Initialize the stand-in.

        :param first: The grid this stand-in returns, as (time, beat position) rows.
        :param second: The grid its narrowed stand-in returns.
        :param num_beats: The meter this stand-in returns.
        """
        self.first = first
        self.second = second
        self.num_beats = num_beats
        self.ranges: list[tuple[float, float]] = []
        self.seen: np.ndarray | None = None
        self.again: _ScriptedTracker | None = None

    def __call__(self, activations: np.ndarray) -> tuple[np.ndarray, int]:
        """Keep what was passed in and return this stand-in's grid and meter."""
        self.seen = activations
        return self.first, self.num_beats

    def narrowed(self, min_bpm: float, max_bpm: float) -> _ScriptedTracker:
        """Record the range and return a stand-in that answers with the second grid, in three."""
        self.ranges.append((min_bpm, max_bpm))
        assert self.second is not None
        self.again = _ScriptedTracker(self.second, num_beats=3)
        return self.again


def _grid(beat_times: np.ndarray) -> np.ndarray:
    """Return (time, beat position) rows for beat times, a bar start every fourth beat."""
    return np.column_stack([beat_times, np.arange(len(beat_times)) % 4 + 1])


def test_decode_one_level_keeps_a_grid_that_needs_nothing() -> None:
    """A grid with one tempo in the band is returned as decoded, with no second decode."""
    first = _grid(_beat_times((30, 200)))
    tracker = _ScriptedTracker(first)

    beats, num_beats = decode_one_level(cast("DBNDownBeatTracker", tracker), np.zeros((10, 2)))

    assert beats is first
    assert num_beats == 4
    assert tracker.ranges == []


def test_decode_one_level_takes_the_second_decode() -> None:
    """A grid with a run is decoded again on the folded range, and that grid and meter win."""
    first = _grid(_beat_times((30, 100), (15, 60), (30, 100)))
    second = _grid(_beat_times((30, 230)))
    tracker = _ScriptedTracker(first, second)
    activations = np.zeros((10, 2))

    beats, num_beats = decode_one_level(cast("DBNDownBeatTracker", tracker), activations)

    assert beats is second
    assert num_beats == 3
    assert tracker.ranges == [pytest.approx(RANGE_AT_100_BPM, abs=1e-3)]
    # both decodes read the very activations that were passed in
    assert tracker.seen is activations
    assert tracker.again is not None
    assert tracker.again.seen is activations


def test_decode_one_level_takes_a_second_grid_with_more_beats() -> None:
    """A track counted under the band comes out with twice the beats."""
    first = _grid(_beat_times((50, 100)))
    second = _grid(_beat_times((25, 200)))
    tracker = _ScriptedTracker(first, second)

    beats, _ = decode_one_level(cast("DBNDownBeatTracker", tracker), np.zeros((10, 2)))

    assert beats is second
    assert tracker.ranges == [pytest.approx((100.0, 144.0))]


@pytest.mark.parametrize("second_beats", [0, 1])
def test_decode_one_level_keeps_the_first_grid_when_the_second_finds_none(
    second_beats: int,
) -> None:
    """A second decode without a grid does not cost the track the one it has."""
    first = _grid(_beat_times((20, 200)))
    second = _grid(_beat_times((40, 100)))[:second_beats]
    tracker = _ScriptedTracker(first, second)

    beats, num_beats = decode_one_level(cast("DBNDownBeatTracker", tracker), np.zeros((10, 2)))

    assert beats is first
    assert num_beats == 4
    assert len(tracker.ranges) == 1


def test_decode_one_level_judges_the_grid_at_the_trackers_frame_rate() -> None:
    """The first grid's tempo is read at the tracker's own frames per second."""
    # beats 0.35 s apart are 171.4 BPM at any frame rate; the spacing is chosen so that
    # read at 50 frames a second it is 17.5 frames, which rounds to another tempo
    first = _grid(np.arange(100) * 0.35)
    tracker = _ScriptedTracker(first, first)
    tracker.fps = 100

    decode_one_level(cast("DBNDownBeatTracker", tracker), np.zeros((10, 2)))

    assert tracker.ranges == [pytest.approx((6000 / 35 / 2 / 1.2, 6000 / 35 / 2 * 1.2))]


def test_narrowed_keeps_the_settings_and_takes_the_range() -> None:
    """A narrowed tracker differs from its origin in the tempo range only."""
    tracker = DBNDownBeatTracker(
        # two meters in an order that is not the default: a narrowed tracker that kept only
        # one of them, or fell back to the default, would show
        beats_per_bar=[4, 3],
        min_bpm=55,
        max_bpm=215,
        fps=100,
        transition_lambda=80.0,
        observation_lambda=12,
        threshold=0.1,
        correct=False,
    )

    narrowed = tracker.narrowed(62.5, 90.0)

    assert narrowed.beats_per_bar == [4, 3]
    assert narrowed.fps == 100
    assert narrowed.transition_lambda == 80.0
    assert narrowed.observation_lambda == 12
    assert narrowed.threshold == 0.1
    assert narrowed.correct is False
    assert [hmm["num_beats"] for hmm in narrowed._hmms] == [4, 3]
    # beat lengths of 67 to 96 frames: 90 down to 62.5 BPM at 100 frames a second
    assert len(narrowed._hmms[0]["positions"]) == sum(range(67, 97)) * 4


def _activations(
    duration: float, beat_frames: np.ndarray, bar_frames: int, quiet: float = 0.05
) -> np.ndarray:
    """
    Return (T, 2) activations with a beat peak on each given frame.

    :param duration: Length in seconds.
    :param beat_frames: Frames that carry a beat peak.
    :param bar_frames: A beat on a multiple of this frame count is a bar start.
    :param quiet: The beat activation between the peaks.
    """
    num_frames = int(duration * FPS)
    beat_act = np.full(num_frames, quiet)
    downbeat_act = np.full(num_frames, 0.4 * quiet)
    for frame in beat_frames:
        if frame < num_frames:
            beat_act[frame] = 0.95
            if frame % bar_frames == 0:
                downbeat_act[frame] = 0.95
    return np.column_stack([np.maximum(beat_act - downbeat_act, 1e-5), downbeat_act])


def _tracker() -> DBNDownBeatTracker:
    """Return a tracker with the meters and tempo range the provider gives its own."""
    return DBNDownBeatTracker(beats_per_bar=[3, 4], min_bpm=55, max_bpm=215, fps=FPS)


def _intervals(beats: np.ndarray) -> np.ndarray:
    """Return a grid's beat intervals in frames."""
    intervals: np.ndarray = np.rint(np.diff(beats[:, 0]) * FPS).astype(int)
    return intervals


def _bar_starts(beats: np.ndarray) -> np.ndarray:
    """Return the frames of a grid's bar starts."""
    frames: np.ndarray = np.rint(beats[beats[:, 1] == 1, 0] * FPS).astype(int)
    return frames


def test_a_track_marked_at_double_time_is_decoded_at_half() -> None:
    """A track the tracker counts at 150 BPM comes out at 75, on the beats that hold its bars."""
    activations = _activations(40.0, np.arange(0, 2000, 20), bar_frames=160)
    tracker = _tracker()

    first, _ = tracker(activations)
    beats, num_beats = decode_one_level(tracker, activations)

    assert np.median(_intervals(first)) == 20
    assert np.median(_intervals(beats)) == 40
    assert num_beats == 4
    # thinned, not moved: every beat of the new grid is a beat of the first one
    assert np.all(np.min(np.abs(beats[:, 0][:, None] - first[:, 0][None, :]), axis=1) <= 0.04)
    assert np.array_equal(_bar_starts(beats), np.arange(0, 2000, 160))


def test_a_stretch_at_double_time_is_decoded_at_the_tracks_tempo() -> None:
    """A stretch the tracker counts at double the track's tempo comes out at one tempo."""
    slow = np.arange(0, 1500, 40)  # 30 s at 75 BPM
    doubled = np.arange(1500, 3000, 20)  # 30 s marked on the half beats too
    slow_again = np.arange(3000, 4500, 40)
    activations = _activations(
        90.0, np.concatenate([slow, doubled, slow_again]), bar_frames=160, quiet=0.01
    )
    tracker = _tracker()

    first, _ = tracker(activations)
    beats, _ = decode_one_level(tracker, activations)

    # the first decode holds the track's tempo in the band and follows the stretch
    assert np.sum(np.abs(_intervals(first) - 40) <= 1) >= 60
    assert np.sum(np.abs(_intervals(first) - 20) <= 1) >= 60
    # the second one does not, and reads the whole track
    assert np.all(np.abs(_intervals(beats) - 40) <= 1)
    assert len(beats) == 113
    assert np.all(_bar_starts(beats) % 160 == 0)


def test_a_part_at_another_tempo_loses_its_beats_in_a_second_decode() -> None:
    """The second decode holds one tempo: the limit of decoding a track again."""
    intro = np.arange(0, 1000, 30)  # 20 s at 100 BPM
    rest = np.arange(1000, 4000, 20)  # 60 s at 150 BPM, which is counted at 75
    activations = _activations(80.0, np.concatenate([intro, rest]), bar_frames=120, quiet=0.01)
    tracker = _tracker()

    first, _ = tracker(activations)
    beats, _ = decode_one_level(tracker, activations)

    # the first decode has the intro on its beats, 30 frames apart
    assert np.sum(_intervals(first) == 30) >= 30
    # the second one has no beat closer than its range allows: 90 BPM is 33 frames
    assert np.all(_intervals(beats) >= 33)

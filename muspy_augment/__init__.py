from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, TypeVar, Union
from muspy.music import Music
import torch
import muspy
import numpy as np


ScorePair = tuple[Music, Music]

@dataclass
class Augmentation:
    probability: float
    
    def __call__(self, scores: ScorePair):
        if torch.rand(1) < self.probability:
            return self.augment(scores)
        else:
            return scores

    def augment(self, scores: ScorePair) -> ScorePair:
        raise NotImplementedError()

@dataclass
class CutMeasures(Augmentation):
    def augment(self, scores: ScorePair):
        score1, score2, aug = cut_phrase(*scores)
        return score1, score2

@dataclass
class RemoveMeasures(Augmentation):
    def augment(self, scores: ScorePair):
        non_tied_measures = get_common_non_tied_measures(scores)
        score1, score2, aug = remove_some_measures(*scores, non_tied_measures)
        return score1, score2

@dataclass
class AddEmptyMeasures(Augmentation):
    def augment(self, scores: ScorePair):
        score1, score2, aug = add_empty_bars(*scores)
        return score1, score2

@dataclass
class EitherOr(Augmentation):
    aug1: Augmentation
    aug2: Augmentation
    aug1_probability: float

    def augment(self, scores: ScorePair):
        if torch.rand(1) < self.aug1_probability:
            return self.aug1(scores)
        else:
            return self.aug2(scores)

def torch_random_choice(list: list[int], n_choices: int, replacement=True):
    """
    Return a list of n_choices random elements from list, with replacement.
    """
    if replacement:
        chosen_indices = torch.randint(0, len(list), (n_choices,))
    else:
        chosen_indices = torch.randperm(len(list))[:n_choices]

    return [list[i] for i in chosen_indices]

def torch_random_int(min_value, max_value) -> int:
    """
    Return a random integer in range [min_value, max_value].
    Note the difference from torch.randint, where the range is [min_value, max_value).
    """
    return int(torch.randint(min_value, max_value + 1, (1,)).item())

fourfour = [muspy.TimeSignature(time=0, numerator=4, denominator=4)]
def add_empty_bars_at(score: Music, bar_indices: dict[int, int]):
    # ASSUME:
    # - Music is only 4/4 without timesig or tempo changes
    # - There are no partial measures (e.g. pickup measures or short last measures).
    assert score.time_signatures == fourfour
    # assert len(score.key_signatures) == 1 ### We don't actually use key signatures anyway
    # assert len(score.tempos) == 1 ## We completely discard tempos for now anyway
    # Take care of beats - add at end, because it doesn't matter where they are added.
    total_bars_added = sum(bar_indices.values())
    beats_to_add = total_bars_added * 4
    last_beat_time = score.beats[-1].time
    beat_duration = score.beats[1].time - score.beats[0].time
    new_score = score.deepcopy()
    for i in range(beats_to_add):
        is_downbeat = (i % 4 == 0)
        beat_time = last_beat_time + beat_duration * (i + 1)
        new_score.beats.append(muspy.Beat(beat_time, is_downbeat))
    
    # To add an empty bar, we simply shift all notes that come after the added bar forward.
    # Note: if there is a tied note before the inserted bar, it will remain as is, tied into the
    # newly inserted "empty" bar - this is fine.
    bar_duration = 4 * beat_duration
    offsets_to_add = {index * bar_duration: num_bars * bar_duration for index, num_bars in bar_indices.items()}
    for track in new_score.tracks:
        offsets = sorted(offsets_to_add)
        cumulative_offset = 0
        assert not track.chords
        for note in track.notes:
            while offsets and note.time >= offsets[0]:
                offset = offsets.pop(0)
                cumulative_offset += offsets_to_add[offset]
            if cumulative_offset:
                note.time += cumulative_offset

    return new_score

    
def add_empty_bars(score1, score2):
    if len(score1.beats) != len(score2.beats):
        # Don't augment if score durations don't match
        return score1, score2, "add_empty_bars: skipped"
    
    # Choose random bar indices (from 0 to min_measure_count) - we will insert an empty bar before
    # each of those.
    # Add between 1 and 5 empty measures.
    num_bars_to_add = int(torch.randint(1, 6, (1,)).item())
    measure_count = get_measure_count(score1)
    # print(num_bars_to_add)
    # We allow to choose the same bar index more than once, so two or more consecutive empty
    # measures can be inserted.
    # We also allow to choose measure_count itself, to insert an empty bar _after_ the last bar.
    bar_indices = torch.randint(0, measure_count + 1, (num_bars_to_add,)).tolist()
    bar_index_counts = Counter(bar_indices)
    # print(bar_index_counts)
    score1_augmented = add_empty_bars_at(score1, bar_index_counts)
    score2_augmented = add_empty_bars_at(score2, bar_index_counts)
    return score1_augmented, score2_augmented, f"add_empty_bars: {bar_index_counts}"


def cut_phrase(score1: Music, score2: Music):
    min_measure_count = min(get_measure_count(score1), get_measure_count(score2))
    if min_measure_count < 4:
        return score1, score2, "cut: skipped"
    rand_value = torch.rand(1)
    cut_measures_from_start = cut_measures_from_end = 0
    if rand_value < 0.3:
        # Cut start - at most half of the bars
        cut_measures_from_start = torch.randint(1, min_measure_count // 2 + 1, (1,)).item()
    elif 0.3 <= rand_value < 0.6:
        # Cut end - at most half of the bars
        cut_measures_from_end = torch.randint(1, min_measure_count // 2 + 1, (1,)).item()
    else:
        # Cut both start and end - at most 1/4 from the start and 1/4 from the end.
        cut_measures_from_start = torch.randint(1, min_measure_count // 4 + 1, (1,)).item()
        cut_measures_from_end = torch.randint(1, min_measure_count // 4 + 1, (1,)).item()
    
    score1_cut = cut_measures_from_phrase(score1, cut_measures_from_start, cut_measures_from_end)
    score2_cut = cut_measures_from_phrase(score2, cut_measures_from_start, cut_measures_from_end)
    return score1_cut, score2_cut, f"cut: start - {cut_measures_from_start}, end - {cut_measures_from_end}"


def get_measure_count(score: Music):
    # Assume fourfour
    # TODO: Adjust for multiple timesigs
    return len(score.beats) // 4


def cut_measures_from_phrase(score: Music, cut_measures_from_start: int, cut_measures_from_end: int):
    end_time = score.get_end_time(infer_last_beat_end=True)
    min_start_time = cut_measures_from_start * 4 * score.resolution
    time_shift = -min_start_time
    max_end_time = end_time - cut_measures_from_end * 4 * score.resolution
    score_new = score.deepcopy()
    for track in score_new:
        track.notes = [
            shift_object(note, time_shift) for note in track.notes
            if note.time >= min_start_time and note.end <= max_end_time
        ]
    # For the beats it's important to use `beat.time < max_end_time` (rather than `<=`).
    score_new.beats = [
        shift_object(beat, time_shift) for beat in score_new.beats
        if beat.time >= min_start_time and beat.time < max_end_time
    ]
    return score_new


BeatOrNote = TypeVar('BeatOrNote', bound=Union[muspy.Beat, muspy.Note])
def shift_object(obj: BeatOrNote, shift: int) -> BeatOrNote:
    obj.time += shift
    return obj


def get_common_non_tied_measures(scores: Iterable[Music]) -> set[int]:
    """
    Return measure indices that are not tied in any of the given scores.
    """
    score_non_tied_measures = [get_non_tied_measures(score) for score in scores]
    return set.intersection(*score_non_tied_measures)


def get_non_tied_measures(score: Music) -> set[int]:
    tied_measures = set()
    for track in score.tracks:
        for note in track.notes:
            start_measure = get_measure(note.start, score)
            end_measure = get_measure(note.end, score, end_time=True)
            if start_measure != end_measure:
                # Note is tied to the next measure, mark both measures as tied.
                tied_measures.update([start_measure, end_measure])
    measure_count = get_measure_count(score)
    all_measures = set(range(measure_count))
    return all_measures - tied_measures


def get_measure(time: float, score: Music, end_time: bool = False) -> int:
    measure_duration = get_measure_duration(score)
    measure_index = int(time // measure_duration)
    if end_time and time % measure_duration == 0:
        # Note end exactly on measure boundary.
        measure_index -= 1
    return measure_index


def remove_some_measures(score1: Music, score2: Music, non_tied_measures: set[int]):
    max_measures_to_remove = len(non_tied_measures) // 2
    if max_measures_to_remove == 0:
        return score1, score2, "remove_some_measures: skipped"
    num_measures_to_remove = torch_random_int(1, max_measures_to_remove)
    measures_to_remove = torch_random_choice(list(non_tied_measures), num_measures_to_remove, replacement=False)
    measures_to_remove = sorted(measures_to_remove)
    score1_cut = remove_measures(score1, measures_to_remove)
    score2_cut = remove_measures(score2, measures_to_remove)
    return score1_cut, score2_cut, f"remove_some_measures: {measures_to_remove}"


def remove_measures(score: Music, measures_to_remove: list[int]):
    """
    Remove the given measure indices from the given score.
    """
    # Prepare two lists of times, which will be used for interpolation later.
    # The first list is the time of each barline measure in the original score (including the final
    # barline). And the second list is the same, but with durations of removed measures set to 0.
    measure_count = get_measure_count(score)
    measure_duration = get_measure_duration(score)
    old_measure_boundaries = range(0, measure_duration * (measure_count + 1), measure_duration)
    new_measure_durations = np.full(measure_count, measure_duration)
    new_measure_durations[measures_to_remove] = 0
    new_measure_end_times = np.cumsum(new_measure_durations)
    new_measure_boundaries = np.pad(new_measure_end_times, (1, 0))
    
    score_new = score.deepcopy()
    for track in score_new.tracks:
        # Remove all notes that are in the removed measures.
        # For all other notes, interpolate their start time to adjust for removed measures.
        track.notes = [
            shift_object_interp(note, old_measure_boundaries, new_measure_boundaries)
            for note in track.notes
            if get_measure(note.time, score) not in measures_to_remove
        ]
    # Remove the number of beats corresponding to the number of measures to remove.
    # Just remove the last beats since order is meaningless. (Assumes single time signature.)
    num_beats_to_remove = len(measures_to_remove) * get_beats_per_measure(score)
    del score_new.beats[-num_beats_to_remove:]
    return score_new

def get_beats_per_measure(score: Music) -> int:
    assert len(score.time_signatures) == 1
    time_signature = score.time_signatures[0]
    return time_signature.denominator

def get_quarter_notes_per_measure(score: Music):
    assert len(score.time_signatures) == 1
    time_signature = score.time_signatures[0]
    return time_signature.numerator / time_signature.denominator * 4

def get_measure_duration(score: Music):
    quarter_notes_per_measure = get_quarter_notes_per_measure(score)
    measure_duration = quarter_notes_per_measure * score.resolution
    assert measure_duration.is_integer()
    return int(measure_duration)

def shift_object_interp(obj: BeatOrNote, old_measure_boundaries, new_measure_boundaries) -> BeatOrNote:
    new_time = np.interp(obj.time, old_measure_boundaries, new_measure_boundaries)
    assert new_time.is_integer()
    obj.time = int(new_time)
    return obj
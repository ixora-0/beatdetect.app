from dataclasses import dataclass, field
from typing import Self

import numpy as np

from .positions import frames_per_position, pos_per_beat


def emission_lp(is_downbeat, is_beat, observation):
    beat_prob = observation[0]
    downbeat_prob = observation[1]
    no_beat_prob = observation[2]
    if is_downbeat:
        return np.log(downbeat_prob + 1e-9)
    elif is_beat:
        return np.log(beat_prob + 1e-9)
    else:
        return np.log(no_beat_prob + 1e-9)


@dataclass(order=True)
class Node:
    lp: float
    state: tuple = field(compare=False)
    prev: Self = field(compare=False, default=None)

    # need to be rounded when use for indexing
    frame: float = field(compare=False, default=0)


def backtrack(node: Node, fps, periods):
    path = []
    while node is not None:
        path.append(node)
        node = node.prev
    path.reverse()

    beats = []
    t = 0
    for node in path:
        period_idx, _, phase = node.state
        period = periods[period_idx]
        if phase % pos_per_beat(period, fps) == 0:
            beat_num = phase // pos_per_beat(period, fps)
            beats.append([t, beat_num + 1])
        t += frames_per_position(period, fps) / fps
    return np.asarray(beats)


def default_beam_width(T, c=450000, w_min=64, w_max=1024):
    return np.clip(round(c / T), w_min, w_max)


def beam_search(config, nn_output, beam_width=None):
    T = nn_output.shape[1]
    fps = config.spectrogram.fps

    if beam_width is None:
        beam_width = default_beam_width(T)

    periods = sorted(
        [(60 / tempo) * config.spectrogram.fps for tempo in config.post.tempo_bins]
    )

    # wrapper for emission_lp
    def calc_emit_lp(state, observation):
        period_idx, _ts, phase = state
        period = periods[period_idx]
        is_downbeat = phase == 0
        is_beat = phase % pos_per_beat(period, fps) == 0
        return emission_lp(is_downbeat, is_beat, observation)

    transitions_file = np.load(config.paths.transitions)
    row_ptrs = transitions_file["row_ptrs"]
    transition_lp = transitions_file["lp"]
    states = transitions_file["states"]
    states_to_idx = transitions_file["states_to_idx"]
    possible_next_states = transitions_file["possible_next_states"]
    init_dist_file = np.load(config.paths.init_dist)
    init_period_dist = init_dist_file["period"]

    start_states_probs = np.array(
        [
            init_period_dist[state[0]] + calc_emit_lp(state, nn_output[:, 0])
            for state in states
        ]
    )  # not normalized, which is probably fine
    start_states_idxs = np.argsort(start_states_probs)[-beam_width:]
    beam = [
        Node(start_states_probs[state_idx], states[state_idx])
        for state_idx in start_states_idxs
    ]
    best_node, best_lp = None, -np.inf

    while beam:
        node = beam.pop()
        period_idx = node.state[0]
        period = periods[period_idx]
        # Frame of next position
        next_frame = node.frame + frames_per_position(period, fps)

        if round(next_frame) >= T:
            if node.lp > best_lp:
                best_node = node
                best_lp = node.lp
            continue

        next_observation = nn_output[:, round(next_frame)]

        # Expand transitions
        state_idx = states_to_idx[*node.state]
        start, end = row_ptrs[state_idx], row_ptrs[state_idx + 1]
        for p in range(start, end):
            next_state_idx = possible_next_states[p]

            next_state = states[next_state_idx]
            emit_lp = calc_emit_lp(next_state, next_observation)

            next_lp = node.lp + transition_lp[p] + emit_lp
            beam.append(Node(next_lp, next_state, node, next_frame))

        beam.sort(key=lambda n: n.lp)
        beam = beam[-beam_width:]
        # # heapq only faster when avg children > 100-200
        # beam = heapq.nlargest(beam_width, beam)[::-1]

    beats = backtrack(best_node, fps, periods)
    return beats, best_lp

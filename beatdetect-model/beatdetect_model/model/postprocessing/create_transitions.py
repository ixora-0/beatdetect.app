import numpy as np
from scipy.special import logsumexp
from tqdm import tqdm

from ...config_loader import Config, load_config
from .positions import pos_per_beat


def period_transition(periods, period_idx, prev_period_idx, pi, lambda1, lambda2):
    period, prev_period = periods[period_idx], periods[prev_period_idx]
    if period_idx == 0:
        left = period - (periods[1] - period) / 2
    else:
        left = (periods[period_idx - 1] + period) / 2
    if period_idx == len(periods) - 1:
        right = period + (period - periods[-2]) / 2
    else:
        right = (period + periods[period_idx + 1]) / 2
    a, b = np.log(left / prev_period), np.log(right / prev_period)

    def lncp(lam):
        return -np.log(2) + logsumexp(b=[1, -1], a=[-lam * a, -lam * b])

    def lncm(lam):
        return -np.log(2) + logsumexp(b=[1, -1], a=[lam * b, lam * a])

    def lncpm(lam):
        return logsumexp(b=[1, -0.5, -0.5], a=[0, -lam * b, lam * a])

    if a >= 0:
        lnc = lncp
    elif b < 0:
        lnc = lncm
    else:
        lnc = lncpm

    return logsumexp(b=[1 - pi, pi], a=[lnc(lambda1), lnc(lambda2)])


def time_signature_transition(ts, ts_pr):
    return 0 if ts == ts_pr else -np.inf


def main(config: Config):
    fps = config.spectrogram.fps
    # create list of states
    periods = sorted([(60 / tempo) * fps for tempo in config.post.tempo_bins])
    max_phase = max(
        max(config.post.time_signatures) * pos_per_beat(period, fps)
        for period in periods
    )
    states = []
    states_to_idx = np.full(
        (
            len(periods),
            len(config.post.time_signatures),
            max_phase,
        ),
        -1,
        dtype=np.int16,
    )
    for period_idx, period in enumerate(periods):
        for ts_idx, beats_per_bar in enumerate(config.post.time_signatures):
            pos_per_bar = beats_per_bar * pos_per_beat(period, fps)
            for phase in range(pos_per_bar):
                s = (period_idx, ts_idx, phase)
                states_to_idx[s] = len(states)
                states.append(s)

    # period transition matrix
    num_periods = len(periods)
    period_transition_matrix = np.zeros((num_periods, num_periods))
    for state_idx in range(num_periods):
        for j in range(num_periods):
            period_transition_matrix[state_idx, j] = period_transition(
                periods,
                state_idx,
                j,
                config.post.pi,
                config.post.lambda1,
                config.post.lambda2,
            )

    # state transition matrix
    num_states = len(states)
    transition_lp = [[] for _ in range(num_states)]
    possible_next_states = [[] for _ in range(num_states)]
    for state_idx, state in tqdm(
        enumerate(states),
        desc="Building transition matrix",
        unit="states",
        total=num_states,
    ):
        period_idx, ts_idx, phase = state
        period = periods[period_idx]
        p = pos_per_beat(period, fps)
        pos_per_bar = config.post.time_signatures[ts_idx] * p
        next_is_beat = (phase + 1) % p == 0

        # next is not beat -> phase + 1
        # next is beat -> other periods (recalc phase)
        # next is downbeat -> potentially other tempos, other periods, phase = 0

        if (phase + 1) % pos_per_bar == 0:
            # End of bar: allow meter change
            # possible_ts_idx = list(range(len(config.post.time_signatures)))
            possible_next_ts_idx = [ts_idx]
        else:
            # Stay in current meter mid-bar
            possible_next_ts_idx = [ts_idx]

        # Only change period on beat
        if next_is_beat:
            possible_period_idx = range(len(periods))
        else:
            possible_period_idx = [period_idx]

        for next_ts_idx in possible_next_ts_idx:
            for next_period_idx in possible_period_idx:
                if next_period_idx == period_idx:
                    next_phase = (phase + 1) % pos_per_bar
                else:
                    # period change, assumes next is beat and same time signature
                    # recalculate phase
                    # since pos per beat might change when period change
                    ts = config.post.time_signatures[ts_idx]
                    beat_num = ((phase + 1) // p) % ts  # % for downbeat case
                    next_period = periods[next_period_idx]
                    next_phase = beat_num * pos_per_beat(next_period, fps)

                lp_period = period_transition_matrix[next_period_idx, period_idx]
                lp_meter = time_signature_transition(next_ts_idx, ts_idx)
                lp = lp_period + lp_meter
                if np.isneginf(lp):
                    continue

                next_state = (next_period_idx, next_ts_idx, next_phase)
                next_state_idx = states_to_idx[next_state]
                transition_lp[state_idx].append(lp)
                possible_next_states[state_idx].append(next_state_idx)

    # Build CSR
    row_ptrs = np.empty(num_states + 1, dtype=np.int64)
    next_states_csr = []
    lp_csr = []
    ptr = 0
    row_ptrs[0] = 0
    for state_idx in range(num_states):
        # flatten
        next_states_csr.extend(possible_next_states[state_idx])
        lp_csr.extend(transition_lp[state_idx])
        row_length = len(transition_lp[state_idx])
        ptr += row_length
        row_ptrs[state_idx + 1] = ptr

    print(f"Saving to {config.paths.transitions}")
    config.paths.transitions.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.paths.transitions,
        row_ptrs=np.array(row_ptrs, dtype=np.uint32),
        lp=np.array(lp_csr, dtype=np.float32),
        states=np.array(states, dtype=np.uint8),
        states_to_idx=states_to_idx,
        possible_next_states=np.array(next_states_csr, dtype=np.uint16),
    )

    print("Done.")


if __name__ == "__main__":
    config = load_config()
    main(config)

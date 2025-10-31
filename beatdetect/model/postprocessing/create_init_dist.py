import numpy as np
import polars as pl

from beatdetect.config_loader import Config, load_config
from beatdetect.utils.paths import iterate_beat_files


def main(config: Config):
    fps = config.spectrogram.fps
    all_periods = []
    for beats_file in iterate_beat_files(config):
        beats = (
            pl.read_csv(
                beats_file,
                separator="\t",
                has_header=False,
            )
            .get_columns()[0]
            .to_list()
        )

        beats = np.array(sorted(beats))
        period = np.diff(beats * fps)
        period = period[period > 0]
        all_periods.append(period)

    all_periods = np.concatenate(all_periods)

    print(f"Loaded {len(all_periods)} songs, {len(all_periods)} beat intervals total.")

    periods = sorted([(60 / tempo) * fps for tempo in config.post.tempo_bins])

    periods_count = np.zeros(len(periods))
    for p in all_periods:
        idx = np.argmin(np.abs(periods - p))
        periods_count[idx] += 1
    init_period_dist = periods_count / periods_count.sum()
    init_period_dist = np.log(init_period_dist + 1e-9)

    print(f"Saving into {config.paths.init_dist}")
    np.savez(config.paths.init_dist, period=init_period_dist)


if __name__ == "__main__":
    config = load_config()
    main(config)

def pos_per_beat(period, fps):
    base_period = (60 / 120) * fps  # period for 120 BPM reference
    base_pos_per_beat = 3  # 3 positions per beat at 120 BPM
    return int(round(base_pos_per_beat * period / base_period))


def frames_per_position(period, fps):
    return period / pos_per_beat(period, fps)

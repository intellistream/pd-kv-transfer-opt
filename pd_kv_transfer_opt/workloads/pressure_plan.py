"""Expand a measured cohort with a frozen higher-rate wave schedule."""

from decimal import Decimal

from pd_kv_transfer_opt.common import read_json, sha256


def expand(path):
    spec = read_json(path)
    parent = path.with_name(spec["source_plan"])
    if sha256(parent) != spec["source_plan_sha256"]:
        raise ValueError("Source cohort has changed.")
    source = read_json(parent)
    if "source_plan" in source:
        source = expand(parent)
    period = Decimal(str(spec["wave_period_s"]))
    within = Decimal(str(source["within_wave_interval_s"]))
    if not 0 < within * (source["wave_size"] - 1) < period:
        raise ValueError("The within-wave arrivals must fit into each wave.")
    source_targets = source["target_requests"]
    waves, remainder = divmod(len(source_targets), source["wave_size"])
    if remainder or not waves:
        raise ValueError("The source cohort must contain complete waves.")
    for index, row in enumerate(source_targets):
        if row["wave"] != index // source["wave_size"] + 1:
            raise ValueError("Unexpected order in the source cohort.")
    targets = []
    repetitions = spec.get("repeat_count", 1)
    if not isinstance(repetitions, int) or repetitions < 1:
        raise ValueError("Expected a positive replay repetition count.")
    for repetition in range(repetitions):
        for index, source_target in enumerate(source_targets):
            wave, slot = divmod(index, source["wave_size"])
            number = repetition * waves + wave
            target = {**source_target, "wave": number + 1, "offered_s": float(number * period + slot * within)}
            if repetitions > 1:
                target["id"] = f"rep{repetition + 1:02d}-{source_target['id']}"
            targets.append(target)
    bounds = spec["analysis_windows_s"]["arrival"]
    if bounds != [targets[0]["offered_s"], targets[-1]["offered_s"]]:
        raise ValueError("The arrival window must match the scheduled requests.")
    return {**source, **spec, "target_requests": targets}

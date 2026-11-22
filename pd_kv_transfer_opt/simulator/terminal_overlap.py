"""Replay an early successor wave against still-running predecessor generation."""

from collections import defaultdict
from math import ceil

from pd_kv_transfer_opt.simulator.terminal_prefill_calibration import paired_quality
from pd_kv_transfer_opt.simulator.terminal_supply import (
    CAP_ORDERS, HOLDOUT, ORDERS, arrange_successors, offsets_for_order, prefill_weights, protect_order,
    root_deliveries, routed_wave, scored_prefill,
)
from pd_kv_transfer_opt.simulator.terminal_wave import decode_deliveries, ongoing_supply_gap


def early_release_probe(workload, selected, models, x, guard_s, credit_slack_s):
    """Keep one arrival plan for both caps, delaying only successors with unfinished parents."""
    pairs = workload["pairs"]
    by_child = {pair["successor"]: pair for pair in pairs}
    ingress = {key: x.fit["offset_s"] + x.fit["s_per_byte"] * x.sizes[key] for key in (
        selected["root_offers"] | selected["successor_offers"])}
    weights = prefill_weights(x.sizes)
    roots = sorted(selected["root_offers"], key=selected["root_offers"].__getitem__)
    root_ends, latest_credit, trials = defaultdict(float), 0.0, []
    layout = None
    for service, error, refs in models:
        runs = []
        for label, (root_residual, _) in x.profiles.items():
            offsets = offsets_for_order(roots, root_residual)
            arrivals = {key: t + ingress[key] + offsets[key] for key, t in selected["root_offers"].items()}
            roles = scored_prefill(arrivals, weights, {})
            if layout is not None and roles != layout:
                raise ValueError("The diagnostic requires reproducible root ingress")
            layout = roles
            order = sorted(arrivals, key=arrivals.__getitem__)
            work = x.prefixes.work(order, roles)
            for order_name, ref, eos in zip(ORDERS, refs, selected["eos_samples"], strict=True):
                slow = {**ref, "cold_tokens_s_per_P": ref["cold_tokens_s_per_P"] * x.cold_fraction}
                root = root_deliveries(order, roles, arrivals, work, eos, x.sizes, slow)
                first = decode_deliveries(*root, x.decode, 16, ref["root_handoff_s"])["per_request"]
                latest_credit = max(latest_credit, max(row["first_s"] for row in first.values()))
                for key, row in first.items():
                    root_ends[key] = max(root_ends[key], row["end_s"])
                runs.append((label, order_name, ref, eos, root, first))
        trials.append((service, error, runs))
    ready = {key: root_ends[pair["root"]] + guard_s for key, pair in by_child.items()}
    start = ceil((latest_credit + guard_s) * 10) / 10 + credit_slack_s
    groups = defaultdict(list)
    late = []
    for key, pair in by_child.items():
        if ready[key] <= start:
            groups[pair["task"]].append(pair)
        else:
            late.append(key)
    for rows in groups.values():
        rows.sort(key=lambda row: (row["checkpoint_blocks"], row["root"]))
    early, _ = arrange_successors(groups, layout, weights)
    order = early + sorted(late, key=lambda key: (ready[key], key))
    clocks = {name: offsets_for_order(order, child_residual) for name, (_, child_residual) in x.profiles.items()}
    offers = {key: max(ready[key], start + ingress[order[0]] + (.11 + (i - 1) * .022 if i else 0) - ingress[key])
              for i, key in enumerate(order)}
    local_attention = x.prefixes.earlier_root_attention(pairs, layout)
    holds = defaultdict(float)
    for _ in range(8):
        expected = {key: t + ingress[key] for key, t in offers.items()}
        offers, nominal = protect_order(order, offers, expected, list(clocks.values()))
        shifts, fitted = {}, []
        first_offer = min(offers.values())
        for service, error, runs in trials:
            cases = {}
            for label, state, ref, eos, roots_work, root_alone in runs:
                arrivals = {key: t + clocks[label][key] for key, t in nominal.items()}
                if min(arrivals.values()) <= max(r["first_s"] for r in root_alone.values()):
                    return {"reason": "A root proxy score was still outstanding", "earliest_offer_s": first_offer}
                arms, roles = {}, {}
                for cap in CAP_ORDERS[ORDERS.index(state)]:
                    row, dest = routed_wave(arrivals, layout, x.sizes, weights, eos, ref, service, x.decode, cap,
                                            first_offer, by_child, x.ledger, lead_credit_s=error["r78_positive_first_reply_error_s"],
                                            root_work=roots_work, local_attention=local_attention)
                    if row is None:
                        return {"reason": "No stable native routing for the offered overlap", "earliest_offer_s": first_offer}
                    arms[str(cap)], roles[str(cap)] = row, dest
                    for key, pair in by_child.items():
                        required = row["sim"]["per_request"][pair["root"]]["end_s"] + guard_s
                        if offers[key] < required - 1e-8:
                            shifts[key] = max(shifts.get(key, 0.0), required)
                low, high = (arms[cap]["report"] for cap in ("100", "200"))
                cases[f"{label}/{state}"] = {"arms": {"100": low, "200": high},
                    "100_low_loss_fraction": 1 - high["duration_s"] / low["duration_s"],
                    "same_P_routes": sum(roles["100"][key] == roles["200"][key] for key in order),
                    "ongoing_D_supply_gap": ongoing_supply_gap(arms["100"], arms["200"])}
            quality = paired_quality(cases, len(pairs), x.r69["r69_max_200_Gbps_in_20_s"],
                                     error["r78_positive_finish_gap_error_s"])
            fitted.append({"service": service, "quality": quality, "main_ABBA": {
                key: cases[f"{HOLDOUT}/{key}-cap200/{key}"] for key in ORDERS}})
        if not shifts:
            return {"earliest_offer_s": first_offer, "initial_ready_successors": len(early),
                    "successor_offers_s": offers, "extra_dependency_hold_s": dict(holds), "models": fitted}
        for key, after in shifts.items():
            holds[key] += after - offers[key]
            offers[key] = after
    return {"reason": "Parent EOS bounds did not settle", "earliest_offer_s": min(offers.values())}

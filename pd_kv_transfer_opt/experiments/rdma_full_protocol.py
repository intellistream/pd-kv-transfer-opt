"""Repeated hardware-cap comparison, low-load controls, and separate trace windows."""
from pd_kv_transfer_opt.experiments.rdma_hard_cap_protocol import OUTPUT_TOKENS, PREFIXES, PROMPT_LENGTHS, WORKLOAD_SHA256, request_id


RUN_ID = '2p2d-20261118-natural-repro-r14'
RATE = 16.0
STOCK = 1600
ORDERS = ((100, 400, 200, STOCK), (STOCK, 200, 400, 100), (200, 100, STOCK, 400))
CASES = (('warm', None, 2),) + tuple(
    (f'b{block}-cap{limit}', limit, 96)
    for block, order in enumerate(ORDERS, 1)
    for limit in order
) + (('low1-cap100', 100, 12), ('low1-cap400', 400, 12),
     ('low2-cap400', 400, 12), ('low2-cap100', 100, 12),
     ('profile-cap100', 100, 24), ('profile-cap400', 400, 24),
     ('profile-cap1600', STOCK, 24))


def offered_rate(case):
    return 0.5 if case.startswith('low') else RATE

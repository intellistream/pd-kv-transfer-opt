"""Fixed workload and port rates for the D-to-P hardware-shaper experiment."""

RUN_ID = '2p2d-20261118-natural-repro-r13'
PREFIXES = ('regex-chess:120', 'winning-avg-corewars:145')
PROMPT_LENGTHS = (223256, 238600)
WORKLOAD_SHA256 = '84edc8dad9c7704184c9ace8d0088b8758c6b5197536641b31c4e306140ed021'
OUTPUT_TOKENS = 16
OFFERED_REQUESTS_S = 16.0
CASES = (('warm', None, 2), ('cap100a', 100, 96), ('cap200', 200, 96),
         ('cap400', 400, 96), ('cap100b', 100, 96))


def request_id(case, index):
    return f'{case}-{index:03}'

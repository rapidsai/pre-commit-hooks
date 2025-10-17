from collections.abc import Hashable, Sequence

def distance(
    s1: Sequence[Hashable],
    s2: Sequence[Hashable],
    weights=(1, 1, 1),
    processor=None,
    score_cutoff=None,
    score_hint=None,
) -> int: ...

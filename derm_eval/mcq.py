"""Multiple-choice answer format: per-case options = the case label + distractors, shuffled.

Distractors are either drawn per case from the SCIN label pool (mcq_weighting) or, with
mcq_distractors, the same fixed list for every case (only the option order changes per case).

Options are deterministic per (case, mcq_seed), so a case shows the same options in the same order
in every stage, mode and sample; only the context differs between stages.
"""
import json
import random

from .config import resolve


def load_pool(cfg):
    """{label: case count} of distractor candidates from cfg['mcq_pool_path']."""
    with open(resolve(cfg["mcq_pool_path"])) as f:
        return json.load(f)["labels"]


def build_options(case_id, label, pool, n_choices=4, seed=0, weighting="uniform", fixed=None):
    """Shuffled list of n_choices condition names: `label` plus n_choices - 1 distinct distractors.

    fixed:     use exactly these distractors for every case (overrides weighting).
    weighting: "uniform" draws distractors uniformly over distinct condition names;
               "frequency" draws them in proportion to how many SCIN cases carry each name.
    """
    rng = random.Random(f"mcq:{seed}:{case_id}")
    if fixed:
        fixed = list(fixed)
        if len(fixed) != n_choices - 1 or len(set(fixed)) != len(fixed) or label in fixed:
            raise ValueError(f"mcq_distractors must be {n_choices - 1} distinct names other than {label!r}, got {fixed}")
        unknown = [d for d in fixed if d not in pool]
        if unknown:
            raise ValueError(f"mcq_distractors not in the SCIN label pool (check spelling): {unknown}")
        options = fixed + [label]
        rng.shuffle(options)
        return options
    candidates = [name for name in pool if name != label]
    if len(candidates) < n_choices - 1:
        raise ValueError(f"distractor pool has only {len(candidates)} labels, need {n_choices - 1}")
    if weighting == "uniform":
        distractors = rng.sample(candidates, n_choices - 1)
    elif weighting == "frequency":
        distractors, weights = [], [pool[c] for c in candidates]
        while len(distractors) < n_choices - 1:
            pick = rng.choices(candidates, weights)[0]
            if pick not in distractors:
                distractors.append(pick)
    else:
        raise ValueError(f"unknown mcq_weighting {weighting!r}")
    options = distractors + [label]
    rng.shuffle(options)
    return options


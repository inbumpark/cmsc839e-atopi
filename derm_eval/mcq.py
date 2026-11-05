"""Multiple-choice answer format: per-case options = the case label + random distractors, shuffled.

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


def build_options(case_id, label, pool, n_choices=4, seed=0, weighting="uniform"):
    """Shuffled list of n_choices condition names: `label` plus n_choices - 1 distinct distractors.

    weighting: "uniform" draws distractors uniformly over distinct condition names;
               "frequency" draws them in proportion to how many SCIN cases carry each name.
    """
    candidates = [name for name in pool if name != label]
    if len(candidates) < n_choices - 1:
        raise ValueError(f"distractor pool has only {len(candidates)} labels, need {n_choices - 1}")
    rng = random.Random(f"mcq:{seed}:{case_id}")
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


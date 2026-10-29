"""Map free-text diagnoses to canonical SCIN labels using configs/labels.yaml."""
import difflib
import re

import yaml

UNMATCHED = "UNMATCHED"


def clean(s):
    s = str(s).lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class Normalizer:
    def __init__(self, labels_path=None, extra_labels=(), fuzzy_cutoff=0.85):
        mapping = yaml.safe_load(open(labels_path)) if labels_path else {}
        mapping = dict(mapping or {})
        for label in extra_labels:          # labels present in the data are always matchable
            mapping.setdefault(label, [])
        self.alias_to_label = {}
        for label, aliases in mapping.items():
            for alias in [label, *(aliases or [])]:
                self.alias_to_label.setdefault(clean(alias), label)
        self.aliases = sorted(self.alias_to_label, key=len, reverse=True)
        self.patterns = [(a, re.compile(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])")) for a in self.aliases if a]
        self.fuzzy_cutoff = fuzzy_cutoff

    def __call__(self, text):
        if not text:
            return UNMATCHED
        t = clean(text)
        if t in self.alias_to_label:
            return self.alias_to_label[t]
        # Earliest alias mentioned in the answer wins; the longest alias breaks ties
        # (so "seborrheic dermatitis" beats "dermatitis").
        best = None
        for alias, pat in self.patterns:
            m = pat.search(t)
            if m and (best is None or m.start() < best[0]):
                best = (m.start(), alias)
        if best:
            return self.alias_to_label[best[1]]
        close = difflib.get_close_matches(t, self.aliases, n=1, cutoff=self.fuzzy_cutoff)
        return self.alias_to_label[close[0]] if close else UNMATCHED

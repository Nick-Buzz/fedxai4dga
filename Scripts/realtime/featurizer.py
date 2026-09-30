"""Online feature extraction: one raw domain name in, one scaled model row out.

The offline pipeline (``Data/Raw/pipeline.sh``) produced the training features
in three steps that are replayed here, per domain, exactly as a DNS sensor
would have to:

  1. ``reduce_and_label.py``  strips the public suffix (Mozilla PSL) so that
     ``juiseforced.com`` becomes ``juiseforced``;
  2. ``feature_extractor.py`` computes 50 lexical features on that prefix,
     including the n-gram reputation against the Tranco top-100k whitelist
     and a ``wordninja`` word segmentation;
  3. ``Ratio_DeciDig`` is dropped by the 0.9 correlation screen, and min-max
     scaling is applied with the minimum and maximum of the TRAINING split.

The feature functions are imported from ``feature_extractor`` rather than
re-implemented, so the timed code is the code that built the dataset.  The one
exception is the character-frequency block, which the original returns as a
CSV string; here the same 36 counts are produced as integers directly.
``Scripts/realtime/build_preprocessor.py`` verifies that the output matches the
stored test features.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Scripts.Preprocessing import feature_extractor as fx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "Data" / "Raw"
PREPROCESSOR_FILE = PROJECT_ROOT / "Results" / "realtime" / "preprocessor.json"

# Column order of labeled_dataset_features.csv (feature_extractor's header).
RAW_FEATURES = (
    ["Length", "Max_DeciDig_Seq", "Max_Let_Seq"]
    + [f"Freq_{c}" for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"]
    + ["Spec_Char_Freq", "Ratio_Spec_Char", "DeciDig_Freq", "Ratio_DeciDig",
       "Vowel_Freq", "Vowel_Ratio", "Max_Gap", "Reputation", "Words_Freq",
       "Words_Mean", "Entropy"]
)
DROPPED_BY_CORRELATION = ("Ratio_DeciDig",)
MODEL_FEATURES = [f for f in RAW_FEATURES if f not in DROPPED_BY_CORRELATION]
_KEEP = np.array([RAW_FEATURES.index(f) for f in MODEL_FEATURES])
_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def load_suffixes(path: Path = RAW_DIR / "public_suffixes_list_v2.csv") -> frozenset:
    with open(path, encoding="utf-8") as fh:
        return frozenset(line.strip() for line in fh)


def load_whitelist(path: Path = RAW_DIR / "tranco_top100k.txt") -> frozenset:
    """N-gram whitelist, built as ``feature_extractor.get_ngram_whitelist``."""
    names, seen = [], set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            name = line.strip()
            if name not in seen:
                seen.add(name)
                names.append(name)
                if len(names) == 100_000:
                    break
    ngrams: set = set()
    for name in names:
        fx.find_ngrams(fx.get_label(name), ngrams)
    return frozenset(ngrams)


def strip_public_suffix(name: str, suffixes: frozenset) -> str | None:
    """``reduce_and_label.py``'s suffix removal; None when nothing remains."""
    labels = name.split(".")
    labels.reverse()
    candidate, index = labels[0], 0
    try:
        while candidate in suffixes:
            index += 1
            candidate = labels[index] + "." + candidate
    except IndexError:
        return None
    labels.reverse()
    return ".".join(labels[: len(labels) - index])


def word_features(prefix: str) -> tuple[int, float]:
    """``find_words_number`` and ``find_words_mean_length`` from one segmentation.

    The original functions each run ``wordninja.split`` on every label, so the
    segmentation, about three quarters of the extraction time, is done twice.
    Same output, half the work.
    """
    lengths = [len(w) for label in prefix.split(".") for w in fx.wordninja.split(label) if len(w) > 2]
    return len(lengths), (float(np.mean(lengths)) if lengths else 0)


def raw_features(prefix: str, whitelist: frozenset, fast: bool = False) -> list[float]:
    """The 50 unscaled features of ``feature_extractor.export_features``."""
    special = fx.find_special_char_frequency(prefix)
    integers = fx.find_integer_frequency(prefix)
    vowels = fx.find_vowel_frequency(prefix)
    if fast:
        words_number, words_mean = word_features(prefix)
    else:
        words_number, words_mean = fx.find_words_number(prefix), fx.find_words_mean_length(prefix)
    return [
        fx.find_length(prefix),
        fx.find_max_digit_sequence(prefix),
        fx.find_max_string_sequence(prefix),
        *[prefix.count(c) for c in _CHARS],
        special,
        fx.find_ratio_special_char(prefix, special),
        integers,
        fx.find_integer_ratio(prefix, integers),
        vowels,
        fx.find_vowels_ratio(prefix, vowels),
        fx.find_maximum_gap_between_dots(prefix),
        fx.find_reputation(prefix, whitelist),
        words_number,
        words_mean,
        fx.get_shannon_entropy(prefix),
    ]


class DomainFeaturizer:
    """Stateless after construction; construction loads the PSL and whitelist."""

    def __init__(self, preprocessor_file: Path = PREPROCESSOR_FILE, fast: bool = False):
        self.fast = fast
        spec = json.loads(Path(preprocessor_file).read_text(encoding="utf-8"))
        if spec["features"] != MODEL_FEATURES:
            raise ValueError("preprocessor.json feature order differs from MODEL_FEATURES")
        self.minimum = np.asarray(spec["minimum"], dtype=np.float64)
        span = np.asarray(spec["maximum"], dtype=np.float64) - self.minimum
        self.inv_span = np.where(span > 0, 1.0 / np.where(span > 0, span, 1.0), 0.0)
        self.suffixes = load_suffixes()
        self.whitelist = load_whitelist()
        self.n_features = len(MODEL_FEATURES)

    def prefix(self, domain: str) -> str | None:
        return strip_public_suffix(domain.strip().rstrip(".").lower(), self.suffixes)

    def transform_one(self, domain: str, out: np.ndarray | None = None) -> np.ndarray:
        prefix = self.prefix(domain)
        if not prefix:
            # A bare public suffix has no registrable label to judge; score the
            # name itself so the caller still receives a verdict.
            prefix = domain.strip().rstrip(".").lower()
        raw = np.asarray(raw_features(prefix, self.whitelist, self.fast), dtype=np.float64)[_KEEP]
        scaled = (raw - self.minimum) * self.inv_span
        if out is None:
            return scaled.astype(np.float32)
        out[:] = scaled
        return out

    def transform(self, domains) -> np.ndarray:
        X = np.empty((len(domains), self.n_features), dtype=np.float32)
        for i, domain in enumerate(domains):
            self.transform_one(domain, X[i])
        return X

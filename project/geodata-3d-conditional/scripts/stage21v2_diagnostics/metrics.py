"""Fixed retrospective ranking summaries; no fitting or threshold optimization."""
import numpy as np


def average_precision(y_true, scores):
    """Non-interpolated AP, grouping all equal-score observations at one threshold.

    Returns thresholds in decreasing order, with recall increasing. AP is the
    sum of recall increments times precision after the entire tied group enters.
    This is the binary, unweighted sklearn average_precision_score definition.
    """
    truth = np.asarray(y_true, dtype=bool).ravel()
    score = np.asarray(scores).ravel()
    if truth.shape != score.shape or not len(score) or not np.isfinite(score).all():
        raise ValueError("AP requires equal, finite, nonempty vectors")
    order = np.argsort(-score, kind="stable")
    ordered = score[order]
    ends = np.r_[np.flatnonzero(ordered[1:] != ordered[:-1]), len(score)-1]
    tp = np.cumsum(truth[order], dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    positives = int(truth.sum())
    recall = tp / positives if positives else np.zeros(len(ends))
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else None
    return ap, dict(threshold=ordered[ends], recall=recall, precision=precision,
                    true_positive=tp, selected_count=ends+1)


def fixed_top5(scores):
    """Select at least ceil(5% * n) values, including ALL ties at that rank."""
    score = np.asarray(scores).ravel()
    if not len(score) or not np.isfinite(score).all():
        raise ValueError("top5 requires finite nonempty scores")
    requested = int(np.ceil(.05 * len(score)))
    boundary = len(score)-requested
    threshold = np.partition(score, boundary)[boundary]
    return score >= threshold, float(threshold), requested


def quantiles(values):
    values = np.asarray(values).ravel()
    levels = dict(min=0, p05=.05, p10=.1, p25=.25, median=.5, p75=.75,
                  p90=.9, p95=.95, max=1)
    if not len(values):
        return {name: None for name in levels}
    return dict(zip(levels, map(float, np.quantile(values, list(levels.values())))))

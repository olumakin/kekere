import math
from typing import Dict, Tuple, List, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression

class PlattCalibrator:
    """
    Platt Scaling probability calibrator.
    Fits logistic regression on log-odds (logit) of raw model predictions.
    """

    def __init__(self):
        self.clf = LogisticRegression(C=1.0, solver="lbfgs")
        self.is_fitted = False

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(p, 1e-5, 1.0 - 1e-5)
        return np.log(p_clipped / (1.0 - p_clipped))

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray):
        """
        Fit Platt scaling on historical raw probabilities and binary labels (1=draw, 0=not draw).
        """
        if len(raw_probs) < 20 or len(np.unique(y_true)) < 2:
            # Insufficient samples or single class, keep identity
            self.is_fitted = False
            return self

        logits = self._logit(raw_probs).reshape(-1, 1)
        self.clf.fit(logits, y_true)
        self.is_fitted = True
        return self

    def predict_proba(self, raw_probs: np.ndarray) -> np.ndarray:
        """Calibrate raw probabilities."""
        if not self.is_fitted:
            return raw_probs

        logits = self._logit(raw_probs).reshape(-1, 1)
        calibrated = self.clf.predict_proba(logits)[:, 1]
        return calibrated

    def calibrate_single(self, raw_prob: float) -> float:
        if not self.is_fitted:
            return raw_prob
        return float(self.predict_proba(np.array([raw_prob]))[0])


class CoxCalibrationEvaluator:
    """
    Evaluates Cox logistic calibration model:
    logit(P(Y=1)) = alpha (intercept) + beta (slope) * logit(p_pred)
    Ideal: slope = 1.0, intercept = 0.0
    """

    @staticmethod
    def evaluate(probs: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        if len(probs) < 20 or len(np.unique(y_true)) < 2:
            return {"slope": 1.0, "intercept": 0.0}

        p_clipped = np.clip(probs, 1e-5, 1.0 - 1e-5)
        logits = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

        # Unregularized (high C) logistic regression to measure true slope and intercept
        model = LogisticRegression(C=1e5, solver="lbfgs", max_iter=500)
        model.fit(logits, y_true)

        slope = float(model.coef_[0][0])
        intercept = float(model.intercept_[0])

        return {
            "slope": round(slope, 4),
            "intercept": round(intercept, 4),
        }


def compute_calibration_metrics(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """
    Computes Brier Score, Log Loss, and Expected Calibration Error (ECE).
    """
    n = len(probs)
    if n == 0:
        return {"brier_score": 0.0, "log_loss": 0.0, "ece": 0.0}

    probs = np.clip(probs, 1e-5, 1.0 - 1e-5)
    
    # Brier Score
    brier = float(np.mean((probs - y_true) ** 2))

    # Log Loss
    log_loss = float(-np.mean(y_true * np.log(probs) + (1.0 - y_true) * np.log(1.0 - probs)))

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return {
        "brier_score": round(brier, 4),
        "log_loss": round(log_loss, 4),
        "ece": round(ece, 4),
    }

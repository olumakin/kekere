from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from skynet.models.calibration import compute_calibration_metrics

class SecondStageEvaluator:
    """
    Evaluates whether a second-stage classifier (Random Forest) improves out-of-sample
    Brier score and CLV compared to Dixon-Coles baseline alone (§4 test).
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, random_state: int = 42):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            min_samples_leaf=10
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        X: features array where column 0 is Dixon-Coles draw prob, followed by engineered features
        y: binary outcomes (1=draw, 0=not draw)
        """
        if len(X) < 30 or len(np.unique(y)) < 2:
            return self
        self.rf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            # Fallback to Dixon-Coles raw prob (first column)
            return X[:, 0]
        return self.rf.predict_proba(X)[:, 1]

    @staticmethod
    def evaluate_gate_comparison(
        dc_probs: np.ndarray,
        rf_probs: np.ndarray,
        y_true: np.ndarray,
        devigged_closing_draw_odds: np.ndarray
    ) -> Dict[str, Any]:
        """
        Determines whether 2nd-stage classifier satisfies §4 criteria:
        Must strictly improve BOTH Brier score AND average CLV over Dixon-Coles alone.
        """
        dc_metrics = compute_calibration_metrics(dc_probs, y_true)
        rf_metrics = compute_calibration_metrics(rf_probs, y_true)

        # Calculate CLV: model edge against devigged closing market prob (1 / closing_odds)
        market_closing_probs = 1.0 / np.clip(devigged_closing_draw_odds, 1.01, 50.0)
        dc_clv = float(np.mean(dc_probs - market_closing_probs))
        rf_clv = float(np.mean(rf_probs - market_closing_probs))

        # Improvement: lower Brier score is better; higher CLV is better
        brier_improved = rf_metrics["brier_score"] < dc_metrics["brier_score"]
        clv_improved = rf_clv > dc_clv
        should_adopt = brier_improved and clv_improved

        return {
            "should_adopt_second_stage": should_adopt,
            "dixon_coles": {
                "brier_score": dc_metrics["brier_score"],
                "avg_clv": round(dc_clv, 4),
            },
            "random_forest": {
                "brier_score": rf_metrics["brier_score"],
                "avg_clv": round(rf_clv, 4),
            },
            "brier_improved": brier_improved,
            "clv_improved": clv_improved,
        }

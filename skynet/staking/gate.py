from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class GateResult:
    is_passed: bool
    status: str # 'ENABLED' or 'DISABLED'
    bet_count: int
    roi: float
    avg_clv: float
    cox_slope: float
    cox_intercept: float
    failures: List[str]
    details: Dict[str, Any]

class GoNoGoGate:
    """
    Evaluates institutional risk gate (§2 & §7):
    Staking remains DISABLED unless:
      1. Minimum bet count >= min_bets (default 150)
      2. Return on Investment (ROI) > 0.0
      3. Average Closing Line Value (CLV) > 0.0
      4. Cox calibration slope in [0.8, 1.2]
      5. Cox calibration intercept in [-0.05, 0.05]
    """

    def __init__(
        self,
        min_bets: int = 150,
        min_roi: float = 0.0,
        min_clv: float = 0.0,
        cox_slope_min: float = 0.8,
        cox_slope_max: float = 1.2,
        cox_intercept_min: float = -0.05,
        cox_intercept_max: float = 0.05
    ):
        self.min_bets = min_bets
        self.min_roi = min_roi
        self.min_clv = min_clv
        self.cox_slope_min = cox_slope_min
        self.cox_slope_max = cox_slope_max
        self.cox_intercept_min = cox_intercept_min
        self.cox_intercept_max = cox_intercept_max

    def evaluate(
        self,
        bet_count: int,
        roi: float,
        avg_clv: float,
        cox_slope: float,
        cox_intercept: float
    ) -> GateResult:
        failures = []

        if bet_count < self.min_bets:
            failures.append(f"Insufficient bets: {bet_count} < {self.min_bets}")

        if roi <= self.min_roi:
            failures.append(f"ROI not positive: {roi:.2%} <= {self.min_roi:.2%}")

        if avg_clv <= self.min_clv:
            failures.append(f"CLV not positive: {avg_clv:.2%} <= {self.min_clv:.2%}")

        if not (self.cox_slope_min <= cox_slope <= self.cox_slope_max):
            failures.append(f"Cox slope {cox_slope:.4f} outside [{self.cox_slope_min}, {self.cox_slope_max}]")

        if not (self.cox_intercept_min <= cox_intercept <= self.cox_intercept_max):
            failures.append(f"Cox intercept {cox_intercept:.4f} outside [{self.cox_intercept_min}, {self.cox_intercept_max}]")

        is_passed = len(failures) == 0
        status = "ENABLED" if is_passed else "DISABLED"

        details = {
            "bet_count": bet_count,
            "min_bets_required": self.min_bets,
            "roi": round(roi, 4),
            "avg_clv": round(avg_clv, 4),
            "cox_slope": round(cox_slope, 4),
            "cox_intercept": round(cox_intercept, 4),
            "criteria_met": {
                "bets": bet_count >= self.min_bets,
                "roi": roi > self.min_roi,
                "clv": avg_clv > self.min_clv,
                "cox_slope": self.cox_slope_min <= cox_slope <= self.cox_slope_max,
                "cox_intercept": self.cox_intercept_min <= cox_intercept <= self.cox_intercept_max,
            }
        }

        return GateResult(
            is_passed=is_passed,
            status=status,
            bet_count=bet_count,
            roi=roi,
            avg_clv=avg_clv,
            cox_slope=cox_slope,
            cox_intercept=cox_intercept,
            failures=failures,
            details=details
        )

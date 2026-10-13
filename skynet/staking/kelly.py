from typing import Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class StakeRecommendation:
    stake_pct: float
    stake_amount: float
    is_bet: bool
    reason: str
    formula_inputs: Dict[str, Any]

class KellyCalculator:
    """
    Fractional Kelly Criterion calculator with bankroll caps, daily exposure limits,
    and losing streak circuit breakers.
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_stake_pct: float = 0.02,
        daily_exposure_limit_pct: float = 0.06,
        circuit_breaker_loss_streak: int = 5,
        base_bankroll: float = 10000.0
    ):
        self.kelly_fraction = kelly_fraction
        self.max_stake_pct = max_stake_pct
        self.daily_exposure_limit_pct = daily_exposure_limit_pct
        self.circuit_breaker_loss_streak = circuit_breaker_loss_streak
        self.bankroll = base_bankroll
        self.current_loss_streak = 0
        self.daily_staked_pct = 0.0

    def reset_daily_exposure(self):
        self.daily_staked_pct = 0.0

    def record_bet_outcome(self, is_win: bool, stake_amount: float, decimal_odds: float):
        if is_win:
            self.current_loss_streak = 0
            self.bankroll += stake_amount * (decimal_odds - 1.0)
        else:
            self.current_loss_streak += 1
            self.bankroll = max(0.0, self.bankroll - stake_amount)

    def calculate_stake(
        self,
        draw_prob: float,
        draw_odds: float,
        edge: float,
        draw_prob_threshold: float = 0.31,
        min_edge: float = 0.03,
        current_daily_staked: float = 0.0
    ) -> StakeRecommendation:
        """
        Calculates suggested stake for a fixture.
        Returns full recommendation even if target thresholds are not met (for audit trail).
        """
        b = draw_odds - 1.0
        p = draw_prob
        q = 1.0 - p

        # Raw Kelly: (b*p - q) / b
        raw_kelly = (b * p - q) / b if b > 0 else 0.0
        fractional_kelly = max(0.0, self.kelly_fraction * raw_kelly)

        formula_inputs = {
            "draw_prob": round(draw_prob, 4),
            "draw_odds": round(draw_odds, 4),
            "b": round(b, 4),
            "edge": round(edge, 4),
            "kelly_fraction_multiplier": self.kelly_fraction,
            "raw_kelly": round(raw_kelly, 4),
            "unconstrained_pct": round(fractional_kelly, 4),
            "current_loss_streak": self.current_loss_streak,
            "bankroll": round(self.bankroll, 2),
        }

        # Check threshold qualification
        if draw_prob < draw_prob_threshold:
            return StakeRecommendation(
                stake_pct=0.0,
                stake_amount=0.0,
                is_bet=False,
                reason=f"Draw prob ({draw_prob:.1%}) below threshold ({draw_prob_threshold:.1%})",
                formula_inputs=formula_inputs
            )

        if edge < min_edge:
            return StakeRecommendation(
                stake_pct=0.0,
                stake_amount=0.0,
                is_bet=False,
                reason=f"Edge ({edge:.1%}) below min edge ({min_edge:.1%})",
                formula_inputs=formula_inputs
            )

        if fractional_kelly <= 0:
            return StakeRecommendation(
                stake_pct=0.0,
                stake_amount=0.0,
                is_bet=False,
                reason="Negative or zero Kelly expectation",
                formula_inputs=formula_inputs
            )

        # 1. Apply max stake cap
        staked_pct = min(fractional_kelly, self.max_stake_pct)

        # 2. Apply circuit breaker if on losing streak
        circuit_breaker_active = False
        if self.current_loss_streak >= self.circuit_breaker_loss_streak:
            staked_pct *= 0.5 # Halve stake during protracted losing streak
            circuit_breaker_active = True
        formula_inputs["circuit_breaker_active"] = circuit_breaker_active

        # 3. Apply daily exposure limit
        remaining_daily_capacity = max(0.0, self.daily_exposure_limit_pct - current_daily_staked)
        if staked_pct > remaining_daily_capacity:
            staked_pct = remaining_daily_capacity
        formula_inputs["remaining_daily_capacity"] = round(remaining_daily_capacity, 4)

        if staked_pct <= 0:
            return StakeRecommendation(
                stake_pct=0.0,
                stake_amount=0.0,
                is_bet=False,
                reason="Daily exposure limit reached",
                formula_inputs=formula_inputs
            )

        stake_amount = round(self.bankroll * staked_pct, 2)

        return StakeRecommendation(
            stake_pct=round(staked_pct, 4),
            stake_amount=stake_amount,
            is_bet=True,
            reason="VALUE_BET",
            formula_inputs=formula_inputs
        )

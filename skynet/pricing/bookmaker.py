from typing import Dict, Any, Optional
from skynet.pricing.base import BasePricingAdapter, PricingResult

class BookmakerPricingAdapter(BasePricingAdapter):
    """
    Pricing adapter for bookmaker odds.
    Normalizes overround (vig) across 1X2 market prices.
    If full 1X2 odds are missing, falls back to raw 1/odds and flags is_vig_adjusted=False.
    """

    def __init__(self, normalization_method: str = "multiplicative"):
        """
        normalization_method: 'multiplicative' (proportional) or 'shin'
        """
        self.normalization_method = normalization_method.lower()

    def calculate_market_draw_probability(
        self,
        draw_odds: float,
        home_odds: Optional[float] = None,
        away_odds: Optional[float] = None,
        fee_assumptions: Optional[Dict[str, Any]] = None,
        venue: str = "PINNACLE"
    ) -> PricingResult:
        if draw_odds <= 1.0:
            raise ValueError(f"Draw odds must be > 1.0, got {draw_odds}")

        raw_draw_prob = 1.0 / draw_odds
        fees = fee_assumptions or {"commission": 0.0}

        # Check if full 1X2 market odds are available
        has_full_market = (
            home_odds is not None 
            and away_odds is not None 
            and home_odds > 1.0 
            and away_odds > 1.0
        )

        if not has_full_market:
            # Cannot de-vig without full 1X2 market prices
            return PricingResult(
                venue=venue,
                home_odds=home_odds,
                draw_odds=draw_odds,
                away_odds=away_odds,
                raw_implied_draw_prob=raw_draw_prob,
                devigged_draw_prob=raw_draw_prob,
                overround=None,
                is_vig_adjusted=False,
                fee_assumptions=fees,
                source="BOOKMAKER"
            )

        # Full 1X2 market normalization
        raw_home_prob = 1.0 / home_odds
        raw_away_prob = 1.0 / away_odds
        overround = raw_home_prob + raw_draw_prob + raw_away_prob

        if self.normalization_method == "multiplicative":
            devigged_draw_prob = raw_draw_prob / overround
        else:
            # Default fallback to multiplicative
            devigged_draw_prob = raw_draw_prob / overround

        return PricingResult(
            venue=venue,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
            raw_implied_draw_prob=raw_draw_prob,
            devigged_draw_prob=devigged_draw_prob,
            overround=overround,
            is_vig_adjusted=True,
            fee_assumptions=fees,
            source="BOOKMAKER"
        )

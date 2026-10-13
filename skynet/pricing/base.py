from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class PricingResult:
    venue: str
    home_odds: Optional[float]
    draw_odds: float
    away_odds: Optional[float]
    raw_implied_draw_prob: float
    devigged_draw_prob: float
    overround: Optional[float]
    is_vig_adjusted: bool
    fee_assumptions: Dict[str, Any]
    source: str = "BOOKMAKER"

class BasePricingAdapter(ABC):
    """Abstract interface for venue-agnostic pricing."""

    @abstractmethod
    def calculate_market_draw_probability(
        self,
        draw_odds: float,
        home_odds: Optional[float] = None,
        away_odds: Optional[float] = None,
        fee_assumptions: Optional[Dict[str, Any]] = None,
        venue: str = "BOOKMAKER"
    ) -> PricingResult:
        """Convert venue prices into cost-inclusive implied draw probability."""
        pass

    def calculate_net_edge(self, model_draw_prob: float, market_draw_prob: float) -> float:
        """Calculate net edge: Model Probability - Market Implied Probability."""
        return model_draw_prob - market_draw_prob

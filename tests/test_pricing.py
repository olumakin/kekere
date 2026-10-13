import pytest
from skynet.pricing.bookmaker import BookmakerPricingAdapter

def test_devigging_full_1x2_market():
    adapter = BookmakerPricingAdapter()
    # Pinnacle odds: Home 2.50, Draw 3.20, Away 3.10
    # Raw probs: 1/2.50 = 0.40, 1/3.20 = 0.3125, 1/3.10 = 0.32258 -> Overround = 1.03508
    res = adapter.calculate_market_draw_probability(
        draw_odds=3.20,
        home_odds=2.50,
        away_odds=3.10,
        venue="PINNACLE"
    )

    assert res.venue == "PINNACLE"
    assert res.is_vig_adjusted is True
    assert res.overround is not None
    assert round(res.overround, 4) == 1.0351
    # Devigged draw prob = 0.3125 / 1.03508 ~ 0.3019
    assert round(res.devigged_draw_prob, 4) == 0.3019
    assert round(res.raw_implied_draw_prob, 4) == 0.3125

def test_devigging_missing_1x2_unadjusted_flag():
    adapter = BookmakerPricingAdapter()
    # Only draw odds provided
    res = adapter.calculate_market_draw_probability(
        draw_odds=3.40,
        home_odds=None,
        away_odds=None,
        venue="EXCHANGE"
    )

    assert res.is_vig_adjusted is False
    assert res.overround is None
    assert round(res.devigged_draw_prob, 4) == round(1.0 / 3.40, 4)

def test_net_edge_calculation():
    adapter = BookmakerPricingAdapter()
    edge = adapter.calculate_net_edge(model_draw_prob=0.34, market_draw_prob=0.30)
    assert round(edge, 4) == 0.04

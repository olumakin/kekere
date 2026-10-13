import pytest
from skynet.staking.kelly import KellyCalculator
from skynet.staking.gate import GoNoGoGate

def test_kelly_calculator_staking_logic():
    kelly = KellyCalculator(
        kelly_fraction=0.25,
        max_stake_pct=0.02,
        daily_exposure_limit_pct=0.06,
        base_bankroll=10000.0
    )

    # Value bet: prob 0.35, odds 3.20 (implied ~0.3125), edge +0.05
    # b = 2.20, p = 0.35, q = 0.65 -> raw Kelly = (2.2*0.35 - 0.65)/2.2 = (0.77 - 0.65)/2.2 = 0.12/2.2 ~ 0.0545
    # Fractional Kelly (0.25x) = 0.0136 (1.36%) -> stake = £136.36
    rec = kelly.calculate_stake(
        draw_prob=0.35,
        draw_odds=3.20,
        edge=0.05,
        draw_prob_threshold=0.31,
        min_edge=0.03
    )

    assert rec.is_bet is True
    assert rec.stake_pct > 0.01
    assert rec.stake_pct <= 0.02 # within max stake cap
    assert rec.stake_amount > 100.0

def test_kelly_below_threshold_no_bet():
    kelly = KellyCalculator()
    rec = kelly.calculate_stake(
        draw_prob=0.28, # below 0.31
        draw_odds=3.20,
        edge=0.04,
        draw_prob_threshold=0.31,
        min_edge=0.03
    )
    assert rec.is_bet is False
    assert rec.stake_pct == 0.0
    assert "below threshold" in rec.reason

def test_circuit_breaker_loss_streak():
    kelly = KellyCalculator(circuit_breaker_loss_streak=3, base_bankroll=10000.0)
    for _ in range(3):
        kelly.record_bet_outcome(is_win=False, stake_amount=100.0, decimal_odds=3.20)

    # Circuit breaker active
    rec = kelly.calculate_stake(
        draw_prob=0.40,
        draw_odds=3.20,
        edge=0.08,
        draw_prob_threshold=0.31,
        min_edge=0.03
    )
    assert rec.formula_inputs["circuit_breaker_active"] is True

def test_go_no_go_gate_conditions():
    gate = GoNoGoGate(
        min_bets=150,
        min_roi=0.0,
        min_clv=0.0,
        cox_slope_min=0.8,
        cox_slope_max=1.2,
        cox_intercept_min=-0.05,
        cox_intercept_max=0.05
    )

    # Scenario 1: All criteria passed
    pass_res = gate.evaluate(
        bet_count=180,
        roi=0.045, # +4.5% ROI
        avg_clv=0.022, # +2.2% CLV
        cox_slope=1.02, # in [0.8, 1.2]
        cox_intercept=0.01 # in [-0.05, 0.05]
    )
    assert pass_res.is_passed is True
    assert pass_res.status == "ENABLED"
    assert len(pass_res.failures) == 0

    # Scenario 2: Bet count too low (< 150)
    fail_res = gate.evaluate(
        bet_count=120,
        roi=0.08,
        avg_clv=0.03,
        cox_slope=1.0,
        cox_intercept=0.0
    )
    assert fail_res.is_passed is False
    assert fail_res.status == "DISABLED"
    assert any("Insufficient bets" in f for f in fail_res.failures)

    # Scenario 3: Negative ROI
    fail_roi = gate.evaluate(
        bet_count=200,
        roi=-0.02,
        avg_clv=0.02,
        cox_slope=1.0,
        cox_intercept=0.0
    )
    assert fail_roi.is_passed is False
    assert fail_roi.status == "DISABLED"
    assert any("ROI not positive" in f for f in fail_roi.failures)

    # Scenario 4: Cox calibration slope out of range
    fail_cox = gate.evaluate(
        bet_count=200,
        roi=0.03,
        avg_clv=0.02,
        cox_slope=0.65, # < 0.8
        cox_intercept=0.0
    )
    assert fail_cox.is_passed is False
    assert fail_cox.status == "DISABLED"
    assert any("Cox slope" in f for f in fail_cox.failures)

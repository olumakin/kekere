from datetime import datetime, timedelta
import numpy as np
import pytest
from skynet.models.dixon_coles import DixonColesModel
from skynet.models.calibration import PlattCalibrator, CoxCalibrationEvaluator, compute_calibration_metrics
from skynet.models.second_stage import SecondStageEvaluator

def test_dixon_coles_low_score_tau():
    # Test Dixon-Coles tau adjustments
    tau_00 = DixonColesModel._tau(0, 0, lam=1.2, mu=1.0, rho=-0.05)
    # 1 - (1.2 * 1.0 * -0.05) = 1 + 0.06 = 1.06
    assert round(tau_00, 4) == 1.06

    tau_01 = DixonColesModel._tau(0, 1, lam=1.2, mu=1.0, rho=-0.05)
    # 1 + (1.2 * -0.05) = 0.94
    assert round(tau_01, 4) == 0.94

    tau_22 = DixonColesModel._tau(2, 2, lam=1.2, mu=1.0, rho=-0.05)
    assert tau_22 == 1.0

def test_dixon_coles_fit_and_predict():
    base_date = datetime(2021, 8, 1)
    teams = ["Norwich", "Watford", "Swansea", "Coventry"]
    matches = []
    
    # Generate mock season history
    for i in range(40):
        h = teams[i % len(teams)]
        a = teams[(i + 1) % len(teams)]
        matches.append({
            "home_team": h,
            "away_team": a,
            "fthg": (i % 3),
            "ftag": ((i + 1) % 3),
            "match_date": base_date + timedelta(days=i * 3)
        })

    model = DixonColesModel(xi=0.0019)
    model.fit(matches)

    assert model.is_fitted is True
    assert len(model.attacks) == 4
    assert len(model.defenses) == 4

    p_draw = model.predict_draw_prob("Norwich", "Watford")
    assert 0.15 <= p_draw <= 0.45

    probs_1x2 = model.predict_1x2_probs("Norwich", "Watford")
    total_prob = probs_1x2["home"] + probs_1x2["draw"] + probs_1x2["away"]
    assert round(total_prob, 4) == 1.0

def test_platt_and_cox_calibration():
    # Synthetic predictions and targets
    np.random.seed(42)
    raw_probs = np.random.uniform(0.20, 0.40, size=100)
    # Ground truth binary targets
    y_true = np.random.binomial(1, raw_probs)

    calibrator = PlattCalibrator()
    calibrator.fit(raw_probs, y_true)
    assert calibrator.is_fitted is True

    cal_probs = calibrator.predict_proba(raw_probs)
    assert len(cal_probs) == 100
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))

    # Cox calibration evaluation
    cox_res = CoxCalibrationEvaluator.evaluate(cal_probs, y_true)
    assert "slope" in cox_res
    assert "intercept" in cox_res

    # Calibration error metrics
    metrics = compute_calibration_metrics(cal_probs, y_true)
    assert metrics["brier_score"] >= 0.0
    assert metrics["log_loss"] >= 0.0

def test_second_stage_evaluator():
    np.random.seed(42)
    N = 100
    dc_probs = np.random.uniform(0.25, 0.35, size=N)
    features = np.random.randn(N, 4)
    X = np.column_stack([dc_probs, features])
    y_true = np.random.binomial(1, 0.28, size=N)
    devigged_closing_odds = np.full(N, 3.20)

    evaluator = SecondStageEvaluator()
    evaluator.fit(X[:80], y_true[:80])
    rf_probs = evaluator.predict_proba(X[80:])

    comparison = SecondStageEvaluator.evaluate_gate_comparison(
        dc_probs=dc_probs[80:],
        rf_probs=rf_probs,
        y_true=y_true[80:],
        devigged_closing_draw_odds=devigged_closing_odds[80:]
    )

    assert "should_adopt_second_stage" in comparison
    assert isinstance(comparison["should_adopt_second_stage"], (bool, np.bool_))

import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from skynet.db.schema import Base, Match, HistoricalOdds, ModelVersion, PredictionAudit
from skynet.models.dixon_coles import DixonColesModel
from skynet.models.calibration import PlattCalibrator
from skynet.models.registry import ModelRegistry
from skynet.staking.kelly import KellyCalculator

def test_audit_snapshot_and_reproducibility(tmp_path):
    # Setup test in-memory SQLite DB
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionTest = sessionmaker(bind=engine)
    session = SessionTest()

    # Create model artifact
    models_dir = tmp_path / "models"
    registry = ModelRegistry(models_dir=models_dir)

    dc = DixonColesModel(xi=0.0019)
    # Mock parameters
    dc.attacks = {"Burnley": 0.2, "Sheffield United": 0.1}
    dc.defenses = {"Burnley": -0.1, "Sheffield United": -0.05}
    dc.home_adv = 0.25
    dc.rho = -0.05
    dc.is_fitted = True

    calibrator = PlattCalibrator()

    cutoff_dt = datetime(2022, 1, 1, 12, 0)
    db_model = registry.save_and_register_model(
        session=session,
        model_obj={"dc": dc, "calibrator": calibrator},
        version_tag="v1.0.0",
        model_type="DIXON_COLES_TEST",
        training_cutoff=cutoff_dt,
        hyperparameters={"xi": 0.0019}
    )

    # Fixture
    match_dt = datetime(2022, 1, 2, 15, 0)
    match = Match(
        id="E1_2122_2022-01-02_Burnley_Sheffield_United",
        season="2122",
        division="E1",
        match_date=match_dt,
        home_team="Burnley",
        away_team="Sheffield United",
        fthg=1,
        ftag=1,
        ftr="D",
        status="FINISHED"
    )
    session.add(match)
    session.flush()

    # Generate inference
    raw_draw = dc.predict_draw_prob("Burnley", "Sheffield United")
    cal_draw = calibrator.calibrate_single(raw_draw)

    odds_snap = {
        "venue": "PINNACLE",
        "home_odds": 2.40,
        "draw_odds": 3.30,
        "away_odds": 3.20,
        "overround": 1.032,
        "is_vig_adjusted": True,
        "fee_assumptions": {"commission": 0.0}
    }
    market_draw_prob = (1.0 / 3.30) / 1.032
    net_edge = cal_draw - market_draw_prob

    kelly = KellyCalculator()
    stake_rec = kelly.calculate_stake(cal_draw, 3.30, net_edge)

    feature_snap = {
        "elo_home": 1560.0,
        "elo_away": 1540.0,
        "elo_diff": 20.0,
        "draw_rate_combined": 0.30,
        "rest_days_home": 7.0,
        "rest_days_away": 6.0,
        "rest_diff": 1.0
    }

    audit_entry = PredictionAudit(
        id=f"pred_{match.id}_{db_model.id}",
        match_id=match.id,
        model_version_id=db_model.id,
        prediction_timestamp=match_dt,
        feature_snapshot=feature_snap,
        odds_snapshot=odds_snap,
        model_draw_prob_raw=raw_draw,
        model_draw_prob_calibrated=cal_draw,
        market_implied_draw_prob=market_draw_prob,
        net_edge=net_edge,
        is_vig_adjusted=True,
        recommendation="BET" if stake_rec.is_bet else "NO_BET",
        suggested_stake_pct=stake_rec.stake_pct,
        suggested_stake_amount=stake_rec.stake_amount,
        staking_formula_inputs=stake_rec.formula_inputs,
        gating_status="DISABLED",
        final_result="D",
        closing_odds_snapshot={"venue": "PINNACLE", "draw_odds": 3.25},
        realized_clv=cal_draw - (1.0 / 3.25),
        realized_pnl=stake_rec.stake_amount * (3.30 - 1.0) if stake_rec.is_bet else 0.0
    )
    session.add(audit_entry)
    session.commit()

    # Verification: Test 100% Reproducibility
    loaded_audit = session.query(PredictionAudit).filter_by(id=audit_entry.id).first()
    assert loaded_audit is not None

    # Load serialized model from artifact path recorded in database
    model_record = session.query(ModelVersion).filter_by(id=loaded_audit.model_version_id).first()
    loaded_pack = registry.load_model(model_record.artifact_path)
    reproduced_dc = loaded_pack["dc"]
    reproduced_calibrator = loaded_pack["calibrator"]

    # Re-run inference using stored feature/model state
    reproduced_raw = reproduced_dc.predict_draw_prob(match.home_team, match.away_team)
    reproduced_cal = reproduced_calibrator.calibrate_single(reproduced_raw)

    assert round(reproduced_raw, 6) == round(loaded_audit.model_draw_prob_raw, 6)
    assert round(reproduced_cal, 6) == round(loaded_audit.model_draw_prob_calibrated, 6)

    # Re-calculate stake from snapshot
    reproduced_stake = kelly.calculate_stake(
        reproduced_cal,
        loaded_audit.odds_snapshot["draw_odds"],
        loaded_audit.net_edge
    )
    assert reproduced_stake.stake_pct == loaded_audit.suggested_stake_pct

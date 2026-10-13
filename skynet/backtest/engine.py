from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from skynet.config import config
from skynet.db.schema import Match, HistoricalOdds, ModelVersion, PredictionAudit
from skynet.pricing.bookmaker import BookmakerPricingAdapter
from skynet.features.engineering import FeatureEngine
from skynet.models.dixon_coles import DixonColesModel
from skynet.models.calibration import PlattCalibrator, CoxCalibrationEvaluator, compute_calibration_metrics
from skynet.models.registry import ModelRegistry
from skynet.staking.kelly import KellyCalculator
from skynet.staking.gate import GoNoGoGate, GateResult

class WalkForwardBacktest:
    """
    Institutional Walk-Forward Backtesting Engine for EFL Championship.
    Enforces strict temporal integrity, audit snapshot recording,
    and Go/No-Go Gate verification over historical seasons.
    """

    def __init__(
        self,
        session: Session,
        evaluation_seasons: Optional[List[str]] = None,
        burn_in_seasons: Optional[List[str]] = None,
        draw_prob_threshold: Optional[float] = None,
        min_edge: Optional[float] = None,
        kelly_fraction: Optional[float] = None,
        retrain_interval_days: int = 30
    ):
        self.session = session
        self.eval_seasons = evaluation_seasons or config.evaluation_seasons
        self.burn_in_seasons = burn_in_seasons or config.burn_in_seasons
        self.prob_threshold = draw_prob_threshold if draw_prob_threshold is not None else config.draw_prob_threshold
        self.min_edge = min_edge if min_edge is not None else config.min_edge
        self.kelly_fraction = kelly_fraction if kelly_fraction is not None else config.kelly_fraction
        self.retrain_interval_days = retrain_interval_days

        self.pricing_adapter = BookmakerPricingAdapter()
        self.model_registry = ModelRegistry()
        self.gate = GoNoGoGate(
            min_bets=config.min_bets,
            min_roi=config.min_roi,
            min_clv=config.min_clv,
            cox_slope_min=config.cox_slope_min,
            cox_slope_max=config.cox_slope_max,
            cox_intercept_min=config.cox_intercept_min,
            cox_intercept_max=config.cox_intercept_max
        )

    def run(self) -> Dict[str, Any]:
        """Executes full walk-forward backtest across configured historical seasons."""
        # 1. Fetch all finished matches in chronological order
        all_seasons = self.burn_in_seasons + self.eval_seasons
        matches = (
            self.session.query(Match)
            .filter(Match.season.in_(all_seasons), Match.status == "FINISHED")
            .order_by(Match.match_date.asc(), Match.id.asc())
            .all()
        )

        if not matches:
            raise ValueError("No matches found in database for backtest.")

        feature_engine = FeatureEngine()
        kelly_calc = KellyCalculator(kelly_fraction=self.kelly_fraction)

        # Separate burn-in matches from evaluation matches
        burn_in_data: List[Dict[str, Any]] = []
        eval_matches: List[Match] = []

        for m in matches:
            if m.season in self.burn_in_seasons:
                burn_in_data.append({
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "fthg": m.fthg,
                    "ftag": m.ftag,
                    "ftr": m.ftr,
                    "match_date": m.match_date,
                    "season": m.season,
                })
                # Burn in features and Elo
                feature_engine.calculate_pre_match_features(m.home_team, m.away_team, m.match_date, m.season)
                feature_engine.record_match_result(m.home_team, m.away_team, m.match_date, m.fthg, m.ftag, m.ftr, m.season)
            else:
                eval_matches.append(m)

        if not eval_matches:
            raise ValueError("No evaluation matches found.")

        # Walk-forward tracking
        training_history = list(burn_in_data)
        current_dc_model = DixonColesModel(xi=config.dixon_coles_xi)
        current_calibrator = PlattCalibrator()
        last_retrain_date: Optional[datetime] = None
        current_model_version_id: Optional[str] = None

        audit_records: List[Dict[str, Any]] = []
        season_stats: Dict[str, Dict[str, Any]] = {s: {"bets": 0, "wins": 0, "staked": 0.0, "returned": 0.0, "pnl": 0.0, "clv_sum": 0.0} for s in self.eval_seasons}
        
        # Track daily stakes for exposure limit
        current_day: Optional[datetime.date] = None
        daily_staked_pct = 0.0

        for match in eval_matches:
            match_date = match.match_date
            match_day = match_date.date()

            if current_day != match_day:
                current_day = match_day
                daily_staked_pct = 0.0
                kelly_calc.reset_daily_exposure()

            # Retrain model if interval elapsed or first evaluation match
            need_retrain = (
                last_retrain_date is None or 
                (match_date - last_retrain_date).days >= self.retrain_interval_days
            )
            if need_retrain and len(training_history) >= 40:
                current_dc_model = DixonColesModel(xi=config.dixon_coles_xi)
                current_dc_model.fit(training_history, current_date=match_date)
                last_retrain_date = match_date

                # Train Platt Calibrator on rolling historical training fold predictions
                sample_preds = []
                sample_y = []
                # Use last 300 matches for calibration fitting
                calib_subset = training_history[-300:]
                for h_m in calib_subset:
                    try:
                        p_raw = current_dc_model.predict_draw_prob(h_m["home_team"], h_m["away_team"])
                        sample_preds.append(p_raw)
                        sample_y.append(1 if h_m["ftr"] == "D" else 0)
                    except Exception:
                        pass
                if len(sample_preds) >= 30:
                    current_calibrator.fit(np.array(sample_preds), np.array(sample_y))

                # Register model version in DB
                db_model = self.model_registry.save_and_register_model(
                    session=self.session,
                    model_obj={"dc": current_dc_model, "calibrator": current_calibrator},
                    version_tag=f"v1.0_{match.season}",
                    model_type="DIXON_COLES_PLATT",
                    training_cutoff=match_date,
                    hyperparameters={"xi": config.dixon_coles_xi, "threshold": self.prob_threshold, "min_edge": self.min_edge}
                )
                current_model_version_id = db_model.id

            # 1. Feature snapshot strictly pre-match
            features = feature_engine.calculate_pre_match_features(
                match.home_team, match.away_team, match.match_date, match.season
            )

            # 2. Model Prediction
            raw_draw_prob = current_dc_model.predict_draw_prob(match.home_team, match.away_team)
            calibrated_draw_prob = current_calibrator.calibrate_single(raw_draw_prob)

            # 3. Market Odds Snapshot (Opening odds: Pinnacle preferred, fallback to Market Avg or Bet365)
            opening_odds = (
                self.session.query(HistoricalOdds)
                .filter(HistoricalOdds.match_id == match.id, HistoricalOdds.odds_type == "OPENING")
                .all()
            )
            selected_odds = None
            for venue_cand in ["PINNACLE", "BET365", "MARKET_AVG"]:
                for o in opening_odds:
                    if o.venue == venue_cand and o.draw_odds > 1.0:
                        selected_odds = o
                        break
                if selected_odds:
                    break

            if selected_odds is None:
                # If no opening odds stored, check closing odds or skip
                continue

            market_devigged_prob = selected_odds.devigged_draw_prob or (1.0 / selected_odds.draw_odds)
            net_edge = calibrated_draw_prob - market_devigged_prob

            odds_snapshot = {
                "venue": selected_odds.venue,
                "odds_type": selected_odds.odds_type,
                "home_odds": selected_odds.home_odds,
                "draw_odds": selected_odds.draw_odds,
                "away_odds": selected_odds.away_odds,
                "overround": selected_odds.overround,
                "is_vig_adjusted": selected_odds.overround is not None,
                "fee_assumptions": selected_odds.fee_assumptions,
            }

            # 4. Closing Odds Snapshot (for CLV calculation)
            closing_odds_rec = (
                self.session.query(HistoricalOdds)
                .filter(HistoricalOdds.match_id == match.id, HistoricalOdds.venue == "PINNACLE", HistoricalOdds.odds_type == "CLOSING")
                .first()
            )
            closing_devigged_prob = None
            closing_snapshot = None
            realized_clv = None
            if closing_odds_rec and closing_odds_rec.draw_odds > 1.0:
                closing_devigged_prob = closing_odds_rec.devigged_draw_prob or (1.0 / closing_odds_rec.draw_odds)
                realized_clv = calibrated_draw_prob - closing_devigged_prob
                closing_snapshot = {
                    "venue": closing_odds_rec.venue,
                    "draw_odds": closing_odds_rec.draw_odds,
                    "devigged_draw_prob": closing_devigged_prob
                }

            # 5. Kelly Staking Decision
            stake_rec = kelly_calc.calculate_stake(
                draw_prob=calibrated_draw_prob,
                draw_odds=selected_odds.draw_odds,
                edge=net_edge,
                draw_prob_threshold=self.prob_threshold,
                min_edge=self.min_edge,
                current_daily_staked=daily_staked_pct
            )

            # Update daily staked
            if stake_rec.is_bet:
                daily_staked_pct += stake_rec.stake_pct

            # 6. Realized PnL
            is_actual_draw = (match.ftr == "D")
            realized_pnl = 0.0
            if stake_rec.is_bet and stake_rec.stake_amount > 0:
                if is_actual_draw:
                    realized_pnl = stake_rec.stake_amount * (selected_odds.draw_odds - 1.0)
                else:
                    realized_pnl = -stake_rec.stake_amount
                kelly_calc.record_bet_outcome(is_actual_draw, stake_rec.stake_amount, selected_odds.draw_odds)

                # Track season stats
                s_stat = season_stats[match.season]
                s_stat["bets"] += 1
                if is_actual_draw:
                    s_stat["wins"] += 1
                s_stat["staked"] += stake_rec.stake_amount
                s_stat["returned"] += (stake_rec.stake_amount + realized_pnl)
                s_stat["pnl"] += realized_pnl
                if realized_clv is not None:
                    s_stat["clv_sum"] += realized_clv

            # 7. Audit Logging (Strictly logs every prediction including 'NO_BET')
            pred_id = f"pred_{match.id}_{current_model_version_id or 'init'}"
            audit_obj = {
                "id": pred_id,
                "match_id": match.id,
                "model_version_id": current_model_version_id or "DC_baseline",
                "prediction_timestamp": match.match_date,
                "feature_snapshot": features,
                "odds_snapshot": odds_snapshot,
                "model_draw_prob_raw": round(raw_draw_prob, 4),
                "model_draw_prob_calibrated": round(calibrated_draw_prob, 4),
                "market_implied_draw_prob": round(market_devigged_prob, 4),
                "net_edge": round(net_edge, 4),
                "is_vig_adjusted": odds_snapshot["is_vig_adjusted"],
                "recommendation": "BET" if stake_rec.is_bet else "NO_BET",
                "suggested_stake_pct": stake_rec.stake_pct,
                "suggested_stake_amount": stake_rec.stake_amount,
                "staking_formula_inputs": stake_rec.formula_inputs,
                "gating_status": "DISABLED", # Checked globally at gate evaluation
                "final_result": match.ftr,
                "closing_odds_snapshot": closing_snapshot,
                "realized_clv": round(realized_clv, 4) if realized_clv is not None else None,
                "realized_pnl": round(realized_pnl, 2) if stake_rec.is_bet else 0.0,
            }
            audit_records.append(audit_obj)

            # Persist or update in DB
            db_audit = self.session.query(PredictionAudit).filter_by(id=pred_id).first()
            if not db_audit:
                db_audit = PredictionAudit(**audit_obj)
                self.session.add(db_audit)
            else:
                for k, v in audit_obj.items():
                    setattr(db_audit, k, v)

            # 8. Advance chronological state post-match
            feature_engine.record_match_result(
                match.home_team, match.away_team, match.match_date, match.fthg, match.ftag, match.ftr, match.season
            )
            training_history.append({
                "home_team": match.home_team,
                "away_team": match.away_team,
                "fthg": match.fthg,
                "ftag": match.ftag,
                "ftr": match.ftr,
                "match_date": match.match_date,
                "season": match.season,
            })

        self.session.flush()

        # 9. Aggregate Validation Metrics
        df_audit = pd.DataFrame(audit_records)
        df_bets = df_audit[df_audit["recommendation"] == "BET"]

        total_bets = len(df_bets)
        total_staked = float(df_bets["suggested_stake_amount"].sum()) if total_bets > 0 else 0.0
        total_pnl = float(df_bets["realized_pnl"].sum()) if total_bets > 0 else 0.0
        roi = (total_pnl / total_staked) if total_staked > 0 else 0.0

        # Win rate
        winning_bets = int((df_bets["final_result"] == "D").sum()) if total_bets > 0 else 0
        win_rate = (winning_bets / total_bets) if total_bets > 0 else 0.0

        # Closing Line Value (CLV)
        valid_clv = df_bets["realized_clv"].dropna()
        avg_clv = float(valid_clv.mean()) if len(valid_clv) > 0 else 0.0

        # Calibration metrics across all predictions
        all_cal_probs = df_audit["model_draw_prob_calibrated"].to_numpy()
        all_y = (df_audit["final_result"] == "D").astype(int).to_numpy()
        cal_metrics = compute_calibration_metrics(all_cal_probs, all_y)

        # Cox Calibration slope and intercept
        cox_res = CoxCalibrationEvaluator.evaluate(all_cal_probs, all_y)

        # Max Drawdown
        max_dd = 0.0
        if total_bets > 0:
            cum_pnl = df_bets["realized_pnl"].cumsum()
            running_max = np.maximum.accumulate(cum_pnl)
            drawdowns = running_max - cum_pnl
            max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

        # Evaluate Go/No-Go Gate
        gate_res: GateResult = self.gate.evaluate(
            bet_count=total_bets,
            roi=roi,
            avg_clv=avg_clv,
            cox_slope=cox_res["slope"],
            cox_intercept=cox_res["intercept"]
        )

        # Update gating status in audit records
        for rec in audit_records:
            rec["gating_status"] = gate_res.status
        self.session.query(PredictionAudit).update({"gating_status": gate_res.status})
        self.session.commit()

        # Season-by-season breakdown
        seasonal_breakdown = {}
        for s, st in season_stats.items():
            s_bets = st["bets"]
            s_staked = st["staked"]
            s_pnl = st["pnl"]
            s_roi = (s_pnl / s_staked) if s_staked > 0 else 0.0
            seasonal_breakdown[s] = {
                "bets": s_bets,
                "wins": st["wins"],
                "win_rate": round(st["wins"] / s_bets, 4) if s_bets > 0 else 0.0,
                "staked": round(s_staked, 2),
                "pnl": round(s_pnl, 2),
                "roi": round(s_roi, 4),
                "avg_clv": round(st["clv_sum"] / s_bets, 4) if s_bets > 0 else 0.0
            }

        return {
            "total_matches_evaluated": len(df_audit),
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "win_rate": round(win_rate, 4),
            "total_staked": round(total_staked, 2),
            "net_pnl": round(total_pnl, 2),
            "roi": round(roi, 4),
            "max_drawdown": round(max_dd, 2),
            "avg_clv": round(avg_clv, 4),
            "brier_score": cal_metrics["brier_score"],
            "log_loss": cal_metrics["log_loss"],
            "ece": cal_metrics["ece"],
            "cox_calibration": cox_res,
            "gate": {
                "is_passed": gate_res.is_passed,
                "status": gate_res.status,
                "failures": gate_res.failures,
                "details": gate_res.details,
            },
            "seasonal_breakdown": seasonal_breakdown,
        }

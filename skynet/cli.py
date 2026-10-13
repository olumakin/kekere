import sys
import argparse
import json
from datetime import datetime
from typing import List

from skynet.config import config
from skynet.db.session import init_db, get_db_session
from skynet.db.schema import Match, HistoricalOdds, PredictionAudit, ModelVersion
from skynet.ingestion.football_data import FootballDataIngestor
from skynet.backtest.engine import WalkForwardBacktest
from skynet.pricing.bookmaker import BookmakerPricingAdapter
from skynet.staking.kelly import KellyCalculator

def cmd_ingest(args):
    print("=" * 60)
    print(" SKYNET CHAMPIONSHIP: DATA INGESTION PIPELINE")
    print("=" * 60)
    init_db()
    ingestor = FootballDataIngestor()

    seasons = args.seasons.split(",") if args.seasons else (config.burn_in_seasons + config.evaluation_seasons)
    total_added = 0

    with get_db_session() as session:
        for s in seasons:
            s = s.strip()
            print(f"[*] Ingesting season '{s}' from football-data.co.uk...")
            try:
                added = ingestor.sync_season_to_db(session, s, force_download=args.force)
                total_added += added
                print(f"    -> Successfully ingested season {s} (new matches added: {added})")
            except Exception as e:
                print(f"    [!] Error syncing season {s}: {e}")

    print(f"\n[+] Ingestion complete. Total new matches stored: {total_added}")

def cmd_backtest(args):
    print("=" * 60)
    print(" SKYNET CHAMPIONSHIP: WALK-FORWARD BACKTEST ENGINE")
    print("=" * 60)
    init_db()

    with get_db_session() as session:
        # Check match count
        match_count = session.query(Match).count()
        if match_count == 0:
            print("[!] Database contains 0 matches. Please run `ingest` first.")
            return

        print(f"[*] Loaded {match_count} total matches from database.")
        print(f"[*] Burn-in seasons: {config.burn_in_seasons}")
        print(f"[*] Evaluation seasons (5 completed): {config.evaluation_seasons}")
        print(f"[*] Draw probability threshold: {args.threshold:.1%}")
        print(f"[*] Minimum edge threshold: {args.edge:.1%}")
        print(f"[*] Kelly multiplier: {args.kelly}x")
        print("[*] Running walk-forward model training & out-of-sample evaluation...\n")

        engine = WalkForwardBacktest(
            session=session,
            draw_prob_threshold=args.threshold,
            min_edge=args.edge,
            kelly_fraction=args.kelly,
            retrain_interval_days=args.retrain_days
        )

        results = engine.run()

        print("-" * 60)
        print(" AGGREGATE PERFORMANCE METRICS")
        print("-" * 60)
        print(f"Total Matches Evaluated:   {results['total_matches_evaluated']}")
        print(f"Bets Placed (Value Bets):  {results['total_bets']}")
        print(f"Winning Bets (Draws):      {results['winning_bets']} ({results['win_rate']:.1%})")
        print(f"Total Staked:              £{results['total_staked']:,.2f}")
        print(f"Net Profit / Loss:         £{results['net_pnl']:,.2f}")
        print(f"Return on Investment (ROI):{results['roi']:.2%}")
        print(f"Average Closing Line Value:{results['avg_clv']:.2%}")
        print(f"Max Drawdown:              £{results['max_drawdown']:,.2f}")
        print(f"Brier Score:               {results['brier_score']:.4f}")
        print(f"Log Loss:                  {results['log_loss']:.4f}")
        print(f"Expected Calib. Error(ECE):{results['ece']:.4f}")
        print(f"Cox Calibration Slope:     {results['cox_calibration']['slope']:.4f}")
        print(f"Cox Calibration Intercept: {results['cox_calibration']['intercept']:.4f}")

        print("\n" + "-" * 60)
        print(" SEASONAL BREAKDOWN")
        print("-" * 60)
        print(f"{'Season':<8} | {'Bets':<6} | {'Wins':<6} | {'Win Rate':<10} | {'Staked':<12} | {'PnL':<12} | {'ROI':<8} | {'Avg CLV':<8}")
        print("-" * 75)
        for s, st in results["seasonal_breakdown"].items():
            print(f"{s:<8} | {st['bets']:<6} | {st['wins']:<6} | {st['win_rate']:<10.1%} | £{st['staked']:<11,.2f} | £{st['pnl']:<11,.2f} | {st['roi']:<8.2%} | {st['avg_clv']:<8.2%}")

        print("\n" + "=" * 60)
        print(" GO / NO-GO GATE EVALUATION (§2 & §7)")
        print("=" * 60)
        gate = results["gate"]
        status_color = "[PASSED - STAKING ENABLED]" if gate["is_passed"] else "[FAILED - STAKING DISABLED]"
        print(f"GATE STATUS: {status_color}")
        print(f"  1. Minimum Bet Count (>=150):           {'PASS' if gate['details']['criteria_met']['bets'] else 'FAIL'} ({gate['details']['bet_count']})")
        print(f"  2. Positive ROI (>0.0%):                {'PASS' if gate['details']['criteria_met']['roi'] else 'FAIL'} ({gate['details']['roi']:.2%})")
        print(f"  3. Positive CLV (>0.0%):                {'PASS' if gate['details']['criteria_met']['clv'] else 'FAIL'} ({gate['details']['avg_clv']:.2%})")
        print(f"  4. Cox Slope in [0.8, 1.2]:             {'PASS' if gate['details']['criteria_met']['cox_slope'] else 'FAIL'} ({gate['details']['cox_slope']:.4f})")
        print(f"  5. Cox Intercept in [-0.05, 0.05]:      {'PASS' if gate['details']['criteria_met']['cox_intercept'] else 'FAIL'} ({gate['details']['cox_intercept']:.4f})")

        if gate["failures"]:
            print("\nGate Blockers:")
            for f in gate["failures"]:
                print(f"  - {f}")
        print("=" * 60)

def cmd_predict(args):
    print("=" * 60)
    print(" SKYNET CHAMPIONSHIP: FIXTURE PREDICTOR")
    print("=" * 60)
    init_db()

    with get_db_session() as session:
        # Load latest model
        latest_model_ver = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()
        if not latest_model_ver:
            print("[!] No trained model version found. Please run backtest first.")
            return

        from skynet.models.registry import ModelRegistry
        reg = ModelRegistry()
        model_pack = reg.load_model(latest_model_ver.artifact_path)
        dc_model = model_pack["dc"]
        calibrator = model_pack["calibrator"]

        # Calculate prediction
        raw_draw = dc_model.predict_draw_prob(args.home, args.away)
        cal_draw = calibrator.calibrate_single(raw_draw)

        # Devig market odds
        pricing = BookmakerPricingAdapter()
        pr = pricing.calculate_market_draw_probability(
            draw_odds=args.draw_odds,
            home_odds=args.home_odds,
            away_odds=args.away_odds,
            venue=args.venue
        )

        edge = cal_draw - pr.devigged_draw_prob

        # Kelly staking
        kelly = KellyCalculator(kelly_fraction=config.kelly_fraction)
        stake = kelly.calculate_stake(
            draw_prob=cal_draw,
            draw_odds=args.draw_odds,
            edge=edge,
            draw_prob_threshold=config.draw_prob_threshold,
            min_edge=config.min_edge
        )

        print(f"Fixture:               {args.home} vs {args.away}")
        print(f"Venue / Price:         {args.venue} @ {args.draw_odds:.2f}")
        print(f"Raw Draw Prob:         {raw_draw:.1%}")
        print(f"Calibrated Draw Prob:  {cal_draw:.1%}")
        print(f"Market Implied Draw:   {pr.devigged_draw_prob:.1%} (Vig-adjusted: {pr.is_vig_adjusted})")
        print(f"Calculated Net Edge:   {edge:+.2%}")
        print(f"Recommendation:        {'BET' if stake.is_bet else 'NO_BET'} ({stake.reason})")
        print(f"Suggested Stake:       {stake.stake_pct:.2%} (£{stake.stake_amount:,.2f})")
        print(f"Staking Gate Status:   {session.query(PredictionAudit).first().gating_status if session.query(PredictionAudit).first() else 'DISABLED'}")

def cmd_audit(args):
    print("=" * 60)
    print(" SKYNET CHAMPIONSHIP: AUDIT LOG INSPECTOR")
    print("=" * 60)
    init_db()

    with get_db_session() as session:
        records = (
            session.query(PredictionAudit)
            .order_by(PredictionAudit.prediction_timestamp.desc())
            .limit(args.limit)
            .all()
        )
        if not records:
            print("[!] No prediction audit logs found.")
            return

        print(f"Showing last {len(records)} audit records:")
        print(f"{'Timestamp':<16} | {'Match ID':<35} | {'Model Prob':<10} | {'Edge':<8} | {'Rec':<7} | {'Stake%':<7} | {'Result':<6} | {'PnL'}")
        print("-" * 105)
        for r in records:
            ts = r.prediction_timestamp.strftime("%Y-%m-%d %H:%M")
            mid = (r.match_id[:32] + "...") if len(r.match_id) > 35 else r.match_id
            print(f"{ts:<16} | {mid:<35} | {r.model_draw_prob_calibrated:<10.1%} | {r.net_edge:<+8.1%} | {r.recommendation:<7} | {r.suggested_stake_pct:<7.1%} | {r.final_result or '-':<6} | £{r.realized_pnl or 0.0:,.2f}")

def main():
    parser = argparse.ArgumentParser(description="Skynet Championship Draw Predictor CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest EFL Championship datasets")
    p_ingest.add_argument("--seasons", type=str, default=None, help="Comma-separated season codes e.g. 1920,2021,2122,2223,2324,2425")
    p_ingest.add_argument("--force", action="store_true", help="Force re-download of CSV files")
    p_ingest.set_defaults(func=cmd_ingest)

    # Backtest
    p_bt = subparsers.add_parser("backtest", help="Run walk-forward backtest")
    p_bt.add_argument("--threshold", type=float, default=config.draw_prob_threshold, help="Min draw prob threshold (e.g. 0.31)")
    p_bt.add_argument("--edge", type=float, default=config.min_edge, help="Min net edge threshold (e.g. 0.03)")
    p_bt.add_argument("--kelly", type=float, default=config.kelly_fraction, help="Kelly fraction multiplier (e.g. 0.25)")
    p_bt.add_argument("--retrain-days", type=int, default=30, help="Model retrain frequency in days")
    p_bt.set_defaults(func=cmd_backtest)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict fixture draw probability and value")
    p_pred.add_argument("--home", type=str, required=True, help="Home team name")
    p_pred.add_argument("--away", type=str, required=True, help="Away team name")
    p_pred.add_argument("--draw-odds", type=float, required=True, help="Market draw odds (decimal)")
    p_pred.add_argument("--home-odds", type=float, default=None, help="Market home odds (optional)")
    p_pred.add_argument("--away-odds", type=float, default=None, help="Market away odds (optional)")
    p_pred.add_argument("--venue", type=str, default="PINNACLE", help="Venue / Bookmaker name")
    p_pred.set_defaults(func=cmd_predict)

    # Audit
    p_audit = subparsers.add_parser("audit", help="Inspect prediction audit logs")
    p_audit.add_argument("--limit", type=int, default=20, help="Number of records to display")
    p_audit.set_defaults(func=cmd_audit)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

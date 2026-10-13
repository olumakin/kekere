import os
import json
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Skynet Championship | EFL Draw Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1f77b4;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .gate-passed {
        background-color: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #c3e6cb;
        font-weight: 600;
    }
    .gate-failed {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 6px;
        border: 1px solid #f5c6cb;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Imports from skynet package
from skynet.config import config
from skynet.db.session import init_db, get_db_session
from skynet.db.schema import Match, HistoricalOdds, PredictionAudit, ModelVersion
from skynet.pricing.bookmaker import BookmakerPricingAdapter
from skynet.staking.kelly import KellyCalculator
from skynet.staking.gate import GoNoGoGate
from skynet.models.dixon_coles import DixonColesModel
from skynet.models.calibration import PlattCalibrator, CoxCalibrationEvaluator, compute_calibration_metrics
from skynet.backtest.engine import WalkForwardBacktest
from skynet.ingestion.football_data import FootballDataIngestor

# Ensure database is initialized
init_db()

@st.cache_data(show_spinner=False)
def load_audit_data():
    """Loads prediction audit records from database into a pandas DataFrame."""
    with get_db_session() as session:
        records = (
            session.query(PredictionAudit)
            .order_by(PredictionAudit.prediction_timestamp.asc())
            .all()
        )
        if not records:
            return pd.DataFrame()

        data = []
        for r in records:
            odds_snap = r.odds_snapshot or {}
            feat_snap = r.feature_snapshot or {}
            closing_snap = r.closing_odds_snapshot or {}
            
            data.append({
                "id": r.id,
                "match_id": r.match_id,
                "timestamp": r.prediction_timestamp,
                "season": r.match.season if r.match else "Unknown",
                "home_team": r.match.home_team if r.match else "Unknown",
                "away_team": r.match.away_team if r.match else "Unknown",
                "model_draw_prob_raw": r.model_draw_prob_raw,
                "model_draw_prob_calibrated": r.model_draw_prob_calibrated,
                "market_implied_draw_prob": r.market_implied_draw_prob,
                "draw_odds": odds_snap.get("draw_odds", np.nan),
                "home_odds": odds_snap.get("home_odds", np.nan),
                "away_odds": odds_snap.get("away_odds", np.nan),
                "venue": odds_snap.get("venue", "PINNACLE"),
                "net_edge": r.net_edge,
                "recommendation": r.recommendation,
                "suggested_stake_pct": r.suggested_stake_pct,
                "suggested_stake_amount": r.suggested_stake_amount,
                "final_result": r.final_result,
                "is_draw": (r.final_result == "D") if r.final_result else None,
                "realized_clv": r.realized_clv,
                "realized_pnl": r.realized_pnl,
                "gating_status": r.gating_status,
                "elo_diff": feat_snap.get("elo_diff", 0.0),
                "draw_rate_combined": feat_snap.get("draw_rate_combined", 0.28),
            })

        df = pd.DataFrame(data)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["month"] = df["timestamp"].dt.to_period("M").astype(str)
        return df

def bootstrap_system_if_empty():
    """Checks if DB is empty and runs ingestion + backtest."""
    with get_db_session() as session:
        match_count = session.query(Match).count()
        audit_count = session.query(PredictionAudit).count()

    if match_count == 0:
        st.warning("⚠️ Database is uninitialized. Auto-syncing EFL Championship datasets...")
        with st.spinner("Downloading historical seasons (2019/20 - 2025/26)..."):
            ingestor = FootballDataIngestor()
            with get_db_session() as session:
                for s in config.burn_in_seasons + config.evaluation_seasons:
                    try:
                        ingestor.sync_season_to_db(session, s)
                    except Exception as e:
                        st.error(f"Error syncing season {s}: {e}")
        st.success("✅ Historical datasets ingested.")

    if audit_count == 0:
        st.info("⚙️ Running initial walk-forward backtest to generate audit snapshots...")
        with st.spinner("Executing walk-forward model training & calibration..."):
            with get_db_session() as session:
                engine = WalkForwardBacktest(session=session)
                engine.run()
        st.success("✅ Backtest complete. Audit records generated.")
        load_audit_data.clear()

# Auto bootstrap if required
bootstrap_system_if_empty()
df_audit = load_audit_data()

# Sidebar Navigation & Settings
st.sidebar.title("⚽ Skynet Championship")
st.sidebar.caption("EFL Championship Draw Predictor v4.6")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "📊 Executive Overview",
        "📈 Performance & PnL",
        "🎯 Model Calibration & Cox",
        "📋 Audit Trail & Predictions",
        "🔮 Live Match Predictor"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")
st.sidebar.markdown(f"**Min Draw Prob:** `{config.draw_prob_threshold:.1%}`")
st.sidebar.markdown(f"**Min Net Edge:** `{config.min_edge:.1%}`")
st.sidebar.markdown(f"**Kelly Fraction:** `{config.kelly_fraction}x`")
st.sidebar.markdown(f"**Max Stake Cap:** `{config.max_stake_pct:.1%}`")
st.sidebar.markdown(f"**Daily Exposure Limit:** `{config.daily_exposure_limit_pct:.1%}`")

# Main Header
st.title("Skynet Championship — Predictive Analytics & Value Engine")

if df_audit.empty:
    st.info("No audit records found. Please trigger data ingestion and backtest via the CLI or bootstrap above.")
    st.stop()

df_bets = df_audit[df_audit["recommendation"] == "BET"].copy()

# ----------------------------------------------------
# 1. EXECUTIVE OVERVIEW
# ----------------------------------------------------
if nav_choice == "📊 Executive Overview":
    st.header("Executive Summary & Risk Governance")

    # Metrics calculation
    total_matches = len(df_audit)
    total_bets = len(df_bets)
    winning_bets = int((df_bets["final_result"] == "D").sum()) if total_bets > 0 else 0
    win_rate = (winning_bets / total_bets) if total_bets > 0 else 0.0
    total_staked = df_bets["suggested_stake_amount"].sum() if total_bets > 0 else 0.0
    net_pnl = df_bets["realized_pnl"].sum() if total_bets > 0 else 0.0
    roi = (net_pnl / total_staked) if total_staked > 0 else 0.0
    avg_clv = df_bets["realized_clv"].dropna().mean() if total_bets > 0 else 0.0

    # Max Drawdown
    if total_bets > 0:
        cum_pnl = df_bets["realized_pnl"].cumsum()
        running_max = np.maximum.accumulate(cum_pnl)
        drawdowns = running_max - cum_pnl
        max_dd = float(np.max(drawdowns))
    else:
        max_dd = 0.0

    # Cox calibration
    all_cal_probs = df_audit["model_draw_prob_calibrated"].to_numpy()
    all_y = (df_audit["final_result"] == "D").astype(int).to_numpy()
    cox_res = CoxCalibrationEvaluator.evaluate(all_cal_probs, all_y)
    cal_metrics = compute_calibration_metrics(all_cal_probs, all_y)

    # Gate Evaluation
    gate = GoNoGoGate(
        min_bets=config.min_bets,
        min_roi=config.min_roi,
        min_clv=config.min_clv,
        cox_slope_min=config.cox_slope_min,
        cox_slope_max=config.cox_slope_max,
        cox_intercept_min=config.cox_intercept_min,
        cox_intercept_max=config.cox_intercept_max
    )
    gate_res = gate.evaluate(
        bet_count=total_bets,
        roi=roi,
        avg_clv=avg_clv,
        cox_slope=cox_res["slope"],
        cox_intercept=cox_res["intercept"]
    )

    # Gate Banner
    if gate_res.is_passed:
        st.markdown(
            '<div class="gate-passed">🟢 <strong>GO / NO-GO GATE PASSED: STAKING ENABLED</strong> — Model meets all institutional risk criteria.</div>',
            unsafe_allow_html=True
        )
    else:
        reasons = ", ".join(gate_res.failures)
        st.markdown(
            f'<div class="gate-failed">🔴 <strong>GO / NO-GO GATE: STAKING STRICTLY DISABLED</strong> — Suggested stakes are logged for paper-trading only.<br>Blockers: {reasons}</div>',
            unsafe_allow_html=True
        )

    st.write("")

    # KPI Columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches Evaluated", f"{total_matches:,}")
        st.metric("Value Bets Flagged", f"{total_bets:,}", f"{total_bets/total_matches:.1%} of matches")
    with col2:
        st.metric("Win Rate", f"{win_rate:.1%}", f"{winning_bets}/{total_bets} draws")
        st.metric("Total Staked", f"£{total_staked:,.2f}")
    with col3:
        pnl_delta = f"{roi:.2%} ROI"
        st.metric("Net Realized P&L", f"£{net_pnl:,.2f}", pnl_delta)
        st.metric("Max Drawdown", f"£{max_dd:,.2f}")
    with col4:
        st.metric("Average CLV", f"{avg_clv:+.2%}", "vs. Pinnacle Closing")
        st.metric("Brier Score", f"{cal_metrics['brier_score']:.4f}")

    st.markdown("---")
    st.subheader("Section 2 & 7 Institutional Risk Gate Breakdown")
    
    crit_cols = st.columns(5)
    with crit_cols[0]:
        st.metric("1. Min Bets (≥150)", f"{total_bets}", "✅ PASS" if total_bets >= 150 else "❌ FAIL")
    with crit_cols[1]:
        st.metric("2. Positive ROI (>0%)", f"{roi:.2%}", "✅ PASS" if roi > 0 else "❌ FAIL")
    with crit_cols[2]:
        st.metric("3. Positive CLV (>0%)", f"{avg_clv:.2%}", "✅ PASS" if avg_clv > 0 else "❌ FAIL")
    with crit_cols[3]:
        st.metric("4. Cox Slope [0.8, 1.2]", f"{cox_res['slope']:.3f}", "✅ PASS" if 0.8 <= cox_res['slope'] <= 1.2 else "❌ FAIL")
    with crit_cols[4]:
        st.metric("5. Cox Intercept [±0.05]", f"{cox_res['intercept']:.3f}", "✅ PASS" if -0.05 <= cox_res['intercept'] <= 0.05 else "❌ FAIL")

# ----------------------------------------------------
# 2. PERFORMANCE & PNL
# ----------------------------------------------------
elif nav_choice == "📈 Performance & PnL":
    st.header("Financial Performance & Historical Reporting")

    if df_bets.empty:
        st.info("No value bets placed in current backtest dataset.")
    else:
        df_bets["cum_pnl"] = df_bets["realized_pnl"].cumsum()
        df_bets["cum_staked"] = df_bets["suggested_stake_amount"].cumsum()
        df_bets["cum_roi"] = (df_bets["cum_pnl"] / df_bets["cum_staked"]) * 100.0

        # PnL Line Chart
        fig_pnl = px.line(
            df_bets,
            x="timestamp",
            y="cum_pnl",
            title="Cumulative Realized Profit & Loss (£)",
            labels={"cum_pnl": "Cumulative P&L (£)", "timestamp": "Match Date"},
            markers=True,
            line_shape="linear"
        )
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_pnl.update_layout(hovermode="x unified", template="plotly_white")
        st.plotly_chart(fig_pnl, use_container_width=True)

        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Seasonal Performance Breakdown")
            season_group = df_bets.groupby("season").agg(
                Bets=("id", "count"),
                Wins=("is_draw", "sum"),
                Staked=("suggested_stake_amount", "sum"),
                PnL=("realized_pnl", "sum"),
                Avg_CLV=("realized_clv", "mean")
            ).reset_index()
            season_group["Win_Rate"] = (season_group["Wins"] / season_group["Bets"]).map("{:.1%}".format)
            season_group["ROI"] = (season_group["PnL"] / season_group["Staked"]).map("{:.2%}".format)
            season_group["Avg_CLV"] = season_group["Avg_CLV"].map("{:+.2%}".format)
            season_group["Staked"] = season_group["Staked"].map("£{:,.2f}".format)
            season_group["PnL"] = season_group["PnL"].map("£{:,.2f}".format)

            st.dataframe(season_group, use_container_width=True)

        with col_b:
            st.subheader("Monthly P&L Distribution")
            month_group = df_bets.groupby("month").agg(
                Bets=("id", "count"),
                PnL=("realized_pnl", "sum")
            ).reset_index()

            fig_month = px.bar(
                month_group,
                x="month",
                y="PnL",
                title="Monthly Realized Net P&L (£)",
                color="PnL",
                color_continuous_scale=["#d9534f", "#5cb85c"]
            )
            fig_month.update_layout(template="plotly_white")
            st.plotly_chart(fig_month, use_container_width=True)

# ----------------------------------------------------
# 3. MODEL CALIBRATION & COX MONITORING
# ----------------------------------------------------
elif nav_choice == "🎯 Model Calibration & Cox":
    st.header("Probability Calibration & Monitoring Engine")

    all_cal_probs = df_audit["model_draw_prob_calibrated"].to_numpy()
    all_raw_probs = df_audit["model_draw_prob_raw"].to_numpy()
    all_y = (df_audit["final_result"] == "D").astype(int).to_numpy()

    cox_res = CoxCalibrationEvaluator.evaluate(all_cal_probs, all_y)
    cal_metrics = compute_calibration_metrics(all_cal_probs, all_y)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cox Calibration Slope", f"{cox_res['slope']:.4f}", "Target: 1.0 (0.8 - 1.2)")
    c2.metric("Cox Calibration Intercept", f"{cox_res['intercept']:.4f}", "Target: 0.0 (±0.05)")
    c3.metric("Brier Score", f"{cal_metrics['brier_score']:.4f}")
    c4.metric("Expected Calib. Error (ECE)", f"{cal_metrics['ece']:.4f}")

    st.write("")

    # Reliability Diagram (Binned Observed vs Predicted)
    n_bins = 8
    bins = np.linspace(0.15, 0.40, n_bins + 1)
    df_eval = df_audit.copy()
    df_eval["bin"] = pd.cut(df_eval["model_draw_prob_calibrated"], bins=bins)
    
    bin_stats = df_eval.groupby("bin", observed=False).agg(
        pred_mean=("model_draw_prob_calibrated", "mean"),
        obs_mean=("is_draw", "mean"),
        count=("id", "count")
    ).dropna().reset_index()

    fig_cal = go.Figure()
    # Reference perfect calibration line
    fig_cal.add_trace(go.Scatter(
        x=[0.15, 0.40],
        y=[0.15, 0.40],
        mode="lines",
        line=dict(dash="dash", color="gray"),
        name="Perfect Calibration"
    ))
    # Observed vs Predicted points
    fig_cal.add_trace(go.Scatter(
        x=bin_stats["pred_mean"],
        y=bin_stats["obs_mean"],
        mode="lines+markers",
        marker=dict(size=10, color="#1f77b4"),
        name="Calibrated Model"
    ))

    fig_cal.update_layout(
        title="Reliability Diagram: Predicted vs. Observed Draw Frequency",
        xaxis_title="Mean Predicted Draw Probability",
        yaxis_title="Empirical Draw Frequency",
        template="plotly_white"
    )
    st.plotly_chart(fig_cal, use_container_width=True)

# ----------------------------------------------------
# 4. AUDIT TRAIL & PREDICTIONS EXPLORER
# ----------------------------------------------------
elif nav_choice == "📋 Audit Trail & Predictions":
    st.header("Predictions Audit Trail Explorer")
    st.caption("Permanent, immutable log of every prediction (including No-Bet outcomes) per Section 5.")

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_rec = st.selectbox("Recommendation Filter", ["ALL", "BET", "NO_BET"])
    with col_f2:
        seasons_avail = ["ALL"] + sorted(list(df_audit["season"].unique()))
        selected_season = st.selectbox("Season Filter", seasons_avail)
    with col_f3:
        search_team = st.text_input("Search Team", placeholder="e.g. Leeds, Sunderland")

    filtered_df = df_audit.copy()
    if selected_rec != "ALL":
        filtered_df = filtered_df[filtered_df["recommendation"] == selected_rec]
    if selected_season != "ALL":
        filtered_df = filtered_df[filtered_df["season"] == selected_season]
    if search_team:
        t = search_team.strip().lower()
        filtered_df = filtered_df[
            filtered_df["home_team"].str.lower().str.contains(t) |
            filtered_df["away_team"].str.lower().str.contains(t)
        ]

    st.write(f"Showing **{len(filtered_df):,}** matching prediction snapshots:")

    display_cols = [
        "timestamp", "home_team", "away_team", "model_draw_prob_calibrated",
        "market_implied_draw_prob", "draw_odds", "net_edge", "recommendation",
        "suggested_stake_pct", "final_result", "realized_pnl", "realized_clv"
    ]

    # Format dataframe for clean table view
    df_view = filtered_df[display_cols].copy()
    df_view["timestamp"] = df_view["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
    df_view["model_draw_prob_calibrated"] = df_view["model_draw_prob_calibrated"].map("{:.1%}".format)
    df_view["market_implied_draw_prob"] = df_view["market_implied_draw_prob"].map("{:.1%}".format)
    df_view["net_edge"] = df_view["net_edge"].map("{:+.2%}".format)
    df_view["suggested_stake_pct"] = df_view["suggested_stake_pct"].map("{:.2%}".format)
    df_view["draw_odds"] = df_view["draw_odds"].map("{:.2f}".format)
    df_view["realized_pnl"] = df_view["realized_pnl"].map("£{:,.2f}".format)
    df_view["realized_clv"] = df_view["realized_clv"].map(lambda x: f"{x:+.2%}" if pd.notna(x) else "-")

    st.dataframe(df_view, use_container_width=True, height=500)

# ----------------------------------------------------
# 5. LIVE MATCH PREDICTOR
# ----------------------------------------------------
elif nav_choice == "🔮 Live Match Predictor":
    st.header("Interactive Live Fixture Value Predictor")
    st.caption("Calculate draw probabilities, vig-adjusted market odds, and Kelly stakes for upcoming Championship fixtures.")

    with get_db_session() as session:
        latest_model = session.query(ModelVersion).order_by(ModelVersion.created_at.desc()).first()

    if not latest_model:
        st.error("No serialized model version registered.")
        st.stop()

    from skynet.models.registry import ModelRegistry
    reg = ModelRegistry()
    model_pack = reg.load_model(latest_model.artifact_path)
    dc_model = model_pack["dc"]
    calibrator = model_pack["calibrator"]

    # Championship clubs
    clubs = sorted(dc_model.teams) if dc_model.teams else ["Leeds", "Sunderland", "Preston", "Stoke", "Norwich", "Watford"]

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        home_team = st.selectbox("Home Team", clubs, index=0)
    with p_col2:
        away_team = st.selectbox("Away Team", clubs, index=1 if len(clubs) > 1 else 0)

    st.subheader("Market Odds (1X2 Decimal)")
    o_col1, o_col2, o_col3, o_col4 = st.columns(4)
    with o_col1:
        market_draw = st.number_input("Draw Odds", min_value=1.05, max_value=20.0, value=3.40, step=0.05)
    with o_col2:
        market_home = st.number_input("Home Win Odds", min_value=1.05, max_value=20.0, value=2.20, step=0.05)
    with o_col3:
        market_away = st.number_input("Away Win Odds", min_value=1.05, max_value=20.0, value=3.30, step=0.05)
    with o_col4:
        venue = st.selectbox("Bookmaker Venue", ["PINNACLE", "BET365", "MARKET_AVG"])

    if st.button("Calculate Value & Suggested Stake", type="primary"):
        # Model inference
        raw_draw = dc_model.predict_draw_prob(home_team, away_team)
        cal_draw = calibrator.calibrate_single(raw_draw)

        # De-vigging
        adapter = BookmakerPricingAdapter()
        pricing_res = adapter.calculate_market_draw_probability(
            draw_odds=market_draw,
            home_odds=market_home,
            away_odds=market_away,
            venue=venue
        )

        edge = cal_draw - pricing_res.devigged_draw_prob

        # Staking
        kelly = KellyCalculator(kelly_fraction=config.kelly_fraction)
        stake_rec = kelly.calculate_stake(
            draw_prob=cal_draw,
            draw_odds=market_draw,
            edge=edge,
            draw_prob_threshold=config.draw_prob_threshold,
            min_edge=config.min_edge
        )

        st.write("")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Model Draw Prob (Calibrated)", f"{cal_draw:.1%}", f"Raw: {raw_draw:.1%}")
        r2.metric("Market Implied Draw", f"{pricing_res.devigged_draw_prob:.1%}", f"Overround: {pricing_res.overround:.3f}")
        r3.metric("Calculated Net Edge", f"{edge:+.2%}", "Model vs Market")
        r4.metric("Recommendation", "BET" if stake_rec.is_bet else "NO BET", stake_rec.reason)

        if stake_rec.is_bet:
            st.success(f"🎯 **VALUE ALERT**: Recommended Stake: `{stake_rec.stake_pct:.2%}` (£{stake_rec.stake_amount:,.2f}) — Staking Status: **DISABLED** (Paper-trade only per §2 gate).")
        else:
            st.info(f"ℹ️ **No Bet**: {stake_rec.reason}")

        # Poisson score matrix heatmap
        st.subheader("Expected Score Distribution Matrix")
        matrix = dc_model.predict_score_matrix(home_team, away_team)[:6, :6] # up to 5x5 goals
        fig_mat = px.imshow(
            matrix,
            labels=dict(x=f"{away_team} Goals", y=f"{home_team} Goals", color="Probability"),
            x=[str(i) for i in range(6)],
            y=[str(i) for i in range(6)],
            text_auto=".1%",
            color_continuous_scale="Viridis",
            title=f"Bivariate Score Probability Heatmap ({home_team} vs {away_team})"
        )
        st.plotly_chart(fig_mat, use_container_width=True)

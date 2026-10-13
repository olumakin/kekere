from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, JSON, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Match(Base):
    """Represents a football fixture/match."""
    __tablename__ = "matches"

    id = Column(String(64), primary_key=True) # Unique ID e.g. E1_202122_2021-08-07_Home_Away
    season = Column(String(16), nullable=False, index=True) # e.g. "2122"
    division = Column(String(16), default="E1", nullable=False) # EFL Championship
    match_date = Column(DateTime, nullable=False, index=True)
    home_team = Column(String(100), nullable=False, index=True)
    away_team = Column(String(100), nullable=False, index=True)
    
    # Full-time scores and result
    fthg = Column(Integer, nullable=True) # Home goals
    ftag = Column(Integer, nullable=True) # Away goals
    ftr = Column(String(2), nullable=True) # 'H', 'D', 'A'
    status = Column(String(20), default="FINISHED", nullable=False) # 'FINISHED', 'SCHEDULED'
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    historical_odds = relationship("HistoricalOdds", back_populates="match", cascade="all, delete-orphan")
    predictions = relationship("PredictionAudit", back_populates="match", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_match_date_teams", "match_date", "home_team", "away_team"),
    )

    def is_draw(self) -> Optional[bool]:
        if self.ftr is None:
            return None
        return self.ftr == "D"


class HistoricalOdds(Base):
    """Historical odds snapshot for a match."""
    __tablename__ = "historical_odds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(64), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True)
    venue = Column(String(50), nullable=False) # e.g. 'PINNACLE', 'MARKET_AVG', 'BET365'
    odds_type = Column(String(20), nullable=False) # 'OPENING', 'CLOSING', 'LIVE'
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    home_odds = Column(Float, nullable=False)
    draw_odds = Column(Float, nullable=False)
    away_odds = Column(Float, nullable=False)

    # Calculated market properties
    overround = Column(Float, nullable=True)
    devigged_draw_prob = Column(Float, nullable=True)
    fee_assumptions = Column(JSON, default=dict)

    match = relationship("Match", back_populates="historical_odds")

    __table_args__ = (
        Index("ix_odds_match_venue_type", "match_id", "venue", "odds_type"),
    )


class ModelVersion(Base):
    """Registry of trained model versions and serialized artifacts."""
    __tablename__ = "model_versions"

    id = Column(String(64), primary_key=True) # e.g. "DC_v1.0.0_20210801"
    version_tag = Column(String(32), nullable=False, index=True) # e.g. "v1.0.0"
    model_type = Column(String(50), nullable=False) # e.g. "DIXON_COLES", "ENSEMBLE_RF"
    artifact_path = Column(String(255), nullable=False)
    model_hash = Column(String(64), nullable=True)
    training_cutoff = Column(DateTime, nullable=False)
    hyperparameters = Column(JSON, default=dict)
    metadata_info = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    predictions = relationship("PredictionAudit", back_populates="model_version")


class PredictionAudit(Base):
    """Complete, immutable audit log of every prediction (including No Bet)."""
    __tablename__ = "predictions_audit"

    id = Column(String(64), primary_key=True) # e.g. pred_<match_id>_<model_id>
    match_id = Column(String(64), ForeignKey("matches.id", ondelete="CASCADE"), nullable=False, index=True)
    model_version_id = Column(String(64), ForeignKey("model_versions.id"), nullable=False, index=True)
    prediction_timestamp = Column(DateTime, nullable=False, index=True)

    # Snapshots
    feature_snapshot = Column(JSON, nullable=False) # Exact features used at inference
    odds_snapshot = Column(JSON, nullable=False) # Venue, prices, type, fees

    # Probabilities & Edge
    model_draw_prob_raw = Column(Float, nullable=False)
    model_draw_prob_calibrated = Column(Float, nullable=False)
    market_implied_draw_prob = Column(Float, nullable=False) # Vig-adjusted
    net_edge = Column(Float, nullable=False) # Calibrated - Market
    is_vig_adjusted = Column(Boolean, default=True, nullable=False)

    # Strategy & Staking Decision
    recommendation = Column(String(16), nullable=False) # 'BET' or 'NO_BET'
    suggested_stake_pct = Column(Float, nullable=False) # Fractional Kelly %
    suggested_stake_amount = Column(Float, nullable=True)
    staking_formula_inputs = Column(JSON, nullable=False) # Kelly fraction, bankroll, edge, b
    gating_status = Column(String(20), nullable=False) # 'ENABLED' or 'DISABLED'

    # Ground Truth & Evaluation (Updated post-match)
    final_result = Column(String(2), nullable=True) # 'H', 'D', 'A'
    closing_odds_snapshot = Column(JSON, nullable=True) # Pinnacle closing odds
    realized_clv = Column(Float, nullable=True) # Closing line value edge
    realized_pnl = Column(Float, nullable=True) # Profit/loss if staked

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    match = relationship("Match", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")

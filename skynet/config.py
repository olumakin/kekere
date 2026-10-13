import os
from pathlib import Path
from typing import Any, Dict, List
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "settings.yaml"

class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._raw = config_dict
        
        # Database
        db_conf = config_dict.get("database", {})
        self.db_url = os.environ.get("DATABASE_URL", db_conf.get("url", "sqlite:///skynet_championship.db"))
        self.db_echo = db_conf.get("echo", False)
        
        # Data
        data_conf = config_dict.get("data", {})
        self.raw_dir = Path(data_conf.get("raw_dir", "data/raw"))
        self.models_dir = Path(data_conf.get("models_dir", "data/models"))
        self.football_data_base_url = data_conf.get("football_data_base_url", "https://www.football-data.co.uk/mmz4281")
        self.division_code = data_conf.get("division_code", "E1")
        self.burn_in_seasons: List[str] = data_conf.get("burn_in_seasons", ["1920", "2021"])
        self.evaluation_seasons: List[str] = data_conf.get("evaluation_seasons", ["2122", "2223", "2324", "2425", "2526"])
        
        # Strategy
        strat_conf = config_dict.get("strategy", {})
        self.draw_prob_threshold = float(strat_conf.get("draw_prob_threshold", 0.31))
        self.min_edge = float(strat_conf.get("min_edge", 0.03))
        self.kelly_fraction = float(strat_conf.get("kelly_fraction", 0.25))
        self.max_stake_pct = float(strat_conf.get("max_stake_pct", 0.02))
        self.daily_exposure_limit_pct = float(strat_conf.get("daily_exposure_limit_pct", 0.06))
        self.circuit_breaker_loss_streak = int(strat_conf.get("circuit_breaker_loss_streak", 5))
        
        # Go/No-Go Gate
        gate_conf = config_dict.get("gate", {})
        self.min_bets = int(gate_conf.get("min_bets", 150))
        self.min_roi = float(gate_conf.get("min_roi", 0.0))
        self.min_clv = float(gate_conf.get("min_clv", 0.0))
        self.cox_slope_min = float(gate_conf.get("cox_slope_min", 0.8))
        self.cox_slope_max = float(gate_conf.get("cox_slope_max", 1.2))
        self.cox_intercept_min = float(gate_conf.get("cox_intercept_min", -0.05))
        self.cox_intercept_max = float(gate_conf.get("cox_intercept_max", 0.05))

        # Dixon-Coles
        dc_conf = config_dict.get("dixon_coles", {})
        self.dixon_coles_xi = float(dc_conf.get("xi", 0.0019))
        self.dixon_coles_max_goals = int(dc_conf.get("max_goals", 10))

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> "Config":
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}
        return cls(data)

# Singleton global instance
config = Config.load()

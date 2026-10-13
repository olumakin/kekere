from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
from skynet.features.elo import EloTracker

class FeatureEngine:
    """
    Chronological feature generator guaranteeing zero future lookahead.
    Tracks team match histories, rest days, and draw rates as matches are processed in order.
    """

    def __init__(self):
        self.elo_tracker = EloTracker()
        # Team match history: list of dicts with match_date, ftr, is_home, goals_for, goals_against
        self.team_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.team_last_date: Dict[str, datetime] = {}
        self.current_season: Optional[str] = None

    def start_season(self, season: str, teams: Optional[List[str]] = None):
        self.elo_tracker.start_season(season, teams)
        self.current_season = season

    def calculate_pre_match_features(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        season: str
    ) -> Dict[str, Any]:
        """
        Generate feature set strictly prior to kickoff.
        Zero data leakage: relies ONLY on completed matches prior to match_date.
        """
        # Ensure season rollover if changed
        if season != self.current_season:
            self.start_season(season)

        # 1. Elo features
        elo_home = self.elo_tracker.get_rating(home_team, season)
        elo_away = self.elo_tracker.get_rating(away_team, season)
        elo_diff = elo_home - elo_away

        # 2. Historical Draw Rates
        def get_draw_rate(history: List[Dict[str, Any]], n: int) -> float:
            recent = history[-n:]
            if not recent:
                return 0.28 # Historical Championship base draw rate (~28%)
            draws = sum(1 for m in recent if m["ftr"] == "D")
            return draws / len(recent)

        home_hist = self.team_history[home_team]
        away_hist = self.team_history[away_team]

        draw_rate_home_10 = get_draw_rate(home_hist, 10)
        draw_rate_away_10 = get_draw_rate(away_hist, 10)
        draw_rate_home_30 = get_draw_rate(home_hist, 30)
        draw_rate_away_30 = get_draw_rate(away_hist, 30)
        draw_rate_combined = (draw_rate_home_10 + draw_rate_away_10) / 2.0

        # 3. Rest Days (capped at 30 days)
        def get_rest_days(team: str) -> float:
            if team not in self.team_last_date:
                return 7.0 # Default rest for first match / after break
            delta = (match_date - self.team_last_date[team]).total_seconds() / 86400.0
            return max(1.0, min(delta, 30.0))

        rest_home = get_rest_days(home_team)
        rest_away = get_rest_days(away_team)
        rest_diff = rest_home - rest_away

        return {
            "elo_home": round(elo_home, 2),
            "elo_away": round(elo_away, 2),
            "elo_diff": round(elo_diff, 2),
            "draw_rate_home_10": round(draw_rate_home_10, 4),
            "draw_rate_away_10": round(draw_rate_away_10, 4),
            "draw_rate_home_30": round(draw_rate_home_30, 4),
            "draw_rate_away_30": round(draw_rate_away_30, 4),
            "draw_rate_combined": round(draw_rate_combined, 4),
            "rest_days_home": round(rest_home, 1),
            "rest_days_away": round(rest_away, 1),
            "rest_diff": round(rest_diff, 1),
        }

    def record_match_result(
        self,
        home_team: str,
        away_team: str,
        match_date: datetime,
        fthg: int,
        ftag: int,
        ftr: str,
        season: str
    ):
        """
        Record match outcome into state AFTER the match has concluded.
        Updates Elo and chronological history.
        """
        if ftr not in ("H", "D", "A"):
            return

        goal_diff = fthg - ftag

        # Update Elo
        self.elo_tracker.update_match(home_team, away_team, ftr, season, goal_diff)

        # Update team history
        self.team_history[home_team].append({
            "match_date": match_date,
            "ftr": ftr,
            "is_home": True,
            "goals_for": fthg,
            "goals_against": ftag,
        })
        self.team_history[away_team].append({
            "match_date": match_date,
            "ftr": ftr,
            "is_home": False,
            "goals_for": ftag,
            "goals_against": fthg,
        })

        # Update last match dates
        self.team_last_date[home_team] = match_date
        self.team_last_date[away_team] = match_date

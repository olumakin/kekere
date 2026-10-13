from typing import Dict, Optional, Tuple
from datetime import datetime
from skynet.features.churn import get_team_prior_offset

class EloTracker:
    """
    Chronological Elo rating tracker for football clubs.
    Enforces strict temporal order and seasonal mean-reversion with churn priors.
    """

    def __init__(self, base_rating: float = 1500.0, k_factor: float = 24.0, home_advantage: float = 65.0):
        self.base_rating = base_rating
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings: Dict[str, float] = {}
        self.current_season: Optional[str] = None

    def start_season(self, season: str, teams_in_season: Optional[list] = None):
        """Handle season rollover: apply mean reversion and initialize promoted/relegated priors."""
        if self.current_season == season:
            return

        self.current_season = season
        # Mean reversion for existing teams: pull 25% towards base
        reversion_factor = 0.75
        for team in list(self.ratings.keys()):
            self.ratings[team] = self.base_rating + reversion_factor * (self.ratings[team] - self.base_rating)

        # Set priors for newly appearing clubs
        if teams_in_season:
            for team in teams_in_season:
                if team not in self.ratings:
                    offset = get_team_prior_offset(season, team)
                    self.ratings[team] = self.base_rating + offset

    def get_rating(self, team: str, season: Optional[str] = None) -> float:
        """Get team rating as-of current state. Initializes with churn prior if unseen."""
        if team not in self.ratings:
            offset = get_team_prior_offset(season or self.current_season or "", team)
            self.ratings[team] = self.base_rating + offset
        return self.ratings[team]

    def get_expected_score(self, home_team: str, away_team: str, season: Optional[str] = None) -> Tuple[float, float]:
        r_home = self.get_rating(home_team, season)
        r_away = self.get_rating(away_team, season)
        exp_home = 1.0 / (1.0 + 10.0 ** ((r_away - (r_home + self.home_advantage)) / 400.0))
        exp_away = 1.0 - exp_home
        return exp_home, exp_away

    def update_match(self, home_team: str, away_team: str, ftr: str, season: Optional[str] = None, goal_diff: int = 0):
        """
        Updates Elo post-match.
        ftr: 'H' (home win), 'D' (draw), 'A' (away win)
        """
        exp_home, exp_away = self.get_expected_score(home_team, away_team, season)

        if ftr == "H":
            actual_home = 1.0
            actual_away = 0.0
        elif ftr == "D":
            actual_home = 0.5
            actual_away = 0.5
        elif ftr == "A":
            actual_home = 0.0
            actual_away = 1.0
        else:
            return # Unplayed or invalid

        # Goal difference multiplier
        margin = abs(goal_diff)
        if margin <= 1:
            g_mult = 1.0
        elif margin == 2:
            g_mult = 1.5
        else:
            g_mult = (11.0 + margin) / 8.0

        r_h = self.get_rating(home_team, season)
        r_a = self.get_rating(away_team, season)

        delta_h = self.k_factor * g_mult * (actual_home - exp_home)
        delta_a = self.k_factor * g_mult * (actual_away - exp_away)

        self.ratings[home_team] = r_h + delta_h
        self.ratings[away_team] = r_a + delta_a

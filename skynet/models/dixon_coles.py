import math
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial

class DixonColesModel:
    """
    Dixon-Coles bivariate Poisson model with low-score adjustment and exponential time decay.
    Reference: Dixon, M. J., & Coles, S. G. (1997).
    Modelling Association Football Scores and Inefficiencies in the Football Betting Market.
    """

    def __init__(self, xi: float = 0.0019, max_goals: int = 10):
        self.xi = xi # Decay rate per day (~0.0019 gives half-life ~ 365 days)
        self.max_goals = max_goals
        self.teams: List[str] = []
        self.team_to_idx: Dict[str, int] = {}
        self.params: Optional[np.ndarray] = None
        self.home_adv: float = 0.25 # gamma
        self.rho: float = -0.05 # low score correlation
        self.attacks: Dict[str, float] = {}
        self.defenses: Dict[str, float] = {}
        self.is_fitted: bool = False

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        """Dixon-Coles low score adjustment factor."""
        if x == 0 and y == 0:
            val = 1.0 - lam * mu * rho
        elif x == 0 and y == 1:
            val = 1.0 + lam * rho
        elif x == 1 and y == 0:
            val = 1.0 + mu * rho
        elif x == 1 and y == 1:
            val = 1.0 - rho
        else:
            val = 1.0
        return max(val, 1e-6)

    def fit(self, matches: List[Dict[str, Any]], current_date: Optional[datetime] = None):
        """
        Fit model parameters via maximum weighted log-likelihood.
        matches: list of dicts with 'home_team', 'away_team', 'fthg', 'ftag', 'match_date'
        """
        if not matches:
            raise ValueError("No matches provided to fit Dixon-Coles model.")

        # Identify unique teams
        all_teams = set()
        for m in matches:
            all_teams.add(m["home_team"])
            all_teams.add(m["away_team"])
        self.teams = sorted(list(all_teams))
        self.team_to_idx = {t: i for i, t in enumerate(self.teams)}
        n_teams = len(self.teams)

        if n_teams < 2:
            raise ValueError("Need at least 2 teams to fit Dixon-Coles model.")

        # Reference date for time decay
        if current_date is None:
            current_date = max(m["match_date"] for m in matches)

        # Precompute arrays for fast objective evaluation
        h_idx = np.array([self.team_to_idx[m["home_team"]] for m in matches], dtype=int)
        a_idx = np.array([self.team_to_idx[m["away_team"]] for m in matches], dtype=int)
        x_vals = np.array([m["fthg"] for m in matches], dtype=int)
        y_vals = np.array([m["ftag"] for m in matches], dtype=int)

        # Calculate time decay weights
        days_diff = np.array([(current_date - m["match_date"]).total_seconds() / 86400.0 for m in matches])
        days_diff = np.maximum(days_diff, 0.0)
        weights = np.exp(-self.xi * days_diff)

        # Initial parameters:
        # alpha (attack) [0:n], beta (defense) [n:2n], gamma (home_adv) [2n], rho [2n+1]
        init_params = np.zeros(2 * n_teams + 2)
        init_params[2 * n_teams] = 0.25 # initial home advantage
        init_params[2 * n_teams + 1] = -0.05 # initial rho

        def loss_fn(p):
            alpha = p[:n_teams]
            beta = p[n_teams:2*n_teams]
            gamma = p[2*n_teams]
            rho = p[2*n_teams + 1]

            # Zero-sum constraint penalty on attack parameters: sum(alpha) = 0
            constraint_penalty = 1000.0 * (np.sum(alpha) ** 2)

            # Compute lambda and mu
            lam = np.exp(alpha[h_idx] + beta[a_idx] + gamma)
            mu = np.exp(alpha[a_idx] + beta[h_idx])

            # Bound lambdas to prevent numerical overflow/underflow
            lam = np.clip(lam, 1e-4, 15.0)
            mu = np.clip(mu, 1e-4, 15.0)

            # Dixon-Coles adjustment tau
            tau_vals = np.ones_like(lam)
            m00 = (x_vals == 0) & (y_vals == 0)
            m01 = (x_vals == 0) & (y_vals == 1)
            m10 = (x_vals == 1) & (y_vals == 0)
            m11 = (x_vals == 1) & (y_vals == 1)

            tau_vals[m00] = 1.0 - lam[m00] * mu[m00] * rho
            tau_vals[m01] = 1.0 + lam[m01] * rho
            tau_vals[m10] = 1.0 + mu[m10] * rho
            tau_vals[m11] = 1.0 - rho
            tau_vals = np.maximum(tau_vals, 1e-6)

            # Weighted log-likelihood
            log_l = np.log(tau_vals) + x_vals * np.log(lam) - lam + y_vals * np.log(mu) - mu
            nll = -np.sum(weights * log_l) + constraint_penalty
            return nll

        bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(0.0, 1.0), (-0.25, 0.25)]

        res = minimize(loss_fn, init_params, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200, "disp": False})

        # Save fitted parameters
        self.params = res.x
        alphas = self.params[:n_teams]
        # Normalize alphas exactly to mean 0
        alphas = alphas - np.mean(alphas)
        betas = self.params[n_teams:2*n_teams]
        self.home_adv = float(self.params[2*n_teams])
        self.rho = float(self.params[2*n_teams + 1])

        self.attacks = {team: float(alphas[i]) for i, team in enumerate(self.teams)}
        self.defenses = {team: float(betas[i]) for i, team in enumerate(self.teams)}
        self.is_fitted = True

    def predict_score_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Computes bivariate goal probability matrix (max_goals x max_goals)."""
        alpha_h = self.attacks.get(home_team, 0.0)
        beta_a = self.defenses.get(away_team, 0.0)
        alpha_a = self.attacks.get(away_team, 0.0)
        beta_h = self.defenses.get(home_team, 0.0)

        lam = math.exp(alpha_h + beta_a + self.home_adv)
        mu = math.exp(alpha_a + beta_h)

        max_g = self.max_goals
        probs = np.zeros((max_g + 1, max_g + 1))

        # Poisson pmfs
        p_x = np.array([(lam**x * math.exp(-lam)) / factorial(x) for x in range(max_g + 1)])
        p_y = np.array([(mu**y * math.exp(-mu)) / factorial(y) for y in range(max_g + 1)])

        for x in range(max_g + 1):
            for y in range(max_g + 1):
                tau = self._tau(x, y, lam, mu, self.rho)
                probs[x, y] = tau * p_x[x] * p_y[y]

        total = np.sum(probs)
        if total > 0:
            probs /= total
        return probs

    def predict_draw_prob(self, home_team: str, away_team: str) -> float:
        """Predicts the probability of a draw."""
        matrix = self.predict_score_matrix(home_team, away_team)
        return float(np.trace(matrix))

    def predict_1x2_probs(self, home_team: str, away_team: str) -> Dict[str, float]:
        """Returns 1X2 probabilities for home win, draw, and away win."""
        matrix = self.predict_score_matrix(home_team, away_team)
        p_draw = float(np.trace(matrix))
        p_home = float(np.sum(np.tril(matrix, -1)))
        p_away = float(np.sum(np.triu(matrix, 1)))
        return {
            "home": p_home,
            "draw": p_draw,
            "away": p_away,
        }

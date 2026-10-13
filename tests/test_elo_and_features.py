from datetime import datetime, timedelta
import pytest
from skynet.features.elo import EloTracker
from skynet.features.churn import get_team_prior_offset
from skynet.features.engineering import FeatureEngine

def test_churn_priors():
    # Relegated from Premier League in 2122: Fulham (+60)
    assert get_team_prior_offset("2122", "Fulham") == 60.0
    # Promoted from League One in 2122: Peterborough (-60)
    assert get_team_prior_offset("2122", "Peterborough") == -60.0
    # Regular Championship club
    assert get_team_prior_offset("2122", "Bristol City") == 0.0

def test_elo_tracker_initialization_and_update():
    tracker = EloTracker()
    tracker.start_season("2122", ["Fulham", "Peterborough"])
    
    # Fulham gets 1560, Peterborough gets 1440
    assert tracker.get_rating("Fulham", "2122") == 1560.0
    assert tracker.get_rating("Peterborough", "2122") == 1440.0

    # Fulham wins 3-0 against Peterborough (margin = 3)
    tracker.update_match("Fulham", "Peterborough", "H", "2122", goal_diff=3)
    assert tracker.get_rating("Fulham", "2122") > 1560.0
    assert tracker.get_rating("Peterborough", "2122") < 1440.0

def test_zero_leakage_feature_engine():
    engine = FeatureEngine()
    engine.start_season("2122")
    
    t0 = datetime(2021, 8, 7, 15, 0)
    # Match 1: Sheffield United vs Birmingham
    f0 = engine.calculate_pre_match_features("Sheffield United", "Birmingham", t0, "2122")
    # Both teams have 0 prior matches -> default rest 7 days, default draw rate 28%
    assert f0["rest_days_home"] == 7.0
    assert f0["draw_rate_home_10"] == 0.28

    # Match concludes as Draw 0-0
    engine.record_match_result("Sheffield United", "Birmingham", t0, 0, 0, "D", "2122")

    # Match 2: 7 days later
    t1 = t0 + timedelta(days=7)
    f1 = engine.calculate_pre_match_features("Sheffield United", "Swansea", t1, "2122")
    assert f1["rest_days_home"] == 7.0
    # Sheffield United now has 1 match in history which was a draw -> draw_rate = 1.0 (100%)
    assert f1["draw_rate_home_10"] == 1.0
    # Swansea has 0 matches
    assert f1["draw_rate_away_10"] == 0.28

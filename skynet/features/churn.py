from typing import Dict, Set

# Known promoted and relegated clubs for EFL Championship by season code
# Relegated from Premier League (Stronger prior: +60)
# Promoted from League One (Weaker prior: -60)
CHURN_REGISTRY: Dict[str, Dict[str, Set[str]]] = {
    "1920": {
        "relegated_from_pl": {"Cardiff", "Fulham", "Huddersfield"},
        "promoted_from_l1": {"Luton", "Barnsley", "Charlton"},
    },
    "2021": {
        "relegated_from_pl": {"Bournemouth", "Watford", "Norwich"},
        "promoted_from_l1": {"Coventry", "Rotherham", "Wycombe"},
    },
    "2122": {
        "relegated_from_pl": {"Fulham", "West Brom", "Sheffield United"},
        "promoted_from_l1": {"Hull", "Peterborough", "Blackpool"},
    },
    "2223": {
        "relegated_from_pl": {"Burnley", "Watford", "Norwich"},
        "promoted_from_l1": {"Wigan", "Rotherham", "Sunderland"},
    },
    "2324": {
        "relegated_from_pl": {"Leicester", "Leeds", "Southampton"},
        "promoted_from_l1": {"Plymouth", "Ipswich", "Sheffield Weds"},
    },
    "2425": {
        "relegated_from_pl": {"Luton", "Burnley", "Sheffield United"},
        "promoted_from_l1": {"Portsmouth", "Derby", "Oxford"},
    },
    "2526": {
        "relegated_from_pl": {"Leicester", "Ipswich", "Southampton"},
        "promoted_from_l1": {"Birmingham", "Wrexham", "Stockport"},
    },
}

def get_team_prior_offset(season: str, team: str) -> float:
    """Returns starting Elo offset based on promotion/relegation churn."""
    season_churn = CHURN_REGISTRY.get(season, {})
    rel = season_churn.get("relegated_from_pl", set())
    prom = season_churn.get("promoted_from_l1", set())

    # Case-insensitive / partial match check
    for r in rel:
        if r.lower() in team.lower() or team.lower() in r.lower():
            return 60.0 # Prior boost for relegated PL teams
    for p in prom:
        if p.lower() in team.lower() or team.lower() in p.lower():
            return -60.0 # Prior discount for promoted League One teams
    return 0.0

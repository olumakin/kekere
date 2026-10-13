from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional

class SportsApiAdapter(ABC):
    """
    Abstract interface for premium sports APIs (API-Football, Sportmonks).
    Used for live fixture feeds and stats ingestion.
    """

    @abstractmethod
    def fetch_fixtures(self, league_id: int, season: str, from_date: datetime, to_date: datetime) -> List[Dict[str, Any]]:
        """Fetch list of scheduled fixtures and match metadata."""
        pass

    @abstractmethod
    def fetch_live_odds(self, fixture_id: str) -> Dict[str, Any]:
        """Fetch real-time 1X2 market odds across supported bookmakers."""
        pass

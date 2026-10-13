import os
import io
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests
import pandas as pd
from sqlalchemy.orm import Session

from skynet.config import config
from skynet.db.schema import Match, HistoricalOdds
from skynet.pricing.bookmaker import BookmakerPricingAdapter

class FootballDataIngestor:
    """
    Ingestion pipeline for football-data.co.uk EFL Championship datasets.
    Handles caching, multi-format date parsing, odds extraction, and DB synchronization.
    """

    def __init__(self, raw_dir: Optional[Path] = None, base_url: Optional[str] = None):
        self.raw_dir = raw_dir or config.raw_dir
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url or config.football_data_base_url
        self.pricing_adapter = BookmakerPricingAdapter()

    def get_season_csv_path(self, season: str) -> Path:
        return self.raw_dir / f"E1_{season}.csv"

    def fetch_season_csv(self, season: str, force_download: bool = False) -> Path:
        """Downloads season CSV from football-data.co.uk if not cached."""
        target_path = self.get_season_csv_path(season)
        if target_path.exists() and not force_download:
            return target_path

        url = f"{self.base_url}/{season}/E1.csv"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200 and len(resp.content) > 100:
                with open(target_path, "wb") as f:
                    f.write(resp.content)
                return target_path
            else:
                # Some seasons might not exist yet or have alternative naming
                raise RuntimeError(f"HTTP {resp.status_code} fetching {url}")
        except Exception as e:
            if target_path.exists():
                return target_path # Fall back to existing cached file if network fails
            raise RuntimeError(f"Could not download {url}: {e}")

    @staticmethod
    def parse_match_date(date_str: str, time_str: Optional[str] = None) -> datetime:
        """Parses dates formatted as DD/MM/YYYY or DD/MM/YY."""
        date_str = str(date_str).strip()
        time_part = "15:00"
        if time_str and pd.notna(time_str) and str(time_str).strip():
            time_part = str(time_str).strip()

        # Try various date formats
        for fmt in ("%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d"):
            try:
                d = datetime.strptime(date_str, fmt)
                hour, minute = map(int, time_part.split(":")[:2])
                return d.replace(hour=hour, minute=minute)
            except Exception:
                continue

        raise ValueError(f"Unable to parse match date: '{date_str}'")

    def parse_csv(self, file_path: Path, season: str) -> pd.DataFrame:
        """Reads CSV, cleans column headers and filters non-empty rows."""
        df = pd.read_csv(file_path, encoding="latin1")
        # Strip column names
        df.columns = [c.strip() for c in df.columns]
        # Filter rows with missing essentials
        df = df[df["HomeTeam"].notna() & df["AwayTeam"].notna() & df["Date"].notna()]
        df["season"] = season
        return df

    def sync_season_to_db(self, session: Session, season: str, force_download: bool = False) -> int:
        """Parses CSV and inserts/updates matches and odds into database."""
        csv_path = self.fetch_season_csv(season, force_download=force_download)
        df = self.parse_csv(csv_path, season)

        matches_count = 0
        for _, row in df.iterrows():
            home_team = str(row["HomeTeam"]).strip()
            away_team = str(row["AwayTeam"]).strip()
            time_str = row.get("Time") if "Time" in row else None
            try:
                match_date = self.parse_match_date(row["Date"], time_str)
            except Exception:
                continue

            date_tag = match_date.strftime("%Y-%m-%d")
            match_id = f"E1_{season}_{date_tag}_{home_team}_{away_team}".replace(" ", "_")

            # Score and result
            fthg = int(row["FTHG"]) if pd.notna(row.get("FTHG")) else None
            ftag = int(row["FTAG"]) if pd.notna(row.get("FTAG")) else None
            ftr = str(row["FTR"]).strip() if pd.notna(row.get("FTR")) else None
            status = "FINISHED" if ftr in ("H", "D", "A") else "SCHEDULED"

            # Check if match exists
            match = session.query(Match).filter_by(id=match_id).first()
            if not match:
                match = Match(
                    id=match_id,
                    season=season,
                    division="E1",
                    match_date=match_date,
                    home_team=home_team,
                    away_team=away_team,
                    fthg=fthg,
                    ftag=ftag,
                    ftr=ftr,
                    status=status
                )
                session.add(match)
                session.flush()
                matches_count += 1
            else:
                match.fthg = fthg
                match.ftag = ftag
                match.ftr = ftr
                match.status = status

            # Odds ingestion
            def add_odds(venue: str, odds_type: str, h_col: str, d_col: str, a_col: str):
                h = float(row[h_col]) if h_col in row and pd.notna(row[h_col]) else None
                d = float(row[d_col]) if d_col in row and pd.notna(row[d_col]) else None
                a = float(row[a_col]) if a_col in row and pd.notna(row[a_col]) else None

                if d is not None and d > 1.0:
                    pricing = self.pricing_adapter.calculate_market_draw_probability(
                        draw_odds=d, home_odds=h, away_odds=a, venue=venue
                    )
                    existing_odds = session.query(HistoricalOdds).filter_by(
                        match_id=match_id, venue=venue, odds_type=odds_type
                    ).first()
                    if not existing_odds:
                        odds_obj = HistoricalOdds(
                            match_id=match_id,
                            venue=venue,
                            odds_type=odds_type,
                            home_odds=h if h else 0.0,
                            draw_odds=d,
                            away_odds=a if a else 0.0,
                            overround=pricing.overround,
                            devigged_draw_prob=pricing.devigged_draw_prob,
                            fee_assumptions={"commission": 0.0}
                        )
                        session.add(odds_obj)

            # 1. Pinnacle Opening (PSH/PSD/PSA)
            add_odds("PINNACLE", "OPENING", "PSH", "PSD", "PSA")
            # 2. Pinnacle Closing (PSCH/PSCD/PSCA)
            add_odds("PINNACLE", "CLOSING", "PSCH", "PSCD", "PSCA")
            # 3. Market Average Opening (AvgH/AvgD/AvgA)
            add_odds("MARKET_AVG", "OPENING", "AvgH", "AvgD", "AvgA")
            # 4. Market Average Closing (AvgCH/AvgCD/AvgCA)
            add_odds("MARKET_AVG", "CLOSING", "AvgCH", "AvgCD", "AvgCA")
            # 5. Bet365 Opening (B365H/B365D/B365A)
            add_odds("BET365", "OPENING", "B365H", "B365D", "B365A")

        return matches_count

#!/usr/bin/env python3
"""
Weather Market Paper Trader
Implements ColdMath's 3 trading strategies on Polymarket temperature bucket markets.

Strategies:
  1. Layered No Hedge    — Buy "No" on multiple adjacent bands the forecast excludes
  2. Lottery Tickets     — Buy cheap "Yes" on near-miss bands; Kelly-sized by forecast edge
  3. High Conviction     — Single large "No" on bands impossibly far from forecast

Portfolio:  $500 total  |  S1=$200  S2=$100  S3=$200  (separate bankrolls)

S2 Kelly sizing:
  - Estimate true probability p_est using normal distribution around NWS forecast
    with σ = S2_FORECAST_SIGMA (4°F, typical 1-day NWS error)
  - edge = p_est − market_yes_price
  - Only bet when edge ≥ S2_MIN_EDGE (5%)
  - Kelly fraction f* = edge / (1 − market_yes_price)
  - Bet size = S2_KELLY_FRAC × f* × s2_balance  (half-Kelly, capped at S2_MAX_BET_FRAC)

Modes:
  python paper_trader.py --backtest             # Backtest on historical parquet data
  python paper_trader.py --forward              # Scan live markets, print signals only
  python paper_trader.py --forward --live       # Scan + record paper trades
  python paper_trader.py --positions            # Show open positions with current prices
  python paper_trader.py --reset                # Reset portfolio to starting balance
  python paper_trader.py --backtest --lookback 6  # Limit backtest to last N months
"""

import re
import json
import argparse
import requests
from math import erf, sqrt
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

STARTING_BALANCE = 500.0   # Total across all strategies
S1_STARTING      = 0.0     # S1 suspended
S2_STARTING      = 100.0   # S2 (Lottery Tickets) bankroll
S3_STARTING      = 400.0   # S3 (High Conviction) bankroll

# Polymarket non-linear taker fee (no fee on settlement):
#   fee_per_share = feeRate × p × (p × (1-p))^exponent
#   effective_price = p + fee_per_share = p × (1 + feeRate × (p × (1-p))^exponent)
# With exponent=1: fee ≈ 0 at extremes (p→0 or p→1), max at p=0.5.
# Examples: p=0.03 → +0.006% fee; p=0.97 → +0.006%; p=0.50 → +0.50%
POLYMARKET_FEE_RATE     = 0.02
POLYMARKET_FEE_EXPONENT = 1

STATE_FILE = Path(__file__).parent / "paper_trader_state.json"
# Parquet data lives in prediction-market-analysis (kept separate from trading logs)
DATA_DIR   = Path(__file__).parent.parent.parent / "prediction-market-analysis" / "data" / "polymarket" / "markets"

# Strategy 1: Layered No Hedge
S1_NO_MIN     = 0.80   # Min "No" price to enter
S1_NO_MAX     = 0.97   # Max "No" price (too illiquid above)
S1_MIN_DIST_F = 6      # Skip bands within 6°F of forecast (forecast ≈ ±5°F noise)
S1_MAX_LAYERS = 5      # Max bands per city/date; each layer = s1_balance / S1_MAX_LAYERS

# Strategy 2: Lottery Tickets — Kelly-sized
S2_YES_MIN        = 0.050  # Only buy Yes priced at or above this (avoid dust)
S2_YES_MAX        = 0.150  # Only buy Yes priced below this
S2_WINDOW_F       = 7      # Only bands within ±7°F of forecast
S2_MIN_EDGE       = 0.05   # Min edge (p_est − market_price) to bet; user spec: 5%
S2_KELLY_FRAC     = 0.5    # Half-Kelly fraction for safety
S2_FORECAST_SIGMA = 4.0    # NWS forecast std dev in °F (typical 1-day error)
S2_MAX_BET_FRAC   = 0.30   # Cap single bet at 30% of S2 balance

# Strategy 3: High Conviction — Kelly-sized No positions
S3_NO_MIN    = 0.95   # "No" must be priced >= this
S3_NO_MAX    = 0.99   # "No" must be priced <= this (above 99% market is too illiquid)
S3_DIST_F    = 15     # Band midpoint >= 15°F from forecast
S3_MIN_EDGE  = 0.01   # Min edge on No side (p_no_win − no_price); user spec: 1%
S3_KELLY_FRAC = 0.5   # Half-Kelly fraction
S3_STOP_LOSS  = 0.04  # Exit S3 if No price drops by 4% from entry (e.g. 0.996 → 0.956)

# City registry — NWS endpoints for US cities
NWS_ENDPOINTS = {
    "new-york-city": "https://api.weather.gov/gridpoints/OKX/37,39/forecast/hourly",
    "chicago":       "https://api.weather.gov/gridpoints/LOT/66,77/forecast/hourly",
    "miami":         "https://api.weather.gov/gridpoints/MFL/106,51/forecast/hourly",
    "dallas":        "https://api.weather.gov/gridpoints/FWD/87,107/forecast/hourly",
    "seattle":       "https://api.weather.gov/gridpoints/SEW/124,61/forecast/hourly",
    "atlanta":       "https://api.weather.gov/gridpoints/FFC/50,82/forecast/hourly",
}
STATION_IDS = {
    "new-york-city": "KLGA",
    "chicago":       "KORD",
    "miami":         "KMIA",
    "dallas":        "KDAL",
    "seattle":       "KSEA",
    "atlanta":       "KATL",
}

# All tracked cities (including London via Open-Meteo)
CITIES = {
    "new-york-city": {"lat": 40.7772,  "lon": -73.8726,  "name": "New York City", "tz": "America/New_York"},
    "chicago":       {"lat": 41.9742,  "lon": -87.9073,  "name": "Chicago",        "tz": "America/Chicago"},
    "miami":         {"lat": 25.7959,  "lon": -80.2870,  "name": "Miami",          "tz": "America/New_York"},
    "dallas":        {"lat": 32.8471,  "lon": -96.8518,  "name": "Dallas",         "tz": "America/Chicago"},
    "seattle":       {"lat": 47.4502,  "lon": -122.3088, "name": "Seattle",        "tz": "America/Los_Angeles"},
    "atlanta":       {"lat": 33.6407,  "lon": -84.4277,  "name": "Atlanta",        "tz": "America/New_York"},
    "london":        {"lat": 51.4775,  "lon": -0.4614,   "name": "London",         "tz": "Europe/London"},
}

# Maps question city strings → city slugs
CITY_ALIASES: dict[str, str] = {
    "new york city":            "new-york-city",
    "nyc":                      "new-york-city",
    "new york":                 "new-york-city",
    "new york's central park":  "new-york-city",
    "central park":             "new-york-city",
    "chicago":                  "chicago",
    "miami":                    "miami",
    "dallas":                   "dallas",
    "seattle":                  "seattle",
    "atlanta":                  "atlanta",
    "london":                   "london",
}

# Polymarket event slug prefixes per city
CITY_SLUGS: dict[str, list[str]] = {
    "new-york-city": ["new-york-city", "nyc"],
    "chicago":       ["chicago"],
    "miami":         ["miami"],
    "dallas":        ["dallas"],
    "seattle":       ["seattle"],
    "atlanta":       ["atlanta"],
    "london":        ["london"],
}

MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

# =============================================================================
# COLORS
# =============================================================================

class C:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    GRAY   = "\033[90m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"

def ok(msg):   print(f"{C.GREEN}  ✓ {msg}{C.RESET}")
def warn(msg): print(f"{C.YELLOW}  ⚠ {msg}{C.RESET}")
def info(msg): print(f"{C.CYAN}  {msg}{C.RESET}")
def skip(msg): print(f"{C.GRAY}  – {msg}{C.RESET}")
def err(msg):  print(f"{C.RED}  ✗ {msg}{C.RESET}")

# =============================================================================
# STATE MANAGEMENT
# =============================================================================

def load_state() -> dict:
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        # Backfill per-strategy balances for states created before this feature
        if "s1_balance" not in s:
            s["s1_balance"] = S1_STARTING
            s["s2_balance"] = S2_STARTING
            s["s3_balance"] = S3_STARTING
        return s
    except FileNotFoundError:
        return {
            "balance":          STARTING_BALANCE,
            "s1_balance":       S1_STARTING,
            "s2_balance":       S2_STARTING,
            "s3_balance":       S3_STARTING,
            "starting_balance": STARTING_BALANCE,
            "positions":        {},
            "trades":           [],
            "total_trades":     0,
            "wins":             0,
            "losses":           0,
            "peak_balance":     STARTING_BALANCE,
        }

def save_state(state: dict):
    # Keep overall balance in sync with per-strategy balances
    state["balance"] = round(
        state["s1_balance"] + state["s2_balance"] + state["s3_balance"], 2
    )
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

def reset_state():
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    print(
        f"{C.GREEN}  ✓ Reset — total: ${STARTING_BALANCE:,.2f}  "
        f"(S1=${S1_STARTING:.0f}  S2=${S2_STARTING:.0f}  S3=${S3_STARTING:.0f}){C.RESET}"
    )

def _strategy_balance_key(strategy: str) -> str:
    if strategy.startswith("S1"):
        return "s1_balance"
    if strategy.startswith("S2"):
        return "s2_balance"
    return "s3_balance"

# =============================================================================
# WEATHER: NWS LIVE FORECAST  (US cities, from bot_v1.py)
# =============================================================================

def get_nws_forecast(city_slug: str) -> dict[str, int]:
    """Return {date_str: max_temp_f} for next ~4 days using NWS hourly + station obs."""
    forecast_url = NWS_ENDPOINTS.get(city_slug)
    station_id   = STATION_IDS.get(city_slug)
    daily_max: dict[str, int] = {}
    headers = {"User-Agent": "paper-trader/1.0"}

    if station_id:
        try:
            r = requests.get(
                f"https://api.weather.gov/stations/{station_id}/observations?limit=48",
                timeout=10, headers=headers,
            )
            for obs in r.json().get("features", []):
                props = obs["properties"]
                date  = props.get("timestamp", "")[:10]
                tc    = props.get("temperature", {}).get("value")
                if tc is not None:
                    tf = round(tc * 9 / 5 + 32)
                    if date not in daily_max or tf > daily_max[date]:
                        daily_max[date] = tf
        except Exception as e:
            warn(f"NWS obs error ({city_slug}): {e}")

    if forecast_url:
        try:
            r = requests.get(forecast_url, timeout=10, headers=headers)
            for p in r.json()["properties"]["periods"]:
                date = p["startTime"][:10]
                temp = p["temperature"]
                if p.get("temperatureUnit") == "C":
                    temp = round(temp * 9 / 5 + 32)
                if date not in daily_max or temp > daily_max[date]:
                    daily_max[date] = temp
        except Exception as e:
            warn(f"NWS forecast error ({city_slug}): {e}")

    return daily_max

# =============================================================================
# WEATHER: OPEN-METEO  (historical + international, free, no key)
# =============================================================================

def get_openmeteo_temps(lat: float, lon: float, tz: str,
                        start_date: str, end_date: str) -> dict[str, int]:
    """
    Fetch daily max temps via Open-Meteo.
    For historical dates use archive-api; for future dates use forecast api.
    Returns {date_str: max_temp_f}.
    """
    today = datetime.now(timezone.utc).date().isoformat()
    results: dict[str, int] = {}

    if start_date <= today:
        hist_end = min(end_date, today)
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={hist_end}"
            f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
            f"&timezone={tz.replace('/', '%2F')}"
        )
        try:
            r = requests.get(url, timeout=15)
            data = r.json().get("daily", {})
            for d, t in zip(data.get("time", []), data.get("temperature_2m_max", [])):
                if t is not None:
                    results[d] = round(t)
        except Exception as e:
            warn(f"Open-Meteo historical error: {e}")

    if end_date > today:
        fcast_start = max(start_date, today)
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={fcast_start}&end_date={end_date}"
            f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
            f"&timezone={tz.replace('/', '%2F')}"
        )
        try:
            r = requests.get(url, timeout=15)
            data = r.json().get("daily", {})
            for d, t in zip(data.get("time", []), data.get("temperature_2m_max", [])):
                if t is not None:
                    results[d] = round(t)
        except Exception as e:
            warn(f"Open-Meteo forecast error: {e}")

    return results

def get_forecast_for_city(city_slug: str) -> dict[str, int]:
    """Return {date_str: max_temp_f} for next 4 days. Uses NWS for US, Open-Meteo for others."""
    city = CITIES[city_slug]
    today = datetime.now(timezone.utc).date()
    start = today.isoformat()
    end   = (today + timedelta(days=3)).isoformat()

    if city_slug in NWS_ENDPOINTS:
        nws = get_nws_forecast(city_slug)
        if nws:
            return nws

    return get_openmeteo_temps(city["lat"], city["lon"], city["tz"], start, end)

# =============================================================================
# MARKET QUESTION PARSER
# =============================================================================

def parse_market_question(question: str, default_year: Optional[int] = None) -> Optional[dict]:
    """
    Parse a Polymarket temperature bucket market question.

    Handles:
      "Will the highest temperature in Dallas be 72-73°F on March 13?"
      "Will the highest temperature in NYC be 56°F or below on April 6?"
      "Will the highest temperature in London be 71°F or higher on May 23?"
      "Will the highest temperature in London be between 67–68°F on May 23?"

    Returns dict with keys: city_slug, city_name, date_str, low, high, midpoint
    or None if not parseable.
    """
    if not question:
        return None
    q = question.lower().strip()

    if "°f" not in q and "°F" not in question:
        return None
    if not any(kw in q for kw in ["highest temperature", "high temperature", "high temp"]):
        return None

    # --- City ---
    city_slug: Optional[str] = None
    city_name: Optional[str] = None
    for alias, slug in sorted(CITY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in q:
            city_slug = slug
            city_name = CITIES[slug]["name"]
            break
    if city_slug is None:
        return None

    # --- Temperature range --- (handle both hyphen and en-dash)
    q_norm = q.replace("\u2013", "-").replace("\u2014", "-")
    low: Optional[int] = None
    high: Optional[int] = None

    m = re.search(r"(\d+)-(\d+)\s*°f", q_norm)
    if m:
        low, high = int(m.group(1)), int(m.group(2))
    elif re.search(r"(\d+)\s*°f\s+or\s+below", q_norm):
        m2 = re.search(r"(\d+)\s*°f\s+or\s+below", q_norm)
        low, high = -999, int(m2.group(1))
    elif re.search(r"(\d+)\s*°f\s+or\s+(higher|above)", q_norm):
        m2 = re.search(r"(\d+)\s*°f\s+or\s+(higher|above)", q_norm)
        low, high = int(m2.group(1)), 999
    elif re.search(r"(\d+)\s*°f\s+or\s+lower", q_norm):
        m2 = re.search(r"(\d+)\s*°f\s+or\s+lower", q_norm)
        low, high = -999, int(m2.group(1))

    if low is None or high is None:
        return None

    # --- Date ---
    date_str: Optional[str] = None
    year = default_year or datetime.now().year
    for i, month_name in enumerate(MONTHS):
        if month_name[:3] not in q_norm:
            continue
        m_date = re.search(rf"{month_name[:3]}\w*\s+(\d{{1,2}}),?\s*(\d{{4}})?", q_norm)
        if m_date:
            day = int(m_date.group(1))
            if m_date.group(2):
                year = int(m_date.group(2))
            try:
                date_str = datetime(year, i + 1, day).strftime("%Y-%m-%d")
            except ValueError:
                pass
            break

    if date_str is None:
        return None

    if low == -999 and high != 999:
        midpoint = float(high)
    elif high == 999 and low != -999:
        midpoint = float(low)
    else:
        midpoint = (low + high) / 2.0

    return {
        "city_slug": city_slug,
        "city_name": city_name,
        "date_str":  date_str,
        "low":       low,
        "high":      high,
        "midpoint":  midpoint,
    }

# =============================================================================
# ENTRY PRICE MODEL  (backtest only — no live price history in parquet data)
# =============================================================================

def estimate_entry_prices(distance: float, is_correct_band: bool) -> tuple[float, float]:
    """Returns (yes_price, no_price) estimate given distance from actual temp."""
    if is_correct_band:
        return 0.60, 0.40
    if distance <= 5:
        return 0.15, 0.85
    if distance <= 12:
        return 0.05, 0.95
    return 0.025, 0.975

# =============================================================================
# KELLY PROBABILITY MODEL  (S2 lottery tickets)
# =============================================================================

def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """Standard normal CDF using math.erf (no scipy needed)."""
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2.0))))

def pm_eff_price(p: float) -> float:
    """
    Polymarket taker fee-inclusive effective price per share.
      fee = feeRate × p × (p × (1-p))^exponent
      effective = p + fee = p × (1 + feeRate × (p×(1-p))^exponent)
    Fee approaches zero at extremes (p→0 or p→1) and peaks near p=0.5.
    No fee applied on settlement.
    """
    variance = p * (1.0 - p)
    fee = POLYMARKET_FEE_RATE * p * (variance ** POLYMARKET_FEE_EXPONENT)
    return p + fee


def estimate_bucket_prob(low: int, high: int, forecast_temp: float,
                         sigma: float = S2_FORECAST_SIGMA) -> float:
    """
    Estimate the true probability that the day's high temperature lands in
    [low, high]°F, modelling the forecast as N(forecast_temp, sigma^2).

    Uses ±0.5°F bucket expansion to account for integer rounding.
    Open-ended buckets (low=-999 or high=999) are handled as tails.
    """
    if low == -999 and high != 999:
        return _norm_cdf(high + 0.5, forecast_temp, sigma)
    if high == 999 and low != -999:
        return 1.0 - _norm_cdf(low - 0.5, forecast_temp, sigma)
    return _norm_cdf(high + 0.5, forecast_temp, sigma) - _norm_cdf(low - 0.5, forecast_temp, sigma)

# =============================================================================
# POLYMARKET DATA: LOCAL PARQUET LOADER
# =============================================================================

def load_historical_weather_markets(lookback_months: int = 12) -> pd.DataFrame:
    """
    Load all closed temperature bucket markets from local parquet files.
    Returns DataFrame with parsed city, date, temp range, and winner columns.
    """
    if not DATA_DIR.exists():
        err(f"Data directory not found: {DATA_DIR}")
        return pd.DataFrame()

    parquet_files = sorted(DATA_DIR.glob("*.parquet"))
    if not parquet_files:
        err(f"No parquet files in {DATA_DIR}")
        return pd.DataFrame()

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_months * 30.44)
    print(f"  Scanning {len(parquet_files)} parquet files "
          f"(cutoff: {cutoff.date().isoformat()})...")

    dfs = []
    for fpath in parquet_files:
        try:
            df = pd.read_parquet(fpath, columns=[
                "id", "question", "outcomes", "outcome_prices",
                "end_date", "closed",
            ])
            mask = (
                df["closed"].fillna(False)
                & df["question"].str.contains(r"\d+.*°[Ff]", na=False, regex=True)
                & df["question"].str.contains(
                    r"highest temperature|high temp", na=False, case=False, regex=True
                )
            )
            subset = df[mask]
            if not subset.empty:
                dfs.append(subset)
        except Exception as e:
            warn(f"Error reading {fpath.name}: {e}")

    if not dfs:
        warn("No matching markets found in parquet files.")
        return pd.DataFrame()

    all_mkts = pd.concat(dfs, ignore_index=True)
    all_mkts["end_date"] = pd.to_datetime(all_mkts["end_date"], utc=True, errors="coerce")
    all_mkts = all_mkts[all_mkts["end_date"].notna() & (all_mkts["end_date"] >= cutoff)]

    print(f"  {len(all_mkts)} closed weather markets after date filter")

    rows = []
    for _, row in all_mkts.iterrows():
        end_year = row["end_date"].year if pd.notna(row["end_date"]) else None
        parsed = parse_market_question(row["question"], default_year=end_year)
        if not parsed:
            continue
        if parsed["city_slug"] not in CITIES:
            continue

        try:
            prices   = json.loads(row["outcome_prices"])
            outcomes = json.loads(row["outcomes"])
            pf       = [float(x) for x in prices]
            max_p    = max(pf)
            if max_p < 0.85:
                continue
            winner = outcomes[pf.index(max_p)]
        except Exception:
            continue

        rows.append({
            "id":        row["id"],
            "question":  row["question"],
            "end_date":  row["end_date"],
            "winner":    winner,
            **parsed,
        })

    if not rows:
        warn("No parseable markets found.")
        return pd.DataFrame()

    return pd.DataFrame(rows)

# =============================================================================
# POLYMARKET API: GAMMA API  (live markets)
# =============================================================================

def get_live_markets(city_slug: str, date: datetime) -> list[dict]:
    """
    Fetch all active temperature bucket markets for a city/date from the Gamma API.
    Returns list of dicts: {id, question, yes_price, no_price, low, high, midpoint, hours_left}.
    """
    month = MONTHS[date.month - 1]
    day   = date.day
    year  = date.year
    slug_city_variants = CITY_SLUGS.get(city_slug, [city_slug])

    headers = {"User-Agent": "paper-trader/1.0"}
    event   = None

    for slug_city in slug_city_variants:
        slug = f"highest-temperature-in-{slug_city}-on-{month}-{day}-{year}"
        url  = f"https://gamma-api.polymarket.com/events?slug={slug}"
        try:
            r    = requests.get(url, timeout=10, headers=headers)
            data = r.json()
            if data and isinstance(data, list) and data:
                event = data[0]
                break
        except Exception:
            pass

    if not event:
        return []

    markets = []
    for mkt in event.get("markets", []):
        question = mkt.get("question", "")
        parsed   = parse_market_question(question, default_year=year)
        if not parsed:
            continue
        try:
            prices    = json.loads(mkt.get("outcomePrices", "[0.5,0.5]"))
            yes_price = float(prices[0])
            no_price  = float(prices[1])
        except Exception:
            continue

        hours_left = _hours_until(event.get("endDate") or mkt.get("endDate"))
        if hours_left < 1.0:
            continue

        markets.append({
            "id":         mkt.get("id", ""),
            "question":   question,
            "yes_price":  yes_price,
            "no_price":   no_price,
            "low":        parsed["low"],
            "high":       parsed["high"],
            "midpoint":   parsed["midpoint"],
            "hours_left": hours_left,
        })

    return markets

def _hours_until(end_date_str: Optional[str]) -> float:
    try:
        if not end_date_str:
            return 999.0
        dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return max(0.0, (dt - datetime.now(timezone.utc)).total_seconds() / 3600)
    except Exception:
        return 999.0

# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_layered_no(
    markets: list[dict], forecast_temp: int, balance_dict: dict,
) -> list[dict]:
    """
    S1: Buy "No" on all bands that don't contain the forecast temp.
    Each layer = s1_balance / S1_MAX_LAYERS (adapts as bankroll grows/shrinks).
    Only enters if no_price in [S1_NO_MIN, S1_NO_MAX] and distance >= S1_MIN_DIST_F.
    """
    s1_balance = balance_dict.get("s1_balance", S1_STARTING)
    layer_size = s1_balance / S1_MAX_LAYERS

    non_forecast = [m for m in markets
                    if not (m["low"] <= forecast_temp <= m["high"])]
    non_forecast.sort(key=lambda m: abs(m["midpoint"] - forecast_temp))

    signals = []
    for mkt in non_forecast[:S1_MAX_LAYERS]:
        no_price = mkt["no_price"]
        distance = abs(mkt["midpoint"] - forecast_temp)
        if distance < S1_MIN_DIST_F:
            continue
        if not (S1_NO_MIN <= no_price <= S1_NO_MAX):
            continue
        if layer_size < 1.0:
            continue
        eff_no = pm_eff_price(no_price)
        signals.append({
            "strategy":              "S1_layered_no",
            "market_id":             mkt["id"],
            "question":              mkt["question"],
            "outcome":               "No",
            "entry_price":           no_price,
            "effective_entry_price": round(eff_no, 6),
            "size_usd":              round(layer_size, 2),
            "shares":                round(layer_size / eff_no, 2),
            "distance_from_fcst": round(distance, 1),
            "forecast_temp":      forecast_temp,
            "low":                mkt["low"],
            "high":               mkt["high"],
        })
    return signals


def strategy_lottery_tickets(
    markets: list[dict], forecast_temp: int, balance_dict: dict,
) -> list[dict]:
    """
    S2: Buy cheap "Yes" on near-miss bands using Kelly criterion.

    Kelly sizing:
      p_est = normal CDF probability of bucket given N(forecast, sigma^2)
      edge  = p_est − market_yes_price
      f*    = edge / (1 − market_yes_price)   [Kelly fraction for binary bet]
      bet   = S2_KELLY_FRAC × f* × s2_balance  (half-Kelly, capped at S2_MAX_BET_FRAC)

    Only bets when edge >= S2_MIN_EDGE (5%).
    """
    s2_balance = balance_dict.get("s2_balance", S2_STARTING)
    signals = []

    for mkt in markets:
        yes_price = mkt["yes_price"]
        if yes_price < S2_YES_MIN or yes_price >= S2_YES_MAX:
            continue
        distance = abs(mkt["midpoint"] - forecast_temp)
        if distance > S2_WINDOW_F or (mkt["low"] <= forecast_temp <= mkt["high"]):
            continue

        # Kelly: estimate true probability from weather model, fee-adjusted
        p_est   = estimate_bucket_prob(mkt["low"], mkt["high"], float(forecast_temp))
        eff_yes = pm_eff_price(yes_price)   # fee ≈ 0 at p→0; see pm_eff_price()
        edge    = p_est - eff_yes           # edge vs fee-inclusive cost
        if edge < S2_MIN_EDGE:
            continue

        # f* = edge / (1 - eff_yes), then half-Kelly
        kelly_f = edge / max(1.0 - eff_yes, 1e-6)
        bet = S2_KELLY_FRAC * kelly_f * s2_balance
        bet = min(bet, S2_MAX_BET_FRAC * s2_balance)
        bet = max(bet, 1.0)
        if s2_balance < 1.0:
            continue

        signals.append({
            "strategy":              "S2_lottery",
            "market_id":             mkt["id"],
            "question":              mkt["question"],
            "outcome":               "Yes",
            "entry_price":           yes_price,
            "effective_entry_price": round(eff_yes, 6),
            "size_usd":              round(bet, 2),
            "shares":                round(bet / eff_yes, 2),  # shares after fee
            "distance_from_fcst":    round(distance, 1),
            "forecast_temp":         forecast_temp,
            "low":                   mkt["low"],
            "high":                  mkt["high"],
            "p_est":                 round(p_est, 4),
            "edge":                  round(edge, 4),
            "kelly_f":               round(kelly_f, 4),
        })
    return signals


def strategy_high_conviction(
    markets: list[dict], forecast_temp: int, balance_dict: dict,
) -> list[dict]:
    """
    S3: Kelly-sized "No" on bands impossibly far from forecast.

    Kelly sizing (No side):
      p_est    = normal CDF probability that temp lands IN the band (= No loses)
      p_no_win = 1 - p_est  (probability No wins)
      edge     = p_no_win - no_price
      f*       = edge / (1 - no_price)   [Kelly fraction for binary No bet]
      bet      = S3_KELLY_FRAC × f* × s3_balance  (half-Kelly, capped at s3_balance)

    Requires no_price >= S3_NO_MIN, distance >= S3_DIST_F, and edge >= S3_MIN_EDGE (1%).
    Picks the candidate with the highest Kelly-adjusted bet size.
    """
    s3_balance = balance_dict.get("s3_balance", S3_STARTING)
    if s3_balance < 1.0:
        return []

    signals = []
    for m in markets:
        no_price = m["no_price"]
        distance = abs(m["midpoint"] - forecast_temp)
        if no_price < S3_NO_MIN or no_price > S3_NO_MAX or distance < S3_DIST_F:
            continue
        if m["low"] <= forecast_temp <= m["high"]:
            continue

        p_est    = estimate_bucket_prob(m["low"], m["high"], float(forecast_temp))
        p_no_win = 1.0 - p_est
        eff_no   = pm_eff_price(no_price)   # fee ≈ 0 at p→1; see pm_eff_price()
        edge     = p_no_win - eff_no
        if edge <= 0:
            continue  # market overprices No after fee — skip

        # 1% floor: for extreme bands (no_price≈0.998), raw edge is tiny but
        # the trade is still valid high-conviction. Floor ensures proper sizing.
        effective_edge = max(edge, S3_MIN_EDGE)

        denom   = max(1.0 - eff_no, 1e-6)
        kelly_f = effective_edge / denom
        bet     = S3_KELLY_FRAC * kelly_f * s3_balance
        bet     = min(bet, s3_balance)
        bet     = max(bet, 1.0)

        signals.append({
            "strategy":              "S3_high_conviction",
            "market_id":             m["id"],
            "question":              m["question"],
            "outcome":               "No",
            "entry_price":           no_price,
            "effective_entry_price": round(eff_no, 6),
            "size_usd":              round(bet, 2),
            "shares":                round(bet / eff_no, 2),
            "distance_from_fcst":    round(distance, 1),
            "forecast_temp":         forecast_temp,
            "low":                   m["low"],
            "high":                  m["high"],
            "p_est":                 round(p_est, 6),
            "edge":                  round(edge, 4),
            "effective_edge":        round(effective_edge, 4),
            "kelly_f":               round(kelly_f, 4),
        })

    if not signals:
        return []

    # Pick the single best candidate by Kelly-adjusted bet size
    best = max(signals, key=lambda s: s["size_usd"])
    return [best]


def apply_all_strategies(
    markets: list[dict], forecast_temp: int, balance_dict: dict,
) -> list[dict]:
    signals = []
    # S1 suspended — see strategy_layered_no() for implementation
    signals.extend(strategy_lottery_tickets(markets, forecast_temp, balance_dict))
    signals.extend(strategy_high_conviction(markets, forecast_temp, balance_dict))
    return signals

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(lookback_months: int = 12):
    """
    Backtest all 3 strategies on historical closed Polymarket weather markets.
    """
    print(f"\n{C.BOLD}{C.CYAN}═══ BACKTEST MODE  (last {lookback_months} months) ═══{C.RESET}\n")

    df = load_historical_weather_markets(lookback_months)
    if df.empty:
        return

    groups   = list(df.groupby(["city_slug", "date_str"]))
    n_groups = len(groups)
    print(f"  {n_groups} unique city/date combos to simulate\n")

    balance_dict = {
        "s1_balance": S1_STARTING,
        "s2_balance": S2_STARTING,
        "s3_balance": S3_STARTING,
    }
    all_trades: list[dict] = []
    by_strat: dict[str, list[dict]] = {
        "S1_layered_no": [], "S2_lottery": [], "S3_high_conviction": [],
    }

    for i, ((city_slug, date_str), group) in enumerate(groups):
        city = CITIES[city_slug]

        temps = get_openmeteo_temps(
            city["lat"], city["lon"], city["tz"], date_str, date_str
        )
        actual_temp = temps.get(date_str)
        if actual_temp is None:
            skip(f"[{i+1}/{n_groups}] {city_slug} {date_str} — no temp data")
            continue

        markets_sim: list[dict] = []
        for _, row in group.iterrows():
            is_correct = (row["low"] <= actual_temp <= row["high"])
            dist       = abs(row["midpoint"] - actual_temp)
            yes_p, no_p = estimate_entry_prices(dist, is_correct)
            markets_sim.append({
                "id":        str(row["id"]),
                "question":  row["question"],
                "yes_price": yes_p,
                "no_price":  no_p,
                "low":       row["low"],
                "high":      row["high"],
                "midpoint":  row["midpoint"],
                "winner":    row["winner"],
            })

        signals = apply_all_strategies(markets_sim, actual_temp, balance_dict)

        for sig in signals:
            mkt = next((m for m in markets_sim if m["id"] == sig["market_id"]), None)
            if not mkt:
                continue

            won   = mkt["winner"] == sig["outcome"]
            price = sig["entry_price"]
            size  = sig["size_usd"]
            bkey  = _strategy_balance_key(sig["strategy"])

            if won:
                pnl = sig["shares"] * (1.0 - price)
            else:
                pnl = -size

            balance_dict[bkey] = round(balance_dict[bkey] + pnl, 2)

            trade = {
                "strategy":    sig["strategy"],
                "city":        city_slug,
                "date":        date_str,
                "question":    sig["question"][:70],
                "outcome":     sig["outcome"],
                "actual_temp": actual_temp,
                "entry_price": price,
                "size":        size,
                "pnl":         round(pnl, 2),
                "won":         won,
            }
            all_trades.append(trade)
            by_strat[sig["strategy"]].append(trade)

    total_balance = sum(balance_dict.values())
    _print_backtest_summary(all_trades, by_strat, balance_dict, total_balance)

    if all_trades:
        out = Path(__file__).parent / "backtest_results.csv"
        pd.DataFrame(all_trades).to_csv(out, index=False)
        ok(f"Full results → {out}")


def _print_backtest_summary(trades: list[dict], by_strat: dict,
                             balance_dict: dict, final_balance: float):
    print(f"\n{'═' * 60}")
    print(f"{C.BOLD}BACKTEST SUMMARY{C.RESET}")
    print(f"{'═' * 60}\n")

    if not trades:
        warn("No trades were simulated — check city filter / date range.")
        return

    total_pnl = final_balance - STARTING_BALANCE
    ret_pct   = total_pnl / STARTING_BALANCE * 100
    wins      = sum(1 for t in trades if t["won"])
    color     = C.GREEN if total_pnl >= 0 else C.RED

    print(f"  Starting balance:  ${STARTING_BALANCE:>10,.2f}  "
          f"(S1=${S1_STARTING:.0f}  S2=${S2_STARTING:.0f}  S3=${S3_STARTING:.0f})")
    print(f"  Final balance:     {color}${final_balance:>10,.2f}{C.RESET}  "
          f"(S1=${balance_dict['s1_balance']:.2f}  "
          f"S2=${balance_dict['s2_balance']:.2f}  "
          f"S3=${balance_dict['s3_balance']:.2f})")
    print(f"  Total P&L:         {color}{'+'if total_pnl>=0 else ''}${total_pnl:,.2f}  "
          f"({ret_pct:+.1f}%){C.RESET}")
    print(f"  Total trades:      {len(trades)}")
    print(f"  Win rate:          {wins}/{len(trades)}  "
          f"({wins/len(trades)*100:.1f}%)")

    strat_labels = {
        "S1_layered_no":      "S1 — Layered No Hedge",
        "S2_lottery":         "S2 — Lottery Tickets (Kelly)",
        "S3_high_conviction": "S3 — High Conviction",
    }
    print(f"\n{C.BOLD}By Strategy:{C.RESET}")
    for key, label in strat_labels.items():
        ts = by_strat[key]
        if not ts:
            print(f"\n  {C.BOLD}{label}{C.RESET}  {C.GRAY}(no trades){C.RESET}")
            continue
        pnl = sum(t["pnl"] for t in ts)
        wr  = sum(1 for t in ts if t["won"])
        col = C.GREEN if pnl >= 0 else C.RED
        avg = pnl / len(ts)
        print(f"\n  {C.BOLD}{label}{C.RESET}")
        print(f"    Trades:    {len(ts)}")
        print(f"    Win rate:  {wr}/{len(ts)}  ({wr/len(ts)*100:.1f}%)")
        print(f"    P&L:       {col}{'+'if pnl>=0 else ''}${pnl:,.2f}{C.RESET}")
        print(f"    Avg/trade: {'+'if avg>=0 else ''}${avg:.2f}")

    sorted_trades = sorted(trades, key=lambda t: t["pnl"], reverse=True)
    print(f"\n{C.BOLD}Top 5 wins:{C.RESET}")
    for t in sorted_trades[:5]:
        print(f"  {C.GREEN}+${t['pnl']:.2f}{C.RESET}  [{t['strategy']}]  {t['question']}")
    print(f"\n{C.BOLD}Top 5 losses:{C.RESET}")
    for t in sorted_trades[-5:]:
        print(f"  {C.RED}${t['pnl']:.2f}{C.RESET}  [{t['strategy']}]  {t['question']}")

    print(f"\n{C.GRAY}  ⚠  Entry prices are ESTIMATED (see ENTRY PRICE MODEL).{C.RESET}")

# =============================================================================
# FORWARD PAPER TRADING ENGINE
# =============================================================================

def run_forward(execute: bool = False):
    """
    Scan live Polymarket weather markets, apply all 3 strategies,
    and optionally record paper trades.
    """
    print(f"\n{C.BOLD}{C.CYAN}═══ FORWARD PAPER TRADER ═══{C.RESET}")
    state     = load_state()
    positions = state["positions"]
    mode_str  = f"{C.GREEN}EXECUTE{C.RESET}" if execute else f"{C.YELLOW}SCAN ONLY{C.RESET}"

    total_bal  = state["s1_balance"] + state["s2_balance"] + state["s3_balance"]
    start_bal  = state["starting_balance"]
    ret        = (total_bal - start_bal) / start_bal * 100
    ret_col    = C.GREEN if ret >= 0 else C.RED

    print(f"\n  Mode:      {mode_str}")
    print(f"  Balance:   {C.BOLD}${total_bal:,.2f}{C.RESET}  "
          f"(started ${start_bal:,.2f}  {ret_col}{ret:+.1f}%{C.RESET})")
    print(f"  Buckets:   S1=${state['s1_balance']:.2f}  "
          f"S2=${state['s2_balance']:.2f}  "
          f"S3=${state['s3_balance']:.2f}")
    print(f"  Trades:    W:{state['wins']} / L:{state['losses']}  |  "
          f"Open positions: {len(positions)}")

    # --- CHECK EXITS ---
    print(f"\n{C.BOLD}Checking exits...{C.RESET}")
    _check_exits(state, execute)

    # --- SCAN FOR ENTRIES ---
    print(f"\n{C.BOLD}Scanning for signals (next 4 days)...{C.RESET}")
    all_signals: list[dict] = []

    balance_dict = {
        "s1_balance": state["s1_balance"],
        "s2_balance": state["s2_balance"],
        "s3_balance": state["s3_balance"],
    }

    for city_slug, city_info in CITIES.items():
        print(f"\n  {C.BOLD}{city_info['name']}{C.RESET}")

        forecast = get_forecast_for_city(city_slug)
        if not forecast:
            skip("No forecast available")
            continue

        for days_ahead in range(0, 4):
            target_date = datetime.now() + timedelta(days=days_ahead)
            date_str    = target_date.strftime("%Y-%m-%d")
            fcst_temp   = forecast.get(date_str)

            if fcst_temp is None:
                continue

            live_markets = get_live_markets(city_slug, target_date)
            if not live_markets:
                skip(f"{date_str}: no active markets found via Gamma API")
                continue

            info(f"{date_str}: forecast {fcst_temp}°F | {len(live_markets)} buckets")

            signals = apply_all_strategies(live_markets, fcst_temp, balance_dict)
            for sig in signals:
                mid = sig["market_id"]
                if mid in positions:
                    skip(f"Already open: {sig['question'][:55]}...")
                    continue

                label   = {"S1_layered_no": "S1", "S2_lottery": "S2",
                           "S3_high_conviction": "S3"}[sig["strategy"]]
                bkey    = _strategy_balance_key(sig["strategy"])
                avail   = balance_dict[bkey]

                # Show Kelly details for S2 and S3
                kelly_info = ""
                if sig["strategy"] in ("S2_lottery", "S3_high_conviction"):
                    kelly_info = (
                        f"  p_est={sig.get('p_est',0):.2%}  "
                        f"edge={sig.get('edge',0):.2%}  "
                        f"f*={sig.get('kelly_f',0):.3f}"
                    )

                print(
                    f"\n    {C.GREEN}[{label}] {sig['outcome']} @ "
                    f"{sig['entry_price']:.3f}{C.RESET}  "
                    f"${sig['size_usd']:.2f}  dist={sig['distance_from_fcst']}°F"
                    f"{kelly_info}"
                )
                print(f"    {sig['question'][:70]}")

                all_signals.append(sig)

                if execute:
                    if avail < sig["size_usd"]:
                        skip(f"Insufficient S{label[1]} balance (${avail:.2f})")
                        continue
                    balance_dict[bkey] = round(avail - sig["size_usd"], 2)
                    positions[mid] = {
                        "strategy":              sig["strategy"],
                        "question":              sig["question"],
                        "outcome":               sig["outcome"],
                        "entry_price":           sig["entry_price"],
                        "effective_entry_price": sig.get("effective_entry_price",
                                                         sig["entry_price"]),
                        "shares":                sig["shares"],
                        "cost":                  sig["size_usd"],
                        "city":          city_slug,
                        "date_str":      date_str,
                        "forecast_temp": fcst_temp,
                        "low":           sig["low"],
                        "high":          sig["high"],
                        "opened_at":     datetime.now().isoformat(),
                    }
                    state["total_trades"] += 1
                    state["trades"].append({
                        "type":        "entry",
                        "strategy":    sig["strategy"],
                        "question":    sig["question"],
                        "outcome":     sig["outcome"],
                        "entry_price": sig["entry_price"],
                        "size":        sig["size_usd"],
                        "opened_at":   datetime.now().isoformat(),
                    })
                    ok(f"Position opened — ${sig['size_usd']:.2f} from {label} bankroll "
                       f"(remaining: ${balance_dict[bkey]:.2f})")

    if not all_signals:
        skip("No signals found this run")

    if execute:
        state["s1_balance"] = round(balance_dict["s1_balance"], 2)
        state["s2_balance"] = round(balance_dict["s2_balance"], 2)
        state["s3_balance"] = round(balance_dict["s3_balance"], 2)
        state["positions"]  = positions
        total = state["s1_balance"] + state["s2_balance"] + state["s3_balance"]
        state["peak_balance"] = max(state.get("peak_balance", total), total)
        save_state(state)

    total_now = balance_dict["s1_balance"] + balance_dict["s2_balance"] + balance_dict["s3_balance"]
    print(f"\n{'─' * 50}")
    info(f"Balance:  ${total_now:,.2f}  "
         f"(S1=${balance_dict['s1_balance']:.2f}  "
         f"S2=${balance_dict['s2_balance']:.2f}  "
         f"S3=${balance_dict['s3_balance']:.2f})")
    info(f"Signals:  {len(all_signals)}")
    if not execute:
        print(f"\n  {C.YELLOW}[Pass --live to record trades]{C.RESET}")


def _check_exits(state: dict, execute: bool):
    """Check open positions against live prices; close if resolved or price >= 0.95."""
    positions = state["positions"]
    if not positions:
        skip("No open positions")
        return

    headers  = {"User-Agent": "paper-trader/1.0"}
    to_close: list[str] = []

    for mid, pos in positions.items():
        try:
            r    = requests.get(
                f"https://gamma-api.polymarket.com/markets/{mid}",
                timeout=5, headers=headers,
            )
            data     = r.json()
            prices   = json.loads(data.get("outcomePrices", "[0.5,0.5]"))
            outcomes = json.loads(data.get("outcomes", '["Yes","No"]'))
            idx      = outcomes.index(pos["outcome"]) if pos["outcome"] in outcomes else 0
            curr     = float(prices[idx])
            closed   = data.get("closed", False)
        except Exception:
            continue

        eff_entry = pos.get("effective_entry_price", pos["entry_price"])
        pnl       = (curr - eff_entry) * pos["shares"]
        pnl_str = (f"{C.GREEN}+${pnl:.2f}{C.RESET}" if pnl >= 0
                   else f"{C.RED}-${abs(pnl):.2f}{C.RESET}")

        strat = pos.get("strategy", "S1_layered_no")
        is_s3 = strat.startswith("S3")
        # S3: hold to resolution; only stop out on a 4% price drop (e.g. 0.996 → 0.956)
        # S2: take profit when price >= 0.95
        stop_loss = curr < eff_entry * (1 - S3_STOP_LOSS)
        should_exit = closed or (stop_loss if is_s3 else curr >= 0.95)
        if should_exit:
            bkey  = _strategy_balance_key(strat)
            ok(f"CLOSE {pos['outcome']} @ {curr:.3f}  |  {pnl_str}  |  "
               f"{pos['question'][:50]}")
            if execute:
                proceeds = round(pos["cost"] + pnl, 2)
                state[bkey] = round(state.get(bkey, 0.0) + proceeds, 2)
                if pnl > 0:
                    state["wins"] += 1
                else:
                    state["losses"] += 1
                state["trades"].append({
                    "type":        "exit",
                    "strategy":    strat,
                    "question":    pos["question"],
                    "outcome":     pos["outcome"],
                    "entry_price": pos["entry_price"],
                    "exit_price":  curr,
                    "pnl":         round(pnl, 2),
                    "closed_at":   datetime.now().isoformat(),
                })
                to_close.append(mid)
        else:
            info(f"HOLD {pos['outcome']} @ {curr:.3f}  |  {pnl_str}  |  "
                 f"{pos['question'][:50]}")

    if execute:
        for mid in to_close:
            del positions[mid]

# =============================================================================
# SHOW POSITIONS
# =============================================================================

def show_positions():
    state     = load_state()
    positions = state["positions"]
    print(f"\n{C.BOLD}Open Positions ({len(positions)}){C.RESET}")
    if not positions:
        skip("No open positions")
        return

    headers   = {"User-Agent": "paper-trader/1.0"}
    total_pnl = 0.0

    for mid, pos in positions.items():
        try:
            r    = requests.get(
                f"https://gamma-api.polymarket.com/markets/{mid}",
                timeout=5, headers=headers,
            )
            data     = r.json()
            prices   = json.loads(data.get("outcomePrices", "[0.5,0.5]"))
            outcomes = json.loads(data.get("outcomes", '["Yes","No"]'))
            idx      = outcomes.index(pos["outcome"]) if pos["outcome"] in outcomes else 0
            curr     = float(prices[idx])
        except Exception:
            curr = pos["entry_price"]

        eff_entry = pos.get("effective_entry_price", pos["entry_price"])
        pnl       = (curr - eff_entry) * pos["shares"]
        total_pnl += pnl
        pnl_str   = (f"{C.GREEN}+${pnl:.2f}{C.RESET}" if pnl >= 0
                     else f"{C.RED}-${abs(pnl):.2f}{C.RESET}")
        strat     = pos.get("strategy", "?")

        print(f"\n  [{strat}]  {pos['outcome']}  entry={pos['entry_price']:.3f}"
              f"(eff={eff_entry:.4f})  now={curr:.3f}  |  {pnl_str}")
        print(f"  {pos['question'][:70]}")
        print(f"  Cost: ${pos['cost']:.2f}  |  Shares: {pos['shares']:.1f}  |  "
              f"City: {pos.get('city', '?')}  Date: {pos.get('date_str', '?')}")

    pnl_col = C.GREEN if total_pnl >= 0 else C.RED
    s1b = state.get("s1_balance", 0)
    s2b = state.get("s2_balance", 0)
    s3b = state.get("s3_balance", 0)
    print(f"\n  {'─' * 40}")
    print(f"  Available: S1=${s1b:.2f}  S2=${s2b:.2f}  S3=${s3b:.2f}")
    print(f"  Open PnL:  {pnl_col}{'+'if total_pnl>=0 else ''}${total_pnl:.2f}{C.RESET}")
    print(f"  W/L:       {state['wins']}/{state['losses']}")

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weather Market Paper Trader — ColdMath strategies on Polymarket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python paper_trader.py --backtest\n"
            "  python paper_trader.py --backtest --lookback 6\n"
            "  python paper_trader.py --forward\n"
            "  python paper_trader.py --forward --live\n"
            "  python paper_trader.py --positions\n"
            "  python paper_trader.py --reset\n"
        ),
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--backtest",  action="store_true", help="Backtest on historical parquet data")
    grp.add_argument("--forward",   action="store_true", help="Scan live markets for signals")
    grp.add_argument("--positions", action="store_true", help="Show current open positions")
    grp.add_argument("--reset",     action="store_true", help="Reset portfolio to starting balance")

    parser.add_argument("--live",     action="store_true",
                        help="(with --forward) Record paper trades to state file")
    parser.add_argument("--lookback", type=int, default=12,
                        help="Backtest lookback window in months (default: 12)")

    args = parser.parse_args()

    if args.reset:
        reset_state()
    elif args.positions:
        show_positions()
    elif args.backtest:
        run_backtest(lookback_months=args.lookback)
    elif args.forward:
        run_forward(execute=args.live)

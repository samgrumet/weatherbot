#!/usr/bin/env python3
"""
Weather Trading Bot v2 — Polymarket
Kelly Criterion + Expected Value simulation.

Usage:
    python weather_bot_v2.py           # Paper mode with $1000 virtual balance
    python weather_bot_v2.py --live    # Real trades
    python weather_bot_v2.py --positions
    python weather_bot_v2.py --reset   # Reset simulation balance
"""

import re
import json
import argparse
import requests
from datetime import datetime, timezone, timedelta

# =============================================================================
# CONFIG
# =============================================================================

with open("config.json") as f:
    _cfg = json.load(f)

ENTRY_THRESHOLD   = _cfg.get("entry_threshold", 0.15)
EXIT_THRESHOLD    = _cfg.get("exit_threshold", 0.45)
MAX_TRADES        = _cfg.get("max_trades_per_run", 5)
MIN_HOURS_LEFT    = _cfg.get("min_hours_to_resolution", 2)
PRICE_DROP_SIGNAL = _cfg.get("price_drop_threshold", 0.10)

# Kelly + EV settings
NOAA_ACCURACY     = 0.78        # NOAA forecast accuracy for 1-3 day predictions
KELLY_FRACTION    = 0.25        # Use 1/4 Kelly for safety (full Kelly is too aggressive)
MAX_POSITION_PCT  = 0.10        # Never bet more than 10% of balance on one trade
MIN_EV            = 0.05        # Minimum EV to enter (5 cents per dollar risked)
SIM_BALANCE       = 1000.0      # Starting virtual balance

LOCATIONS = {
    "NYC":     {"lat": 40.77,  "lon": -73.87, "name": "New York City"},
    "Chicago": {"lat": 41.97,  "lon": -87.90, "name": "Chicago"},
    "Seattle": {"lat": 47.45,  "lon": -122.30, "name": "Seattle"},
    "Atlanta": {"lat": 33.64,  "lon": -84.43,  "name": "Atlanta"},
    "Dallas":  {"lat": 32.90,  "lon": -97.04,  "name": "Dallas"},
    "Miami":   {"lat": 25.80,  "lon": -80.29,  "name": "Miami"},
}

ACTIVE_LOCATIONS = _cfg.get("locations", "NYC,Chicago,Seattle,Atlanta,Dallas,Miami").split(",")
ACTIVE_LOCATIONS = [l.strip() for l in ACTIVE_LOCATIONS]
MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]

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

def ok(msg):   print(f"{C.GREEN}  ✅ {msg}{C.RESET}")
def warn(msg): print(f"{C.YELLOW}  ⚠️  {msg}{C.RESET}")
def info(msg): print(f"{C.CYAN}  {msg}{C.RESET}")
def skip(msg): print(f"{C.GRAY}  ⏸️  {msg}{C.RESET}")

# =============================================================================
# KELLY CRITERION + EV
# =============================================================================

def calculate_ev(our_prob: float, market_price: float) -> float:
    """
    Expected Value per $1 risked.
    EV = (our_prob * payout) - (1 - our_prob) * 1
    payout = (1 / market_price) - 1  (net profit per $1 if we win)
    
    Example: our_prob=0.75, price=0.08
    payout = 1/0.08 - 1 = 11.5x
    EV = 0.75 * 11.5 - 0.25 = 8.375 - 0.25 = +$8.12 per $1 risked
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    payout = (1.0 / market_price) - 1.0
    ev = (our_prob * payout) - (1.0 - our_prob)
    return round(ev, 4)


def calculate_kelly(our_prob: float, market_price: float) -> float:
    """
    Kelly Criterion: optimal fraction of bankroll to bet.
    f* = (p * b - q) / b
    where:
        p = our probability of winning
        q = 1 - p (probability of losing)
        b = net odds (payout per $1 bet)
    
    We apply KELLY_FRACTION (0.25) for safety — fractional Kelly.
    Result is capped at MAX_POSITION_PCT (10% of balance).
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 / market_price) - 1.0  # net odds
    p = our_prob
    q = 1.0 - p
    kelly = (p * b - q) / b
    kelly = max(0.0, kelly)                    # never negative
    kelly = kelly * KELLY_FRACTION             # fractional Kelly
    kelly = min(kelly, MAX_POSITION_PCT)       # cap at max position
    return round(kelly, 4)


def calculate_position_size(kelly_fraction: float, balance: float) -> float:
    """Convert Kelly fraction to dollar amount."""
    return round(kelly_fraction * balance, 2)

# =============================================================================
# SIMULATION STATE
# =============================================================================

SIM_FILE = "simulation.json"

def load_sim() -> dict:
    try:
        with open(SIM_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "balance": SIM_BALANCE,
            "starting_balance": SIM_BALANCE,
            "positions": {},
            "trades": [],
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "peak_balance": SIM_BALANCE,
        }

def save_sim(sim: dict):
    with open(SIM_FILE, "w") as f:
        json.dump(sim, f, indent=2)

def reset_sim():
    import os
    if os.path.exists(SIM_FILE):
        os.remove(SIM_FILE)
    if os.path.exists("positions.json"):
        os.remove("positions.json")
    print(f"{C.GREEN}  ✅ Simulation reset — balance back to ${SIM_BALANCE:.2f}{C.RESET}")

# =============================================================================
# OPEN-METEO FORECAST
# =============================================================================

def get_forecast(location: str) -> dict:
    loc = LOCATIONS[location]
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&daily=temperature_2m_max&temperature_unit=fahrenheit&forecast_days=4"
    )
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        result = {}
        for date, temp in zip(data["daily"]["time"], data["daily"]["temperature_2m_max"]):
            result[date] = round(temp, 1)
        return result
    except Exception as e:
        warn(f"Forecast error for {location}: {e}")
        return {}

# =============================================================================
# POLYMARKET API
# =============================================================================

def get_polymarket_event(location_slug: str, month: str, day: int, year: int) -> dict:
    slug = f"highest-temperature-in-{location_slug}-on-{month}-{day}-{year}"
    url = f"https://gamma-api.polymarket.com/events?slug={slug}"
    try:
        r = requests.get(url, timeout=10)
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
    except Exception as e:
        warn(f"Polymarket API error: {e}")
    return None

def get_price_history(market_id: str) -> list:
    url = f"https://clob.polymarket.com/prices-history?market={market_id}&interval=1d&fidelity=60"
    try:
        r = requests.get(url, timeout=10)
        return r.json().get("history", [])
    except Exception:
        return []

# =============================================================================
# PARSING
# =============================================================================

def parse_temp_range(question: str) -> tuple:
    if not question:
        return None
    if "or below" in question.lower():
        m = re.search(r'(\d+)°F or below', question, re.IGNORECASE)
        if m: return (-999, int(m.group(1)))
    if "or higher" in question.lower():
        m = re.search(r'(\d+)°F or higher', question, re.IGNORECASE)
        if m: return (int(m.group(1)), 999)
    m = re.search(r'between (\d+)-(\d+)°F', question, re.IGNORECASE)
    if m: return (int(m.group(1)), int(m.group(2)))
    return None

def temp_in_range(temp: float, rng: tuple) -> bool:
    return rng[0] <= temp <= rng[1]

def hours_until_resolution(event: dict) -> float:
    try:
        end_date = event.get("endDate") or event.get("end_date_iso")
        if not end_date: return 999
        end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        delta = (end_dt - datetime.now(timezone.utc)).total_seconds() / 3600
        return max(0, delta)
    except Exception:
        return 999

def detect_price_drop(history: list) -> dict:
    if not history or len(history) < 2:
        return {"dropped": False, "change": 0}
    recent = history[-1].get("p", 0.5)
    lookback = min(96, len(history) - 1)
    old = history[-lookback].get("p", recent)
    if old == 0: return {"dropped": False, "change": 0}
    change = (recent - old) / old
    return {"dropped": change < -PRICE_DROP_SIGNAL, "change": change}

# =============================================================================
# SHOW POSITIONS
# =============================================================================

def show_positions():
    sim = load_sim()
    positions = sim["positions"]
    print(f"\n{C.BOLD}📊 Open Positions:{C.RESET}")
    if not positions:
        print("  No open positions")
        return

    total_pnl = 0
    for mid, pos in positions.items():
        try:
            url = f"https://gamma-api.polymarket.com/markets/{mid}"
            r = requests.get(url, timeout=5)
            current_price = float(r.json().get("outcomePrices", ["0.5"])[0])
        except Exception:
            current_price = pos["entry_price"]

        pnl = (current_price - pos["entry_price"]) * pos["shares"]
        total_pnl += pnl
        pnl_str = f"{C.GREEN}+${pnl:.2f}{C.RESET}" if pnl >= 0 else f"{C.RED}-${abs(pnl):.2f}{C.RESET}"
        print(f"\n  • {pos['question'][:65]}...")
        print(f"    Entry: ${pos['entry_price']:.3f} | Now: ${current_price:.3f} | "
              f"Shares: {pos['shares']:.1f} | PnL: {pnl_str}")
        print(f"    Kelly used: {pos['kelly_pct']:.1%} | EV: {pos['ev']:.2f} | Cost: ${pos['cost']:.2f}")

    balance_str = f"${sim['balance']:.2f}"
    pnl_color = C.GREEN if total_pnl >= 0 else C.RED
    print(f"\n  Balance:    {balance_str}")
    print(f"  Open PnL:   {pnl_color}{'+'if total_pnl>=0 else ''}{total_pnl:.2f}{C.RESET}")
    print(f"  Total trades: {sim['total_trades']} | W/L: {sim['wins']}/{sim['losses']}")

# =============================================================================
# MAIN STRATEGY
# =============================================================================

def run(dry_run: bool = True):
    print(f"\n{C.BOLD}{C.CYAN}🌤  Weather Trading Bot v2 — Kelly + EV Edition{C.RESET}")
    print("=" * 55)

    sim = load_sim()
    balance = sim["balance"]
    positions = sim["positions"]

    mode = f"{C.YELLOW}PAPER MODE{C.RESET}" if dry_run else f"{C.RED}LIVE MODE{C.RESET}"
    starting = sim["starting_balance"]
    total_return = (balance - starting) / starting * 100
    return_str = f"{C.GREEN}+{total_return:.1f}%{C.RESET}" if total_return >= 0 else f"{C.RED}{total_return:.1f}%{C.RESET}"

    print(f"\n  Mode:            {mode}")
    print(f"  Virtual balance: {C.BOLD}${balance:.2f}{C.RESET} (started ${starting:.2f}, {return_str})")
    print(f"  Kelly fraction:  {KELLY_FRACTION:.0%} of full Kelly")
    print(f"  Max per trade:   {MAX_POSITION_PCT:.0%} of balance")
    print(f"  Min EV:          {MIN_EV:.2f} per $1 risked")
    print(f"  NOAA accuracy:   {NOAA_ACCURACY:.0%}")
    print(f"  Trades W/L:      {sim['wins']}/{sim['losses']}")

    forecast_cache = {}
    trades_executed = 0
    opportunities = 0

    # Check exits
    print(f"\n{C.BOLD}📤 Checking exits...{C.RESET}")
    exits_found = 0
    for mid, pos in list(positions.items()):
        try:
            url = f"https://gamma-api.polymarket.com/markets/{mid}"
            r = requests.get(url, timeout=5)
            current_price = float(r.json().get("outcomePrices", ["0.5"])[0])
        except Exception:
            continue

        if current_price >= EXIT_THRESHOLD:
            exits_found += 1
            pnl = (current_price - pos["entry_price"]) * pos["shares"]
            ok(f"EXIT: {pos['question'][:50]}...")
            info(f"Price ${current_price:.3f} >= exit ${EXIT_THRESHOLD:.2f} | PnL: +${pnl:.2f}")

            if not dry_run:
                balance += pos["cost"] + pnl
                sim["wins"] += 1 if pnl > 0 else 0
                sim["losses"] += 1 if pnl <= 0 else 0
                sim["trades"].append({
                    "type": "exit", "question": pos["question"],
                    "entry_price": pos["entry_price"], "exit_price": current_price,
                    "pnl": round(pnl, 2), "cost": pos["cost"],
                    "closed_at": datetime.now().isoformat(),
                })
                del positions[mid]
                ok(f"Closed position — PnL: {'+'if pnl>=0 else ''}{pnl:.2f}")
            else:
                skip("Paper mode — not selling")

    if exits_found == 0:
        skip("No exit opportunities")

    # Scan entries
    print(f"\n{C.BOLD}🔍 Scanning for entry signals...{C.RESET}")

    for loc_key in ACTIVE_LOCATIONS:
        loc_key = loc_key.strip()
        if loc_key not in LOCATIONS:
            warn(f"Unknown location: {loc_key}")
            continue

        loc_data = LOCATIONS[loc_key]
        loc_slug = loc_key.lower().replace(" ", "-")

        if loc_key not in forecast_cache:
            forecast_cache[loc_key] = get_forecast(loc_key)

        forecast = forecast_cache[loc_key]
        if not forecast:
            continue

        for i in range(0, 3):
            date = datetime.now() + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            month = MONTHS[date.month - 1]
            day = date.day
            year = date.year

            forecast_temp = forecast.get(date_str)
            if forecast_temp is None:
                continue

            event = get_polymarket_event(loc_slug, month, day, year)
            if not event:
                continue

            hours_left = hours_until_resolution(event)

            print(f"\n{C.BOLD}📍 {loc_data['name']} — {date_str}{C.RESET}")
            info(f"Forecast: {forecast_temp}°F | Resolves in: {hours_left:.0f}h")

            if hours_left < MIN_HOURS_LEFT:
                skip(f"Resolves in {hours_left:.0f}h — too soon")
                continue

            # Find matching bucket
            matched = None
            for market in event.get("markets", []):
                question = market.get("question", "")
                rng = parse_temp_range(question)
                if rng and temp_in_range(forecast_temp, rng):
                    try:
                        prices = json.loads(market.get("outcomePrices", "[0.5,0.5]"))
                        yes_price = float(prices[0])
                    except Exception:
                        continue
                    matched = {"market": market, "question": question,
                               "price": yes_price, "range": rng}
                    break

            if not matched:
                skip(f"No bucket found for {forecast_temp}°F")
                continue

            price = matched["price"]
            market_id = matched["market"].get("id", "")
            question = matched["question"]

            info(f"Bucket: {question[:60]}")
            info(f"Market price: ${price:.3f}")

            # Trend check
            history = get_price_history(market_id)
            trend = detect_price_drop(history)
            if trend["dropped"]:
                info(f"📉 Price dropped {abs(trend['change']):.0%} in 24h — stronger signal")

            # ── KELLY + EV CALCULATION ──
            our_prob = NOAA_ACCURACY  # base accuracy

            # Boost if strong trend signal
            if trend["dropped"] and abs(trend["change"]) > 0.20:
                our_prob = min(0.90, our_prob + 0.05)

            ev = calculate_ev(our_prob, price)
            kelly_pct = calculate_kelly(our_prob, price)
            position_size = calculate_position_size(kelly_pct, balance)

            print(f"\n  {C.BOLD}📐 Kelly + EV Analysis:{C.RESET}")
            info(f"  Our probability:  {our_prob:.0%}")
            info(f"  Market implies:   {price:.1%}")
            info(f"  Edge:             {our_prob - price:.1%}")

            ev_color = C.GREEN if ev > 0 else C.RED
            print(f"  {C.CYAN}  EV per $1:        {ev_color}{ev:+.2f}{C.RESET}")
            print(f"  {C.CYAN}  Kelly fraction:   {kelly_pct:.1%} of balance{C.RESET}")
            print(f"  {C.CYAN}  Position size:    ${position_size:.2f}{C.RESET}")

            # Entry checks
            if price >= ENTRY_THRESHOLD:
                skip(f"Price ${price:.3f} above threshold ${ENTRY_THRESHOLD:.2f}")
                continue

            if ev < MIN_EV:
                skip(f"EV {ev:.2f} below minimum {MIN_EV:.2f} — skip")
                continue

            if kelly_pct <= 0:
                skip("Kelly says no edge — skip")
                continue

            opportunities += 1
            ok(f"ENTRY signal! EV={ev:+.2f} | Kelly={kelly_pct:.1%} | Size=${position_size:.2f}")

            if market_id in positions:
                skip("Already in this market")
                continue

            if trades_executed >= MAX_TRADES:
                skip(f"Max trades ({MAX_TRADES}) reached")
                continue

            if position_size < 0.50:
                skip(f"Position size ${position_size:.2f} too small — skip")
                continue

            shares = position_size / price
            info(f"Buying {shares:.1f} shares @ ${price:.3f} = ${position_size:.2f}")

            if not dry_run:
                balance -= position_size
                positions[market_id] = {
                    "question": question,
                    "entry_price": price,
                    "shares": shares,
                    "cost": position_size,
                    "kelly_pct": kelly_pct,
                    "ev": ev,
                    "our_prob": our_prob,
                    "date": date_str,
                    "location": loc_key,
                    "forecast_temp": forecast_temp,
                    "opened_at": datetime.now().isoformat(),
                }
                sim["total_trades"] += 1
                sim["trades"].append({
                    "type": "entry", "question": question,
                    "entry_price": price, "shares": shares,
                    "cost": position_size, "kelly_pct": kelly_pct,
                    "ev": ev, "our_prob": our_prob,
                    "opened_at": datetime.now().isoformat(),
                })
                trades_executed += 1
                ok(f"Bought {shares:.1f} shares — ${position_size:.2f} deducted from balance")
            else:
                skip("Paper mode — not buying")
                trades_executed += 1

    # Save simulation state
    if not dry_run:
        sim["balance"] = round(balance, 2)
        sim["positions"] = positions
        sim["peak_balance"] = max(sim["peak_balance"], balance)
        save_sim(sim)

    # Summary
    print(f"\n{'=' * 55}")
    print(f"{C.BOLD}📊 Summary:{C.RESET}")
    info(f"Opportunities found: {opportunities}")
    info(f"Trades executed:     {trades_executed}")
    info(f"Exits found:         {exits_found}")
    info(f"Balance:             ${balance:.2f}")

    if dry_run:
        print(f"\n  {C.YELLOW}[PAPER MODE — use --live to simulate trades against real prices]{C.RESET}")

# =============================================================================
# LIVE MONITOR — updates prices every N seconds, auto-exits on threshold
# =============================================================================

import time as _time

def monitor(interval: int = 10):
    """
    Background monitor — fetches live prices from Polymarket every N seconds,
    updates PnL in simulation.json so the dashboard stays current.
    Auto-exits positions when price hits EXIT_THRESHOLD.

    Run: python polymarket_weather_bot.py --monitor
    Stop: Ctrl+C
    """
    print(f"\n{C.BOLD}{C.CYAN}📡 Live Monitor — refreshing every {interval}s{C.RESET}")
    print(f"  Dashboard will update automatically")
    print(f"  Auto-exit threshold: ${EXIT_THRESHOLD:.2f}")
    print(f"  Press Ctrl+C to stop\n")

    while True:
        try:
            sim = load_sim()
            positions = sim.get("positions", {})

            if not positions:
                print(f"{C.GRAY}  {_time.strftime('%H:%M:%S')} — No open positions{C.RESET}")
                _time.sleep(interval)
                continue

            total_pnl = 0

            for mid, pos in list(positions.items()):
                # Fetch current price from Polymarket
                try:
                    url = f"https://gamma-api.polymarket.com/markets/{mid}"
                    r = requests.get(url, timeout=5)
                    data = r.json()
                    prices = json.loads(data.get("outcomePrices", "[0.5,0.5]"))
                    current_price = float(prices[0])
                except Exception:
                    current_price = pos.get("current_price", pos["entry_price"])

                pnl = (current_price - pos["entry_price"]) * pos["shares"]
                pos["current_price"] = round(current_price, 4)
                pos["pnl"] = round(pnl, 2)
                total_pnl += pnl

                pnl_str = f"{C.GREEN}+${pnl:.2f}{C.RESET}" if pnl >= 0 else f"{C.RED}-${abs(pnl):.2f}{C.RESET}"
                print(f"  {C.GRAY}{_time.strftime('%H:%M:%S')}{C.RESET}  "
                      f"{pos['question'][:45]}...  "
                      f"${current_price:.3f}  {pnl_str}")

                # Auto-exit if price hit threshold
                if current_price >= EXIT_THRESHOLD:
                    ok(f"AUTO EXIT: {pos['question'][:50]}... PnL: +${pnl:.2f}")
                    sim["balance"] = round(sim["balance"] + pos["cost"] + pnl, 2)
                    sim["wins"] += 1 if pnl > 0 else 0
                    sim["losses"] += 1 if pnl <= 0 else 0
                    sim["trades"].append({
                        "type": "exit",
                        "question": pos["question"],
                        "entry_price": pos["entry_price"],
                        "exit_price": current_price,
                        "pnl": round(pnl, 2),
                        "cost": pos["cost"],
                        "kelly_pct": pos.get("kelly_pct", 0),
                        "ev": pos.get("ev", 0),
                        "location": pos.get("location", ""),
                        "date": pos.get("date", ""),
                        "our_prob": pos.get("our_prob", 0),
                        "closed_at": datetime.now().isoformat(),
                    })
                    del sim["positions"][mid]

            sim["positions"] = {k: v for k, v in sim["positions"].items()}
            sim["peak_balance"] = max(sim.get("peak_balance", sim["balance"]), sim["balance"])

            total_str = f"{C.GREEN}+${total_pnl:.2f}{C.RESET}" if total_pnl >= 0 else f"{C.RED}-${abs(total_pnl):.2f}{C.RESET}"
            print(f"  {'─'*60}")
            print(f"  Open PnL: {total_str}  |  Balance: ${sim['balance']:.2f}  |  "
                  f"Positions: {len(sim['positions'])}\n")

            save_sim(sim)

        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}  Monitor stopped{C.RESET}")
            break
        except Exception as e:
            warn(f"Monitor error: {e}")

        _time.sleep(interval)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weather Trading Bot v2 — Kelly + EV")
    parser.add_argument("--live", action="store_true", help="Execute trades (updates simulation balance)")
    parser.add_argument("--positions", action="store_true", help="Show open positions")
    parser.add_argument("--reset", action="store_true", help="Reset simulation to $1000")
    parser.add_argument("--monitor", action="store_true", help="Live price monitor — updates dashboard every 10s")
    parser.add_argument("--interval", type=int, default=10, help="Monitor refresh interval in seconds (default: 10)")
    args = parser.parse_args()

    if args.reset:
        reset_sim()
    elif args.positions:
        show_positions()
    elif args.monitor:
        monitor(interval=args.interval)
    else:

        run(dry_run=not args.live)

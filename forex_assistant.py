"""
FOREX ASSISTANT  -  watches the market for you while you are at work.

What it does
  1. Gets live price data (free, Yahoo Finance)
  2. Finds setups: liquidity sweep reversal + trend breakout
  3. Backtests each setup on ~2 years of data (real win rate, not a guess)
  4. Checks high-impact news (ForexFactory calendar) and warns / blocks
  5. Sends you an alert on Telegram with entry, stop loss, target, and score

How to use
  pip install yfinance pandas numpy requests
  python forex_assistant.py backtest     # see how setups performed in the past
  python forex_assistant.py scan         # check the market once now
  python forex_assistant.py watch        # keep running, check every 5 minutes
  python forex_assistant.py once         # one check, only new candles (for cloud runner)
  python forex_assistant.py news         # show today's high-impact news

IMPORTANT: this is a decision helper, not a money machine. The "score" is the
historical win rate of that setup, shrunk toward break-even so small samples
don't fool you. Past results do not guarantee future results. Always use a
stop loss and risk a small % per trade.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ----------------------------- SETTINGS -------------------------------------
SYMBOLS = {            # your name : Yahoo symbol
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "XAUUSD": "GC=F",  # gold futures (closest free feed to spot gold)
}
CURRENCIES = {         # which currencies' news affects each symbol
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "XAUUSD": ["USD"],
}
INTERVAL = "1h"        # candle size: "15m" only gives 60 days, "1h" gives 2 years
PERIOD = "730d"
LOOKBACK = 20          # bars used to find liquidity highs/lows
STOP_ATR = 1.5         # stop loss = 1.5 x ATR
RR = 2.0               # target = 2 x stop distance
MAX_BARS = 48          # a trade is closed (at market) if nothing hits in 48 bars
NEWS_BLOCK_MIN = 30    # no new trade within 30 min before/after high-impact news
MIN_SCORE = 50.0       # only alert if historical score >= this %
MIN_TRADES = 30        # need at least this many past trades to trust a setup

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")      # from @BotFather on Telegram
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")  # your chat id (@userinfobot)
CHECK_EVERY_SEC = 300
# -----------------------------------------------------------------------------


def get_data(yahoo_symbol):
    df = yf.Ticker(yahoo_symbol).history(period=PERIOD, interval=INTERVAL)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    return df


def add_indicators(df):
    df = df.copy()
    df["ema50"] = df.close.ewm(span=50, adjust=False).mean()
    df["ema200"] = df.close.ewm(span=200, adjust=False).mean()
    prev_close = df.close.shift()
    tr = pd.concat([df.high - df.low,
                    (df.high - prev_close).abs(),
                    (df.low - prev_close).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=14, adjust=False).mean()
    delta = df.close.diff()
    gain = delta.clip(lower=0).ewm(span=14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    # liquidity levels = highest high / lowest low of the previous N bars
    df["liq_high"] = df.high.shift(1).rolling(LOOKBACK).max()
    df["liq_low"] = df.low.shift(1).rolling(LOOKBACK).min()
    return df


def find_signals(df):
    """Returns two arrays: direction (+1 buy, -1 sell, 0 none) and setup name."""
    c, h, l = df.close, df.high, df.low
    up = c > df.ema200
    down = c < df.ema200

    # 1) Liquidity sweep: price grabs stops above/below recent range, then closes back inside
    sweep_buy = (l < df.liq_low) & (c > df.liq_low) & up
    sweep_sell = (h > df.liq_high) & (c < df.liq_high) & down

    # 2) Trend breakout: close breaks the range in the direction of the trend
    brk_buy = (c > df.liq_high) & (df.ema50 > df.ema200) & (c > df.ema50) & (df.rsi < 75)
    brk_sell = (c < df.liq_low) & (df.ema50 < df.ema200) & (c < df.ema50) & (df.rsi > 25)

    direction = np.zeros(len(df), dtype=int)
    name = np.array([""] * len(df), dtype=object)
    for cond, d, n in [(brk_buy, 1, "Trend breakout"), (brk_sell, -1, "Trend breakout"),
                       (sweep_buy, 1, "Liquidity sweep"), (sweep_sell, -1, "Liquidity sweep")]:
        m = cond.fillna(False).values & (direction == 0)
        direction[m] = d
        name[m] = n
    return direction, name


def simulate(df, i, d):
    """Trade opens at next bar open. Returns result in R (+RR win, -1 loss, partial if timeout)."""
    if i + 1 >= len(df):
        return None
    atr = df.atr.iloc[i]
    entry = df.open.iloc[i + 1]
    risk = STOP_ATR * atr
    if not risk or np.isnan(risk):
        return None
    stop = entry - d * risk
    target = entry + d * risk * RR
    end = min(i + 1 + MAX_BARS, len(df))
    for j in range(i + 1, end):
        hi, lo = df.high.iloc[j], df.low.iloc[j]
        hit_stop = lo <= stop if d == 1 else hi >= stop
        hit_tp = hi >= target if d == 1 else lo <= target
        if hit_stop:           # if both hit in same bar, assume the worst (stop first)
            return -1.0
        if hit_tp:
            return RR
    last = df.close.iloc[end - 1]
    return (last - entry) * d / risk


def backtest(df):
    """Returns {(setup, direction): {n, wins, winrate, expectancy}} plus a per-setup summary."""
    df = add_indicators(df)
    direction, name = find_signals(df)
    results = {}
    for i in range(210, len(df) - 1):
        if direction[i] == 0:
            continue
        r = simulate(df, i, direction[i])
        if r is None:
            continue
        results.setdefault(name[i], []).append(r)
    out = {}
    for setup, rs in results.items():
        rs = np.array(rs)
        n = len(rs)
        wins = int((rs > 0).sum())
        out[setup] = {
            "n": n,
            "winrate": wins / n * 100,
            "expectancy": rs.mean(),            # average R per trade
            "score": shrunk_winrate(wins, n),
        }
    return out


def shrunk_winrate(wins, n, k=20):
    """Pull small samples toward break-even so 3 lucky trades don't look like 100%."""
    p0 = 1 / (1 + RR)
    return (wins + k * p0) / (n + k) * 100


# ----------------------------- NEWS ------------------------------------------
def get_news():
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    try:
        data = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"}).json()
    except Exception as e:
        print("News feed problem:", e)
        return []
    events = []
    for e in data:
        if e.get("impact") != "High":
            continue
        try:
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
        except Exception:
            continue
        events.append({"time": t, "ccy": e.get("country", ""), "title": e.get("title", ""),
                       "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
    return sorted(events, key=lambda x: x["time"])


def news_near(symbol, events, now=None):
    now = now or datetime.now(timezone.utc)
    ccys = CURRENCIES[symbol]
    hit = [e for e in events if e["ccy"] in ccys
           and abs((e["time"] - now).total_seconds()) <= NEWS_BLOCK_MIN * 60]
    nxt = [e for e in events if e["ccy"] in ccys and now < e["time"] <= now + timedelta(hours=12)]
    return hit, nxt


def show_news():
    now = datetime.now(timezone.utc)
    print("High-impact news (your PC time):")
    for e in get_news():
        if e["time"] >= now - timedelta(hours=2):
            print(f"  {e['time'].astimezone():%a %H:%M}  {e['ccy']}  {e['title']}  "
                  f"(forecast {e['forecast']}, previous {e['previous']})")


# ----------------------------- ALERTS ----------------------------------------
def send(msg):
    print("\n" + msg + "\n")
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                          data={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=15)
        except Exception as e:
            print("Telegram problem:", e)


_last_alert = {}


def scan(stats_cache, events, fresh_minutes=None):
    for sym, ysym in SYMBOLS.items():
        try:
            df = add_indicators(get_data(ysym)).iloc[:-1]   # drop the still-forming candle
        except Exception as e:
            print(sym, "data problem:", e)
            continue
        direction, name = find_signals(df)
        d, setup = direction[-1], name[-1]
        if d == 0:
            print(f"{sym}: no setup now")
            continue
        bar_time = df.index[-1]
        if fresh_minutes is not None:      # cloud mode: only alert for a candle that just closed
            bar_close = bar_time.tz_convert("UTC") + pd.Timedelta(INTERVAL)
            age_min = (datetime.now(timezone.utc) - bar_close).total_seconds() / 60
            if age_min > fresh_minutes:
                print(f"{sym}: {setup} is old ({age_min:.0f} min), skipped")
                continue
        if _last_alert.get((sym, setup)) == bar_time:
            continue
        stats = stats_cache.get(sym, {}).get(setup)
        if not stats or stats["n"] < MIN_TRADES:
            print(f"{sym}: {setup} found but not enough history to trust it")
            continue
        if stats["score"] < MIN_SCORE:
            print(f"{sym}: {setup} found but score {stats['score']:.0f}% is below {MIN_SCORE:.0f}%")
            continue

        last = df.iloc[-1]
        entry = last.close
        risk = STOP_ATR * last.atr
        sl, tp = entry - d * risk, entry + d * risk * RR
        hit, nxt = news_near(sym, events)
        side = "BUY" if d == 1 else "SELL"
        liq = "above" if (d == -1 and setup == "Liquidity sweep") else "below" if setup == "Liquidity sweep" else ""
        lines = [
            f"{side} {sym}  |  {setup}",
            f"Score: {stats['score']:.0f}%  (past win rate {stats['winrate']:.0f}% in {stats['n']} trades, "
            f"avg {stats['expectancy']:+.2f}R per trade)",
            f"Entry ~{entry:.5g}   Stop {sl:.5g}   Target {tp:.5g}   (RR 1:{RR:g})",
            f"Trend: {'UP' if last.close > last.ema200 else 'DOWN'}   RSI {last.rsi:.0f}",
        ]
        if liq:
            lines.append(f"Stops were swept {liq} the last {LOOKBACK}-bar range.")
        if hit:
            lines.append("NEWS WARNING: " + ", ".join(f"{e['ccy']} {e['title']}" for e in hit)
                         + " - high risk, better to SKIP.")
        elif nxt:
            e = nxt[0]
            lines.append(f"Next news: {e['ccy']} {e['title']} at {e['time'].astimezone():%H:%M}")
        lines.append("Not financial advice. Risk small. Always use the stop.")
        send("\n".join(lines))
        _last_alert[(sym, setup)] = bar_time


def build_stats():
    cache = {}
    for sym, ysym in SYMBOLS.items():
        try:
            cache[sym] = backtest(get_data(ysym))
        except Exception as e:
            print(sym, "backtest problem:", e)
    return cache


def print_stats(cache):
    print(f"\nBACKTEST  (timeframe {INTERVAL}, stop {STOP_ATR} ATR, target RR 1:{RR:g})")
    print(f"{'Symbol':8}{'Setup':18}{'Trades':>7}{'Win%':>7}{'Score%':>8}{'Avg R':>8}")
    for sym, d in cache.items():
        for setup, s in d.items():
            print(f"{sym:8}{setup:18}{s['n']:>7}{s['winrate']:>7.1f}{s['score']:>8.1f}{s['expectancy']:>8.2f}")
    print("\nAvg R above 0 = profitable on past data. Below 0 = losing, do not trade it.\n")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    if mode == "news":
        show_news()
    elif mode == "backtest":
        print_stats(build_stats())
    elif mode == "scan":
        cache = build_stats()
        scan(cache, get_news())
    elif mode == "once":               # used by the free cloud runner (GitHub Actions)
        if os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch":   # you pressed the Run button
            send("Forex bot is working. You will get a message here when a good setup appears.")
        scan(build_stats(), get_news(), fresh_minutes=50)
    elif mode == "watch":
        cache = build_stats()
        print_stats(cache)
        last_refresh = time.time()
        while True:
            if time.time() - last_refresh > 24 * 3600:     # refresh stats daily
                cache, last_refresh = build_stats(), time.time()
            scan(cache, get_news())
            time.sleep(CHECK_EVERY_SEC)
    else:
        print(__doc__)

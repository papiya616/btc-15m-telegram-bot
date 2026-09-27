
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ================= SETTINGS =================

DAYS = 183
TRAIN_DAYS = 122

GRANULARITY = 900
CHUNK_CANDLES = 250
LOOKBACK = 100

ATR_TP = 1.5
ATR_SL = 1.0
MAX_HOLD = 12

COOLDOWN = 12
MIN_TRAIN_SIGNALS = 50

# Estimated fee + slippage per side
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002
ROUND_TRIP_COST = 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)

BASE_URL = (
    "https://api.exchange.coinbase.com/"
    "products/BTC-USD/candles"
)


# ================= DOWNLOAD DATA =================

def download_data():
    print("BTC 6-Month Train + Validation Backtest")
    print("Downloading BTC 15M candles...")

    end_ts = int(
        datetime.now(timezone.utc).timestamp() // GRANULARITY
    ) * GRANULARITY

    start_ts = end_ts - DAYS * 86400
    current = start_ts
    all_rows = []

    session = requests.Session()
    session.headers.update({
        "User-Agent": "BTC-15M-Backtest/1.0"
    })

    while current < end_ts:
        chunk_end = min(
            current + GRANULARITY * CHUNK_CANDLES,
            end_ts
        )

        start_iso = datetime.fromtimestamp(
            current, timezone.utc
        ).isoformat(timespec="seconds")

        end_iso = datetime.fromtimestamp(
            chunk_end, timezone.utc
        ).isoformat(timespec="seconds")

        print(
            "Downloading:",
            start_iso[:10],
            "to",
            end_iso[:10]
        )

        params = {
            "start": start_iso,
            "end": end_iso,
            "granularity": GRANULARITY
        }

        response = None

        for attempt in range(3):
            try:
                response = session.get(
                    BASE_URL,
                    params=params,
                    timeout=30
                )
                response.raise_for_status()
                break
            except Exception as error:
                print(
                    f"Download attempt {attempt + 1} failed:",
                    error
                )
                if attempt == 2:
                    raise RuntimeError(
                        "Could not download BTC data. "
                        "Please run the workflow again later."
                    )
                time.sleep(2)

        rows = response.json()

        if not isinstance(rows, list):
            raise RuntimeError(
                "Unexpected Coinbase API response."
            )

        all_rows.extend(rows)

        # IMPORTANT: seconds, not minutes
        current = chunk_end
        time.sleep(0.15)

    if not all_rows:
        raise RuntimeError("No BTC data downloaded.")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "time", "low", "high",
            "open", "close", "volume"
        ]
    )

    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["time"] = pd.to_datetime(
        df["time"], unit="s", utc=True
    )

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print()
    print(f"Total candles downloaded: {len(df):,}")
    print(f"First candle: {df['time'].iloc[0]}")
    print(f"Last candle:  {df['time'].iloc[-1]}")
    print()

    return df


# ================= INDICATORS =================

def calculate_indicators(df):
    df = df.copy()

    df["ema9"] = df["close"].ewm(
        span=9, adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21, adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50, adjust=False
    ).mean()

    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14, adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = (
        100 - 100 / (1 + rs)
    ).fillna(50)

    ema12 = df["close"].ewm(
        span=12, adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26, adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9, adjust=False
    ).mean()

    previous_close = df["close"].shift(1)

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs()
    ], axis=1).max(axis=1)

    df["atr"] = tr.rolling(14).mean()

    df["volume_avg"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"].replace(0, np.nan)
    ).fillna(1)

    # Use only candles before the current candle
    previous_high = (
        df["high"].shift(1).rolling(20).max()
    )

    previous_low = (
        df["low"].shift(1).rolling(20).min()
    )

    df["breakout_up"] = df["close"] > previous_high
    df["breakout_down"] = df["close"] < previous_low

    df["trend_up"] = (
        (df["ema9"] > df["ema21"]) &
        (df["ema21"] > df["ema50"])
    )

    df["trend_down"] = (
        (df["ema9"] < df["ema21"]) &
        (df["ema21"] < df["ema50"])
    )

    return df


# ================= MARKET PATTERN =================

def get_pattern(row):
    if row["trend_up"]:
        trend = "TREND_UP"
    elif row["trend_down"]:
        trend = "TREND_DOWN"
    else:
        trend = "SIDEWAYS"

    if row["rsi"] >= 60:
        rsi_state = "RSI_STRONG"
    elif row["rsi"] <= 40:
        rsi_state = "RSI_WEAK"
    else:
        rsi_state = "RSI_NEUTRAL"

    macd_state = (
        "MACD_UP"
        if row["macd"] > row["macd_signal"]
        else "MACD_DOWN"
    )

    if row["volume_ratio"] >= 1.3:
        volume_state = "VOLUME_STRONG"
    elif row["volume_ratio"] <= 0.7:
        volume_state = "VOLUME_WEAK"
    else:
        volume_state = "VOLUME_NORMAL"

    if row["breakout_up"]:
        breakout = "BREAKOUT_UP"
    elif row["breakout_down"]:
        breakout = "BREAKOUT_DOWN"
    else:
        breakout = "NO_BREAKOUT"

    return (
        f"{trend}|{rsi_state}|{macd_state}|"
        f"{volume_state}|{breakout}"
    )


# ================= SIGNAL =================

def get_signal(row):
    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return "WAIT"

    buy_score = 0
    sell_score = 0

    if row["ema9"] > row["ema21"]:
        buy_score += 2
    elif row["ema9"] < row["ema21"]:
        sell_score += 2

    if row["trend_up"]:
        buy_score += 2
    elif row["trend_down"]:
        sell_score += 2

    if 50 < row["rsi"] < 70:
        buy_score += 1
    elif 30 < row["rsi"] < 50:
        sell_score += 1

    if row["macd"] > row["macd_signal"]:
        buy_score += 2
    elif row["macd"] < row["macd_signal"]:
        sell_score += 2

    if row["volume_ratio"] >= 1.3:
        if buy_score > sell_score:
            buy_score += 1
        elif sell_score > buy_score:
            sell_score += 1

    if row["breakout_up"]:
        buy_score += 2

    if row["breakout_down"]:
        sell_score += 2

    if not row["trend_up"] and not row["trend_down"]:
        return "WAIT"

    if buy_score >= 6 and buy_score > sell_score:
        return "BUY"

    if sell_score >= 6 and sell_score > buy_score:
        return "SELL"

    return "WAIT"


# ================= TEST ONE TRADE =================

def test_trade(df, index, signal):
    row = df.iloc[index]

    entry = float(row["close"])
    atr = float(row["atr"])

    if not np.isfinite(atr) or atr <= 0 or entry <= 0:
        return None

    risk_fraction = ATR_SL * atr / entry

    if signal == "BUY":
        tp = entry + ATR_TP * atr
        sl = entry - ATR_SL * atr
    else:
        tp = entry - ATR_TP * atr
        sl = entry + ATR_SL * atr

    last_index = min(index + MAX_HOLD, len(df) - 1)

    for j in range(index + 1, last_index + 1):
        candle = df.iloc[j]

        if signal == "BUY":
            hit_tp = candle["high"] >= tp
            hit_sl = candle["low"] <= sl
        else:
            hit_tp = candle["low"] <= tp
            hit_sl = candle["high"] >= sl

        # Conservative: if both hit in one candle,
        # count the stop-loss first.
        if hit_tp and hit_sl:
            gross_return = -risk_fraction
            result = "SL"
            exit_index = j
            break

        if hit_sl:
            gross_return = -risk_fraction
            result = "SL"
            exit_index = j
            break

        if hit_tp:
            gross_return = ATR_TP * atr / entry
            result = "TP"
            exit_index = j
            break
    else:
        exit_index = last_index
        exit_price = float(df.iloc[exit_index]["close"])

        if signal == "BUY":
            gross_return = (exit_price - entry) / entry
        else:
            gross_return = (entry - exit_price) / entry

        result = "TIMEOUT"

    net_return = gross_return - ROUND_TRIP_COST
    net_r = net_return / risk_fraction

    return {
        "result": result,
        "r": net_r,
        "exit_index": exit_index
    }


# ================= TRAINING =================

def training_phase(df, start, end):
    print("=" * 65)
    print("TRAINING: First approximately 4 months")
    print("=" * 65)

    stats = defaultdict(lambda: {
        "signals": 0,
        "tp": 0,
        "sl": 0,
        "timeout": 0,
        "total_r": 0.0
    })

    last_exit = -999999

    for i in range(max(LOOKBACK, start), end):
        row = df.iloc[i]
        signal = get_signal(row)

        if signal == "WAIT":
            continue

        # One position at a time, then cooldown
        if i - last_exit <= COOLDOWN:
            continue

        trade = test_trade(df, i, signal)

        if trade is None:
            continue

        key = (signal, get_pattern(row))
        s = stats[key]

        s["signals"] += 1
        s[trade["result"].lower()] += 1
        s["total_r"] += trade["r"]

        last_exit = trade["exit_index"]

    selected = {}

    print("\nPatterns passing training filter:\n")

    for key, s in stats.items():
        n = s["signals"]

        if n < MIN_TRAIN_SIGNALS:
            continue

        avg_r = s["total_r"] / n

        if avg_r > 0:
            selected[key] = {
                "signals": n,
                "avg_r": avg_r,
                "tp_rate": s["tp"] / n * 100,
                "sl_rate": s["sl"] / n * 100
            }

            signal, pattern = key

            print(
                f"{signal} | n={n} | "
                f"TP={s['tp']/n*100:.1f}% | "
                f"SL={s['sl']/n*100:.1f}% | "
                f"Avg R={avg_r:+.3f} | {pattern}"
            )

    print(f"\nSelected patterns: {len(selected)}\n")

    return selected


# ================= VALIDATION =================

def validation_phase(df, start, end, selected):
    print("=" * 65)
    print("VALIDATION: Last approximately 2 months")
    print("=" * 65)

    stats = {
        "BUY": {
            "signals": 0, "tp": 0, "sl": 0,
            "timeout": 0, "total_r": 0.0
        },
        "SELL": {
            "signals": 0, "tp": 0, "sl": 0,
            "timeout": 0, "total_r": 0.0
        }
    }

    last_exit = -999999

    for i in range(max(LOOKBACK, start), end):
        row = df.iloc[i]
        signal = get_signal(row)

        if signal == "WAIT":
            continue

        pattern = get_pattern(row)

        # Only use patterns selected from training data
        if (signal, pattern) not in selected:
            continue

        if i - last_exit <= COOLDOWN:
            continue

        trade = test_trade(df, i, signal)

        if trade is None:
            continue

        s = stats[signal]
        s["signals"] += 1
        s[trade["result"].lower()] += 1
        s["total_r"] += trade["r"]

        last_exit = trade["exit_index"]

    total_trades = 0
    total_r = 0.0

    print()

    for signal in ["BUY", "SELL"]:
        s = stats[signal]
        n = s["signals"]

        total_trades += n
        total_r += s["total_r"]

        if n:
            tp_pct = s["tp"] / n * 100
            sl_pct = s["sl"] / n * 100
            timeout_pct = s["timeout"] / n * 100
            avg_r = s["total_r"] / n
        else:
            tp_pct = sl_pct = timeout_pct = avg_r = 0

        print(
            f"{signal}: {n} trades | "
            f"TP {s['tp']} ({tp_pct:.1f}%) | "
            f"SL {s['sl']} ({sl_pct:.1f}%) | "
            f"Timeout {s['timeout']} ({timeout_pct:.1f}%) | "
            f"Total R {s['total_r']:+.2f} | "
            f"Average R {avg_r:+.3f}"
        )

    print("\n" + "-" * 65)
    print(f"VALIDATION TOTAL TRADES: {total_trades}")
    print(f"VALIDATION TOTAL R: {total_r:+.2f}")

    if total_trades:
        print(f"AVERAGE R PER TRADE: {total_r / total_trades:+.3f}")

    print()

    return total_trades, total_r


# ================= MAIN =================

def main():
    df = download_data()
    df = calculate_indicators(df)

    total = len(df)

    if total < 500:
        raise RuntimeError(
            f"Not enough candles downloaded: {total}"
        )

    # Chronological split: approximately 4 months / 2 months
    train_end = int(total * TRAIN_DAYS / DAYS)
    validation_start = train_end

    print(f"Total candles: {total:,}")
    print(f"Training range: 0 to {train_end:,}")
    print(
        f"Validation range: {validation_start:,} "
        f"to {total:,}"
    )
    print(f"TP: {ATR_TP} x ATR")
    print(f"SL: {ATR_SL} x ATR")
    print(f"Maximum holding: {MAX_HOLD * 15} minutes")
    print(f"Round-trip estimated cost: {ROUND_TRIP_COST*100:.2f}%")
    print()

    selected = training_phase(
        df, LOOKBACK, train_end
    )

    total_trades, total_r = validation_phase(
        df, validation_start, total, selected
    )

    print("=" * 65)
    print("FINAL VALIDATION RESULT")
    print("=" * 65)

    if total_trades == 0:
        print("No validation trades passed the training filter.")
    elif total_r > 0:
        print("Validation total R is positive.")
        print("This is historical evidence, not a guarantee of profit.")
    else:
        print("Validation total R is not positive.")
        print("Do not switch the live bot to this strategy yet.")

    print("Backtest finished.")


if __name__ == "__main__":
    main()

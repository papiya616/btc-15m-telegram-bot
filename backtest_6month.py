import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ============================================================
# SETTINGS
# ============================================================

DAYS = 183

TRAIN_DAYS = 122       # প্রায় 4 মাস
VALIDATION_DAYS = 61    # প্রায় 2 মাস

GRANULARITY = 900       # 15 minutes
CHUNK_CANDLES = 250

LOOKBACK = 100

ATR_TP = 1.5
ATR_SL = 1.0

MAX_HOLD_CANDLES = 12  # 3 hours

MIN_TRAIN_SIGNALS = 50

COOLDOWN_CANDLES = 12  # একটি trade শেষ না হওয়া পর্যন্ত নতুন trade নয়

FEE_PER_SIDE = 0.0005      # 0.05%
SLIPPAGE_PER_SIDE = 0.0002  # 0.02%

# ============================================================
# DOWNLOAD BTC 15M DATA
# ============================================================

def download_data():

    print("🚀 BTC 6-Month Train + Validation Backtest Started!")
    print()
    print("📥 Downloading 6 months of BTC 15M data...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current = start_time

    while current < end_time:

        chunk_end = min(
            current + timedelta(minutes=GRANULARITY * CHUNK_CANDLES),
            end_time
        )

        start_iso = current.isoformat()
        end_iso = chunk_end.isoformat()

        print(
            f"Downloading: "
            f"{current.strftime('%Y-%m-%d')} "
            f"to "
            f"{chunk_end.strftime('%Y-%m-%d')}"
        )

        url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

        params = {
            "start": start_iso,
            "end": end_iso,
            "granularity": GRANULARITY
        }

        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()

            data = r.json()

            if isinstance(data, list):
                all_rows.extend(data)

        except Exception as e:
            print("⚠️ Download error:", e)

        current = chunk_end

    if not all_rows:
        raise RuntimeError("❌ No BTC data downloaded.")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "time",
            "low",
            "high",
            "open",
            "close",
            "volume"
        ]
    )

    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print()
    print(f"✅ Total candles: {len(df):,}")
    print(
        f"📅 From: {df['time'].iloc[0]}"
        f"\n📅 To:   {df['time'].iloc[-1]}"
    )
    print()

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].fillna(50)

    # MACD
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # ATR
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # Volume
    df["volume_avg"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"].replace(0, np.nan)
    )

    # Recent support / resistance
    df["support"] = df["low"].rolling(20).min()
    df["resistance"] = df["high"].rolling(20).max()

    # Breakout
    previous_high = df["high"].shift(1).rolling(20).max()
    previous_low = df["low"].shift(1).rolling(20).min()

    df["breakout_up"] = df["close"] > previous_high
    df["breakout_down"] = df["close"] < previous_low

    # Trend
    df["trend_up"] = (
        (df["ema9"] > df["ema21"]) &
        (df["ema21"] > df["ema50"])
    )

    df["trend_down"] = (
        (df["ema9"] < df["ema21"]) &
        (df["ema21"] < df["ema50"])
    )

    return df


# ============================================================
# CREATE MARKET PATTERN
# ============================================================

def get_pattern(row):

    # Trend
    if row["trend_up"]:
        trend = "TREND_UP"

    elif row["trend_down"]:
        trend = "TREND_DOWN"

    else:
        trend = "SIDEWAYS"


    # RSI
    if row["rsi"] >= 60:
        rsi_state = "RSI_STRONG"

    elif row["rsi"] <= 40:
        rsi_state = "RSI_WEAK"

    else:
        rsi_state = "RSI_NEUTRAL"


    # MACD
    if row["macd"] > row["macd_signal"]:
        macd_state = "MACD_UP"

    else:
        macd_state = "MACD_DOWN"


    # Volume
    if row["volume_ratio"] >= 1.3:
        volume_state = "VOLUME_STRONG"

    elif row["volume_ratio"] <= 0.7:
        volume_state = "VOLUME_WEAK"

    else:
        volume_state = "VOLUME_NORMAL"


    # Breakout
    if row["breakout_up"]:
        breakout = "BREAKOUT_UP"

    elif row["breakout_down"]:
        breakout = "BREAKOUT_DOWN"

    else:
        breakout = "NO_BREAKOUT"


    return (
        f"{trend}|"
        f"{rsi_state}|"
        f"{macd_state}|"
        f"{volume_state}|"
        f"{breakout}"
    )


# ============================================================
# SIGNAL
# ============================================================

def get_signal(row):

    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return "WAIT"

    if pd.isna(row["ema50"]):
        return "WAIT"

    buy_score = 0
    sell_score = 0

    # EMA
    if row["ema9"] > row["ema21"]:
        buy_score += 2

    elif row["ema9"] < row["ema21"]:
        sell_score += 2

    # Trend
    if row["trend_up"]:
        buy_score += 2

    elif row["trend_down"]:
        sell_score += 2

    # RSI
    if 50 < row["rsi"] < 70:
        buy_score += 1

    elif 30 < row["rsi"] < 50:
        sell_score += 1

    # MACD
    if row["macd"] > row["macd_signal"]:
        buy_score += 2

    elif row["macd"] < row["macd_signal"]:
        sell_score += 2

    # Volume
    if row["volume_ratio"] >= 1.3:

        if buy_score > sell_score:
            buy_score += 1

        elif sell_score > buy_score:
            sell_score += 1

    # Breakout
    if row["breakout_up"]:
        buy_score += 2

    if row["breakout_down"]:
        sell_score += 2

    # Avoid sideways
    if not row["trend_up"] and not row["trend_down"]:
        return "WAIT"

    # Signal
    if buy_score >= 6 and buy_score > sell_score:
        return "BUY"

    if sell_score >= 6 and sell_score > buy_score:
        return "SELL"

    return "WAIT"


# ============================================================
# TEST ONE TRADE
# ============================================================

def test_trade(df, index, signal):

    entry = df.iloc[index]

    entry_price = entry["close"]
    atr = entry["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if signal == "BUY":

        tp = entry_price + (atr * ATR_TP)
        sl = entry_price - (atr * ATR_SL)

    else:

        tp = entry_price - (atr * ATR_TP)
        sl = entry_price + (atr * ATR_SL)


    end_index = min(
        index + MAX_HOLD_CANDLES,
        len(df) - 1
    )


    for j in range(index + 1, end_index + 1):

        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if signal == "BUY":

            hit_tp = high >= tp
            hit_sl = low <= sl

        else:

            hit_tp = low <= tp
            hit_sl = high >= sl


        # If both happen inside same candle,
        # assume SL first (conservative).
        if hit_tp and hit_sl:

            return {
                "result": "SL",
                "r": -1.0,
                "exit_price": sl,
                "exit_index": j
            }

        if hit_tp:

            return {
                "result": "TP",
                "r": 1.5,
                "exit_price": tp,
                "exit_index": j
            }

        if hit_sl:

            return {
                "result": "SL",
                "r": -1.0,
                "exit_price": sl,
                "exit_index": j
            }


    # Timeout
    exit_price = df.iloc[end_index]["close"]

    if signal == "BUY":
        raw_return = (exit_price - entry_price) / entry_price
    else:
        raw_return = (entry_price - exit_price) / entry_price


    # Approximate trading costs
    cost = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

    net_return = raw_return - cost

    r_value = net_return / (ATR_SL * atr / entry_price)

    return {
        "result": "TIMEOUT",
        "r": r_value,
        "exit_price": exit_price,
        "exit_index": end_index
    }


# ============================================================
# TRAINING
# ============================================================

def training_phase(df, train_start, train_end):

    print("=" * 70)
    print("🧠 TRAINING PHASE")
    print("=" * 70)

    pattern_stats = defaultdict(
        lambda: {
            "signals": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "r_total": 0.0
        }
    )

    last_trade_index = {
        "BUY": -999999,
        "SELL": -999999
    }


    for i in range(
        max(LOOKBACK, train_start),
        train_end
    ):

        row = df.iloc[i]

        signal = get_signal(row)

        if signal == "WAIT":
            continue

        # Cooldown
        if i - last_trade_index[signal] < COOLDOWN_CANDLES:
            continue

        result = test_trade(
            df,
            i,
            signal
        )

        if result is None:
            continue

        pattern = get_pattern(row)

        stats = pattern_stats[
            (signal, pattern)
        ]

        stats["signals"] += 1

        stats[result["result"].lower()] += 1

        stats["r_total"] += result["r"]

        last_trade_index[signal] = result["exit_index"]


    # Select patterns
    selected = {}

    print()
    print("📊 TRAINING PATTERNS")
    print()

    for key, stats in pattern_stats.items():

        signal, pattern = key

        n = stats["signals"]

        if n < MIN_TRAIN_SIGNALS:
            continue

        tp_rate = stats["tp"] / n
        sl_rate = stats["sl"] / n
        avg_r = stats["r_total"] / n

        # Require positive average R
        if avg_r > 0:

            selected[key] = {
                "signals": n,
                "tp_rate": tp_rate,
                "sl_rate": sl_rate,
                "avg_r": avg_r
            }

            print(
                f"{signal:4} | "
                f"{n:4} signals | "
                f"TP {tp_rate*100:5.1f}% | "
                f"SL {sl_rate*100:5.1f}% | "
                f"Avg R {avg_r:+.3f} | "
                f"{pattern}"
            )


    print()
    print(
        f"✅ Selected training patterns: "
        f"{len(selected)}"
    )

    if not selected:
        print(
            "⚠️ No pattern passed the training filter."
        )

    print()

    return selected


# ============================================================
# VALIDATION
# ============================================================

def validation_phase(df, validation_start, validation_end, selected):

    print("=" * 70)
    print("🧪 VALIDATION PHASE")
    print("=" * 70)

    results = {
        "BUY": {
            "signals": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "r": 0.0
        },
        "SELL": {
            "signals": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "r": 0.0
        }
    }


    last_trade_index = {
        "BUY": -999999,
        "SELL": -999999
    }


    trades = []


    for i in range(
        max(LOOKBACK, validation_start),
        validation_end
    ):

        row = df.iloc[i]

        signal = get_signal(row)

        if signal == "WAIT":
            continue

        pattern = get_pattern(row)

        key = (signal, pattern)

        # ONLY patterns learned during training
        if key not in selected:
            continue

        if i - last_trade_index[signal] < COOLDOWN_CANDLES:
            continue

        result = test_trade(
            df,
            i,
            signal
        )

        if result is None:
            continue

        results[signal]["signals"] += 1

        results[signal][
            result["result"].lower()
        ] += 1

        results[signal]["r"] += result["r"]

        trades.append({
            "time": row["time"],
            "signal": signal,
            "pattern": pattern,
            "result": result["result"],
            "r": result["r"]
        })

        last_trade_index[signal] = result["exit_index"]


    print()

    total_r = 0.0
    total_trades = 0

    for signal in ["BUY", "SELL"]:

        s = results[signal]

        n = s["signals"]

        total_r += s["r"]
        total_trades += n

        if n > 0:

            tp_rate = s["tp"] / n * 100
            sl_rate = s["sl"] / n * 100
            timeout_rate = s["timeout"] / n * 100
            avg_r = s["r"] / n

        else:

            tp_rate = 0
            sl_rate = 0
            timeout_rate = 0
            avg_r = 0


        print(
            f"{signal}: "
            f"{n} trades | "
            f"TP {s['tp']} ({tp_rate:.1f}%) | "
            f"SL {s['sl']} ({sl_rate:.1f}%) | "
            f"Timeout {s['timeout']} ({timeout_rate:.1f}%) | "
            f"Total R {s['r']:+.2f} | "
            f"Avg R {avg_r:+.3f}"
        )


    print()
    print("-" * 70)

    print(
        f"📌 VALIDATION TOTAL: "
        f"{total_trades} trades"
    )

    print(
        f"📈 Total R: "
        f"{total_r:+.2f}"
    )

    if total_trades > 0:

        print(
            f"📊 Average R/trade: "
            f"{total_r / total_trades:+.3f}"
        )

    print()

    return results, trades


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_data()

    df = calculate_indicators(df)

    total = len(df)

    # Use chronological split
    train_start = LOOKBACK

    train_end = int(
        total * (TRAIN_DAYS / DAYS)
    )

    validation_start = train_end

    validation_end = total

    print("=" * 70)

    print(
        f"🧠 Training candles: "
        f"{train_start:,} → {train_end:,}"
    )

    print(
        f"🧪 Validation candles: "
        f"{validation_start:,} → {validation_end:,}"
    )

    print()

    print(
        f"🎯 TP = {ATR_TP} × ATR"
    )

    print(
        f"🛑 SL = {ATR_SL} × ATR"
    )

    print(
        f"⏱️ Max holding = "
        f"{MAX_HOLD_CANDLES * 15 / 60:.1f} hours"
    )

    print(
        f"🔒 Minimum training signals/pattern = "
        f"{MIN_TRAIN_SIGNALS}"
    )

    print(
        f"⏳ Cooldown = "
        f"{COOLDOWN_CANDLES * 15} minutes"
    )

    print()


    # -----------------------------
    # TRAIN
    # -----------------------------

    selected = training_phase(
        df,
        train_start,
        train_end
    )


    # -----------------------------
    # VALIDATE
    # -----------------------------

    results, trades = validation_phase(
        df,
        validation_start,
        validation_end,
        selected
    )


    # -----------------------------
    # FINAL VERDICT
    # -----------------------------

    print("=" * 70)
    print("🏁 FINAL VALIDATION RESULT")
    print("=" * 70)

    total_r = sum(
        results[s]["r"]
        for s in ["BUY", "SELL"]
    )

    total_trades = sum(
        results[s]["signals"]
        for s in ["BUY", "SELL"]
    )

    if total_trades == 0:

        print(
            "⚠️ No validation trades."
        )

        print(
            "The training filter was too strict."
        )

    elif total_r > 0:

        print(
            "🟢 Validation Total R is POSITIVE."
        )

        print(
            f"Total R = {total_r:+.2f}"
        )

        print(
            "This does NOT automatically mean "
            "the strategy is profitable live."
        )

        print(
            "Fees, slippage and live execution "
            "still matter."
        )

    else:

        print(
            "🔴 Validation Total R is NEGATIVE."
        )

        print(
            f"Total R = {total_r:+.2f}"
        )

        print(
            "Do NOT change the live bot to this "
            "strategy yet."
        )

    print()

    print("✅ Backtest finished.")


if __name__ == "__main__":
    main()

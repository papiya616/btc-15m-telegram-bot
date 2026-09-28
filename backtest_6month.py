import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M ENTRY-QUALITY BACKTEST
# ============================================================

SYMBOL = "BTC-USD"
GRANULARITY = 900          # 15 minutes
DAYS = 183

CHUNK_CANDLES = 250

TP_ATR = 1.5
SL_ATR = 1.0
MAX_HOLD_CANDLES = 12     # 3 hours

ROUND_TRIP_COST = 0.0014   # 0.14%

COOLDOWN_CANDLES = 12

MIN_TRAIN_TRADES = 20

# Train = first ~4 months
TRAIN_DAYS = 122


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    print("🚀 BTC 6-Month Entry-Quality Backtest Started!")
    print()
    print("📥 Downloading BTC 15M historical data...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current = start_time

    while current < end_time:

        chunk_end = min(
            current + timedelta(minutes=GRANULARITY // 60 * CHUNK_CANDLES),
            end_time
        )

        start_ts = int(current.timestamp())
        end_ts = int(chunk_end.timestamp())

        url = f"https://api.exchange.coinbase.com/products/{SYMBOL}/candles"

        params = {
            "granularity": GRANULARITY,
            "start": start_ts,
            "end": end_ts
        }

        print(
            "Downloading:",
            current.strftime("%Y-%m-%d %H:%M"),
            "to",
            chunk_end.strftime("%Y-%m-%d %H:%M")
        )

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if isinstance(data, list):
                all_rows.extend(data)

        except Exception as e:

            print("Download error:", e)

        current = chunk_end

        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("No BTC data downloaded.")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume"
        ]
    )

    df = df.drop_duplicates("timestamp")

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print()
    print("Total candles downloaded:", len(df))

    print(
        "First candle:",
        df["datetime"].iloc[0]
    )

    print(
        "Last candle:",
        df["datetime"].iloc[-1]
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["ema21"] = df["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    df["ema50"] = df["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    delta = df["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # ATR
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] -
        prev_close
    ).abs()

    tr3 = (
        df["low"] -
        prev_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # Volume average
    df["volume_avg"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    # Recent high/low
    df["recent_high"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["recent_low"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # ATR percentage
    df["atr_pct"] = (
        df["atr"] /
        df["close"]
    )

    # ATR regime
    df["atr_avg"] = (
        df["atr"]
        .rolling(100)
        .mean()
    )

    return df


# ============================================================
# CANDLE / ENTRY QUALITY
# ============================================================

def candle_features(row):

    candle_range = row["high"] - row["low"]

    if candle_range <= 0:
        return {
            "bullish_candle": False,
            "bearish_candle": False,
            "bull_rejection": False,
            "bear_rejection": False
        }

    body = abs(
        row["close"] -
        row["open"]
    )

    upper_wick = (
        row["high"] -
        max(
            row["open"],
            row["close"]
        )
    )

    lower_wick = (
        min(
            row["open"],
            row["close"]
        ) -
        row["low"]
    )

    bullish = (
        row["close"] >
        row["open"]
    )

    bearish = (
        row["close"] <
        row["open"]
    )

    bull_rejection = (
        lower_wick >= body * 1.2
        and
        row["close"] >
        row["low"] + candle_range * 0.55
    )

    bear_rejection = (
        upper_wick >= body * 1.2
        and
        row["close"] <
        row["low"] + candle_range * 0.45
    )

    return {
        "bullish_candle": bullish,
        "bearish_candle": bearish,
        "bull_rejection": bull_rejection,
        "bear_rejection": bear_rejection
    }


# ============================================================
# SIGNAL
# ============================================================

def get_signal(df, i):

    row = df.iloc[i]

    needed = [
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "atr",
        "volume_ratio",
        "recent_high",
        "recent_low",
        "atr_avg"
    ]

    if any(pd.isna(row[x]) for x in needed):
        return "WAIT", None

    price = row["close"]
    atr = row["atr"]

    if atr <= 0:
        return "WAIT", None

    features = candle_features(row)

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    trend_up = (
        row["ema9"] >
        row["ema21"] >
        row["ema50"]
    )

    trend_down = (
        row["ema9"] <
        row["ema21"] <
        row["ema50"]
    )

    if not trend_up and not trend_down:
        return "WAIT", None

    # --------------------------------------------------------
    # PULLBACK DISTANCE
    # --------------------------------------------------------

    distance_ema21 = (
        abs(price - row["ema21"]) /
        atr
    )

    distance_ema50 = (
        abs(price - row["ema50"]) /
        atr
    )

    # Avoid chasing price too far from EMA21
    not_overextended = (
        distance_ema21 <= 1.8
    )

    # --------------------------------------------------------
    # PULLBACK
    # --------------------------------------------------------

    pullback_buy = (
        row["low"] <= row["ema21"] + atr * 0.35
        and
        row["close"] > row["ema21"]
    )

    pullback_sell = (
        row["high"] >= row["ema21"] - atr * 0.35
        and
        row["close"] < row["ema21"]
    )

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    macd_buy = (
        row["macd"] >
        row["macd_signal"]
    )

    macd_sell = (
        row["macd"] <
        row["macd_signal"]
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    rsi_buy = (
        row["rsi"] >= 50
        and
        row["rsi"] <= 68
    )

    rsi_sell = (
        row["rsi"] <= 50
        and
        row["rsi"] >= 32
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    volume_ok = (
        row["volume_ratio"] >= 1.0
    )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    volatility_ok = (
        row["atr"] >= row["atr_avg"] * 0.75
        and
        row["atr"] <= row["atr_avg"] * 2.0
    )

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    breakout_up = (
        price >
        row["recent_high"]
    )

    breakout_down = (
        price <
        row["recent_low"]
    )

    # --------------------------------------------------------
    # BUY SCORE
    # --------------------------------------------------------

    buy_score = 0

    if trend_up:
        buy_score += 2

    if pullback_buy:
        buy_score += 3

    if features["bull_rejection"]:
        buy_score += 2

    if features["bullish_candle"]:
        buy_score += 1

    if macd_buy:
        buy_score += 2

    if rsi_buy:
        buy_score += 1

    if volume_ok:
        buy_score += 1

    if volatility_ok:
        buy_score += 1

    if breakout_up:
        buy_score += 1

    # --------------------------------------------------------
    # SELL SCORE
    # --------------------------------------------------------

    sell_score = 0

    if trend_down:
        sell_score += 2

    if pullback_sell:
        sell_score += 3

    if features["bear_rejection"]:
        sell_score += 2

    if features["bearish_candle"]:
        sell_score += 1

    if macd_sell:
        sell_score += 2

    if rsi_sell:
        sell_score += 1

    if volume_ok:
        sell_score += 1

    if volatility_ok:
        sell_score += 1

    if breakout_down:
        sell_score += 1

    # --------------------------------------------------------
    # ENTRY FILTER
    # --------------------------------------------------------

    if not not_overextended:
        return "WAIT", None

    if not volatility_ok:
        return "WAIT", None

    # BUY
    if (
        trend_up
        and
        pullback_buy
        and
        buy_score >= 8
        and
        buy_score > sell_score
    ):

        pattern = (
            "BUY|"
            f"PULLBACK_{int(pullback_buy)}|"
            f"REJECTION_{int(features['bull_rejection'])}|"
            f"MACD_{int(macd_buy)}|"
            f"RSI_{int(rsi_buy)}|"
            f"VOL_{int(volume_ok)}"
        )

        return "BUY", pattern

    # SELL
    if (
        trend_down
        and
        pullback_sell
        and
        sell_score >= 8
        and
        sell_score > buy_score
    ):

        pattern = (
            "SELL|"
            f"PULLBACK_{int(pullback_sell)}|"
            f"REJECTION_{int(features['bear_rejection'])}|"
            f"MACD_{int(macd_sell)}|"
            f"RSI_{int(rsi_sell)}|"
            f"VOL_{int(volume_ok)}"
        )

        return "SELL", pattern

    return "WAIT", None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, entry_index, direction):

    entry = df.iloc[entry_index]

    entry_price = entry["close"]
    atr = entry["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp = (
            entry_price +
            atr * TP_ATR
        )

        sl = (
            entry_price -
            atr * SL_ATR
        )

    else:

        tp = (
            entry_price -
            atr * TP_ATR
        )

        sl = (
            entry_price +
            atr * SL_ATR
        )

    end_index = min(
        entry_index + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    for j in range(
        entry_index + 1,
        end_index + 1
    ):

        candle = df.iloc[j]

        if direction == "BUY":

            hit_tp = candle["high"] >= tp
            hit_sl = candle["low"] <= sl

            # Conservative assumption:
            # if both happen in same candle,
            # count SL first.
            if hit_tp and hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 - ROUND_TRIP_COST / SL_ATR
                }

            if hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 - ROUND_TRIP_COST / SL_ATR
                }

            if hit_tp:
                return {
                    "result": "TP",
                    "r": TP_ATR - ROUND_TRIP_COST / SL_ATR
                }

        else:

            hit_tp = candle["low"] <= tp
            hit_sl = candle["high"] >= sl

            if hit_tp and hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 - ROUND_TRIP_COST / SL_ATR
                }

            if hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 - ROUND_TRIP_COST / SL_ATR
                }

            if hit_tp:
                return {
                    "result": "TP",
                    "r": TP_ATR - ROUND_TRIP_COST / SL_ATR
                }

    # TIMEOUT
    final_price = df.iloc[end_index]["close"]

    if direction == "BUY":

        gross_return = (
            final_price -
            entry_price
        ) / (
            entry_price
        )

    else:

        gross_return = (
            entry_price -
            final_price
        ) / (
            entry_price
        )

    net_return = (
        gross_return -
        ROUND_TRIP_COST
    )

    r = net_return / (
        atr / entry_price
    )

    return {
        "result": "TIMEOUT",
        "r": r
    }


# ============================================================
# BACKTEST RANGE
# ============================================================

def run_backtest(
    df,
    start_index,
    end_index,
    allowed_patterns=None
):

    trades = []

    last_trade_index = -999999

    for i in range(
        max(120, start_index),
        min(end_index, len(df) - MAX_HOLD_CANDLES)
    ):

        if (
            i - last_trade_index
            < COOLDOWN_CANDLES
        ):
            continue

        signal, pattern = get_signal(
            df,
            i
        )

        if signal == "WAIT":
            continue

        if (
            allowed_patterns is not None
            and
            pattern not in allowed_patterns
        ):
            continue

        trade = simulate_trade(
            df,
            i,
            signal
        )

        if trade is None:
            continue

        trades.append({
            "index": i,
            "datetime": df.iloc[i]["datetime"],
            "direction": signal,
            "pattern": pattern,
            "result": trade["result"],
            "r": trade["r"]
        })

        last_trade_index = i

    return pd.DataFrame(trades)


# ============================================================
# STATISTICS
# ============================================================

def print_stats(name, trades):

    if trades.empty:

        print()
        print(name)
        print("No trades.")

        return

    total = len(trades)

    tp = (
        trades["result"] == "TP"
    ).sum()

    sl = (
        trades["result"] == "SL"
    ).sum()

    timeout = (
        trades["result"] == "TIMEOUT"
    ).sum()

    total_r = trades["r"].sum()

    avg_r = trades["r"].mean()

    win_rate = (
        tp / total * 100
    )

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Trades: {total}"
    )

    print(
        f"TP: {tp} ({tp / total * 100:.1f}%)"
    )

    print(
        f"SL: {sl} ({sl / total * 100:.1f}%)"
    )

    print(
        f"Timeout: {timeout} ({timeout / total * 100:.1f}%)"
    )

    print(
        f"TP Win Rate: {win_rate:.1f}%"
    )

    print(
        f"Total R: {total_r:.2f}"
    )

    print(
        f"Average R: {avg_r:.3f}"
    )


# ============================================================
# PATTERN ANALYSIS
# ============================================================

def pattern_analysis(trades, direction):

    data = trades[
        trades["direction"] == direction
    ]

    if data.empty:
        return

    grouped = (
        data
        .groupby("pattern")
        .agg(
            trades=("r", "count"),
            total_r=("r", "sum"),
            avg_r=("r", "mean"),
            tp=("result", lambda x: (x == "TP").sum()),
            sl=("result", lambda x: (x == "SL").sum()),
            timeout=("result", lambda x: (x == "TIMEOUT").sum())
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["trades"] >= MIN_TRAIN_TRADES
    ]

    grouped = grouped.sort_values(
        "avg_r",
        ascending=False
    )

    print()
    print()
    print(
        f"📊 {direction} TRAINING PATTERNS"
    )

    print("-" * 100)

    if grouped.empty:

        print("No patterns with enough trades.")

        return grouped

    for _, row in grouped.head(15).iterrows():

        total = row["trades"]

        tp_pct = (
            row["tp"] /
            total *
            100
        )

        sl_pct = (
            row["sl"] /
            total *
            100
        )

        print(
            f"{int(total):4d} trades | "
            f"TP {tp_pct:5.1f}% | "
            f"SL {sl_pct:5.1f}% | "
            f"Avg R {row['avg_r']:7.3f} | "
            f"{row['pattern']}"
        )

    return grouped


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(trades):

    if trades.empty:
        return 0

    equity = trades["r"].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    return drawdown.min()


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_data()

    df = calculate_indicators(df)

    print()

    print(
        "Total candles:",
        len(df)
    )

    train_end_time = (
        df["datetime"].iloc[0]
        +
        timedelta(days=TRAIN_DAYS)
    )

    train_end = df[
        df["datetime"] <
        train_end_time
    ].index[-1]

    validation_start = train_end

    print(
        "Training range:",
        0,
        "to",
        train_end
    )

    print(
        "Validation range:",
        validation_start,
        "to",
        len(df)
    )

    print()

    print(
        f"TP: {TP_ATR} x ATR"
    )

    print(
        f"SL: {SL_ATR} x ATR"
    )

    print(
        "Maximum holding:",
        MAX_HOLD_CANDLES * 15,
        "minutes"
    )

    print(
        "Round-trip estimated cost:",
        ROUND_TRIP_COST * 100,
        "%"
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    print()
    print("🔵 RUNNING TRAINING BACKTEST...")

    train_trades = run_backtest(
        df,
        120,
        train_end
    )

    print_stats(
        "TRAINING TOTAL",
        train_trades
    )

    buy_patterns = pattern_analysis(
        train_trades,
        "BUY"
    )

    sell_patterns = pattern_analysis(
        train_trades,
        "SELL"
    )

    # --------------------------------------------------------
    # SELECT ONLY POSITIVE TRAINING PATTERNS
    # --------------------------------------------------------

    selected_buy = set()
    selected_sell = set()

    if buy_patterns is not None:

        positive_buy = buy_patterns[
            buy_patterns["avg_r"] > 0
        ]

        selected_buy = set(
            positive_buy["pattern"].head(5)
        )

    if sell_patterns is not None:

        positive_sell = sell_patterns[
            sell_patterns["avg_r"] > 0
        ]

        selected_sell = set(
            positive_sell["pattern"].head(5)
        )

    print()
    print("=" * 70)
    print("🎯 SELECTED TRAINING PATTERNS")
    print("=" * 70)

    print()
    print("BUY:")

    if selected_buy:

        for p in selected_buy:
            print("  ", p)

    else:

        print(
            "   NONE — no positive BUY pattern found."
        )

    print()
    print("SELL:")

    if selected_sell:

        for p in selected_sell:
            print("  ", p)

    else:

        print(
            "   NONE — no positive SELL pattern found."
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print()
    print("🟢 RUNNING UNTOUCHED VALIDATION...")

    allowed = (
        selected_buy |
        selected_sell
    )

    if not allowed:

        print()
        print(
            "⚠️ STOP: Training produced no positive patterns."
        )

        print(
            "The strategy will NOT be tested as if it were successful."
        )

        return

    validation_trades = run_backtest(
        df,
        validation_start,
        len(df) - MAX_HOLD_CANDLES,
        allowed_patterns=allowed
    )

    print_stats(
        "VALIDATION TOTAL",
        validation_trades
    )

    buy_validation = validation_trades[
        validation_trades["direction"] == "BUY"
    ]

    sell_validation = validation_trades[
        validation_trades["direction"] == "SELL"
    ]

    print_stats(
        "VALIDATION BUY",
        buy_validation
    )

    print_stats(
        "VALIDATION SELL",
        sell_validation
    )

    # --------------------------------------------------------
    # DRAWDOWN
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("📉 VALIDATION RISK")
    print("=" * 70)

    print(
        f"Maximum Drawdown: "
        f"{max_drawdown(validation_trades):.2f} R"
    )

    # --------------------------------------------------------
    # FINAL DECISION
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("🏁 FINAL RESULT")
    print("=" * 70)

    if validation_trades.empty:

        print(
            "No validation trades."
        )

    else:

        avg_r = validation_trades["r"].mean()
        total_r = validation_trades["r"].sum()

        print(
            f"Validation trades: "
            f"{len(validation_trades)}"
        )

        print(
            f"Validation Total R: "
            f"{total_r:.2f}"
        )

        print(
            f"Validation Average R: "
            f"{avg_r:.3f}"
        )

        if (
            total_r > 0
            and
            avg_r > 0
        ):

            print()
            print(
                "⚠️ Validation is positive."
            )

            print(
                "This does NOT guarantee live profitability."
            )

        else:

            print()
            print(
                "❌ Validation is not positive."
            )

            print(
                "Do NOT use these rules for live trading yet."
            )

    print()
    print("✅ Backtest finished.")


if __name__ == "__main__":
    main()

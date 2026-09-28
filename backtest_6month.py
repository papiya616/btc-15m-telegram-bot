import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M WALK-FORWARD BACKTEST
# ============================================================

SYMBOL = "BTC-USD"
GRANULARITY = 900
DAYS = 183
CHUNK_CANDLES = 250

TP_ATR = 1.5
SL_ATR = 1.0
MAX_HOLD_CANDLES = 12

ROUND_TRIP_COST = 0.0014
COOLDOWN_CANDLES = 12

MIN_TRAIN_TRADES = 20

# 2 months training -> 1 month validation
TRAIN_DAYS = 60
VALIDATION_DAYS = 30


# ============================================================
# DOWNLOAD
# ============================================================

def download_data():

    print("🚀 BTC 6-Month Walk-Forward Backtest Started!")
    print()
    print("📥 Downloading BTC 15M historical data...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []
    current = start_time

    while current < end_time:

        chunk_end = min(
            current + timedelta(minutes=15 * CHUNK_CANDLES),
            end_time
        )

        start_ts = int(current.timestamp())
        end_ts = int(chunk_end.timestamp())

        print(
            "Downloading:",
            current.strftime("%Y-%m-%d %H:%M"),
            "to",
            chunk_end.strftime("%Y-%m-%d %H:%M")
        )

        url = (
            "https://api.exchange.coinbase.com/"
            f"products/{SYMBOL}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": start_ts,
            "end": end_ts
        }

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

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(drop=True)

    print()
    print(
        "Total candles downloaded:",
        len(df)
    )

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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

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

    df["atr"] = true_range.rolling(
        14
    ).mean()

    # Volume
    df["volume_avg"] = df["volume"].rolling(
        20
    ).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    # ATR regime
    df["atr_avg"] = df["atr"].rolling(
        100
    ).mean()

    return df


# ============================================================
# CANDLE FEATURES
# ============================================================

def candle_features(row):

    candle_range = (
        row["high"] -
        row["low"]
    )

    if candle_range <= 0:

        return {
            "bullish": False,
            "bearish": False,
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
        row["low"] +
        candle_range * 0.55
    )

    bear_rejection = (
        upper_wick >= body * 1.2
        and
        row["close"] <
        row["low"] +
        candle_range * 0.45
    )

    return {
        "bullish": bullish,
        "bearish": bearish,
        "bull_rejection": bull_rejection,
        "bear_rejection": bear_rejection
    }


# ============================================================
# SIGNAL
# ============================================================

def get_signal(df, i):

    row = df.iloc[i]

    required = [
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "atr",
        "volume_ratio",
        "atr_avg"
    ]

    for x in required:

        if pd.isna(row[x]):
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
    # PULLBACK
    # --------------------------------------------------------

    pullback_buy = (
        row["low"] <=
        row["ema21"] +
        atr * 0.35
        and
        row["close"] >
        row["ema21"]
    )

    pullback_sell = (
        row["high"] >=
        row["ema21"] -
        atr * 0.35
        and
        row["close"] <
        row["ema21"]
    )

    # --------------------------------------------------------
    # OVEREXTENSION FILTER
    # --------------------------------------------------------

    distance_ema21 = (
        abs(price - row["ema21"]) /
        atr
    )

    not_overextended = (
        distance_ema21 <= 1.8
    )

    if not not_overextended:
        return "WAIT", None

    # --------------------------------------------------------
    # MACD
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
        row["atr"] >=
        row["atr_avg"] * 0.75
        and
        row["atr"] <=
        row["atr_avg"] * 2.0
    )

    if not volatility_ok:
        return "WAIT", None

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

    if features["bullish"]:
        buy_score += 1

    if macd_buy:
        buy_score += 2

    if rsi_buy:
        buy_score += 1

    if volume_ok:
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

    if features["bearish"]:
        sell_score += 1

    if macd_sell:
        sell_score += 2

    if rsi_sell:
        sell_score += 1

    if volume_ok:
        sell_score += 1

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

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
            f"REJECTION_{int(features['bull_rejection'])}|"
            f"MACD_{int(macd_buy)}|"
            f"RSI_{int(rsi_buy)}|"
            f"VOL_{int(volume_ok)}"
        )

        return "BUY", pattern

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

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
        entry_index +
        MAX_HOLD_CANDLES,
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

            if hit_tp and hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 -
                    ROUND_TRIP_COST /
                    SL_ATR
                }

            if hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 -
                    ROUND_TRIP_COST /
                    SL_ATR
                }

            if hit_tp:
                return {
                    "result": "TP",
                    "r": TP_ATR -
                    ROUND_TRIP_COST /
                    SL_ATR
                }

        else:

            hit_tp = candle["low"] <= tp
            hit_sl = candle["high"] >= sl

            if hit_tp and hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 -
                    ROUND_TRIP_COST /
                    SL_ATR
                }

            if hit_sl:
                return {
                    "result": "SL",
                    "r": -1.0 -
                    ROUND_TRIP_COST /
                    SL_ATR
                }

            if hit_tp:
                return {
                    "result": "TP",
                    "r": TP_ATR -
                    ROUND_TRIP_COST /
                    SL_ATR
                }

    # TIMEOUT
    final_price = df.iloc[
        end_index
    ]["close"]

    if direction == "BUY":

        gross = (
            final_price -
            entry_price
        ) / entry_price

    else:

        gross = (
            entry_price -
            final_price
        ) / entry_price

    net = gross - ROUND_TRIP_COST

    r = net / (
        atr / entry_price
    )

    return {
        "result": "TIMEOUT",
        "r": r
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df,
    start_index,
    end_index,
    allowed_patterns=None
):

    trades = []

    last_trade_index = -999999

    end_index = min(
        end_index,
        len(df) -
        MAX_HOLD_CANDLES
    )

    for i in range(
        max(120, start_index),
        end_index
    ):

        if (
            i -
            last_trade_index
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

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    if trades.empty:

        print("No trades.")
        return

    total = len(trades)

    tp = (
        trades["result"] ==
        "TP"
    ).sum()

    sl = (
        trades["result"] ==
        "SL"
    ).sum()

    timeout = (
        trades["result"] ==
        "TIMEOUT"
    ).sum()

    total_r = trades["r"].sum()
    avg_r = trades["r"].mean()

    print(
        f"Trades: {total}"
    )

    print(
        f"TP: {tp} "
        f"({tp / total * 100:.1f}%)"
    )

    print(
        f"SL: {sl} "
        f"({sl / total * 100:.1f}%)"
    )

    print(
        f"Timeout: {timeout} "
        f"({timeout / total * 100:.1f}%)"
    )

    print(
        f"Total R: {total_r:.2f}"
    )

    print(
        f"Average R: {avg_r:.3f}"
    )


# ============================================================
# POSITIVE PATTERNS FROM TRAINING
# ============================================================

def get_positive_patterns(
    trades,
    direction
):

    data = trades[
        trades["direction"] ==
        direction
    ]

    if data.empty:
        return set()

    grouped = (
        data
        .groupby("pattern")
        .agg(
            trades=("r", "count"),
            avg_r=("r", "mean"),
            total_r=("r", "sum")
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["trades"] >=
        MIN_TRAIN_TRADES
    ]

    grouped = grouped[
        grouped["avg_r"] > 0
    ]

    grouped = grouped.sort_values(
        "avg_r",
        ascending=False
    )

    return set(
        grouped["pattern"].head(5)
    )


# ============================================================
# MAX DRAWDOWN
# ============================================================

def max_drawdown(trades):

    if trades.empty:
        return 0.0

    equity = trades["r"].cumsum()

    peak = equity.cummax()

    drawdown = equity - peak

    return drawdown.min()


# ============================================================
# WALK-FORWARD
# ============================================================

def main():

    df = download_data()

    df = calculate_indicators(df)

    print()
    print(
        "Total candles:",
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

    print()
    print(
        "🔄 WALK-FORWARD TEST"
    )

    print(
        "Training:",
        TRAIN_DAYS,
        "days"
    )

    print(
        "Validation:",
        VALIDATION_DAYS,
        "days"
    )

    # --------------------------------------------------------
    # CREATE MONTH-LIKE WINDOWS
    # --------------------------------------------------------

    first_time = df["datetime"].iloc[0]

    current_train_start = first_time

    all_validation = []

    window_number = 1

    while True:

        train_start_time = current_train_start

        train_end_time = (
            train_start_time +
            timedelta(days=TRAIN_DAYS)
        )

        validation_end_time = (
            train_end_time +
            timedelta(days=VALIDATION_DAYS)
        )

        if validation_end_time > df["datetime"].iloc[-1]:
            break

        train_start = df[
            df["datetime"] >=
            train_start_time
        ].index[0]

        train_end_list = df[
            df["datetime"] <
            train_end_time
        ].index

        validation_start_list = df[
            df["datetime"] >=
            train_end_time
        ].index

        validation_end_list = df[
            df["datetime"] <
            validation_end_time
        ].index

        if (
            len(train_end_list) == 0
            or
            len(validation_start_list) == 0
            or
            len(validation_end_list) == 0
        ):
            break

        train_end = train_end_list[-1]

        validation_start = (
            validation_start_list[0]
        )

        validation_end = (
            validation_end_list[-1]
        )

        print()
        print("=" * 70)
        print(
            f"🔵 WINDOW {window_number}"
        )
        print("=" * 70)

        print(
            "Training:",
            df["datetime"].iloc[
                train_start
            ],
            "→",
            df["datetime"].iloc[
                train_end
            ]
        )

        print(
            "Validation:",
            df["datetime"].iloc[
                validation_start
            ],
            "→",
            df["datetime"].iloc[
                validation_end
            ]
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_trades = run_backtest(
            df,
            train_start,
            train_end
        )

        print_stats(
            "TRAINING",
            train_trades
        )

        buy_patterns = get_positive_patterns(
            train_trades,
            "BUY"
        )

        sell_patterns = get_positive_patterns(
            train_trades,
            "SELL"
        )

        allowed_patterns = (
            buy_patterns |
            sell_patterns
        )

        print()
        print(
            "Positive BUY patterns:",
            len(buy_patterns)
        )

        print(
            "Positive SELL patterns:",
            len(sell_patterns)
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if allowed_patterns:

            validation_trades = run_backtest(
                df,
                validation_start,
                validation_end,
                allowed_patterns
            )

        else:

            validation_trades = pd.DataFrame()

        print_stats(
            "VALIDATION",
            validation_trades
        )

        if not validation_trades.empty:

            validation_copy = (
                validation_trades.copy()
            )

            validation_copy[
                "window"
            ] = window_number

            all_validation.append(
                validation_copy
            )

        current_train_start = (
            current_train_start +
            timedelta(days=VALIDATION_DAYS)
        )

        window_number += 1

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()
    print("=" * 70)
    print("🏁 WALK-FORWARD FINAL RESULT")
    print("=" * 70)

    if not all_validation:

        print(
            "No validation trades were produced."
        )

        return

    final_validation = pd.concat(
        all_validation,
        ignore_index=True
    )

    print_stats(
        "ALL VALIDATION TRADES",
        final_validation
    )

    print()
    print(
        f"Validation Maximum Drawdown: "
        f"{max_drawdown(final_validation):.2f} R"
    )

    # --------------------------------------------------------
    # WINDOW SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("📊 VALIDATION WINDOW SUMMARY")
    print("=" * 70)

    for window in sorted(
        final_validation["window"].unique()
    ):

        data = final_validation[
            final_validation["window"] ==
            window
        ]

        print(
            f"Window {int(window)} | "
            f"Trades: {len(data)} | "
            f"Total R: {data['r'].sum():.2f} | "
            f"Avg R: {data['r'].mean():.3f}"
        )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "A positive total result alone is NOT enough."
    )

    print(
        "We want the strategy to remain reasonably "
        "stable across multiple validation windows."
    )

    print(
        "Do NOT switch the live Telegram bot to this "
        "strategy yet."
    )

    print()
    print("✅ Walk-forward backtest finished.")


if __name__ == "__main__":
    main()

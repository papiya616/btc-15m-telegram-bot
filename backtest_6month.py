import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M ENTRY-QUALITY + TP/SL WALK-FORWARD BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900
CHUNK_CANDLES = 250

DAYS = 183

TRAIN_DAYS = 60
VALIDATION_DAYS = 30

MAX_HOLD_CANDLES = 12
COOLDOWN_CANDLES = 12

ROUND_TRIP_COST = 0.0014

MIN_TRAIN_TRADES = 30

# TP / SL combinations tested ONLY on training data
TP_SL_OPTIONS = [
    (1.0, 1.0),
    (1.2, 1.0),
    (1.5, 1.0),
    (1.8, 1.0),
    (2.0, 1.0),
    (1.5, 0.8),
    (1.8, 0.8),
    (2.0, 0.8),
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_candles():

    print("🚀 BTC 6-Month Entry Quality Backtest Started!")
    print()
    print("📥 Downloading BTC-USD 15M data...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    rows = []

    current_start = int(start_time.timestamp())
    final_end = int(end_time.timestamp())

    while current_start < final_end:

        current_end = min(
            current_start +
            GRANULARITY * CHUNK_CANDLES,
            final_end
        )

        url = (
            f"https://api.exchange.coinbase.com/"
            f"products/{PRODUCT}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": datetime.fromtimestamp(
                current_start,
                timezone.utc
            ).isoformat(),
            "end": datetime.fromtimestamp(
                current_end,
                timezone.utc
            ).isoformat()
        }

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if data:
                rows.extend(data)

            print(
                "Downloading:",
                datetime.fromtimestamp(
                    current_start,
                    timezone.utc
                ).strftime("%Y-%m-%d"),
                "to",
                datetime.fromtimestamp(
                    current_end,
                    timezone.utc
                ).strftime("%Y-%m-%d")
            )

            current_start = current_end

            time.sleep(0.2)

        except Exception as e:

            print("Download error:", e)
            time.sleep(3)

    if not rows:
        raise RuntimeError(
            "No market data downloaded."
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume"
        ]
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

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

    df = df.dropna().reset_index(
        drop=True
    )

    print()
    print(
        "Total candles downloaded:",
        len(df)
    )

    print(
        "First candle:",
        df.iloc[0]["timestamp"]
    )

    print(
        "Last candle: ",
        df.iloc[-1]["timestamp"]
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

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    df["rsi"] = (
        100 -
        (100 / (1 + rs))
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

    df["macd"] = (
        ema12 - ema26
    )

    df["macd_signal"] = df[
        "macd"
    ].ewm(
        span=9,
        adjust=False
    ).mean()

    # ATR
    prev_close = df[
        "close"
    ].shift(1)

    tr1 = (
        df["high"] -
        df["low"]
    )

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

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # Volume
    df["volume_ma"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_ma"]
    )

    # Candle
    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["upper_wick"] = (
        df["high"] -
        df[
            ["open", "close"]
        ].max(axis=1)
    )

    df["lower_wick"] = (
        df[
            ["open", "close"]
        ].min(axis=1)
        -
        df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(
            0,
            np.nan
        )
    )

    # Recent levels
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

    df["atr_median"] = (
        df["atr_pct"]
        .rolling(96)
        .median()
    )

    return df


# ============================================================
# REGIME
# ============================================================

def get_regime(row):

    if any(
        pd.isna(row[x])
        for x in [
            "ema9",
            "ema21",
            "ema50",
            "atr",
            "atr_pct",
            "atr_median"
        ]
    ):
        return "UNKNOWN"

    # Avoid extreme volatility
    if row["atr_pct"] > (
        row["atr_median"] * 1.8
    ):
        return "HIGH_VOL"

    if (
        row["ema9"] >
        row["ema21"] >
        row["ema50"]
    ):
        return "UP"

    if (
        row["ema9"] <
        row["ema21"] <
        row["ema50"]
    ):
        return "DOWN"

    return "CHOPPY"


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def bullish_confirmation(row):

    if row["range"] <= 0:
        return False

    strong_body = (
        row["body_ratio"] >= 0.45
    )

    bullish = (
        row["close"] >
        row["open"]
    )

    close_high = (
        row["high"] -
        row["close"]
        <=
        row["range"] * 0.30
    )

    lower_rejection = (
        row["lower_wick"] >
        row["body"] * 0.8
    )

    return (
        bullish and
        strong_body and
        close_high
    ) or lower_rejection


def bearish_confirmation(row):

    if row["range"] <= 0:
        return False

    strong_body = (
        row["body_ratio"] >= 0.45
    )

    bearish = (
        row["close"] <
        row["open"]
    )

    close_low = (
        row["close"] -
        row["low"]
        <=
        row["range"] * 0.30
    )

    upper_rejection = (
        row["upper_wick"] >
        row["body"] * 0.8
    )

    return (
        bearish and
        strong_body and
        close_low
    ) or upper_rejection


# ============================================================
# ENTRY QUALITY
# ============================================================

def get_entry_setup(df, i):

    row = df.iloc[i]

    regime = get_regime(row)

    if regime in [
        "UNKNOWN",
        "CHOPPY",
        "HIGH_VOL"
    ]:
        return "WAIT"

    if pd.isna(row["atr"]) or row["atr"] <= 0:
        return "WAIT"

    distance = (
        abs(
            row["close"] -
            row["ema21"]
        ) /
        row["atr"]
    )

    # Do not chase extended move
    if distance > 1.5:
        return "WAIT"

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if regime == "UP":

        trend = (
            row["ema9"] >
            row["ema21"] >
            row["ema50"]
        )

        price_above = (
            row["close"] >
            row["ema21"]
        )

        macd_ok = (
            row["macd"] >
            row["macd_signal"]
        )

        rsi_ok = (
            48 <= row["rsi"] <= 68
        )

        pullback = (
            abs(
                row["close"] -
                row["ema21"]
            ) /
            row["atr"]
            <= 0.9
        )

        confirmation = (
            bullish_confirmation(row)
        )

        volume_ok = (
            row["volume_ratio"] >= 0.85
        )

        if all([
            trend,
            price_above,
            macd_ok,
            rsi_ok,
            pullback,
            confirmation,
            volume_ok
        ]):
            return "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if regime == "DOWN":

        trend = (
            row["ema9"] <
            row["ema21"] <
            row["ema50"]
        )

        price_below = (
            row["close"] <
            row["ema21"]
        )

        macd_ok = (
            row["macd"] <
            row["macd_signal"]
        )

        rsi_ok = (
            32 <= row["rsi"] <= 52
        )

        pullback = (
            abs(
                row["close"] -
                row["ema21"]
            ) /
            row["atr"]
            <= 0.9
        )

        confirmation = (
            bearish_confirmation(row)
        )

        volume_ok = (
            row["volume_ratio"] >= 0.85
        )

        if all([
            trend,
            price_below,
            macd_ok,
            rsi_ok,
            pullback,
            confirmation,
            volume_ok
        ]):
            return "SELL"

    return "WAIT"


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    df,
    entry_index,
    signal,
    tp_atr,
    sl_atr
):

    entry = df.iloc[
        entry_index
    ]["close"]

    atr = df.iloc[
        entry_index
    ]["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if signal == "BUY":

        tp = (
            entry +
            tp_atr * atr
        )

        sl = (
            entry -
            sl_atr * atr
        )

    else:

        tp = (
            entry -
            tp_atr * atr
        )

        sl = (
            entry +
            sl_atr * atr
        )

    last_index = min(
        entry_index +
        MAX_HOLD_CANDLES,
        len(df) - 1
    )

    risk_pct = (
        sl_atr *
        atr /
        entry
    )

    if risk_pct <= 0:
        return None

    for j in range(
        entry_index + 1,
        last_index + 1
    ):

        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if signal == "BUY":

            # Conservative:
            # if both touched, SL first
            if (
                low <= sl and
                high >= tp
            ):
                net_return = (
                    -sl_atr *
                    atr /
                    entry
                ) - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "r": net_return /
                    risk_pct
                }

            if low <= sl:

                net_return = (
                    -sl_atr *
                    atr /
                    entry
                ) - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "r": net_return /
                    risk_pct
                }

            if high >= tp:

                net_return = (
                    tp_atr *
                    atr /
                    entry
                ) - ROUND_TRIP_COST

                return {
                    "result": "TP",
                    "r": net_return /
                    risk_pct
                }

        else:

            if (
                high >= sl and
                low <= tp
            ):
                net_return = (
                    -sl_atr *
                    atr /
                    entry
                ) - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "r": net_return /
                    risk_pct
                }

            if high >= sl:

                net_return = (
                    -sl_atr *
                    atr /
                    entry
                ) - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "r": net_return /
                    risk_pct
                }

            if low <= tp:

                net_return = (
                    tp_atr *
                    atr /
                    entry
                ) - ROUND_TRIP_COST

                return {
                    "result": "TP",
                    "r": net_return /
                    risk_pct
                }

    # Timeout
    exit_price = df.iloc[
        last_index
    ]["close"]

    if signal == "BUY":

        gross_return = (
            exit_price /
            entry
        ) - 1

    else:

        gross_return = (
            entry /
            exit_price
        ) - 1

    net_return = (
        gross_return -
        ROUND_TRIP_COST
    )

    return {
        "result": "TIMEOUT",
        "r": net_return /
        risk_pct
    }


# ============================================================
# COLLECT TRADES
# ============================================================

def collect_trades(
    df,
    start_index,
    end_index,
    tp_atr,
    sl_atr
):

    trades = []

    last_trade = -999999

    for i in range(
        start_index,
        end_index
    ):

        if i < 120:
            continue

        if (
            i -
            last_trade
            < COOLDOWN_CANDLES
        ):
            continue

        signal = get_entry_setup(
            df,
            i
        )

        if signal not in [
            "BUY",
            "SELL"
        ]:
            continue

        result = simulate_trade(
            df,
            i,
            signal,
            tp_atr,
            sl_atr
        )

        if result is None:
            continue

        result["signal"] = signal
        result["index"] = i
        result["timestamp"] = (
            df.iloc[i]["timestamp"]
        )

        trades.append(result)

        last_trade = i

    return trades


# ============================================================
# STATISTICS
# ============================================================

def stats(trades):

    if not trades:

        return {
            "count": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "total_r": 0,
            "avg_r": 0,
            "max_dd": 0
        }

    rs = [
        x["r"]
        for x in trades
    ]

    results = [
        x["result"]
        for x in trades
    ]

    equity = 0
    peak = 0
    max_dd = 0

    for r in rs:

        equity += r

        peak = max(
            peak,
            equity
        )

        max_dd = min(
            max_dd,
            equity - peak
        )

    return {
        "count": len(trades),
        "tp": results.count("TP"),
        "sl": results.count("SL"),
        "timeout": results.count(
            "TIMEOUT"
        ),
        "total_r": sum(rs),
        "avg_r": np.mean(rs),
        "max_dd": max_dd
    }


def print_stats(
    title,
    trades
):

    s = stats(trades)

    print()
    print(title)

    if s["count"] == 0:

        print("No trades.")
        return

    total = s["count"]

    print(
        f"Trades: {total}"
    )

    print(
        f"TP: {s['tp']} "
        f"({s['tp']/total*100:.1f}%)"
    )

    print(
        f"SL: {s['sl']} "
        f"({s['sl']/total*100:.1f}%)"
    )

    print(
        f"Timeout: {s['timeout']} "
        f"({s['timeout']/total*100:.1f}%)"
    )

    print(
        f"Total R: {s['total_r']:.2f}"
    )

    print(
        f"Average R: {s['avg_r']:.3f}"
    )

    print(
        f"Max Drawdown: {s['max_dd']:.2f}R"
    )


# ============================================================
# SELECT TP/SL ON TRAINING ONLY
# ============================================================

def select_tp_sl(
    df,
    train_start,
    train_end
):

    print()
    print(
        "========== TRAINING TP/SL TEST =========="
    )

    candidates = []

    for tp_atr, sl_atr in TP_SL_OPTIONS:

        trades = collect_trades(
            df,
            train_start,
            train_end,
            tp_atr,
            sl_atr
        )

        s = stats(trades)

        if s["count"] < MIN_TRAIN_TRADES:
            print(
                f"TP {tp_atr} / SL {sl_atr}: "
                f"only {s['count']} trades - skipped"
            )
            continue

        print(
            f"TP {tp_atr} / SL {sl_atr} | "
            f"Trades {s['count']} | "
            f"TP {s['tp']/s['count']*100:.1f}% | "
            f"SL {s['sl']/s['count']*100:.1f}% | "
            f"Total R {s['total_r']:.2f} | "
            f"Avg R {s['avg_r']:.3f}"
        )

        # Require positive expectancy
        if s["avg_r"] > 0:

            candidates.append(
                (
                    s["avg_r"],
                    s["count"],
                    tp_atr,
                    sl_atr
                )
            )

    if not candidates:

        print()
        print(
            "❌ No positive TP/SL configuration "
            "found in training."
        )

        return None

    # Rank by average R
    candidates.sort(
        key=lambda x: (
            x[0],
            x[1]
        ),
        reverse=True
    )

    best = candidates[0]

    print()
    print(
        "✅ Selected training TP/SL:"
    )

    print(
        f"TP = {best[2]} ATR"
    )

    print(
        f"SL = {best[3]} ATR"
    )

    print(
        f"Training Avg R = {best[0]:.3f}"
    )

    print(
        f"Training Trades = {best[1]}"
    )

    return (
        best[2],
        best[3]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_candles()

    print()
    print(
        "📊 Calculating indicators..."
    )

    df = calculate_indicators(
        df
    )

    print(
        "✅ Indicators calculated."
    )

    print()
    print(
        "========== SETTINGS =========="
    )

    print(
        "Maximum holding:",
        MAX_HOLD_CANDLES * 15,
        "minutes"
    )

    print(
        "Cooldown:",
        COOLDOWN_CANDLES * 15,
        "minutes"
    )

    print(
        "Estimated round-trip cost:",
        ROUND_TRIP_COST * 100,
        "%"
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

    start_time = df[
        "timestamp"
    ].iloc[0]

    final_time = df[
        "timestamp"
    ].iloc[-1]

    window_start = start_time

    all_validation = []

    window = 1

    while True:

        train_start_time = (
            window_start
        )

        train_end_time = (
            train_start_time +
            timedelta(days=TRAIN_DAYS)
        )

        validation_start_time = (
            train_end_time
        )

        validation_end_time = (
            validation_start_time +
            timedelta(days=VALIDATION_DAYS)
        )

        if validation_end_time > final_time:
            break

        train_start = df.index[
            df["timestamp"] >=
            train_start_time
        ][0]

        train_end = df.index[
            df["timestamp"] >=
            train_end_time
        ][0]

        validation_start = df.index[
            df["timestamp"] >=
            validation_start_time
        ][0]

        validation_end_list = df.index[
            df["timestamp"] <=
            validation_end_time
        ]

        if len(validation_end_list) == 0:
            break

        validation_end = (
            validation_end_list[-1] + 1
        )

        print()
        print(
            "=========================================="
        )

        print(
            f"WINDOW {window}"
        )

        print(
            "=========================================="
        )

        print(
            "Training:",
            train_start_time,
            "→",
            train_end_time
        )

        print(
            "Validation:",
            validation_start_time,
            "→",
            validation_end_time
        )

        # ----------------------------------------------------
        # Find TP/SL using TRAINING ONLY
        # ----------------------------------------------------

        selected = select_tp_sl(
            df,
            train_start,
            train_end
        )

        if selected is None:

            print()
            print(
                "⚠️ No positive training setup."
            )

            print(
                "Validation for this window "
                "is skipped."
            )

            window_start = (
                window_start +
                timedelta(
                    days=VALIDATION_DAYS
                )
            )

            window += 1

            continue

        tp_atr, sl_atr = selected

        # ----------------------------------------------------
        # Training result
        # ----------------------------------------------------

        training_trades = collect_trades(
            df,
            train_start,
            train_end,
            tp_atr,
            sl_atr
        )

        print_stats(
            "SELECTED TRAINING RESULT",
            training_trades
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_trades = collect_trades(
            df,
            validation_start,
            validation_end,
            tp_atr,
            sl_atr
        )

        print()
        print(
            f"LOCKED TP/SL FOR VALIDATION: "
            f"{tp_atr} ATR / {sl_atr} ATR"
        )

        print_stats(
            "VALIDATION RESULT",
            validation_trades
        )

        all_validation.extend(
            validation_trades
        )

        window_start = (
            window_start +
            timedelta(
                days=VALIDATION_DAYS
            )
        )

        window += 1

    # ========================================================
    # OVERALL
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "========== WALK-FORWARD OVERALL =========="
    )

    print(
        "=========================================="
    )

    print_stats(
        "ALL VALIDATION TRADES",
        all_validation
    )

    # BUY / SELL
    buy = [
        x for x in all_validation
        if x["signal"] == "BUY"
    ]

    sell = [
        x for x in all_validation
        if x["signal"] == "SELL"
    ]

    print_stats(
        "VALIDATION BUY",
        buy
    )

    print_stats(
        "VALIDATION SELL",
        sell
    )

    # ========================================================
    # FINAL
    # ========================================================

    s = stats(
        all_validation
    )

    print()
    print(
        "=========================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=========================================="
    )

    if s["count"] == 0:

        print(
            "❌ No validation trades."
        )

        print(
            "No reliable edge was demonstrated."
        )

    elif s["total_r"] > 0:

        print(
            "📈 Validation Total R is positive."
        )

        print(
            "This is only a historical result."
        )

        print(
            "It does NOT prove future profitability."
        )

        print(
            "More out-of-sample testing is required."
        )

    else:

        print(
            "📉 Validation Total R is negative."
        )

        print(
            "Current entry/exit rules do not "
            "show a positive out-of-sample edge."
        )

        print(
            "Do NOT put these rules into the "
            "live bot yet."
        )

    print()
    print(
        "🏁 Backtest finished."
    )


if __name__ == "__main__":
    main()

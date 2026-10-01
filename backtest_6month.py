import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 6-MONTH TP/SL STRUCTURE TEST
# Same entry logic, different TP/SL
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900
DAYS = 183
CHUNK_CANDLES = 250

ROUND_TRIP_COST = 0.0014

MAX_HOLD_CANDLES = 12
COOLDOWN_CANDLES = 12

TRAIN_DAYS = 120
VALIDATION_DAYS = 63

# TP / SL combinations to test
TP_SL_CONFIGS = [
    (0.50, 0.50),
    (0.75, 0.50),
    (1.00, 0.50),

    (0.75, 0.75),
    (1.00, 0.75),
    (1.25, 0.75),

    (1.00, 1.00),
    (1.25, 1.00),
    (1.50, 1.00),

    (1.50, 1.25),
    (2.00, 1.00),
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_data():

    print("\n" + "=" * 70)
    print("🚀 BTC 6-MONTH TP/SL STRUCTURE TEST")
    print("=" * 70)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    rows = []
    cursor = start_time

    while cursor < end_time:

        chunk_end = min(
            cursor + timedelta(seconds=GRANULARITY * CHUNK_CANDLES),
            end_time
        )

        print(
            f"Downloading: "
            f"{cursor.strftime('%Y-%m-%d %H:%M')} "
            f"to "
            f"{chunk_end.strftime('%Y-%m-%d %H:%M')}"
        )

        url = (
            f"https://api.exchange.coinbase.com/"
            f"products/{PRODUCT}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat()
        }

        try:
            r = requests.get(
                url,
                params=params,
                timeout=30
            )

            r.raise_for_status()

            data = r.json()

            if isinstance(data, list):
                rows.extend(data)

        except Exception as e:

            print("❌ Download error:", e)
            time.sleep(2)
            continue

        cursor = chunk_end
        time.sleep(0.25)

    if not rows:
        raise RuntimeError("No BTC data downloaded.")

    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "low",
            "high",
            "open",
            "close",
            "volume"
        ]
    )

    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    )

    df = df.drop_duplicates("time")
    df = df.sort_values("time")
    df = df.reset_index(drop=True)

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

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print("Total candles:", len(df))
    print("First candle :", df["time"].iloc[0])
    print("Last candle  :", df["time"].iloc[-1])

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


# ============================================================
# 15M INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

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

    df["rsi"] = calculate_rsi(df["close"])

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

    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()

    df["volume_avg"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    df["body_atr"] = (
        (df["close"] - df["open"]).abs()
        / df["atr"]
    )

    df["bullish"] = (
        df["close"] > df["open"]
    )

    df["bearish"] = (
        df["close"] < df["open"]
    )

    df["ema_distance"] = (
        (df["close"] - df["ema21"]).abs()
        / df["atr"]
    )

    return df


# ============================================================
# 1H REGIME
# ============================================================

def build_1h_regime(df):

    hourly = (
        df.set_index("time")
        .resample("1h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
        .reset_index()
    )

    hourly["ema20"] = hourly["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    hourly["ema50"] = hourly["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    hourly["ema100"] = hourly["close"].ewm(
        span=100,
        adjust=False
    ).mean()

    hourly["slope20"] = (
        hourly["ema20"] -
        hourly["ema20"].shift(3)
    )

    hourly["slope50"] = (
        hourly["ema50"] -
        hourly["ema50"].shift(3)
    )

    hourly["regime"] = "SIDEWAYS"

    strong_up = (
        (hourly["close"] > hourly["ema20"]) &
        (hourly["ema20"] > hourly["ema50"]) &
        (hourly["ema50"] > hourly["ema100"]) &
        (hourly["slope20"] > 0) &
        (hourly["slope50"] > 0)
    )

    strong_down = (
        (hourly["close"] < hourly["ema20"]) &
        (hourly["ema20"] < hourly["ema50"]) &
        (hourly["ema50"] < hourly["ema100"]) &
        (hourly["slope20"] < 0) &
        (hourly["slope50"] < 0)
    )

    normal_up = (
        (hourly["close"] > hourly["ema20"]) &
        (hourly["ema20"] > hourly["ema50"])
    )

    normal_down = (
        (hourly["close"] < hourly["ema20"]) &
        (hourly["ema20"] < hourly["ema50"])
    )

    hourly.loc[normal_up, "regime"] = "NORMAL_UP"
    hourly.loc[normal_down, "regime"] = "NORMAL_DOWN"
    hourly.loc[strong_up, "regime"] = "STRONG_UP"
    hourly.loc[strong_down, "regime"] = "STRONG_DOWN"

    # Completed 1H candle only
    hourly["available_time"] = (
        hourly["time"] +
        pd.Timedelta(hours=1)
    )

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    return hourly[
        ["available_time", "regime"]
    ]


# ============================================================
# MERGE 1H
# ============================================================

def merge_regime(df, hourly):

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    return pd.merge_asof(
        df.sort_values("time"),
        hourly.sort_values("available_time"),
        left_on="time",
        right_on="available_time",
        direction="backward"
    )


# ============================================================
# SAME ENTRY LOGIC AS PREVIOUS TEST
# ============================================================

def get_signal(row):

    if row["regime"] == "SIDEWAYS":
        return "WAIT"

    buy = 0
    sell = 0

    # BUY trend
    if row["regime"] in [
        "STRONG_UP",
        "NORMAL_UP"
    ]:
        buy += 2

    if (
        row["ema9"] >
        row["ema21"] >
        row["ema50"]
    ):
        buy += 2

    if row["close"] > row["ema21"]:
        buy += 1

    if row["macd_hist"] > 0:
        buy += 2

    if 50 <= row["rsi"] <= 68:
        buy += 1

    if row["volume_ratio"] >= 1.2:
        buy += 1

    if (
        row["bullish"] and
        row["body_atr"] >= 0.25
    ):
        buy += 1

    if row["ema_distance"] > 1.5:
        buy -= 3

    # SELL trend
    if row["regime"] in [
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:
        sell += 2

    if (
        row["ema9"] <
        row["ema21"] <
        row["ema50"]
    ):
        sell += 2

    if row["close"] < row["ema21"]:
        sell += 1

    if row["macd_hist"] < 0:
        sell += 2

    if 32 <= row["rsi"] <= 50:
        sell += 1

    if row["volume_ratio"] >= 1.2:
        sell += 1

    if (
        row["bearish"] and
        row["body_atr"] >= 0.25
    ):
        sell += 1

    if row["ema_distance"] > 1.5:
        sell -= 3

    if buy >= 6 and buy > sell:
        return "BUY"

    if sell >= 6 and sell > buy:
        return "SELL"

    return "WAIT"


# ============================================================
# ONE TRADE
# ============================================================

def simulate_trade(
    df,
    index,
    direction,
    tp_atr,
    sl_atr
):

    entry = float(df.iloc[index]["close"])
    atr = float(df.iloc[index]["atr"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    risk_pct = (
        sl_atr * atr / entry
    )

    if risk_pct <= 0:
        return None

    if direction == "BUY":

        tp = entry + tp_atr * atr
        sl = entry - sl_atr * atr

    else:

        tp = entry - tp_atr * atr
        sl = entry + sl_atr * atr

    end = min(
        index + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    mfe = 0
    mae = 0

    for j in range(index + 1, end + 1):

        high = float(df.iloc[j]["high"])
        low = float(df.iloc[j]["low"])

        if direction == "BUY":

            mfe = max(
                mfe,
                high / entry - 1
            )

            mae = min(
                mae,
                low / entry - 1
            )

            # Conservative: SL first
            if low <= sl and high >= tp:
                gross = -risk_pct
                net = gross - ROUND_TRIP_COST

                return (
                    "SL",
                    net,
                    net / risk_pct,
                    mfe,
                    mae
                )

            if low <= sl:

                gross = -risk_pct
                net = gross - ROUND_TRIP_COST

                return (
                    "SL",
                    net,
                    net / risk_pct,
                    mfe,
                    mae
                )

            if high >= tp:

                reward_pct = (
                    tp_atr * atr / entry
                )

                gross = reward_pct
                net = gross - ROUND_TRIP_COST

                return (
                    "TP",
                    net,
                    net / risk_pct,
                    mfe,
                    mae
                )

        else:

            mfe = max(
                mfe,
                entry / low - 1
            )

            mae = min(
                mae,
                entry / high - 1
            )

            if high >= sl and low <= tp:

                gross = -risk_pct
                net = gross - ROUND_TRIP_COST

                return (
                    "SL",
                    net,
                    net / risk_pct,
                    mfe,
                    mae
                )

            if high >= sl:

                gross = -risk_pct
                net = gross - ROUND_TRIP_COST

                return (
                    "SL",
                    net,
                    net / risk_pct,
                    mfe,
                    mae
                )

            if low <= tp:

                reward_pct = (
                    tp_atr * atr / entry
                )

                gross = reward_pct
                net = gross - ROUND_TRIP_COST

                return (
                    "TP",
                    net,
                    net / risk_pct,
                    mfe,
                    mae
                )

    # TIMEOUT
    exit_price = float(
        df.iloc[end]["close"]
    )

    if direction == "BUY":

        gross = (
            exit_price / entry - 1
        )

    else:

        gross = (
            entry / exit_price - 1
        )

    net = gross - ROUND_TRIP_COST

    return (
        "TIMEOUT",
        net,
        net / risk_pct,
        mfe,
        mae
    )


# ============================================================
# RUN CONFIGURATION
# ============================================================

def run_config(
    df,
    start,
    end,
    tp_atr,
    sl_atr
):

    results = []

    i = start

    while i < end - MAX_HOLD_CANDLES:

        signal = get_signal(
            df.iloc[i]
        )

        if signal == "WAIT":

            i += 1
            continue

        result = simulate_trade(
            df,
            i,
            signal,
            tp_atr,
            sl_atr
        )

        if result is None:

            i += 1
            continue

        outcome, net, r, mfe, mae = result

        results.append({
            "time": df.iloc[i]["time"],
            "direction": signal,
            "regime": df.iloc[i]["regime"],
            "result": outcome,
            "net_return": net,
            "R": r,
            "mfe": mfe,
            "mae": mae
        })

        # Prevent overlapping trades
        i += COOLDOWN_CANDLES

    return pd.DataFrame(results)


# ============================================================
# REPORT
# ============================================================

def summarize(trades):

    if trades.empty:
        return None

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

    return {
        "trades": total,
        "tp_pct": tp / total * 100,
        "sl_pct": sl / total * 100,
        "timeout_pct": timeout / total * 100,
        "total_return": trades["net_return"].sum(),
        "avg_return": trades["net_return"].mean(),
        "total_r": trades["R"].sum(),
        "avg_r": trades["R"].mean(),
        "avg_mfe": trades["mfe"].mean(),
        "avg_mae": trades["mae"].mean()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_data()

    print("\n📊 Calculating indicators...")
    df = calculate_indicators(df)

    print("📊 Building 1H regime...")
    hourly = build_1h_regime(df)

    print("📊 Merging 1H regime...")
    df = merge_regime(
        df,
        hourly
    )

    df = df.dropna(
        subset=[
            "ema50",
            "rsi",
            "macd_hist",
            "atr",
            "volume_ratio",
            "regime"
        ]
    ).reset_index(drop=True)

    print(
        "Final usable candles:",
        len(df)
    )

    # ========================================================
    # TRAIN / VALIDATION
    # ========================================================

    last_time = df["time"].iloc[-1]

    validation_start = (
        last_time -
        pd.Timedelta(days=VALIDATION_DAYS)
    )

    train_start = (
        validation_start -
        pd.Timedelta(days=TRAIN_DAYS)
    )

    train_idx = df.index[
        (df["time"] >= train_start) &
        (df["time"] < validation_start)
    ]

    validation_idx = df.index[
        df["time"] >= validation_start
    ]

    train_start_idx = train_idx.min()
    train_end_idx = train_idx.max()

    validation_start_idx = validation_idx.min()
    validation_end_idx = validation_idx.max()

    print("\n" + "=" * 70)
    print("WALK-FORWARD")
    print("=" * 70)

    print(
        "Training:",
        df.iloc[train_start_idx]["time"],
        "→",
        df.iloc[train_end_idx]["time"]
    )

    print(
        "Validation:",
        df.iloc[validation_start_idx]["time"],
        "→",
        df.iloc[validation_end_idx]["time"]
    )

    # ========================================================
    # TRAINING CONFIGURATIONS
    # ========================================================

    print("\n" + "=" * 70)
    print("🧪 TRAINING TP/SL CONFIGURATIONS")
    print("=" * 70)

    training_results = []

    for tp_atr, sl_atr in TP_SL_CONFIGS:

        trades = run_config(
            df,
            train_start_idx,
            train_end_idx,
            tp_atr,
            sl_atr
        )

        s = summarize(trades)

        if s is None:
            continue

        training_results.append({
            "tp": tp_atr,
            "sl": sl_atr,
            **s
        })

        print(
            f"TP {tp_atr:.2f} ATR | "
            f"SL {sl_atr:.2f} ATR | "
            f"Trades {s['trades']:3d} | "
            f"TP {s['tp_pct']:5.1f}% | "
            f"SL {s['sl_pct']:5.1f}% | "
            f"TotalR {s['total_r']:+8.2f} | "
            f"AvgR {s['avg_r']:+.3f}"
        )

    if not training_results:

        print("No training trades.")
        return

    # ========================================================
    # SELECT FROM TRAINING ONLY
    # ========================================================

    positive = [
        x for x in training_results
        if x["avg_r"] > 0
    ]

    if positive:

        selected = max(
            positive,
            key=lambda x: x["avg_r"]
        )

        print(
            "\n✅ Positive training configuration found."
        )

    else:

        selected = max(
            training_results,
            key=lambda x: x["avg_r"]
        )

        print(
            "\n⚠️ No positive training configuration."
        )

        print(
            "Using highest training AvgR only "
            "for diagnostic validation."
        )

    tp_atr = selected["tp"]
    sl_atr = selected["sl"]

    print(
        f"Selected from TRAINING only: "
        f"TP={tp_atr:.2f} ATR, "
        f"SL={sl_atr:.2f} ATR"
    )

    # ========================================================
    # UNSEEN VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("🔬 UNSEEN VALIDATION")
    print("=" * 70)

    validation_trades = run_config(
        df,
        validation_start_idx,
        validation_end_idx,
        tp_atr,
        sl_atr
    )

    s = summarize(
        validation_trades
    )

    if s is None:

        print("No validation trades.")
        return

    print(
        f"Configuration: "
        f"TP {tp_atr:.2f} ATR | "
        f"SL {sl_atr:.2f} ATR"
    )

    print(
        f"Trades       : {s['trades']}"
    )

    print(
        f"TP           : {s['tp_pct']:.1f}%"
    )

    print(
        f"SL           : {s['sl_pct']:.1f}%"
    )

    print(
        f"Timeout      : {s['timeout_pct']:.1f}%"
    )

    print(
        f"Total return : {s['total_return'] * 100:+.2f}%"
    )

    print(
        f"Average return: {s['avg_return'] * 100:+.3f}%"
    )

    print(
        f"Total R      : {s['total_r']:+.2f}"
    )

    print(
        f"Average R    : {s['avg_r']:+.3f}"
    )

    print(
        f"Average MFE  : {s['avg_mfe'] * 100:.3f}%"
    )

    print(
        f"Average MAE  : {s['avg_mae'] * 100:.3f}%"
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION BY DIRECTION")
    print("=" * 70)

    for direction in ["BUY", "SELL"]:

        subset = validation_trades[
            validation_trades["direction"] ==
            direction
        ]

        if subset.empty:
            continue

        ss = summarize(subset)

        print(
            f"{direction:<5} | "
            f"Trades={ss['trades']:3d} | "
            f"TP={ss['tp_pct']:5.1f}% | "
            f"SL={ss['sl_pct']:5.1f}% | "
            f"TotalR={ss['total_r']:+.2f} | "
            f"AvgR={ss['avg_r']:+.3f}"
        )

    # ========================================================
    # REGIME
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION BY 1H REGIME")
    print("=" * 70)

    for regime in sorted(
        validation_trades["regime"].unique()
    ):

        subset = validation_trades[
            validation_trades["regime"] ==
            regime
        ]

        ss = summarize(subset)

        print(
            f"{regime:<15} | "
            f"Trades={ss['trades']:3d} | "
            f"TotalR={ss['total_r']:+.2f} | "
            f"AvgR={ss['avg_r']:+.3f}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    validation_trades.to_csv(
        "btc_tp_sl_validation.csv",
        index=False
    )

    print(
        "\nSaved: btc_tp_sl_validation.csv"
    )

    print("\n" + "=" * 70)
    print("✅ TP/SL TEST COMPLETE")
    print("=" * 70)

    print(
        "⚠️ Historical backtest only."
    )

    print(
        "No live Telegram strategy was changed."
    )


if __name__ == "__main__":
    main()

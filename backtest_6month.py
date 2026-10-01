import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 6-MONTH DYNAMIC EXIT / MFE-MAE BACKTEST
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

# Dynamic exit configurations
EXIT_CONFIGS = [
    ("TP_020", 0.20, 0.50),
    ("TP_030", 0.30, 0.50),
    ("TP_040", 0.40, 0.50),
    ("TP_050", 0.50, 0.50),

    ("TP_030_SL020", 0.30, 0.20),
    ("TP_040_SL025", 0.40, 0.25),
    ("TP_050_SL030", 0.50, 0.30),

    ("TRAIL_020", 0.20, 0.20),
    ("TRAIL_030", 0.30, 0.20),
    ("TRAIL_040", 0.40, 0.25),

    ("TIME_60M", 999, 60),
    ("TIME_90M", 999, 90),
    ("TIME_120M", 999, 120),
]


# ============================================================
# DOWNLOAD
# ============================================================

def download_data():

    print("\n" + "=" * 70)
    print("🚀 BTC 6-MONTH DYNAMIC EXIT BACKTEST")
    print("=" * 70)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    rows = []
    cursor = start_time

    while cursor < end_time:

        chunk_end = min(
            cursor + timedelta(
                seconds=GRANULARITY * CHUNK_CANDLES
            ),
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
        raise RuntimeError(
            "No BTC data downloaded."
        )

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

    df = (
        df
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
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

    df = df.dropna().reset_index(drop=True)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print(
        "Total candles:",
        len(df)
    )

    print(
        "First candle :",
        df["time"].iloc[0]
    )

    print(
        "Last candle  :",
        df["time"].iloc[-1]
    )

    return df


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    series,
    period=14
):

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

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# INDICATORS
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

    df["rsi"] = calculate_rsi(
        df["close"]
    )

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

    df["macd_signal"] = df["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    previous_close = df["close"].shift(1)

    tr1 = (
        df["high"] -
        df["low"]
    )

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        df["tr"]
        .rolling(14)
        .mean()
    )

    df["volume_avg"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    df["body_atr"] = (
        (
            df["close"] -
            df["open"]
        ).abs()
        / df["atr"]
    )

    df["bullish"] = (
        df["close"] >
        df["open"]
    )

    df["bearish"] = (
        df["close"] <
        df["open"]
    )

    df["ema_distance"] = (
        (
            df["close"] -
            df["ema21"]
        ).abs()
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

    hourly["ema20"] = (
        hourly["close"]
        .ewm(span=20, adjust=False)
        .mean()
    )

    hourly["ema50"] = (
        hourly["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    hourly["ema100"] = (
        hourly["close"]
        .ewm(span=100, adjust=False)
        .mean()
    )

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

    hourly.loc[
        normal_up,
        "regime"
    ] = "NORMAL_UP"

    hourly.loc[
        normal_down,
        "regime"
    ] = "NORMAL_DOWN"

    hourly.loc[
        strong_up,
        "regime"
    ] = "STRONG_UP"

    hourly.loc[
        strong_down,
        "regime"
    ] = "STRONG_DOWN"

    # Only completed 1H candle is available
    hourly["available_time"] = (
        hourly["time"] +
        pd.Timedelta(hours=1)
    )

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    return hourly[
        [
            "available_time",
            "regime"
        ]
    ]


# ============================================================
# MERGE 1H REGIME
# ============================================================

def merge_regime(
    df,
    hourly
):

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    return pd.merge_asof(
        df.sort_values("time"),
        hourly.sort_values(
            "available_time"
        ),
        left_on="time",
        right_on="available_time",
        direction="backward"
    )


# ============================================================
# ENTRY SIGNAL
# ============================================================

def get_signal(row):

    if row["regime"] == "SIDEWAYS":
        return "WAIT"

    buy = 0
    sell = 0

    # BUY

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

    # SELL

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
# DYNAMIC TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    index,
    direction,
    config_name,
    target,
    stop
):

    entry = float(
        df.iloc[index]["close"]
    )

    if direction == "BUY":

        favorable_target = (
            entry *
            (1 + target / 100)
        )

        adverse_stop = (
            entry *
            (1 - stop / 100)
        )

    else:

        favorable_target = (
            entry *
            (1 - target / 100)
        )

        adverse_stop = (
            entry *
            (1 + stop / 100)
        )

    max_candles = MAX_HOLD_CANDLES

    # TIME based configurations
    if config_name.startswith("TIME_"):

        minutes = stop
        max_candles = max(
            1,
            int(minutes / 15)
        )

    end = min(
        index + max_candles,
        len(df) - 1
    )

    mfe = 0.0
    mae = 0.0

    entry_price = entry

    for j in range(
        index + 1,
        end + 1
    ):

        high = float(
            df.iloc[j]["high"]
        )

        low = float(
            df.iloc[j]["low"]
        )

        if direction == "BUY":

            favorable = (
                high /
                entry_price - 1
            )

            adverse = (
                low /
                entry_price - 1
            )

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            # For fixed TP/SL
            if not config_name.startswith("TIME_"):

                if (
                    low <= adverse_stop
                    and
                    high >= favorable_target
                ):

                    # Conservative assumption
                    # when both happen in same candle
                    gross = -stop / 100

                    net = (
                        gross -
                        ROUND_TRIP_COST
                    )

                    return {
                        "result": "SL",
                        "return": net,
                        "mfe": mfe,
                        "mae": mae
                    }

                if low <= adverse_stop:

                    gross = -stop / 100

                    net = (
                        gross -
                        ROUND_TRIP_COST
                    )

                    return {
                        "result": "SL",
                        "return": net,
                        "mfe": mfe,
                        "mae": mae
                    }

                if high >= favorable_target:

                    gross = target / 100

                    net = (
                        gross -
                        ROUND_TRIP_COST
                    )

                    return {
                        "result": "TP",
                        "return": net,
                        "mfe": mfe,
                        "mae": mae
                    }

        else:

            favorable = (
                entry_price /
                low - 1
            )

            adverse = (
                entry_price /
                high - 1
            )

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            if not config_name.startswith("TIME_"):

                if (
                    high >= adverse_stop
                    and
                    low <= favorable_target
                ):

                    gross = -stop / 100

                    net = (
                        gross -
                        ROUND_TRIP_COST
                    )

                    return {
                        "result": "SL",
                        "return": net,
                        "mfe": mfe,
                        "mae": mae
                    }

                if high >= adverse_stop:

                    gross = -stop / 100

                    net = (
                        gross -
                        ROUND_TRIP_COST
                    )

                    return {
                        "result": "SL",
                        "return": net,
                        "mfe": mfe,
                        "mae": mae
                    }

                if low <= favorable_target:

                    gross = target / 100

                    net = (
                        gross -
                        ROUND_TRIP_COST
                    )

                    return {
                        "result": "TP",
                        "return": net,
                        "mfe": mfe,
                        "mae": mae
                    }

    # TIME EXIT
    exit_price = float(
        df.iloc[end]["close"]
    )

    if direction == "BUY":

        gross = (
            exit_price /
            entry_price - 1
        )

    else:

        gross = (
            entry_price /
            exit_price - 1
        )

    net = (
        gross -
        ROUND_TRIP_COST
    )

    return {
        "result": "TIMEOUT",
        "return": net,
        "mfe": mfe,
        "mae": mae
    }


# ============================================================
# RUN CONFIG
# ============================================================

def run_config(
    df,
    start,
    end,
    config_name,
    target,
    stop
):

    trades = []

    i = start

    while i < end - 1:

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
            config_name,
            target,
            stop
        )

        trades.append({
            "time": df.iloc[i]["time"],
            "direction": signal,
            "regime": df.iloc[i]["regime"],
            "result": result["result"],
            "return": result["return"],
            "mfe": result["mfe"],
            "mae": result["mae"]
        })

        i += COOLDOWN_CANDLES

    return pd.DataFrame(
        trades
    )


# ============================================================
# SUMMARY
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

    total_return = (
        trades["return"].sum()
    )

    avg_return = (
        trades["return"].mean()
    )

    return {
        "trades": total,
        "tp": tp,
        "sl": sl,
        "timeout": timeout,
        "tp_pct": tp / total * 100,
        "sl_pct": sl / total * 100,
        "timeout_pct": timeout / total * 100,
        "total_return": total_return,
        "avg_return": avg_return,
        "avg_mfe": trades["mfe"].mean(),
        "avg_mae": trades["mae"].mean()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_data()

    print(
        "\n📊 Calculating indicators..."
    )

    df = calculate_indicators(df)

    print(
        "📊 Building 1H regime..."
    )

    hourly = build_1h_regime(df)

    print(
        "📊 Merging 1H regime..."
    )

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
    ).reset_index(
        drop=True
    )

    print(
        "Final usable candles:",
        len(df)
    )

    last_time = df["time"].iloc[-1]

    validation_start = (
        last_time -
        pd.Timedelta(
            days=VALIDATION_DAYS
        )
    )

    train_start = (
        validation_start -
        pd.Timedelta(
            days=TRAIN_DAYS
        )
    )

    train_indices = df.index[
        (df["time"] >= train_start) &
        (df["time"] < validation_start)
    ]

    validation_indices = df.index[
        df["time"] >= validation_start
    ]

    train_start_idx = train_indices.min()
    train_end_idx = train_indices.max()

    validation_start_idx = validation_indices.min()
    validation_end_idx = validation_indices.max()

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
    # TRAINING
    # ========================================================

    print("\n" + "=" * 70)
    print("🧪 TRAINING DYNAMIC EXIT CONFIGURATIONS")
    print("=" * 70)

    training_results = []

    for config_name, target, stop in EXIT_CONFIGS:

        trades = run_config(
            df,
            train_start_idx,
            train_end_idx,
            config_name,
            target,
            stop
        )

        summary = summarize(
            trades
        )

        if summary is None:
            continue

        training_results.append({
            "config": config_name,
            "target": target,
            "stop": stop,
            **summary
        })

        print(
            f"{config_name:<18} "
            f"Trades={summary['trades']:3d} | "
            f"TP={summary['tp_pct']:5.1f}% | "
            f"SL={summary['sl_pct']:5.1f}% | "
            f"Total={summary['total_return']*100:+7.2f}% | "
            f"Avg={summary['avg_return']*100:+.3f}%"
        )

    # ========================================================
    # SELECT TRAINING CONFIG
    # ========================================================

    if not training_results:

        print(
            "\n❌ No training results."
        )

        return

    selected = max(
        training_results,
        key=lambda x: x["avg_return"]
    )

    print("\n" + "=" * 70)
    print("SELECTED FROM TRAINING ONLY")
    print("=" * 70)

    print(
        "Configuration:",
        selected["config"]
    )

    print(
        "Training Avg:",
        f"{selected['avg_return']*100:+.3f}%"
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
        selected["config"],
        selected["target"],
        selected["stop"]
    )

    validation_summary = summarize(
        validation_trades
    )

    if validation_summary is None:

        print(
            "No validation trades."
        )

        return

    print(
        f"Configuration : {selected['config']}"
    )

    print(
        f"Trades        : "
        f"{validation_summary['trades']}"
    )

    print(
        f"TP            : "
        f"{validation_summary['tp_pct']:.1f}%"
    )

    print(
        f"SL            : "
        f"{validation_summary['sl_pct']:.1f}%"
    )

    print(
        f"Timeout       : "
        f"{validation_summary['timeout_pct']:.1f}%"
    )

    print(
        f"Total return  : "
        f"{validation_summary['total_return']*100:+.2f}%"
    )

    print(
        f"Average return: "
        f"{validation_summary['avg_return']*100:+.3f}%"
    )

    print(
        f"Average MFE   : "
        f"{validation_summary['avg_mfe']*100:+.3f}%"
    )

    print(
        f"Average MAE   : "
        f"{validation_summary['avg_mae']*100:+.3f}%"
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION BY DIRECTION")
    print("=" * 70)

    for direction in [
        "BUY",
        "SELL"
    ]:

        subset = validation_trades[
            validation_trades["direction"] ==
            direction
        ]

        summary = summarize(
            subset
        )

        if summary is None:
            continue

        print(
            f"{direction:<5} | "
            f"Trades={summary['trades']:3d} | "
            f"TP={summary['tp_pct']:5.1f}% | "
            f"SL={summary['sl_pct']:5.1f}% | "
            f"Total={summary['total_return']*100:+.2f}% | "
            f"Avg={summary['avg_return']*100:+.3f}%"
        )

    # ========================================================
    # REGIME
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION BY 1H REGIME")
    print("=" * 70)

    for regime in sorted(
        validation_trades[
            "regime"
        ].unique()
    ):

        subset = validation_trades[
            validation_trades["regime"] ==
            regime
        ]

        summary = summarize(
            subset
        )

        print(
            f"{regime:<15} | "
            f"Trades={summary['trades']:3d} | "
            f"Total={summary['total_return']*100:+.2f}% | "
            f"Avg={summary['avg_return']*100:+.3f}%"
        )

    # ========================================================
    # SAVE
    # ========================================================

    validation_trades.to_csv(
        "btc_dynamic_exit_validation.csv",
        index=False
    )

    print(
        "\nSaved:"
        " btc_dynamic_exit_validation.csv"
    )

    print("\n" + "=" * 70)
    print("✅ DYNAMIC EXIT TEST COMPLETE")
    print("=" * 70)

    print(
        "⚠️ Historical backtest only."
    )

    print(
        "No live Telegram strategy was changed."
    )


if __name__ == "__main__":
    main()

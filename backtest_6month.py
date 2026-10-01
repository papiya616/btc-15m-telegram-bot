import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 6-MONTH REAL TRADE VALIDATION
# 1H TREND + 15M ENTRY + VOLUME + MACD + RSI
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900
DAYS = 183
CHUNK_CANDLES = 250

# Trade settings
TP_ATR = 1.5
SL_ATR = 1.0

MAX_HOLD_CANDLES = 12       # 3 hours
COOLDOWN_CANDLES = 12       # 3 hours

ROUND_TRIP_COST = 0.0014    # 0.14%

# Walk-forward
TRAIN_DAYS = 120             # ~4 months
VALIDATION_DAYS = 63         # ~2 months


# ============================================================
# DOWNLOAD
# ============================================================

def download_data():

    print("\n" + "=" * 70)
    print("🚀 BTC 6-MONTH REAL TRADE VALIDATION")
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

        url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

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

    for c in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[c] = pd.to_numeric(
            df[c],
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

def rsi(series, period=14):

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

def calculate_15m(df):

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

    df["rsi"] = rsi(df["close"])

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

    # Candle body
    df["body"] = (
        (df["close"] - df["open"]).abs()
        / df["atr"]
    )

    # Candle direction
    df["bullish_candle"] = (
        df["close"] > df["open"]
    )

    df["bearish_candle"] = (
        df["close"] < df["open"]
    )

    # Distance from EMA21
    df["ema21_distance"] = (
        (df["close"] - df["ema21"]).abs()
        / df["atr"]
    )

    return df


# ============================================================
# 1H REGIME
# ============================================================

def build_1h(df):

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

    # Only completed 1H candle becomes available
    hourly["available_time"] = (
        hourly["time"] +
        pd.Timedelta(hours=1)
    )

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    return hourly[
        [
            "available_time",
            "regime"
        ]
    ]


# ============================================================
# MERGE 1H REGIME
# ============================================================

def merge_regime(df, hourly):

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    df = pd.merge_asof(
        df.sort_values("time"),
        hourly.sort_values("available_time"),
        left_on="time",
        right_on="available_time",
        direction="backward"
    )

    return df


# ============================================================
# SIGNAL
# ============================================================

def get_signal(row):

    regime = row["regime"]

    # SIDEWAYS = no trade
    if regime == "SIDEWAYS":
        return "WAIT"

    # ========================================================
    # BUY
    # ========================================================

    buy = 0

    if regime in [
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

    # RSI:
    # We don't chase extreme RSI blindly.
    if 50 <= row["rsi"] <= 68:
        buy += 1

    # Strong volume
    if row["volume_ratio"] >= 1.2:
        buy += 1

    # Bullish candle
    if (
        row["bullish_candle"] and
        row["body"] >= 0.25
    ):
        buy += 1

    # Don't enter if extremely extended
    if row["ema21_distance"] > 1.5:
        buy -= 3

    # ========================================================
    # SELL
    # ========================================================

    sell = 0

    if regime in [
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
        row["bearish_candle"] and
        row["body"] >= 0.25
    ):
        sell += 1

    if row["ema21_distance"] > 1.5:
        sell -= 3

    # ========================================================
    # FINAL
    # ========================================================

    if buy >= 6 and buy > sell:
        return "BUY"

    if sell >= 6 and sell > buy:
        return "SELL"

    return "WAIT"


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, index, direction):

    entry = df.iloc[index]["close"]
    atr = df.iloc[index]["atr"]

    if not np.isfinite(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp = entry + TP_ATR * atr
        sl = entry - SL_ATR * atr

    else:

        tp = entry - TP_ATR * atr
        sl = entry + SL_ATR * atr

    end = min(
        index + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    mfe = 0
    mae = 0

    for j in range(index + 1, end + 1):

        high = df.iloc[j]["high"]
        low = df.iloc[j]["low"]

        if direction == "BUY":

            favorable = (
                high / entry - 1
            )

            adverse = (
                low / entry - 1
            )

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            # Conservative:
            # if both TP and SL touched,
            # assume SL first.
            if low <= sl and high >= tp:
                return {
                    "result": "SL",
                    "return": -SL_ATR / 1.0 -
                             ROUND_TRIP_COST / atr * atr,
                    "mfe": mfe,
                    "mae": mae
                }

            if low <= sl:
                return {
                    "result": "SL",
                    "return": -SL_ATR -
                             ROUND_TRIP_COST / atr * atr,
                    "mfe": mfe,
                    "mae": mae
                }

            if high >= tp:
                return {
                    "result": "TP",
                    "return": TP_ATR -
                             ROUND_TRIP_COST / atr * atr,
                    "mfe": mfe,
                    "mae": mae
                }

        else:

            favorable = (
                entry - low
            ) / entry

            adverse = (
                high - entry
            ) / entry

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            if high >= sl and low <= tp:
                return {
                    "result": "SL",
                    "return": -SL_ATR -
                             ROUND_TRIP_COST / atr * atr,
                    "mfe": mfe,
                    "mae": mae
                }

            if high >= sl:
                return {
                    "result": "SL",
                    "return": -SL_ATR -
                             ROUND_TRIP_COST / atr * atr,
                    "mfe": mfe,
                    "mae": mae
                }

            if low <= tp:
                return {
                    "result": "TP",
                    "return": TP_ATR -
                             ROUND_TRIP_COST / atr * atr,
                    "mfe": mfe,
                    "mae": mae
                }

    # Timeout
    exit_price = df.iloc[end]["close"]

    if direction == "BUY":
        gross = (
            exit_price / entry - 1
        )
    else:
        gross = (
            entry / exit_price - 1
        )

    net = gross - ROUND_TRIP_COST

    return {
        "result": "TIMEOUT",
        "return": net,
        "mfe": mfe,
        "mae": mae
    }


# ============================================================
# FIXED TRADE RESULT CALCULATION
# ============================================================

def simulate_trade(df, index, direction):

    entry = float(df.iloc[index]["close"])
    atr = float(df.iloc[index]["atr"])

    if not np.isfinite(atr) or atr <= 0:
        return None

    if direction == "BUY":
        tp = entry + TP_ATR * atr
        sl = entry - SL_ATR * atr
    else:
        tp = entry - TP_ATR * atr
        sl = entry + SL_ATR * atr

    end = min(
        index + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    mfe = 0.0
    mae = 0.0

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

            # Conservative SL-first
            if low <= sl and high >= tp:
                gross = -SL_ATR * atr / entry
                net = gross - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "net_return": net,
                    "R": net / (SL_ATR * atr / entry),
                    "mfe": mfe,
                    "mae": mae
                }

            if low <= sl:

                gross = -SL_ATR * atr / entry
                net = gross - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "net_return": net,
                    "R": net / (SL_ATR * atr / entry),
                    "mfe": mfe,
                    "mae": mae
                }

            if high >= tp:

                gross = TP_ATR * atr / entry
                net = gross - ROUND_TRIP_COST

                return {
                    "result": "TP",
                    "net_return": net,
                    "R": net / (SL_ATR * atr / entry),
                    "mfe": mfe,
                    "mae": mae
                }

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

                gross = -SL_ATR * atr / entry
                net = gross - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "net_return": net,
                    "R": net / (SL_ATR * atr / entry),
                    "mfe": mfe,
                    "mae": mae
                }

            if high >= sl:

                gross = -SL_ATR * atr / entry
                net = gross - ROUND_TRIP_COST

                return {
                    "result": "SL",
                    "net_return": net,
                    "R": net / (SL_ATR * atr / entry),
                    "mfe": mfe,
                    "mae": mae
                }

            if low <= tp:

                gross = TP_ATR * atr / entry
                net = gross - ROUND_TRIP_COST

                return {
                    "result": "TP",
                    "net_return": net,
                    "R": net / (SL_ATR * atr / entry),
                    "mfe": mfe,
                    "mae": mae
                }

    # ========================================================
    # TIMEOUT
    # ========================================================

    exit_price = float(
        df.iloc[end]["close"]
    )

    if direction == "BUY":
        gross = exit_price / entry - 1
    else:
        gross = entry / exit_price - 1

    net = gross - ROUND_TRIP_COST

    risk = SL_ATR * atr / entry

    return {
        "result": "TIMEOUT",
        "net_return": net,
        "R": net / risk,
        "mfe": mfe,
        "mae": mae
    }


# ============================================================
# RUN PERIOD
# ============================================================

def run_period(df, start, end):

    trades = []

    i = start

    while i < end - MAX_HOLD_CANDLES:

        signal = get_signal(
            df.iloc[i]
        )

        if signal == "WAIT":
            i += 1
            continue

        trade = simulate_trade(
            df,
            i,
            signal
        )

        if trade is None:
            i += 1
            continue

        trade["index"] = i
        trade["time"] = df.iloc[i]["time"]
        trade["direction"] = signal
        trade["regime"] = df.iloc[i]["regime"]

        trades.append(trade)

        # No overlapping trades
        i += COOLDOWN_CANDLES

    return pd.DataFrame(trades)


# ============================================================
# REPORT
# ============================================================

def report(trades, title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if trades.empty:

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

    total_return = (
        trades["net_return"].sum()
    )

    avg_return = (
        trades["net_return"].mean()
    )

    total_r = trades["R"].sum()
    avg_r = trades["R"].mean()

    print("Trades:", total)

    print(
        f"TP      : {tp} "
        f"({tp / total * 100:.1f}%)"
    )

    print(
        f"SL      : {sl} "
        f"({sl / total * 100:.1f}%)"
    )

    print(
        f"Timeout : {timeout} "
        f"({timeout / total * 100:.1f}%)"
    )

    print(
        f"Total return : "
        f"{total_return * 100:+.2f}%"
    )

    print(
        f"Average return : "
        f"{avg_return * 100:+.3f}%"
    )

    print(
        f"Total R : {total_r:+.2f}"
    )

    print(
        f"Average R : {avg_r:+.3f}"
    )

    print(
        f"Average MFE : "
        f"{trades['mfe'].mean() * 100:.3f}%"
    )

    print(
        f"Average MAE : "
        f"{trades['mae'].mean() * 100:.3f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_data()

    print("\n📊 Calculating 15M indicators...")

    df = calculate_15m(df)

    print("📊 Building completed 1H regime...")

    hourly = build_1h(df)

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
    # 120-DAY TRAIN + 63-DAY VALIDATION
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

    train_mask = (
        (df["time"] >= train_start) &
        (df["time"] < validation_start)
    )

    validation_mask = (
        df["time"] >= validation_start
    )

    train_start_idx = df.index[
        train_mask
    ].min()

    train_end_idx = df.index[
        train_mask
    ].max()

    validation_start_idx = df.index[
        validation_mask
    ].min()

    validation_end_idx = df.index[
        validation_mask
    ].max()

    print("\n" + "=" * 70)
    print("WALK-FORWARD PERIODS")
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

    train_trades = run_period(
        df,
        train_start_idx,
        train_end_idx
    )

    report(
        train_trades,
        "🧪 TRAINING RESULTS"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_trades = run_period(
        df,
        validation_start_idx,
        validation_end_idx
    )

    report(
        validation_trades,
        "🔬 UNSEEN VALIDATION RESULTS"
    )

    # ========================================================
    # VALIDATION BY DIRECTION
    # ========================================================

    if not validation_trades.empty:

        for direction in [
            "BUY",
            "SELL"
        ]:

            subset = validation_trades[
                validation_trades["direction"] ==
                direction
            ]

            report(
                subset,
                f"VALIDATION — {direction}"
            )

        # ====================================================
        # VALIDATION BY REGIME
        # ====================================================

        print("\n" + "=" * 70)
        print("🔎 VALIDATION BY 1H REGIME")
        print("=" * 70)

        for regime in sorted(
            validation_trades["regime"].unique()
        ):

            subset = validation_trades[
                validation_trades["regime"] ==
                regime
            ]

            print(
                f"{regime:<15} | "
                f"Trades={len(subset):3d} | "
                f"AvgR={subset['R'].mean():+.3f} | "
                f"TotalR={subset['R'].sum():+.2f}"
            )

        # ====================================================
        # SAVE VALIDATION
        # ====================================================

        validation_trades.to_csv(
            "btc_validation_trades.csv",
            index=False
        )

        print(
            "\nSaved:"
            " btc_validation_trades.csv"
        )

    print("\n" + "=" * 70)
    print("✅ BACKTEST COMPLETE")
    print("=" * 70)

    print(
        "\n⚠️ This is historical testing only."
    )

    print(
        "No live Telegram BUY/SELL rules were changed."
    )


if __name__ == "__main__":
    main()

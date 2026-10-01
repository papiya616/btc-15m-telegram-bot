import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M SIGNAL QUALITY FILTER BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900
CHUNK_CANDLES = 250
LOOKBACK_DAYS = 183

TRAIN_DAYS = 120
VALIDATION_DAYS = 63

ROUND_TRIP_COST = 0.0014
COOLDOWN_CANDLES = 12

MIN_TRAIN_TRADES = 30

# ============================================================
# DOWNLOAD
# ============================================================

def download_candles():

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=LOOKBACK_DAYS)

    all_rows = []

    chunk_seconds = GRANULARITY * CHUNK_CANDLES
    current_start = int(start_time.timestamp())
    final_end = int(end_time.timestamp())

    while current_start < final_end:

        current_end = min(
            current_start + chunk_seconds,
            final_end
        )

        url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

        params = {
            "granularity": GRANULARITY,
            "start": datetime.fromtimestamp(
                current_start, timezone.utc
            ).isoformat(),
            "end": datetime.fromtimestamp(
                current_end, timezone.utc
            ).isoformat()
        }

        print(
            "Downloading:",
            datetime.fromtimestamp(
                current_start, timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "to",
            datetime.fromtimestamp(
                current_end, timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
        )

        try:
            r = requests.get(
                url,
                params=params,
                timeout=30
            )

            r.raise_for_status()

            data = r.json()

            if isinstance(data, list):
                all_rows.extend(data)

        except Exception as e:
            print("Download error:", e)

        current_start = current_end
        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("No BTC data downloaded.")

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

    df["time"] = pd.to_datetime(
        df["time"],
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

    df = (
        df.dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
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

    df["atr"] = true_range.rolling(
        14
    ).mean()

    # Volume ratio
    df["volume_avg"] = df["volume"].rolling(
        20
    ).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    # Candle body
    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_atr"] = (
        df["body"] /
        df["atr"]
    )

    # Distance from EMA21
    df["ema21_distance_atr"] = (
        (df["close"] - df["ema21"]).abs()
        / df["atr"]
    )

    return df


# ============================================================
# 1H REGIME
# ============================================================

def build_1h_regime(df):

    temp = df.set_index("time")

    hourly = temp.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna().reset_index()

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

    hourly["ema20_slope"] = (
        hourly["ema20"] -
        hourly["ema20"].shift(3)
    )

    hourly["ema50_slope"] = (
        hourly["ema50"] -
        hourly["ema50"].shift(3)
    )

    def regime(row):

        if (
            row["ema20"] >
            row["ema50"] >
            row["ema100"] and
            row["ema20_slope"] > 0 and
            row["ema50_slope"] > 0
        ):
            return "STRONG_UP"

        if (
            row["ema20"] <
            row["ema50"] <
            row["ema100"] and
            row["ema20_slope"] < 0 and
            row["ema50_slope"] < 0
        ):
            return "STRONG_DOWN"

        if row["ema20"] > row["ema50"]:
            return "NORMAL_UP"

        if row["ema20"] < row["ema50"]:
            return "NORMAL_DOWN"

        return "SIDEWAYS"

    hourly["regime"] = hourly.apply(
        regime,
        axis=1
    )

    # Only COMPLETED hourly candle information
    hourly["available_time"] = (
        hourly["time"] +
        pd.Timedelta(hours=1)
    )

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    hourly = hourly[
        [
            "available_time",
            "regime"
        ]
    ]

    df["time"] = pd.to_datetime(
        df["time"],
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
# BASE SIGNAL
# ============================================================

def get_base_signal(row):

    if pd.isna(row["regime"]):
        return None

    if row["regime"] == "SIDEWAYS":
        return None

    buy_score = 0
    sell_score = 0

    # 1H regime
    if row["regime"] in [
        "STRONG_UP",
        "NORMAL_UP"
    ]:
        buy_score += 2

    if row["regime"] in [
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:
        sell_score += 2

    # EMA alignment
    if (
        row["ema9"] >
        row["ema21"] >
        row["ema50"]
    ):
        buy_score += 2

    if (
        row["ema9"] <
        row["ema21"] <
        row["ema50"]
    ):
        sell_score += 2

    # Price vs EMA21
    if row["close"] > row["ema21"]:
        buy_score += 1

    if row["close"] < row["ema21"]:
        sell_score += 1

    # MACD
    if row["macd_hist"] > 0:
        buy_score += 2

    if row["macd_hist"] < 0:
        sell_score += 2

    # RSI
    if 50 <= row["rsi"] <= 68:
        buy_score += 1

    if 32 <= row["rsi"] <= 50:
        sell_score += 1

    # Volume
    if row["volume_ratio"] >= 1.2:

        if buy_score > sell_score:
            buy_score += 1

        elif sell_score > buy_score:
            sell_score += 1

    # Candle
    if row["body_atr"] >= 0.25:

        if row["close"] > row["open"]:
            buy_score += 1

        elif row["close"] < row["open"]:
            sell_score += 1

    # Overextension penalty
    if row["ema21_distance_atr"] > 1.5:

        buy_score -= 3
        sell_score -= 3

    if (
        buy_score >= 6 and
        buy_score > sell_score
    ):
        return "BUY"

    if (
        sell_score >= 6 and
        sell_score > buy_score
    ):
        return "SELL"

    return None


# ============================================================
# CONDITION BUCKETS
# ============================================================

def get_conditions(row, direction):

    conditions = []

    # Direction
    conditions.append(
        ("DIRECTION", direction)
    )

    # 1H regime
    conditions.append(
        ("REGIME", row["regime"])
    )

    # RSI
    rsi = row["rsi"]

    if rsi < 35:
        rsi_bucket = "RSI_LT35"
    elif rsi < 45:
        rsi_bucket = "RSI_35_45"
    elif rsi < 55:
        rsi_bucket = "RSI_45_55"
    elif rsi < 65:
        rsi_bucket = "RSI_55_65"
    else:
        rsi_bucket = "RSI_GT65"

    conditions.append(
        ("RSI", rsi_bucket)
    )

    # MACD
    conditions.append(
        (
            "MACD",
            "MACD_UP"
            if row["macd_hist"] > 0
            else "MACD_DOWN"
        )
    )

    # Volume
    vr = row["volume_ratio"]

    if vr < 0.8:
        volume_bucket = "VOL_LT08"
    elif vr <= 1.2:
        volume_bucket = "VOL_08_12"
    else:
        volume_bucket = "VOL_GT12"

    conditions.append(
        ("VOLUME", volume_bucket)
    )

    # EMA21 distance
    dist = row["ema21_distance_atr"]

    if dist < 0.25:
        distance_bucket = "DIST_LT025"
    elif dist < 0.50:
        distance_bucket = "DIST_025_050"
    elif dist < 1.00:
        distance_bucket = "DIST_050_100"
    else:
        distance_bucket = "DIST_GT100"

    conditions.append(
        ("DISTANCE", distance_bucket)
    )

    # Candle body
    body = row["body_atr"]

    if body < 0.25:
        body_bucket = "BODY_LT025"
    elif body < 0.50:
        body_bucket = "BODY_025_050"
    else:
        body_bucket = "BODY_GT050"

    conditions.append(
        ("BODY", body_bucket)
    )

    return conditions


# ============================================================
# FORWARD MOVEMENT
# ============================================================

def calculate_forward_stats(
    df,
    index,
    direction
):

    entry = df.iloc[index]["close"]

    future = df.iloc[
        index + 1:
    ]

    if future.empty:
        return None

    max_high = future["high"].max()
    min_low = future["low"].min()

    if direction == "BUY":

        mfe = (
            (max_high - entry)
            / entry
        )

        mae = (
            (min_low - entry)
            / entry
        )

        ret15 = get_future_return(
            df,
            index,
            1,
            direction
        )

        ret30 = get_future_return(
            df,
            index,
            2,
            direction
        )

        ret60 = get_future_return(
            df,
            index,
            4,
            direction
        )

        ret120 = get_future_return(
            df,
            index,
            8,
            direction
        )

    else:

        mfe = (
            (entry - min_low)
            / entry
        )

        mae = (
            (entry - max_high)
            / entry
        )

        ret15 = get_future_return(
            df,
            index,
            1,
            direction
        )

        ret30 = get_future_return(
            df,
            index,
            2,
            direction
        )

        ret60 = get_future_return(
            df,
            index,
            4,
            direction
        )

        ret120 = get_future_return(
            df,
            index,
            8,
            direction
        )

    return {
        "mfe": mfe,
        "mae": mae,
        "ret15": ret15,
        "ret30": ret30,
        "ret60": ret60,
        "ret120": ret120
    }


def get_future_return(
    df,
    index,
    candles,
    direction
):

    future_index = index + candles

    if future_index >= len(df):
        return np.nan

    entry = df.iloc[index]["close"]
    future = df.iloc[
        future_index
    ]["close"]

    if direction == "BUY":
        return (
            future - entry
        ) / entry

    return (
        entry - future
    ) / entry


# ============================================================
# COLLECT SIGNAL DATA
# ============================================================

def collect_signals(df):

    records = []

    last_trade_index = -999999

    for i in range(
        120,
        len(df) - 8
    ):

        row = df.iloc[i]

        if pd.isna(row["atr"]):
            continue

        direction = get_base_signal(row)

        if direction is None:
            continue

        # Cooldown
        if (
            i - last_trade_index
            < COOLDOWN_CANDLES
        ):
            continue

        stats = calculate_forward_stats(
            df,
            i,
            direction
        )

        if stats is None:
            continue

        # Cost-adjusted returns
        stats["ret15"] -= ROUND_TRIP_COST
        stats["ret30"] -= ROUND_TRIP_COST
        stats["ret60"] -= ROUND_TRIP_COST
        stats["ret120"] -= ROUND_TRIP_COST

        conditions = get_conditions(
            row,
            direction
        )

        record = {
            "time": row["time"],
            "direction": direction,
            "regime": row["regime"],
            "rsi": row["rsi"],
            "volume_ratio": row["volume_ratio"],
            "ema21_distance_atr":
                row["ema21_distance_atr"],
            "body_atr":
                row["body_atr"],
            "mfe": stats["mfe"],
            "mae": stats["mae"],
            "ret15": stats["ret15"],
            "ret30": stats["ret30"],
            "ret60": stats["ret60"],
            "ret120": stats["ret120"],
        }

        for name, value in conditions:
            record[name] = value

        records.append(record)

        last_trade_index = i

    return pd.DataFrame(records)


# ============================================================
# CONDITION TEST
# ============================================================

def analyze_conditions(
    train_df,
    validation_df
):

    candidate_rows = []

    condition_columns = [
        "DIRECTION",
        "REGIME",
        "RSI",
        "MACD",
        "VOLUME",
        "DISTANCE",
        "BODY"
    ]

    # --------------------------------------------------------
    # Single conditions
    # --------------------------------------------------------

    for column in condition_columns:

        for value in train_df[column].dropna().unique():

            subset = train_df[
                train_df[column] == value
            ]

            if len(subset) < MIN_TRAIN_TRADES:
                continue

            avg_return = subset[
                "ret60"
            ].mean()

            candidate_rows.append({
                "type": "SINGLE",
                "condition":
                    f"{column}={value}",
                "column1": column,
                "value1": value,
                "column2": "",
                "value2": "",
                "train_trades":
                    len(subset),
                "train_avg_return":
                    avg_return
            })

    # --------------------------------------------------------
    # Two-condition combinations
    # --------------------------------------------------------

    for a in range(
        len(condition_columns)
    ):

        for b in range(
            a + 1,
            len(condition_columns)
        ):

            col1 = condition_columns[a]
            col2 = condition_columns[b]

            grouped = train_df.groupby(
                [col1, col2]
            )

            for (
                (value1, value2),
                subset
            ) in grouped:

                if len(subset) < MIN_TRAIN_TRADES:
                    continue

                avg_return = subset[
                    "ret60"
                ].mean()

                candidate_rows.append({
                    "type": "DOUBLE",
                    "condition":
                        f"{col1}={value1} AND "
                        f"{col2}={value2}",
                    "column1": col1,
                    "value1": value1,
                    "column2": col2,
                    "value2": value2,
                    "train_trades":
                        len(subset),
                    "train_avg_return":
                        avg_return
                })

    candidates = pd.DataFrame(
        candidate_rows
    )

    if candidates.empty:
        return candidates

    # Only positive training conditions
    candidates = candidates[
        candidates["train_avg_return"] > 0
    ].copy()

    candidates = candidates.sort_values(
        "train_avg_return",
        ascending=False
    )

    return candidates


# ============================================================
# APPLY CONDITION TO VALIDATION
# ============================================================

def apply_condition(
    df,
    candidate
):

    mask = (
        df[candidate["column1"]]
        == candidate["value1"]
    )

    if (
        candidate["type"] == "DOUBLE"
    ):
        mask &= (
            df[candidate["column2"]]
            == candidate["value2"]
        )

    return df[mask]


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("🚀 BTC SIGNAL QUALITY BACKTEST")
    print("==========================================")
    print()

    print("📥 Downloading BTC 15M data...")

    df = download_candles()

    print()
    print(
        "Total candles:",
        len(df)
    )

    print(
        "First candle:",
        df["time"].iloc[0]
    )

    print(
        "Last candle:",
        df["time"].iloc[-1]
    )

    print()

    print("📊 Calculating indicators...")

    df = calculate_indicators(df)

    print(
        "📊 Building completed 1H regime..."
    )

    df = build_1h_regime(df)

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
    # SIGNALS
    # ========================================================

    print()
    print("🔎 Collecting base signals...")

    signals = collect_signals(df)

    if signals.empty:
        print("No signals found.")
        return

    print(
        "Base signals:",
        len(signals)
    )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    start_time = signals["time"].min()

    train_end = (
        start_time +
        pd.Timedelta(days=TRAIN_DAYS)
    )

    validation_end = (
        train_end +
        pd.Timedelta(days=VALIDATION_DAYS)
    )

    train = signals[
        signals["time"] < train_end
    ].copy()

    validation = signals[
        (
            signals["time"] >= train_end
        ) &
        (
            signals["time"] <= validation_end
        )
    ].copy()

    print()
    print("==========================================")
    print("TRAIN / VALIDATION")
    print("==========================================")

    print(
        "Training:",
        train["time"].min(),
        "→",
        train["time"].max()
    )

    print(
        "Validation:",
        validation["time"].min(),
        "→",
        validation["time"].max()
    )

    print(
        "Training trades:",
        len(train)
    )

    print(
        "Validation trades:",
        len(validation)
    )

    # ========================================================
    # TRAIN CONDITIONS
    # ========================================================

    print()
    print("==========================================")
    print("🔎 TRAINING CONDITION SEARCH")
    print("==========================================")

    candidates = analyze_conditions(
        train,
        validation
    )

    if candidates.empty:

        print()
        print(
            "❌ No condition had positive "
            "training average return."
        )

        print(
            "Therefore validation is not "
            "run."
        )

        signals.to_csv(
            "btc_signal_quality_all.csv",
            index=False
        )

        return

    print()
    print(
        "Positive training conditions:",
        len(candidates)
    )

    print()

    print(
        candidates[
            [
                "type",
                "condition",
                "train_trades",
                "train_avg_return"
            ]
        ].head(20).to_string(
            index=False
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("==========================================")
    print("🧪 UNSEEN VALIDATION")
    print("==========================================")

    validation_results = []

    # Test up to 20 conditions selected
    # ONLY from training data.
    selected = candidates.head(20)

    for _, candidate in selected.iterrows():

        subset = apply_condition(
            validation,
            candidate
        )

        if subset.empty:
            continue

        result = candidate.to_dict()

        result["validation_trades"] = len(
            subset
        )

        result["validation_avg_return"] = (
            subset["ret60"].mean()
        )

        result["validation_avg_mfe"] = (
            subset["mfe"].mean()
        )

        result["validation_avg_mae"] = (
            subset["mae"].mean()
        )

        result["validation_15m"] = (
            subset["ret15"].mean()
        )

        result["validation_30m"] = (
            subset["ret30"].mean()
        )

        result["validation_60m"] = (
            subset["ret60"].mean()
        )

        result["validation_120m"] = (
            subset["ret120"].mean()
        )

        result["hit_plus_03"] = (
            subset["mfe"] >= 0.003
        ).mean()

        result["hit_plus_05"] = (
            subset["mfe"] >= 0.005
        ).mean()

        result["bad_move_03"] = (
            subset["mae"] <= -0.003
        ).mean()

        validation_results.append(
            result
        )

    results = pd.DataFrame(
        validation_results
    )

    if results.empty:

        print(
            "No selected training "
            "conditions appeared in validation."
        )

    else:

        results = results.sort_values(
            "validation_avg_return",
            ascending=False
        )

        print()

        print(
            results[
                [
                    "condition",
                    "train_trades",
                    "train_avg_return",
                    "validation_trades",
                    "validation_avg_return",
                    "validation_avg_mfe",
                    "validation_avg_mae",
                    "validation_60m",
                    "hit_plus_03",
                    "hit_plus_05",
                    "bad_move_03"
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # BASELINE
    # ========================================================

    print()
    print("==========================================")
    print("📌 BASELINE")
    print("==========================================")

    print(
        "Validation trades:",
        len(validation)
    )

    print(
        "15M average:",
        f"{validation['ret15'].mean() * 100:.3f}%"
    )

    print(
        "30M average:",
        f"{validation['ret30'].mean() * 100:.3f}%"
    )

    print(
        "60M average:",
        f"{validation['ret60'].mean() * 100:.3f}%"
    )

    print(
        "120M average:",
        f"{validation['ret120'].mean() * 100:.3f}%"
    )

    print(
        "Average MFE:",
        f"{validation['mfe'].mean() * 100:.3f}%"
    )

    print(
        "Average MAE:",
        f"{validation['mae'].mean() * 100:.3f}%"
    )

    # ========================================================
    # SAVE
    # ========================================================

    signals.to_csv(
        "btc_signal_quality_all.csv",
        index=False
    )

    candidates.to_csv(
        "btc_signal_quality_training.csv",
        index=False
    )

    results.to_csv(
        "btc_signal_quality_validation.csv",
        index=False
    )

    print()
    print("==========================================")
    print("💾 FILES SAVED")
    print("==========================================")

    print(
        "btc_signal_quality_all.csv"
    )

    print(
        "btc_signal_quality_training.csv"
    )

    print(
        "btc_signal_quality_validation.csv"
    )

    print()
    print("✅ BACKTEST COMPLETE")
    print()


if __name__ == "__main__":
    main()

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone


# =========================================================
# SETTINGS
# =========================================================

PRODUCT = "BTC-USD"

GRANULARITY = 900          # 15 minutes
DAYS = 183
CHUNK_CANDLES = 250

FORWARD_CANDLES = 12       # 3 hours


# =========================================================
# DOWNLOAD BTC DATA
# =========================================================

def download_data():

    print("🚀 BTC 6-Month Signal Failure Analysis Started!")
    print()
    print("📥 Downloading 6 months of BTC 15M data...")
    print()

    end_time = datetime.now(timezone.utc)

    start_time = (
        end_time -
        timedelta(days=DAYS)
    )

    all_candles = []

    current_start = start_time

    while current_start < end_time:

        current_end = min(
            current_start +
            timedelta(
                seconds=
                GRANULARITY *
                (CHUNK_CANDLES - 1)
            ),
            end_time
        )

        start_epoch = int(
            current_start.timestamp()
        )

        end_epoch = int(
            current_end.timestamp()
        )

        print(
            f"Downloading: "
            f"{current_start.strftime('%Y-%m-%d %H:%M')} "
            f"to "
            f"{current_end.strftime('%Y-%m-%d %H:%M')}"
        )

        url = (
            f"https://api.exchange.coinbase.com/"
            f"products/{PRODUCT}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": start_epoch,
            "end": end_epoch
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
                all_candles.extend(data)

        except Exception as e:

            print(
                "Download error:",
                e
            )

        current_start = (
            current_end +
            timedelta(seconds=GRANULARITY)
        )

        time.sleep(0.25)

    if not all_candles:

        raise RuntimeError(
            "No BTC data downloaded."
        )

    df = pd.DataFrame(
        all_candles,
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

    df["time"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(
        drop=True
    )

    print()
    print(
        f"✅ Total candles downloaded: "
        f"{len(df):,}"
    )

    print(
        f"First candle: "
        f"{df['time'].iloc[0]}"
    )

    print(
        f"Last candle: "
        f"{df['time'].iloc[-1]}"
    )

    return df


# =========================================================
# 15M INDICATORS
# =========================================================

def calculate_indicators(df):

    df = df.copy()

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    delta = df["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    ema12 = df["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["macd"] = (
        ema12 -
        ema26
    )

    df["macd_signal"] = (
        df["macd"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # -----------------------------------------------------
    # ATR
    # -----------------------------------------------------

    previous_close = (
        df["close"].shift(1)
    )

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

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range.rolling(
            14
        ).mean()
    )

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    volume_ma = (
        df["volume"].rolling(
            20
        ).mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        volume_ma.replace(
            0,
            np.nan
        )
    )

    # -----------------------------------------------------
    # CANDLE BODY
    # -----------------------------------------------------

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_atr"] = (
        df["body"] /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    # -----------------------------------------------------
    # EMA21 DISTANCE
    # -----------------------------------------------------

    df["ema21_distance_atr"] = (
        (
            df["close"] -
            df["ema21"]
        ).abs()
        /
        df["atr"].replace(
            0,
            np.nan
        )
    )

    return df


# =========================================================
# 1H REGIME
# =========================================================

def build_1h_regime(df):

    temp = df.copy()

    temp["time"] = pd.to_datetime(
        temp["time"],
        utc=True
    )

    temp = temp.set_index(
        "time"
    )

    hourly = temp.resample(
        "1h",
        label="right",
        closed="right"
    ).agg({

        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"

    })

    hourly = hourly.dropna()

    hourly = hourly.reset_index()

    # -----------------------------------------------------
    # 1H EMA
    # -----------------------------------------------------

    hourly["ema20"] = (
        hourly["close"].ewm(
            span=20,
            adjust=False
        ).mean()
    )

    hourly["ema50"] = (
        hourly["close"].ewm(
            span=50,
            adjust=False
        ).mean()
    )

    hourly["ema100"] = (
        hourly["close"].ewm(
            span=100,
            adjust=False
        ).mean()
    )

    # -----------------------------------------------------
    # Slopes
    # -----------------------------------------------------

    hourly["slope20"] = (
        hourly["ema20"] -
        hourly["ema20"].shift(3)
    )

    hourly["slope50"] = (
        hourly["ema50"] -
        hourly["ema50"].shift(3)
    )

    # -----------------------------------------------------
    # Regime
    # -----------------------------------------------------

    def get_regime(row):

        if (
            row["ema20"] >
            row["ema50"] >
            row["ema100"]
            and
            row["slope20"] > 0
            and
            row["slope50"] > 0
        ):

            return "STRONG_UP"

        if (
            row["ema20"] <
            row["ema50"] <
            row["ema100"]
            and
            row["slope20"] < 0
            and
            row["slope50"] < 0
        ):

            return "STRONG_DOWN"

        if (
            row["ema20"] >
            row["ema50"]
        ):

            return "NORMAL_UP"

        if (
            row["ema20"] <
            row["ema50"]
        ):

            return "NORMAL_DOWN"

        return "SIDEWAYS"

    hourly["regime"] = hourly.apply(
        get_regime,
        axis=1
    )

    hourly["available_time"] = (
        hourly["time"]
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


# =========================================================
# MERGE REGIME
# =========================================================

def merge_regime(df, hourly):

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    hourly = hourly.copy()

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    df = df.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    hourly = hourly.sort_values(
        "available_time"
    ).reset_index(
        drop=True
    )

    regime_data = hourly[
        [
            "available_time",
            "regime"
        ]
    ].copy()

    df = pd.merge_asof(
        df,
        regime_data,
        left_on="time",
        right_on="available_time",
        direction="backward"
    )

    if "available_time" in df.columns:

        df = df.drop(
            columns=[
                "available_time"
            ]
        )

    df["regime"] = (
        df["regime"].fillna(
            "SIDEWAYS"
        )
    )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    )

    df = df.sort_values(
        "time"
    ).reset_index(
        drop=True
    )

    return df


# =========================================================
# BASE SIGNAL
# =========================================================

def get_base_signal(row):

    regime = row["regime"]

    if regime == "SIDEWAYS":

        return "WAIT"

    buy_score = 0
    sell_score = 0

    # -----------------------------------------------------
    # REGIME
    # -----------------------------------------------------

    if regime in [
        "STRONG_UP",
        "NORMAL_UP"
    ]:

        buy_score += 2

    if regime in [
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:

        sell_score += 2

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # PRICE VS EMA21
    # -----------------------------------------------------

    if (
        row["close"] >
        row["ema21"]
    ):

        buy_score += 1

    if (
        row["close"] <
        row["ema21"]
    ):

        sell_score += 1

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    if row["macd_hist"] > 0:

        buy_score += 2

    if row["macd_hist"] < 0:

        sell_score += 2

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    if (
        50 <= row["rsi"] <= 68
    ):

        buy_score += 1

    if (
        32 <= row["rsi"] <= 50
    ):

        sell_score += 1

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if (
        row["volume_ratio"] >= 1.2
    ):

        if buy_score > sell_score:

            buy_score += 1

        elif sell_score > buy_score:

            sell_score += 1

    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

    if (
        row["body_atr"] >= 0.25
    ):

        if (
            row["close"] >
            row["open"]
        ):

            buy_score += 1

        elif (
            row["close"] <
            row["open"]
        ):

            sell_score += 1

    # -----------------------------------------------------
    # OVEREXTENSION
    # -----------------------------------------------------

    if (
        row["ema21_distance_atr"] >
        1.5
    ):

        buy_score -= 3
        sell_score -= 3

    # -----------------------------------------------------
    # FINAL
    # -----------------------------------------------------

    if (
        buy_score >= 6
        and
        buy_score > sell_score
    ):

        return "BUY"

    if (
        sell_score >= 6
        and
        sell_score > buy_score
    ):

        return "SELL"

    return "WAIT"


# =========================================================
# COLLECT SIGNALS
# =========================================================

def collect_signals(df):

    signals = []

    last_signal_index = -999

    for i in range(
        120,
        len(df) - FORWARD_CANDLES - 1
    ):

        if (
            i -
            last_signal_index
        ) < 12:

            continue

        signal = get_base_signal(
            df.iloc[i]
        )

        if signal in [
            "BUY",
            "SELL"
        ]:

            signals.append({
                "index": i,
                "signal": signal
            })

            last_signal_index = i

    return signals


# =========================================================
# ANALYZE ONE SIGNAL
# =========================================================

def analyze_signal(
    df,
    index,
    direction
):

    row = df.iloc[index]

    entry_price = row["close"]

    end_index = min(
        index +
        FORWARD_CANDLES,
        len(df) - 1
    )

    future = df.iloc[
        index + 1:
        end_index + 1
    ]

    if future.empty:

        return None

    # -----------------------------------------------------
    # Forward returns
    # -----------------------------------------------------

    returns = {}

    for minutes in [
        15,
        30,
        60,
        120,
        180
    ]:

        candles = (
            minutes // 15
        )

        target_index = (
            index +
            candles
        )

        if (
            target_index <
            len(df)
        ):

            future_price = (
                df.iloc[
                    target_index
                ]["close"]
            )

            if direction == "BUY":

                ret = (
                    future_price -
                    entry_price
                ) / entry_price

            else:

                ret = (
                    entry_price -
                    future_price
                ) / entry_price

            # Cost-adjusted
            ret -= 0.0014

            returns[
                f"return_{minutes}"
            ] = ret

        else:

            returns[
                f"return_{minutes}"
            ] = np.nan

    # -----------------------------------------------------
    # MFE / MAE
    # -----------------------------------------------------

    if direction == "BUY":

        favorable_moves = (
            future["high"] -
            entry_price
        ) / entry_price

        adverse_moves = (
            future["low"] -
            entry_price
        ) / entry_price

    else:

        favorable_moves = (
            entry_price -
            future["low"]
        ) / entry_price

        adverse_moves = (
            entry_price -
            future["high"]
        ) / entry_price

    mfe = favorable_moves.max()

    mae = adverse_moves.min()

    # -----------------------------------------------------
    # Thresholds
    # -----------------------------------------------------

    if direction == "BUY":

        plus_020 = (
            favorable_moves >= 0.002
        ).any()

        plus_030 = (
            favorable_moves >= 0.003
        ).any()

        plus_050 = (
            favorable_moves >= 0.005
        ).any()

        plus_075 = (
            favorable_moves >= 0.0075
        ).any()

        plus_100 = (
            favorable_moves >= 0.010
        ).any()

        minus_020 = (
            adverse_moves <= -0.002
        ).any()

        minus_030 = (
            adverse_moves <= -0.003
        ).any()

        minus_050 = (
            adverse_moves <= -0.005
        ).any()

    else:

        plus_020 = (
            favorable_moves >= 0.002
        ).any()

        plus_030 = (
            favorable_moves >= 0.003
        ).any()

        plus_050 = (
            favorable_moves >= 0.005
        ).any()

        plus_075 = (
            favorable_moves >= 0.0075
        ).any()

        plus_100 = (
            favorable_moves >= 0.010
        ).any()

        minus_020 = (
            adverse_moves <= -0.002
        ).any()

        minus_030 = (
            adverse_moves <= -0.003
        ).any()

        minus_050 = (
            adverse_moves <= -0.005
        ).any()

    # -----------------------------------------------------
    # Condition buckets
    # -----------------------------------------------------

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

    volume = row["volume_ratio"]

    if volume < 0.8:

        volume_bucket = "VOL_LT08"

    elif volume <= 1.2:

        volume_bucket = "VOL_08_12"

    else:

        volume_bucket = "VOL_GT12"

    distance = row[
        "ema21_distance_atr"
    ]

    if distance < 0.25:

        distance_bucket = "DIST_LT025"

    elif distance < 0.50:

        distance_bucket = "DIST_025_050"

    elif distance <= 1.0:

        distance_bucket = "DIST_050_100"

    else:

        distance_bucket = "DIST_GT100"

    body = row["body_atr"]

    if body < 0.25:

        body_bucket = "BODY_LT025"

    elif body < 0.50:

        body_bucket = "BODY_025_050"

    else:

        body_bucket = "BODY_GT050"

    if row["ema9"] > row["ema21"] > row["ema50"]:

        ema_alignment = "EMA_UP"

    elif row["ema9"] < row["ema21"] < row["ema50"]:

        ema_alignment = "EMA_DOWN"

    else:

        ema_alignment = "EMA_MIXED"

    macd_state = (
        "MACD_UP"
        if row["macd_hist"] > 0
        else
        "MACD_DOWN"
    )

    return {

        "time":
            row["time"],

        "direction":
            direction,

        "regime":
            row["regime"],

        "rsi_bucket":
            rsi_bucket,

        "volume_bucket":
            volume_bucket,

        "distance_bucket":
            distance_bucket,

        "body_bucket":
            body_bucket,

        "ema_alignment":
            ema_alignment,

        "macd_state":
            macd_state,

        "rsi":
            rsi,

        "volume_ratio":
            volume,

        "ema_distance_atr":
            distance,

        "body_atr":
            body,

        "mfe":
            mfe,

        "mae":
            mae,

        "plus_020":
            plus_020,

        "plus_030":
            plus_030,

        "plus_050":
            plus_050,

        "plus_075":
            plus_075,

        "plus_100":
            plus_100,

        "minus_020":
            minus_020,

        "minus_030":
            minus_030,

        "minus_050":
            minus_050,

        **returns
    }


# =========================================================
# BUILD ANALYSIS DATA
# =========================================================

def build_analysis(df):

    signals = collect_signals(
        df
    )

    print()
    print(
        f"Base signals: "
        f"{len(signals):,}"
    )

    rows = []

    for n, signal in enumerate(
        signals
    ):

        result = analyze_signal(
            df,
            signal["index"],
            signal["signal"]
        )

        if result is not None:

            rows.append(
                result
            )

    analysis = pd.DataFrame(
        rows
    )

    return analysis


# =========================================================
# SUMMARY
# =========================================================

def print_overall(
    analysis
):

    print()
    print("=" * 80)
    print(
        "OVERALL MARKET MOVEMENT AFTER SIGNAL"
    )
    print("=" * 80)

    print(
        f"Observations: "
        f"{len(analysis):,}"
    )

    for minutes in [
        15,
        30,
        60,
        120,
        180
    ]:

        col = (
            f"return_{minutes}"
        )

        avg = (
            analysis[col].mean()
        )

        up = (
            analysis[col] > 0
        ).mean()

        down = (
            analysis[col] < 0
        ).mean()

        print(
            f"{minutes}M | "
            f"Avg Return "
            f"{avg * 100:.3f}% | "
            f"UP "
            f"{up * 100:.1f}% | "
            f"DOWN "
            f"{down * 100:.1f}%"
        )

    print()
    print(
        f"Average MFE: "
        f"{analysis['mfe'].mean() * 100:.3f}%"
    )

    print(
        f"Average MAE: "
        f"{analysis['mae'].mean() * 100:.3f}%"
    )

    print()

    for col, label in [
        ("plus_020", "+0.20%"),
        ("plus_030", "+0.30%"),
        ("plus_050", "+0.50%"),
        ("plus_075", "+0.75%"),
        ("plus_100", "+1.00%")
    ]:

        print(
            f"Favorable {label}: "
            f"{analysis[col].mean() * 100:.1f}%"
        )

    print()

    for col, label in [
        ("minus_020", "-0.20%"),
        ("minus_030", "-0.30%"),
        ("minus_050", "-0.50%")
    ]:

        print(
            f"Adverse {label}: "
            f"{analysis[col].mean() * 100:.1f}%"
        )


# =========================================================
# GROUP ANALYSIS
# =========================================================

def group_analysis(
    analysis,
    column,
    title
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    grouped = (
        analysis
        .groupby(column)
        .agg(

            count=(
                column,
                "size"
            ),

            avg_60m=(
                "return_60",
                "mean"
            ),

            avg_120m=(
                "return_120",
                "mean"
            ),

            avg_180m=(
                "return_180",
                "mean"
            ),

            avg_mfe=(
                "mfe",
                "mean"
            ),

            avg_mae=(
                "mae",
                "mean"
            ),

            favorable_030=(
                "plus_030",
                "mean"
            ),

            favorable_050=(
                "plus_050",
                "mean"
            ),

            favorable_075=(
                "plus_075",
                "mean"
            ),

            adverse_030=(
                "minus_030",
                "mean"
            ),

            adverse_050=(
                "minus_050",
                "mean"
            )
        )
        .sort_values(
            "count",
            ascending=False
        )
    )

    for idx, row in grouped.iterrows():

        print(
            f"{idx} | "
            f"N={int(row['count'])} | "
            f"60M={row['avg_60m'] * 100:.3f}% | "
            f"120M={row['avg_120m'] * 100:.3f}% | "
            f"180M={row['avg_180m'] * 100:.3f}% | "
            f"MFE={row['avg_mfe'] * 100:.3f}% | "
            f"MAE={row['avg_mae'] * 100:.3f}% | "
            f"+0.3={row['favorable_030'] * 100:.1f}% | "
            f"+0.5={row['favorable_050'] * 100:.1f}% | "
            f"+0.75={row['favorable_075'] * 100:.1f}% | "
            f"-0.3={row['adverse_030'] * 100:.1f}% | "
            f"-0.5={row['adverse_050'] * 100:.1f}%"
        )


# =========================================================
# DIRECTION + CONDITION ANALYSIS
# =========================================================

def direction_analysis(
    analysis
):

    print()
    print("=" * 80)
    print(
        "BUY VS SELL"
    )
    print("=" * 80)

    for direction in [
        "BUY",
        "SELL"
    ]:

        subset = analysis[
            analysis["direction"] ==
            direction
        ]

        if subset.empty:

            continue

        print()

        print(
            f"{direction} | "
            f"N={len(subset)} | "
            f"15M={subset['return_15'].mean() * 100:.3f}% | "
            f"30M={subset['return_30'].mean() * 100:.3f}% | "
            f"60M={subset['return_60'].mean() * 100:.3f}% | "
            f"120M={subset['return_120'].mean() * 100:.3f}% | "
            f"180M={subset['return_180'].mean() * 100:.3f}% | "
            f"MFE={subset['mfe'].mean() * 100:.3f}% | "
            f"MAE={subset['mae'].mean() * 100:.3f}%"
        )


# =========================================================
# CONDITIONAL COMBINATIONS
# =========================================================

def conditional_analysis(
    analysis
):

    print()
    print("=" * 80)
    print(
        "IMPORTANT CONDITION COMBINATIONS"
    )
    print("=" * 80)

    conditions = [

        (
            "STRONG_UP + EMA_UP",
            (
                (analysis["regime"] == "STRONG_UP")
                &
                (analysis["ema_alignment"] == "EMA_UP")
            )
        ),

        (
            "STRONG_DOWN + EMA_DOWN",
            (
                (analysis["regime"] == "STRONG_DOWN")
                &
                (analysis["ema_alignment"] == "EMA_DOWN")
            )
        ),

        (
            "STRONG_UP + MACD_UP",
            (
                (analysis["regime"] == "STRONG_UP")
                &
                (analysis["macd_state"] == "MACD_UP")
            )
        ),

        (
            "STRONG_DOWN + MACD_DOWN",
            (
                (analysis["regime"] == "STRONG_DOWN")
                &
                (analysis["macd_state"] == "MACD_DOWN")
            )
        ),

        (
            "HIGH_VOLUME + EMA_UP",
            (
                (analysis["volume_bucket"] == "VOL_GT12")
                &
                (analysis["ema_alignment"] == "EMA_UP")
            )
        ),

        (
            "HIGH_VOLUME + EMA_DOWN",
            (
                (analysis["volume_bucket"] == "VOL_GT12")
                &
                (analysis["ema_alignment"] == "EMA_DOWN")
            )
        ),

        (
            "RSI_LOW + MACD_DOWN",
            (
                (analysis["rsi_bucket"] == "RSI_LT35")
                &
                (analysis["macd_state"] == "MACD_DOWN")
            )
        ),

        (
            "RSI_HIGH + MACD_UP",
            (
                (analysis["rsi_bucket"] == "RSI_GT65")
                &
                (analysis["macd_state"] == "MACD_UP")
            )
        )
    ]

    for name, mask in conditions:

        subset = analysis[
            mask
        ]

        if len(subset) < 20:

            continue

        print()

        print(
            f"{name} | "
            f"N={len(subset)} | "
            f"60M={subset['return_60'].mean() * 100:.3f}% | "
            f"120M={subset['return_120'].mean() * 100:.3f}% | "
            f"180M={subset['return_180'].mean() * 100:.3f}% | "
            f"MFE={subset['mfe'].mean() * 100:.3f}% | "
            f"MAE={subset['mae'].mean() * 100:.3f}% | "
            f"+0.5={subset['plus_050'].mean() * 100:.1f}% | "
            f"-0.3={subset['minus_030'].mean() * 100:.1f}%"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    # -----------------------------------------------------
    # Download
    # -----------------------------------------------------

    df = download_data()

    # -----------------------------------------------------
    # Indicators
    # -----------------------------------------------------

    df = calculate_indicators(
        df
    )

    # -----------------------------------------------------
    # 1H regime
    # -----------------------------------------------------

    hourly = build_1h_regime(
        df
    )

    # -----------------------------------------------------
    # Merge
    # -----------------------------------------------------

    df = merge_regime(
        df,
        hourly
    )

    # -----------------------------------------------------
    # Clean
    # -----------------------------------------------------

    required = [

        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd_hist",
        "atr",
        "volume_ratio",
        "body_atr",
        "ema21_distance_atr"
    ]

    df = df.dropna(
        subset=required
    ).reset_index(
        drop=True
    )

    print()
    print(
        f"Final usable candles: "
        f"{len(df):,}"
    )

    # -----------------------------------------------------
    # Build analysis
    # -----------------------------------------------------

    print()
    print(
        "📊 Analyzing signal behavior..."
    )

    analysis = build_analysis(
        df
    )

    if analysis.empty:

        print(
            "❌ No analysis observations."
        )

        return

    # -----------------------------------------------------
    # Overall
    # -----------------------------------------------------

    print_overall(
        analysis
    )

    # -----------------------------------------------------
    # Direction
    # -----------------------------------------------------

    direction_analysis(
        analysis
    )

    # -----------------------------------------------------
    # 1H regime
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "regime",
        "1H MARKET REGIME"
    )

    # -----------------------------------------------------
    # EMA
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "ema_alignment",
        "15M EMA ALIGNMENT"
    )

    # -----------------------------------------------------
    # RSI
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "rsi_bucket",
        "RSI CONDITIONS"
    )

    # -----------------------------------------------------
    # MACD
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "macd_state",
        "MACD CONDITIONS"
    )

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "volume_bucket",
        "VOLUME CONDITIONS"
    )

    # -----------------------------------------------------
    # EMA distance
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "distance_bucket",
        "DISTANCE FROM EMA21"
    )

    # -----------------------------------------------------
    # Candle
    # -----------------------------------------------------

    group_analysis(
        analysis,
        "body_bucket",
        "CANDLE STRENGTH"
    )

    # -----------------------------------------------------
    # Combinations
    # -----------------------------------------------------

    conditional_analysis(
        analysis
    )

    # -----------------------------------------------------
    # SAVE FULL CSV
    # -----------------------------------------------------

    analysis.to_csv(
        "btc_signal_failure_analysis.csv",
        index=False
    )

    print()
    print(
        "💾 Saved:"
    )

    print(
        "btc_signal_failure_analysis.csv"
    )

    print()
    print("=" * 80)
    print(
        "✅ SIGNAL FAILURE ANALYSIS COMPLETE"
    )
    print("=" * 80)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()

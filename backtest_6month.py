import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M - 3 HOUR ENTRY QUALITY BACKTEST
# Corrected MFE / MAE
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900
CHUNK_CANDLES = 250
LOOKBACK_DAYS = 183

TRAIN_DAYS = 120
VALIDATION_DAYS = 63

ROUND_TRIP_COST = 0.0014

# Maximum trade duration = 3 hours
MAX_HOLD_CANDLES = 12

# Minimum gap between signals = 3 hours
COOLDOWN_CANDLES = 12

MIN_TRAIN_TRADES = 30


# ============================================================
# DOWNLOAD BTC DATA
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

        print(
            "Downloading:",
            datetime.fromtimestamp(
                current_start,
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M"),
            "to",
            datetime.fromtimestamp(
                current_end,
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M")
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

            print(
                "Download error:",
                e
            )

        current_start = current_end

        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError(
            "No BTC data downloaded."
        )

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
        df
        .dropna()
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

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    df["rsi"] = (
        100 -
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

    df["macd"] = (
        ema12 -
        ema26
    )

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

    # Candle body
    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_atr"] = (
        df["body"] /
        df["atr"]
    )

    # EMA21 distance
    df["ema21_distance_atr"] = (
        (
            df["close"] -
            df["ema21"]
        ).abs()
        /
        df["atr"]
    )

    return df


# ============================================================
# 1H REGIME
# ============================================================

def build_1h_regime(df):

    temp = df.set_index(
        "time"
    )

    hourly = temp.resample(
        "1h"
    ).agg({
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

    def get_regime(row):

        if (
            row["ema20"] >
            row["ema50"] >
            row["ema100"]
            and
            row["ema20_slope"] > 0
            and
            row["ema50_slope"] > 0
        ):
            return "STRONG_UP"

        if (
            row["ema20"] <
            row["ema50"] <
            row["ema100"]
            and
            row["ema20_slope"] < 0
            and
            row["ema50_slope"] < 0
        ):
            return "STRONG_DOWN"

        if row["ema20"] > row["ema50"]:
            return "NORMAL_UP"

        if row["ema20"] < row["ema50"]:
            return "NORMAL_DOWN"

        return "SIDEWAYS"

    hourly["regime"] = hourly.apply(
        get_regime,
        axis=1
    )

    # IMPORTANT:
    # Only use completed 1H candle
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

    hourly = hourly[
        [
            "available_time",
            "regime"
        ]
    ]

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    df = pd.merge_asof(
        df.sort_values("time"),
        hourly.sort_values(
            "available_time"
        ),
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

    # EMA
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

    # Strong volume
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

    # Overextension
    if row["ema21_distance_atr"] > 1.5:

        buy_score -= 3
        sell_score -= 3

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

    return None


# ============================================================
# CORRECT 3-HOUR MFE / MAE
# ============================================================

def calculate_trade_window(
    df,
    index,
    direction
):

    entry_price = df.iloc[index]["close"]

    end_index = min(
        index + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    future = df.iloc[
        index + 1:
        end_index + 1
    ]

    if future.empty:
        return None

    if direction == "BUY":

        max_high = future["high"].max()
        min_low = future["low"].min()

        mfe = (
            max_high -
            entry_price
        ) / entry_price

        mae = (
            min_low -
            entry_price
        ) / entry_price

    else:

        min_low = future["low"].min()
        max_high = future["high"].max()

        mfe = (
            entry_price -
            min_low
        ) / entry_price

        mae = (
            entry_price -
            max_high
        ) / entry_price

    results = {
        "mfe": mfe,
        "mae": mae
    }

    # 15 / 30 / 60 / 120 / 180 minutes
    for minutes in [
        15,
        30,
        60,
        120,
        180
    ]:

        candles = minutes // 15

        target_index = index + candles

        if target_index >= len(df):
            results[
                f"ret_{minutes}"
            ] = np.nan
            continue

        future_close = df.iloc[
            target_index
        ]["close"]

        if direction == "BUY":

            ret = (
                future_close -
                entry_price
            ) / entry_price

        else:

            ret = (
                entry_price -
                future_close
            ) / entry_price

        # subtract assumed round-trip cost
        ret -= ROUND_TRIP_COST

        results[
            f"ret_{minutes}"
        ] = ret

    return results


# ============================================================
# COLLECT BASE SIGNALS
# ============================================================

def collect_signals(df):

    records = []

    last_signal_index = -999999

    for i in range(
        120,
        len(df) - MAX_HOLD_CANDLES
    ):

        row = df.iloc[i]

        if pd.isna(row["atr"]):
            continue

        direction = get_base_signal(
            row
        )

        if direction is None:
            continue

        # 3-hour cooldown
        if (
            i -
            last_signal_index
            <
            COOLDOWN_CANDLES
        ):
            continue

        movement = calculate_trade_window(
            df,
            i,
            direction
        )

        if movement is None:
            continue

        record = {

            "time":
                row["time"],

            "direction":
                direction,

            "regime":
                row["regime"],

            "rsi":
                row["rsi"],

            "volume_ratio":
                row["volume_ratio"],

            "ema21_distance_atr":
                row["ema21_distance_atr"],

            "body_atr":
                row["body_atr"],

            "mfe":
                movement["mfe"],

            "mae":
                movement["mae"]
        }

        for minutes in [
            15,
            30,
            60,
            120,
            180
        ]:

            record[
                f"ret_{minutes}"
            ] = movement[
                f"ret_{minutes}"
            ]

        records.append(record)

        last_signal_index = i

    return pd.DataFrame(records)


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    title,
    data
):

    print()
    print(
        "------------------------------------------"
    )
    print(title)
    print(
        "------------------------------------------"
    )

    if data.empty:

        print("No trades.")
        return

    print(
        "Trades:",
        len(data)
    )

    print(
        "BUY:",
        len(
            data[
                data["direction"] ==
                "BUY"
            ]
        )
    )

    print(
        "SELL:",
        len(
            data[
                data["direction"] ==
                "SELL"
            ]
        )
    )

    for minutes in [
        15,
        30,
        60,
        120,
        180
    ]:

        value = data[
            f"ret_{minutes}"
        ].mean()

        print(
            f"{minutes}M average:",
            f"{value * 100:.3f}%"
        )

    print(
        "Average MFE:",
        f"{data['mfe'].mean() * 100:.3f}%"
    )

    print(
        "Average MAE:",
        f"{data['mae'].mean() * 100:.3f}%"
    )

    print(
        "MFE >= 0.20%:",
        f"{(data['mfe'] >= 0.002).mean() * 100:.1f}%"
    )

    print(
        "MFE >= 0.30%:",
        f"{(data['mfe'] >= 0.003).mean() * 100:.1f}%"
    )

    print(
        "MFE >= 0.50%:",
        f"{(data['mfe'] >= 0.005).mean() * 100:.1f}%"
    )

    print(
        "MFE >= 0.75%:",
        f"{(data['mfe'] >= 0.0075).mean() * 100:.1f}%"
    )

    print(
        "MAE <= -0.20%:",
        f"{(data['mae'] <= -0.002).mean() * 100:.1f}%"
    )

    print(
        "MAE <= -0.30%:",
        f"{(data['mae'] <= -0.003).mean() * 100:.1f}%"
    )

    print(
        "MAE <= -0.50%:",
        f"{(data['mae'] <= -0.005).mean() * 100:.1f}%"
    )


# ============================================================
# CONDITION ANALYSIS
# ============================================================

def condition_analysis(
    train,
    validation
):

    print()
    print(
        "=========================================="
    )
    print(
        "🔎 ENTRY QUALITY CONDITIONS"
    )
    print(
        "=========================================="
    )

    conditions = []

    # Direction
    for direction in [
        "BUY",
        "SELL"
    ]:

        conditions.append(
            (
                f"DIRECTION={direction}",
                lambda d, x=direction:
                    d["direction"] == x
            )
        )

    # Regime
    for regime in [
        "STRONG_UP",
        "NORMAL_UP",
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:

        conditions.append(
            (
                f"REGIME={regime}",
                lambda d, x=regime:
                    d["regime"] == x
            )
        )

    # RSI
    conditions.extend([
        (
            "RSI<35",
            lambda d:
                d["rsi"] < 35
        ),
        (
            "RSI 35-45",
            lambda d:
                (
                    d["rsi"] >= 35
                ) &
                (
                    d["rsi"] < 45
                )
        ),
        (
            "RSI 45-55",
            lambda d:
                (
                    d["rsi"] >= 45
                ) &
                (
                    d["rsi"] < 55
                )
        ),
        (
            "RSI 55-65",
            lambda d:
                (
                    d["rsi"] >= 55
                ) &
                (
                    d["rsi"] < 65
                )
        ),
        (
            "RSI>65",
            lambda d:
                d["rsi"] >= 65
        )
    ])

    # Volume
    conditions.extend([
        (
            "VOLUME<0.8",
            lambda d:
                d["volume_ratio"] < 0.8
        ),
        (
            "VOLUME 0.8-1.2",
            lambda d:
                (
                    d["volume_ratio"] >= 0.8
                ) &
                (
                    d["volume_ratio"] <= 1.2
                )
        ),
        (
            "VOLUME>1.2",
            lambda d:
                d["volume_ratio"] > 1.2
        )
    ])

    # EMA distance
    conditions.extend([
        (
            "DIST<0.25",
            lambda d:
                d["ema21_distance_atr"] < 0.25
        ),
        (
            "DIST 0.25-0.50",
            lambda d:
                (
                    d["ema21_distance_atr"] >= 0.25
                ) &
                (
                    d["ema21_distance_atr"] < 0.50
                )
        ),
        (
            "DIST 0.50-1.00",
            lambda d:
                (
                    d["ema21_distance_atr"] >= 0.50
                ) &
                (
                    d["ema21_distance_atr"] < 1.00
                )
        ),
        (
            "DIST>1.00",
            lambda d:
                d["ema21_distance_atr"] >= 1.00
        )
    ])

    # Candle body
    conditions.extend([
        (
            "BODY<0.25",
            lambda d:
                d["body_atr"] < 0.25
        ),
        (
            "BODY 0.25-0.50",
            lambda d:
                (
                    d["body_atr"] >= 0.25
                ) &
                (
                    d["body_atr"] < 0.50
                )
        ),
        (
            "BODY>0.50",
            lambda d:
                d["body_atr"] >= 0.50
        )
    ])

    rows = []

    for name, function in conditions:

        train_subset = train[
            function(train)
        ]

        if len(train_subset) < MIN_TRAIN_TRADES:
            continue

        rows.append({

            "condition":
                name,

            "train_trades":
                len(train_subset),

            "train_60m":
                train_subset[
                    "ret_60"
                ].mean(),

            "train_180m":
                train_subset[
                    "ret_180"
                ].mean(),

            "train_mfe":
                train_subset[
                    "mfe"
                ].mean(),

            "train_mae":
                train_subset[
                    "mae"
                ].mean()
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    # Only positive 60M training conditions
    result = result[
        result["train_60m"] > 0
    ].copy()

    result = result.sort_values(
        "train_60m",
        ascending=False
    )

    print()

    if result.empty:

        print(
            "No positive training "
            "condition found."
        )

        return result

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# VALIDATE SELECTED CONDITIONS
# ============================================================

def validate_conditions(
    candidates,
    train,
    validation
):

    if candidates.empty:
        return pd.DataFrame()

    print()
    print(
        "=========================================="
    )
    print(
        "🧪 UNSEEN VALIDATION"
    )
    print(
        "=========================================="
    )

    lookup = {}

    # Same conditions used above
    lookup["DIRECTION=BUY"] = (
        lambda d:
            d["direction"] == "BUY"
    )

    lookup["DIRECTION=SELL"] = (
        lambda d:
            d["direction"] == "SELL"
    )

    for regime in [
        "STRONG_UP",
        "NORMAL_UP",
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:

        lookup[
            f"REGIME={regime}"
        ] = (
            lambda d, x=regime:
                d["regime"] == x
        )

    lookup["RSI<35"] = (
        lambda d:
            d["rsi"] < 35
    )

    lookup["RSI 35-45"] = (
        lambda d:
            (
                d["rsi"] >= 35
            ) &
            (
                d["rsi"] < 45
            )
    )

    lookup["RSI 45-55"] = (
        lambda d:
            (
                d["rsi"] >= 45
            ) &
            (
                d["rsi"] < 55
            )
    )

    lookup["RSI 55-65"] = (
        lambda d:
            (
                d["rsi"] >= 55
            ) &
            (
                d["rsi"] < 65
            )
    )

    lookup["RSI>65"] = (
        lambda d:
            d["rsi"] >= 65
    )

    lookup["VOLUME<0.8"] = (
        lambda d:
            d["volume_ratio"] < 0.8
    )

    lookup["VOLUME 0.8-1.2"] = (
        lambda d:
            (
                d["volume_ratio"] >= 0.8
            ) &
            (
                d["volume_ratio"] <= 1.2
            )
    )

    lookup["VOLUME>1.2"] = (
        lambda d:
            d["volume_ratio"] > 1.2
    )

    lookup["DIST<0.25"] = (
        lambda d:
            d["ema21_distance_atr"] < 0.25
    )

    lookup["DIST 0.25-0.50"] = (
        lambda d:
            (
                d["ema21_distance_atr"] >= 0.25
            ) &
            (
                d["ema21_distance_atr"] < 0.50
            )
    )

    lookup["DIST 0.50-1.00"] = (
        lambda d:
            (
                d["ema21_distance_atr"] >= 0.50
            ) &
            (
                d["ema21_distance_atr"] < 1.00
            )
    )

    lookup["DIST>1.00"] = (
        lambda d:
            d["ema21_distance_atr"] >= 1.00
    )

    lookup["BODY<0.25"] = (
        lambda d:
            d["body_atr"] < 0.25
    )

    lookup["BODY 0.25-0.50"] = (
        lambda d:
            (
                d["body_atr"] >= 0.25
            ) &
            (
                d["body_atr"] < 0.50
            )
    )

    lookup["BODY>0.50"] = (
        lambda d:
            d["body_atr"] >= 0.50
    )

    results = []

    # Maximum 10 training-selected conditions
    for _, candidate in candidates.head(10).iterrows():

        name = candidate["condition"]

        if name not in lookup:
            continue

        subset = validation[
            lookup[name](validation)
        ]

        if subset.empty:
            continue

        results.append({

            "condition":
                name,

            "train_trades":
                candidate[
                    "train_trades"
                ],

            "train_60m":
                candidate[
                    "train_60m"
                ],

            "validation_trades":
                len(subset),

            "validation_15m":
                subset[
                    "ret_15"
                ].mean(),

            "validation_30m":
                subset[
                    "ret_30"
                ].mean(),

            "validation_60m":
                subset[
                    "ret_60"
                ].mean(),

            "validation_120m":
                subset[
                    "ret_120"
                ].mean(),

            "validation_180m":
                subset[
                    "ret_180"
                ].mean(),

            "validation_mfe":
                subset[
                    "mfe"
                ].mean(),

            "validation_mae":
                subset[
                    "mae"
                ].mean(),

            "mfe_03":
                (
                    subset["mfe"] >= 0.003
                ).mean(),

            "mfe_05":
                (
                    subset["mfe"] >= 0.005
                ).mean(),

            "mae_03":
                (
                    subset["mae"] <= -0.003
                ).mean()
        })

    result = pd.DataFrame(results)

    if result.empty:

        print(
            "No selected condition "
            "appeared in validation."
        )

        return result

    print()

    print(
        result.to_string(
            index=False
        )
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "=========================================="
    )
    print(
        "🚀 BTC 3-HOUR ENTRY QUALITY BACKTEST"
    )
    print(
        "=========================================="
    )
    print()

    print(
        "📥 Downloading BTC 15M data..."
    )

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

    print(
        "📊 Calculating indicators..."
    )

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
    ).reset_index(
        drop=True
    )

    print(
        "Final usable candles:",
        len(df)
    )

    # --------------------------------------------------------
    # Signals
    # --------------------------------------------------------

    print()
    print(
        "🔎 Collecting base signals..."
    )

    signals = collect_signals(df)

    print(
        "Base signals:",
        len(signals)
    )

    if signals.empty:
        print(
            "No signals found."
        )
        return

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    start_time = signals[
        "time"
    ].min()

    train_end = (
        start_time +
        pd.Timedelta(
            days=TRAIN_DAYS
        )
    )

    validation_end = (
        train_end +
        pd.Timedelta(
            days=VALIDATION_DAYS
        )
    )

    train = signals[
        signals["time"] <
        train_end
    ].copy()

    validation = signals[
        (
            signals["time"] >=
            train_end
        )
        &
        (
            signals["time"] <=
            validation_end
        )
    ].copy()

    print()
    print(
        "=========================================="
    )
    print(
        "TRAIN / VALIDATION"
    )
    print(
        "=========================================="
    )

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

    # --------------------------------------------------------
    # Full baseline
    # --------------------------------------------------------

    print_summary(
        "TRAINING BASELINE",
        train
    )

    print_summary(
        "VALIDATION BASELINE",
        validation
    )

    # --------------------------------------------------------
    # Condition analysis
    # --------------------------------------------------------

    candidates = condition_analysis(
        train,
        validation
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validation_results = (
        validate_conditions(
            candidates,
            train,
            validation
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    signals.to_csv(
        "btc_entry_quality_3h_all.csv",
        index=False
    )

    candidates.to_csv(
        "btc_entry_quality_3h_training.csv",
        index=False
    )

    validation_results.to_csv(
        "btc_entry_quality_3h_validation.csv",
        index=False
    )

    print()
    print(
        "=========================================="
    )
    print(
        "💾 FILES SAVED"
    )
    print(
        "=========================================="
    )

    print(
        "btc_entry_quality_3h_all.csv"
    )

    print(
        "btc_entry_quality_3h_training.csv"
    )

    print(
        "btc_entry_quality_3h_validation.csv"
    )

    print()
    print(
        "✅ BACKTEST COMPLETE"
    )
    print()


if __name__ == "__main__":
    main()

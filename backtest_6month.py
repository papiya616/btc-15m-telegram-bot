import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone


# ============================================================
# SETTINGS
# ============================================================

PRODUCT = "BTC-USD"

GRANULARITY = 900          # 15 minutes
CHUNK_CANDLES = 250

DAYS = 183

TP_ATR = 1.5
SL_ATR = 1.0

MAX_HOLD_CANDLES = 12      # 180 minutes

ROUND_TRIP_COST = 0.0014   # 0.14%

COOLDOWN_CANDLES = 12

TRAIN_DAYS = 60
VALIDATION_DAYS = 30


# ============================================================
# DOWNLOAD 15M DATA
# ============================================================

def download_data():

    print("\n📥 Downloading 6 months of BTC 15M data...\n")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current = start_time

    chunk_minutes = 15 * CHUNK_CANDLES

    while current < end_time:

        chunk_end = min(
            current + timedelta(minutes=chunk_minutes),
            end_time
        )

        print(
            "Downloading:",
            current.strftime("%Y-%m-%d %H:%M"),
            "to",
            chunk_end.strftime("%Y-%m-%d %H:%M")
        )

        url = (
            "https://api.exchange.coinbase.com/"
            "products/BTC-USD/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": current.isoformat(),
            "end": chunk_end.isoformat()
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

            print("❌ Download error:", e)

            time.sleep(2)

        current = chunk_end

        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError(
            "No market data downloaded."
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

    # IMPORTANT:
    # Force exact same datetime dtype.
    df["time"] = pd.to_datetime(
        df["time"],
        unit="s",
        utc=True
    ).astype("datetime64[ns, UTC]")

    numeric_cols = [
        "low",
        "high",
        "open",
        "close",
        "volume"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = (
        df
        .dropna()
        .drop_duplicates(
            subset=["time"]
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    print("\n✅ Data download complete.")

    print(
        "Total candles:",
        len(df)
    )

    print(
        "First candle:",
        df["time"].iloc[0]
    )

    print(
        "Last candle: ",
        df["time"].iloc[-1]
    )

    return df


# ============================================================
# 15M INDICATORS
# ============================================================

def add_indicators(df):

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

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        tr.rolling(14).mean()
    )

    # Volume

    df["volume_ma20"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_ma20"]
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

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(
            0,
            np.nan
        )
    )

    return df


# ============================================================
# BUILD COMPLETED 1H CANDLES
# ============================================================

def build_1h_data(df):

    temp = df.copy()

    temp["time"] = pd.to_datetime(
        temp["time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    temp = temp.set_index(
        "time"
    )

    h1 = (
        temp
        .resample(
            "1h",
            label="left",
            closed="left"
        )
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

    # A 1H candle becomes known only after
    # that hour has completed.

    h1["available_time"] = (
        h1["time"] +
        pd.Timedelta(hours=1)
    )

    h1["available_time"] = pd.to_datetime(
        h1["available_time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    # 1H EMA

    h1["ema20"] = (
        h1["close"]
        .ewm(
            span=20,
            adjust=False
        )
        .mean()
    )

    h1["ema50"] = (
        h1["close"]
        .ewm(
            span=50,
            adjust=False
        )
        .mean()
    )

    h1["ema100"] = (
        h1["close"]
        .ewm(
            span=100,
            adjust=False
        )
        .mean()
    )

    # 1H ATR

    prev_close = (
        h1["close"].shift(1)
    )

    tr1 = (
        h1["high"] -
        h1["low"]
    )

    tr2 = (
        h1["high"] -
        prev_close
    ).abs()

    tr3 = (
        h1["low"] -
        prev_close
    ).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    h1["atr"] = (
        tr.rolling(14).mean()
    )

    # Trend slope

    h1["ema20_slope"] = (
        h1["ema20"] -
        h1["ema20"].shift(3)
    )

    h1["ema50_slope"] = (
        h1["ema50"] -
        h1["ema50"].shift(3)
    )

    return h1


# ============================================================
# MERGE 1H + 15M
# ============================================================

def merge_timeframes(df, h1):

    df = df.copy()
    h1 = h1.copy()

    # --------------------------------------------------------
    # IMPORTANT FIX:
    # Force BOTH merge columns to exactly the same dtype.
    # --------------------------------------------------------

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    h1["available_time"] = pd.to_datetime(
        h1["available_time"],
        utc=True
    ).astype(
        "datetime64[ns, UTC]"
    )

    df = (
        df
        .sort_values("time")
        .reset_index(drop=True)
    )

    h1 = (
        h1
        .sort_values("available_time")
        .reset_index(drop=True)
    )

    h1_cols = [
        "available_time",
        "close",
        "ema20",
        "ema50",
        "ema100",
        "atr",
        "ema20_slope",
        "ema50_slope"
    ]

    h1 = h1[h1_cols].copy()

    h1 = h1.rename(
        columns={
            "close": "h1_close",
            "ema20": "h1_ema20",
            "ema50": "h1_ema50",
            "ema100": "h1_ema100",
            "atr": "h1_atr",
            "ema20_slope":
                "h1_ema20_slope",
            "ema50_slope":
                "h1_ema50_slope"
        }
    )

    # Final dtype safety check

    df["time"] = (
        pd.to_datetime(
            df["time"],
            utc=True
        )
        .dt.tz_convert("UTC")
        .astype("datetime64[ns, UTC]")
    )

    h1["available_time"] = (
        pd.to_datetime(
            h1["available_time"],
            utc=True
        )
        .dt.tz_convert("UTC")
        .astype("datetime64[ns, UTC]")
    )

    merged = pd.merge_asof(
        df,
        h1,
        left_on="time",
        right_on="available_time",
        direction="backward"
    )

    return merged


# ============================================================
# 1H REGIME
# ============================================================

def get_1h_regime(row):

    values = [
        row["h1_close"],
        row["h1_ema20"],
        row["h1_ema50"],
        row["h1_ema100"],
        row["h1_ema20_slope"],
        row["h1_ema50_slope"]
    ]

    if any(
        pd.isna(x)
        for x in values
    ):

        return "UNKNOWN"

    # Strong bullish

    if (
        row["h1_close"] >
        row["h1_ema20"]
        and
        row["h1_ema20"] >
        row["h1_ema50"]
        and
        row["h1_ema50"] >
        row["h1_ema100"]
        and
        row["h1_ema20_slope"] > 0
        and
        row["h1_ema50_slope"] > 0
    ):

        return "STRONG_UP"

    # Strong bearish

    if (
        row["h1_close"] <
        row["h1_ema20"]
        and
        row["h1_ema20"] <
        row["h1_ema50"]
        and
        row["h1_ema50"] <
        row["h1_ema100"]
        and
        row["h1_ema20_slope"] < 0
        and
        row["h1_ema50_slope"] < 0
    ):

        return "STRONG_DOWN"

    # Normal bullish

    if (
        row["h1_close"] >
        row["h1_ema50"]
        and
        row["h1_ema20"] >
        row["h1_ema50"]
    ):

        return "NORMAL_UP"

    # Normal bearish

    if (
        row["h1_close"] <
        row["h1_ema50"]
        and
        row["h1_ema20"] <
        row["h1_ema50"]
    ):

        return "NORMAL_DOWN"

    return "SIDEWAYS"


# ============================================================
# 15M SIGNAL
# ============================================================

def get_signal(row):

    regime = get_1h_regime(row)

    # ========================================================
    # BUY
    # ========================================================

    if regime in [
        "STRONG_UP",
        "NORMAL_UP"
    ]:

        trend_ok = (
            row["ema9"] >
            row["ema21"] >
            row["ema50"]
        )

        pullback_distance = abs(
            row["close"] -
            row["ema21"]
        )

        pullback_ok = (
            pullback_distance <=
            0.9 * row["atr"]
        )

        bullish_candle = (
            row["close"] >
            row["open"]
            and
            row["body_ratio"] >= 0.30
        )

        momentum_ok = (
            row["macd_hist"] > 0
            and
            row["rsi"] >= 48
            and
            row["rsi"] <= 68
        )

        volume_ok = (
            row["volume_ratio"] >= 0.85
        )

        not_extended = (
            abs(
                row["close"] -
                row["ema21"]
            )
            <=
            1.5 * row["atr"]
        )

        if (
            trend_ok
            and pullback_ok
            and bullish_candle
            and momentum_ok
            and volume_ok
            and not_extended
        ):

            return "BUY", regime

    # ========================================================
    # SELL
    # ========================================================

    if regime in [
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:

        trend_ok = (
            row["ema9"] <
            row["ema21"] <
            row["ema50"]
        )

        pullback_distance = abs(
            row["close"] -
            row["ema21"]
        )

        pullback_ok = (
            pullback_distance <=
            0.9 * row["atr"]
        )

        bearish_candle = (
            row["close"] <
            row["open"]
            and
            row["body_ratio"] >= 0.30
        )

        momentum_ok = (
            row["macd_hist"] < 0
            and
            row["rsi"] >= 32
            and
            row["rsi"] <= 52
        )

        volume_ok = (
            row["volume_ratio"] >= 0.85
        )

        not_extended = (
            abs(
                row["close"] -
                row["ema21"]
            )
            <=
            1.5 * row["atr"]
        )

        if (
            trend_ok
            and
            pullback_ok
            and
            bearish_candle
            and
            momentum_ok
            and
            volume_ok
            and
            not_extended
        ):

            return "SELL", regime

    return "WAIT", regime


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    df,
    index,
    direction
):

    row = df.iloc[index]

    entry = row["close"]
    atr = row["atr"]

    if (
        pd.isna(atr)
        or
        atr <= 0
    ):

        return None

    if direction == "BUY":

        tp = (
            entry +
            TP_ATR * atr
        )

        sl = (
            entry -
            SL_ATR * atr
        )

    else:

        tp = (
            entry -
            TP_ATR * atr
        )

        sl = (
            entry +
            SL_ATR * atr
        )

    end_index = min(
        index +
        MAX_HOLD_CANDLES,
        len(df) - 1
    )

    result = "TIMEOUT"

    exit_price = (
        df.iloc[end_index]["close"]
    )

    exit_index = end_index

    max_favorable = 0.0
    max_adverse = 0.0

    for j in range(
        index + 1,
        end_index + 1
    ):

        candle = df.iloc[j]

        if direction == "BUY":

            favorable = (
                candle["high"] -
                entry
            ) / entry

            adverse = (
                candle["low"] -
                entry
            ) / entry

            max_favorable = max(
                max_favorable,
                favorable
            )

            max_adverse = min(
                max_adverse,
                adverse
            )

            hit_tp = (
                candle["high"] >= tp
            )

            hit_sl = (
                candle["low"] <= sl
            )

        else:

            favorable = (
                entry -
                candle["low"]
            ) / entry

            adverse = (
                entry -
                candle["high"]
            ) / entry

            max_favorable = max(
                max_favorable,
                favorable
            )

            max_adverse = min(
                max_adverse,
                adverse
            )

            hit_tp = (
                candle["low"] <= tp
            )

            hit_sl = (
                candle["high"] >= sl
            )

        # Conservative assumption:
        # SL first if both happen in same candle.

        if hit_sl:

            result = "SL"

            exit_price = sl
            exit_index = j

            break

        if hit_tp:

            result = "TP"

            exit_price = tp
            exit_index = j

            break

    # Return

    if direction == "BUY":

        gross_return = (
            exit_price -
            entry
        ) / entry

    else:

        gross_return = (
            entry -
            exit_price
        ) / entry

    net_return = (
        gross_return -
        ROUND_TRIP_COST
    )

    sl_distance = (
        SL_ATR * atr
    ) / entry

    if sl_distance > 0:

        r_multiple = (
            net_return /
            sl_distance
        )

    else:

        r_multiple = 0

    return {
        "direction": direction,
        "result": result,
        "entry_price": entry,
        "exit_price": exit_price,
        "entry_index": index,
        "exit_index": exit_index,
        "net_return": net_return,
        "r": r_multiple,
        "mfe": max_favorable,
        "mae": max_adverse
    }


# ============================================================
# RUN BACKTEST
# ============================================================

def run_backtest(
    df,
    start_index,
    end_index
):

    trades = []

    next_allowed_index = (
        start_index
    )

    for i in range(
        start_index,
        end_index
    ):

        if i < next_allowed_index:
            continue

        row = df.iloc[i]

        if pd.isna(row["atr"]):
            continue

        signal, regime = (
            get_signal(row)
        )

        if signal == "WAIT":
            continue

        trade = simulate_trade(
            df,
            i,
            signal
        )

        if trade is None:
            continue

        trade["time"] = (
            row["time"]
        )

        trade["regime"] = regime

        trades.append(trade)

        next_allowed_index = (
            trade["exit_index"] +
            COOLDOWN_CANDLES
        )

    return trades


# ============================================================
# PRINT STATS
# ============================================================

def print_stats(
    trades,
    title
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    if not trades:

        print("No trades.")
        return

    result_df = pd.DataFrame(
        trades
    )

    total = len(result_df)

    tp = (
        result_df["result"] ==
        "TP"
    ).sum()

    sl = (
        result_df["result"] ==
        "SL"
    ).sum()

    timeout = (
        result_df["result"] ==
        "TIMEOUT"
    ).sum()

    total_r = (
        result_df["r"].sum()
    )

    avg_r = (
        result_df["r"].mean()
    )

    total_return = (
        result_df["net_return"].sum()
    )

    avg_return = (
        result_df["net_return"].mean()
    )

    avg_mfe = (
        result_df["mfe"].mean()
    )

    avg_mae = (
        result_df["mae"].mean()
    )

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
        f"Total Return: "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Average Return: "
        f"{avg_return * 100:.3f}%"
    )

    print(
        f"Total R: {total_r:.2f}"
    )

    print(
        f"Average R: {avg_r:.3f}"
    )

    print(
        f"Average MFE: "
        f"{avg_mfe * 100:.2f}%"
    )

    print(
        f"Average MAE: "
        f"{avg_mae * 100:.2f}%"
    )


# ============================================================
# DIRECTION STATS
# ============================================================

def direction_stats(trades):

    if not trades:
        return

    df = pd.DataFrame(
        trades
    )

    print("\n📊 Direction breakdown")

    for direction in [
        "BUY",
        "SELL"
    ]:

        part = df[
            df["direction"] ==
            direction
        ]

        if part.empty:
            continue

        print(
            f"\n{direction}: "
            f"{len(part)} trades"
        )

        print(
            f"TP: "
            f"{(part['result'] == 'TP').mean() * 100:.1f}%"
        )

        print(
            f"SL: "
            f"{(part['result'] == 'SL').mean() * 100:.1f}%"
        )

        print(
            f"Total R: "
            f"{part['r'].sum():.2f}"
        )

        print(
            f"Average R: "
            f"{part['r'].mean():.3f}"
        )


# ============================================================
# REGIME STATS
# ============================================================

def regime_stats(trades):

    if not trades:
        return

    df = pd.DataFrame(
        trades
    )

    print("\n📈 Market regime breakdown")

    for regime, part in (
        df.groupby("regime")
    ):

        print(
            f"\n{regime}: "
            f"{len(part)} trades"
        )

        print(
            f"TP: "
            f"{(part['result'] == 'TP').mean() * 100:.1f}%"
        )

        print(
            f"SL: "
            f"{(part['result'] == 'SL').mean() * 100:.1f}%"
        )

        print(
            f"Total R: "
            f"{part['r'].sum():.2f}"
        )

        print(
            f"Average R: "
            f"{part['r'].mean():.3f}"
        )


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward(df):

    print("\n")
    print("#" * 70)
    print(
        "🚀 MULTI-TIMEFRAME WALK-FORWARD BACKTEST"
    )
    print("#" * 70)

    print(
        f"\nTP = {TP_ATR} x ATR"
    )

    print(
        f"SL = {SL_ATR} x ATR"
    )

    print(
        f"Maximum holding = "
        f"{MAX_HOLD_CANDLES * 15} minutes"
    )

    print(
        f"Estimated round-trip cost = "
        f"{ROUND_TRIP_COST * 100:.2f}%"
    )

    print(
        f"Training = {TRAIN_DAYS} days"
    )

    print(
        f"Validation = "
        f"{VALIDATION_DAYS} days"
    )

    all_validation = []

    window_start = 0
    window_number = 1

    while True:

        train_end_time = (
            df["time"].iloc[
                window_start
            ]
            +
            pd.Timedelta(
                days=TRAIN_DAYS
            )
        )

        validation_end_time = (
            train_end_time
            +
            pd.Timedelta(
                days=VALIDATION_DAYS
            )
        )

        train_candidates = df.index[
            df["time"] >=
            train_end_time
        ]

        validation_candidates = df.index[
            df["time"] >=
            validation_end_time
        ]

        if (
            len(train_candidates) == 0
            or
            len(validation_candidates) == 0
        ):
            break

        train_end = int(
            train_candidates[0]
        )

        validation_end = int(
            validation_candidates[0]
        )

        print("\n" + "-" * 70)

        print(
            f"WINDOW {window_number}"
        )

        print(
            "Training:",
            df["time"].iloc[
                window_start
            ],
            "→",
            df["time"].iloc[
                train_end - 1
            ]
        )

        print(
            "Validation:",
            df["time"].iloc[
                train_end
            ],
            "→",
            df["time"].iloc[
                validation_end - 1
            ]
        )

        # Training

        training_trades = run_backtest(
            df,
            window_start,
            train_end
        )

        print_stats(
            training_trades,
            f"WINDOW {window_number} TRAINING"
        )

        # Validation

        validation_trades = run_backtest(
            df,
            train_end,
            validation_end
        )

        print_stats(
            validation_trades,
            f"WINDOW {window_number} VALIDATION"
        )

        if validation_trades:

            all_validation.extend(
                validation_trades
            )

        # Move forward 30 days

        window_start += (
            30 * 96
        )

        window_number += 1

        if window_start >= train_end:

            break

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("#" * 70)
    print(
        "📊 OVERALL VALIDATION RESULT"
    )
    print("#" * 70)

    print_stats(
        all_validation,
        "ALL WALK-FORWARD VALIDATION"
    )

    direction_stats(
        all_validation
    )

    regime_stats(
        all_validation
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n🚀 BTC 6-Month Multi-Timeframe Backtest Started!"
    )

    # Download

    df = download_data()

    # 15M indicators

    print(
        "\n📊 Calculating 15M indicators..."
    )

    df = add_indicators(
        df
    )

    # 1H

    print(
        "📊 Building completed 1H candles..."
    )

    h1 = build_1h_data(
        df
    )

    # Merge

    print(
        "🔗 Combining 1H trend + 15M entry..."
    )

    df = merge_timeframes(
        df,
        h1
    )

    # Remove rows without enough history

    df = df.dropna(
        subset=[
            "ema50",
            "rsi",
            "macd_hist",
            "atr",
            "h1_ema20",
            "h1_ema50",
            "h1_ema100"
        ]
    ).reset_index(
        drop=True
    )

    print(
        "\nFinal usable candles:",
        len(df)
    )

    # Backtest

    walk_forward(
        df
    )

    print(
        "\n✅ Backtest finished."
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()

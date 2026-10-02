import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone


# =========================================================
# SETTINGS
# =========================================================

PRODUCT = "BTC-USD"

GRANULARITY = 900
DAYS = 183
CHUNK_CANDLES = 250

TRAIN_DAYS = 120
VALIDATION_DAYS = 63

MAX_HOLD_CANDLES = 12       # 3 hours
COOLDOWN_CANDLES = 12       # 3 hours

ROUND_TRIP_COST = 0.0014    # 0.14%

MIN_RR = 1.50

# Setup variations
PULLBACK_ATR_LEVELS = [
    0.50,
    0.75,
    1.00
]

BODY_LEVELS = [
    0.25,
    0.40,
    0.55
]

VOLUME_LEVELS = [
    0.80,
    1.00,
    1.20
]


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data():

    print("🚀 BTC High-Quality Setup Backtest Started!")
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

def calculate_15m(df):

    df = df.copy()

    # EMA
    df["ema9"] = (
        df["close"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["ema21"] = (
        df["close"]
        .ewm(span=21, adjust=False)
        .mean()
    )

    df["ema50"] = (
        df["close"]
        .ewm(span=50, adjust=False)
        .mean()
    )

    # RSI
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

    # MACD
    ema12 = (
        df["close"]
        .ewm(span=12, adjust=False)
        .mean()
    )

    ema26 = (
        df["close"]
        .ewm(span=26, adjust=False)
        .mean()
    )

    df["macd"] = (
        ema12 -
        ema26
    )

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    df["macd_hist"] = (
        df["macd"] -
        df["macd_signal"]
    )

    # ATR
    prev_close = (
        df["close"].shift(1)
    )

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
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(axis=1)

    df["atr"] = (
        true_range
        .rolling(14)
        .mean()
    )

    # Volume
    volume_ma = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        volume_ma.replace(
            0,
            np.nan
        )
    )

    # Candle
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

    # Candle range
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

    # Distance from EMA21
    df["ema21_distance"] = (
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

def build_1h(df):

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

    def regime(row):

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
        regime,
        axis=1
    )

    hourly["available_time"] = (
        pd.to_datetime(
            hourly["time"],
            utc=True
        )
        .astype(
            "datetime64[ns, UTC]"
        )
    )

    return hourly[
        [
            "available_time",
            "regime"
        ]
    ]


# =========================================================
# MERGE 1H REGIME
# =========================================================

def merge_regime(
    df,
    hourly
):

    df = df.copy()

    df["time"] = (
        pd.to_datetime(
            df["time"],
            utc=True
        )
        .astype(
            "datetime64[ns, UTC]"
        )
    )

    hourly = hourly.copy()

    hourly["available_time"] = (
        pd.to_datetime(
            hourly["available_time"],
            utc=True
        )
        .astype(
            "datetime64[ns, UTC]"
        )
    )

    df = df.sort_values(
        "time"
    )

    hourly = hourly.sort_values(
        "available_time"
    )

    df = pd.merge_asof(
        df,
        hourly,
        left_on="time",
        right_on="available_time",
        direction="backward"
    )

    df["regime"] = (
        df["regime"]
        .fillna("SIDEWAYS")
    )

    return df


# =========================================================
# HIGH QUALITY SETUP
# =========================================================

def find_setup(
    df,
    i,
    pullback_atr,
    body_min,
    volume_min
):

    row = df.iloc[i]

    # -----------------------------------------------------
    # 1H regime
    # -----------------------------------------------------

    regime = row["regime"]

    if regime == "SIDEWAYS":

        return None

    # -----------------------------------------------------
    # Basic values
    # -----------------------------------------------------

    close = row["close"]
    open_price = row["open"]
    high = row["high"]
    low = row["low"]

    ema9 = row["ema9"]
    ema21 = row["ema21"]
    ema50 = row["ema50"]

    atr = row["atr"]

    if (
        pd.isna(atr)
        or atr <= 0
    ):

        return None

    # -----------------------------------------------------
    # BUY TREND
    # -----------------------------------------------------

    bullish_trend = (
        regime in [
            "STRONG_UP",
            "NORMAL_UP"
        ]
        and
        ema9 > ema21 > ema50
        and
        close > ema21
        and
        row["macd_hist"] > 0
    )

    # -----------------------------------------------------
    # SELL TREND
    # -----------------------------------------------------

    bearish_trend = (
        regime in [
            "STRONG_DOWN",
            "NORMAL_DOWN"
        ]
        and
        ema9 < ema21 < ema50
        and
        close < ema21
        and
        row["macd_hist"] < 0
    )

    # -----------------------------------------------------
    # Volume
    # -----------------------------------------------------

    if (
        row["volume_ratio"] <
        volume_min
    ):

        return None

    # -----------------------------------------------------
    # Candle strength
    # -----------------------------------------------------

    if (
        row["body_atr"] <
        body_min
    ):

        return None

    if (
        row["body_ratio"] <
        0.35
    ):

        return None

    # =====================================================
    # BUY SETUP
    # =====================================================

    if bullish_trend:

        # Candle must close bullish
        if close <= open_price:

            return None

        # RSI healthy
        if not (
            48 <= row["rsi"] <= 68
        ):

            return None

        # Pullback to EMA21
        distance = (
            abs(close - ema21)
            / atr
        )

        if distance > pullback_atr:

            return None

        # Not overextended
        if (
            distance >
            1.5
        ):

            return None

        # Recent pullback check:
        # previous 3 candles must have
        # touched / approached EMA21

        previous = df.iloc[
            i - 3:i
        ]

        touched_ema = (
            (
                previous["low"] <=
                previous["ema21"] +
                previous["atr"] * 0.35
            )
            .any()
        )

        if not touched_ema:

            return None

        return "BUY"

    # =====================================================
    # SELL SETUP
    # =====================================================

    if bearish_trend:

        if close >= open_price:

            return None

        if not (
            32 <= row["rsi"] <= 52
        ):

            return None

        distance = (
            abs(close - ema21)
            / atr
        )

        if distance > pullback_atr:

            return None

        if (
            distance >
            1.5
        ):

            return None

        previous = df.iloc[
            i - 3:i
        ]

        touched_ema = (
            (
                previous["high"] >=
                previous["ema21"] -
                previous["atr"] * 0.35
            )
            .any()
        )

        if not touched_ema:

            return None

        return "SELL"

    return None


# =========================================================
# CONFIRMATION
# =========================================================

def confirm_entry(
    df,
    signal_index,
    direction
):

    signal = df.iloc[
        signal_index
    ]

    signal_close = signal["close"]

    signal_high = signal["high"]

    signal_low = signal["low"]

    # Wait maximum 2 candles
    for j in range(
        signal_index + 1,
        signal_index + 3
    ):

        if j >= len(df):

            break

        candle = df.iloc[j]

        # -------------------------------------------------
        # BUY
        # -------------------------------------------------

        if direction == "BUY":

            # Confirmation:
            # close above signal high
            if (
                candle["close"] >
                signal_high
            ):

                return j

            # Or strong continuation
            if (
                candle["close"] >
                signal_close
                and
                candle["close"] >
                candle["open"]
                and
                candle["body_atr"] >= 0.40
            ):

                return j

        # -------------------------------------------------
        # SELL
        # -------------------------------------------------

        if direction == "SELL":

            if (
                candle["close"] <
                signal_low
            ):

                return j

            if (
                candle["close"] <
                signal_close
                and
                candle["close"] <
                candle["open"]
                and
                candle["body_atr"] >= 0.40
            ):

                return j

    return None


# =========================================================
# TRADE SIMULATION
# =========================================================

def simulate_trade(
    df,
    entry_index,
    direction
):

    entry = df.iloc[
        entry_index
    ]

    entry_price = (
        entry["close"]
    )

    atr = entry["atr"]

    if (
        pd.isna(atr)
        or atr <= 0
    ):

        return None

    # -----------------------------------------------------
    # Dynamic risk
    # -----------------------------------------------------

    risk = atr

    # TP at 1.5R
    reward = (
        risk *
        MIN_RR
    )

    if direction == "BUY":

        stop_price = (
            entry_price -
            risk
        )

        target_price = (
            entry_price +
            reward
        )

    else:

        stop_price = (
            entry_price +
            risk
        )

        target_price = (
            entry_price -
            reward
        )

    end_index = min(
        entry_index +
        MAX_HOLD_CANDLES,
        len(df) - 1
    )

    future = df.iloc[
        entry_index + 1:
        end_index + 1
    ]

    if future.empty:

        return None

    mfe_values = []

    mae_values = []

    outcome = "TIMEOUT"

    exit_price = (
        future.iloc[-1]["close"]
    )

    exit_index = (
        future.index[-1]
    )

    for idx, candle in future.iterrows():

        if direction == "BUY":

            favorable = (
                candle["high"] -
                entry_price
            ) / entry_price

            adverse = (
                candle["low"] -
                entry_price
            ) / entry_price

            mfe_values.append(
                favorable
            )

            mae_values.append(
                adverse
            )

            hit_stop = (
                candle["low"] <=
                stop_price
            )

            hit_target = (
                candle["high"] >=
                target_price
            )

            # Conservative:
            # if both happen same candle,
            # assume stop first.

            if hit_stop:

                outcome = "SL"

                exit_price = (
                    stop_price
                )

                exit_index = idx

                break

            if hit_target:

                outcome = "TP"

                exit_price = (
                    target_price
                )

                exit_index = idx

                break

        else:

            favorable = (
                entry_price -
                candle["low"]
            ) / entry_price

            adverse = (
                entry_price -
                candle["high"]
            ) / entry_price

            mfe_values.append(
                favorable
            )

            mae_values.append(
                adverse
            )

            hit_stop = (
                candle["high"] >=
                stop_price
            )

            hit_target = (
                candle["low"] <=
                target_price
            )

            if hit_stop:

                outcome = "SL"

                exit_price = (
                    stop_price
                )

                exit_index = idx

                break

            if hit_target:

                outcome = "TP"

                exit_price = (
                    target_price
                )

                exit_index = idx

                break

    # -----------------------------------------------------
    # Return
    # -----------------------------------------------------

    if direction == "BUY":

        gross_return = (
            exit_price -
            entry_price
        ) / entry_price

    else:

        gross_return = (
            entry_price -
            exit_price
        ) / entry_price

    net_return = (
        gross_return -
        ROUND_TRIP_COST
    )

    # -----------------------------------------------------
    # R
    # -----------------------------------------------------

    risk_pct = (
        risk /
        entry_price
    )

    if risk_pct > 0:

        r_value = (
            net_return /
            risk_pct
        )

    else:

        r_value = 0

    return {

        "direction":
            direction,

        "entry_time":
            entry["time"],

        "entry_price":
            entry_price,

        "exit_time":
            df.loc[
                exit_index,
                "time"
            ],

        "outcome":
            outcome,

        "gross_return":
            gross_return,

        "net_return":
            net_return,

        "r":
            r_value,

        "mfe":
            max(
                mfe_values
            ) if mfe_values else 0,

        "mae":
            min(
                mae_values
            ) if mae_values else 0,

        "regime":
            entry["regime"]
    }


# =========================================================
# RUN ONE CONFIG
# =========================================================

def run_config(
    df,
    start_time,
    end_time,
    pullback_atr,
    body_min,
    volume_min
):

    trades = []

    last_trade_index = -999

    for i in range(
        120,
        len(df) -
        MAX_HOLD_CANDLES -
        3
    ):

        current_time = (
            df.iloc[i]["time"]
        )

        if (
            current_time <
            start_time
        ):

            continue

        if (
            current_time >
            end_time
        ):

            break

        if (
            i -
            last_trade_index
        ) < COOLDOWN_CANDLES:

            continue

        direction = find_setup(
            df,
            i,
            pullback_atr,
            body_min,
            volume_min
        )

        if direction is None:

            continue

        entry_index = confirm_entry(
            df,
            i,
            direction
        )

        if entry_index is None:

            continue

        entry_time = (
            df.iloc[
                entry_index
            ]["time"]
        )

        if (
            entry_time <
            start_time
        ):

            continue

        if (
            entry_time >
            end_time
        ):

            continue

        trade = simulate_trade(
            df,
            entry_index,
            direction
        )

        if trade is not None:

            trades.append(
                trade
            )

            last_trade_index = (
                entry_index
            )

    return trades


# =========================================================
# STATISTICS
# =========================================================

def stats(
    trades
):

    if not trades:

        return None

    data = pd.DataFrame(
        trades
    )

    total = len(data)

    tp = (
        data["outcome"] ==
        "TP"
    ).sum()

    sl = (
        data["outcome"] ==
        "SL"
    ).sum()

    timeout = (
        data["outcome"] ==
        "TIMEOUT"
    ).sum()

    total_return = (
        data["net_return"]
        .sum()
    )

    avg_return = (
        data["net_return"]
        .mean()
    )

    total_r = (
        data["r"].sum()
    )

    avg_r = (
        data["r"].mean()
    )

    avg_mfe = (
        data["mfe"].mean()
    )

    avg_mae = (
        data["mae"].mean()
    )

    return {

        "trades":
            total,

        "tp":
            tp,

        "sl":
            sl,

        "timeout":
            timeout,

        "tp_pct":
            tp / total,

        "sl_pct":
            sl / total,

        "timeout_pct":
            timeout / total,

        "total_return":
            total_return,

        "avg_return":
            avg_return,

        "total_r":
            total_r,

        "avg_r":
            avg_r,

        "avg_mfe":
            avg_mfe,

        "avg_mae":
            avg_mae
    }


# =========================================================
# PRINT STATS
# =========================================================

def print_stats(
    name,
    result
):

    if result is None:

        print(
            f"{name}: NO TRADES"
        )

        return

    print(
        f"{name} | "
        f"Trades={result['trades']} | "
        f"TP={result['tp_pct'] * 100:.1f}% | "
        f"SL={result['sl_pct'] * 100:.1f}% | "
        f"Timeout={result['timeout_pct'] * 100:.1f}% | "
        f"Return={result['total_return'] * 100:.2f}% | "
        f"Avg={result['avg_return'] * 100:.3f}% | "
        f"TotalR={result['total_r']:.2f} | "
        f"AvgR={result['avg_r']:.3f} | "
        f"MFE={result['avg_mfe'] * 100:.3f}% | "
        f"MAE={result['avg_mae'] * 100:.3f}%"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    df = download_data()

    print()
    print("📊 Calculating 15M indicators...")

    df = calculate_15m(
        df
    )

    print(
        "📊 Building completed 1H regime..."
    )

    hourly = build_1h(
        df
    )

    df = merge_regime(
        df,
        hourly
    )

    required = [
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd_hist",
        "atr",
        "volume_ratio",
        "body_atr",
        "body_ratio",
        "ema21_distance"
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
    # Time periods
    # -----------------------------------------------------

    data_start = (
        df["time"].iloc[0]
    )

    data_end = (
        df["time"].iloc[-1]
    )

    validation_start = (
        data_end -
        timedelta(
            days=VALIDATION_DAYS
        )
    )

    training_start = (
        validation_start -
        timedelta(
            days=TRAIN_DAYS
        )
    )

    print()
    print(
        f"Training: "
        f"{training_start} "
        f"→ "
        f"{validation_start}"
    )

    print(
        f"Validation: "
        f"{validation_start} "
        f"→ "
        f"{data_end}"
    )

    # -----------------------------------------------------
    # TRAINING CONFIGS
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "TRAINING HIGH-QUALITY SETUPS"
    )
    print("=" * 80)

    configs = []

    for pullback in PULLBACK_ATR_LEVELS:

        for body in BODY_LEVELS:

            for volume in VOLUME_LEVELS:

                trades = run_config(
                    df,
                    training_start,
                    validation_start,
                    pullback,
                    body,
                    volume
                )

                result = stats(
                    trades
                )

                name = (
                    f"PB{pullback:.2f}"
                    f"_BODY{body:.2f}"
                    f"_VOL{volume:.2f}"
                )

                print_stats(
                    name,
                    result
                )

                if (
                    result is not None
                    and
                    result["trades"] >= 20
                    and
                    result["avg_return"] > 0
                    and
                    result["avg_r"] > 0
                ):

                    configs.append({
                        "name": name,
                        "pullback": pullback,
                        "body": body,
                        "volume": volume,
                        "result": result
                    })

    # -----------------------------------------------------
    # Positive configs
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "POSITIVE TRAINING CONFIGURATIONS"
    )
    print("=" * 80)

    if not configs:

        print(
            "❌ No positive training configuration."
        )

        print()
        print(
            "The setup did not demonstrate "
            "a positive edge in training."
        )

        return

    configs = sorted(
        configs,
        key=lambda x:
        x["result"]["avg_r"],
        reverse=True
    )

    for config in configs:

        result = (
            config["result"]
        )

        print(
            f"{config['name']} | "
            f"Trades={result['trades']} | "
            f"AvgR={result['avg_r']:.3f} | "
            f"Return={result['avg_return'] * 100:.3f}%"
        )

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    print()
    print("=" * 80)
    print(
        "UNSEEN VALIDATION"
    )
    print("=" * 80)

    validation_rows = []

    for config in configs:

        trades = run_config(
            df,
            validation_start,
            data_end,
            config["pullback"],
            config["body"],
            config["volume"]
        )

        result = stats(
            trades
        )

        print_stats(
            config["name"],
            result
        )

        if result is not None:

            row = {
                "config":
                    config["name"],

                **result
            }

            validation_rows.append(
                row
            )

    # -----------------------------------------------------
    # SAVE
    # -----------------------------------------------------

    if validation_rows:

        pd.DataFrame(
            validation_rows
        ).to_csv(
            "btc_high_quality_validation.csv",
            index=False
        )

        print()
        print(
            "💾 Saved:"
        )

        print(
            "btc_high_quality_validation.csv"
        )

    print()
    print("=" * 80)
    print(
        "✅ HIGH-QUALITY SETUP BACKTEST COMPLETE"
    )
    print("=" * 80)


# =========================================================
# START
# =========================================================

if __name__ == "__main__":

    main()

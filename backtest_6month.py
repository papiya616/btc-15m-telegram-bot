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

TRAIN_DAYS = 120           # ~4 months
VALIDATION_DAYS = 63        # ~2 months

COOLDOWN_CANDLES = 12      # 3 hours
EVAL_CANDLES = 12          # 3 hours

ROUND_TRIP_COST = 0.0014   # 0.14%

MIN_TRAIN_TRADES = 30

# Confirmation tests
CONFIRM_CONFIGS = [
    ("CLOSE_1", 1, "close"),
    ("CLOSE_2", 2, "close"),
    ("CLOSE_3", 3, "close"),

    ("BREAK_005_1", 1, "break_005"),
    ("BREAK_005_2", 2, "break_005"),
    ("BREAK_005_3", 3, "break_005"),

    ("BREAK_010_1", 1, "break_010"),
    ("BREAK_010_2", 2, "break_010"),
    ("BREAK_010_3", 3, "break_010"),

    ("BREAK_015_1", 1, "break_015"),
    ("BREAK_015_2", 2, "break_015"),
    ("BREAK_015_3", 3, "break_015"),
]

# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data():
    print("🚀 BTC 6-Month Confirmation Backtest Started!")
    print()
    print("📥 Downloading 6 months of BTC 15M data...")
    print()

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_candles = []

    current_start = start_time

    while current_start < end_time:

        current_end = min(
            current_start +
            timedelta(seconds=GRANULARITY * (CHUNK_CANDLES - 1)),
            end_time
        )

        start_epoch = int(current_start.timestamp())
        end_epoch = int(current_end.timestamp())

        print(
            f"Downloading: "
            f"{current_start.strftime('%Y-%m-%d %H:%M')} "
            f"to "
            f"{current_end.strftime('%Y-%m-%d %H:%M')}"
        )

        url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

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
            print("Download error:", e)

        current_start = current_end + timedelta(seconds=GRANULARITY)

        time.sleep(0.25)

    if not all_candles:
        raise RuntimeError("No BTC data downloaded.")

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

    df = df.drop_duplicates(subset=["timestamp"])

    df["time"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.sort_values("time").reset_index(drop=True)

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

    df = df.dropna().reset_index(drop=True)

    print()
    print(f"✅ Total candles downloaded: {len(df):,}")

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
# INDICATORS
# =========================================================

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

    rs = avg_gain / avg_loss.replace(0, np.nan)

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

    df["atr"] = true_range.rolling(14).mean()

    # Volume ratio
    volume_ma = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        volume_ma.replace(0, np.nan)
    )

    # Candle body
    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["body_atr"] = (
        df["body"] /
        df["atr"].replace(0, np.nan)
    )

    # Distance from EMA21
    df["ema21_distance_atr"] = (
        (df["close"] - df["ema21"]).abs() /
        df["atr"].replace(0, np.nan)
    )

    return df


# =========================================================
# 1H MARKET REGIME
# =========================================================

def build_1h_regime(df):

    temp = df.set_index("time").copy()

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

    hourly = hourly.dropna().reset_index()

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

    def regime(row):

        if (
            row["ema20"] >
            row["ema50"] >
            row["ema100"] and
            row["slope20"] > 0 and
            row["slope50"] > 0
        ):
            return "STRONG_UP"

        if (
            row["ema20"] <
            row["ema50"] <
            row["ema100"] and
            row["slope20"] < 0 and
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

    hourly["available_time"] = hourly["time"]

    hourly["time"] = pd.to_datetime(
        hourly["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    return hourly[
        [
            "time",
            "available_time",
            "regime"
        ]
    ]


def merge_regime(df, hourly):

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    hourly = hourly.copy()

    hourly["time"] = pd.to_datetime(
        hourly["time"],
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

    df["regime"] = df["regime"].fillna(
        "SIDEWAYS"
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
    # MARKET REGIME
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
    # EMA ALIGNMENT
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

    if row["close"] > row["ema21"]:
        buy_score += 1

    if row["close"] < row["ema21"]:
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

    if 50 <= row["rsi"] <= 68:
        buy_score += 1

    if 32 <= row["rsi"] <= 50:
        sell_score += 1

    # -----------------------------------------------------
    # VOLUME
    # -----------------------------------------------------

    if row["volume_ratio"] >= 1.2:

        if buy_score > sell_score:
            buy_score += 1

        elif sell_score > buy_score:
            sell_score += 1

    # -----------------------------------------------------
    # CANDLE
    # -----------------------------------------------------

    if row["body_atr"] >= 0.25:

        if row["close"] > row["open"]:
            buy_score += 1

        elif row["close"] < row["open"]:
            sell_score += 1

    # -----------------------------------------------------
    # OVEREXTENSION
    # -----------------------------------------------------

    if row["ema21_distance_atr"] > 1.5:

        buy_score -= 3
        sell_score -= 3

    # -----------------------------------------------------
    # FINAL SIGNAL
    # -----------------------------------------------------

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

    return "WAIT"


# =========================================================
# CONFIRMATION
# =========================================================

def check_confirmation(
    df,
    signal_index,
    direction,
    max_wait,
    confirmation_type
):

    signal_row = df.iloc[signal_index]

    signal_close = signal_row["close"]
    signal_high = signal_row["high"]
    signal_low = signal_row["low"]

    last_index = min(
        signal_index + max_wait,
        len(df) - 1
    )

    for j in range(
        signal_index + 1,
        last_index + 1
    ):

        row = df.iloc[j]

        # -------------------------------------------------
        # BUY
        # -------------------------------------------------

        if direction == "BUY":

            if confirmation_type == "close":

                if row["close"] > signal_close:

                    return j

            elif confirmation_type == "break_005":

                level = signal_high * 1.0005

                if row["high"] >= level:

                    return j

            elif confirmation_type == "break_010":

                level = signal_high * 1.0010

                if row["high"] >= level:

                    return j

            elif confirmation_type == "break_015":

                level = signal_high * 1.0015

                if row["high"] >= level:

                    return j

        # -------------------------------------------------
        # SELL
        # -------------------------------------------------

        if direction == "SELL":

            if confirmation_type == "close":

                if row["close"] < signal_close:

                    return j

            elif confirmation_type == "break_005":

                level = signal_low * 0.9995

                if row["low"] <= level:

                    return j

            elif confirmation_type == "break_010":

                level = signal_low * 0.9990

                if row["low"] <= level:

                    return j

            elif confirmation_type == "break_015":

                level = signal_low * 0.9985

                if row["low"] <= level:

                    return j

    return None


# =========================================================
# BASE SIGNAL COLLECTION
# =========================================================

def collect_base_signals(df):

    signals = []

    last_signal_index = -999

    for i in range(120, len(df) - EVAL_CANDLES - 5):

        if (
            i -
            last_signal_index
        ) < COOLDOWN_CANDLES:

            continue

        signal = get_base_signal(
            df.iloc[i]
        )

        if signal in [
            "BUY",
            "SELL"
        ]:

            signals.append({
                "signal_index": i,
                "signal": signal
            })

            last_signal_index = i

    return signals


# =========================================================
# CONFIRMED TRADE
# =========================================================

def simulate_confirmed_trade(
    df,
    signal_index,
    entry_index,
    direction
):

    if entry_index >= len(df) - 1:
        return None

    entry_row = df.iloc[entry_index]

    entry_price = entry_row["close"]

    # ATR at actual entry
    atr = entry_row["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp_price = (
            entry_price +
            1.5 * atr
        )

        sl_price = (
            entry_price -
            1.0 * atr
        )

    else:

        tp_price = (
            entry_price -
            1.5 * atr
        )

        sl_price = (
            entry_price +
            1.0 * atr
        )

    end_index = min(
        entry_index +
        EVAL_CANDLES,
        len(df) - 1
    )

    result = "TIMEOUT"

    exit_price = df.iloc[end_index]["close"]

    mfe = 0.0
    mae = 0.0

    for j in range(
        entry_index + 1,
        end_index + 1
    ):

        row = df.iloc[j]

        if direction == "BUY":

            favorable = (
                row["high"] -
                entry_price
            ) / entry_price

            adverse = (
                row["low"] -
                entry_price
            ) / entry_price

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            hit_tp = (
                row["high"] >= tp_price
            )

            hit_sl = (
                row["low"] <= sl_price
            )

        else:

            favorable = (
                entry_price -
                row["low"]
            ) / entry_price

            adverse = (
                entry_price -
                row["high"]
            ) / entry_price

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            hit_tp = (
                row["low"] <= tp_price
            )

            hit_sl = (
                row["high"] >= sl_price
            )

        # Conservative:
        # if both happen in same candle,
        # assume SL first.

        if hit_sl:

            result = "SL"

            exit_price = sl_price

            break

        if hit_tp:

            result = "TP"

            exit_price = tp_price

            break

    # -----------------------------------------------------
    # GROSS RETURN
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
    # R MULTIPLE
    # -----------------------------------------------------

    risk_price = atr

    if direction == "BUY":

        r_multiple = (
            exit_price -
            entry_price
        ) / risk_price

    else:

        r_multiple = (
            entry_price -
            exit_price
        ) / risk_price

    return {
        "signal_index": signal_index,
        "entry_index": entry_index,
        "signal_time": df.iloc[
            signal_index
        ]["time"],

        "entry_time": df.iloc[
            entry_index
        ]["time"],

        "direction": direction,

        "signal_close": df.iloc[
            signal_index
        ]["close"],

        "entry_price": entry_price,

        "exit_price": exit_price,

        "result": result,

        "gross_return": gross_return,

        "net_return": net_return,

        "r_multiple": r_multiple,

        "mfe": mfe,

        "mae": mae,

        "regime": df.iloc[
            signal_index
        ]["regime"],

        "rsi": df.iloc[
            signal_index
        ]["rsi"],

        "volume_ratio": df.iloc[
            signal_index
        ]["volume_ratio"]
    }


# =========================================================
# RUN CONFIGURATION
# =========================================================

def run_configuration(
    df,
    signals,
    config
):

    name, max_wait, confirmation_type = config

    trades = []

    last_entry_index = -999

    for s in signals:

        signal_index = s["signal_index"]
        direction = s["signal"]

        if (
            signal_index -
            last_entry_index
        ) < COOLDOWN_CANDLES:

            continue

        entry_index = check_confirmation(
            df,
            signal_index,
            direction,
            max_wait,
            confirmation_type
        )

        if entry_index is None:
            continue

        # Avoid overlapping trades
        if (
            entry_index -
            last_entry_index
        ) < COOLDOWN_CANDLES:

            continue

        trade = simulate_confirmed_trade(
            df,
            signal_index,
            entry_index,
            direction
        )

        if trade is not None:

            trade["configuration"] = name

            trades.append(trade)

            last_entry_index = entry_index

    return pd.DataFrame(trades)


# =========================================================
# SUMMARY
# =========================================================

def summarize(
    trades,
    title
):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if trades.empty:

        print("No trades.")
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

    avg_return = (
        trades["net_return"].mean()
    )

    total_return = (
        trades["net_return"].sum()
    )

    avg_r = (
        trades["r_multiple"].mean()
    )

    total_r = (
        trades["r_multiple"].sum()
    )

    avg_mfe = (
        trades["mfe"].mean()
    )

    avg_mae = (
        trades["mae"].mean()
    )

    print(f"Trades: {total}")

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
        f"Total return: "
        f"{total_return * 100:.2f}%"
    )

    print(
        f"Average return: "
        f"{avg_return * 100:.3f}%"
    )

    print(
        f"Total R: "
        f"{total_r:.2f}"
    )

    print(
        f"Average R: "
        f"{avg_r:.3f}"
    )

    print(
        f"Average MFE: "
        f"{avg_mfe * 100:.3f}%"
    )

    print(
        f"Average MAE: "
        f"{avg_mae * 100:.3f}%"
    )

    return {
        "trades": total,
        "tp": tp,
        "sl": sl,
        "timeout": timeout,
        "avg_return": avg_return,
        "total_return": total_return,
        "avg_r": avg_r,
        "total_r": total_r,
        "avg_mfe": avg_mfe,
        "avg_mae": avg_mae
    }


# =========================================================
# CONDITION DETAILS
# =========================================================

def print_direction_stats(trades):

    print()
    print("DIRECTION RESULTS")
    print("-" * 70)

    for direction in [
        "BUY",
        "SELL"
    ]:

        subset = trades[
            trades["direction"] ==
            direction
        ]

        if subset.empty:
            continue

        print(
            f"{direction}: "
            f"{len(subset)} trades | "
            f"Avg Return "
            f"{subset['net_return'].mean() * 100:.3f}% | "
            f"Avg R "
            f"{subset['r_multiple'].mean():.3f} | "
            f"MFE "
            f"{subset['mfe'].mean() * 100:.3f}% | "
            f"MAE "
            f"{subset['mae'].mean() * 100:.3f}%"
        )


def print_regime_stats(trades):

    print()
    print("REGIME RESULTS")
    print("-" * 70)

    for regime, group in trades.groupby(
        "regime"
    ):

        print(
            f"{regime}: "
            f"{len(group)} trades | "
            f"Avg Return "
            f"{group['net_return'].mean() * 100:.3f}% | "
            f"Avg R "
            f"{group['r_multiple'].mean():.3f}"
        )


# =========================================================
# MAIN
# =========================================================

def main():

    df = download_data()

    df = calculate_indicators(
        df
    )

    hourly = build_1h_regime(
        df
    )

    df = merge_regime(
        df,
        hourly
    )

    df = df.dropna().reset_index(
        drop=True
    )

    print()
    print(
        f"Final usable candles: "
        f"{len(df):,}"
    )

    print()
    print("📊 Creating base signals...")

    signals = collect_base_signals(
        df
    )

    print(
        f"Base signals: "
        f"{len(signals):,}"
    )

    if not signals:
        print("No base signals found.")
        return

    # =====================================================
    # DATE SPLIT
    # =====================================================

    last_time = df["time"].iloc[-1]

    validation_start = (
        last_time -
        timedelta(days=VALIDATION_DAYS)
    )

    training_start = (
        validation_start -
        timedelta(days=TRAIN_DAYS)
    )

    print()
    print(
        f"Training period: "
        f"{training_start} "
        f"→ "
        f"{validation_start}"
    )

    print(
        f"Validation period: "
        f"{validation_start} "
        f"→ "
        f"{last_time}"
    )

    # =====================================================
    # SPLIT BASE SIGNALS
    # =====================================================

    train_signals = []

    validation_signals = []

    for s in signals:

        t = df.iloc[
            s["signal_index"]
        ]["time"]

        if (
            training_start <= t <
            validation_start
        ):

            train_signals.append(s)

        elif t >= validation_start:

            validation_signals.append(s)

    print()
    print(
        f"Training base signals: "
        f"{len(train_signals)}"
    )

    print(
        f"Validation base signals: "
        f"{len(validation_signals)}"
    )

    # =====================================================
    # TEST ALL CONFIRMATION CONFIGS
    # =====================================================

    training_results = []

    print()
    print("=" * 70)
    print("TRAINING CONFIRMATION CONFIGURATIONS")
    print("=" * 70)

    for config in CONFIRM_CONFIGS:

        name = config[0]

        trades = run_configuration(
            df,
            train_signals,
            config
        )

        if trades.empty:

            print(
                f"{name}: "
                f"NO TRADES"
            )

            continue

        stats = summarize(
            trades,
            name
        )

        if stats is not None:

            training_results.append({
                "config": config,
                "stats": stats,
                "trades": trades
            })

    # =====================================================
    # SELECT POSITIVE TRAINING CONFIGS
    # =====================================================

    positive = [
        x for x in training_results
        if (
            x["stats"]["trades"] >=
            MIN_TRAIN_TRADES
            and
            x["stats"]["avg_return"] > 0
        )
    ]

    print()
    print("=" * 70)
    print("POSITIVE TRAINING CONFIGURATIONS")
    print("=" * 70)

    if not positive:

        print(
            "❌ No confirmation configuration "
            "had positive training return."
        )

        print()
        print(
            "Therefore validation will NOT "
            "force a configuration."
        )

        # Save all training results
        all_training = []

        for x in training_results:

            all_training.append({
                "config": x["config"][0],
                "trades": x["stats"]["trades"],
                "avg_return": x["stats"]["avg_return"],
                "total_return": x["stats"]["total_return"],
                "avg_r": x["stats"]["avg_r"],
                "total_r": x["stats"]["total_r"],
                "avg_mfe": x["stats"]["avg_mfe"],
                "avg_mae": x["stats"]["avg_mae"]
            })

        pd.DataFrame(
            all_training
        ).to_csv(
            "btc_confirmation_training.csv",
            index=False
        )

        print()
        print(
            "Saved: "
            "btc_confirmation_training.csv"
        )

        return

    positive.sort(
        key=lambda x:
        x["stats"]["avg_return"],
        reverse=True
    )

    print()

    for x in positive:

        print(
            f"{x['config'][0]} | "
            f"{x['stats']['trades']} trades | "
            f"Avg Return "
            f"{x['stats']['avg_return'] * 100:.3f}% | "
            f"Avg R "
            f"{x['stats']['avg_r']:.3f}"
        )

    # =====================================================
    # VALIDATION
    # =====================================================

    print()
    print("=" * 70)
    print("UNSEEN VALIDATION")
    print("=" * 70)

    validation_all = []

    # Use positive training configurations only
    for x in positive:

        config = x["config"]

        trades = run_configuration(
            df,
            validation_signals,
            config
        )

        if trades.empty:

            print(
                f"{config[0]}: "
                f"No validation trades."
            )

            continue

        summarize(
            trades,
            f"VALIDATION - {config[0]}"
        )

        print_direction_stats(
            trades
        )

        print_regime_stats(
            trades
        )

        validation_all.append(
            trades
        )

    # =====================================================
    # SAVE VALIDATION
    # =====================================================

    if validation_all:

        validation_df = pd.concat(
            validation_all,
            ignore_index=True
        )

        validation_df.to_csv(
            "btc_confirmation_validation.csv",
            index=False
        )

        print()
        print(
            "Saved: "
            "btc_confirmation_validation.csv"
        )

    print()
    print("=" * 70)
    print("✅ CONFIRMATION BACKTEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

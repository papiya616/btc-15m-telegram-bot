import requests
import pandas as pd
import numpy as np
import time as time_module
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 6-MONTH ENTRY TIMING BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
DAYS = 183
CHUNK_CANDLES = 250

TP_ATR = 1.5
SL_ATR = 1.0
MAX_HOLD_CANDLES = 12      # 3 hours
COOLDOWN_CANDLES = 12      # 3 hours
ROUND_TRIP_COST = 0.0014   # 0.14%

TRAIN_DAYS = 60
VALID_DAYS = 30

# Small pre-defined grid.
# These are selected BEFORE validation.
PULLBACK_ATR_OPTIONS = [0.30, 0.50, 0.75, 1.00]
OVEREXTENSION_ATR_OPTIONS = [0.75, 1.00, 1.25, 1.50]
CONFIRM_BODY_OPTIONS = [0.25, 0.40, 0.55]

MIN_TRAIN_TRADES = 25


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_candles():
    print("🚀 BTC 6-Month Entry Timing Backtest Started!")
    print()
    print("📥 Downloading 6 months of BTC 15M data...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []
    current = start_time

    while current < end_time:
        chunk_end = min(
            current + timedelta(seconds=GRANULARITY * CHUNK_CANDLES),
            end_time
        )

        params = {
            "granularity": GRANULARITY,
            "start": current.isoformat(),
            "end": chunk_end.isoformat()
        }

        print(
            f"Downloading: "
            f"{current.strftime('%Y-%m-%d %H:%M')} "
            f"to {chunk_end.strftime('%Y-%m-%d %H:%M')}"
        )

        try:
            response = requests.get(
                f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles",
                params=params,
                timeout=30
            )

            response.raise_for_status()

            rows = response.json()

            if rows:
                all_rows.extend(rows)

        except Exception as e:
            print("Download error:", e)
            time_module.sleep(2)

        current = chunk_end
        time_module.sleep(0.15)

    if not all_rows:
        raise RuntimeError("No candle data downloaded.")

    df = pd.DataFrame(
        all_rows,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume"
        ]
    )

    df["time"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    ).dt.tz_convert("UTC")

    df = df.drop_duplicates("time")
    df = df.sort_values("time").reset_index(drop=True)

    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print()
    print(f"✅ Total candles downloaded: {len(df):,}")
    print(f"First candle: {df['time'].iloc[0]}")
    print(f"Last candle : {df['time'].iloc[-1]}")

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    df = df.copy()

    # EMA
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema21"] = df["close"].ewm(span=21, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

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

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi"] = 100 - (100 / (1 + rs))

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

    # ATR
    prev_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.rolling(14).mean()

    # Volume ratio
    df["volume_avg"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"].replace(0, np.nan)
    )

    # Candle body
    candle_range = (
        df["high"] - df["low"]
    ).replace(0, np.nan)

    df["body_strength"] = (
        (df["close"] - df["open"]).abs() /
        candle_range
    )

    # Candle direction
    df["bull_candle"] = df["close"] > df["open"]
    df["bear_candle"] = df["close"] < df["open"]

    # Distance from EMA21
    df["ema21_distance_atr"] = (
        (df["close"] - df["ema21"]).abs() /
        df["atr"].replace(0, np.nan)
    )

    # Recent high / low
    df["recent_high"] = df["high"].rolling(20).max().shift(1)
    df["recent_low"] = df["low"].rolling(20).min().shift(1)

    return df


# ============================================================
# ENTRY SETUP
# ============================================================

def get_setup(df, i):

    row = df.iloc[i]

    required = [
        row["ema9"],
        row["ema21"],
        row["ema50"],
        row["rsi"],
        row["macd"],
        row["macd_signal"],
        row["atr"],
        row["volume_ratio"],
        row["body_strength"]
    ]

    if any(pd.isna(x) for x in required):
        return None

    # --------------------------------------------------------
    # BUY setup
    # --------------------------------------------------------

    buy = True

    if not (
        row["ema9"] > row["ema21"] > row["ema50"]
    ):
        buy = False

    if row["close"] <= row["ema21"]:
        buy = False

    if row["macd"] <= row["macd_signal"]:
        buy = False

    if not (48 <= row["rsi"] <= 68):
        buy = False

    if row["volume_ratio"] < 0.85:
        buy = False

    # --------------------------------------------------------
    # SELL setup
    # --------------------------------------------------------

    sell = True

    if not (
        row["ema9"] < row["ema21"] < row["ema50"]
    ):
        sell = False

    if row["close"] >= row["ema21"]:
        sell = False

    if row["macd"] >= row["macd_signal"]:
        sell = False

    if not (32 <= row["rsi"] <= 52):
        sell = False

    if row["volume_ratio"] < 0.85:
        sell = False

    if buy and not sell:
        return "BUY"

    if sell and not buy:
        return "SELL"

    return None


# ============================================================
# ENTRY TIMING
# ============================================================

def timing_entry(
    df,
    setup_index,
    direction,
    pullback_atr,
    overextension_atr,
    confirm_body
):
    """
    IMPORTANT:
    Uses only information available BEFORE entry.

    No future MFE/MAE is used here.
    """

    row = df.iloc[setup_index]

    atr = row["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    distance = abs(
        row["close"] - row["ema21"]
    ) / atr

    # --------------------------------------------------------
    # Reject if already too far from EMA21.
    # --------------------------------------------------------

    if distance > overextension_atr:
        return None

    # --------------------------------------------------------
    # Entry should be reasonably close to EMA21.
    # --------------------------------------------------------

    if distance > pullback_atr:
        return None

    # --------------------------------------------------------
    # Confirmation candle strength
    # --------------------------------------------------------

    if row["body_strength"] < confirm_body:
        return None

    if direction == "BUY":

        if not row["bull_candle"]:
            return None

        if row["close"] <= row["open"]:
            return None

        entry_price = row["close"]

        return {
            "direction": "BUY",
            "entry_index": setup_index,
            "entry_price": entry_price,
            "atr": atr
        }

    if direction == "SELL":

        if not row["bear_candle"]:
            return None

        if row["close"] >= row["open"]:
            return None

        entry_price = row["close"]

        return {
            "direction": "SELL",
            "entry_index": setup_index,
            "entry_price": entry_price,
            "atr": atr
        }

    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(df, trade):

    i = trade["entry_index"]

    entry = trade["entry_price"]
    atr = trade["atr"]
    direction = trade["direction"]

    if direction == "BUY":

        tp = entry + TP_ATR * atr
        sl = entry - SL_ATR * atr

    else:

        tp = entry - TP_ATR * atr
        sl = entry + SL_ATR * atr

    end_index = min(
        i + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    mfe = 0.0
    mae = 0.0

    result = None
    exit_price = None
    exit_index = None

    for j in range(i + 1, end_index + 1):

        candle = df.iloc[j]

        if direction == "BUY":

            favorable = (
                candle["high"] - entry
            ) / entry * 100

            adverse = (
                candle["low"] - entry
            ) / entry * 100

            mfe = max(mfe, favorable)
            mae = min(mae, adverse)

            hit_tp = candle["high"] >= tp
            hit_sl = candle["low"] <= sl

        else:

            favorable = (
                entry - candle["low"]
            ) / entry * 100

            adverse = (
                entry - candle["high"]
            ) / entry * 100

            mfe = max(mfe, favorable)
            mae = min(mae, adverse)

            hit_tp = candle["low"] <= tp
            hit_sl = candle["high"] >= sl

        # Conservative:
        # if TP and SL happen in same candle,
        # assume SL happened first.
        if hit_tp and hit_sl:

            result = "SL"
            exit_price = sl
            exit_index = j
            break

        if hit_tp:

            result = "TP"
            exit_price = tp
            exit_index = j
            break

        if hit_sl:

            result = "SL"
            exit_price = sl
            exit_index = j
            break

    # Timeout
    if result is None:

        exit_index = end_index
        exit_price = df.iloc[end_index]["close"]
        result = "TIMEOUT"

    # Return
    if direction == "BUY":

        gross_return = (
            (exit_price - entry) /
            entry
        )

    else:

        gross_return = (
            (entry - exit_price) /
            entry
        )

    net_return = (
        gross_return -
        ROUND_TRIP_COST
    )

    risk_amount = SL_ATR * atr / entry

    if risk_amount > 0:
        r_multiple = net_return / risk_amount
    else:
        r_multiple = 0

    return {
        "direction": direction,
        "entry_index": i,
        "exit_index": exit_index,
        "result": result,
        "return_pct": net_return * 100,
        "r": r_multiple,
        "mfe": mfe,
        "mae": mae
    }


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_config(
    df,
    start_index,
    end_index,
    pullback_atr,
    overextension_atr,
    confirm_body
):

    trades = []

    i = start_index

    while i < end_index:

        setup = get_setup(df, i)

        if setup is None:
            i += 1
            continue

        trade = timing_entry(
            df,
            i,
            setup,
            pullback_atr,
            overextension_atr,
            confirm_body
        )

        if trade is None:
            i += 1
            continue

        result = simulate_trade(df, trade)

        trades.append(result)

        # Cooldown
        i = result["exit_index"] + COOLDOWN_CANDLES

    return trades


# ============================================================
# STATISTICS
# ============================================================

def stats(trades):

    if not trades:
        return {
            "trades": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "tp_pct": 0,
            "sl_pct": 0,
            "timeout_pct": 0,
            "total_return": 0,
            "avg_return": 0,
            "total_r": 0,
            "avg_r": 0,
            "avg_mfe": 0,
            "avg_mae": 0
        }

    total = len(trades)

    tp = sum(
        t["result"] == "TP"
        for t in trades
    )

    sl = sum(
        t["result"] == "SL"
        for t in trades
    )

    timeout = sum(
        t["result"] == "TIMEOUT"
        for t in trades
    )

    total_return = sum(
        t["return_pct"]
        for t in trades
    )

    total_r = sum(
        t["r"]
        for t in trades
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
        "avg_return": total_return / total,

        "total_r": total_r,
        "avg_r": total_r / total,

        "avg_mfe": np.mean([
            t["mfe"]
            for t in trades
        ]),

        "avg_mae": np.mean([
            t["mae"]
            for t in trades
        ])
    }


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_candles()

    df = add_indicators(df)

    df = df.dropna(
        subset=[
            "ema9",
            "ema21",
            "ema50",
            "rsi",
            "macd",
            "macd_signal",
            "atr",
            "volume_ratio",
            "body_strength"
        ]
    ).reset_index(drop=True)

    print()
    print(f"📊 Final usable candles: {len(df):,}")
    print()

    print("=" * 70)
    print("SETTINGS")
    print("=" * 70)

    print(f"TP: {TP_ATR} × ATR")
    print(f"SL: {SL_ATR} × ATR")
    print(f"Max hold: {MAX_HOLD_CANDLES * 15} minutes")
    print(f"Cooldown: {COOLDOWN_CANDLES * 15} minutes")
    print(f"Round-trip cost: {ROUND_TRIP_COST * 100:.2f}%")
    print(f"Training: {TRAIN_DAYS} days")
    print(f"Validation: {VALID_DAYS} days")
    print()

    # --------------------------------------------------------
    # Walk-forward windows
    # --------------------------------------------------------

    start_time = df["time"].iloc[0]
    end_time = df["time"].iloc[-1]

    window_start = start_time

    validation_results = []

    window_number = 1

    while True:

        train_start = window_start
        train_end = train_start + timedelta(days=TRAIN_DAYS)

        valid_start = train_end
        valid_end = valid_start + timedelta(days=VALID_DAYS)

        if valid_end > end_time:
            break

        train_mask = (
            (df["time"] >= train_start) &
            (df["time"] < train_end)
        )

        valid_mask = (
            (df["time"] >= valid_start) &
            (df["time"] < valid_end)
        )

        train_indices = np.where(train_mask)[0]
        valid_indices = np.where(valid_mask)[0]

        if len(train_indices) == 0 or len(valid_indices) == 0:
            break

        train_start_i = train_indices[0]
        train_end_i = train_indices[-1]

        valid_start_i = valid_indices[0]
        valid_end_i = valid_indices[-1]

        print()
        print("=" * 70)
        print(f"🔎 WALK-FORWARD WINDOW {window_number}")
        print("=" * 70)

        print(
            f"Training : {train_start} → {train_end}"
        )

        print(
            f"Validation: {valid_start} → {valid_end}"
        )

        # ----------------------------------------------------
        # TRAINING
        # ----------------------------------------------------

        training_results = []

        for pullback in PULLBACK_ATR_OPTIONS:

            for overext in OVEREXTENSION_ATR_OPTIONS:

                # Invalid combination
                if overext < pullback:
                    continue

                for body in CONFIRM_BODY_OPTIONS:

                    trades = run_config(
                        df,
                        train_start_i,
                        train_end_i,
                        pullback,
                        overext,
                        body
                    )

                    s = stats(trades)

                    if s["trades"] >= MIN_TRAIN_TRADES:

                        training_results.append({
                            "pullback": pullback,
                            "overext": overext,
                            "body": body,
                            **s
                        })

        print()
        print("TRAINING CONFIGURATIONS")
        print("-" * 70)

        if not training_results:

            print(
                "No configuration reached minimum training trades."
            )

            window_start = valid_start
            window_number += 1
            continue

        # Sort by Avg R
        training_results.sort(
            key=lambda x: x["avg_r"],
            reverse=True
        )

        # Show every configuration
        for r in training_results:

            print(
                f"Pullback {r['pullback']:.2f} ATR | "
                f"Overext {r['overext']:.2f} ATR | "
                f"Body {r['body']:.2f} | "
                f"Trades {r['trades']:3d} | "
                f"TP {r['tp_pct']:5.1f}% | "
                f"SL {r['sl_pct']:5.1f}% | "
                f"AvgR {r['avg_r']:+.3f}"
            )

        # ----------------------------------------------------
        # IMPORTANT:
        # Choose ONLY from training.
        # Do NOT use validation to choose parameters.
        # ----------------------------------------------------

        positive_training = [
            r for r in training_results
            if r["avg_r"] > 0
        ]

        print()

        if not positive_training:

            print(
                "⚠️ No positive Avg R configuration in training."
            )

            print(
                "➡️ Validation skipped for this window."
            )

        else:

            # Take best training configuration
            selected = positive_training[0]

            print(
                "✅ SELECTED FROM TRAINING ONLY:"
            )

            print(
                f"Pullback: {selected['pullback']:.2f} ATR"
            )

            print(
                f"Overextension: {selected['overext']:.2f} ATR"
            )

            print(
                f"Body strength: {selected['body']:.2f}"
            )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            validation_trades = run_config(
                df,
                valid_start_i,
                valid_end_i,
                selected["pullback"],
                selected["overext"],
                selected["body"]
            )

            vs = stats(validation_trades)

            print()
            print("VALIDATION RESULT")
            print("-" * 70)

            print(
                f"Trades: {vs['trades']}"
            )

            print(
                f"TP: {vs['tp']} ({vs['tp_pct']:.1f}%)"
            )

            print(
                f"SL: {vs['sl']} ({vs['sl_pct']:.1f}%)"
            )

            print(
                f"Timeout: "
                f"{vs['timeout']} "
                f"({vs['timeout_pct']:.1f}%)"
            )

            print(
                f"Total return: "
                f"{vs['total_return']:+.2f}%"
            )

            print(
                f"Average return: "
                f"{vs['avg_return']:+.3f}%"
            )

            print(
                f"Total R: "
                f"{vs['total_r']:+.2f}"
            )

            print(
                f"Average R: "
                f"{vs['avg_r']:+.3f}"
            )

            print(
                f"Average MFE: "
                f"{vs['avg_mfe']:+.3f}%"
            )

            print(
                f"Average MAE: "
                f"{vs['avg_mae']:+.3f}%"
            )

            validation_results.append({
                "window": window_number,
                **selected,
                "validation": vs
            })

        window_start = valid_start
        window_number += 1

    # ========================================================
    # OVERALL VALIDATION
    # ========================================================

    print()
    print()
    print("=" * 70)
    print("📊 OVERALL VALIDATION")
    print("=" * 70)

    if not validation_results:

        print(
            "No validation configuration passed the training filter."
        )

        print(
            "This means the strategy did not demonstrate a "
            "positive training edge in these windows."
        )

        return

    all_validation_trades = []

    for item in validation_results:

        # Reconstruct validation period
        w = item["window"]

        # We don't have stored trades here,
        # so report window-level validation results.
        v = item["validation"]

        print(
            f"Window {w}: "
            f"{v['trades']} trades | "
            f"TP {v['tp_pct']:.1f}% | "
            f"SL {v['sl_pct']:.1f}% | "
            f"AvgR {v['avg_r']:+.3f}"
        )

    print()
    print("=" * 70)
    print("IMPORTANT")
    print("=" * 70)

    print(
        "Validation results were NOT used to choose parameters."
    )

    print(
        "Future MFE/MAE was NOT used to decide entry."
    )

    print(
        "A positive training result does NOT guarantee future performance."
    )

    print(
        "Do not switch the live Telegram bot to this strategy "
        "unless validation remains positive across additional data."
    )


if __name__ == "__main__":
    main()

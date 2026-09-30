import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M — 6 MONTH ENTRY TIMING + MFE/MAE BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
CHUNK_CANDLES = 250
DAYS = 183

TRAIN_DAYS = 60
VALID_DAYS = 30

TP_ATR = 1.5
SL_ATR = 1.0

MAX_HOLD_CANDLES = 12      # 3 hours
COOLDOWN_CANDLES = 12      # 3 hours

COST = 0.0014              # 0.14% round-trip

MIN_TRAIN_TRADES = 10

API_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current = start_time

    print("🚀 BTC 6-Month Entry Timing Backtest Started!")
    print()

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
            f"to "
            f"{chunk_end.strftime('%Y-%m-%d %H:%M')}"
        )

        try:
            r = requests.get(
                API_URL,
                params=params,
                timeout=30
            )

            r.raise_for_status()

            data = r.json()

            if isinstance(data, list):
                all_rows.extend(data)

        except Exception as e:
            print("Download error:", e)

        current = chunk_end

    if not all_rows:
        raise RuntimeError("No market data downloaded.")

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

    print()
    print("Total candles downloaded:", len(df))

    if len(df):
        print("First candle:", df["time"].iloc[0])
        print("Last candle: ", df["time"].iloc[-1])

    return df


# ============================================================
# INDICATORS
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
    df["volume_avg"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    # Candle body
    df["body"] = (
        df["close"] -
        df["open"]
    )

    df["body_pct"] = (
        df["body"].abs() /
        df["close"]
    )

    # Recent high / low
    df["recent_high"] = (
        df["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    df["recent_low"] = (
        df["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # Distance from EMA21 in ATR
    df["ema21_distance_atr"] = (
        (df["close"] - df["ema21"]).abs()
        / df["atr"]
    )

    return df


# ============================================================
# SIGNAL
# ============================================================

def get_signal(row):

    needed = [
        "close",
        "ema9",
        "ema21",
        "ema50",
        "rsi",
        "macd",
        "macd_signal",
        "atr",
        "volume_ratio"
    ]

    for x in needed:
        if pd.isna(row[x]):
            return None

    buy_score = 0
    sell_score = 0

    # Trend
    if (
        row["ema9"] >
        row["ema21"] >
        row["ema50"]
    ):
        buy_score += 3

    if (
        row["ema9"] <
        row["ema21"] <
        row["ema50"]
    ):
        sell_score += 3

    # Price location
    if row["close"] > row["ema21"]:
        buy_score += 1

    if row["close"] < row["ema21"]:
        sell_score += 1

    # MACD
    if row["macd"] > row["macd_signal"]:
        buy_score += 2

    if row["macd"] < row["macd_signal"]:
        sell_score += 2

    # RSI
    if 50 <= row["rsi"] <= 68:
        buy_score += 1

    if 32 <= row["rsi"] < 50:
        sell_score += 1

    # Volume
    if row["volume_ratio"] >= 0.85:
        if buy_score > sell_score:
            buy_score += 1
        elif sell_score > buy_score:
            sell_score += 1

    # Avoid overextended entry
    if row["ema21_distance_atr"] > 1.5:
        return None

    # Candle confirmation
    if row["body"] > 0:
        buy_score += 1

    if row["body"] < 0:
        sell_score += 1

    if buy_score >= 7 and buy_score > sell_score:
        return "BUY"

    if sell_score >= 7 and sell_score > buy_score:
        return "SELL"

    return None


# ============================================================
# MFE / MAE ANALYSIS
# ============================================================

def analyze_trade(df, entry_index, direction):

    entry = df.iloc[entry_index]

    entry_price = entry["close"]
    atr = entry["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp_price = (
            entry_price +
            TP_ATR * atr
        )

        sl_price = (
            entry_price -
            SL_ATR * atr
        )

    else:

        tp_price = (
            entry_price -
            TP_ATR * atr
        )

        sl_price = (
            entry_price +
            SL_ATR * atr
        )

    end_index = min(
        entry_index + MAX_HOLD_CANDLES,
        len(df) - 1
    )

    future = df.iloc[
        entry_index + 1:
        end_index + 1
    ]

    if len(future) == 0:
        return None

    # --------------------------------------------------------
    # MFE / MAE
    # --------------------------------------------------------

    if direction == "BUY":

        best_price = future["high"].max()

        worst_price = future["low"].min()

        mfe_pct = (
            (best_price - entry_price)
            / entry_price
            * 100
        )

        mae_pct = (
            (worst_price - entry_price)
            / entry_price
            * 100
        )

    else:

        best_price = future["low"].min()

        worst_price = future["high"].max()

        mfe_pct = (
            (entry_price - best_price)
            / entry_price
            * 100
        )

        mae_pct = (
            (entry_price - worst_price)
            / entry_price
            * 100
        )

    # --------------------------------------------------------
    # Did TP or SL happen?
    # --------------------------------------------------------

    result = "TIMEOUT"

    exit_price = future["close"].iloc[-1]

    exit_index = future.index[-1]

    for idx, candle in future.iterrows():

        if direction == "BUY":

            hit_tp = candle["high"] >= tp_price
            hit_sl = candle["low"] <= sl_price

        else:

            hit_tp = candle["low"] <= tp_price
            hit_sl = candle["high"] >= sl_price

        # Conservative:
        # if both happen same candle -> SL first
        if hit_tp and hit_sl:

            result = "SL"
            exit_price = sl_price
            exit_index = idx
            break

        elif hit_tp:

            result = "TP"
            exit_price = tp_price
            exit_index = idx
            break

        elif hit_sl:

            result = "SL"
            exit_price = sl_price
            exit_index = idx
            break

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

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
        COST
    )

    # R based on ATR stop distance
    risk_pct = (
        SL_ATR * atr /
        entry_price
    )

    if risk_pct > 0:
        r_multiple = (
            net_return /
            risk_pct
        )
    else:
        r_multiple = 0

    return {
        "direction": direction,
        "result": result,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "mfe_pct": mfe_pct,
        "mae_pct": mae_pct,
        "return_pct": net_return * 100,
        "r": r_multiple,
        "entry_index": entry_index,
        "exit_index": exit_index
    }


# ============================================================
# GENERATE ALL TRADES
# ============================================================

def generate_trades(df, start_index, end_index):

    trades = []

    next_allowed = start_index

    for i in range(
        start_index,
        end_index
    ):

        if i < next_allowed:
            continue

        row = df.iloc[i]

        signal = get_signal(row)

        if signal is None:
            continue

        trade = analyze_trade(
            df,
            i,
            signal
        )

        if trade is None:
            continue

        trades.append(trade)

        # no overlapping trades
        next_allowed = (
            trade["exit_index"] +
            COOLDOWN_CANDLES
        )

    return trades


# ============================================================
# SUMMARY
# ============================================================

def summarize(trades, title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not trades:

        print("No trades.")
        return

    df = pd.DataFrame(trades)

    total = len(df)

    tp = (
        df["result"] == "TP"
    ).sum()

    sl = (
        df["result"] == "SL"
    ).sum()

    timeout = (
        df["result"] == "TIMEOUT"
    ).sum()

    print("Trades:", total)

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
        f"{df['return_pct'].sum():.2f}%"
    )

    print(
        f"Average return: "
        f"{df['return_pct'].mean():.3f}%"
    )

    print(
        f"Total R: "
        f"{df['r'].sum():.2f}"
    )

    print(
        f"Average R: "
        f"{df['r'].mean():.3f}"
    )

    print(
        f"Average MFE: "
        f"{df['mfe_pct'].mean():.3f}%"
    )

    print(
        f"Average MAE: "
        f"{df['mae_pct'].mean():.3f}%"
    )

    # --------------------------------------------------------
    # How often price reached useful levels
    # --------------------------------------------------------

    print()
    print("MFE analysis:")

    for level in [
        0.10,
        0.20,
        0.30,
        0.50,
        0.75,
        1.00
    ]:

        count = (
            df["mfe_pct"] >= level
        ).sum()

        print(
            f"MFE >= {level:.2f}%: "
            f"{count}/{total} "
            f"({count / total * 100:.1f}%)"
        )

    print()
    print("MAE analysis:")

    for level in [
        -0.10,
        -0.20,
        -0.30,
        -0.50,
        -0.75,
        -1.00
    ]:

        count = (
            df["mae_pct"] <= level
        ).sum()

        print(
            f"MAE <= {level:.2f}%: "
            f"{count}/{total} "
            f"({count / total * 100:.1f}%)"
        )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    print()
    print("Direction:")

    for direction in [
        "BUY",
        "SELL"
    ]:

        part = df[
            df["direction"] == direction
        ]

        if len(part) == 0:
            continue

        print()
        print(direction)

        print(
            "Trades:",
            len(part)
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
            f"Avg return: "
            f"{part['return_pct'].mean():.3f}%"
        )

        print(
            f"Avg R: "
            f"{part['r'].mean():.3f}"
        )

        print(
            f"Avg MFE: "
            f"{part['mfe_pct'].mean():.3f}%"
        )

        print(
            f"Avg MAE: "
            f"{part['mae_pct'].mean():.3f}%"
        )


# ============================================================
# ENTRY TIMING ANALYSIS
# ============================================================

def entry_timing_analysis(trades):

    print()
    print("=" * 70)
    print("ENTRY TIMING ANALYSIS")
    print("=" * 70)

    if not trades:
        print("No trades.")
        return

    df = pd.DataFrame(trades)

    # MFE minus absolute MAE
    df["movement_quality"] = (
        df["mfe_pct"] +
        df["mae_pct"]
    )

    print(
        f"Average MFE: "
        f"{df['mfe_pct'].mean():.3f}%"
    )

    print(
        f"Average MAE: "
        f"{df['mae_pct'].mean():.3f}%"
    )

    print(
        f"Average movement quality: "
        f"{df['movement_quality'].mean():.3f}%"
    )

    # --------------------------------------------------------
    # Entry quality buckets
    # --------------------------------------------------------

    print()
    print("Entry quality buckets:")

    buckets = [
        ("MFE < 0.20%", df["mfe_pct"] < 0.20),
        (
            "MFE 0.20-0.40%",
            (
                (df["mfe_pct"] >= 0.20) &
                (df["mfe_pct"] < 0.40)
            )
        ),
        (
            "MFE 0.40-0.75%",
            (
                (df["mfe_pct"] >= 0.40) &
                (df["mfe_pct"] < 0.75)
            )
        ),
        (
            "MFE >= 0.75%",
            df["mfe_pct"] >= 0.75
        )
    ]

    for name, mask in buckets:

        part = df[mask]

        if len(part) == 0:
            continue

        print(
            f"{name}: "
            f"{len(part)} trades | "
            f"Avg return "
            f"{part['return_pct'].mean():.3f}% | "
            f"Avg R "
            f"{part['r'].mean():.3f}"
        )

    # --------------------------------------------------------
    # Maximum adverse movement
    # --------------------------------------------------------

    print()
    print("Bad-entry analysis:")

    bad_entry = df[
        df["mae_pct"] <= -0.30
    ]

    print(
        f"MAE <= -0.30%: "
        f"{len(bad_entry)}/{len(df)} "
        f"({len(bad_entry) / len(df) * 100:.1f}%)"
    )

    good_entry = df[
        df["mae_pct"] > -0.20
    ]

    print(
        f"MAE > -0.20%: "
        f"{len(good_entry)}/{len(df)} "
        f"({len(good_entry) / len(df) * 100:.1f}%)"
    )


# ============================================================
# WALK FORWARD
# ============================================================

def run_walk_forward(df):

    start_time = df["time"].min()
    end_time = df["time"].max()

    current_train_start = start_time

    all_validation = []

    window = 1

    while True:

        train_start = (
            current_train_start
        )

        train_end = (
            train_start +
            timedelta(days=TRAIN_DAYS)
        )

        valid_start = train_end

        valid_end = (
            valid_start +
            timedelta(days=VALID_DAYS)
        )

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

        train_indices = np.where(
            train_mask
        )[0]

        valid_indices = np.where(
            valid_mask
        )[0]

        if len(train_indices) == 0:
            break

        if len(valid_indices) == 0:
            break

        train_start_idx = train_indices[0]
        train_end_idx = train_indices[-1]

        valid_start_idx = valid_indices[0]
        valid_end_idx = valid_indices[-1]

        print()
        print("#" * 70)
        print(f"WALK-FORWARD WINDOW {window}")
        print("#" * 70)

        print(
            "Training:",
            train_start,
            "->",
            train_end
        )

        print(
            "Validation:",
            valid_start,
            "->",
            valid_end
        )

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_trades = generate_trades(
            df,
            train_start_idx,
            train_end_idx
        )

        summarize(
            train_trades,
            f"WINDOW {window} TRAINING"
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        valid_trades = generate_trades(
            df,
            valid_start_idx,
            valid_end_idx
        )

        summarize(
            valid_trades,
            f"WINDOW {window} VALIDATION"
        )

        entry_timing_analysis(
            valid_trades
        )

        all_validation.extend(
            valid_trades
        )

        current_train_start = (
            current_train_start +
            timedelta(days=30)
        )

        window += 1

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    summarize(
        all_validation,
        "OVERALL WALK-FORWARD VALIDATION"
    )

    entry_timing_analysis(
        all_validation
    )

    print()
    print("=" * 70)
    print("BACKTEST FINISHED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_data()

    df = add_indicators(df)

    # Remove incomplete indicator rows
    df = df.dropna(
        subset=[
            "ema50",
            "rsi",
            "macd",
            "macd_signal",
            "atr",
            "volume_ratio"
        ]
    ).reset_index(drop=True)

    print()
    print(
        "Final usable candles:",
        len(df)
    )

    print()
    print("Settings:")
    print(
        f"TP: {TP_ATR} x ATR"
    )
    print(
        f"SL: {SL_ATR} x ATR"
    )
    print(
        "Maximum holding:",
        MAX_HOLD_CANDLES * 15,
        "minutes"
    )
    print(
        "Cooldown:",
        COOLDOWN_CANDLES * 15,
        "minutes"
    )
    print(
        f"Round-trip estimated cost: "
        f"{COST * 100:.2f}%"
    )

    print()
    print(
        f"Training: {TRAIN_DAYS} days"
    )

    print(
        f"Validation: {VALID_DAYS} days"
    )

    run_walk_forward(df)


if __name__ == "__main__":
    main()

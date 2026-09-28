import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict

# ============================================================
# BTC 15M TRAIN + VALIDATION BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
DAYS = 183                 # ~6 months

CHUNK_CANDLES = 250        # Coinbase max candles per request
LOOKBACK = 100

TP_ATR = 1.5
SL_ATR = 1.0

MAX_HOLD_CANDLES = 12      # 12 x 15m = 3 hours

ROUND_TRIP_COST = 0.0014   # 0.14%

# Training / validation
TRAIN_DAYS = 122            # ~4 months
VALIDATION_DAYS = 61        # ~2 months

# Pattern filter
MIN_TRAIN_SIGNALS = 30

# Maximum number of selected patterns per side
MAX_BUY_PATTERNS = 5
MAX_SELL_PATTERNS = 5


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    print("Downloading BTC 15M candles...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current = start_time

    while current < end_time:

        chunk_end = min(
            current + timedelta(minutes=GRANULARITY // 60 * CHUNK_CANDLES),
            end_time
        )

        start_epoch = int(current.timestamp())
        end_epoch = int(chunk_end.timestamp())

        print(
            f"Downloading: "
            f"{current.strftime('%Y-%m-%d')} to "
            f"{chunk_end.strftime('%Y-%m-%d')}"
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
                all_rows.extend(data)

        except Exception as e:

            print("Download error:", e)

        current = chunk_end

    if not all_rows:
        raise RuntimeError("No BTC data downloaded.")

    # Coinbase:
    # [time, low, high, open, close, volume]

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

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    df = df.dropna()

    print()
    print("Total candles downloaded:", len(df))

    if len(df) > 0:

        print(
            "First candle:",
            df["timestamp"].iloc[0]
        )

        print(
            "Last candle: ",
            df["timestamp"].iloc[-1]
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

    avg_gain = gain.rolling(
        14
    ).mean()

    avg_loss = loss.rolling(
        14
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
    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

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

    df["atr"] = true_range.rolling(
        14
    ).mean()

    # Volume average
    df["volume_avg"] = df["volume"].rolling(
        20
    ).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    # 20-candle breakout
    df["previous_high_20"] = (
        df["high"]
        .shift(1)
        .rolling(20)
        .max()
    )

    df["previous_low_20"] = (
        df["low"]
        .shift(1)
        .rolling(20)
        .min()
    )

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def get_market_regime(row):

    ema9 = row["ema9"]
    ema21 = row["ema21"]
    ema50 = row["ema50"]
    atr = row["atr"]
    close = row["close"]

    if pd.isna(atr) or atr <= 0:
        return "SIDEWAYS"

    # Strong upward trend
    if (
        ema9 > ema21
        and ema21 > ema50
        and (ema9 - ema50) > atr * 0.5
    ):
        return "TREND_UP"

    # Strong downward trend
    if (
        ema9 < ema21
        and ema21 < ema50
        and (ema50 - ema9) > atr * 0.5
    ):
        return "TREND_DOWN"

    return "SIDEWAYS"


# ============================================================
# PATTERN
# ============================================================

def get_pattern(row):

    regime = get_market_regime(row)

    rsi = row["rsi"]

    if pd.isna(rsi):
        rsi_state = "RSI_NEUTRAL"

    elif rsi >= 60:
        rsi_state = "RSI_STRONG"

    elif rsi <= 40:
        rsi_state = "RSI_WEAK"

    else:
        rsi_state = "RSI_NEUTRAL"

    if row["macd"] > row["macd_signal"]:
        macd_state = "MACD_UP"
    else:
        macd_state = "MACD_DOWN"

    volume_ratio = row["volume_ratio"]

    if pd.isna(volume_ratio):
        volume_state = "VOLUME_NORMAL"

    elif volume_ratio >= 1.5:
        volume_state = "VOLUME_STRONG"

    elif volume_ratio <= 0.7:
        volume_state = "VOLUME_WEAK"

    else:
        volume_state = "VOLUME_NORMAL"

    if (
        row["close"] >
        row["previous_high_20"]
    ):
        breakout_state = "BREAKOUT_UP"

    elif (
        row["close"] <
        row["previous_low_20"]
    ):
        breakout_state = "BREAKOUT_DOWN"

    else:
        breakout_state = "NO_BREAKOUT"

    pattern = (
        f"{regime}|"
        f"{rsi_state}|"
        f"{macd_state}|"
        f"{volume_state}|"
        f"{breakout_state}"
    )

    return pattern


# ============================================================
# SIGNAL
# ============================================================

def get_signal(row):

    regime = get_market_regime(row)

    # Do not trade sideways markets
    if regime == "SIDEWAYS":
        return "WAIT"

    buy_score = 0
    sell_score = 0

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if row["ema9"] > row["ema21"]:
        buy_score += 2

    elif row["ema9"] < row["ema21"]:
        sell_score += 2

    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    if regime == "TREND_UP":
        buy_score += 2

    elif regime == "TREND_DOWN":
        sell_score += 2

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if row["rsi"] >= 55:
        buy_score += 1

    elif row["rsi"] <= 45:
        sell_score += 1

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if row["macd"] > row["macd_signal"]:
        buy_score += 2

    elif row["macd"] < row["macd_signal"]:
        sell_score += 2

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    if row["volume_ratio"] >= 1.5:

        if buy_score > sell_score:
            buy_score += 1

        elif sell_score > buy_score:
            sell_score += 1

    # --------------------------------------------------------
    # BREAKOUT
    # --------------------------------------------------------

    if (
        row["close"] >
        row["previous_high_20"]
    ):
        buy_score += 2

    elif (
        row["close"] <
        row["previous_low_20"]
    ):
        sell_score += 2

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

    if (
        buy_score >= 6
        and buy_score > sell_score
    ):
        return "BUY"

    if (
        sell_score >= 6
        and sell_score > buy_score
    ):
        return "SELL"

    return "WAIT"


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    signal
):

    row = df.iloc[entry_index]

    entry_price = row["close"]

    atr = row["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    risk_distance = atr * SL_ATR

    reward_distance = atr * TP_ATR

    if signal == "BUY":

        tp_price = (
            entry_price +
            reward_distance
        )

        sl_price = (
            entry_price -
            risk_distance
        )

    elif signal == "SELL":

        tp_price = (
            entry_price -
            reward_distance
        )

        sl_price = (
            entry_price +
            risk_distance
        )

    else:
        return None

    last_index = min(
        entry_index +
        MAX_HOLD_CANDLES,
        len(df) - 1
    )

    for future_index in range(
        entry_index + 1,
        last_index + 1
    ):

        future = df.iloc[
            future_index
        ]

        high = future["high"]
        low = future["low"]

        # ----------------------------------------------------
        # BUY
        # ----------------------------------------------------

        if signal == "BUY":

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

            # Conservative assumption:
            # if both happen inside same candle,
            # assume SL happened first.
            if hit_tp and hit_sl:

                gross_r = -1.0

                net_return = (
                    -risk_distance /
                    entry_price
                )

                net_return -= ROUND_TRIP_COST

                net_r = (
                    net_return /
                    (risk_distance /
                     entry_price)
                )

                return {
                    "result": "SL",
                    "r": net_r
                }

            if hit_sl:

                gross_r = -1.0

                net_return = (
                    -risk_distance /
                    entry_price
                )

                net_return -= ROUND_TRIP_COST

                net_r = (
                    net_return /
                    (risk_distance /
                     entry_price)
                )

                return {
                    "result": "SL",
                    "r": net_r
                }

            if hit_tp:

                net_return = (
                    reward_distance /
                    entry_price
                )

                net_return -= ROUND_TRIP_COST

                net_r = (
                    net_return /
                    (risk_distance /
                     entry_price)
                )

                return {
                    "result": "TP",
                    "r": net_r
                }

        # ----------------------------------------------------
        # SELL
        # ----------------------------------------------------

        elif signal == "SELL":

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

            if hit_tp and hit_sl:

                net_return = (
                    -risk_distance /
                    entry_price
                )

                net_return -= ROUND_TRIP_COST

                net_r = (
                    net_return /
                    (risk_distance /
                     entry_price)
                )

                return {
                    "result": "SL",
                    "r": net_r
                }

            if hit_sl:

                net_return = (
                    -risk_distance /
                    entry_price
                )

                net_return -= ROUND_TRIP_COST

                net_r = (
                    net_return /
                    (risk_distance /
                     entry_price)
                )

                return {
                    "result": "SL",
                    "r": net_r
                }

            if hit_tp:

                net_return = (
                    reward_distance /
                    entry_price
                )

                net_return -= ROUND_TRIP_COST

                net_r = (
                    net_return /
                    (risk_distance /
                     entry_price)
                )

                return {
                    "result": "TP",
                    "r": net_r
                }

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    final_price = df.iloc[
        last_index
    ]["close"]

    if signal == "BUY":

        raw_return = (
            final_price -
            entry_price
        ) / entry_price

    else:

        raw_return = (
            entry_price -
            final_price
        ) / entry_price

    net_return = (
        raw_return -
        ROUND_TRIP_COST
    )

    net_r = (
        net_return /
        (risk_distance /
         entry_price)
    )

    return {
        "result": "TIMEOUT",
        "r": net_r
    }


# ============================================================
# TRAINING STATISTICS
# ============================================================

def collect_training_statistics(
    df,
    start_index,
    end_index
):

    statistics = defaultdict(
        lambda: {
            "BUY": [],
            "SELL": []
        }
    )

    for i in range(
        max(start_index, LOOKBACK),
        min(end_index, len(df) - MAX_HOLD_CANDLES)
    ):

        row = df.iloc[i]

        if (
            pd.isna(row["atr"])
            or pd.isna(row["rsi"])
            or pd.isna(row["volume_ratio"])
            or pd.isna(row["previous_high_20"])
            or pd.isna(row["previous_low_20"])
        ):
            continue

        signal = get_signal(row)

        if signal == "WAIT":
            continue

        pattern = get_pattern(row)

        trade = simulate_trade(
            df,
            i,
            signal
        )

        if trade is None:
            continue

        statistics[pattern][signal].append(
            trade
        )

    return statistics


# ============================================================
# PRINT TRAINING CANDIDATES
# ============================================================

def calculate_stats(trades):

    if not trades:
        return None

    total = len(trades)

    tp = sum(
        1 for x in trades
        if x["result"] == "TP"
    )

    sl = sum(
        1 for x in trades
        if x["result"] == "SL"
    )

    timeout = sum(
        1 for x in trades
        if x["result"] == "TIMEOUT"
    )

    total_r = sum(
        x["r"] for x in trades
    )

    avg_r = total_r / total

    return {
        "n": total,
        "tp": tp,
        "sl": sl,
        "timeout": timeout,
        "tp_rate": tp / total,
        "sl_rate": sl / total,
        "timeout_rate": timeout / total,
        "total_r": total_r,
        "avg_r": avg_r
    }


def print_training_candidates(
    statistics
):

    print()
    print("=" * 100)
    print("TRAINING PATTERN RESULTS")
    print("=" * 100)

    candidates = {
        "BUY": [],
        "SELL": []
    }

    for pattern, sides in statistics.items():

        for side in ["BUY", "SELL"]:

            trades = sides[side]

            stats = calculate_stats(
                trades
            )

            if stats is None:
                continue

            if stats["n"] < MIN_TRAIN_SIGNALS:
                continue

            # More flexible filter:
            #
            # 1. At least 30 trades
            # 2. Average R cannot be too negative
            # 3. Either TP rate is reasonable
            #    OR average R is reasonably good
            #
            if stats["avg_r"] < -0.10:
                continue

            if (
                stats["tp_rate"] < 0.30
                and stats["avg_r"] < 0.05
            ):
                continue

            candidates[side].append(
                (
                    pattern,
                    stats
                )
            )

    # Sort by average R
    for side in ["BUY", "SELL"]:

        candidates[side].sort(
            key=lambda x: (
                x[1]["avg_r"],
                x[1]["tp_rate"],
                x[1]["n"]
            ),
            reverse=True
        )

    for side in ["BUY", "SELL"]:

        print()
        print(f"{side} TRAINING CANDIDATES")
        print("-" * 100)

        if not candidates[side]:

            print(
                "No patterns passed the training filter."
            )

            continue

        print(
            f"{'N':>5} "
            f"{'TP%':>7} "
            f"{'SL%':>7} "
            f"{'TO%':>7} "
            f"{'Total R':>10} "
            f"{'Avg R':>9}  "
            f"Pattern"
        )

        print("-" * 100)

        for pattern, stats in candidates[side]:

            print(
                f"{stats['n']:>5} "
                f"{stats['tp_rate'] * 100:>6.1f}% "
                f"{stats['sl_rate'] * 100:>6.1f}% "
                f"{stats['timeout_rate'] * 100:>6.1f}% "
                f"{stats['total_r']:>+10.2f} "
                f"{stats['avg_r']:>+9.3f}  "
                f"{pattern}"
            )

    return candidates


# ============================================================
# SELECT TOP TRAINING PATTERNS
# ============================================================

def select_patterns(
    candidates
):

    selected = {
        "BUY": set(),
        "SELL": set()
    }

    for side, maximum in [
        ("BUY", MAX_BUY_PATTERNS),
        ("SELL", MAX_SELL_PATTERNS)
    ]:

        for pattern, stats in candidates[side][
            :maximum
        ]:

            selected[side].add(
                pattern
            )

    return selected


# ============================================================
# VALIDATION
# ============================================================

def run_validation(
    df,
    validation_start,
    validation_end,
    selected_patterns
):

    results = {
        "BUY": [],
        "SELL": []
    }

    last_trade_index = {
        "BUY": -999999,
        "SELL": -999999
    }

    for i in range(
        max(
            validation_start,
            LOOKBACK
        ),
        min(
            validation_end,
            len(df) - MAX_HOLD_CANDLES
        )
    ):

        row = df.iloc[i]

        if (
            pd.isna(row["atr"])
            or pd.isna(row["rsi"])
            or pd.isna(row["volume_ratio"])
            or pd.isna(row["previous_high_20"])
            or pd.isna(row["previous_low_20"])
        ):
            continue

        signal = get_signal(row)

        if signal == "WAIT":
            continue

        pattern = get_pattern(row)

        if pattern not in selected_patterns[signal]:
            continue

        # ----------------------------------------------------
        # NO OVERLAPPING TRADES
        # ----------------------------------------------------

        if (
            i -
            last_trade_index[signal]
            <= MAX_HOLD_CANDLES
        ):
            continue

        trade = simulate_trade(
            df,
            i,
            signal
        )

        if trade is None:
            continue

        trade["index"] = i
        trade["time"] = row["timestamp"]

        results[signal].append(
            trade
        )

        last_trade_index[signal] = i

    return results


# ============================================================
# PRINT VALIDATION RESULTS
# ============================================================

def print_validation_results(
    results
):

    print()
    print("=" * 100)
    print("VALIDATION RESULTS")
    print("=" * 100)

    total_trades = 0
    total_r = 0.0

    for side in ["BUY", "SELL"]:

        trades = results[side]

        total = len(trades)

        tp = sum(
            1 for x in trades
            if x["result"] == "TP"
        )

        sl = sum(
            1 for x in trades
            if x["result"] == "SL"
        )

        timeout = sum(
            1 for x in trades
            if x["result"] == "TIMEOUT"
        )

        side_total_r = sum(
            x["r"] for x in trades
        )

        avg_r = (
            side_total_r / total
            if total > 0
            else 0
        )

        total_trades += total
        total_r += side_total_r

        if total > 0:

            tp_pct = tp / total * 100
            sl_pct = sl / total * 100
            timeout_pct = timeout / total * 100

        else:

            tp_pct = 0
            sl_pct = 0
            timeout_pct = 0

        print()
        print(
            f"{side}: "
            f"{total} trades | "
            f"TP {tp} ({tp_pct:.1f}%) | "
            f"SL {sl} ({sl_pct:.1f}%) | "
            f"Timeout {timeout} ({timeout_pct:.1f}%) | "
            f"Total R {side_total_r:+.2f} | "
            f"Average R {avg_r:+.3f}"
        )

    print()
    print(
        f"VALIDATION TOTAL TRADES: "
        f"{total_trades}"
    )

    print(
        f"VALIDATION TOTAL R: "
        f"{total_r:+.2f}"
    )

    if total_trades > 0:

        print(
            f"VALIDATION AVERAGE R: "
            f"{total_r / total_trades:+.3f}"
        )

    return total_trades, total_r


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("🚀 BTC 6-Month Train + Validation Backtest")
    print()

    df = download_data()

    df = calculate_indicators(
        df
    )

    total_candles = len(df)

    # Approximately 4 months
    train_candles = int(
        total_candles *
        TRAIN_DAYS /
        DAYS
    )

    validation_start = train_candles

    validation_end = total_candles

    print()
    print(
        "Total candles:",
        total_candles
    )

    print(
        "Training range:",
        0,
        "to",
        train_candles
    )

    print(
        "Validation range:",
        validation_start,
        "to",
        validation_end
    )

    print()
    print(
        f"TP: {TP_ATR} x ATR"
    )

    print(
        f"SL: {SL_ATR} x ATR"
    )

    print(
        f"Maximum holding: "
        f"{MAX_HOLD_CANDLES * 15} minutes"
    )

    print(
        f"Round-trip estimated cost: "
        f"{ROUND_TRIP_COST * 100:.2f}%"
    )

    print()
    print("=" * 100)
    print("TRAINING: First approximately 4 months")
    print("=" * 100)

    statistics = collect_training_statistics(
        df,
        0,
        train_candles
    )

    candidates = print_training_candidates(
        statistics
    )

    selected = select_patterns(
        candidates
    )

    print()
    print("=" * 100)
    print("SELECTED PATTERNS")
    print("=" * 100)

    print()
    print(
        "Selected BUY patterns:",
        len(selected["BUY"])
    )

    for pattern in selected["BUY"]:
        print(
            "  BUY  ->",
            pattern
        )

    print()
    print(
        "Selected SELL patterns:",
        len(selected["SELL"])
    )

    for pattern in selected["SELL"]:
        print(
            "  SELL ->",
            pattern
        )

    print()
    print("=" * 100)
    print("VALIDATION: Last approximately 2 months")
    print("=" * 100)

    results = run_validation(
        df,
        validation_start,
        validation_end,
        selected
    )

    total_trades, total_r = (
        print_validation_results(
            results
        )
    )

    print()
    print("=" * 100)
    print("FINAL VALIDATION RESULT")
    print("=" * 100)

    if total_trades == 0:

        print()
        print(
            "No validation trades passed "
            "the training pattern filter."
        )

    else:

        print()

        if total_r > 0:

            print(
                "Validation total R is POSITIVE:",
                f"{total_r:+.2f}"
            )

        elif total_r < 0:

            print(
                "Validation total R is NEGATIVE:",
                f"{total_r:+.2f}"
            )

        else:

            print(
                "Validation total R is approximately ZERO."
            )

        print()
        print(
            "Important: this is a historical backtest, "
            "not a guarantee of future performance."
        )

    print()
    print("Backtest finished.")


if __name__ == "__main__":
    main()

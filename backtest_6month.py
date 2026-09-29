import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M MARKET REGIME + PULLBACK WALK-FORWARD BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
CHUNK_CANDLES = 250

DAYS = 183

TP_ATR = 1.5
SL_ATR = 1.0

MAX_HOLD_CANDLES = 12     # 3 hours
COOLDOWN_CANDLES = 12     # 3 hours

ROUND_TRIP_COST = 0.0014  # 0.14%

TRAIN_DAYS = 60
VALIDATION_DAYS = 30

MIN_PATTERN_TRADES = 15
MAX_PATTERNS_PER_SIDE = 5


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_candles():

    print("🚀 BTC 6-Month Regime Backtest Started!")
    print()
    print("📥 Downloading BTC-USD 15M data...")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current_start = int(start_time.timestamp())

    while current_start < int(end_time.timestamp()):

        current_end = min(
            current_start + GRANULARITY * CHUNK_CANDLES,
            int(end_time.timestamp())
        )

        url = (
            f"https://api.exchange.coinbase.com/products/"
            f"{PRODUCT}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": datetime.fromtimestamp(
                current_start, timezone.utc
            ).isoformat(),
            "end": datetime.fromtimestamp(
                current_end, timezone.utc
            ).isoformat()
        }

        try:

            response = requests.get(
                url,
                params=params,
                timeout=30
            )

            response.raise_for_status()

            data = response.json()

            if data:

                all_rows.extend(data)

                print(
                    "Downloading:",
                    datetime.fromtimestamp(
                        current_start,
                        timezone.utc
                    ).strftime("%Y-%m-%d"),
                    "to",
                    datetime.fromtimestamp(
                        current_end,
                        timezone.utc
                    ).strftime("%Y-%m-%d")
                )

        except Exception as e:

            print("Download error:", e)

            time.sleep(3)

            continue

        current_start = current_end

        time.sleep(0.25)

    if not all_rows:
        raise RuntimeError("No market data downloaded.")

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

    df = df.sort_values("timestamp")

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

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna().reset_index(drop=True)

    print()
    print("Total candles downloaded:", len(df))

    print(
        "First candle:",
        df.iloc[0]["timestamp"]
    )

    print(
        "Last candle: ",
        df.iloc[-1]["timestamp"]
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):

    df = df.copy()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

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
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = true_range.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # ATR percentage
    # --------------------------------------------------------

    df["atr_pct"] = (
        df["atr"] /
        df["close"]
    )

    # --------------------------------------------------------
    # ATR volatility baseline
    # --------------------------------------------------------

    df["atr_median"] = (
        df["atr_pct"]
        .rolling(96)
        .median()
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    df["volume_ma"] = (
        df["volume"]
        .rolling(20)
        .mean()
    )

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_ma"]
    )

    # --------------------------------------------------------
    # Recent highs / lows
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Candle properties
    # --------------------------------------------------------

    df["body"] = (
        df["close"] -
        df["open"]
    ).abs()

    df["range"] = (
        df["high"] -
        df["low"]
    )

    df["upper_wick"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_wick"] = (
        df[["open", "close"]].min(axis=1) -
        df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        df["range"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Returns
    # --------------------------------------------------------

    df["return_3"] = (
        df["close"] /
        df["close"].shift(3) - 1
    )

    df["return_6"] = (
        df["close"] /
        df["close"].shift(6) - 1
    )

    # --------------------------------------------------------
    # Trend strength
    #
    # This is a simplified ADX-style calculation.
    # --------------------------------------------------------

    up_move = (
        df["high"] -
        df["high"].shift(1)
    )

    down_move = (
        df["low"].shift(1) -
        df["low"]
    )

    plus_dm = np.where(
        (up_move > down_move) &
        (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) &
        (down_move > 0),
        down_move,
        0
    )

    atr_safe = df["atr"].replace(
        0,
        np.nan
    )

    plus_di = (
        pd.Series(
            plus_dm,
            index=df.index
        ).ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / atr_safe
    ) * 100

    minus_di = (
        pd.Series(
            minus_dm,
            index=df.index
        ).ewm(
            alpha=1 / 14,
            adjust=False
        ).mean()
        / atr_safe
    ) * 100

    dx = (
        (plus_di - minus_di).abs()
        /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    ) * 100

    df["adx"] = dx.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    return df


# ============================================================
# MARKET REGIME
# ============================================================

def get_regime(row):

    close = row["close"]

    ema9 = row["ema9"]
    ema21 = row["ema21"]
    ema50 = row["ema50"]

    adx = row["adx"]

    atr_pct = row["atr_pct"]
    atr_median = row["atr_median"]

    if any(
        pd.isna(x)
        for x in [
            close,
            ema9,
            ema21,
            ema50,
            adx,
            atr_pct,
            atr_median
        ]
    ):
        return "UNKNOWN"

    # High volatility
    if atr_pct > atr_median * 1.8:
        return "HIGH_VOL"

    # Strong trend
    if (
        ema9 > ema21 > ema50
        and adx >= 22
        and close > ema21
    ):
        return "STRONG_UP"

    if (
        ema9 < ema21 < ema50
        and adx >= 22
        and close < ema21
    ):
        return "STRONG_DOWN"

    # Normal trend
    if (
        ema9 > ema21 > ema50
        and adx >= 16
    ):
        return "NORMAL_UP"

    if (
        ema9 < ema21 < ema50
        and adx >= 16
    ):
        return "NORMAL_DOWN"

    # Everything else
    return "CHOPPY"


# ============================================================
# PULLBACK DETECTION
# ============================================================

def get_pullback(row):

    close = row["close"]
    ema21 = row["ema21"]
    ema50 = row["ema50"]
    atr = row["atr"]

    if any(
        pd.isna(x)
        for x in [
            close,
            ema21,
            ema50,
            atr
        ]
    ):
        return "NONE"

    if atr <= 0:
        return "NONE"

    distance_21 = abs(
        close - ema21
    ) / atr

    distance_50 = abs(
        close - ema50
    ) / atr

    # Price reasonably close to EMA21
    if distance_21 <= 0.8:

        if close >= ema21:
            return "PULLBACK_21"

        return "PULLBACK_21_BELOW"

    # Price near EMA50
    if distance_50 <= 1.0:

        if close >= ema50:
            return "PULLBACK_50"

        return "PULLBACK_50_BELOW"

    return "NONE"


# ============================================================
# CANDLE CONFIRMATION
# ============================================================

def bullish_confirmation(row):

    if row["range"] <= 0:
        return False

    # Bullish candle
    bullish_body = (
        row["close"] > row["open"]
    )

    # Body should have reasonable size
    strong_body = (
        row["body_ratio"] >= 0.45
    )

    # Close should be toward candle high
    close_near_high = (
        (
            row["high"] -
            row["close"]
        )
        <= row["range"] * 0.30
    )

    # Bullish rejection from lower area
    bullish_rejection = (
        row["lower_wick"] >
        row["body"] * 0.8
    )

    return (
        bullish_body and
        strong_body and
        close_near_high
    ) or bullish_rejection


def bearish_confirmation(row):

    if row["range"] <= 0:
        return False

    bearish_body = (
        row["close"] < row["open"]
    )

    strong_body = (
        row["body_ratio"] >= 0.45
    )

    close_near_low = (
        (
            row["close"] -
            row["low"]
        )
        <= row["range"] * 0.30
    )

    bearish_rejection = (
        row["upper_wick"] >
        row["body"] * 0.8
    )

    return (
        bearish_body and
        strong_body and
        close_near_low
    ) or bearish_rejection


# ============================================================
# SIGNAL
# ============================================================

def get_signal(df, i):

    row = df.iloc[i]

    regime = get_regime(row)

    pullback = get_pullback(row)

    # --------------------------------------------------------
    # Never trade these
    # --------------------------------------------------------

    if regime in [
        "UNKNOWN",
        "CHOPPY",
        "HIGH_VOL"
    ]:
        return "WAIT"

    # --------------------------------------------------------
    # Avoid overextended entries
    # --------------------------------------------------------

    distance_from_ema21 = (
        abs(
            row["close"] -
            row["ema21"]
        )
        /
        row["atr"]
    )

    if distance_from_ema21 > 1.5:
        return "WAIT"

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if regime in [
        "STRONG_UP",
        "NORMAL_UP"
    ]:

        score = 0

        # Trend alignment
        if (
            row["ema9"] >
            row["ema21"] >
            row["ema50"]
        ):
            score += 2

        # Price above EMA21
        if row["close"] > row["ema21"]:
            score += 1

        # MACD
        if row["macd"] > row["macd_signal"]:
            score += 2

        # RSI
        if 50 <= row["rsi"] <= 68:
            score += 1

        # Volume
        if row["volume_ratio"] >= 1.10:
            score += 1

        # Pullback
        if pullback in [
            "PULLBACK_21",
            "PULLBACK_50"
        ]:
            score += 2

        # Candle confirmation
        if bullish_confirmation(row):
            score += 2

        # Breakout
        if (
            row["close"] >
            row["recent_high"]
        ):
            score += 1

        if score >= 7:
            return "BUY"

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    if regime in [
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ]:

        score = 0

        # Trend alignment
        if (
            row["ema9"] <
            row["ema21"] <
            row["ema50"]
        ):
            score += 2

        # Price below EMA21
        if row["close"] < row["ema21"]:
            score += 1

        # MACD
        if row["macd"] < row["macd_signal"]:
            score += 2

        # RSI
        if 32 <= row["rsi"] <= 50:
            score += 1

        # Volume
        if row["volume_ratio"] >= 1.10:
            score += 1

        # Pullback
        if pullback in [
            "PULLBACK_21_BELOW",
            "PULLBACK_50_BELOW"
        ]:
            score += 2

        # Candle confirmation
        if bearish_confirmation(row):
            score += 2

        # Breakdown
        if (
            row["close"] <
            row["recent_low"]
        ):
            score += 1

        if score >= 7:
            return "SELL"

    return "WAIT"


# ============================================================
# PATTERN
# ============================================================

def get_pattern(df, i, signal):

    row = df.iloc[i]

    regime = get_regime(row)

    pullback = get_pullback(row)

    if row["rsi"] >= 60:
        rsi_state = "RSI_HIGH"

    elif row["rsi"] <= 40:
        rsi_state = "RSI_LOW"

    else:
        rsi_state = "RSI_MID"

    if row["macd"] > row["macd_signal"]:
        macd_state = "MACD_UP"
    else:
        macd_state = "MACD_DOWN"

    if row["volume_ratio"] >= 1.10:
        volume_state = "VOL_STRONG"

    elif row["volume_ratio"] < 0.80:
        volume_state = "VOL_WEAK"

    else:
        volume_state = "VOL_NORMAL"

    if signal == "BUY":

        candle = (
            "BULL_CONFIRM"
            if bullish_confirmation(row)
            else "NO_CONFIRM"
        )

    else:

        candle = (
            "BEAR_CONFIRM"
            if bearish_confirmation(row)
            else "NO_CONFIRM"
        )

    return "|".join([
        regime,
        pullback,
        rsi_state,
        macd_state,
        volume_state,
        candle
    ])


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    signal
):

    entry = df.iloc[
        entry_index
    ]["close"]

    atr = df.iloc[
        entry_index
    ]["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if signal == "BUY":

        tp_price = (
            entry +
            TP_ATR * atr
        )

        sl_price = (
            entry -
            SL_ATR * atr
        )

    else:

        tp_price = (
            entry -
            TP_ATR * atr
        )

        sl_price = (
            entry +
            SL_ATR * atr
        )

    end_index = min(
        entry_index +
        MAX_HOLD_CANDLES,
        len(df) - 1
    )

    for j in range(
        entry_index + 1,
        end_index + 1
    ):

        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        # ----------------------------------------------------
        # Conservative assumption:
        # if both TP and SL occur in same candle,
        # assume SL happened first.
        # ----------------------------------------------------

        if signal == "BUY":

            if (
                low <= sl_price
                and high >= tp_price
            ):
                return {
                    "result": "SL",
                    "r": -1.0 - ROUND_TRIP_COST / (
                        SL_ATR * atr / entry
                    )
                }

            if low <= sl_price:

                gross_r = -1.0

                net_return = (
                    gross_r *
                    (SL_ATR * atr / entry)
                    -
                    ROUND_TRIP_COST
                )

                net_r = (
                    net_return /
                    (SL_ATR * atr / entry)
                )

                return {
                    "result": "SL",
                    "r": net_r
                }

            if high >= tp_price:

                gross_return = (
                    TP_ATR * atr / entry
                )

                net_return = (
                    gross_return -
                    ROUND_TRIP_COST
                )

                net_r = (
                    net_return /
                    (SL_ATR * atr / entry)
                )

                return {
                    "result": "TP",
                    "r": net_r
                }

        else:

            if (
                high >= sl_price
                and low <= tp_price
            ):
                return {
                    "result": "SL",
                    "r": -1.0 - ROUND_TRIP_COST / (
                        SL_ATR * atr / entry
                    )
                }

            if high >= sl_price:

                gross_r = -1.0

                net_return = (
                    gross_r *
                    (SL_ATR * atr / entry)
                    -
                    ROUND_TRIP_COST
                )

                net_r = (
                    net_return /
                    (SL_ATR * atr / entry)
                )

                return {
                    "result": "SL",
                    "r": net_r
                }

            if low <= tp_price:

                gross_return = (
                    TP_ATR * atr / entry
                )

                net_return = (
                    gross_return -
                    ROUND_TRIP_COST
                )

                net_r = (
                    net_return /
                    (SL_ATR * atr / entry)
                )

                return {
                    "result": "TP",
                    "r": net_r
                }

    # --------------------------------------------------------
    # TIMEOUT
    # --------------------------------------------------------

    exit_price = df.iloc[
        end_index
    ]["close"]

    if signal == "BUY":

        gross_return = (
            exit_price / entry
        ) - 1

    else:

        gross_return = (
            entry / exit_price
        ) - 1

    net_return = (
        gross_return -
        ROUND_TRIP_COST
    )

    risk_return = (
        SL_ATR * atr / entry
    )

    net_r = (
        net_return /
        risk_return
    )

    return {
        "result": "TIMEOUT",
        "r": net_r
    }


# ============================================================
# COLLECT TRADES
# ============================================================

def collect_trades(
    df,
    start_index,
    end_index,
    allowed_patterns=None
):

    trades = []

    last_trade_index = -999999

    for i in range(
        start_index,
        end_index
    ):

        if i < 120:
            continue

        if (
            i -
            last_trade_index
            < COOLDOWN_CANDLES
        ):
            continue

        signal = get_signal(
            df,
            i
        )

        if signal not in [
            "BUY",
            "SELL"
        ]:
            continue

        pattern = get_pattern(
            df,
            i,
            signal
        )

        # ----------------------------------------------------
        # If training selected patterns are supplied,
        # only use those.
        # ----------------------------------------------------

        if allowed_patterns is not None:

            if pattern not in allowed_patterns.get(
                signal,
                set()
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
        trade["signal"] = signal
        trade["pattern"] = pattern
        trade["timestamp"] = df.iloc[i]["timestamp"]

        trades.append(trade)

        last_trade_index = i

    return trades


# ============================================================
# STATISTICS
# ============================================================

def statistics(trades):

    if not trades:

        return {
            "count": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0,
            "total_r": 0,
            "avg_r": 0,
            "max_dd": 0
        }

    results = [
        t["result"]
        for t in trades
    ]

    r_values = [
        t["r"]
        for t in trades
    ]

    equity = 0
    peak = 0
    max_dd = 0

    for r in r_values:

        equity += r

        peak = max(
            peak,
            equity
        )

        drawdown = (
            equity -
            peak
        )

        max_dd = min(
            max_dd,
            drawdown
        )

    return {
        "count": len(trades),
        "tp": results.count("TP"),
        "sl": results.count("SL"),
        "timeout": results.count("TIMEOUT"),
        "total_r": sum(r_values),
        "avg_r": np.mean(r_values),
        "max_dd": max_dd
    }


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_stats(title, trades):

    s = statistics(trades)

    total = s["count"]

    if total == 0:

        print()
        print(title)
        print("No trades.")
        return

    print()
    print(title)

    print(
        f"Trades: {total}"
    )

    print(
        f"TP: {s['tp']} "
        f"({s['tp'] / total * 100:.1f}%)"
    )

    print(
        f"SL: {s['sl']} "
        f"({s['sl'] / total * 100:.1f}%)"
    )

    print(
        f"Timeout: {s['timeout']} "
        f"({s['timeout'] / total * 100:.1f}%)"
    )

    print(
        f"Total R: {s['total_r']:.2f}"
    )

    print(
        f"Average R: {s['avg_r']:.3f}"
    )

    print(
        f"Max Drawdown: {s['max_dd']:.2f}R"
    )


# ============================================================
# SELECT PATTERNS FROM TRAINING
# ============================================================

def select_patterns(
    trades
):

    pattern_stats = {}

    for trade in trades:

        key = (
            trade["signal"],
            trade["pattern"]
        )

        if key not in pattern_stats:

            pattern_stats[key] = []

        pattern_stats[key].append(
            trade["r"]
        )

    selected = {
        "BUY": set(),
        "SELL": set()
    }

    print()
    print("========== TRAINING PATTERNS ==========")

    for signal in [
        "BUY",
        "SELL"
    ]:

        candidates = []

        for (
            side,
            pattern
        ), values in pattern_stats.items():

            if side != signal:
                continue

            if len(values) < MIN_PATTERN_TRADES:
                continue

            avg_r = np.mean(values)

            total_r = np.sum(values)

            # IMPORTANT:
            # Only positive training expectancy.
            if avg_r > 0:

                candidates.append(
                    (
                        avg_r,
                        len(values),
                        total_r,
                        pattern
                    )
                )

        candidates.sort(
            reverse=True
        )

        print()
        print(
            signal,
            "positive patterns:",
            len(candidates)
        )

        for item in candidates[
            :MAX_PATTERNS_PER_SIDE
        ]:

            avg_r, count, total_r, pattern = item

            selected[signal].add(
                pattern
            )

            print(
                f"{count} trades | "
                f"Avg R {avg_r:.3f} | "
                f"Total R {total_r:.2f} | "
                f"{pattern}"
            )

    return selected


# ============================================================
# MARKET REGIME REPORT
# ============================================================

def regime_report(
    trades,
    title
):

    print()
    print(title)

    if not trades:

        print("No trades.")
        return

    groups = {}

    for trade in trades:

        regime = trade[
            "pattern"
        ].split("|")[0]

        if regime not in groups:
            groups[regime] = []

        groups[regime].append(
            trade
        )

    for regime, group in sorted(
        groups.items()
    ):

        s = statistics(group)

        print(
            f"{regime}: "
            f"{s['count']} trades | "
            f"TP {s['tp']/s['count']*100:.1f}% | "
            f"SL {s['sl']/s['count']*100:.1f}% | "
            f"Avg R {s['avg_r']:.3f}"
        )


# ============================================================
# MAIN WALK-FORWARD TEST
# ============================================================

def main():

    df = download_candles()

    print()
    print("📊 Calculating indicators...")

    df = calculate_indicators(
        df
    )

    print("✅ Indicators calculated.")

    total_days = (
        df["timestamp"].iloc[-1] -
        df["timestamp"].iloc[0]
    ).total_seconds() / 86400

    print()
    print(
        f"Actual data span: "
        f"{total_days:.1f} days"
    )

    print()
    print("========== SETTINGS ==========")

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
        f"Cooldown: "
        f"{COOLDOWN_CANDLES * 15} minutes"
    )

    print(
        f"Round-trip estimated cost: "
        f"{ROUND_TRIP_COST * 100:.2f}%"
    )

    print(
        f"Training: "
        f"{TRAIN_DAYS} days"
    )

    print(
        f"Validation: "
        f"{VALIDATION_DAYS} days"
    )

    # --------------------------------------------------------
    # Walk-forward windows
    # --------------------------------------------------------

    start_time = (
        df["timestamp"].iloc[0]
    )

    final_time = (
        df["timestamp"].iloc[-1]
    )

    window_start = start_time

    all_validation_trades = []

    window_number = 1

    while True:

        train_start_time = (
            window_start
        )

        train_end_time = (
            train_start_time +
            timedelta(days=TRAIN_DAYS)
        )

        validation_start_time = (
            train_end_time
        )

        validation_end_time = (
            validation_start_time +
            timedelta(days=VALIDATION_DAYS)
        )

        if validation_end_time > final_time:
            break

        train_start = df.index[
            df["timestamp"] >=
            train_start_time
        ][0]

        train_end = df.index[
            df["timestamp"] >=
            train_end_time
        ][0]

        validation_start = df.index[
            df["timestamp"] >=
            validation_start_time
        ][0]

        validation_end_list = df.index[
            df["timestamp"] <=
            validation_end_time
        ]

        if len(validation_end_list) == 0:
            break

        validation_end = (
            validation_end_list[-1] + 1
        )

        print()
        print(
            "=========================================="
        )

        print(
            f"WINDOW {window_number}"
        )

        print(
            "=========================================="
        )

        print(
            "Training:",
            train_start_time,
            "→",
            train_end_time
        )

        print(
            "Validation:",
            validation_start_time,
            "→",
            validation_end_time
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        training_trades = collect_trades(
            df,
            train_start,
            train_end
        )

        print_stats(
            "TRAINING RESULT",
            training_trades
        )

        regime_report(
            training_trades,
            "TRAINING REGIME"
        )

        # ----------------------------------------------------
        # Select positive patterns
        # ----------------------------------------------------

        selected_patterns = select_patterns(
            training_trades
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_trades = collect_trades(
            df,
            validation_start,
            validation_end,
            selected_patterns
        )

        print_stats(
            "VALIDATION RESULT",
            validation_trades
        )

        regime_report(
            validation_trades,
            "VALIDATION REGIME"
        )

        all_validation_trades.extend(
            validation_trades
        )

        window_number += 1

        # Move forward by 30 days
        window_start = (
            window_start +
            timedelta(days=VALIDATION_DAYS)
        )


    # ========================================================
    # OVERALL VALIDATION
    # ========================================================

    print()
    print(
        "=========================================="
    )

    print(
        "========== WALK-FORWARD OVERALL =========="
    )

    print(
        "=========================================="
    )

    print_stats(
        "ALL VALIDATION TRADES",
        all_validation_trades
    )

    # --------------------------------------------------------
    # Direction breakdown
    # --------------------------------------------------------

    buy_trades = [
        t for t in all_validation_trades
        if t["signal"] == "BUY"
    ]

    sell_trades = [
        t for t in all_validation_trades
        if t["signal"] == "SELL"
    ]

    print_stats(
        "ALL VALIDATION BUY",
        buy_trades
    )

    print_stats(
        "ALL VALIDATION SELL",
        sell_trades
    )

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    s = statistics(
        all_validation_trades
    )

    print()
    print(
        "=========================================="
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=========================================="
    )

    if s["count"] == 0:

        print(
            "❌ No validation trades."
        )

        print(
            "The filters may be too strict."
        )

    elif s["total_r"] > 0:

        print(
            "📈 Validation Total R is positive."
        )

        print(
            "This does NOT prove future profitability."
        )

        print(
            "More out-of-sample testing is required."
        )

    else:

        print(
            "📉 Validation Total R is negative."
        )

        print(
            "Current rules do not show a positive "
            "out-of-sample edge."
        )

        print(
            "Do NOT put these rules into the live bot yet."
        )

    print()
    print("🏁 Backtest finished.")


if __name__ == "__main__":
    main()

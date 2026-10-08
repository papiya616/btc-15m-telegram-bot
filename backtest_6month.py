import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M REGIME REVERSAL / MEAN-REVERSION BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
DAYS = 183                 # ~6 months

TRAIN_DAYS = 120
VALIDATION_DAYS = DAYS - TRAIN_DAYS

# Trading cost assumption
COST = 0.0014              # 0.14%

# Maximum holding time
MAX_HOLD_CANDLES = 12      # 3 hours


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_candles():

    print("\n🚀 BTC 6-Month Regime Reversal Backtest Started!\n")

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    current = start_time

    while current < end_time:

        chunk_end = min(
            current + timedelta(days=3),
            end_time
        )

        print(
            f"Downloading: "
            f"{current.strftime('%Y-%m-%d')} "
            f"to "
            f"{chunk_end.strftime('%Y-%m-%d')}"
        )

        url = (
            f"https://api.exchange.coinbase.com/"
            f"products/{PRODUCT}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": current.isoformat(),
            "end": chunk_end.isoformat()
        }

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

        current = chunk_end

        time.sleep(0.3)

    if not all_rows:

        raise RuntimeError(
            "❌ No BTC data downloaded."
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

    df = df.drop_duplicates(
        subset=["time"]
    )

    df = df.sort_values("time")

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

    df = df.dropna()

    df = df.reset_index(drop=True)

    print(
        f"\n✅ Total candles: {len(df)}"
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    d = df.copy()

    # EMA
    d["ema20"] = d["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    d["ema50"] = d["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    d["ema100"] = d["close"].ewm(
        span=100,
        adjust=False
    ).mean()

    # RSI
    delta = d["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    d["rsi"] = 100 - (
        100 / (1 + rs)
    )

    # MACD
    ema12 = d["close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = d["close"].ewm(
        span=26,
        adjust=False
    ).mean()

    d["macd"] = ema12 - ema26

    d["macd_signal"] = d["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = (
        d["macd"] -
        d["macd_signal"]
    )

    # ATR
    prev_close = d["close"].shift(1)

    tr1 = d["high"] - d["low"]

    tr2 = (
        d["high"] -
        prev_close
    ).abs()

    tr3 = (
        d["low"] -
        prev_close
    ).abs()

    d["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    d["atr"] = d["tr"].rolling(
        14
    ).mean()

    # Volume average
    d["volume_ma"] = d["volume"].rolling(
        20
    ).mean()

    d["volume_ratio"] = (
        d["volume"] /
        d["volume_ma"]
    )

    # Candle structure
    d["body"] = (
        d["close"] -
        d["open"]
    )

    d["body_abs"] = d["body"].abs()

    d["candle_range"] = (
        d["high"] -
        d["low"]
    )

    d["body_ratio"] = (
        d["body_abs"] /
        d["candle_range"].replace(
            0,
            np.nan
        )
    )

    # Distance from EMA20
    d["ema20_distance"] = (
        d["close"] -
        d["ema20"]
    )

    d["ema20_distance_atr"] = (
        d["ema20_distance"] /
        d["atr"]
    )

    # EMA trend strength
    d["ema_gap"] = (
        d["ema20"] -
        d["ema50"]
    )

    d["ema_gap_atr"] = (
        d["ema_gap"] /
        d["atr"]
    )

    # Recent highs/lows
    d["recent_high"] = (
        d["high"]
        .rolling(20)
        .max()
        .shift(1)
    )

    d["recent_low"] = (
        d["low"]
        .rolling(20)
        .min()
        .shift(1)
    )

    # 1H equivalent trend
    # 4 x 15M = 1 hour
    d["close_1h"] = d["close"].rolling(
        4
    ).mean()

    d["ema20_1h"] = d["ema20"].rolling(
        4
    ).mean()

    # Previous candle
    d["prev_close"] = d["close"].shift(1)
    d["prev_open"] = d["open"].shift(1)
    d["prev_high"] = d["high"].shift(1)
    d["prev_low"] = d["low"].shift(1)

    # Reversal candle detection
    d["bullish_candle"] = (
        d["close"] > d["open"]
    )

    d["bearish_candle"] = (
        d["close"] < d["open"]
    )

    # Lower wick
    d["lower_wick"] = (
        np.minimum(
            d["open"],
            d["close"]
        ) -
        d["low"]
    )

    # Upper wick
    d["upper_wick"] = (
        d["high"] -
        np.maximum(
            d["open"],
            d["close"]
        )
    )

    d["lower_wick_ratio"] = (
        d["lower_wick"] /
        d["candle_range"].replace(
            0,
            np.nan
        )
    )

    d["upper_wick_ratio"] = (
        d["upper_wick"] /
        d["candle_range"].replace(
            0,
            np.nan
        )
    )

    return d


# ============================================================
# MARKET REGIME
# ============================================================

def get_regime(row):

    gap = row["ema_gap_atr"]

    if pd.isna(gap):
        return "UNKNOWN"

    if gap >= 1.0:
        return "STRONG_UP"

    if gap >= 0.30:
        return "NORMAL_UP"

    if gap <= -1.0:
        return "STRONG_DOWN"

    if gap <= -0.30:
        return "NORMAL_DOWN"

    return "RANGE"


# ============================================================
# REVERSAL SIGNAL
# ============================================================

def get_signal(row):

    regime = get_regime(row)

    if regime == "UNKNOWN":
        return None

    score_buy = 0
    score_sell = 0

    # --------------------------------------------------------
    # BUY REVERSAL
    # --------------------------------------------------------

    if regime == "STRONG_DOWN":

        # Oversold
        if row["rsi"] <= 30:
            score_buy += 2

        elif row["rsi"] <= 35:
            score_buy += 1

        # Strong lower wick
        if row["lower_wick_ratio"] >= 0.35:
            score_buy += 2

        elif row["lower_wick_ratio"] >= 0.25:
            score_buy += 1

        # Bullish reversal candle
        if row["bullish_candle"]:
            score_buy += 1

        # MACD turning upward
        if row["macd_hist"] >
                row["prev_macd_hist"] \
                if "prev_macd_hist" in row else False:

            score_buy += 1

        # Price stretched below EMA20
        if row["ema20_distance_atr"] <= -1.5:
            score_buy += 2

        elif row["ema20_distance_atr"] <= -1.0:
            score_buy += 1

        # Volume confirmation
        if row["volume_ratio"] >= 1.5:
            score_buy += 1

        # Strong bullish body
        if (
            row["body_ratio"] >= 0.45
            and row["bullish_candle"]
        ):
            score_buy += 1

        if score_buy >= 5:
            return "BUY"

    # --------------------------------------------------------
    # SELL REVERSAL
    # --------------------------------------------------------

    if regime == "STRONG_UP":

        # Overbought
        if row["rsi"] >= 70:
            score_sell += 2

        elif row["rsi"] >= 65:
            score_sell += 1

        # Strong upper wick
        if row["upper_wick_ratio"] >= 0.35:
            score_sell += 2

        elif row["upper_wick_ratio"] >= 0.25:
            score_sell += 1

        # Bearish reversal candle
        if row["bearish_candle"]:
            score_sell += 1

        # MACD turning downward
        if "prev_macd_hist" in row:

            if row["macd_hist"] < row["prev_macd_hist"]:
                score_sell += 1

        # Price stretched above EMA20
        if row["ema20_distance_atr"] >= 1.5:
            score_sell += 2

        elif row["ema20_distance_atr"] >= 1.0:
            score_sell += 1

        # Volume
        if row["volume_ratio"] >= 1.5:
            score_sell += 1

        # Strong bearish body
        if (
            row["body_ratio"] >= 0.45
            and row["bearish_candle"]
        ):
            score_sell += 1

        if score_sell >= 5:
            return "SELL"

    return None


# ============================================================
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    df,
    entry_index,
    direction,
    tp_atr,
    sl_atr
):

    entry = df.iloc[entry_index]

    entry_price = entry["close"]

    atr = entry["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp = entry_price + (
            atr * tp_atr
        )

        sl = entry_price - (
            atr * sl_atr
        )

    else:

        tp = entry_price - (
            atr * tp_atr
        )

        sl = entry_price + (
            atr * sl_atr
        )

    max_index = min(
        entry_index +
        MAX_HOLD_CANDLES,
        len(df) - 1
    )

    for j in range(
        entry_index + 1,
        max_index + 1
    ):

        candle = df.iloc[j]

        high = candle["high"]
        low = candle["low"]

        if direction == "BUY":

            hit_tp = high >= tp
            hit_sl = low <= sl

            # Conservative assumption:
            # if both are hit in same candle,
            # assume SL happened first.
            if hit_sl and hit_tp:
                return {
                    "result": "SL",
                    "return": -sl_atr,
                    "r": -1.0,
                    "exit_index": j
                }

            if hit_sl:

                return {
                    "result": "SL",
                    "return": -sl_atr,
                    "r": -1.0,
                    "exit_index": j
                }

            if hit_tp:

                return {
                    "result": "TP",
                    "return": tp_atr,
                    "r": tp_atr / sl_atr,
                    "exit_index": j
                }

        else:

            hit_tp = low <= tp
            hit_sl = high >= sl

            if hit_sl and hit_tp:

                return {
                    "result": "SL",
                    "return": -sl_atr,
                    "r": -1.0,
                    "exit_index": j
                }

            if hit_sl:

                return {
                    "result": "SL",
                    "return": -sl_atr,
                    "r": -1.0,
                    "exit_index": j
                }

            if hit_tp:

                return {
                    "result": "TP",
                    "return": tp_atr,
                    "r": tp_atr / sl_atr,
                    "exit_index": j
                }

    # Timeout
    final_price = df.iloc[
        max_index
    ]["close"]

    if direction == "BUY":

        raw_return = (
            final_price -
            entry_price
        ) / entry_price

    else:

        raw_return = (
            entry_price -
            final_price
        ) / entry_price

    return {
        "result": "TIMEOUT",
        "return": raw_return,
        "r": raw_return / (
            sl_atr *
            atr /
            entry_price
        ),
        "exit_index": max_index
    }


# ============================================================
# BACKTEST
# ============================================================

def run_backtest(
    df,
    start_index,
    end_index,
    tp_atr,
    sl_atr
):

    trades = []

    cooldown_until = -1

    for i in range(
        start_index,
        end_index
    ):

        if i <= cooldown_until:
            continue

        row = df.iloc[i]

        signal = get_signal(row)

        if signal is None:
            continue

        trade = simulate_trade(
            df,
            i,
            signal,
            tp_atr,
            sl_atr
        )

        if trade is None:
            continue

        # Cost
        entry_price = row["close"]

        if signal == "BUY":

            net_return = (
                trade["return"]
                *
                row["atr"]
                /
                entry_price
            )

        else:

            net_return = (
                trade["return"]
                *
                row["atr"]
                /
                entry_price
            )

        net_return -= COST

        trade["direction"] = signal
        trade["net_return"] = net_return

        trades.append(trade)

        # Prevent immediate repeated entries
        cooldown_until = (
            trade["exit_index"] + 2
        )

    return trades


# ============================================================
# RESULTS
# ============================================================

def print_results(
    name,
    trades
):

    print("\n" + "=" * 65)
    print(name)
    print("=" * 65)

    if not trades:

        print("❌ No trades")
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

    win_rate = (
        tp / total * 100
    )

    avg_return = (
        df["net_return"].mean()
        * 100
    )

    total_return = (
        df["net_return"].sum()
        * 100
    )

    avg_r = df["r"].mean()

    total_r = df["r"].sum()

    print(
        f"Trades       : {total}"
    )

    print(
        f"TP           : {tp} "
        f"({tp / total * 100:.1f}%)"
    )

    print(
        f"SL           : {sl} "
        f"({sl / total * 100:.1f}%)"
    )

    print(
        f"Timeout      : {timeout} "
        f"({timeout / total * 100:.1f}%)"
    )

    print(
        f"Avg Return   : {avg_return:.3f}%"
    )

    print(
        f"Total Return : {total_return:.2f}%"
    )

    print(
        f"Avg R        : {avg_r:.3f}"
    )

    print(
        f"Total R      : {total_r:.2f}"
    )

    print("\nDirection:")

    for direction in [
        "BUY",
        "SELL"
    ]:

        x = df[
            df["direction"] == direction
        ]

        if len(x) == 0:
            continue

        print(
            f"{direction}: "
            f"{len(x)} trades | "
            f"Avg {x['net_return'].mean()*100:.3f}% | "
            f"Total {x['net_return'].sum()*100:.2f}%"
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_candles()

    print("\n📊 Calculating indicators...")

    df = add_indicators(df)

    # Previous MACD histogram
    df["prev_macd_hist"] = (
        df["macd_hist"].shift(1)
    )

    df = df.dropna().reset_index(
        drop=True
    )

    print(
        f"Final usable candles: {len(df)}"
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    total_days = (
        df["time"].iloc[-1] -
        df["time"].iloc[0]
    ).total_seconds() / 86400

    train_cutoff = (
        df["time"].iloc[0] +
        timedelta(days=TRAIN_DAYS)
    )

    train_end = (
        df["time"] < train_cutoff
    ).sum()

    validation_start = train_end

    print("\n" + "=" * 65)

    print(
        "TRAINING:"
    )

    print(
        df["time"].iloc[0],
        "→",
        df["time"].iloc[
            train_end - 1
        ]
    )

    print(
        "\nVALIDATION:"
    )

    print(
        df["time"].iloc[
            validation_start
        ],
        "→",
        df["time"].iloc[-1]
    )

    # --------------------------------------------------------
    # TEST MULTIPLE TP/SL
    # --------------------------------------------------------

    configs = [

        (1.00, 0.75),

        (1.25, 0.75),

        (1.50, 1.00),

        (1.50, 1.25),

        (2.00, 1.00),

        (2.00, 1.25),

        (2.50, 1.25)

    ]

    best_config = None
    best_avg_r = -999

    print(
        "\n\n🔬 TRAINING CONFIGURATION TEST"
    )

    for tp, sl in configs:

        trades = run_backtest(
            df,
            0,
            train_end,
            tp,
            sl
        )

        result = print_results(
            f"TRAIN TP {tp:.2f} ATR / "
            f"SL {sl:.2f} ATR",
            trades
        )

        if result is not None:

            avg_r = result[
                "r"
            ].mean()

            if avg_r > best_avg_r:

                best_avg_r = avg_r

                best_config = (
                    tp,
                    sl
                )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    print(
        "\n\n" + "=" * 65
    )

    print(
        "🎯 VALIDATION USING BEST TRAINING CONFIG"
    )

    print(
        "=" * 65
    )

    if best_config is None:

        print(
            "❌ No valid training configuration."
        )

        return

    tp, sl = best_config

    print(
        f"\nSelected TP/SL: "
        f"{tp:.2f} ATR / {sl:.2f} ATR"
    )

    validation_trades = run_backtest(
        df,
        validation_start,
        len(df),
        tp,
        sl
    )

    validation_df = print_results(
        "VALIDATION",
        validation_trades
    )

    # --------------------------------------------------------
    # SAVE VALIDATION CSV
    # --------------------------------------------------------

    if validation_df is not None:

        validation_df.to_csv(
            "btc_reversal_validation.csv",
            index=False
        )

        print(
            "\n💾 Saved:"
        )

        print(
            "btc_reversal_validation.csv"
        )

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    print(
        "\n\n" + "=" * 65
    )

    print(
        "🏁 FINAL VERDICT"
    )

    print(
        "=" * 65
    )

    if validation_df is None:

        print(
            "❌ No validation trades."
        )

        return

    total_return = (
        validation_df[
            "net_return"
        ].sum()
        * 100
    )

    avg_r = (
        validation_df[
            "r"
        ].mean()
    )

    if (
        total_return > 0
        and avg_r > 0
    ):

        print(
            "🟢 PROMISING: "
            "Validation is positive."
        )

        print(
            "⚠️ Still needs more out-of-sample testing."
        )

    else:

        print(
            "🔴 NOT PROFITABLE:"
        )

        print(
            "Validation does not show a "
            "robust positive edge."
        )

        print(
            "Do NOT put this strategy into "
            "the live Telegram bot yet."
        )


if __name__ == "__main__":
    main()

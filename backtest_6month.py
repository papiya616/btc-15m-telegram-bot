import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# =========================================================
# BTC 15M BREAKOUT + RETEST BACKTEST
# =========================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900  # 15 minutes

TOTAL_DAYS = 183
TRAIN_DAYS = 120
VALIDATION_DAYS = 63

CHUNK_DAYS = 3

LOOKBACK = 20
ATR_PERIOD = 14

# Breakout must exceed level by this ATR amount
BREAKOUT_ATR = 0.10

# Retest can happen within these candles
RETEST_MIN = 1
RETEST_MAX = 4

# Retest tolerance around breakout level
RETEST_ATR = 0.20

# Maximum trade duration
MAX_HOLD_CANDLES = 12   # 3 hours

# Minimum time between trades
COOLDOWN_CANDLES = 12   # 3 hours

# Estimated round-trip cost
COST = 0.0014  # 0.14%

MIN_TRAIN_TRADES = 30
MIN_VALIDATION_TRADES = 20


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_chunk(start_dt, end_dt):

    url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

    params = {
        "granularity": GRANULARITY,
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat()
    }

    r = requests.get(
        url,
        params=params,
        timeout=30
    )

    r.raise_for_status()

    data = r.json()

    if not isinstance(data, list):
        return []

    return data


def download_data():

    print("📥 Downloading BTC 15M data...")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=TOTAL_DAYS)

    chunks = []

    current = start

    while current < end:

        chunk_end = min(
            current + timedelta(days=CHUNK_DAYS),
            end
        )

        print(
            f"Downloading: "
            f"{current.strftime('%Y-%m-%d')} "
            f"to "
            f"{chunk_end.strftime('%Y-%m-%d')}"
        )

        try:

            data = download_chunk(
                current,
                chunk_end
            )

            if data:
                chunks.extend(data)

        except Exception as e:

            print("⚠️ Download error:", e)

        current = chunk_end

    if not chunks:
        raise RuntimeError("No BTC data downloaded.")

    df = pd.DataFrame(
        chunks,
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

    print(f"✅ Total candles: {len(df)}")

    return df


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

    d = df.copy()

    print("⚙️ Calculating indicators...")

    # EMA
    d["ema20"] = d["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    d["ema50"] = d["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # ATR
    prev_close = d["close"].shift(1)

    tr1 = d["high"] - d["low"]

    tr2 = (
        d["high"] - prev_close
    ).abs()

    tr3 = (
        d["low"] - prev_close
    ).abs()

    d["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    d["atr"] = (
        d["tr"]
        .rolling(ATR_PERIOD)
        .mean()
    )

    # Candle properties
    d["body"] = (
        d["close"] - d["open"]
    ).abs()

    d["range"] = (
        d["high"] - d["low"]
    )

    d["body_ratio"] = np.where(
        d["range"] > 0,
        d["body"] / d["range"],
        0
    )

    # Volume
    d["volume_ma"] = (
        d["volume"]
        .rolling(20)
        .mean()
    )

    d["volume_ratio"] = np.where(
        d["volume_ma"] > 0,
        d["volume"] / d["volume_ma"],
        0
    )

    # Previous 20-candle resistance/support
    d["resistance"] = (
        d["high"]
        .shift(1)
        .rolling(LOOKBACK)
        .max()
    )

    d["support"] = (
        d["low"]
        .shift(1)
        .rolling(LOOKBACK)
        .min()
    )

    # Breakout levels
    d["breakout_up"] = (
        d["resistance"]
        + d["atr"] * BREAKOUT_ATR
    )

    d["breakout_down"] = (
        d["support"]
        - d["atr"] * BREAKOUT_ATR
    )

    # Trend
    d["trend"] = "SIDEWAYS"

    d.loc[
        (
            (d["close"] > d["ema20"]) &
            (d["ema20"] > d["ema50"])
        ),
        "trend"
    ] = "UP"

    d.loc[
        (
            (d["close"] < d["ema20"]) &
            (d["ema20"] < d["ema50"])
        ),
        "trend"
    ] = "DOWN"

    d = d.dropna().reset_index(drop=True)

    print(
        f"✅ Final usable candles: {len(d)}"
    )

    return d


# =========================================================
# FIND BREAKOUTS
# =========================================================

def find_breakouts(d, start_idx, end_idx):

    events = []

    i = start_idx

    while i < end_idx:

        row = d.iloc[i]

        atr = row["atr"]

        if not np.isfinite(atr) or atr <= 0:
            i += 1
            continue

        # -------------------------------------------------
        # BULLISH BREAKOUT
        # -------------------------------------------------

        if (
            row["trend"] == "UP"
            and row["close"] > row["breakout_up"]
            and row["close"] > row["open"]
            and row["body_ratio"] >= 0.40
        ):

            level = row["resistance"]

            events.append({
                "index": i,
                "direction": "BUY",
                "level": level
            })

            i += 1
            continue

        # -------------------------------------------------
        # BEARISH BREAKOUT
        # -------------------------------------------------

        if (
            row["trend"] == "DOWN"
            and row["close"] < row["breakout_down"]
            and row["close"] < row["open"]
            and row["body_ratio"] >= 0.40
        ):

            level = row["support"]

            events.append({
                "index": i,
                "direction": "SELL",
                "level": level
            })

        i += 1

    return events


# =========================================================
# FIND RETEST ENTRY
# =========================================================

def find_retest_entry(
    d,
    breakout_idx,
    direction,
    level,
    end_idx
):

    last_idx = min(
        breakout_idx + RETEST_MAX,
        end_idx - 1
    )

    first_idx = (
        breakout_idx + RETEST_MIN
    )

    if first_idx > last_idx:
        return None

    for j in range(
        first_idx,
        last_idx + 1
    ):

        row = d.iloc[j]

        atr = row["atr"]

        if not np.isfinite(atr) or atr <= 0:
            continue

        tolerance = atr * RETEST_ATR

        # =============================================
        # BUY RETEST
        # =============================================

        if direction == "BUY":

            touched = (
                row["low"]
                <= level + tolerance
            )

            reclaimed = (
                row["close"] > level
            )

            bullish = (
                row["close"] > row["open"]
            )

            good_body = (
                row["body_ratio"] >= 0.35
            )

            # Avoid weak retest
            volume_ok = (
                row["volume_ratio"] >= 0.80
            )

            trend_ok = (
                row["close"] > row["ema20"]
            )

            if (
                touched
                and reclaimed
                and bullish
                and good_body
                and volume_ok
                and trend_ok
            ):

                return j

        # =============================================
        # SELL RETEST
        # =============================================

        else:

            touched = (
                row["high"]
                >= level - tolerance
            )

            rejected = (
                row["close"] < level
            )

            bearish = (
                row["close"] < row["open"]
            )

            good_body = (
                row["body_ratio"] >= 0.35
            )

            volume_ok = (
                row["volume_ratio"] >= 0.80
            )

            trend_ok = (
                row["close"] < row["ema20"]
            )

            if (
                touched
                and rejected
                and bearish
                and good_body
                and volume_ok
                and trend_ok
            ):

                return j

    return None


# =========================================================
# SIMULATE TRADE
# =========================================================

def simulate_trade(
    d,
    entry_idx,
    direction,
    tp_atr,
    sl_atr,
    end_idx
):

    entry = d.iloc[entry_idx]

    entry_price = entry["close"]
    atr = entry["atr"]

    if not np.isfinite(atr) or atr <= 0:
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

    last_idx = min(
        entry_idx + MAX_HOLD_CANDLES,
        end_idx - 1
    )

    mfe = 0.0
    mae = 0.0

    result = "TIMEOUT"
    exit_price = d.iloc[last_idx]["close"]
    exit_idx = last_idx

    for j in range(
        entry_idx + 1,
        last_idx + 1
    ):

        row = d.iloc[j]

        if direction == "BUY":

            favorable = (
                row["high"] - entry_price
            ) / entry_price

            adverse = (
                row["low"] - entry_price
            ) / entry_price

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            hit_sl = (
                row["low"] <= sl
            )

            hit_tp = (
                row["high"] >= tp
            )

        else:

            favorable = (
                entry_price - row["low"]
            ) / entry_price

            adverse = (
                entry_price - row["high"]
            ) / entry_price

            mfe = max(
                mfe,
                favorable
            )

            mae = min(
                mae,
                adverse
            )

            hit_sl = (
                row["high"] >= sl
            )

            hit_tp = (
                row["low"] <= tp
            )

        # Conservative assumption:
        # if TP and SL hit same candle,
        # SL is counted first.

        if hit_sl:

            result = "SL"
            exit_price = sl
            exit_idx = j
            break

        if hit_tp:

            result = "TP"
            exit_price = tp
            exit_idx = j
            break

        exit_price = row["close"]
        exit_idx = j

    if direction == "BUY":

        gross_return = (
            exit_price - entry_price
        ) / entry_price

    else:

        gross_return = (
            entry_price - exit_price
        ) / entry_price

    net_return = (
        gross_return - COST
    )

    # R calculation
    if result == "TP":
        r_value = (
            tp_atr / sl_atr
        )

    elif result == "SL":
        r_value = -1.0

    else:

        risk = (
            atr * sl_atr
        ) / entry_price

        if risk > 0:
            r_value = (
                gross_return / risk
            )
        else:
            r_value = 0.0

    return {
        "entry_time": entry["time"],
        "exit_time": d.iloc[exit_idx]["time"],
        "direction": direction,
        "entry": entry_price,
        "exit": exit_price,
        "result": result,
        "gross_return": gross_return,
        "net_return": net_return,
        "r": r_value,
        "mfe": mfe,
        "mae": mae,
        "tp_atr": tp_atr,
        "sl_atr": sl_atr
    }


# =========================================================
# RUN BACKTEST
# =========================================================

def run_backtest(
    d,
    start_idx,
    end_idx,
    tp_atr,
    sl_atr
):

    events = find_breakouts(
        d,
        start_idx,
        end_idx
    )

    trades = []

    last_trade_idx = -999999

    for event in events:

        breakout_idx = event["index"]

        if (
            breakout_idx
            <= last_trade_idx
        ):
            continue

        direction = event["direction"]
        level = event["level"]

        entry_idx = find_retest_entry(
            d,
            breakout_idx,
            direction,
            level,
            end_idx
        )

        if entry_idx is None:
            continue

        if (
            entry_idx
            - last_trade_idx
            < COOLDOWN_CANDLES
        ):
            continue

        trade = simulate_trade(
            d,
            entry_idx,
            direction,
            tp_atr,
            sl_atr,
            end_idx
        )

        if trade is None:
            continue

        trade["breakout_time"] = (
            d.iloc[breakout_idx]["time"]
        )

        trade["level"] = level

        trades.append(trade)

        last_trade_idx = (
            trade_index_after_exit(
                d,
                trade["exit_time"]
            )
        )

    return trades


def trade_index_after_exit(
    d,
    exit_time
):

    matches = d.index[
        d["time"] == exit_time
    ]

    if len(matches) == 0:
        return -999999

    return int(matches[0])


# =========================================================
# RESULTS
# =========================================================

def print_results(
    method_name,
    trades,
    tp_atr,
    sl_atr
):

    if not trades:
        print(
            f"{method_name} | "
            f"TP {tp_atr:.2f} / SL {sl_atr:.2f}"
        )
        print("Trades: 0")
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

    tp_pct = tp / total * 100
    sl_pct = sl / total * 100
    timeout_pct = timeout / total * 100

    total_return = (
        df["net_return"].sum()
        * 100
    )

    avg_return = (
        df["net_return"].mean()
        * 100
    )

    total_r = df["r"].sum()
    avg_r = df["r"].mean()

    mfe = (
        df["mfe"].mean()
        * 100
    )

    mae = (
        df["mae"].mean()
        * 100
    )

    print(
        f"\n{method_name} | "
        f"TP {tp_atr:.2f} / SL {sl_atr:.2f}"
    )

    print(
        f"Trades: {total} | "
        f"TP {tp_pct:.1f}% | "
        f"SL {sl_pct:.1f}% | "
        f"Timeout {timeout_pct:.1f}%"
    )

    print(
        f"Return: {total_return:.2f}% | "
        f"Avg Return: {avg_return:.3f}% | "
        f"Avg R: {avg_r:.3f} | "
        f"Total R: {total_r:.2f}"
    )

    print(
        f"MFE: {mfe:.3f}% | "
        f"MAE: {mae:.3f}%"
    )


# =========================================================
# MAIN
# =========================================================

def main():

    print(
        "\n🚀 BTC 15M "
        "Breakout + Retest Backtest\n"
    )

    d = download_data()

    d = calculate_indicators(d)

    if len(d) < 500:

        raise RuntimeError(
            "Not enough candles."
        )

    first_time = d["time"].iloc[0]
    last_time = d["time"].iloc[-1]

    train_end_time = (
        first_time
        + pd.Timedelta(
            days=TRAIN_DAYS
        )
    )

    validation_start = (
        train_end_time
    )

    validation_end = last_time

    print(
        "\n🧠 Training:",
        first_time,
        "→",
        train_end_time
    )

    print(
        "🧪 Validation:",
        validation_start,
        "→",
        validation_end
    )

    train_start_idx = 100

    train_end_idx = (
        d["time"] < train_end_time
    ).sum()

    validation_start_idx = (
        d["time"] >= validation_start
    ).idxmax()

    validation_end_idx = len(d)

    # =====================================================
    # TEST CONFIGS
    # =====================================================

    configs = [
        (1.00, 0.75),
        (1.25, 0.75),
        (1.50, 1.00),
        (1.50, 1.25),
        (2.00, 1.00),
    ]

    print(
        "\n🔎 Finding Breakout + Retest setups..."
    )

    # =====================================================
    # TRAINING
    # =====================================================

    print(
        "\n"
        "================================================"
    )

    print(
        "TRAINING BREAKOUT + RETEST RESULTS"
    )

    print(
        "================================================"
    )

    positive_configs = []

    all_training_trades = []

    for tp_atr, sl_atr in configs:

        trades = run_backtest(
            d,
            train_start_idx,
            train_end_idx,
            tp_atr,
            sl_atr
        )

        all_training_trades.extend(
            trades
        )

        if trades:

            print_results(
                "BREAKOUT_RETEST",
                trades,
                tp_atr,
                sl_atr
            )

            temp = pd.DataFrame(
                trades
            )

            if len(temp) >= MIN_TRAIN_TRADES:

                avg_r = temp["r"].mean()

                total_return = (
                    temp["net_return"].sum()
                )

                if (
                    avg_r > 0
                    and total_return > 0
                ):

                    positive_configs.append(
                        (
                            tp_atr,
                            sl_atr,
                            avg_r,
                            total_return,
                            len(temp)
                        )
                    )

    # =====================================================
    # TRAINING DECISION
    # =====================================================

    print(
        "\n"
        "================================================"
    )

    print(
        "TRAINING DECISION"
    )

    print(
        "================================================"
    )

    if not positive_configs:

        print(
            "\n❌ No positive training configuration."
        )

        print(
            "Validation is NOT forced."
        )

        print(
            "\n➡️ Breakout + Retest strategy "
            "will NOT go live."
        )

        return

    positive_configs.sort(
        key=lambda x: x[2],
        reverse=True
    )

    print(
        f"\n✅ Positive training configs: "
        f"{len(positive_configs)}"
    )

    for cfg in positive_configs:

        print(
            f"TP {cfg[0]:.2f} / "
            f"SL {cfg[1]:.2f} | "
            f"Trades {cfg[4]} | "
            f"Avg R {cfg[2]:.3f} | "
            f"Return {cfg[3] * 100:.2f}%"
        )

    # Best configuration
    best_tp, best_sl = (
        positive_configs[0][0],
        positive_configs[0][1]
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    print(
        "\n"
        "================================================"
    )

    print(
        "VALIDATION"
    )

    print(
        "================================================"
    )

    validation_trades = run_backtest(
        d,
        validation_start_idx,
        validation_end_idx,
        best_tp,
        best_sl
    )

    if (
        len(validation_trades)
        < MIN_VALIDATION_TRADES
    ):

        print(
            f"\n⚠️ Validation trades "
            f"too few: {len(validation_trades)}"
        )

        print(
            "Validation result is not "
            "strong enough to use live."
        )

        return

    print_results(
        "VALIDATION",
        validation_trades,
        best_tp,
        best_sl
    )

    # =====================================================
    # SAVE
    # =====================================================

    validation_df = pd.DataFrame(
        validation_trades
    )

    validation_df.to_csv(
        "btc_breakout_retest_validation.csv",
        index=False
    )

    print(
        "\n💾 Saved:"
        " btc_breakout_retest_validation.csv"
    )

    # =====================================================
    # FINAL DECISION
    # =====================================================

    avg_r = validation_df["r"].mean()
    total_return = (
        validation_df["net_return"].sum()
    )

    print(
        "\n"
        "================================================"
    )

    print(
        "FINAL DECISION"
    )

    print(
        "================================================"
    )

    if (
        avg_r > 0
        and total_return > 0
    ):

        print(
            "\n🟢 Validation is POSITIVE."
        )

        print(
            "This strategy may be considered "
            "for further paper testing."
        )

        print(
            "⚠️ Do NOT immediately use "
            "real money."
        )

    else:

        print(
            "\n🔴 Validation is NEGATIVE."
        )

        print(
            "Do NOT put this strategy "
            "into the live bot."
        )


if __name__ == "__main__":
    main()

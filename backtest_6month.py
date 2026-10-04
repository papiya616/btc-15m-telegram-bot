import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M LIQUIDITY SWEEP - ENTRY TIMING BACKTEST
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900
TOTAL_DAYS = 183

TRAIN_DAYS = 120
VALIDATION_DAYS = 63

COST = 0.0014

SWING_LOOKBACK = 12

# Sweep sensitivity
SWEEP_ATR = 0.05

# Maximum candles after sweep to wait
MAX_WAIT = 2

# Maximum trade holding time
MAX_HOLD = 12

# Cooldown
COOLDOWN = 12

MIN_TRAIN_TRADES = 30
MIN_VALIDATION_TRADES = 30


# ============================================================
# DOWNLOAD DATA
# ============================================================

def download_data():

    print("📥 Downloading BTC 15M data...")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=TOTAL_DAYS)

    rows = []
    cursor = start

    chunk_seconds = GRANULARITY * 250

    while cursor < end:

        chunk_end = min(
            cursor + timedelta(seconds=chunk_seconds),
            end
        )

        print(
            f"Downloading: "
            f"{cursor.strftime('%Y-%m-%d')} "
            f"to {chunk_end.strftime('%Y-%m-%d')}"
        )

        url = (
            "https://api.exchange.coinbase.com/"
            f"products/{PRODUCT}/candles"
        )

        params = {
            "granularity": GRANULARITY,
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat()
        }

        r = requests.get(
            url,
            params=params,
            timeout=30
        )

        r.raise_for_status()

        data = r.json()

        if data:
            rows.extend(data)

        cursor = chunk_end + timedelta(seconds=1)

    df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "low",
            "high",
            "open",
            "close",
            "volume"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(
            df[col]
        )

    df = df.reset_index(drop=True)

    print(
        f"✅ Total candles: {len(df)}"
    )

    return df


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):

    d = df.copy()

    # EMA
    d["ema9"] = d["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["ema21"] = d["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    # RSI
    delta = d["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    d["rsi"] = (
        100 -
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

    d["macd"] = (
        ema12 -
        ema26
    )

    d["macd_signal"] = d[
        "macd"
    ].ewm(
        span=9,
        adjust=False
    ).mean()

    d["macd_hist"] = (
        d["macd"] -
        d["macd_signal"]
    )

    # ATR
    prev = d["close"].shift(1)

    tr1 = (
        d["high"] -
        d["low"]
    )

    tr2 = (
        d["high"] -
        prev
    ).abs()

    tr3 = (
        d["low"] -
        prev
    ).abs()

    d["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    d["atr"] = d["tr"].rolling(
        14
    ).mean()

    # Volume
    d["volume_avg"] = d[
        "volume"
    ].rolling(20).mean()

    d["volume_ratio"] = (
        d["volume"] /
        d["volume_avg"]
    )

    # Candle body
    d["body"] = (
        d["close"] -
        d["open"]
    ).abs()

    return d


# ============================================================
# 1H TREND
# ============================================================

def add_1h_trend(d):

    h = (
        d.set_index("timestamp")
        .resample("1h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last"
        })
        .dropna()
    )

    h["ema20"] = h[
        "close"
    ].ewm(
        span=20,
        adjust=False
    ).mean()

    h["ema50"] = h[
        "close"
    ].ewm(
        span=50,
        adjust=False
    ).mean()

    h["trend"] = "SIDEWAYS"

    h.loc[
        (
            h["ema20"] >
            h["ema50"]
        ) &
        (
            h["close"] >
            h["ema20"]
        ),
        "trend"
    ] = "UP"

    h.loc[
        (
            h["ema20"] <
            h["ema50"]
        ) &
        (
            h["close"] <
            h["ema20"]
        ),
        "trend"
    ] = "DOWN"

    d = pd.merge_asof(
        d.sort_values(
            "timestamp"
        ),
        h[
            [
                "ema20",
                "ema50",
                "trend"
            ]
        ].reset_index(),
        on="timestamp",
        direction="backward"
    )

    return d


# ============================================================
# FIND LIQUIDITY SWEEPS
# ============================================================

def find_sweeps(d):

    sweeps = []

    for i in range(
        SWING_LOOKBACK + 60,
        len(d) - MAX_WAIT - MAX_HOLD - 1
    ):

        row = d.iloc[i]

        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        trend = row["trend"]

        if trend not in [
            "UP",
            "DOWN"
        ]:
            continue

        previous = d.iloc[
            i - SWING_LOOKBACK:i
        ]

        swing_low = previous[
            "low"
        ].min()

        swing_high = previous[
            "high"
        ].max()

        # BUY sweep
        if trend == "UP":

            if (
                row["low"] <
                swing_low -
                atr * SWEEP_ATR
            ):

                sweeps.append({
                    "sweep_index": i,
                    "direction": "BUY",
                    "level": swing_low
                })

        # SELL sweep
        elif trend == "DOWN":

            if (
                row["high"] >
                swing_high +
                atr * SWEEP_ATR
            ):

                sweeps.append({
                    "sweep_index": i,
                    "direction": "SELL",
                    "level": swing_high
                })

    return sweeps


# ============================================================
# ENTRY TIMING METHODS
# ============================================================

def find_entry(
    d,
    sweep,
    method
):

    i = sweep["sweep_index"]
    direction = sweep["direction"]
    level = sweep["level"]

    last = min(
        i + MAX_WAIT,
        len(d) - 1
    )

    # --------------------------------------------------------
    # METHOD 1
    # Enter immediately at sweep candle close
    # --------------------------------------------------------

    if method == "SWEEP_CLOSE":

        row = d.iloc[i]

        if direction == "BUY":

            if row["close"] > level:

                return {
                    "entry_index": i,
                    "direction": direction,
                    "entry": row["close"]
                }

        else:

            if row["close"] < level:

                return {
                    "entry_index": i,
                    "direction": direction,
                    "entry": row["close"]
                }

        return None

    # --------------------------------------------------------
    # METHOD 2
    # Wait 1 candle for confirmation
    # --------------------------------------------------------

    if method == "CONFIRM_1":

        j = i + 1

        if j >= len(d):
            return None

        row = d.iloc[j]

        if direction == "BUY":

            if (
                row["close"] > row["open"]
                and
                row["close"] > level
            ):

                return {
                    "entry_index": j,
                    "direction": direction,
                    "entry": row["close"]
                }

        else:

            if (
                row["close"] < row["open"]
                and
                row["close"] < level
            ):

                return {
                    "entry_index": j,
                    "direction": direction,
                    "entry": row["close"]
                }

        return None

    # --------------------------------------------------------
    # METHOD 3
    # Wait up to 2 candles
    # --------------------------------------------------------

    if method == "CONFIRM_2":

        for j in range(
            i + 1,
            last + 1
        ):

            row = d.iloc[j]

            if direction == "BUY":

                if (
                    row["close"] > row["open"]
                    and
                    row["close"] > level
                ):

                    return {
                        "entry_index": j,
                        "direction": direction,
                        "entry": row["close"]
                    }

            else:

                if (
                    row["close"] < row["open"]
                    and
                    row["close"] < level
                ):

                    return {
                        "entry_index": j,
                        "direction": direction,
                        "entry": row["close"]
                    }

    return None


# ============================================================
# TEST TRADE
# ============================================================

def test_trade(
    d,
    setup,
    tp_mult,
    sl_mult
):

    i = setup["entry_index"]

    direction = setup[
        "direction"
    ]

    entry = setup["entry"]

    atr = d.iloc[i]["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp = (
            entry +
            atr * tp_mult
        )

        sl = (
            entry -
            atr * sl_mult
        )

    else:

        tp = (
            entry -
            atr * tp_mult
        )

        sl = (
            entry +
            atr * sl_mult
        )

    end = min(
        i + MAX_HOLD,
        len(d) - 1
    )

    result = "TIMEOUT"

    exit_price = d.iloc[
        end
    ]["close"]

    mfe = 0.0
    mae = 0.0

    for j in range(
        i + 1,
        end + 1
    ):

        row = d.iloc[j]

        if direction == "BUY":

            fav = (
                row["high"] -
                entry
            ) / entry

            adv = (
                row["low"] -
                entry
            ) / entry

            mfe = max(
                mfe,
                fav
            )

            mae = min(
                mae,
                adv
            )

            hit_sl = (
                row["low"] <= sl
            )

            hit_tp = (
                row["high"] >= tp
            )

        else:

            fav = (
                entry -
                row["low"]
            ) / entry

            adv = (
                entry -
                row["high"]
            ) / entry

            mfe = max(
                mfe,
                fav
            )

            mae = min(
                mae,
                adv
            )

            hit_sl = (
                row["high"] >= sl
            )

            hit_tp = (
                row["low"] <= tp
            )

        # Conservative SL first
        if hit_sl:

            result = "SL"
            exit_price = sl
            break

        if hit_tp:

            result = "TP"
            exit_price = tp
            break

    if direction == "BUY":

        gross = (
            exit_price -
            entry
        ) / entry

    else:

        gross = (
            entry -
            exit_price
        ) / entry

    net = gross - COST

    risk = (
        sl_mult *
        atr /
        entry
    )

    r = net / risk

    return {
        "direction": direction,
        "result": result,
        "entry_index": i,
        "entry_time": d.iloc[
            i
        ]["timestamp"],
        "entry": entry,
        "exit": exit_price,
        "return": net,
        "r": r,
        "mfe": mfe,
        "mae": mae
    }


# ============================================================
# SUMMARY
# ============================================================

def summary(trades):

    if not trades:
        return None

    x = pd.DataFrame(trades)

    n = len(x)

    tp = (
        x["result"] == "TP"
    ).sum()

    sl = (
        x["result"] == "SL"
    ).sum()

    timeout = (
        x["result"] == "TIMEOUT"
    ).sum()

    return {
        "n": n,
        "tp": tp,
        "sl": sl,
        "timeout": timeout,
        "tp_pct": tp / n * 100,
        "sl_pct": sl / n * 100,
        "timeout_pct": timeout / n * 100,
        "return": x["return"].sum() * 100,
        "avg_return": x["return"].mean() * 100,
        "total_r": x["r"].sum(),
        "avg_r": x["r"].mean(),
        "mfe": x["mfe"].mean() * 100,
        "mae": x["mae"].mean() * 100
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print(
        "🚀 BTC Liquidity Sweep "
        "Entry Timing Backtest"
    )
    print()

    d = download_data()

    print(
        "⚙️ Calculating indicators..."
    )

    d = indicators(d)

    d = add_1h_trend(d)

    d = d.dropna().reset_index(
        drop=True
    )

    print(
        f"✅ Final usable candles: "
        f"{len(d)}"
    )

    start = d.iloc[0]["timestamp"]

    train_end = (
        start +
        timedelta(days=TRAIN_DAYS)
    )

    validation_end = (
        train_end +
        timedelta(days=VALIDATION_DAYS)
    )

    print()
    print(
        f"🧠 Training: "
        f"{start} → {train_end}"
    )

    print(
        f"🧪 Validation: "
        f"{train_end} → {validation_end}"
    )

    print()
    print(
        "🔎 Finding liquidity sweeps..."
    )

    sweeps = find_sweeps(d)

    print(
        f"📊 Total sweeps: "
        f"{len(sweeps)}"
    )

    methods = [
        "SWEEP_CLOSE",
        "CONFIRM_1",
        "CONFIRM_2"
    ]

    configs = [
        (1.00, 0.75),
        (1.25, 0.75),
        (1.50, 1.00),
        (1.50, 1.25),
        (2.00, 1.00)
    ]

    # ========================================================
    # TRAINING
    # ========================================================

    print()
    print(
        "================================================"
    )
    print(
        "TRAINING ENTRY TIMING RESULTS"
    )
    print(
        "================================================"
    )

    positive = []

    for method in methods:

        setups = []

        for sweep in sweeps:

            setup = find_entry(
                d,
                sweep,
                method
            )

            if setup is None:
                continue

            time = d.iloc[
                setup["entry_index"]
            ]["timestamp"]

            if time < train_end:

                setups.append(
                    setup
                )

        print()
        print(
            f"📌 METHOD: {method}"
        )

        print(
            f"Training setups: "
            f"{len(setups)}"
        )

        for tp, sl in configs:

            trades = []

            for setup in setups:

                trade = test_trade(
                    d,
                    setup,
                    tp,
                    sl
                )

                if trade:
                    trades.append(
                        trade
                    )

            s = summary(trades)

            if s is None:
                continue

            print()
            print(
                f"{method} | "
                f"TP {tp:.2f} / "
                f"SL {sl:.2f}"
            )

            print(
                f"Trades: {s['n']} | "
                f"TP {s['tp_pct']:.1f}% | "
                f"SL {s['sl_pct']:.1f}% | "
                f"Timeout {s['timeout_pct']:.1f}%"
            )

            print(
                f"Return: "
                f"{s['return']:.2f}% | "
                f"Avg R: "
                f"{s['avg_r']:.3f} | "
                f"Total R: "
                f"{s['total_r']:.2f}"
            )

            print(
                f"MFE: "
                f"{s['mfe']:.3f}% | "
                f"MAE: "
                f"{s['mae']:.3f}%"
            )

            if (
                s["n"] >= MIN_TRAIN_TRADES
                and
                s["return"] > 0
                and
                s["avg_r"] > 0
            ):

                positive.append(
                    (
                        method,
                        tp,
                        sl,
                        s
                    )
                )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print(
        "================================================"
    )
    print(
        "VALIDATION"
    )
    print(
        "================================================"
    )

    if not positive:

        print()
        print(
            "❌ No positive training configuration."
        )

        print(
            "Validation is NOT forced."
        )

        return

    positive.sort(
        key=lambda x: x[3]["avg_r"],
        reverse=True
    )

    best_method, best_tp, best_sl, best_train = (
        positive[0]
    )

    print()
    print(
        "🏆 Best training configuration:"
    )

    print(
        f"Method: {best_method}"
    )

    print(
        f"TP: {best_tp:.2f} ATR"
    )

    print(
        f"SL: {best_sl:.2f} ATR"
    )

    print(
        f"Training trades: "
        f"{best_train['n']}"
    )

    print(
        f"Training return: "
        f"{best_train['return']:.2f}%"
    )

    print(
        f"Training Avg R: "
        f"{best_train['avg_r']:.3f}"
    )

    # Build validation setups
    val_setups = []

    for sweep in sweeps:

        setup = find_entry(
            d,
            sweep,
            best_method
        )

        if setup is None:
            continue

        time = d.iloc[
            setup["entry_index"]
        ]["timestamp"]

        if (
            time >= train_end
            and
            time <= validation_end
        ):

            val_setups.append(
                setup
            )

    trades = []

    for setup in val_setups:

        trade = test_trade(
            d,
            setup,
            best_tp,
            best_sl
        )

        if trade:
            trades.append(
                trade
            )

    if len(trades) < MIN_VALIDATION_TRADES:

        print()
        print(
            "⚠️ Validation sample too small:"
        )

        print(
            f"{len(trades)} trades"
        )

        return

    v = summary(trades)

    print()
    print(
        "🧪 VALIDATION RESULT"
    )

    print(
        f"Trades: {v['n']}"
    )

    print(
        f"TP: {v['tp']} "
        f"({v['tp_pct']:.1f}%)"
    )

    print(
        f"SL: {v['sl']} "
        f"({v['sl_pct']:.1f}%)"
    )

    print(
        f"Timeout: {v['timeout']} "
        f"({v['timeout_pct']:.1f}%)"
    )

    print(
        f"Return: "
        f"{v['return']:.2f}%"
    )

    print(
        f"Avg Return: "
        f"{v['avg_return']:.3f}%"
    )

    print(
        f"Total R: "
        f"{v['total_r']:.2f}"
    )

    print(
        f"Avg R: "
        f"{v['avg_r']:.3f}"
    )

    print(
        f"MFE: "
        f"{v['mfe']:.3f}%"
    )

    print(
        f"MAE: "
        f"{v['mae']:.3f}%"
    )

    pd.DataFrame(
        trades
    ).to_csv(
        "btc_entry_timing_validation.csv",
        index=False
    )

    print()
    print(
        "💾 Saved:"
    )

    print(
        "btc_entry_timing_validation.csv"
    )

    if (
        v["return"] > 0
        and
        v["avg_r"] > 0
    ):

        print()
        print(
            "🟢 VALIDATION POSITIVE"
        )

    else:

        print()
        print(
            "🔴 VALIDATION NEGATIVE"
        )

        print(
            "Do NOT use this setup in "
            "the live bot yet."
        )


if __name__ == "__main__":
    main()

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 15M LIQUIDITY SWEEP + CONFIRMATION BACKTEST
# Balanced version
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
TOTAL_DAYS = 183

TRAIN_DAYS = 120
VALIDATION_DAYS = 63

COST = 0.0014              # 0.14% round-trip cost
COOLDOWN_CANDLES = 12      # 3 hours
MAX_HOLD_CANDLES = 12      # 3 hours

MIN_TRAIN_TRADES = 30
MIN_VALIDATION_TRADES = 30

# Sweep settings
SWING_LOOKBACK = 12
SWEEP_ATR = 0.05
RECLAIM_ATR = 0.00

# Confirmation
MAX_CONFIRM_CANDLES = 2

# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_candles():
    print("📥 Downloading BTC 15M data...")

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=TOTAL_DAYS)

    all_rows = []
    cursor = start

    chunk_seconds = GRANULARITY * 250

    while cursor < end:
        chunk_end = min(
            cursor + timedelta(seconds=chunk_seconds),
            end
        )

        url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

        params = {
            "granularity": GRANULARITY,
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat()
        }

        print(
            f"Downloading: "
            f"{cursor.strftime('%Y-%m-%d')} "
            f"to {chunk_end.strftime('%Y-%m-%d')}"
        )

        r = requests.get(
            url,
            params=params,
            timeout=30
        )

        r.raise_for_status()

        rows = r.json()

        if rows:
            all_rows.extend(rows)

        cursor = chunk_end
        cursor += timedelta(seconds=1)

    if not all_rows:
        raise RuntimeError("No BTC data received.")

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

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df = df.drop_duplicates(
        subset=["timestamp"]
    )

    df = df.sort_values("timestamp")

    for col in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:
        df[col] = pd.to_numeric(df[col])

    df = df.reset_index(drop=True)

    print(f"✅ Total candles: {len(df)}")

    return df


# ============================================================
# INDICATORS
# ============================================================

def add_indicators(df):

    d = df.copy()

    # 15M EMA
    d["ema9"] = d["close"].ewm(
        span=9,
        adjust=False
    ).mean()

    d["ema21"] = d["close"].ewm(
        span=21,
        adjust=False
    ).mean()

    d["ema50"] = d["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    # RSI
    delta = d["close"].diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

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

    d["atr"] = d["tr"].rolling(14).mean()

    # Volume average
    d["volume_avg"] = d["volume"].rolling(20).mean()

    d["volume_ratio"] = (
        d["volume"] /
        d["volume_avg"]
    )

    # Candle
    d["body"] = (
        d["close"] -
        d["open"]
    ).abs()

    d["range"] = (
        d["high"] -
        d["low"]
    )

    # 1H data
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

    h["ema20"] = h["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    h["ema50"] = h["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    h["atr"] = (
        h["high"] -
        h["low"]
    ).rolling(14).mean()

    h["trend"] = "SIDEWAYS"

    h.loc[
        (h["ema20"] > h["ema50"]) &
        (
            h["close"] >
            h["ema20"]
        ),
        "trend"
    ] = "UP"

    h.loc[
        (h["ema20"] < h["ema50"]) &
        (
            h["close"] <
            h["ema20"]
        ),
        "trend"
    ] = "DOWN"

    d = pd.merge_asof(
        d.sort_values("timestamp"),
        h[
            [
                "ema20",
                "ema50",
                "atr",
                "trend"
            ]
        ].reset_index(),
        on="timestamp",
        direction="backward",
        suffixes=("", "_1h")
    )

    return d


# ============================================================
# LIQUIDITY SWEEP DETECTION
# ============================================================

def detect_sweep(d, i):

    if i < SWING_LOOKBACK + 5:
        return None

    row = d.iloc[i]

    atr = row["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    previous = d.iloc[
        i - SWING_LOOKBACK:i
    ]

    swing_low = previous["low"].min()
    swing_high = previous["high"].max()

    # --------------------------------------------------------
    # BUY SWEEP
    # Price breaks previous low,
    # then closes back above that level.
    # --------------------------------------------------------

    buy_sweep = (
        row["low"] <
        swing_low - atr * SWEEP_ATR
        and
        row["close"] >
        swing_low + atr * RECLAIM_ATR
    )

    # --------------------------------------------------------
    # SELL SWEEP
    # Price breaks previous high,
    # then closes back below that level.
    # --------------------------------------------------------

    sell_sweep = (
        row["high"] >
        swing_high + atr * SWEEP_ATR
        and
        row["close"] <
        swing_high - atr * RECLAIM_ATR
    )

    if buy_sweep:
        return {
            "direction": "BUY",
            "level": swing_low,
            "sweep_index": i
        }

    if sell_sweep:
        return {
            "direction": "SELL",
            "level": swing_high,
            "sweep_index": i
        }

    return None


# ============================================================
# CONFIRMATION
# ============================================================

def confirm_trade(d, sweep):

    direction = sweep["direction"]
    start = sweep["sweep_index"]

    for j in range(
        start,
        min(
            start +
            MAX_CONFIRM_CANDLES +
            1,
            len(d)
        )
    ):

        row = d.iloc[j]

        atr = row["atr"]

        if pd.isna(atr) or atr <= 0:
            continue

        body = abs(
            row["close"] -
            row["open"]
        )

        # BUY confirmation
        if direction == "BUY":

            bullish = (
                row["close"] >
                row["open"]
            )

            above_ema = (
                row["close"] >
                row["ema21"]
            )

            macd_ok = (
                row["macd_hist"] >= 0
            )

            rsi_ok = (
                40 <= row["rsi"] <= 70
            )

            volume_ok = (
                row["volume_ratio"] >= 0.75
            )

            score = sum([
                bullish,
                above_ema,
                macd_ok,
                rsi_ok,
                volume_ok
            ])

            # Need 3 of 5
            if score >= 3:

                return {
                    "entry_index": j,
                    "direction": "BUY",
                    "entry": row["close"],
                    "atr": atr
                }

        # SELL confirmation
        else:

            bearish = (
                row["close"] <
                row["open"]
            )

            below_ema = (
                row["close"] <
                row["ema21"]
            )

            macd_ok = (
                row["macd_hist"] <= 0
            )

            rsi_ok = (
                30 <= row["rsi"] <= 60
            )

            volume_ok = (
                row["volume_ratio"] >= 0.75
            )

            score = sum([
                bearish,
                below_ema,
                macd_ok,
                rsi_ok,
                volume_ok
            ])

            if score >= 3:

                return {
                    "entry_index": j,
                    "direction": "SELL",
                    "entry": row["close"],
                    "atr": atr
                }

    return None


# ============================================================
# TEST ONE TRADE
# ============================================================

def test_trade(
    d,
    setup,
    tp_mult,
    sl_mult
):

    entry_index = setup["entry_index"]
    direction = setup["direction"]

    entry = setup["entry"]
    atr = setup["atr"]

    if pd.isna(atr) or atr <= 0:
        return None

    if direction == "BUY":

        tp = entry + atr * tp_mult
        sl = entry - atr * sl_mult

    else:

        tp = entry - atr * tp_mult
        sl = entry + atr * sl_mult

    end_index = min(
        entry_index +
        MAX_HOLD_CANDLES,
        len(d) - 1
    )

    max_favorable = 0.0
    max_adverse = 0.0

    result = "TIMEOUT"
    exit_price = d.iloc[
        end_index
    ]["close"]

    for j in range(
        entry_index + 1,
        end_index + 1
    ):

        row = d.iloc[j]

        if direction == "BUY":

            favorable = (
                row["high"] -
                entry
            ) / entry

            adverse = (
                row["low"] -
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

            hit_sl = (
                row["low"] <= sl
            )

            hit_tp = (
                row["high"] >= tp
            )

        else:

            favorable = (
                entry -
                row["low"]
            ) / entry

            adverse = (
                entry -
                row["high"]
            ) / entry

            max_favorable = max(
                max_favorable,
                favorable
            )

            max_adverse = min(
                max_adverse,
                adverse
            )

            hit_sl = (
                row["high"] >= sl
            )

            hit_tp = (
                row["low"] <= tp
            )

        # Conservative:
        # if both happen same candle -> SL first
        if hit_sl:

            result = "SL"
            exit_price = sl
            break

        if hit_tp:

            result = "TP"
            exit_price = tp
            break

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
        COST
    )

    risk = (
        sl_mult *
        atr /
        entry
    )

    if risk > 0:
        r_multiple = (
            net_return /
            risk
        )
    else:
        r_multiple = 0

    return {
        "direction": direction,
        "result": result,
        "entry_index": entry_index,
        "entry_time": d.iloc[
            entry_index
        ]["timestamp"],
        "entry": entry,
        "exit": exit_price,
        "return": net_return,
        "r": r_multiple,
        "mfe": max_favorable,
        "mae": max_adverse
    }


# ============================================================
# GENERATE SIGNALS
# ============================================================

def generate_setups(d):

    setups = []

    last_trade_index = -999

    for i in range(
        SWING_LOOKBACK + 60,
        len(d) -
        MAX_HOLD_CANDLES -
        MAX_CONFIRM_CANDLES -
        1
    ):

        if (
            i -
            last_trade_index
            < COOLDOWN_CANDLES
        ):
            continue

        row = d.iloc[i]

        trend = row["trend"]

        if trend not in [
            "UP",
            "DOWN"
        ]:
            continue

        sweep = detect_sweep(
            d,
            i
        )

        if sweep is None:
            continue

        # 1H direction must match sweep
        if (
            trend == "UP"
            and
            sweep["direction"] != "BUY"
        ):
            continue

        if (
            trend == "DOWN"
            and
            sweep["direction"] != "SELL"
        ):
            continue

        setup = confirm_trade(
            d,
            sweep
        )

        if setup is None:
            continue

        # Prevent confirmation from going too far
        if (
            setup["entry_index"] -
            i >
            MAX_CONFIRM_CANDLES
        ):
            continue

        setups.append(setup)

        last_trade_index = setup[
            "entry_index"
        ]

    return setups


# ============================================================
# BACKTEST CONFIG
# ============================================================

CONFIGS = [
    (0.75, 0.50),
    (1.00, 0.50),
    (1.25, 0.75),
    (1.50, 1.00),
    (1.50, 1.25),
    (2.00, 1.00),
]


# ============================================================
# SUMMARY
# ============================================================

def summarize(trades):

    if not trades:
        return None

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

    return {
        "trades": total,
        "tp": tp,
        "sl": sl,
        "timeout": timeout,
        "tp_pct": tp / total * 100,
        "sl_pct": sl / total * 100,
        "timeout_pct": timeout / total * 100,
        "return_pct": df["return"].sum() * 100,
        "avg_return": df["return"].mean() * 100,
        "total_r": df["r"].sum(),
        "avg_r": df["r"].mean(),
        "mfe": df["mfe"].mean() * 100,
        "mae": df["mae"].mean() * 100
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("🚀 BTC Liquidity Sweep Balanced Backtest")
    print()

    df = download_candles()

    print("⚙️ Calculating indicators...")

    df = add_indicators(df)

    df = df.dropna().reset_index(
        drop=True
    )

    print(
        f"✅ Final usable candles: "
        f"{len(df)}"
    )

    start_time = df.iloc[0]["timestamp"]

    train_end = (
        start_time +
        timedelta(days=TRAIN_DAYS)
    )

    validation_end = (
        train_end +
        timedelta(days=VALIDATION_DAYS)
    )

    print()
    print(
        f"🧠 Training: "
        f"{start_time} → {train_end}"
    )

    print(
        f"🧪 Validation: "
        f"{train_end} → {validation_end}"
    )

    print()
    print("🔎 Generating liquidity sweep setups...")

    setups = generate_setups(df)

    print(
        f"📊 Total setups found: "
        f"{len(setups)}"
    )

    if len(setups) < MIN_TRAIN_TRADES:
        print()
        print(
            "⚠️ Too few setups."
        )
        print(
            "Need at least "
            f"{MIN_TRAIN_TRADES} "
            "training trades."
        )
        return

    # ========================================================
    # SPLIT SETUPS
    # ========================================================

    train_setups = [
        s for s in setups
        if s["entry_index"] <
        df.index[
            df["timestamp"] >= train_end
        ][0]
    ]

    validation_setups = [
        s for s in setups
        if s["entry_index"] >=
        df.index[
            df["timestamp"] >= train_end
        ][0]
    ]

    print()
    print(
        f"Training setups: "
        f"{len(train_setups)}"
    )

    print(
        f"Validation setups: "
        f"{len(validation_setups)}"
    )

    print()
    print("================================================")
    print("TRAINING RESULTS")
    print("================================================")

    positive_configs = []

    for tp, sl in CONFIGS:

        trades = []

        for setup in train_setups:

            trade = test_trade(
                df,
                setup,
                tp,
                sl
            )

            if trade:
                trades.append(trade)

        summary = summarize(
            trades
        )

        if summary is None:
            continue

        print()
        print(
            f"TP {tp:.2f} ATR / "
            f"SL {sl:.2f} ATR"
        )

        print(
            f"Trades: {summary['trades']}"
        )

        print(
            f"TP: {summary['tp']} "
            f"({summary['tp_pct']:.1f}%)"
        )

        print(
            f"SL: {summary['sl']} "
            f"({summary['sl_pct']:.1f}%)"
        )

        print(
            f"Timeout: {summary['timeout']} "
            f"({summary['timeout_pct']:.1f}%)"
        )

        print(
            f"Return: "
            f"{summary['return_pct']:.2f}%"
        )

        print(
            f"Avg Return: "
            f"{summary['avg_return']:.3f}%"
        )

        print(
            f"Total R: "
            f"{summary['total_r']:.2f}"
        )

        print(
            f"Avg R: "
            f"{summary['avg_r']:.3f}"
        )

        print(
            f"MFE: "
            f"{summary['mfe']:.3f}%"
        )

        print(
            f"MAE: "
            f"{summary['mae']:.3f}%"
        )

        if (
            summary["trades"] >=
            MIN_TRAIN_TRADES
            and
            summary["return_pct"] > 0
            and
            summary["avg_r"] > 0
        ):

            positive_configs.append(
                (
                    tp,
                    sl,
                    summary
                )
            )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()
    print("================================================")
    print("VALIDATION")
    print("================================================")

    if not positive_configs:

        print()
        print(
            "❌ No positive training configuration."
        )

        print(
            "Validation is intentionally NOT forced."
        )

        return

    # Best training config
    positive_configs.sort(
        key=lambda x: x[2]["avg_r"],
        reverse=True
    )

    best_tp, best_sl, best_train = (
        positive_configs[0]
    )

    print()
    print(
        f"🏆 Best training setup:"
    )

    print(
        f"TP {best_tp:.2f} ATR / "
        f"SL {best_sl:.2f} ATR"
    )

    print(
        f"Training Avg R: "
        f"{best_train['avg_r']:.3f}"
    )

    print(
        f"Training Return: "
        f"{best_train['return_pct']:.2f}%"
    )

    validation_trades = []

    for setup in validation_setups:

        trade = test_trade(
            df,
            setup,
            best_tp,
            best_sl
        )

        if trade:
            validation_trades.append(
                trade
            )

    print()

    if len(validation_trades) < MIN_VALIDATION_TRADES:

        print(
            "⚠️ Validation sample too small."
        )

        print(
            f"Only {len(validation_trades)} "
            "validation trades."
        )

        return

    val = summarize(
        validation_trades
    )

    print(
        f"Validation Trades: "
        f"{val['trades']}"
    )

    print(
        f"TP: {val['tp']} "
        f"({val['tp_pct']:.1f}%)"
    )

    print(
        f"SL: {val['sl']} "
        f"({val['sl_pct']:.1f}%)"
    )

    print(
        f"Timeout: {val['timeout']} "
        f"({val['timeout_pct']:.1f}%)"
    )

    print(
        f"Return: "
        f"{val['return_pct']:.2f}%"
    )

    print(
        f"Avg Return: "
        f"{val['avg_return']:.3f}%"
    )

    print(
        f"Total R: "
        f"{val['total_r']:.2f}"
    )

    print(
        f"Avg R: "
        f"{val['avg_r']:.3f}"
    )

    print(
        f"MFE: "
        f"{val['mfe']:.3f}%"
    )

    print(
        f"MAE: "
        f"{val['mae']:.3f}%"
    )

    # ========================================================
    # SAVE VALIDATION TRADES
    # ========================================================

    pd.DataFrame(
        validation_trades
    ).to_csv(
        "btc_liquidity_sweep_balanced_validation.csv",
        index=False
    )

    print()
    print(
        "💾 Saved:"
    )

    print(
        "btc_liquidity_sweep_balanced_validation.csv"
    )

    print()

    if (
        val["return_pct"] > 0
        and
        val["avg_r"] > 0
    ):

        print(
            "🟢 VALIDATION POSITIVE"
        )

        print(
            "This setup may be worth "
            "further testing."
        )

    else:

        print(
            "🔴 VALIDATION NEGATIVE"
        )

        print(
            "Do NOT use this setup "
            "in the live bot yet."
        )


if __name__ == "__main__":
    main()

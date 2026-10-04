import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone


PRODUCT = "BTC-USD"
GRANULARITY = 900

DAYS = 183
TRAIN_DAYS = 120
VALIDATION_DAYS = 63

MAX_HOLD = 12
COOLDOWN = 12

COST = 0.0014
SWING = 12


# TP / SL configurations
EXITS = [
    ("TP050_SL050", 0.50, 0.50),
    ("TP075_SL050", 0.75, 0.50),
    ("TP100_SL050", 1.00, 0.50),
    ("TP100_SL075", 1.00, 0.75),
    ("TP125_SL075", 1.25, 0.75),
    ("TP150_SL100", 1.50, 1.00),
    ("TP150_SL125", 1.50, 1.25),
    ("TP200_SL100", 2.00, 1.00),
]


# ============================================================
# DOWNLOAD BTC DATA
# ============================================================

def download():

    print("🚀 BTC Price-Action Liquidity Sweep Backtest Started!")
    print()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)

    current = start
    rows = []

    print("📥 Downloading 6 months of BTC 15M data...")
    print()

    while current < end:

        chunk_end = min(
            current + timedelta(minutes=3750),
            end
        )

        print(
            f"Downloading: "
            f"{current:%Y-%m-%d %H:%M} "
            f"to "
            f"{chunk_end:%Y-%m-%d %H:%M}"
        )

        try:

            response = requests.get(
                f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles",
                params={
                    "granularity": GRANULARITY,
                    "start": current.isoformat(),
                    "end": chunk_end.isoformat(),
                },
                timeout=30,
                headers={
                    "User-Agent": "BTC-Backtest/1.0"
                }
            )

            response.raise_for_status()

            data = response.json()

            rows.extend(data)

        except Exception as e:

            print("Download error:", e)

        current = chunk_end + timedelta(minutes=15)

    if not rows:
        raise RuntimeError("No BTC data downloaded.")

    df = pd.DataFrame(
        rows,
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

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = (
        df
        .dropna()
        .drop_duplicates("time")
        .sort_values("time")
        .reset_index(drop=True)
    )

    print()
    print(
        f"✅ Total candles downloaded: "
        f"{len(df):,}"
    )

    print(
        "First candle:",
        df["time"].iloc[0]
    )

    print(
        "Last candle:",
        df["time"].iloc[-1]
    )

    print()

    return df


# ============================================================
# EMA
# ============================================================

def ema(series, period):

    return series.ewm(
        span=period,
        adjust=False
    ).mean()


# ============================================================
# INDICATORS
# ============================================================

def indicators(df):

    print("📊 Calculating 15M indicators...")

    d = df.copy()

    # EMA
    d["ema9"] = ema(
        d["close"],
        9
    )

    d["ema21"] = ema(
        d["close"],
        21
    )

    d["ema50"] = ema(
        d["close"],
        50
    )

    # RSI
    delta = d["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    d["rsi"] = (
        100 -
        100 / (1 + rs)
    )

    d["rsi"] = d["rsi"].fillna(50)

    # ========================================================
    # MACD
    # ========================================================

    macd_fast = ema(
        d["close"],
        12
    )

    macd_slow = ema(
        d["close"],
        26
    )

    d["macd"] = (
        macd_fast -
        macd_slow
    )

    d["macd_signal"] = ema(
        d["macd"],
        9
    )

    # IMPORTANT:
    # Do NOT call this "hist".
    # pandas Series has a .hist() method.
    d["macd_hist"] = (
        d["macd"] -
        d["macd_signal"]
    )

    # ========================================================
    # ATR
    # ========================================================

    previous_close = (
        d["close"].shift(1)
    )

    d["tr"] = pd.concat(
        [
            d["high"] - d["low"],

            (
                d["high"] -
                previous_close
            ).abs(),

            (
                d["low"] -
                previous_close
            ).abs()
        ],
        axis=1
    ).max(axis=1)

    d["atr"] = (
        d["tr"]
        .rolling(14)
        .mean()
    )

    # ========================================================
    # VOLUME
    # ========================================================

    d["volume_ratio"] = (
        d["volume"] /
        d["volume"]
        .rolling(20)
        .mean()
    )

    # ========================================================
    # CANDLE
    # ========================================================

    d["body"] = (
        d["close"] -
        d["open"]
    ).abs()

    d["candle_range"] = (
        d["high"] -
        d["low"]
    ).replace(
        0,
        np.nan
    )

    d["body_atr"] = (
        d["body"] /
        d["atr"]
    )

    d["body_ratio"] = (
        d["body"] /
        d["candle_range"]
    )

    # ========================================================
    # DISTANCE FROM EMA21
    # ========================================================

    d["ema21_distance_atr"] = (
        (
            d["close"] -
            d["ema21"]
        ).abs()
        /
        d["atr"]
    )

    return d


# ============================================================
# 1H MARKET REGIME
# ============================================================

def build_regime(df):

    print("📊 Building completed 1H regime...")

    hourly = (
        df
        .set_index("time")
        .resample(
            "1h",
            label="right",
            closed="right"
        )
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum"
            }
        )
        .dropna()
        .reset_index()
    )

    hourly["ema20"] = ema(
        hourly["close"],
        20
    )

    hourly["ema50"] = ema(
        hourly["close"],
        50
    )

    hourly["ema100"] = ema(
        hourly["close"],
        100
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

        # Strong up
        if (
            row["ema20"] >
            row["ema50"] >
            row["ema100"]
            and
            row["slope20"] > 0
            and
            row["slope50"] > 0
            and
            row["close"] >
            row["ema20"]
        ):

            return "STRONG_UP"

        # Strong down
        if (
            row["ema20"] <
            row["ema50"] <
            row["ema100"]
            and
            row["slope20"] < 0
            and
            row["slope50"] < 0
            and
            row["close"] <
            row["ema20"]
        ):

            return "STRONG_DOWN"

        # Normal up
        if (
            row["ema20"] >
            row["ema50"]
            and
            row["close"] >
            row["ema50"]
        ):

            return "NORMAL_UP"

        # Normal down
        if (
            row["ema20"] <
            row["ema50"]
            and
            row["close"] <
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


# ============================================================
# PREPARE DATA
# ============================================================

def prepare(df):

    regime = build_regime(df)

    d = df.copy()

    d["time"] = (
        pd.to_datetime(
            d["time"],
            utc=True
        )
        .astype(
            "datetime64[ns, UTC]"
        )
    )

    d = pd.merge_asof(
        d.sort_values("time"),
        regime.sort_values(
            "available_time"
        ),
        left_on="time",
        right_on="available_time",
        direction="backward"
    )

    # Previous 12 completed candles
    d["swing_high"] = (
        d["high"]
        .shift(1)
        .rolling(SWING)
        .max()
    )

    d["swing_low"] = (
        d["low"]
        .shift(1)
        .rolling(SWING)
        .min()
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
        "ema21_distance_atr",
        "regime",
        "swing_high",
        "swing_low"
    ]

    d = (
        d
        .dropna(
            subset=required
        )
        .reset_index(
            drop=True
        )
    )

    return d


# ============================================================
# FIND LIQUIDITY SWEEP SETUP
# ============================================================

def find_setup(df, i):

    if i < SWING + 5:
        return None

    row = df.iloc[i]

    atr = float(
        row["atr"]
    )

    if (
        not np.isfinite(atr)
        or atr <= 0
    ):
        return None

    if row["regime"] == "SIDEWAYS":
        return None

    # Don't chase extended move
    if (
        row["ema21_distance_atr"]
        > 1.25
    ):
        return None

    # ========================================================
    # BUY
    # ========================================================

    if row["regime"] in (
        "STRONG_UP",
        "NORMAL_UP"
    ):

        buy_context = (
            row["ema9"] >
            row["ema21"] >
            row["ema50"]

            and
            row["close"] >
            row["ema21"]

            and
            row["macd_hist"] > 0

            and
            45 <= row["rsi"] <= 68

            and
            row["volume_ratio"] >= 0.80

            and
            row["close"] >
            row["open"]

            and
            row["body_atr"] >= 0.25

            and
            row["body_ratio"] >= 0.35
        )

        # Liquidity sweep below recent low
        sweep = (
            row["low"]
            <
            row["swing_low"]
            -
            0.10 * atr

            and

            row["close"]
            >
            row["swing_low"]
            +
            0.05 * atr
        )

        if (
            buy_context
            and
            sweep
        ):

            return {
                "direction": "BUY",
                "setup_index": i,
                "swing_level": float(
                    row["swing_low"]
                ),
                "setup_high": float(
                    row["high"]
                ),
                "setup_low": float(
                    row["low"]
                ),
                "regime": row["regime"]
            }

    # ========================================================
    # SELL
    # ========================================================

    if row["regime"] in (
        "STRONG_DOWN",
        "NORMAL_DOWN"
    ):

        sell_context = (
            row["ema9"] <
            row["ema21"] <
            row["ema50"]

            and
            row["close"] <
            row["ema21"]

            and
            row["macd_hist"] < 0

            and
            32 <= row["rsi"] <= 55

            and
            row["volume_ratio"] >= 0.80

            and
            row["close"] <
            row["open"]

            and
            row["body_atr"] >= 0.25

            and
            row["body_ratio"] >= 0.35
        )

        # Liquidity sweep above recent high
        sweep = (
            row["high"]
            >
            row["swing_high"]
            +
            0.10 * atr

            and

            row["close"]
            <
            row["swing_high"]
            -
            0.05 * atr
        )

        if (
            sell_context
            and
            sweep
        ):

            return {
                "direction": "SELL",
                "setup_index": i,
                "swing_level": float(
                    row["swing_high"]
                ),
                "setup_high": float(
                    row["high"]
                ),
                "setup_low": float(
                    row["low"]
                ),
                "regime": row["regime"]
            }

    return None


# ============================================================
# CONFIRM ENTRY
# ============================================================

def confirm_entry(df, setup):

    i = setup["setup_index"]

    direction = setup["direction"]

    setup_close = float(
        df.iloc[i]["close"]
    )

    swept_level = (
        setup["swing_level"]
    )

    # Wait max 2 candles
    for j in range(
        i + 1,
        min(
            len(df),
            i + 3
        )
    ):

        row = df.iloc[j]

        # BUY confirmation
        if direction == "BUY":

            # Setup cancelled
            if (
                row["close"]
                <
                swept_level
            ):
                return None

            confirmed = (
                row["close"]
                >
                setup["setup_high"]
                or
                (
                    row["close"]
                    >
                    setup_close
                    and
                    row["close"]
                    >
                    row["open"]
                )
            )

            if confirmed:

                return {
                    "entry_index": j,
                    "entry_price": float(
                        row["close"]
                    ),
                    "entry_atr": float(
                        row["atr"]
                    ),
                    "direction": direction,
                    "setup_index": i,
                    "regime": setup["regime"]
                }

        # SELL confirmation
        else:

            if (
                row["close"]
                >
                swept_level
            ):
                return None

            confirmed = (
                row["close"]
                <
                setup["setup_low"]
                or
                (
                    row["close"]
                    <
                    setup_close
                    and
                    row["close"]
                    <
                    row["open"]
                )
            )

            if confirmed:

                return {
                    "entry_index": j,
                    "entry_price": float(
                        row["close"]
                    ),
                    "entry_atr": float(
                        row["atr"]
                    ),
                    "direction": direction,
                    "setup_index": i,
                    "regime": setup["regime"]
                }

    return None


# ============================================================
# SIMULATE TRADE
# ============================================================

def simulate_trade(
    df,
    entry,
    tp_r,
    sl_r
):

    entry_index = (
        entry["entry_index"]
    )

    direction = (
        entry["direction"]
    )

    entry_price = (
        entry["entry_price"]
    )

    atr = entry["entry_atr"]

    if (
        not np.isfinite(entry_price)
        or
        not np.isfinite(atr)
        or
        atr <= 0
    ):
        return None

    setup_row = df.iloc[
        entry["setup_index"]
    ]

    # ========================================================
    # BUY
    # ========================================================

    if direction == "BUY":

        structural_stop = (
            float(
                setup_row[
                    "swing_low"
                ]
            )
            -
            0.15 * atr
        )

        atr_stop = (
            entry_price
            -
            sl_r * atr
        )

        stop = max(
            structural_stop,
            atr_stop
        )

        risk = (
            entry_price -
            stop
        )

        if risk <= 0:
            return None

        target = (
            entry_price
            +
            tp_r * risk
        )

    # ========================================================
    # SELL
    # ========================================================

    else:

        structural_stop = (
            float(
                setup_row[
                    "swing_high"
                ]
            )
            +
            0.15 * atr
        )

        atr_stop = (
            entry_price
            +
            sl_r * atr
        )

        stop = min(
            structural_stop,
            atr_stop
        )

        risk = (
            stop -
            entry_price
        )

        if risk <= 0:
            return None

        target = (
            entry_price
            -
            tp_r * risk
        )

    end_index = min(
        len(df) - 1,
        entry_index + MAX_HOLD
    )

    mfe = 0.0
    mae = 0.0

    result = "TIMEOUT"

    exit_price = float(
        df.iloc[
            end_index
        ]["close"]
    )

    # ========================================================
    # CHECK FUTURE CANDLES
    # ========================================================

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
                row["high"] >= target
            )

            hit_sl = (
                row["low"] <= stop
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
                row["low"] <= target
            )

            hit_sl = (
                row["high"] >= stop
            )

        # Conservative:
        # if both happen in same candle,
        # assume SL happens first.
        if (
            hit_sl
            and
            hit_tp
        ):

            result = "SL"

            exit_price = stop

            break

        if hit_sl:

            result = "SL"

            exit_price = stop

            break

        if hit_tp:

            result = "TP"

            exit_price = target

            break

    # ========================================================
    # RETURN
    # ========================================================

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

    risk_percent = (
        risk /
        entry_price
    )

    if risk_percent > 0:

        r_multiple = (
            net_return /
            risk_percent
        )

    else:

        r_multiple = np.nan

    return {
        "direction": direction,
        "entry_time": df.iloc[
            entry_index
        ]["time"],
        "entry_price": entry_price,
        "stop": stop,
        "target": target,
        "result": result,
        "net_return": net_return,
        "r": r_multiple,
        "mfe": mfe,
        "mae": mae,
        "regime": entry["regime"]
    }


# ============================================================
# RUN BACKTEST
# ============================================================

def run(
    df,
    start_time,
    end_time,
    tp_r,
    sl_r
):

    trades = []

    next_allowed = -1

    start_index = max(
        150,
        SWING + 5
    )

    for i in range(
        start_index,
        len(df)
    ):

        if i < next_allowed:
            continue

        current_time = (
            df.iloc[i]["time"]
        )

        if (
            current_time <
            start_time
        ):
            continue

        if (
            current_time >=
            end_time
        ):
            continue

        setup = find_setup(
            df,
            i
        )

        if setup is None:
            continue

        entry = confirm_entry(
            df,
            setup
        )

        if entry is None:
            continue

        entry_index = (
            entry["entry_index"]
        )

        entry_time = (
            df.iloc[
                entry_index
            ]["time"]
        )

        if entry_time >= end_time:
            continue

        trade = simulate_trade(
            df,
            entry,
            tp_r,
            sl_r
        )

        if trade is None:
            continue

        trades.append(trade)

        # Cooldown
        next_allowed = (
            entry_index +
            COOLDOWN
        )

    return trades


# ============================================================
# SUMMARY
# ============================================================

def summary(trades):

    count = len(trades)

    if count == 0:

        return {
            "trades": 0,
            "tp_pct": 0,
            "sl_pct": 0,
            "timeout_pct": 0,
            "return_pct": 0,
            "avg_r": 0,
            "total_r": 0,
            "mfe": 0,
            "mae": 0
        }

    tp = sum(
        t["result"] == "TP"
        for t in trades
    )

    sl = sum(
        t["result"] == "SL"
        for t in trades
    )

    timeout = (
        count -
        tp -
        sl
    )

    return {
        "trades": count,

        "tp_pct":
            100 * tp / count,

        "sl_pct":
            100 * sl / count,

        "timeout_pct":
            100 * timeout / count,

        "return_pct":
            100 * sum(
                t["net_return"]
                for t in trades
            ),

        "avg_r":
            np.mean([
                t["r"]
                for t in trades
            ]),

        "total_r":
            sum(
                t["r"]
                for t in trades
            ),

        "mfe":
            100 * np.mean([
                t["mfe"]
                for t in trades
            ]),

        "mae":
            100 * np.mean([
                t["mae"]
                for t in trades
            ])
    }


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    name,
    stats
):

    print(
        f"{name:16s} | "
        f"Trades={stats['trades']} | "
        f"TP={stats['tp_pct']:.1f}% | "
        f"SL={stats['sl_pct']:.1f}% | "
        f"Timeout={stats['timeout_pct']:.1f}% | "
        f"Return={stats['return_pct']:.2f}% | "
        f"AvgR={stats['avg_r']:.3f} | "
        f"TotalR={stats['total_r']:.2f} | "
        f"MFE={stats['mfe']:.3f}% | "
        f"MAE={stats['mae']:.3f}%"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = download()

    df = indicators(df)

    df = prepare(df)

    print(
        f"Final usable candles: "
        f"{len(df):,}"
    )

    print()

    data_start = (
        df["time"].iloc[0]
    )

    train_end = (
        data_start +
        timedelta(
            days=TRAIN_DAYS
        )
    )

    validation_end = min(
        train_end +
        timedelta(
            days=VALIDATION_DAYS
        ),
        df["time"].iloc[-1]
    )

    print(
        f"Training: "
        f"{data_start} → {train_end}"
    )

    print(
        f"Validation: "
        f"{train_end} → {validation_end}"
    )

    print()

    # ========================================================
    # TRAINING
    # ========================================================

    print("=" * 80)
    print(
        "TRAINING PRICE-ACTION "
        "CONFIGURATIONS"
    )
    print("=" * 80)

    training_rows = []

    for (
        exit_name,
        tp_r,
        sl_r
    ) in EXITS:

        trades = run(
            df,
            data_start,
            train_end,
            tp_r,
            sl_r
        )

        stats = summary(
            trades
        )

        print_summary(
            exit_name,
            stats
        )

        training_rows.append({
            "exit": exit_name,
            "tp_r": tp_r,
            "sl_r": sl_r,
            **stats
        })

    train_df = pd.DataFrame(
        training_rows
    )

    # Only configurations that
    # genuinely make money in training
    positive = train_df[
        (
            train_df["trades"]
            >= 20
        )
        &
        (
            train_df["return_pct"]
            > 0
        )
        &
        (
            train_df["avg_r"]
            > 0
        )
    ].copy()

    print()

    print("=" * 80)
    print(
        "POSITIVE TRAINING "
        "CONFIGURATIONS"
    )
    print("=" * 80)

    if positive.empty:

        print(
            "❌ No positive training configuration."
        )

        print(
            "The liquidity-sweep setup "
            "did not demonstrate a "
            "positive training edge."
        )

        print(
            "Validation is intentionally "
            "NOT forced."
        )

        return

    positive = positive.sort_values(
        [
            "avg_r",
            "return_pct"
        ],
        ascending=False
    )

    print(
        positive[
            [
                "exit",
                "trades",
                "return_pct",
                "avg_r",
                "total_r"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print()

    print("=" * 80)
    print(
        "UNSEEN VALIDATION"
    )
    print("=" * 80)

    validation_rows = []

    validation_trades = []

    for _, selected in positive.iterrows():

        trades = run(
            df,
            train_end,
            validation_end,
            float(
                selected["tp_r"]
            ),
            float(
                selected["sl_r"]
            )
        )

        stats = summary(
            trades
        )

        print_summary(
            selected["exit"],
            stats
        )

        validation_rows.append({
            "exit":
                selected["exit"],

            "tp_r":
                selected["tp_r"],

            "sl_r":
                selected["sl_r"],

            **stats
        })

        for trade in trades:

            trade["exit_config"] = (
                selected["exit"]
            )

        validation_trades.extend(
            trades
        )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    pd.DataFrame(
        validation_rows
    ).to_csv(
        "btc_liquidity_sweep_validation.csv",
        index=False
    )

    pd.DataFrame(
        validation_trades
    ).to_csv(
        "btc_liquidity_sweep_trades.csv",
        index=False
    )

    print()

    print("💾 Saved:")

    print(
        "btc_liquidity_sweep_validation.csv"
    )

    print(
        "btc_liquidity_sweep_trades.csv"
    )

    # ========================================================
    # FINAL CHECK
    # ========================================================

    if validation_rows:

        validation_df = (
            pd.DataFrame(
                validation_rows
            )
            .sort_values(
                [
                    "avg_r",
                    "return_pct"
                ],
                ascending=False
            )
        )

        best = (
            validation_df.iloc[0]
        )

        print()

        print("=" * 80)
        print(
            "FINAL VALIDATION CHECK"
        )
        print("=" * 80)

        print(
            f"Best validation: "
            f"{best['exit']}"
        )

        print(
            f"Trades: "
            f"{int(best['trades'])}"
        )

        print(
            f"AvgR: "
            f"{best['avg_r']:.3f}"
        )

        print(
            f"TotalR: "
            f"{best['total_r']:.2f}"
        )

        print(
            f"Return: "
            f"{best['return_pct']:.2f}%"
        )

        if (
            best["trades"] >= 20
            and
            best["avg_r"] > 0
            and
            best["return_pct"] > 0
        ):

            print(
                "⚠️ Positive unseen "
                "validation result found."
            )

            print(
                "Paper-test before "
                "any live deployment."
            )

        else:

            print(
                "❌ No positive unseen "
                "validation edge."
            )

            print(
                "Do NOT move this strategy "
                "to the live bot."
            )


if __name__ == "__main__":
    main()

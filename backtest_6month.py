import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone

# =========================================================
# BTC 15M MARKET-STATE / OPPORTUNITY SCAN
# =========================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900

TOTAL_DAYS = 183
CHUNK_DAYS = 3

EMA_FAST = 20
EMA_SLOW = 50
ATR_PERIOD = 14
RSI_PERIOD = 14
LOOKBACK = 20


# =========================================================
# DOWNLOAD
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

    print("\n📥 Downloading BTC 15M data...")

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
        raise RuntimeError(
            "No data downloaded."
        )

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

    print(
        f"✅ Total candles: {len(df)}"
    )

    return df


# =========================================================
# INDICATORS
# =========================================================

def calculate_indicators(df):

    d = df.copy()

    print(
        "⚙️ Calculating market conditions..."
    )

    # EMA
    d["ema20"] = d["close"].ewm(
        span=EMA_FAST,
        adjust=False
    ).mean()

    d["ema50"] = d["close"].ewm(
        span=EMA_SLOW,
        adjust=False
    ).mean()

    # ATR
    prev_close = d["close"].shift(1)

    tr1 = (
        d["high"] - d["low"]
    )

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

    # ATR percentage
    d["atr_pct"] = (
        d["atr"]
        / d["close"]
        * 100
    )

    # RSI
    delta = d["close"].diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .rolling(RSI_PERIOD)
        .mean()
    )

    avg_loss = (
        loss
        .rolling(RSI_PERIOD)
        .mean()
    )

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    d["rsi"] = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # Volume
    d["volume_ma"] = (
        d["volume"]
        .rolling(20)
        .mean()
    )

    d["volume_ratio"] = (
        d["volume"]
        / d["volume_ma"]
    )

    # Candle
    d["body"] = (
        d["close"]
        - d["open"]
    ).abs()

    d["range"] = (
        d["high"]
        - d["low"]
    )

    d["body_ratio"] = np.where(
        d["range"] > 0,
        d["body"] / d["range"],
        0
    )

    # Recent range
    d["range_high"] = (
        d["high"]
        .shift(1)
        .rolling(LOOKBACK)
        .max()
    )

    d["range_low"] = (
        d["low"]
        .shift(1)
        .rolling(LOOKBACK)
        .min()
    )

    d["range_width_pct"] = (
        (
            d["range_high"]
            - d["range_low"]
        )
        / d["close"]
        * 100
    )

    # EMA distance
    d["ema_distance_atr"] = (
        (
            d["close"]
            - d["ema20"]
        )
        / d["atr"]
    )

    # Trend strength
    d["ema_gap_atr"] = (
        (
            d["ema20"]
            - d["ema50"]
        )
        / d["atr"]
    )

    # =====================================================
    # MARKET STATE
    # =====================================================

    d["state"] = "OTHER"

    # Strong uptrend
    d.loc[
        (
            (d["close"] > d["ema20"]) &
            (d["ema20"] > d["ema50"]) &
            (d["ema_gap_atr"] >= 0.50)
        ),
        "state"
    ] = "STRONG_UP"

    # Normal uptrend
    d.loc[
        (
            (d["close"] > d["ema20"]) &
            (d["ema20"] > d["ema50"]) &
            (d["ema_gap_atr"] < 0.50)
        ),
        "state"
    ] = "NORMAL_UP"

    # Strong downtrend
    d.loc[
        (
            (d["close"] < d["ema20"]) &
            (d["ema20"] < d["ema50"]) &
            (d["ema_gap_atr"] <= -0.50)
        ),
        "state"
    ] = "STRONG_DOWN"

    # Normal downtrend
    d.loc[
        (
            (d["close"] < d["ema20"]) &
            (d["ema20"] < d["ema50"]) &
            (d["ema_gap_atr"] > -0.50)
        ),
        "state"
    ] = "NORMAL_DOWN"

    # Range / sideways
    d.loc[
        (
            d["range_width_pct"] < 2.0
        ),
        "state"
    ] = "RANGE"

    # High volatility
    d.loc[
        (
            d["atr_pct"] >= 0.60
        ),
        "state"
    ] = "HIGH_VOL"

    # Low volatility
    d.loc[
        (
            d["atr_pct"] <= 0.20
        ),
        "state"
    ] = "LOW_VOL"

    d = d.dropna().reset_index(
        drop=True
    )

    print(
        f"✅ Final usable candles: {len(d)}"
    )

    return d


# =========================================================
# FUTURE MARKET MOVEMENT
# =========================================================

def calculate_forward_returns(d):

    horizons = {
        15: 1,
        30: 2,
        60: 4,
        120: 8
    }

    for minutes, candles in horizons.items():

        d[
            f"future_{minutes}"
        ] = (
            d["close"].shift(-candles)
            / d["close"]
            - 1
        )

    # Maximum favorable/adverse movement
    # over next 3 hours

    future_highs = []

    future_lows = []

    close_values = (
        d["close"].values
    )

    high_values = (
        d["high"].values
    )

    low_values = (
        d["low"].values
    )

    n = len(d)

    for i in range(n):

        end = min(
            i + 13,
            n
        )

        if i + 1 >= n:

            future_highs.append(
                np.nan
            )

            future_lows.append(
                np.nan
            )

            continue

        future_highs.append(
            np.max(
                high_values[
                    i + 1:end
                ]
            )
        )

        future_lows.append(
            np.min(
                low_values[
                    i + 1:end
                ]
            )
        )

    d["future_high_3h"] = (
        future_highs
    )

    d["future_low_3h"] = (
        future_lows
    )

    d["mfe_up_3h"] = (
        d["future_high_3h"]
        / d["close"]
        - 1
    )

    d["mfe_down_3h"] = (
        d["future_low_3h"]
        / d["close"]
        - 1
    )

    return d


# =========================================================
# STATE ANALYSIS
# =========================================================

def analyze_states(
    d,
    start_time,
    end_time,
    title
):

    x = d[
        (
            d["time"] >= start_time
        )
        &
        (
            d["time"] < end_time
        )
    ].copy()

    print(
        "\n"
        "================================================"
    )

    print(title)

    print(
        "================================================"
    )

    states = sorted(
        x["state"].dropna().unique()
    )

    rows = []

    for state in states:

        s = x[
            x["state"] == state
        ].copy()

        if len(s) < 20:
            continue

        row = {
            "state": state,
            "samples": len(s)
        }

        for minutes in [
            15,
            30,
            60,
            120
        ]:

            col = (
                f"future_{minutes}"
            )

            values = s[col].dropna()

            if len(values) == 0:
                continue

            row[
                f"avg_{minutes}"
            ] = values.mean() * 100

            row[
                f"up_{minutes}"
            ] = (
                values > 0
            ).mean() * 100

            row[
                f"down_{minutes}"
            ] = (
                values < 0
            ).mean() * 100

        mfe_up = (
            s["mfe_up_3h"]
            .dropna()
        )

        mfe_down = (
            s["mfe_down_3h"]
            .dropna()
        )

        row["avg_mfe_up_3h"] = (
            mfe_up.mean() * 100
        )

        row["avg_mfe_down_3h"] = (
            mfe_down.mean() * 100
        )

        # Probability of useful movement
        row["up_30_3h"] = (
            mfe_up >= 0.003
        ).mean() * 100

        row["down_30_3h"] = (
            mfe_down <= -0.003
        ).mean() * 100

        row["up_50_3h"] = (
            mfe_up >= 0.005
        ).mean() * 100

        row["down_50_3h"] = (
            mfe_down <= -0.005
        ).mean() * 100

        rows.append(row)

    result = pd.DataFrame(rows)

    if result.empty:

        print(
            "No sufficient data."
        )

        return result

    for _, r in result.iterrows():

        print(
            f"\n📌 {r['state']}"
        )

        print(
            f"Samples: {int(r['samples'])}"
        )

        for minutes in [
            15,
            30,
            60,
            120
        ]:

            avg = r.get(
                f"avg_{minutes}",
                np.nan
            )

            up = r.get(
                f"up_{minutes}",
                np.nan
            )

            down = r.get(
                f"down_{minutes}",
                np.nan
            )

            print(
                f"{minutes}M | "
                f"Avg {avg:+.3f}% | "
                f"UP {up:.1f}% | "
                f"DOWN {down:.1f}%"
            )

        print(
            f"3H favorable +0.30%: "
            f"{r['up_30_3h']:.1f}%"
        )

        print(
            f"3H favorable -0.30%: "
            f"{r['down_30_3h']:.1f}%"
        )

        print(
            f"3H favorable +0.50%: "
            f"{r['up_50_3h']:.1f}%"
        )

        print(
            f"3H favorable -0.50%: "
            f"{r['down_50_3h']:.1f}%"
        )

    return result


# =========================================================
# DETAILED CONDITION ANALYSIS
# =========================================================

def analyze_conditions(
    d,
    start_time,
    end_time
):

    x = d[
        (
            d["time"] >= start_time
        )
        &
        (
            d["time"] < end_time
        )
    ].copy()

    print(
        "\n"
        "================================================"
    )

    print(
        "CONDITION SCAN"
    )

    print(
        "================================================"
    )

    conditions = {

        "RSI_LT35":
            x["rsi"] < 35,

        "RSI_35_45":
            (
                (x["rsi"] >= 35)
                &
                (x["rsi"] < 45)
            ),

        "RSI_45_55":
            (
                (x["rsi"] >= 45)
                &
                (x["rsi"] < 55)
            ),

        "RSI_55_65":
            (
                (x["rsi"] >= 55)
                &
                (x["rsi"] < 65)
            ),

        "RSI_GT65":
            x["rsi"] > 65,

        "HIGH_VOLUME":
            x["volume_ratio"] >= 1.2,

        "LOW_VOLUME":
            x["volume_ratio"] <= 0.8,

        "PRICE_ABOVE_EMA":
            x["ema_distance_atr"] > 0.5,

        "PRICE_BELOW_EMA":
            x["ema_distance_atr"] < -0.5,

        "NEAR_EMA":
            x["ema_distance_atr"].abs() < 0.25,

        "STRONG_BODY":
            x["body_ratio"] >= 0.60,

        "WEAK_BODY":
            x["body_ratio"] < 0.30,

        "HIGH_VOLATILITY":
            x["atr_pct"] >= 0.60,

        "LOW_VOLATILITY":
            x["atr_pct"] <= 0.20
    }

    rows = []

    for name, mask in conditions.items():

        s = x[mask].copy()

        if len(s) < 30:
            continue

        future = (
            s["future_60"]
            .dropna()
        )

        if len(future) == 0:
            continue

        rows.append({
            "condition": name,
            "samples": len(s),
            "avg_60m": (
                future.mean() * 100
            ),
            "up_60m": (
                (future > 0)
                .mean() * 100
            ),
            "down_60m": (
                (future < 0)
                .mean() * 100
            ),
            "up_30_3h": (
                (
                    s["mfe_up_3h"]
                    >= 0.003
                )
                .mean()
                * 100
            ),
            "down_30_3h": (
                (
                    s["mfe_down_3h"]
                    <= -0.003
                )
                .mean()
                * 100
            )
        })

    result = pd.DataFrame(rows)

    if result.empty:
        print(
            "No conditions with enough data."
        )
        return result

    result = result.sort_values(
        "avg_60m",
        ascending=False
    )

    print(
        "\n📈 BEST CONDITIONS FOR UP MOVEMENT"
    )

    for _, r in result.head(7).iterrows():

        print(
            f"{r['condition']} | "
            f"N={int(r['samples'])} | "
            f"60M Avg={r['avg_60m']:+.3f}% | "
            f"UP={r['up_60m']:.1f}% | "
            f"+0.30% in 3H="
            f"{r['up_30_3h']:.1f}%"
        )

    print(
        "\n📉 BEST CONDITIONS FOR DOWN MOVEMENT"
    )

    result_down = result.sort_values(
        "avg_60m",
        ascending=True
    )

    for _, r in result_down.head(7).iterrows():

        print(
            f"{r['condition']} | "
            f"N={int(r['samples'])} | "
            f"60M Avg={r['avg_60m']:+.3f}% | "
            f"DOWN={r['down_60m']:.1f}% | "
            f"-0.30% in 3H="
            f"{r['down_30_3h']:.1f}%"
        )

    return result


# =========================================================
# SAVE
# =========================================================

def main():

    print(
        "\n🚀 BTC 15M "
        "Market-State / Opportunity Scan\n"
    )

    d = download_data()

    d = calculate_indicators(d)

    d = calculate_forward_returns(d)

    first_time = d["time"].iloc[0]
    last_time = d["time"].iloc[-1]

    train_end = (
        first_time
        + pd.Timedelta(
            days=120
        )
    )

    validation_end = last_time

    print(
        "\n🧠 Training:",
        first_time,
        "→",
        train_end
    )

    print(
        "🧪 Validation:",
        train_end,
        "→",
        validation_end
    )

    # =====================================================
    # TRAINING
    # =====================================================

    train_states = analyze_states(
        d,
        first_time,
        train_end,
        "TRAINING MARKET STATES"
    )

    train_conditions = analyze_conditions(
        d,
        first_time,
        train_end
    )

    # =====================================================
    # VALIDATION
    # =====================================================

    validation_states = analyze_states(
        d,
        train_end,
        validation_end,
        "VALIDATION MARKET STATES"
    )

    validation_conditions = analyze_conditions(
        d,
        train_end,
        validation_end
    )

    # =====================================================
    # SAVE
    # =====================================================

    if not train_states.empty:

        train_states.to_csv(
            "btc_market_state_training.csv",
            index=False
        )

    if not validation_states.empty:

        validation_states.to_csv(
            "btc_market_state_validation.csv",
            index=False
        )

    if not train_conditions.empty:

        train_conditions.to_csv(
            "btc_market_conditions_training.csv",
            index=False
        )

    if not validation_conditions.empty:

        validation_conditions.to_csv(
            "btc_market_conditions_validation.csv",
            index=False
        )

    print(
        "\n💾 Results saved."
    )

    print(
        "\n"
        "================================================"
    )

    print(
        "IMPORTANT"
    )

    print(
        "================================================"
    )

    print(
        "\nএই test এখন কোনো BUY/SELL strategy "
        "নির্বাচন করছে না।"
    )

    print(
        "এটি শুধু দেখাচ্ছে কোন market condition-এ "
        "BTC-এর historical movement ভালো ছিল।"
    )

    print(
        "\n➡️ Output এখানে পাঠাও।"
    )


if __name__ == "__main__":
    main()

import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone

# ============================================================
# BTC 6-MONTH MARKET MOVEMENT / CONDITIONAL ANALYSIS
# ============================================================

PRODUCT = "BTC-USD"
GRANULARITY = 900          # 15 minutes
DAYS = 183

CHUNK_CANDLES = 250
REQUEST_SLEEP = 0.25

HORIZONS = [1, 2, 3, 4, 8]   # 15m, 30m, 45m, 60m, 120m

UP_LEVELS = [0.003, 0.005, 0.0075, 0.010]
DOWN_LEVELS = [0.002, 0.003, 0.005]

MIN_GROUP_SIZE = 100


# ============================================================
# DOWNLOAD
# ============================================================

def download_candles():
    print("\n" + "=" * 70)
    print("🚀 BTC 6-MONTH MARKET MOVEMENT ANALYSIS")
    print("=" * 70)

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=DAYS)

    all_rows = []

    cursor = start_time

    while cursor < end_time:

        chunk_end = min(
            cursor + timedelta(seconds=GRANULARITY * CHUNK_CANDLES),
            end_time
        )

        print(
            f"Downloading: "
            f"{cursor.strftime('%Y-%m-%d %H:%M')} "
            f"to "
            f"{chunk_end.strftime('%Y-%m-%d %H:%M')}"
        )

        url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"

        params = {
            "granularity": GRANULARITY,
            "start": cursor.isoformat(),
            "end": chunk_end.isoformat()
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
            print("❌ Download error:", e)
            time.sleep(2)
            continue

        cursor = chunk_end
        time.sleep(REQUEST_SLEEP)

    if not all_rows:
        raise RuntimeError("No BTC data downloaded.")

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

    df = df.drop_duplicates("time")
    df = df.sort_values("time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)

    print("Total candles:", len(df))
    print("First candle :", df["time"].iloc[0])
    print("Last candle  :", df["time"].iloc[-1])

    return df


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + rs))

    return rsi


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
    df["rsi"] = calculate_rsi(df["close"])

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
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()

    df["tr"] = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    df["atr"] = df["tr"].rolling(14).mean()

    # Volume ratio
    df["volume_avg"] = df["volume"].rolling(20).mean()

    df["volume_ratio"] = (
        df["volume"] /
        df["volume_avg"]
    )

    # Candle body
    df["body"] = (
        (df["close"] - df["open"]).abs()
        / df["atr"]
    )

    # Distance from EMA21 in ATR
    df["ema21_distance_atr"] = (
        (df["close"] - df["ema21"]).abs()
        / df["atr"]
    )

    # EMA alignment
    df["ema_alignment"] = np.select(
        [
            (df["ema9"] > df["ema21"]) &
            (df["ema21"] > df["ema50"]),

            (df["ema9"] < df["ema21"]) &
            (df["ema21"] < df["ema50"])
        ],
        [
            "UP",
            "DOWN"
        ],
        default="MIXED"
    )

    # MACD direction
    df["macd_direction"] = np.select(
        [
            df["macd_hist"] > 0,
            df["macd_hist"] < 0
        ],
        [
            "UP",
            "DOWN"
        ],
        default="FLAT"
    )

    # RSI bucket
    df["rsi_bucket"] = pd.cut(
        df["rsi"],
        bins=[
            -np.inf,
            35,
            45,
            55,
            65,
            np.inf
        ],
        labels=[
            "RSI_<35",
            "RSI_35_45",
            "RSI_45_55",
            "RSI_55_65",
            "RSI_>65"
        ]
    )

    # Volume bucket
    df["volume_bucket"] = pd.cut(
        df["volume_ratio"],
        bins=[
            -np.inf,
            0.8,
            1.2,
            np.inf
        ],
        labels=[
            "VOL_<0.8",
            "VOL_0.8_1.2",
            "VOL_>1.2"
        ]
    )

    # EMA distance bucket
    df["distance_bucket"] = pd.cut(
        df["ema21_distance_atr"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            1.00,
            np.inf
        ],
        labels=[
            "DIST_<0.25ATR",
            "DIST_0.25_0.50ATR",
            "DIST_0.50_1.00ATR",
            "DIST_>1ATR"
        ]
    )

    # Candle body bucket
    df["body_bucket"] = pd.cut(
        df["body"],
        bins=[
            -np.inf,
            0.25,
            0.50,
            np.inf
        ],
        labels=[
            "BODY_<0.25ATR",
            "BODY_0.25_0.50ATR",
            "BODY_>0.50ATR"
        ]
    )

    return df


# ============================================================
# 1H MARKET REGIME
# ============================================================

def build_1h_regime(df):

    hourly = (
        df.set_index("time")
        .resample("1h")
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum"
        })
        .dropna()
        .reset_index()
    )

    hourly["ema20"] = hourly["close"].ewm(
        span=20,
        adjust=False
    ).mean()

    hourly["ema50"] = hourly["close"].ewm(
        span=50,
        adjust=False
    ).mean()

    hourly["ema100"] = hourly["close"].ewm(
        span=100,
        adjust=False
    ).mean()

    hourly["ema20_slope"] = (
        hourly["ema20"] -
        hourly["ema20"].shift(3)
    )

    hourly["ema50_slope"] = (
        hourly["ema50"] -
        hourly["ema50"].shift(3)
    )

    hourly["regime"] = "SIDEWAYS"

    strong_up = (
        (hourly["close"] > hourly["ema20"]) &
        (hourly["ema20"] > hourly["ema50"]) &
        (hourly["ema50"] > hourly["ema100"]) &
        (hourly["ema20_slope"] > 0) &
        (hourly["ema50_slope"] > 0)
    )

    strong_down = (
        (hourly["close"] < hourly["ema20"]) &
        (hourly["ema20"] < hourly["ema50"]) &
        (hourly["ema50"] < hourly["ema100"]) &
        (hourly["ema20_slope"] < 0) &
        (hourly["ema50_slope"] < 0)
    )

    normal_up = (
        (hourly["close"] > hourly["ema20"]) &
        (hourly["ema20"] > hourly["ema50"])
    )

    normal_down = (
        (hourly["close"] < hourly["ema20"]) &
        (hourly["ema20"] < hourly["ema50"])
    )

    hourly.loc[normal_up, "regime"] = "NORMAL_UP"
    hourly.loc[normal_down, "regime"] = "NORMAL_DOWN"
    hourly.loc[strong_up, "regime"] = "STRONG_UP"
    hourly.loc[strong_down, "regime"] = "STRONG_DOWN"

    # Important:
    # Use only COMPLETED 1H candle information.
    hourly["available_time"] = (
        hourly["time"] +
        pd.Timedelta(hours=1)
    )

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    return hourly[
        [
            "available_time",
            "regime"
        ]
    ]


def merge_regime(df, hourly):

    df = df.copy()

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    hourly["available_time"] = pd.to_datetime(
        hourly["available_time"],
        utc=True
    ).astype("datetime64[ns, UTC]")

    df = pd.merge_asof(
        df.sort_values("time"),
        hourly.sort_values("available_time"),
        left_on="time",
        right_on="available_time",
        direction="backward"
    )

    return df


# ============================================================
# FORWARD MOVEMENT ANALYSIS
# ============================================================

def analyze_forward_movement(df):

    print("\n" + "=" * 70)
    print("📊 CALCULATING FUTURE PRICE MOVEMENT")
    print("=" * 70)

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    n = len(df)

    records = []

    for i in range(n):

        if i + max(HORIZONS) >= n:
            break

        entry = close[i]

        if entry <= 0:
            continue

        future_close = {}
        future_high = {}
        future_low = {}

        for h in HORIZONS:

            end = i + h

            future_close[h] = (
                close[end] / entry - 1
            )

            future_high[h] = (
                np.max(high[i + 1:end + 1])
                / entry - 1
            )

            future_low[h] = (
                np.min(low[i + 1:end + 1])
                / entry - 1
            )

        # 120-minute MFE / MAE
        mfe_buy = future_high[8]
        mae_buy = future_low[8]

        mfe_sell = -future_low[8]
        mae_sell = -future_high[8]

        row = df.iloc[i]

        records.append({
            "time": row["time"],
            "close": entry,

            "regime": row["regime"],
            "ema_alignment": row["ema_alignment"],
            "rsi_bucket": str(row["rsi_bucket"]),
            "macd_direction": row["macd_direction"],
            "volume_bucket": str(row["volume_bucket"]),
            "distance_bucket": str(row["distance_bucket"]),
            "body_bucket": str(row["body_bucket"]),

            "rsi": row["rsi"],
            "volume_ratio": row["volume_ratio"],
            "distance_atr": row["ema21_distance_atr"],
            "body_atr": row["body"],

            "ret_15m": future_close[1],
            "ret_30m": future_close[2],
            "ret_45m": future_close[3],
            "ret_60m": future_close[4],
            "ret_120m": future_close[8],

            "mfe_buy_120": mfe_buy,
            "mae_buy_120": mae_buy,

            "mfe_sell_120": mfe_sell,
            "mae_sell_120": mae_sell
        })

    result = pd.DataFrame(records)

    return result


# ============================================================
# OVERALL STATS
# ============================================================

def overall_stats(data):

    print("\n" + "=" * 70)
    print("📈 OVERALL MARKET MOVEMENT")
    print("=" * 70)

    print("Observations:", len(data))

    for horizon, label in [
        ("ret_15m", "15M"),
        ("ret_30m", "30M"),
        ("ret_45m", "45M"),
        ("ret_60m", "60M"),
        ("ret_120m", "120M")
    ]:

        print(
            f"{label:>5} | "
            f"Avg: {data[horizon].mean() * 100:+.3f}% | "
            f"Median: {data[horizon].median() * 100:+.3f}% | "
            f"UP: {(data[horizon] > 0).mean() * 100:.1f}% | "
            f"DOWN: {(data[horizon] < 0).mean() * 100:.1f}%"
        )

    print("\nBUY-side favorable movement:")

    for level in UP_LEVELS:

        hit = (
            data["mfe_buy_120"] >= level
        ).mean() * 100

        print(
            f"  +{level * 100:.2f}% reached: "
            f"{hit:.1f}%"
        )

    print("\nSELL-side favorable movement:")

    for level in UP_LEVELS:

        hit = (
            data["mfe_sell_120"] >= level
        ).mean() * 100

        print(
            f"  -{level * 100:.2f}% reached: "
            f"{hit:.1f}%"
        )

    print("\nBUY-side adverse movement:")

    for level in DOWN_LEVELS:

        hit = (
            data["mae_buy_120"] <= -level
        ).mean() * 100

        print(
            f"  -{level * 100:.2f}% reached: "
            f"{hit:.1f}%"
        )

    print("\nSELL-side adverse movement:")

    for level in DOWN_LEVELS:

        hit = (
            data["mae_sell_120"] <= -level
        ).mean() * 100

        print(
            f"  +{level * 100:.2f}% against SELL: "
            f"{hit:.1f}%"
        )


# ============================================================
# GROUP ANALYSIS
# ============================================================

def group_analysis(data, column, title):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    grouped = (
        data.groupby(column, observed=True)
        .agg(
            samples=("close", "size"),

            avg_60m=("ret_60m", "mean"),

            buy_mfe=("mfe_buy_120", "mean"),

            buy_mae=("mae_buy_120", "mean"),

            sell_mfe=("mfe_sell_120", "mean"),

            sell_mae=("mae_sell_120", "mean")
        )
        .reset_index()
    )

    grouped = grouped[
        grouped["samples"] >= MIN_GROUP_SIZE
    ]

    if grouped.empty:
        print("No groups with enough samples.")
        return

    for _, r in grouped.iterrows():

        buy_05 = (
            data.loc[
                data[column] == r[column],
                "mfe_buy_120"
            ] >= 0.005
        ).mean() * 100

        sell_05 = (
            data.loc[
                data[column] == r[column],
                "mfe_sell_120"
            ] >= 0.005
        ).mean() * 100

        print(
            f"{str(r[column]):<22} | "
            f"N={int(r['samples']):4d} | "
            f"60m={r['avg_60m']*100:+.3f}% | "
            f"BUY +0.50%={buy_05:5.1f}% | "
            f"SELL +0.50%={sell_05:5.1f}%"
        )


# ============================================================
# CONDITION ANALYSIS
# ============================================================

def condition_analysis(data):

    conditions = [
        (
            "STRONG_UP + EMA_UP",
            (
                (data["regime"] == "STRONG_UP") &
                (data["ema_alignment"] == "UP")
            )
        ),

        (
            "STRONG_DOWN + EMA_DOWN",
            (
                (data["regime"] == "STRONG_DOWN") &
                (data["ema_alignment"] == "DOWN")
            )
        ),

        (
            "NORMAL_UP + EMA_UP",
            (
                (data["regime"] == "NORMAL_UP") &
                (data["ema_alignment"] == "UP")
            )
        ),

        (
            "NORMAL_DOWN + EMA_DOWN",
            (
                (data["regime"] == "NORMAL_DOWN") &
                (data["ema_alignment"] == "DOWN")
            )
        ),

        (
            "STRONG_UP + MACD_UP",
            (
                (data["regime"] == "STRONG_UP") &
                (data["macd_direction"] == "UP")
            )
        ),

        (
            "STRONG_DOWN + MACD_DOWN",
            (
                (data["regime"] == "STRONG_DOWN") &
                (data["macd_direction"] == "DOWN")
            )
        ),

        (
            "UP + RSI_45_55",
            (
                (data["ema_alignment"] == "UP") &
                (data["rsi_bucket"] == "RSI_45_55")
            )
        ),

        (
            "DOWN + RSI_45_55",
            (
                (data["ema_alignment"] == "DOWN") &
                (data["rsi_bucket"] == "RSI_45_55")
            )
        )
    ]

    print("\n" + "=" * 70)
    print("🔎 CONDITIONAL MARKET MOVEMENT")
    print("=" * 70)

    for name, mask in conditions:

        subset = data[mask]

        if len(subset) < MIN_GROUP_SIZE:
            continue

        buy_03 = (
            subset["mfe_buy_120"] >= 0.003
        ).mean() * 100

        buy_05 = (
            subset["mfe_buy_120"] >= 0.005
        ).mean() * 100

        buy_075 = (
            subset["mfe_buy_120"] >= 0.0075
        ).mean() * 100

        sell_03 = (
            subset["mfe_sell_120"] >= 0.003
        ).mean() * 100

        sell_05 = (
            subset["mfe_sell_120"] >= 0.005
        ).mean() * 100

        sell_075 = (
            subset["mfe_sell_120"] >= 0.0075
        ).mean() * 100

        print(
            f"\n{name}"
        )

        print(
            f"Samples: {len(subset)}"
        )

        print(
            f"BUY  +0.30%: {buy_03:.1f}% | "
            f"+0.50%: {buy_05:.1f}% | "
            f"+0.75%: {buy_075:.1f}%"
        )

        print(
            f"SELL +0.30%: {sell_03:.1f}% | "
            f"+0.50%: {sell_05:.1f}% | "
            f"+0.75%: {sell_075:.1f}%"
        )

        print(
            f"BUY  avg 120m return: "
            f"{subset['ret_120m'].mean()*100:+.3f}%"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    df = download_candles()

    print("\n📊 Calculating indicators...")

    df = calculate_indicators(df)

    print("📊 Building completed 1H market regime...")

    hourly = build_1h_regime(df)

    print("📊 Merging 1H regime into 15M data...")

    df = merge_regime(df, hourly)

    df = df.dropna(
        subset=[
            "ema50",
            "rsi",
            "macd",
            "atr",
            "volume_ratio",
            "regime"
        ]
    ).reset_index(drop=True)

    print("Final usable candles:", len(df))

    data = analyze_forward_movement(df)

    data = data.dropna().reset_index(drop=True)

    print("Final analysis observations:", len(data))

    # --------------------------------------------------------
    # OVERALL
    # --------------------------------------------------------

    overall_stats(data)

    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    group_analysis(
        data,
        "regime",
        "🏦 1H MARKET REGIME"
    )

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    group_analysis(
        data,
        "ema_alignment",
        "📐 15M EMA ALIGNMENT"
    )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    group_analysis(
        data,
        "rsi_bucket",
        "📊 RSI CONDITIONS"
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    group_analysis(
        data,
        "macd_direction",
        "📈 MACD CONDITIONS"
    )

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    group_analysis(
        data,
        "volume_bucket",
        "🔊 VOLUME CONDITIONS"
    )

    # --------------------------------------------------------
    # DISTANCE FROM EMA21
    # --------------------------------------------------------

    group_analysis(
        data,
        "distance_bucket",
        "📏 DISTANCE FROM EMA21"
    )

    # --------------------------------------------------------
    # CANDLE BODY
    # --------------------------------------------------------

    group_analysis(
        data,
        "body_bucket",
        "🕯️ CANDLE BODY STRENGTH"
    )

    # --------------------------------------------------------
    # COMBINED CONDITIONS
    # --------------------------------------------------------

    condition_analysis(data)

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------

    output_file = "btc_market_movement_analysis.csv"

    data.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        f"Detailed data saved to: {output_file}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is market-movement analysis, "
        "NOT a profitable trading strategy."
    )

    print(
        "No live BUY/SELL rules were changed."
    )


if __name__ == "__main__":
    main()

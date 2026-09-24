import requests
import time
from datetime import datetime, timezone, timedelta

PRODUCT_ID = "BTC-USD"
GRANULARITY = 900
LOOKBACK = 100


def get_candles(start_time, end_time):
    url = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"

    response = requests.get(
        url,
        params={
            "granularity": GRANULARITY,
            "start": start_time.isoformat(),
            "end": end_time.isoformat()
        },
        timeout=30,
        headers={"User-Agent": "BTC-Pattern-Backtest"}
    )

    response.raise_for_status()
    return response.json()


def get_six_month_data():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=183)

    all_candles = []
    current = start
    chunk = timedelta(minutes=15 * 250)

    print("📥 Downloading 6 months of BTC 15M data...")

    while current < end:
        chunk_end = min(current + chunk, end)

        print(
            "Downloading:",
            current.strftime("%Y-%m-%d"),
            "to",
            chunk_end.strftime("%Y-%m-%d")
        )

        try:
            data = get_candles(current, chunk_end)
            all_candles.extend(data)
        except Exception as error:
            print("Download error:", error)

        current = chunk_end
        time.sleep(0.3)

    unique = {}

    for candle in all_candles:
        unique[int(candle[0])] = candle

    candles = list(unique.values())
    candles.sort(key=lambda x: x[0])

    print(f"✅ Total candles collected: {len(candles)}")

    return candles


def ema(values, period):
    multiplier = 2 / (period + 1)
    result = values[0]

    for price in values[1:]:
        result = (price - result) * multiplier + result

    return result


def rsi(closes, period=14):
    changes = [
        closes[i] - closes[i - 1]
        for i in range(1, len(closes))
    ]

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def macd(closes):
    now = ema(closes, 12) - ema(closes, 26)

    previous = closes[:-1]

    if len(previous) >= 26:
        old = ema(previous, 12) - ema(previous, 26)
    else:
        old = now

    return now, old


def analyze_market(history):

    closes = [float(x[4]) for x in history]
    volumes = [float(x[5]) for x in history]

    price = closes[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    current_rsi = rsi(closes)

    macd_now, macd_old = macd(closes)

    recent_volume = sum(volumes[-5:]) / 5
    old_volume = sum(volumes[-20:-5]) / 15

    volume_strong = recent_volume > old_volume * 1.10

    recent_high = max(float(x[2]) for x in history[-20:])
    recent_low = min(float(x[1]) for x in history[-20:])

    resistance_distance = (
        (recent_high - price) / price
    ) * 100

    support_distance = (
        (price - recent_low) / price
    ) * 100

    buy_score = 0
    sell_score = 0

    conditions = []

    # EMA trend
    if ema9 > ema21:
        buy_score += 2
        conditions.append("EMA_BULLISH")

    elif ema9 < ema21:
        sell_score += 2
        conditions.append("EMA_BEARISH")

    # RSI
    if 55 <= current_rsi <= 70:
        buy_score += 2
        conditions.append("RSI_BULLISH")

    elif 30 <= current_rsi <= 45:
        sell_score += 2
        conditions.append("RSI_BEARISH")

    elif current_rsi > 70:
        conditions.append("RSI_OVERBOUGHT")

    elif current_rsi < 30:
        conditions.append("RSI_OVERSOLD")

    else:
        conditions.append("RSI_NEUTRAL")

    # MACD
    if macd_now > 0 and macd_now >= macd_old:
        buy_score += 2
        conditions.append("MACD_BULLISH")

    elif macd_now < 0 and macd_now <= macd_old:
        sell_score += 2
        conditions.append("MACD_BEARISH")

    else:
        conditions.append("MACD_NEUTRAL")

    # Volume
    if volume_strong:
        conditions.append("VOLUME_STRONG")

        if buy_score > sell_score:
            buy_score += 1

        elif sell_score > buy_score:
            sell_score += 1

    else:
        conditions.append("VOLUME_NORMAL")

    # Support / resistance
    if resistance_distance < 0.40:
        buy_score -= 2
        conditions.append("NEAR_RESISTANCE")

    if support_distance < 0.40:
        sell_score -= 2
        conditions.append("NEAR_SUPPORT")

    if buy_score >= 5 and buy_score > sell_score:
        signal = "BUY"

    elif sell_score >= 5 and sell_score > buy_score:
        signal = "SELL"

    else:
        signal = "WAIT"

    return {
        "signal": signal,
        "conditions": conditions,
        "price": price,
        "rsi": current_rsi,
        "ema9": ema9,
        "ema21": ema21
    }


def percent_change(entry, future):
    return ((future - entry) / entry) * 100


def run_pattern_test(candles):

    results = {
        "BUY": {
            "total": 0,
            "up15": 0,
            "up30": 0,
            "up45": 0
        },
        "SELL": {
            "total": 0,
            "down15": 0,
            "down30": 0,
            "down45": 0
        }
    }

    condition_stats = {}

    movement = {
        "BUY": {15: [], 30: [], 45: []},
        "SELL": {15: [], 30: [], 45: []}
    }

    print("🔎 Running 6-month pattern test...")

    for i in range(LOOKBACK, len(candles) - 3):

        history = candles[i - LOOKBACK:i]

        analysis = analyze_market(history)

        signal = analysis["signal"]

        if signal == "WAIT":
            continue

        entry = float(candles[i][4])

        results[signal]["total"] += 1

        condition_key = "|".join(
            sorted(analysis["conditions"])
        )

        if condition_key not in condition_stats:
            condition_stats[condition_key] = {
                "signals": 0,
                "up": 0,
                "down": 0
            }

        condition_stats[condition_key]["signals"] += 1

        for minutes in [15, 30, 45]:

            future_index = i + minutes // 15

            future_price = float(
                candles[future_index][4]
            )

            move = percent_change(
                entry,
                future_price
            )

            movement[signal][minutes].append(move)

            if move > 0:
                condition_stats[condition_key]["up"] += 1

            elif move < 0:
                condition_stats[condition_key]["down"] += 1

            if signal == "BUY":

                if move > 0:
                    results["BUY"][f"up{minutes}"] += 1

            elif signal == "SELL":

                if move < 0:
                    results["SELL"][f"down{minutes}"] += 1

    return results, condition_stats, movement


def average(values):

    if not values:
        return 0

    return sum(values) / len(values)


def print_results(results, condition_stats, movement):

    print()
    print("=" * 65)
    print("📊 BTC 15M — 6 MONTH MARKET PATTERN TEST")
    print("=" * 65)

    for signal in ["BUY", "SELL"]:

        print()
        print(f"📌 {signal} SIGNALS")
        print("-" * 65)

        total = results[signal]["total"]

        print(f"Total signals: {total}")

        for minutes in [15, 30, 45]:

            if signal == "BUY":
                successful = results["BUY"][f"up{minutes}"]
            else:
                successful = results["SELL"][f"down{minutes}"]

            accuracy = (
                successful / total * 100
                if total else 0
            )

            avg_move = average(
                movement[signal][minutes]
            )

            print(
                f"{minutes}M → "
                f"correct direction: "
                f"{successful}/{total} "
                f"({accuracy:.1f}%) | "
                f"average move: {avg_move:.3f}%"
            )

    print()
    print("=" * 65)
    print("🔎 MARKET CONDITION PATTERNS")
    print("=" * 65)

    sorted_conditions = sorted(
        condition_stats.items(),
        key=lambda x: x[1]["signals"],
        reverse=True
    )

    for conditions, data in sorted_conditions[:15]:

        signals = data["signals"]
        up = data["up"]
        down = data["down"]

        up_percent = up / signals * 100
        down_percent = down / signals * 100

        print()
        print(f"Condition: {conditions}")
        print(f"Signals: {signals}")
        print(f"Price UP: {up_percent:.1f}%")
        print(f"Price DOWN: {down_percent:.1f}%")

    print()
    print("=" * 65)
    print("⚠️ This is historical analysis, not a future prediction.")
    print("⚠️ Historical patterns do not guarantee future results.")
    print("=" * 65)


def main():

    print("🚀 BTC 6-Month Pattern Backtest Started!")

    candles = get_six_month_data()

    if len(candles) < 1000:
        raise RuntimeError(
            "Not enough historical data."
        )

    results, conditions, movement = run_pattern_test(
        candles
    )

    print_results(
        results,
        conditions,
        movement
    )


if __name__ == "__main__":
    main()

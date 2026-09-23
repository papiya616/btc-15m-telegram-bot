import requests
import time
from datetime import datetime, timezone, timedelta

PRODUCT_ID = "BTC-USD"
GRANULARITY = 900
CANDLE_LIMIT = 300


def get_candles(start_time, end_time):
    url = f"https://api.exchange.coinbase.com/products/{PRODUCT_ID}/candles"

    params = {
        "granularity": GRANULARITY,
        "start": start_time.isoformat(),
        "end": end_time.isoformat()
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={"User-Agent": "BTC-6-Month-Backtest"}
    )

    response.raise_for_status()
    return response.json()


def get_six_month_data():
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=183)

    all_candles = []
    current = start

    # 250 candles × 15 minutes
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
            candles = get_candles(current, chunk_end)
            all_candles.extend(candles)
        except Exception as e:
            print("Download error:", e)

        current = chunk_end
        time.sleep(0.3)

    # Remove duplicates
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


def calculate_rsi(closes, period=14):
    changes = []

    for i in range(1, len(closes)):
        changes.append(closes[i] - closes[i - 1])

    gains = [max(x, 0) for x in changes]
    losses = [max(-x, 0) for x in changes]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_macd(closes):
    macd_now = ema(closes, 12) - ema(closes, 26)

    previous = closes[:-1]

    if len(previous) >= 26:
        macd_previous = ema(previous, 12) - ema(previous, 26)
    else:
        macd_previous = macd_now

    return macd_now, macd_previous


def get_signal(history):

    closes = [float(c[4]) for c in history]
    volumes = [float(c[5]) for c in history]

    price = closes[-1]

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    rsi = calculate_rsi(closes)

    macd, previous_macd = calculate_macd(closes)

    recent_volume = sum(volumes[-5:]) / 5
    old_volume = sum(volumes[-20:-5]) / 15

    volume_strong = recent_volume > old_volume * 1.10

    recent_high = max(float(c[2]) for c in history[-20:])
    recent_low = min(float(c[1]) for c in history[-20:])

    buy_score = 0
    sell_score = 0

    # EMA
    if ema9 > ema21:
        buy_score += 2

    elif ema9 < ema21:
        sell_score += 2

    # RSI
    if 55 <= rsi <= 70:
        buy_score += 2

    elif 30 <= rsi <= 45:
        sell_score += 2

    # MACD
    if macd > 0 and macd >= previous_macd:
        buy_score += 2

    elif macd < 0 and macd <= previous_macd:
        sell_score += 2

    # Volume
    if volume_strong:

        if buy_score > sell_score:
            buy_score += 1

        elif sell_score > buy_score:
            sell_score += 1

    # Resistance / support
    resistance_distance = (
        (recent_high - price) / price
    ) * 100

    support_distance = (
        (price - recent_low) / price
    ) * 100

    if resistance_distance < 0.40:
        buy_score -= 2

    if support_distance < 0.40:
        sell_score -= 2

    if buy_score >= 5 and buy_score > sell_score:
        return "BUY"

    if sell_score >= 5 and sell_score > buy_score:
        return "SELL"

    return "WAIT"


def run_backtest(candles):

    results = {
        15: {"BUY": [0, 0], "SELL": [0, 0]},
        30: {"BUY": [0, 0], "SELL": [0, 0]},
        45: {"BUY": [0, 0], "SELL": [0, 0]}
    }

    lookback = 100

    print("🔎 Running 6-month historical test...")

    for i in range(lookback, len(candles) - 3):

        history = candles[i - lookback:i]

        signal = get_signal(history)

        if signal == "WAIT":
            continue

        entry_price = float(candles[i][4])

        for minutes in [15, 30, 45]:

            future_index = i + (minutes // 15)

            if future_index >= len(candles):
                continue

            future_price = float(candles[future_index][4])

            results[minutes][signal][1] += 1

            if signal == "BUY" and future_price > entry_price:
                results[minutes][signal][0] += 1

            elif signal == "SELL" and future_price < entry_price:
                results[minutes][signal][0] += 1

    return results


def print_results(results):

    print()
    print("=" * 55)
    print("📊 BTC 15M — 6 MONTH HISTORICAL BACKTEST")
    print("=" * 55)

    for minutes in [15, 30, 45]:

        print()
        print(f"⏱️ {minutes} MINUTES")

        for signal in ["BUY", "SELL"]:

            successful, total = results[minutes][signal]

            if total > 0:
                accuracy = (successful / total) * 100
            else:
                accuracy = 0

            print(
                f"{signal}: "
                f"Successful {successful} / "
                f"Total {total} "
                f"({accuracy:.1f}%)"
            )

    print()
    print("=" * 55)
    print("⚠️ Historical backtest only.")
    print("⚠️ Past results do not guarantee future results.")
    print("=" * 55)


def main():

    print("🚀 BTC 6-Month Backtest Started!")

    candles = get_six_month_data()

    if len(candles) < 1000:
        raise RuntimeError(
            "Not enough historical candle data."
        )

    results = run_backtest(candles)

    print_results(results)


if __name__ == "__main__":
    main()

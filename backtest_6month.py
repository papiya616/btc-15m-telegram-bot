import requests
import time
from datetime import datetime, timezone, timedelta

PRODUCT_ID = "BTC-USD"
GRANULARITY = 900
LOOKBACK = 100

# Test settings
TP_PERCENT = 0.30
SL_PERCENT = 0.20


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
    if len(closes) <= period:
        return 50

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

    # EMA
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

    # Support / Resistance
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


def check_trade(candles, entry_index, signal):

    entry = float(candles[entry_index][4])

    if signal == "BUY":
        tp_price = entry * (1 + TP_PERCENT / 100)
        sl_price = entry * (1 - SL_PERCENT / 100)

    else:
        tp_price = entry * (1 - TP_PERCENT / 100)
        sl_price = entry * (1 + SL_PERCENT / 100)

    max_bars = min(12, len(candles) - entry_index - 1)

    for j in range(1, max_bars + 1):

        candle = candles[entry_index + j]

        high = float(candle[2])
        low = float(candle[1])

        if signal == "BUY":

            hit_tp = high >= tp_price
            hit_sl = low <= sl_price

        else:

            hit_tp = low <= tp_price
            hit_sl = high >= sl_price

        # If both are touched in the same candle,
        # use conservative assumption: SL first.
        if hit_tp and hit_sl:
            return "SL"

        if hit_tp:
            return "TP"

        if hit_sl:
            return "SL"

    return "TIMEOUT"


def run_backtest(candles):

    stats = {
        "BUY": {
            "signals": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0
        },
        "SELL": {
            "signals": 0,
            "tp": 0,
            "sl": 0,
            "timeout": 0
        }
    }

    patterns = {}

    print("🔎 Running corrected pattern + TP/SL test...")

    for i in range(
        LOOKBACK,
        len(candles) - 12
    ):

        history = candles[i - LOOKBACK:i]

        analysis = analyze_market(history)

        signal = analysis["signal"]

        if signal == "WAIT":
            continue

        result = check_trade(
            candles,
            i,
            signal
        )

        stats[signal]["signals"] += 1

        if result == "TP":
            stats[signal]["tp"] += 1

        elif result == "SL":
            stats[signal]["sl"] += 1

        else:
            stats[signal]["timeout"] += 1

        pattern = "|".join(
            sorted(analysis["conditions"])
        )

        if pattern not in patterns:

            patterns[pattern] = {
                "signals": 0,
                "tp": 0,
                "sl": 0,
                "timeout": 0
            }

        patterns[pattern]["signals"] += 1

        if result == "TP":
            patterns[pattern]["tp"] += 1

        elif result == "SL":
            patterns[pattern]["sl"] += 1

        else:
            patterns[pattern]["timeout"] += 1

    return stats, patterns


def print_results(stats, patterns):

    print()
    print("=" * 70)
    print("📊 BTC 15M — 6 MONTH CORRECTED TP/SL BACKTEST")
    print("=" * 70)

    print()
    print(
        f"🎯 Test settings: "
        f"TP = {TP_PERCENT}% | "
        f"SL = {SL_PERCENT}% | "
        f"Max holding = 3 hours"
    )

    for signal in ["BUY", "SELL"]:

        data = stats[signal]

        total = data["signals"]

        tp_rate = (
            data["tp"] / total * 100
            if total else 0
        )

        sl_rate = (
            data["sl"] / total * 100
            if total else 0
        )

        timeout_rate = (
            data["timeout"] / total * 100
            if total else 0
        )

        print()
        print(f"📌 {signal}")
        print("-" * 70)

        print(f"Signals: {total}")
        print(
            f"TP: {data['tp']} "
            f"({tp_rate:.1f}%)"
        )
        print(
            f"SL: {data['sl']} "
            f"({sl_rate:.1f}%)"
        )
        print(
            f"TIMEOUT: {data['timeout']} "
            f"({timeout_rate:.1f}%)"
        )

    print()
    print("=" * 70)
    print("🔎 TOP MARKET PATTERNS")
    print("=" * 70)

    sorted_patterns = sorted(
        patterns.items(),
        key=lambda x: x[1]["signals"],
        reverse=True
    )

    for pattern, data in sorted_patterns[:20]:

        total = data["signals"]

        tp_rate = data["tp"] / total * 100
        sl_rate = data["sl"] / total * 100
        timeout_rate = data["timeout"] / total * 100

        print()
        print(f"Pattern: {pattern}")
        print(f"Signals: {total}")
        print(f"TP first: {data['tp']} ({tp_rate:.1f}%)")
        print(f"SL first: {data['sl']} ({sl_rate:.1f}%)")
        print(
            f"Timeout: "
            f"{data['timeout']} "
            f"({timeout_rate:.1f}%)"
        )

    print()
    print("=" * 70)
    print("⚠️ Historical backtest only.")
    print("⚠️ This does not guarantee future results.")
    print("=" * 70)


def main():

    print("🚀 BTC 6-Month Corrected Pattern Backtest Started!")

    candles = get_six_month_data()

    if len(candles) < 1000:
        raise RuntimeError(
            "Not enough historical data."
        )

    stats, patterns = run_backtest(candles)

    print_results(
        stats,
        patterns
    )


if __name__ == "__main__":
    main()

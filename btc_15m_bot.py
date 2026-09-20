import os
from datetime import datetime
from io import BytesIO

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


CHAT_ID = "5467490148"

PRODUCT_ID = "BTC-USD"
GRANULARITY = 900
CANDLE_LIMIT = 100


def get_token():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN secret is not set.")

    return token


def get_btc_data():

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

    response = requests.get(
        url,
        params={"granularity": GRANULARITY},
        timeout=30,
        headers={"User-Agent": "BTC-15M-Advanced-Bot"}
    )

    response.raise_for_status()

    data = response.json()
    data = sorted(data, key=lambda x: x[0])

    return data[-CANDLE_LIMIT:]


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

    gains = [max(change, 0) for change in changes]
    losses = [max(-change, 0) for change in changes]

    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period

    if avg_loss == 0:
        return 100

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_macd(closes):

    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)

    macd = ema12 - ema26

    # Approximation of MACD signal line
    previous_closes = closes[:-1]

    if len(previous_closes) >= 26:
        previous_macd = (
            ema(previous_closes, 12)
            - ema(previous_closes, 26)
        )
    else:
        previous_macd = macd

    return macd, previous_macd


def calculate_atr(candles, period=14):

    true_ranges = []

    for i in range(1, len(candles)):

        high = float(candles[i][2])
        low = float(candles[i][1])
        previous_close = float(candles[i - 1][4])

        tr = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )

        true_ranges.append(tr)

    return sum(true_ranges[-period:]) / period


def calculate_signal(candles):

    if len(candles) < 30:
        return {
            "signal": "WAIT",
            "score": 0,
            "reason": "Not enough candle data"
        }

    closes = [float(c[4]) for c in candles]
    volumes = [float(c[5]) for c in candles]

    current_price = closes[-1]

    # -------------------------
    # EMA TREND
    # -------------------------

    ema9 = ema(closes, 9)
    ema21 = ema(closes, 21)

    # -------------------------
    # RSI
    # -------------------------

    rsi = calculate_rsi(closes)

    # -------------------------
    # MACD
    # -------------------------

    macd, previous_macd = calculate_macd(closes)

    # -------------------------
    # VOLUME
    # -------------------------

    recent_volume = sum(volumes[-5:]) / 5
    older_volume = sum(volumes[-20:-5]) / 15

    volume_strong = recent_volume > older_volume * 1.10

    # -------------------------
    # SUPPORT / RESISTANCE
    # -------------------------

    recent_high = max(
        float(c[2]) for c in candles[-20:]
    )

    recent_low = min(
        float(c[1]) for c in candles[-20:]
    )

    distance_to_resistance = (
        (recent_high - current_price)
        / current_price
    ) * 100

    distance_to_support = (
        (current_price - recent_low)
        / current_price
    ) * 100

    # -------------------------
    # ATR
    # -------------------------

    atr = calculate_atr(candles)

    # -------------------------
    # SCORING
    # -------------------------

    buy_score = 0
    sell_score = 0

    reasons = []

    # EMA
    if ema9 > ema21:
        buy_score += 2
        reasons.append("EMA bullish")
    elif ema9 < ema21:
        sell_score += 2
        reasons.append("EMA bearish")

    # RSI
    if 55 <= rsi <= 70:
        buy_score += 2
        reasons.append(f"RSI bullish {rsi:.1f}")

    elif 30 <= rsi <= 45:
        sell_score += 2
        reasons.append(f"RSI bearish {rsi:.1f}")

    elif rsi > 70:
        reasons.append(f"RSI overbought {rsi:.1f}")

    elif rsi < 30:
        reasons.append(f"RSI oversold {rsi:.1f}")

    # MACD
    if macd > 0 and macd >= previous_macd:
        buy_score += 2
        reasons.append("MACD bullish")

    elif macd < 0 and macd <= previous_macd:
        sell_score += 2
        reasons.append("MACD bearish")

    # Volume
    if volume_strong:

        if buy_score > sell_score:
            buy_score += 1
            reasons.append("Volume confirmed")

        elif sell_score > buy_score:
            sell_score += 1
            reasons.append("Volume confirmed")

    # Support / Resistance protection
    near_resistance = distance_to_resistance < 0.40
    near_support = distance_to_support < 0.40

    if near_resistance:
        buy_score -= 2
        reasons.append("Near resistance")

    if near_support:
        sell_score -= 2
        reasons.append("Near support")

    # -------------------------
    # FINAL SIGNAL
    # -------------------------

    if buy_score >= 5 and buy_score > sell_score:

        signal = "BUY"

    elif sell_score >= 5 and sell_score > buy_score:

        signal = "SELL"

    else:

        signal = "WAIT"

    score = max(buy_score, sell_score)

    reason_text = " | ".join(reasons)

    return {
        "signal": signal,
        "score": score,
        "reason": reason_text,
        "price": current_price,
        "rsi": rsi,
        "ema9": ema9,
        "ema21": ema21,
        "atr": atr,
        "support": recent_low,
        "resistance": recent_high
    }


def make_chart(candles):

    times = [
        datetime.fromtimestamp(c[0])
        for c in candles
    ]

    lows = [float(c[1]) for c in candles]
    highs = [float(c[2]) for c in candles]
    opens = [float(c[3]) for c in candles]
    closes = [float(c[4]) for c in candles]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(len(candles)):

        open_price = opens[i]
        close_price = closes[i]
        high_price = highs[i]
        low_price = lows[i]

        if close_price >= open_price:
            candle_color = "green"
        else:
            candle_color = "red"

        ax.vlines(
            i,
            low_price,
            high_price,
            linewidth=1
        )

        bottom = min(open_price, close_price)
        height = abs(close_price - open_price)

        if height == 0:
            height = (high_price - low_price) * 0.01

        rectangle = Rectangle(
            (i - 0.3, bottom),
            0.6,
            height,
            facecolor=candle_color,
            edgecolor=candle_color
        )

        ax.add_patch(rectangle)

    ax.set_title("BTC/USD - 15 Minute Advanced Analysis")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")

    step = max(1, len(times) // 10)

    ax.set_xticks(range(0, len(times), step))

    ax.set_xticklabels(
        [
            times[i].strftime("%H:%M")
            for i in range(0, len(times), step)
        ],
        rotation=45
    )

    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    image = BytesIO()

    fig.savefig(
        image,
        format="png",
        dpi=160
    )

    plt.close(fig)

    image.seek(0)

    return image


def send_to_telegram(image, token, analysis):

    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    caption = (
        "🕯️ BTC 15M Advanced Signal\n\n"

        f"📌 Signal: {analysis['signal']}\n"
        f"⭐ Score: {analysis['score']}\n\n"

        f"💰 Price: ${analysis['price']:,.2f}\n"
        f"📊 RSI: {analysis['rsi']:.1f}\n"

        f"📈 EMA9: ${analysis['ema9']:,.2f}\n"
        f"📉 EMA21: ${analysis['ema21']:,.2f}\n\n"

        f"🟢 Support: ${analysis['support']:,.2f}\n"
        f"🔴 Resistance: ${analysis['resistance']:,.2f}\n\n"

        f"🔎 {analysis['reason']}\n\n"

        f"🕒 {now}\n\n"

        "⚠️ Technical analysis only."
    )

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "caption": caption
        },
        files={
            "photo": (
                "btc_15m.png",
                image,
                "image/png"
            )
        },
        timeout=60
    )

    response.raise_for_status()


def main():

    print("🚀 BTC 15M Advanced Signal Bot started!")

    token = get_token()

    print("📊 Getting BTC 15M data...")

    candles = get_btc_data()

    if not candles:
        raise RuntimeError(
            "No BTC candle data received."
        )

    analysis = calculate_signal(candles)

    print(
        f"📌 Signal: {analysis['signal']}"
    )

    print(
        f"⭐ Score: {analysis['score']}"
    )

    print(
        f"🔎 {analysis['reason']}"
    )

    chart = make_chart(candles)

    send_to_telegram(
        chart,
        token,
        analysis
    )

    print(
        "✅ Advanced BTC chart + signal sent!"
    )


if __name__ == "__main__":
    main()

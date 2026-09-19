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
CANDLE_LIMIT = 80


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
        headers={"User-Agent": "BTC-15M-Telegram-Bot"}
    )

    response.raise_for_status()

    data = response.json()
    data = sorted(data, key=lambda x: x[0])

    return data[-CANDLE_LIMIT:]


def calculate_signal(candles):

    closes = [float(candle[4]) for candle in candles]

    if len(closes) < 22:
        return "WAIT", "Not enough data"

    # EMA 9
    ema9 = closes[0]
    multiplier9 = 2 / (9 + 1)

    for price in closes[1:]:
        ema9 = (price - ema9) * multiplier9 + ema9

    # EMA 21
    ema21 = closes[0]
    multiplier21 = 2 / (21 + 1)

    for price in closes[1:]:
        ema21 = (price - ema21) * multiplier21 + ema21

    # RSI 14
    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[-14:]) / 14
    avg_loss = sum(losses[-14:]) / 14

    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

    if ema9 > ema21 and rsi >= 55:
        return "BUY", f"EMA9 > EMA21 | RSI {rsi:.1f}"

    if ema9 < ema21 and rsi <= 45:
        return "SELL", f"EMA9 < EMA21 | RSI {rsi:.1f}"

    return "WAIT", f"EMA9/EMA21 unclear | RSI {rsi:.1f}"


def make_chart(candles):

    times = [
        datetime.fromtimestamp(candle[0])
        for candle in candles
    ]

    lows = [float(candle[1]) for candle in candles]
    highs = [float(candle[2]) for candle in candles]
    opens = [float(candle[3]) for candle in candles]
    closes = [float(candle[4]) for candle in candles]

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

    ax.set_title("BTC/USD - 15 Minute Candlestick")
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


def send_to_telegram(image, token, signal, reason):

    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    caption = (
        "🕯️ BTC 15M Candlestick\n\n"
        f"📌 Signal: {signal}\n"
        f"📊 {reason}\n"
        f"🕒 {now}\n\n"
        "⚠️ Technical signal only."
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

    print("🚀 BTC 15M Signal Bot started!")

    token = get_token()

    print("📊 Getting BTC 15M data...")

    candles = get_btc_data()

    if not candles:
        raise RuntimeError("No BTC candle data received.")

    signal, reason = calculate_signal(candles)

    print(f"📌 Signal: {signal}")
    print(f"📊 Reason: {reason}")

    chart = make_chart(candles)

    send_to_telegram(
        chart,
        token,
        signal,
        reason
    )

    print("✅ BTC chart + signal sent to Telegram!")


if __name__ == "__main__":
    main()

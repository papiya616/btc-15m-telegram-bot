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
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN secret is not set."
        )

    return token


def get_btc_data():

    url = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

    params = {
        "granularity": GRANULARITY
    }

    response = requests.get(
        url,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "BTC-15M-Telegram-Bot"
        }
    )

    response.raise_for_status()

    data = response.json()

    # Coinbase returns:
    # [time, low, high, open, close, volume]
    data = sorted(data, key=lambda x: x[0])

    return data[-CANDLE_LIMIT:]


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

        # Wick
        ax.vlines(
            i,
            low_price,
            high_price,
            linewidth=1
        )

        # Body
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

    ax.set_xticks(
        range(0, len(times), step)
    )

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


def send_to_telegram(image, token):

    url = f"https://api.telegram.org/bot{token}/sendPhoto"

    now = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "caption": (
                "🕯️ BTC 15M Candlestick Chart\n"
                f"🕒 {now}\n"
                "⏳ Next update in 15 minutes."
            )
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

    print("🚀 BTC 15M Cloud Bot started!")

    token = get_token()

    print("📊 Getting BTC 15M data...")

    candles = get_btc_data()

    if not candles:
        raise RuntimeError(
            "No BTC candle data received."
        )

    chart = make_chart(candles)

    send_to_telegram(
        chart,
        token
    )

    print(
        "✅ BTC 15M candlestick sent to Telegram!"
    )


if __name__ == "__main__":
    main()

import requests
import time
from datetime import datetime, timezone, timedelta

PRODUCT_ID = "BTC-USD"
GRANULARITY = 900

LOOKBACK = 100

# ATR based trade settings
TP_ATR_MULTIPLIER = 1.5
SL_ATR_MULTIPLIER = 1.0

# Minimum reward/risk
MIN_RR = 1.3


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
        headers={"User-Agent": "BTC-Advanced-Backtest"}
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

        chunk_end = min(
            current + chunk,
            end
        )

        print(
            "Downloading:",
            current.strftime("%Y-%m-%d"),
            "to",
            chunk_end.strftime("%Y-%m-%d")
        )

        try:

            data = get_candles(
                current,
                chunk_end
            )

            all_candles.extend(data)

        except Exception as error:

            print(
                "Download error:",
                error
            )

        current = chunk_end

        time.sleep(0.3)

    unique = {}

    for candle in all_candles:

        unique[int(candle[0])] = candle

    candles = list(
        unique.values()
    )

    candles.sort(
        key=lambda x: x[0]
    )

    print(
        f"✅ Total candles collected: {len(candles)}"
    )

    return candles


def ema(values, period):

    if len(values) < period:

        return sum(values) / len(values)

    multiplier = 2 / (period + 1)

    result = sum(
        values[:period]
    ) / period

    for price in values[period:]:

        result = (
            (price - result)
            * multiplier
            + result
        )

    return result


def rsi(closes, period=14):

    if len(closes) <= period:

        return 50

    changes = [
        closes[i] - closes[i - 1]
        for i in range(1, len(closes))
    ]

    gains = [
        max(x, 0)
        for x in changes
    ]

    losses = [
        max(-x, 0)
        for x in changes
    ]

    avg_gain = (
        sum(gains[-period:])
        / period
    )

    avg_loss = (
        sum(losses[-period:])
        / period
    )

    if avg_loss == 0:

        return 100

    rs = avg_gain / avg_loss

    return 100 - (
        100 / (1 + rs)
    )


def atr(history, period=14):

    if len(history) < period + 1:

        return 0

    true_ranges = []

    for i in range(
        len(history) - period,
        len(history)
    ):

        current = history[i]

        previous = history[i - 1]

        high = float(current[2])
        low = float(current[1])

        previous_close = float(
            previous[4]
        )

        tr = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )

        true_ranges.append(tr)

    return sum(
        true_ranges
    ) / len(true_ranges)


def macd(closes):

    if len(closes) < 30:

        return 0, 0

    current = (
        ema(closes, 12)
        - ema(closes, 26)
    )

    previous_closes = closes[:-1]

    previous = (
        ema(previous_closes, 12)
        - ema(previous_closes, 26)
    )

    return current, previous


def analyze_market(history):

    closes = [
        float(x[4])
        for x in history
    ]

    highs = [
        float(x[2])
        for x in history
    ]

    lows = [
        float(x[1])
        for x in history
    ]

    volumes = [
        float(x[5])
        for x in history
    ]

    price = closes[-1]

    ema9 = ema(
        closes,
        9
    )

    ema21 = ema(
        closes,
        21
    )

    ema50 = ema(
        closes,
        50
    )

    current_rsi = rsi(
        closes,
        14
    )

    macd_now, macd_old = macd(
        closes
    )

    current_atr = atr(
        history,
        14
    )

    if price > 0:

        atr_percent = (
            current_atr
            / price
            * 100
        )

    else:

        atr_percent = 0

    # --------------------------------
    # Trend strength
    # --------------------------------

    trend_up = (
        ema9 > ema21
        and ema21 > ema50
    )

    trend_down = (
        ema9 < ema21
        and ema21 < ema50
    )

    # --------------------------------
    # Recent range
    # --------------------------------

    recent_high = max(
        highs[-20:]
    )

    recent_low = min(
        lows[-20:]
    )

    range_percent = (
        (recent_high - recent_low)
        / price
        * 100
    )

    # Sideways market detection
    sideways = (
        range_percent < 2.0
        and abs(ema9 - ema21)
        / price
        * 100 < 0.25
    )

    # --------------------------------
    # Volume
    # --------------------------------

    recent_volume = (
        sum(volumes[-5:])
        / 5
    )

    old_volume = (
        sum(volumes[-20:-5])
        / 15
    )

    volume_strong = (
        recent_volume
        > old_volume * 1.15
    )

    # --------------------------------
    # Momentum
    # --------------------------------

    momentum_up = (
        macd_now > 0
        and macd_now > macd_old
    )

    momentum_down = (
        macd_now < 0
        and macd_now < macd_old
    )

    # --------------------------------
    # Breakout
    # --------------------------------

    previous_high = max(
        highs[-21:-1]
    )

    previous_low = min(
        lows[-21:-1]
    )

    breakout_up = (
        price > previous_high
    )

    breakout_down = (
        price < previous_low
    )

    # --------------------------------
    # Scoring
    # --------------------------------

    buy_score = 0
    sell_score = 0

    conditions = []

    # Trend

    if trend_up:

        buy_score += 3

        conditions.append(
            "TREND_UP"
        )

    elif trend_down:

        sell_score += 3

        conditions.append(
            "TREND_DOWN"
        )

    else:

        conditions.append(
            "TREND_WEAK"
        )

    # RSI

    if 52 <= current_rsi <= 68:

        buy_score += 2

        conditions.append(
            "RSI_BUY_ZONE"
        )

    elif 32 <= current_rsi <= 48:

        sell_score += 2

        conditions.append(
            "RSI_SELL_ZONE"
        )

    elif current_rsi > 70:

        conditions.append(
            "RSI_OVERBOUGHT"
        )

    elif current_rsi < 30:

        conditions.append(
            "RSI_OVERSOLD"
        )

    else:

        conditions.append(
            "RSI_NEUTRAL"
        )

    # MACD

    if momentum_up:

        buy_score += 2

        conditions.append(
            "MACD_UP"
        )

    elif momentum_down:

        sell_score += 2

        conditions.append(
            "MACD_DOWN"
        )

    else:

        conditions.append(
            "MACD_FLAT"
        )

    # Volume

    if volume_strong:

        conditions.append(
            "VOLUME_STRONG"
        )

        if buy_score > sell_score:

            buy_score += 1

        elif sell_score > buy_score:

            sell_score += 1

    else:

        conditions.append(
            "VOLUME_NORMAL"
        )

    # Breakout

    if breakout_up:

        buy_score += 2

        conditions.append(
            "BREAKOUT_UP"
        )

    elif breakout_down:

        sell_score += 2

        conditions.append(
            "BREAKOUT_DOWN"
        )

    # Sideways

    if sideways:

        conditions.append(
            "SIDEWAYS"
        )

        # Avoid forcing trades
        buy_score -= 2
        sell_score -= 2

    # --------------------------------
    # Final signal
    # --------------------------------

    if (
        buy_score >= 7
        and buy_score > sell_score
        and not sideways
    ):

        signal = "BUY"

    elif (
        sell_score >= 7
        and sell_score > buy_score
        and not sideways
    ):

        signal = "SELL"

    else:

        signal = "WAIT"

    return {
        "signal": signal,
        "conditions": conditions,
        "price": price,
        "atr": current_atr,
        "atr_percent": atr_percent,
        "rsi": current_rsi,
        "buy_score": buy_score,
        "sell_score": sell_score,
        "trend_up": trend_up,
        "trend_down": trend_down,
        "sideways": sideways
    }


def check_trade(
    candles,
    entry_index,
    analysis
):

    signal = analysis["signal"]

    entry = float(
        candles[entry_index][4]
    )

    current_atr = analysis["atr"]

    if current_atr <= 0:

        return "NO_TRADE"

    tp_distance = (
        current_atr
        * TP_ATR_MULTIPLIER
    )

    sl_distance = (
        current_atr
        * SL_ATR_MULTIPLIER
    )

    # Make sure reward/risk is acceptable

    reward_risk = (
        tp_distance
        / sl_distance
    )

    if reward_risk < MIN_RR:

        return "NO_TRADE"

    if signal == "BUY":

        tp_price = (
            entry
            + tp_distance
        )

        sl_price = (
            entry
            - sl_distance
        )

    else:

        tp_price = (
            entry
            - tp_distance
        )

        sl_price = (
            entry
            + sl_distance
        )

    # Maximum 3 hours
    max_bars = min(
        12,
        len(candles)
        - entry_index
        - 1
    )

    for j in range(
        1,
        max_bars + 1
    ):

        candle = candles[
            entry_index + j
        ]

        high = float(
            candle[2]
        )

        low = float(
            candle[1]
        )

        if signal == "BUY":

            hit_tp = (
                high >= tp_price
            )

            hit_sl = (
                low <= sl_price
            )

        else:

            hit_tp = (
                low <= tp_price
            )

            hit_sl = (
                high >= sl_price
            )

        # Conservative assumption
        # when both happen in one candle

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

    regimes = {

        "TREND_UP": {
            "signals": 0,
            "tp": 0,
            "sl": 0
        },

        "TREND_DOWN": {
            "signals": 0,
            "tp": 0,
            "sl": 0
        },

        "SIDEWAYS": {
            "signals": 0,
            "tp": 0,
            "sl": 0
        },

        "OTHER": {
            "signals": 0,
            "tp": 0,
            "sl": 0
        }
    }

    print(
        "🔎 Running advanced market-regime backtest..."
    )

    for i in range(
        LOOKBACK,
        len(candles) - 12
    ):

        history = candles[
            i - LOOKBACK:i
        ]

        analysis = analyze_market(
            history
        )

        signal = analysis["signal"]

        if signal == "WAIT":

            continue

        result = check_trade(
            candles,
            i,
            analysis
        )

        if result == "NO_TRADE":

            continue

        stats[signal]["signals"] += 1

        if result == "TP":

            stats[signal]["tp"] += 1

        elif result == "SL":

            stats[signal]["sl"] += 1

        else:

            stats[signal]["timeout"] += 1

        # Pattern

        pattern = "|".join(
            sorted(
                analysis["conditions"]
            )
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

        # Market regime

        if analysis["trend_up"]:

            regime = "TREND_UP"

        elif analysis["trend_down"]:

            regime = "TREND_DOWN"

        elif analysis["sideways"]:

            regime = "SIDEWAYS"

        else:

            regime = "OTHER"

        regimes[regime]["signals"] += 1

        if result == "TP":

            regimes[regime]["tp"] += 1

        elif result == "SL":

            regimes[regime]["sl"] += 1

    return (
        stats,
        patterns,
        regimes
    )


def print_results(
    stats,
    patterns,
    regimes
):

    print()
    print("=" * 75)
    print(
        "📊 BTC 15M — ADVANCED 6 MONTH BACKTEST"
    )
    print("=" * 75)

    print()
    print(
        f"🎯 TP = {TP_ATR_MULTIPLIER} × ATR"
    )

    print(
        f"🛑 SL = {SL_ATR_MULTIPLIER} × ATR"
    )

    print(
        f"⚖️ Minimum R:R = {MIN_RR}"
    )

    print(
        "⏱️ Maximum holding time = 3 hours"
    )

    for signal in [
        "BUY",
        "SELL"
    ]:

        data = stats[signal]

        total = data["signals"]

        tp_rate = (
            data["tp"]
            / total
            * 100
            if total
            else 0
        )

        sl_rate = (
            data["sl"]
            / total
            * 100
            if total
            else 0
        )

        timeout_rate = (
            data["timeout"]
            / total
            * 100
            if total
            else 0
        )

        print()
        print(
            f"📌 {signal}"
        )

        print("-" * 75)

        print(
            f"Signals: {total}"
        )

        print(
            f"TP first: "
            f"{data['tp']} "
            f"({tp_rate:.1f}%)"
        )

        print(
            f"SL first: "
            f"{data['sl']} "
            f"({sl_rate:.1f}%)"
        )

        print(
            f"Timeout: "
            f"{data['timeout']} "
            f"({timeout_rate:.1f}%)"
        )

    print()
    print("=" * 75)
    print(
        "🌍 MARKET REGIME RESULTS"
    )
    print("=" * 75)

    for regime, data in regimes.items():

        total = data["signals"]

        if total == 0:

            continue

        tp_rate = (
            data["tp"]
            / total
            * 100
        )

        sl_rate = (
            data["sl"]
            / total
            * 100
        )

        print()
        print(
            f"{regime}: "
            f"{total} signals | "
            f"TP {tp_rate:.1f}% | "
            f"SL {sl_rate:.1f}%"
        )

    print()
    print("=" * 75)
    print(
        "🔎 TOP MARKET PATTERNS"
    )
    print("=" * 75)

    sorted_patterns = sorted(
        patterns.items(),
        key=lambda x: x[1]["signals"],
        reverse=True
    )

    for pattern, data in sorted_patterns[:20]:

        total = data["signals"]

        tp_rate = (
            data["tp"]
            / total
            * 100
        )

        sl_rate = (
            data["sl"]
            / total
            * 100
        )

        timeout_rate = (
            data["timeout"]
            / total
            * 100
        )

        print()
        print(
            f"Pattern: {pattern}"
        )

        print(
            f"Signals: {total}"
        )

        print(
            f"TP first: "
            f"{data['tp']} "
            f"({tp_rate:.1f}%)"
        )

        print(
            f"SL first: "
            f"{data['sl']} "
            f"({sl_rate:.1f}%)"
        )

        print(
            f"Timeout: "
            f"{data['timeout']} "
            f"({timeout_rate:.1f}%)"
        )

    print()
    print("=" * 75)
    print(
        "⚠️ Historical backtest only."
    )

    print(
        "⚠️ Results do not guarantee future performance."
    )

    print("=" * 75)


def main():

    print(
        "🚀 BTC Advanced 6-Month Backtest Started!"
    )

    candles = get_six_month_data()

    if len(candles) < 1000:

        raise RuntimeError(
            "Not enough historical data."
        )

    stats, patterns, regimes = run_backtest(
        candles
    )

    print_results(
        stats,
        patterns,
        regimes
    )


if __name__ == "__main__":

    main()

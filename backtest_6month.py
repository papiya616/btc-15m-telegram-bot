import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

# BTC 15M price-action liquidity-sweep backtest. Historical testing only.
PRODUCT = "BTC-USD"
GRANULARITY = 900
DAYS = 183
TRAIN_DAYS = 120
COST = 0.0014  # assumed round-trip fees/slippage: 0.14%
MAX_HOLD = 12   # 12 x 15-minute candles = 3 hours
COOLDOWN = 2


def download_data():
    print("🚀 BTC Price-Action Liquidity Sweep Backtest Started!\n")
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS)
    current = start
    all_rows = []
    url = f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles"
    print("📥 Downloading 6 months of BTC 15M data...\n")

    while current < end:
        chunk_end = min(current + timedelta(days=3), end)
        print(f"Downloading: {current:%Y-%m-%d %H:%M} to {chunk_end:%Y-%m-%d %H:%M}")
        response = requests.get(
            url,
            params={"granularity": GRANULARITY, "start": current.isoformat(), "end": chunk_end.isoformat()},
            headers={"User-Agent": "btc-15m-backtest/1.0"},
            timeout=30,
        )
        response.raise_for_status()
        rows = response.json()
        if isinstance(rows, list):
            all_rows.extend(rows)
        current = chunk_end
        time.sleep(0.2)

    if not all_rows:
        raise RuntimeError("Coinbase returned no candles.")

    df = pd.DataFrame(all_rows, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.drop_duplicates(subset=["time"]).sort_values("time")
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["low", "high", "open", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    print(f"\n✅ Total candles downloaded: {len(df)}")
    print(f"First candle: {df['time'].iloc[0]}")
    print(f"Last candle:  {df['time'].iloc[-1]}")
    return df


def add_indicators(df):
    d = df.copy()
    d["ema9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["ema21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["ema50"] = d["close"].ewm(span=50, adjust=False).mean()

    delta = d["close"].diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    d["rsi"] = 100 - (100 / (1 + rs))

    macd = d["close"].ewm(span=12, adjust=False).mean() - d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = macd
    d["macd_signal"] = macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    d["prev_macd_hist"] = d["macd_hist"].shift(1)

    prev_close = d["close"].shift(1)
    true_range = pd.concat([
        d["high"] - d["low"],
        (d["high"] - prev_close).abs(),
        (d["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    d["atr"] = true_range.rolling(14).mean()

    d["volume_ma"] = d["volume"].rolling(20).mean()
    d["vr"] = d["volume"] / d["volume_ma"].replace(0, np.nan)
    d["candle_range"] = (d["high"] - d["low"]).replace(0, np.nan)
    d["body"] = d["close"] - d["open"]
    d["body_atr"] = d["body"].abs() / d["atr"]
    d["body_ratio"] = d["body"].abs() / d["candle_range"]
    d["swh"] = d["high"].rolling(20).max().shift(1)
    d["swl"] = d["low"].rolling(20).min().shift(1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick_ratio"] = d["lower_wick"] / d["candle_range"]
    d["upper_wick_ratio"] = d["upper_wick"] / d["candle_range"]

    up = (d["ema9"] > d["ema21"]) & (d["ema21"] > d["ema50"])
    down = (d["ema9"] < d["ema21"]) & (d["ema21"] < d["ema50"])
    d["regime"] = np.select([up, down], ["UP", "DOWN"], default="RANGE")
    return d


def setup(row):
    # Use bracket notation for indicator columns: row.hist conflicts with pandas' hist method.
    needed = [
        "ema9", "ema21", "ema50", "close", "open", "high", "low",
        "macd_hist", "rsi", "vr", "body_atr", "body_ratio", "swh", "swl",
        "regime", "atr", "lower_wick_ratio", "upper_wick_ratio",
    ]
    if any(pd.isna(row[k]) for k in needed) or row["atr"] <= 0:
        return None

    atr = row["atr"]
    bullish_sweep = (
        row["regime"] == "UP"
        and row["ema9"] > row["ema21"] > row["ema50"]
        and row["close"] > row["ema21"]
        and row["macd_hist"] > 0
        and 45 <= row["rsi"] <= 68
        and row["vr"] >= 0.8
        and row["close"] > row["open"]
        and row["body_atr"] >= 0.25
        and row["body_ratio"] >= 0.35
        and row["low"] < row["swl"] - 0.10 * atr
        and row["close"] > row["swl"] + 0.05 * atr
    )
    bearish_sweep = (
        row["regime"] == "DOWN"
        and row["ema9"] < row["ema21"] < row["ema50"]
        and row["close"] < row["ema21"]
        and row["macd_hist"] < 0
        and 32 <= row["rsi"] <= 55
        and row["vr"] >= 0.8
        and row["close"] < row["open"]
        and row["body_atr"] >= 0.25
        and row["body_ratio"] >= 0.35
        and row["high"] > row["swh"] + 0.10 * atr
        and row["close"] < row["swh"] - 0.05 * atr
    )
    if bullish_sweep:
        return "BUY"
    if bearish_sweep:
        return "SELL"
    return None


def simulate_trade(df, signal_i, end_i, direction, tp_atr, sl_atr):
    # Signal is known only after its candle closes; enter at next candle open.
    entry_i = signal_i + 1
    if entry_i >= end_i:
        return None
    entry_row = df.iloc[entry_i]
    entry = float(entry_row["open"])
    atr = float(df.iloc[signal_i]["atr"])
    if not np.isfinite(atr) or atr <= 0 or entry <= 0:
        return None

    if direction == "BUY":
        tp = entry + tp_atr * atr
        sl = entry - sl_atr * atr
    else:
        tp = entry - tp_atr * atr
        sl = entry + sl_atr * atr

    last_i = min(entry_i + MAX_HOLD - 1, end_i - 1)
    exit_i = last_i
    exit_price = float(df.iloc[last_i]["close"])
    result = "TIMEOUT"

    for j in range(entry_i, last_i + 1):
        candle = df.iloc[j]
        if direction == "BUY":
            hit_sl = float(candle["low"]) <= sl
            hit_tp = float(candle["high"]) >= tp
        else:
            hit_sl = float(candle["high"]) >= sl
            hit_tp = float(candle["low"]) <= tp
        # Conservative if both TP and SL are inside the same candle: count SL first.
        if hit_sl:
            result, exit_price, exit_i = "SL", sl, j
            break
        if hit_tp:
            result, exit_price, exit_i = "TP", tp, j
            break

    gross = (exit_price - entry) / entry
    if direction == "SELL":
        gross = -gross
    net = gross - COST
    risk_fraction = sl_atr * atr / entry
    r_multiple = net / risk_fraction if risk_fraction > 0 else np.nan
    return {
        "entry_time": df.iloc[entry_i]["time"],
        "exit_time": df.iloc[exit_i]["time"],
        "direction": direction,
        "entry_price": entry,
        "exit_price": exit_price,
        "result": result,
        "net_return_pct": net * 100,
        "r": r_multiple,
        "exit_index": exit_i,
    }


def run(df, start_i, end_i, tp_atr, sl_atr):
    trades = []
    next_allowed = start_i
    for i in range(start_i, end_i - 1):
        if i < next_allowed:
            continue
        direction = setup(df.iloc[i])
        if direction is None:
            continue
        trade = simulate_trade(df, i, end_i, direction, tp_atr, sl_atr)
        if trade is None:
            continue
        trades.append(trade)
        next_allowed = trade["exit_index"] + COOLDOWN + 1
    return trades


def summarize(trades):
    if not trades:
        return None
    data = pd.DataFrame(trades)
    return {
        "data": data,
        "trades": len(data),
        "tp_pct": (data["result"] == "TP").mean() * 100,
        "sl_pct": (data["result"] == "SL").mean() * 100,
        "timeout_pct": (data["result"] == "TIMEOUT").mean() * 100,
        "avg_net_pct": data["net_return_pct"].mean(),
        "sum_net_pct": data["net_return_pct"].sum(),
        "avg_r": data["r"].mean(),
        "total_r": data["r"].sum(),
    }


def show(title, s):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)
    if s is None:
        print("No trades found.")
        return
    print(f"Trades: {s['trades']}")
    print(f"TP: {s['tp_pct']:.1f}% | SL: {s['sl_pct']:.1f}% | Timeout: {s['timeout_pct']:.1f}%")
    print(f"Average net return/trade: {s['avg_net_pct']:.3f}%")
    print(f"Sum net returns (not compounded): {s['sum_net_pct']:.2f}%")
    print(f"Average R: {s['avg_r']:.3f} | Total R: {s['total_r']:.2f}")


def main():
    df = add_indicators(download_data())
    df = df.replace([np.inf, -np.inf], np.nan)
    required = [
        "ema9", "ema21", "ema50", "atr", "rsi", "macd_hist", "vr",
        "body_atr", "body_ratio", "swh", "swl", "lower_wick_ratio", "upper_wick_ratio",
    ]
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 1000:
        raise RuntimeError(f"Not enough usable candles: {len(df)}")

    cutoff = df["time"].iloc[0] + timedelta(days=TRAIN_DAYS)
    train_end = int((df["time"] < cutoff).sum())
    val_start = train_end
    if train_end < 100 or val_start >= len(df) - 20:
        raise RuntimeError("Not enough data for train/validation split.")

    print(f"\nFinal usable candles: {len(df)}")
    print(f"Training: {df['time'].iloc[0]} → {df['time'].iloc[train_end - 1]}")
    print(f"Validation: {df['time'].iloc[val_start]} → {df['time'].iloc[-1]}")
    print(f"Assumed round-trip cost: {COST * 100:.2f}%")

    configs = [(1.0, 0.75), (1.25, 0.75), (1.5, 1.0), (1.5, 1.25), (2.0, 1.0), (2.0, 1.25)]
    best_config = None
    best_score = -np.inf
    print("\nTRAINING PRICE-ACTION CONFIGURATIONS")
    for tp, sl in configs:
        s = summarize(run(df, 0, train_end, tp, sl))
        show(f"TRAIN TP {tp:.2f} ATR / SL {sl:.2f} ATR", s)
        if s is not None and s["trades"] >= 10 and s["avg_r"] > best_score:
            best_score = s["avg_r"]
            best_config = (tp, sl)

    if best_config is None:
        print("\nNo configuration generated at least 10 training trades. Do not use live.")
        return

    tp, sl = best_config
    print(f"\nSelected using training only: TP {tp:.2f} ATR / SL {sl:.2f} ATR")
    validation = summarize(run(df, val_start, len(df), tp, sl))
    show("OUT-OF-SAMPLE VALIDATION", validation)
    if validation is None:
        print("\nNo validation trades. Do not use live.")
        return

    validation["data"].drop(columns=["exit_index"], errors="ignore").to_csv(
        "btc_liquidity_sweep_validation.csv", index=False
    )
    print("\nSaved: btc_liquidity_sweep_validation.csv")
    if validation["sum_net_pct"] > 0 and validation["avg_r"] > 0 and validation["trades"] >= 20:
        print("VERDICT: Potentially promising, not proven profitable. Test another unseen period.")
    else:
        print("VERDICT: No robust positive validation edge. Do not use live.")


if __name__ == "__main__":
    main()

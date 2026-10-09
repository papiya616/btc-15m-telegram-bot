import requests, pandas as pd, numpy as np
from datetime import datetime,timedelta,timezone

PRODUCT="BTC-USD"; GRANULARITY=900; DAYS=183
TRAIN_DAYS=120; VALIDATION_DAYS=63; MAX_HOLD=12; COOLDOWN=12; COST=.0014
SWING=12
EXITS=[("TP050_SL050",.50,.50),("TP075_SL050",.75,.50),("TP100_SL050",1,.50),("TP100_SL075",1,.75),("TP125_SL075",1.25,.75),("TP150_SL100",1.5,1),("TP150_SL125",1.5,1.25),("TP200_SL100",2,1)]

def download():
    print("🚀 BTC Price-Action Liquidity Sweep Backtest Started!\n")
    end=datetime.now(timezone.utc); start=end-timedelta(days=DAYS); cur=start; rows=[]
    print("📥 Downloading 6 months of BTC 15M data...\n")
    while cur<end:
        ce=min(cur+timedelta(minutes=3750),end)
        print(f"Downloading: {cur:%Y-%m-%d %H:%M} to {ce:%Y-%m-%d %H:%M}")
        try:
            r=requests.get(f"https://api.exchange.coinbase.com/products/{PRODUCT}/candles",
                params={"granularity":GRANULARITY,"start":cur.isoformat(),"end":ce.isoformat()},
                timeout=30,headers={"User-Agent":"BTC-backtest/1.0"})
            r.raise_for_status(); rows+=r.json()
        except Exception as e: print("Download error:",e)
        cur=ce+timedelta(minutes=15)
    df=pd.DataFrame(rows,columns=["time","low","high","open","close","volume"])
    df.time=pd.to_datetime(df.time,unit="s",utc=True)
    for c in ["open","high","low","close","volume"]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna().drop_duplicates("time").sort_values("time").reset_index(drop=True)
    print(f"\n✅ Total candles downloaded: {len(df):,}")
    print("First candle:",df.time.iloc[0]); print("Last candle:",df.time.iloc[-1],"\n")
    return df

def ema(s,n): return s.ewm(span=n,adjust=False).mean()

def indicators(d):
    print("📊 Calculating 15M indicators...")
    d=d.copy(); d["ema9"]=ema(d.close,9); d["ema21"]=ema(d.close,21); d["ema50"]=ema(d.close,50)
    x=d.close.diff(); g=x.clip(lower=0); l=-x.clip(upper=0)
    ag=g.ewm(alpha=1/14,adjust=False).mean(); al=l.ewm(alpha=1/14,adjust=False).mean()
    d["rsi"]=100-100/(1+ag/al.replace(0,np.nan)); d.rsi=d.rsi.fillna(50)
    d["macd"]=ema(d.close,12)-ema(d.close,26); d["macds"]=ema(d.macd,9); d["hist"]=d.macd-d.macds
    pc=d.close.shift(); d["tr"]=pd.concat([d.high-d.low,(d.high-pc).abs(),(d.low-pc).abs()],axis=1).max(axis=1)
    d["atr"]=d.tr.rolling(14).mean(); d["vr"]=d.volume/d.volume.rolling(20).mean()
    d["body"]=(d.close-d.open).abs(); d["rng"]=(d.high-d.low).replace(0,np.nan)
    d["body_atr"]=d.body/d.atr; d["body_ratio"]=d.body/d.rng; d["dist"]=abs(d.close-d.ema21)/d.atr
    return d

def regime(d):
    print("📊 Building completed 1H regime...")
    h=d.set_index("time").resample("1h",label="right",closed="right").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"}).dropna().reset_index()
    h["e20"]=ema(h.close,20); h["e50"]=ema(h.close,50); h["e100"]=ema(h.close,100)
    h["s20"]=h.e20-h.e20.shift(3); h["s50"]=h.e50-h.e50.shift(3)
    def rg(r):
        if r.e20>r.e50>r.e100 and r.s20>0 and r.s50>0 and r.close>r.e20:return "STRONG_UP"
        if r.e20<r.e50<r.e100 and r.s20<0 and r.s50<0 and r.close<r.e20:return "STRONG_DOWN"
        if r.e20>r.e50 and r.close>r.e50:return "NORMAL_UP"
        if r.e20<r.e50 and r.close<r.e50:return "NORMAL_DOWN"
        return "SIDEWAYS"
    h["regime"]=h.apply(rg,axis=1); h["available_time"]=pd.to_datetime(h.time,utc=True).astype("datetime64[ns, UTC]")
    return h[["available_time","regime"]]

def prepare(d):
    h=regime(d); d=d.copy()
    d.time=pd.to_datetime(d.time,utc=True).astype("datetime64[ns, UTC]")
    d=pd.merge_asof(d.sort_values("time"),h.sort_values("available_time"),left_on="time",right_on="available_time",direction="backward")
    d["swh"]=d.high.shift(1).rolling(SWING).max(); d["swl"]=d.low.shift(1).rolling(SWING).min()
    return d.dropna(subset=["ema9","ema21","ema50","rsi","hist","atr","vr","body_atr","body_ratio","regime","swh","swl"]).reset_index(drop=True)

def setup(d,i):
    if i<SWING+5:return None
    r=d.iloc[i]; atr=r.atr
    if r.regime=="SIDEWAYS" or r.dist>1.25:return None
    if r.regime in ("STRONG_UP","NORMAL_UP") and r.ema9>r.ema21>r.ema50 and r.close>r.ema21 and r.hist>0 and 45<=r.rsi<=68 and r.vr>=.8 and r.close>r.open and r.body_atr>=.25 and r.body_ratio>=.35 and r.low<r.swl-.10*atr and r.close>r.swl+.05*atr:
        return {"dir":"BUY","i":i,"level":r.swl,"hi":r.high,"lo":r.low,"reg":r.regime}
    if r.regime in ("STRONG_DOWN","NORMAL_DOWN") and r.ema9<r.ema21<r.ema50 and r.close<r.ema21 and r.hist<0 and 32<=r.rsi<=55 and r.vr>=.8 and r.close<r.open and r.body_atr>=.25 and r.body_ratio>=.35 and r.high>r.swh+.10*atr and r.close<r.swh-.05*atr:
        return {"dir":"SELL","i":i,"level":r.swh,"hi":r.high,"lo":r.low,"reg":r.regime}
    return None

def confirm(d,s):
    for j in range(s["i"]+1,min(len(d),s["i"]+3)):
        r=d.iloc[j]
        if s["dir"]=="BUY":
            if r.close<s["level"]: return None
            if r.close>s["hi"] or (r.close>d.iloc[s["i"]].close and r.close>r.open):
                return j
        else:
            if r.close>s["level"]: return None
            if r.close<s["lo"] or (r.close<d.iloc[s["i"]].close and r.close<r.open):
                return j
    return None

def trade(d,s,j,tp,sl):
    r=d.iloc[j]; p=float(r.close); atr=float(r.atr); sr=d.iloc[s["i"]]
    if s["dir"]=="BUY":
        stop=max(float(sr.swl)-.15*atr,p-sl*atr); risk=p-stop
        if risk<=0:return None
        target=p+tp*risk
    else:
        stop=min(float(sr.swh)+.15*atr,p+sl*atr); risk=stop-p
        if risk<=0:return None
        target=p-tp*risk
    end=min(len(d)-1,j+MAX_HOLD); mfe=mae=0.; result="TIMEOUT"; ex=float(d.iloc[end].close)
    for k in range(j+1,end+1):
        x=d.iloc[k]
        if s["dir"]=="BUY":
            mfe=max(mfe,(x.high-p)/p); mae=min(mae,(x.low-p)/p); ht=x.high>=target; hs=x.low<=stop
        else:
            mfe=max(mfe,(p-x.low)/p); mae=min(mae,(p-x.high)/p); ht=x.low<=target; hs=x.high>=stop
        if hs and ht or hs: result="SL"; ex=stop; break
        if ht: result="TP"; ex=target; break
    gross=(ex-p)/p if s["dir"]=="BUY" else (p-ex)/p; net=gross-COST; rr=net/(risk/p)
    return {"direction":s["dir"],"time":d.iloc[j].time,"result":result,"net_return":net,"r":rr,"mfe":mfe,"mae":mae,"regime":s["reg"]}

def run(d,a,b,tp,sl):
    out=[]; nxt=-1
    for i in range(max(150,SWING+5),len(d)):
        if i<nxt:continue
        t=d.iloc[i].time
        if t<a or t>=b:continue
        s=setup(d,i)
        if not s:continue
        j=confirm(d,s)
        if j is None or d.iloc[j].time>=b:continue
        z=trade(d,s,j,tp,sl)
        if z:out.append(z);nxt=j+COOLDOWN
    return out

def summ(x):
    n=len(x)
    if not n:return dict(trades=0,tp=0,sl=0,timeout=0,ret=0,avgr=0,totalr=0,mfe=0,mae=0)
    tp=sum(q["result"]=="TP" for q in x); sl=sum(q["result"]=="SL" for q in x); to=n-tp-sl
    return dict(trades=n,tp=100*tp/n,sl=100*sl/n,timeout=100*to/n,ret=100*sum(q["net_return"] for q in x),avgr=np.mean([q["r"] for q in x]),totalr=sum(q["r"] for q in x),mfe=100*np.mean([q["mfe"] for q in x]),mae=100*np.mean([q["mae"] for q in x]))

def show(name,s):
    print(f"{name:16s} | Trades={s['trades']} | TP={s['tp']:.1f}% | SL={s['sl']:.1f}% | Timeout={s['timeout']:.1f}% | Return={s['ret']:.2f}% | AvgR={s['avgr']:.3f} | TotalR={s['totalr']:.2f} | MFE={s['mfe']:.3f}% | MAE={s['mae']:.3f}%")

def main():
    d=prepare(indicators(download())); print(f"Final usable candles: {len(d):,}\n")
    a=d.time.iloc[0]; tr=a+timedelta(days=TRAIN_DAYS); va=min(tr+timedelta(days=VALIDATION_DAYS),d.time.iloc[-1])
    print(f"Training: {a} → {tr}\nValidation: {tr} → {va}\n")
    print("="*80);print("TRAINING PRICE-ACTION CONFIGURATIONS");print("="*80)
    rows=[]
    for name,tp,sl in EXITS:
        x=run(d,a,tr,tp,sl); s=summ(x); show(name,s)
        rows.append({"exit":name,"tp":tp,"sl":sl,**s})
    q=pd.DataFrame(rows)
    pos=q[(q.trades>=20)&(q.ret>0)&(q.avgr>0)].sort_values(["avgr","ret"],ascending=False)
    print("\n"+"="*80);print("POSITIVE TRAINING CONFIGURATIONS");print("="*80)
    if pos.empty:
        print("❌ No positive training configuration.");print("The liquidity-sweep setup did not demonstrate a positive training edge.");return
    print(pos[["exit","trades","ret","avgr","totalr"]].to_string(index=False))
    print("\n"+"="*80);print("UNSEEN VALIDATION");print("="*80)
    vr=[]; vt=[]
    for _,z in pos.iterrows():
        x=run(d,tr,va,float(z.tp),float(z.sl)); s=summ(x); show(z.exit,s)
        vr.append({"exit":z.exit,"tp":z.tp,"sl":z.sl,**s})
        for t in x:t["exit"]=z.exit
        vt+=x
    pd.DataFrame(vr).to_csv("btc_liquidity_sweep_validation.csv",index=False)
    pd.DataFrame(vt).to_csv("btc_liquidity_sweep_trades.csv",index=False)
    print("\n💾 Saved validation/trade CSV files.")
    best=pd.DataFrame(vr).sort_values(["avgr","ret"],ascending=False).iloc[0]
    print(f"\nBest validation: {best.exit} | Trades={int(best.trades)} | AvgR={best.avgr:.3f} | TotalR={best.totalr:.2f}")
    if best.trades>=20 and best.avgr>0 and best.ret>0: print("⚠️ Positive unseen result — paper-test before live use.")
    else: print("❌ No positive unseen validation edge. Do NOT move it to the live bot.")

if __name__=="__main__": main()

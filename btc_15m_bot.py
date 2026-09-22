import os
from datetime import datetime
from io import BytesIO
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CHAT_ID='5467490148'
URL='https://api.exchange.coinbase.com/products/BTC-USD/candles'
LIMIT=200

def token():
    t=os.getenv('TELEGRAM_BOT_TOKEN','').strip()
    if not t: raise RuntimeError('TELEGRAM_BOT_TOKEN secret is not set.')
    return t

def data():
    r=requests.get(URL,params={'granularity':900},timeout=30,headers={'User-Agent':'BTC-15M-Bot'})
    r.raise_for_status(); d=sorted(r.json(),key=lambda x:x[0]); return d[-LIMIT:]

def ema(v,p):
    m=2/(p+1); x=v[0]
    for z in v[1:]: x=(z-x)*m+x
    return x

def rsi(c,p=14):
    ch=[c[i]-c[i-1] for i in range(1,len(c))]
    g=sum(max(x,0) for x in ch[-p:])/p; l=sum(max(-x,0) for x in ch[-p:])/p
    return 100 if l==0 else 100-100/(1+g/l)

def macd(c):
    return ema(c,12)-ema(c,26), ema(c[:-1],12)-ema(c[:-1],26)

def signal(cs):
    if len(cs)<30: return {'signal':'WAIT','score':0,'reason':'Not enough data','price':float(cs[-1][4])}
    c=[float(x[4]) for x in cs]; vol=[float(x[5]) for x in cs]; p=c[-1]
    e9,e21=ema(c,9),ema(c,21); rr=rsi(c); m,pm=macd(c)
    rv=sum(vol[-5:])/5; ov=sum(vol[-20:-5])/15
    hi=max(float(x[2]) for x in cs[-20:]); lo=min(float(x[1]) for x in cs[-20:])
    buy=sell=0; why=[]
    if e9>e21: buy+=2; why.append('EMA bullish')
    elif e9<e21: sell+=2; why.append('EMA bearish')
    if 55<=rr<=70: buy+=2; why.append(f'RSI bullish {rr:.1f}')
    elif 30<=rr<=45: sell+=2; why.append(f'RSI bearish {rr:.1f}')
    elif rr>70: why.append(f'RSI overbought {rr:.1f}')
    elif rr<30: why.append(f'RSI oversold {rr:.1f}')
    if m>0 and m>=pm: buy+=2; why.append('MACD bullish')
    elif m<0 and m<=pm: sell+=2; why.append('MACD bearish')
    if rv>ov*1.10:
        if buy>sell: buy+=1; why.append('Volume confirmed')
        elif sell>buy: sell+=1; why.append('Volume confirmed')
    if (hi-p)/p*100<.40: buy-=2; why.append('Near resistance')
    if (p-lo)/p*100<.40: sell-=2; why.append('Near support')
    s='BUY' if buy>=5 and buy>sell else 'SELL' if sell>=5 and sell>buy else 'WAIT'
    return {'signal':s,'score':max(buy,sell),'reason':' | '.join(why),'price':p,'rsi':rr,'ema9':e9,'ema21':e21,'support':lo,'resistance':hi}

def backtest(cs):
    out={h:{'BUY':[0,0],'SELL':[0,0]} for h in (1,2,3)}
    end=len(cs)-3
    for i in range(30,end):
        a=signal(cs[:i+1]); s=a['signal']
        if s not in ('BUY','SELL'): continue
        entry=float(cs[i][4])
        for h in (1,2,3):
            future=float(cs[i+h][4]); out[h][s][1]+=1
            if (s=='BUY' and future>entry) or (s=='SELL' and future<entry): out[h][s][0]+=1
    lines=['📊 Recent Historical Test (same rules, no future data used)']
    for h,label in ((1,'15M'),(2,'30M'),(3,'45M')):
        b,w=out[h]['BUY']; s,st=out[h]['SELL']
        bp=w/b*100 if b else 0; sp=st/s*100 if s else 0
        lines.append(f'\n⏱ {label}\n🟢 BUY: {w}/{b} ({bp:.0f}%)\n🔴 SELL: {st}/{s} ({sp:.0f}%)')
    lines.append('\n⚠️ Historical test only; not a profit guarantee.')
    return '\n'.join(lines)

def chart(cs):
    op=[float(x[3]) for x in cs]; cl=[float(x[4]) for x in cs]; hi=[float(x[2]) for x in cs]; lo=[float(x[1]) for x in cs]
    ts=[datetime.fromtimestamp(x[0]) for x in cs]; fig,ax=plt.subplots(figsize=(12,6))
    for i in range(len(cs)):
        col='green' if cl[i]>=op[i] else 'red'; ax.vlines(i,lo[i],hi[i],linewidth=1); bot=min(op[i],cl[i]); ht=abs(cl[i]-op[i]) or (hi[i]-lo[i])*.01
        ax.add_patch(Rectangle((i-.3,bot),.6,ht,facecolor=col,edgecolor=col))
    step=max(1,len(ts)//10); ax.set_xticks(range(0,len(ts),step)); ax.set_xticklabels([ts[i].strftime('%H:%M') for i in range(0,len(ts),step)],rotation=45); ax.set_title('BTC/USD - 15 Minute Advanced Analysis'); ax.set_ylabel('Price (USD)'); ax.grid(True,alpha=.25); fig.tight_layout()
    b=BytesIO(); fig.savefig(b,format='png',dpi=160); plt.close(fig); b.seek(0); return b

def send(img,t,a,test):
    cap=(f"🕯️ BTC 15M Advanced Signal\n\n📌 Signal: {a['signal']}\n⭐ Score: {a['score']}\n\n💰 Price: ${a['price']:,.2f}\n📊 RSI: {a.get('rsi',0):.1f}\n📈 EMA9: ${a.get('ema9',0):,.2f}\n📉 EMA21: ${a.get('ema21',0):,.2f}\n\n🟢 Support: ${a.get('support',0):,.2f}\n🔴 Resistance: ${a.get('resistance',0):,.2f}\n\n🔎 {a['reason']}\n\n{test}\n\n🕒 {datetime.now():%Y-%m-%d %H:%M:%S}\n\n⚠️ Technical analysis only.")
    r=requests.post(f'https://api.telegram.org/bot{t}/sendPhoto',data={'chat_id':CHAT_ID,'caption':cap},files={'photo':('btc_15m.png',img,'image/png')},timeout=60); r.raise_for_status()

def main():
    print('🚀 BTC 15M Performance Bot started!'); t=token(); cs=data(); a=signal(cs); print('📌 Signal:',a['signal']); test=backtest(cs); print(test); send(chart(cs),t,a,test); print('✅ BTC chart + signal + performance test sent!')

if __name__=='__main__': main()

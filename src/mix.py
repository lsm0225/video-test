"""씬별 음악을 컷 타임라인에 맞춰 자르고, 효과음·내레이션과 섞어 build/audio.wav를 만든다."""
import json, os, numpy as np, librosa, soundfile as sf
from scipy import signal
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); SR=48000
tl=json.load(open(f"{ROOT}/build/timeline.json")); cut={c["id"]:c for c in tl["cuts"]}; N=int((tl["total"]+0.2)*SR)
rng=np.random.default_rng(7)
def db(x): return 10**(x/20)
def rms(x): return float(np.sqrt(np.mean(x**2))+1e-9)
def place(buf,x,t,g=1.0):
    i=int(round(t*SR)); 
    if i<0: x=x[...,-i:]; i=0
    n=min(x.shape[-1],buf.shape[-1]-i)
    if n>0: buf[...,i:i+n]+=g*x[...,:n]
def env(n,a,r):  # 페이드 인/아웃(초)
    e=np.ones(n); na=min(n,int(a*SR)); nr=min(n,int(r*SR))
    if na>0: e[:na]=np.sin(np.linspace(0,np.pi/2,na))**2
    if nr>0: e[-nr:]*=np.cos(np.linspace(0,np.pi/2,nr))**2
    return e
def bp(x,lo,hi,order=2):
    sos=signal.butter(order,[lo,hi],btype="band",fs=SR,output="sos"); return signal.sosfilt(sos,x)
def lp(x,f,order=2): return signal.sosfilt(signal.butter(order,f,btype="low",fs=SR,output="sos"),x)
def hp(x,f,order=2): return signal.sosfilt(signal.butter(order,f,btype="high",fs=SR,output="sos"),x)
def noise(d): return rng.standard_normal(int(d*SR))
def st(x,pan=0.0): return np.stack([x*np.sqrt((1-pan)/2*2)/np.sqrt(2)*np.sqrt(2) if False else x*np.cos((pan+1)*np.pi/4), x*np.sin((pan+1)*np.pi/4)])*np.sqrt(2)/1.0

# ---------- 1) 음악
music=np.zeros((2,N)); HARD={"특화·드럼","합격·스웰","댄스","등굣길·밝음","동아리·리듬","시설·경쾌함"}  # 컷과 동시에 치고 들어오는 구간
TARGET={"오프닝":-19,"학업·따뜻함":-22,"시설·경쾌함":-20,"특화·드럼":-17.5,"도서실·차분함":-23,"동아리·리듬":-18.5,"등굣길·밝음":-20,"급식·포근함":-22,"찬양":-21.5,"댄스":-19,"합격·스웰":-18,"엔딩·빌드업":-17,"엔딩·종지":-17}
tracks={}
segs=tl["music"]
for i,m in enumerate(segs):
    k=m["track"]
    if k not in tracks: tracks[k]=librosa.load(f"{ROOT}/assets/audio/music/{k}.mp3",sr=SR,mono=False)[0]
    y=tracks[k]; nxt=segs[i+1] if i+1<len(segs) else None
    gap = nxt is None or nxt["tl_in"]-m["tl_out"]>0.5      # 뒤에 음악이 없는 구간
    nhard = nxt is not None and (nxt["name"] in HARD)
    tail = 0.0 if nxt is None else (0.12 if nhard else 1.0)
    if m["name"]=="엔딩·빌드업": tail=0.25
    if gap and nxt is not None: tail=0.6
    a = 0.02 if (m["name"] in HARD or m["name"]=="오프닝") else (0.25 if m.get("jump") else 0.9)
    dur=m["tl_out"]-m["tl_in"]+tail; s0=int(m["src_in"]*SR); x=y[:,s0:s0+int(dur*SR)].copy()
    body=x[:,:int((m["tl_out"]-m["tl_in"])*SR)]; g=db(TARGET[m["name"]])/rms(body)
    if m["name"]=="엔딩·종지": g=prev_g           # 같은 곡의 이어 붙임: 음량 유지
    prev_g=g
    r = tail if nxt is not None else 0.01
    x*=env(x.shape[1],a,r)*g
    place(music,x,m["tl_in"])
    print(f'{m["name"]:10s} {m["title"]:28s} {m["tl_in"]:7.2f}-{m["tl_out"]:7.2f} src {m["src_in"]:6.2f} gain {20*np.log10(g):+.1f}dB')
# 연출용 음량 오토메이션
auto=np.ones(N)
def dip(t0,t1,depth_db,a=0.25,r=0.5):
    i0,i1=int(t0*SR),int(t1*SR); e=np.ones(N); e[i0:i1]=db(depth_db)
    na,nr=int(a*SR),int(r*SR); e[max(0,i0-na):i0]=np.linspace(1,db(depth_db),min(na,i0)); e[i1:i1+nr]=np.linspace(db(depth_db),1,nr)[:max(0,min(nr,N-i1))]
    return e
c15=cut["C15"]; auto*=dip(c15["start"]+0.05,c15["start"]+1.5,-14,a=0.05,r=0.5)    # C-15 음악 잠깐 다운 → 숨소리 → 다시 업
for n in tl["na"]: auto*=dip(n["start"]-0.1,n["start"]+n["dur"]+0.05,-7.5,a=0.3,r=0.6)  # 내레이션 구간 덕킹
auto=np.maximum(auto,db(-16))
c29=cut["C29a"]; c31=cut["C31"]
ramp=np.ones(N); i0,i1=int(c29["start"]*SR),int(c31["start"]*SR); ramp[i0:i1]=np.linspace(db(-4),1,i1-i0)   # 엔딩 빌드업: 점점 크게
music*=auto*ramp

# ---------- 2) 내레이션
voice=np.zeros((2,N))
for n in tl["na"]:
    y,_=librosa.load(f'{ROOT}/assets/audio/na/{n["key"]}.wav',sr=SR); y=hp(y,70); y*=db(-17)/rms(y[np.abs(y)>0.01]) if np.any(np.abs(y)>0.01) else 1
    y*=env(len(y),0.01,0.03); place(voice,np.stack([y,y]),n["start"])

# ---------- 3) 효과음 (전부 합성)
fx=np.zeros((2,N))
def whoosh(d=0.6,f0=400,f1=3000,peak=0.5):
    n=int(d*SR); x=noise(d); t=np.linspace(0,1,n); f=f0*(f1/f0)**t; out=np.zeros(n); B=1024
    for i in range(0,n,B):
        fc=f[min(i+B//2,n-1)]; out[i:i+B]=bp(np.concatenate([x[max(0,i-2048):i],x[i:i+B]]),fc*0.7,min(fc*1.4,20000))[-len(x[i:i+B]):]
    e=np.where(t<peak,(t/peak)**2,((1-t)/(1-peak))**1.5); return out*e
def thump(f=62,d=0.5,drop=0.5):
    t=np.arange(int(d*SR))/SR; ph=2*np.pi*np.cumsum(f*(1-drop*(t/d)))/SR; return np.sin(ph)*np.exp(-t*9)
def blip(f=1100,d=0.16):
    t=np.arange(int(d*SR))/SR; return (np.sin(2*np.pi*f*t)+0.4*np.sin(2*np.pi*f*1.5*t))*np.exp(-t*30)
def pan2(x,p): return np.stack([x*np.cos((p+1)*np.pi/4),x*np.sin((p+1)*np.pi/4)])
c01,c02,c03=cut["C01"],cut["C02"],cut["C03"]; rel=c03["start"]+0.42
# 바람: C-01~02 내내, 갈수록 긴장 고조 → 화살 발사 순간 끊김
d=rel; w=noise(d); t=np.arange(len(w))/SR
wind=bp(w,180,700)*(0.55+0.45*np.sin(2*np.pi*0.13*t+1))+0.5*bp(noise(d),700,2200)*(0.5+0.5*np.sin(2*np.pi*0.21*t))
wind*=np.clip(t/1.5,0,1)*(0.6+0.4*t/d); wind*=db(-27)/rms(wind); wind*=env(len(wind),0.01,0.06)
place(fx,pan2(wind,-0.15)+pan2(np.roll(wind,3000),0.25)*0.6,0)
# 긴장 드론: C-02에서 서서히
dd=rel-c02["start"]; t=np.arange(int(dd*SR))/SR; dr=(np.sin(2*np.pi*55*t)+0.5*np.sin(2*np.pi*82.5*t)+0.25*np.sin(2*np.pi*110.3*t))*(t/dd)**2
place(fx,pan2(dr*db(-24),0),c02["start"])
# 화살 발사: 시위 튕김 + 날아가는 소리 + 저음 타격 (이 순간 음악 첫 박이 들어온다)
tw=hp(noise(0.09),2500)*np.exp(-np.arange(int(0.09*SR))/SR*60)
place(fx,pan2(tw*db(-10),-0.3),rel); place(fx,pan2(whoosh(0.5,3500,500,0.15)*db(-12),0.4),rel+0.01); place(fx,pan2(thump(70,0.6)*db(-8),0),rel)
# 장면 전환 휘익
for cid,g in [("C05",-22),("C09",-20),("C10",-24),("C12",-20),("C19",-21),("C20",-21),("C24c",-18),("C33",-22)]:
    x=whoosh(0.7,300,2600,0.72); place(fx,pan2(x*db(g)/rms(x)*0.25,rng.uniform(-.4,.4)),cut[cid]["start"]-0.5)
# C-06 벽이 열리는 순간
c=cut["C06"]; x=whoosh(1.4,150,900,0.5); place(fx,pan2(x*db(-24)/rms(x)*0.25,0),c["start"]+c["dur"]*0.4)
# C-09 엠블럼 등장/흡수, C-11 키워드, C-12 시설 라벨
c=cut["C09"]
for i in range(5): place(fx,pan2(blip(880*2**(i*2/12))*db(-21),(i-2)*0.25),c["start"]+0.35+i*0.3+0.2)
x=whoosh(0.8,600,4000,0.8); place(fx,pan2(x*db(-22)/rms(x)*0.25,0),c["start"]+c["dur"]*0.66)
c=cut["C11"]
for i in range(int(c["dur"]/0.28)): place(fx,pan2(blip(1500+300*(i%4),0.07)*db(-27),rng.uniform(-.6,.6)),c["start"]+0.3+i*0.28)
c=cut["C12"]
for i in range(7): place(fx,pan2(blip(784*2**([0,2,4,5,7,9,12][i]/12))*db(-22),(i-3)*0.2),c["start"]+c["dur"]*0.36+i*(c["dur"]*0.42/7)+0.1)
# C-14 발차기 타격
c=cut["C14"]; th=c["start"]+c["dur"]*0.53
x=whoosh(0.3,500,2500,0.85); place(fx,pan2(x*db(-17)/rms(x)*0.25,-0.3),th-0.28)
hit=lp(noise(0.12),1800)*np.exp(-np.arange(int(0.12*SR))/SR*45); place(fx,pan2(hit*db(-9),0.1),th); place(fx,pan2(thump(85,0.35,0.6)*db(-7),0),th)
# C-15 거친 숨소리 두 번
c=cut["C15"]
for j,(off,dur,f) in enumerate([(0.12,0.5,1500),(0.68,0.6,900),(1.4,0.42,1500)]):
    n=int(dur*SR); t=np.linspace(0,1,n); b=bp(noise(dur),f*0.5,f*1.6)*np.sin(np.pi*t)**1.5; place(fx,pan2(b*db(-21)/rms(b)*0.3,0.05),c["start"]+off)
# C-25 합격 발표 직전: 음악 없이 심장 박동 + 얇은 긴장음 → 상승음 → C-26에서 터짐
c=cut["C25"]; t0=c["start"]; k=0; tt=0.35; gap=0.92
while tt<c["dur"]-0.3:
    place(fx,pan2(thump(52,0.3,0.3)*db(-13),0),t0+tt); place(fx,pan2(thump(44,0.3,0.3)*db(-17),0),t0+tt+0.24); tt+=gap; gap*=0.86
t=np.arange(int(c["dur"]*SR))/SR; hi=(np.sin(2*np.pi*1975*t)+0.6*np.sin(2*np.pi*2960*t))*(t/c["dur"])**2*env(len(t),0.01,0.08); place(fx,pan2(hi*db(-37),0),t0)
x=whoosh(1.3,300,6000,0.93); place(fx,pan2(x*db(-17)/rms(x)*0.25,0),cut["C26"]["start"]-1.25)
place(fx,pan2(thump(58,0.9,0.4)*db(-9),0),cut["C26"]["start"])
# C-31 슬로건 순간의 저음 임팩트
place(fx,pan2(thump(50,1.2,0.35)*db(-8),0),cut["C31"]["start"])
x=whoosh(1.0,300,5000,0.92); place(fx,pan2(x*db(-19)/rms(x)*0.25,0),cut["C31"]["start"]-0.95)

# ---------- 4) 합치기
mix=music+voice+fx
end=np.ones(N); nf=int(0.8*SR); end[-nf:]=np.linspace(1,0,nf); mix*=end
pk=np.max(np.abs(mix)); print("peak",round(float(pk),3))
mix=np.tanh(mix*1.0/max(1.0,pk/0.98)*1.15)/np.tanh(1.15)*0.9
os.makedirs(f"{ROOT}/build",exist_ok=True)
sf.write(f"{ROOT}/build/audio_raw.wav",mix.T,SR)
for name,b in [("music",music),("voice",voice),("fx",fx)]: sf.write(f"{ROOT}/build/stem_{name}.wav",(b*end).T,SR)
# 장면 간 강약(무음 오프닝, 정적 구간)을 살리기 위해 압축 없이 전체 음량만 -16 LUFS로 맞춘다
import subprocess, re
r=subprocess.run(["ffmpeg","-hide_banner","-i",f"{ROOT}/build/audio_raw.wav","-af","loudnorm=print_format=json","-f","null","-"],capture_output=True,text=True).stderr
I=float(re.search(r'"input_i"\s*:\s*"(-?[\d.]+)"',r).group(1)); gain=-16-I; print("integrated",I,"LUFS → gain",round(gain,2),"dB")
os.system(f'ffmpeg -y -loglevel error -i "{ROOT}/build/audio_raw.wav" -af "volume={gain}dB,alimiter=limit=0.84:attack=3:release=60:level=disabled" -ar 48000 "{ROOT}/build/audio.wav"')
print("ok")

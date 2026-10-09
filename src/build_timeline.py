"""컷 길이를 내레이션 길이와 씬별 음악 박자에 맞춰 확정하고 timeline.json을 만든다."""
import json, numpy as np, librosa, os
H=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
info=json.load(open(f"{H}/assets/audio/music/info.json")); na=json.load(open(f"{H}/assets/audio/na/na.json"))
_cache={}
def load(k):
    if k not in _cache: _cache[k]=librosa.load(f"{H}/assets/audio/music/{k}.mp3",sr=22050)[0]
    return _cache[k]
def beats(k): return np.array(info[k]["beats"])
def nearest_beat(k,t): b=beats(k); return float(b[np.argmin(abs(b-t))])
def jump(k,approx,win=2.5):
    """approx 근처에서 음량이 가장 크게 뛰는 지점(구간 진입점)"""
    y=load(k); sr=22050; rms=librosa.feature.rms(y=y,frame_length=4096,hop_length=256)[0]
    t=np.arange(len(rms))*256/sr; n=int(0.35*sr/256); d=np.array([rms[i:i+n].mean()-rms[max(0,i-n):i].mean() if i>0 else 0 for i in range(len(rms))])
    m=(t>approx-win)&(t<approx+win); tj=t[m][np.argmax(d[m])]
    on=librosa.onset.onset_detect(y=y,sr=sr,units="time",backtrack=True); c=on[abs(on-tj)<0.4]
    return float(c[np.argmin(abs(c-tj))]) if len(c) else float(tj)
def end_time(k):
    y=load(k); rms=librosa.feature.rms(y=y,frame_length=4096,hop_length=256)[0]; t=np.arange(len(rms))*256/22050
    return float(t[np.where(rms>0.3*np.percentile(rms,95))[0][-1]])

# (컷, 목표 길이, 내레이션 키, 내레이션 시작 오프셋)
def C(id,d,nk=None,no=0.6,**kw): return dict(id=id,d=d,na=nk,na_off=no,**kw)
SEGS=[
 dict(name="무음+바람", track=None, scene=1, cuts=[C("C01",4.0),C("C02",3.6)]),
 dict(name="오프닝", track="daybreak", scene=1, src=("first",0), lead=0.45, cuts=[C("C03",2.6),C("C04",6.6,"C04",1.8)]),
 dict(name="학업·따뜻함", track="home", scene=2, src=("beat",8.0), cuts=[C("C05",5.6,"C05",0.7),C("C06",6.2,"C06",0.7),C("C07",6.0,"C07",0.6),C("C08",3.0)]),
 dict(name="시설·경쾌함", track="familiar", scene=2, src=("jump",8.0), cuts=[C("C09",6.4,"C09",0.8),C("C10",3.0),C("C11",4.6,"C11",0.6),C("C12",6.4,"C12",0.6)]),
 dict(name="특화·드럼", track="journey", scene=3, src=("jump",16.0), cuts=[C("C13",3.0),C("C14",3.4),C("C15",5.6,"C15",1.7)]),
 dict(name="도서실·차분함", track="millie", scene=3, src=("beat",28.5), cuts=[C("C16",4.8,"C16",0.8)]),
 dict(name="동아리·리듬", track="journey", scene=3, src=("jump",80.0), cuts=[C("C17",3.4),C("C18",2.4),C("C19",4.4)]),
 dict(name="등굣길·밝음", track="deserve", scene=4, src=("jump",6.0), cuts=[C("C20",6.6,"C20",0.7),C("C21",3.0)]),
 dict(name="급식·포근함", track="home", scene=4, src=("beat",48.0), cuts=[C("C22",5.6,"C22",0.6),C("C23",2.6)]),
 dict(name="찬양", track="faithlove", scene=4, src=("beat",16.5), cuts=[C("C24a",2.2,"C24",0.5,span=True),C("C24b",2.2)]),
 dict(name="댄스", track="synth", scene=4, src=("beat",10.0), cuts=[C("C24c",3.2)]),
 dict(name="긴장·정적", track=None, scene=4, cuts=[C("C25",4.2)]),
 dict(name="합격·스웰", track="millie", scene=4, src=("jump",108.0), cuts=[C("C26",5.4,"C26",1.2),C("C27",4.0,"C27",0.5,span=True),C("C28",4.4)]),
 dict(name="엔딩·빌드업", track="findyou", scene=5, src=("climax",56.0), cuts=[C("C29a",1.5,"C29",0.5,span=True),C("C29b",1.5),C("C29c",1.5),C("C29d",1.5),C("C30",4.4,"C30",1.0,force_end="climax"),C("C31",7.2,"C31",0.8),C("C32",2.2),C("C33",3.2),C("C34",4.6)]),
]
t=0.0; cuts=[]; music=[]; nas=[]; allbeats=[]
for sg in SEGS:
    k=sg["track"]; tl0=t+sg.get("lead",0)
    if k:
        mode,val=sg["src"]
        if mode=="first": src0=info[k]["first"]
        elif mode=="beat": src0=nearest_beat(k,val)
        elif mode=="jump": src0=jump(k,val)
        elif mode=="climax":
            clim=jump(k,val); pre=sum(c["d"] for c in sg["cuts"] if c["id"] in("C29a","C29b","C29c","C29d","C30")); src0=nearest_beat(k,clim-pre); clim=nearest_beat(k,clim)
        tlb=tl0+(beats(k)-src0); tlb=tlb[tlb>=tl0-1e-6]
    seg_start=t; pieces=[]
    for c in sg["cuts"]:
        mn=1.2
        if c["na"] and not c.get("span"): mn=c["na_off"]+na[c["na"]]["dur"]+0.45
        want=t+max(c["d"],mn); end=want
        if k and c["id"]!="C34":
            if c.get("force_end")=="climax": end=tl0+(clim-src0)
            else:
                cand=tlb[tlb>=t+mn-1e-6]; end=float(cand[np.argmin(abs(cand-want))]) if len(cand) else want
        cuts.append(dict(id=c["id"],start=round(t,3),dur=round(end-t,3),scene=sg["scene"],seg=sg["name"]))
        if c["na"]: nas.append(dict(key=c["na"],start=round(t+c["na_off"],3),dur=na[c["na"]]["dur"],text=na[c["na"]]["text"]))
        t=end
    if k:
        m=dict(track=k,title=info[k]["title"],name=sg["name"],tl_in=round(tl0,3),tl_out=round(t,3),src_in=round(src0,3))
        if sg["src"][0]=="climax":
            # 클라이맥스 7초 뒤(C31 끝)에서 곡의 마지막 마디로 건너뛰어, 로고 컷에서 곡이 자연스럽게 끝나게 한다
            c31=[c for c in cuts if c["id"]=="C31"][0]; c34=[c for c in cuts if c["id"]=="C34"][0]
            jt=c31["start"]+c31["dur"]; E=end_time(k); want_src=E-((c34["start"]+1.2)-jt)
            b=beats(k); i=int(np.argmin(abs(b-(src0+(jt-tl0))))); js=[j for j in range(len(b)) if (j-i)%8==0]; j=min(js,key=lambda j:abs(b[j]-want_src))
            m["tl_out"]=round(jt,3); music.append(m)
            m2=dict(track=k,title=info[k]["title"],name="엔딩·종지",tl_in=round(jt,3),tl_out=round(jt+(info[k]["last"]-b[j])+0.5,3),src_in=round(float(b[j]),3),jump=True,end_src=round(E,3))
            music.append(m2); allbeats+= [float(x) for x in (jt+(b[j:]-b[j]))]
            allbeats+=[float(x) for x in tlb[(tlb>=tl0)&(tlb<jt)]]
        else:
            music.append(m); allbeats+=[float(x) for x in tlb[(tlb>=tl0)&(tlb<t)]]
total=t
json.dump(dict(fps=30,total=round(total,3),cuts=cuts,music=music,na=nas,beats=sorted(round(b,3) for b in allbeats)),open(f"{H}/build/timeline.json","w"),ensure_ascii=False,indent=1)
for c in cuts: print(f'{c["id"]:5s} {c["start"]:7.2f} {c["dur"]:5.2f}  S{c["scene"]} {c["seg"]}')
for m in music: print(m)
print("total",round(total,2))

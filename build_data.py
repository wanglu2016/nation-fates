import urllib.request,json,time,concurrent.futures,pathlib,numpy as np,datetime
ROOT=pathlib.Path(__file__).resolve().parent; CACHE=ROOT/'cache';CACHE.mkdir(exist_ok=True)
CODES=['NY.GDP.PCAP.PP.KD','SP.DYN.LE00.IN','SP.URB.TOTL.IN.ZS','SP.DYN.TFRT.IN','SP.POP.65UP.TO.ZS','NV.IND.TOTL.ZS','SP.POP.TOTL']
def get(code):
    f=CACHE/(code+'.json')
    if f.exists(): return code,json.loads(f.read_text())
    url='https://api.worldbank.org/v2/country/all/indicator/'+code+'?format=json&source=2&date=1990:2023&per_page=20000'
    if code=='countries':url='https://api.worldbank.org/v2/country?format=json&per_page=400'
    for attempt in range(5):
        try:
            d=json.loads(urllib.request.urlopen(url,timeout=90).read()); assert isinstance(d,list) and len(d)==2 and isinstance(d[1],list);assert d[0]['pages']==1
            f.write_text(json.dumps(d));print(code,len(d[1]),flush=True);return code,d
        except Exception as e:
            print(code,'retry',attempt,str(e)[:120],flush=True);time.sleep(1)
    raise RuntimeError(code)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool: raw=dict(pool.map(get,CODES+['countries']))
series={code:{} for code in CODES}
for code in CODES:
    for r in raw[code][1]:
        if r['value'] is not None:series[code].setdefault(r['countryiso3code'],{})[int(r['date'])]=r['value']
regions={'EAS':'东亚与太平洋','ECS':'欧洲与中亚','LCN':'拉丁美洲与加勒比','MEA':'中东与北非','NAC':'北美','SAS':'南亚','SSF':'撒哈拉以南非洲'}
extra={'ISL','LUX','SGP','QAT','BHR','EST','MUS','CYP','BTN','TLS'}
countries=[]; vectors=[]; refs=[]
for c in raw['countries'][1]:
    cid=c['id'];pop=series[CODES[6]].get(cid,{}).get(2023,0)
    if c['region']['id']=='NA' or (pop<2000000 and cid not in extra):continue
    rows=[]; observed=0
    for year in range(1990,2024):
        vals=[]; flags=[]
        for code in CODES[:6]:
            s=series[code].get(cid,{});v=s.get(year);interp=False
            if v is None:
                prev=next((y for y in range(year-1,year-3,-1) if y in s),None);nxt=next((y for y in range(year+1,year+3) if y in s),None)
                if prev and nxt and nxt-prev<=3:v=s[prev]+(s[nxt]-s[prev])*(year-prev)/(nxt-prev);interp=True
            vals.append(round(v,4) if v is not None else None);flags.append(interp)
        rows.append({'y':year,'v':vals,'i':[i for i,f in enumerate(flags) if f]})
        if None not in vals:
            observed+=1;vectors.append([np.log(vals[0])]+vals[1:]);refs.append((len(countries),len(rows)-1))
    if observed<5:
        # Remove rows already queued for this country when insufficient.
        n=observed
        if n:vectors=vectors[:-n];refs=refs[:-n]
        continue
    countries.append({'id':cid,'name':c['name'],'iso2':c['iso2Code'],'region':c['region']['id'],'pop':round(pop),'rows':rows})
X=np.array(vectors);center=np.mean(X,axis=0);scale=np.std(X,axis=0);Z=(X-center)/scale
u,s,vh=np.linalg.svd(Z,full_matrices=False);basis=vh[:3].copy()
# Consistent orientation: prosperity right, longevity up; no substantive labels on PCA axes.
if basis[0,0]<0:basis[0]*=-1
if basis[1,1]<0:basis[1]*=-1
P=Z@basis.T
for (ci,ri),p in zip(refs,P):countries[ci]['rows'][ri]['p']=[round(float(v),5) for v in p]
meta={'title':'国家命运','start':1990,'end':2023,'retrieved':str(datetime.date.today()),'lastUpdated':raw[CODES[0]][0].get('lastupdated'),'source':'World Bank · World Development Indicators','indicators':[{'code':c,'name':raw[c][1][0]['indicator']['value']} for c in CODES[:6]],'center':center.tolist(),'scale':scale.tolist(),'basis':basis.tolist(),'explained':(s[:3]**2/sum(s**2)).tolist(),'regions':regions,'completeRows':len(vectors),'observedValues':sum(sum(v is not None for v in r['v'])-len(r['i']) for c in countries for r in c['rows']),'interpolatedValues':sum(len(r['i']) for c in countries for r in c['rows'])}
(ROOT/'data.json').write_text(json.dumps({'meta':meta,'countries':countries},ensure_ascii=False,separators=(',',':')))
print(json.dumps({'countries':len(countries),'completeRows':len(vectors),'explained':meta['explained'],'regions':sorted(set(c['region'] for c in countries)),'size':(ROOT/'data.json').stat().st_size}))

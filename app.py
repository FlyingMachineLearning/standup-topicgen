import random, re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

st.set_page_config(page_title="Topic Well", page_icon="🪣", layout="wide")

STOPWORDS = {
    "yeah","yes","okay","ok","right","like","know","mean","thing","things",
    "gonna","gotta","wanna","really","just","actually","literally","uh","um",
    "hey","oh","ah","man","dude","people","person","say","said","says","go",
    "going","went","get","got","getting","come","came","look","think","kind",
    "sort","stuff","little","lot","way","time","good"
}

def clean(t):
    t = t.replace("\ufeff"," ").replace("\r\n","\n")
    t = re.sub(r"(?m)^\s*(?:\[\s*)?(?:\d{1,2}:)?\d{1,2}:\d{2}(?:\.\d+)?(?:\s*\])?\s*[-–—:]?\s*","",t)
    t = re.sub(r"\[(?:applause|laughter|laughs|music|cheering|cheers|audience|inaudible)[^\]]*\]"," ",t,flags=re.I)
    return re.sub(r"\s+"," ",t).strip()

def chunks(t, size=150, overlap=25):
    words=t.split()
    step=max(40,size-overlap)
    out=[]
    for i in range(0,len(words),step):
        c=words[i:i+size]
        if len(c)<35 and out: break
        out.append(" ".join(c))
        if i+size>=len(words): break
    return out

def corpus_from(files):
    rows=[]
    for f in files:
        txt=clean(f.getvalue().decode("utf-8",errors="replace"))
        for i,c in enumerate(chunks(txt),1):
            rows.append({"source":f.name,"chunk_id":i,"text":c})
    return pd.DataFrame(rows)

def model_topics(corpus,n_topics,extra):
    sw=sorted(set(ENGLISH_STOP_WORDS)|STOPWORDS|set(extra))
    v=TfidfVectorizer(
        stop_words=sw, lowercase=True, strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']{1,}\b",
        ngram_range=(1,2),
        min_df=2 if len(corpus)>=12 else 1,
        max_df=.93 if len(corpus)>=8 else 1.0,
        sublinear_tf=True, max_features=12000
    )
    X=v.fit_transform(corpus["text"])
    k=max(2,min(n_topics,X.shape[0]-1,X.shape[1]-1))
    nmf=NMF(n_components=k,init="nndsvda",random_state=42,max_iter=700,l1_ratio=.05)
    W=nmf.fit_transform(X)
    H=nmf.components_
    terms=np.asarray(v.get_feature_names_out())
    strength=W.sum(axis=0); total=strength.sum() or 1
    rows=[]
    for ti,comp in enumerate(H):
        ids=comp.argsort()[::-1][:10]
        words=terms[ids].tolist()
        best=int(np.argmax(W[:,ti]))
        rows.append({
            "topic_id":ti+1,
            "label":" / ".join(words[:3]),
            "terms":words,
            "prevalence":float(strength[ti]/total),
            "example":corpus.iloc[best]["text"],
            "source":corpus.iloc[best]["source"]
        })
    return pd.DataFrame(rows).sort_values("prevalence",ascending=False).reset_index(drop=True)

def draw_topic(df, weighted):
    if weighted:
        p=df["prevalence"].to_numpy()
        p=p/p.sum()
        return df.iloc[int(np.random.choice(len(df),p=p))]
    return df.iloc[random.randrange(len(df))]

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem}
.title{font-size:3rem;font-weight:800;line-height:1}
.sub{opacity:.7;margin:.4rem 0 1.5rem}
.card{border:1px solid rgba(127,127,127,.25);border-radius:18px;padding:18px;margin-bottom:14px;min-height:145px}
.rank{font-size:.8rem;opacity:.5}
.ct{font-size:1.22rem;font-weight:750;margin:.35rem 0 .55rem}
.terms{font-size:.92rem;opacity:.78}
.share{font-size:.78rem;opacity:.5;margin-top:.8rem}
.draw{border:2px solid rgba(127,127,127,.3);border-radius:24px;padding:28px;text-align:center;margin:10px 0 16px}
.dsmall{text-transform:uppercase;letter-spacing:.12em;font-size:.75rem;opacity:.5}
.dbig{font-size:2rem;font-weight:800;margin:.5rem 0}
</style>
""",unsafe_allow_html=True)

st.markdown('<div class="title">🪣 Topic Well</div>',unsafe_allow_html=True)
st.markdown('<div class="sub">Broad stand-up subjects, distilled into jumping-off points.</div>',unsafe_allow_html=True)

with st.sidebar:
    st.header("Build the Well")
    n=st.slider("Broad topics",6,30,14)
    weighted=st.toggle("Weight draws by corpus prevalence",value=False)
    show_examples=st.toggle("Show representative transcript chunks",value=False)
    extra_text=st.text_area("Extra filler words to ignore",placeholder="comic name, city name, verbal tic...")

files=st.file_uploader("Upload stand-up transcripts",type=["txt","md"],accept_multiple_files=True)

if not files:
    st.info("Upload your transcript corpus and Topic Well will extract the recurring subject neighborhoods.")
    st.stop()

corpus=corpus_from(files)
if len(corpus)<4:
    st.error("Not enough usable transcript text yet.")
    st.stop()

extra={x.strip().lower() for x in re.split(r"[,\n]",extra_text) if x.strip()}
try:
    topics=model_topics(corpus,n,extra)
except ValueError as e:
    st.error("The vocabulary became too small after filtering. Remove stopwords or upload more text.")
    st.stop()

a,b,c=st.columns(3)
a.metric("Transcripts",len(files))
b.metric("Chunks",len(corpus))
c.metric("Topics",len(topics))
st.divider()

left,right=st.columns([1,2],gap="large")

with left:
    st.subheader("Draw from the Well")
    if "drawn" not in st.session_state:
        st.session_state.drawn=draw_topic(topics,weighted).to_dict()
    if st.button("🎲 Give me a jumping-off point",use_container_width=True):
        st.session_state.drawn=draw_topic(topics,weighted).to_dict()
    d=st.session_state.drawn
    st.markdown(
        f'<div class="draw"><div class="dsmall">Bucket pull</div>'
        f'<div class="dbig">{d["label"]}</div>'
        f'<div>{" · ".join(d["terms"][3:9])}</div></div>',
        unsafe_allow_html=True
    )
    st.caption("The big label is the neighborhood; the smaller terms are nearby doors.")
    if show_examples:
        with st.expander("Where this came from"):
            st.write(d["example"]); st.caption(d["source"])

with right:
    st.subheader("The Well")
    st.caption("Recurring subject neighborhoods, ordered by prevalence.")
    for start in range(0,len(topics),2):
        cols=st.columns(2)
        for off,col in enumerate(cols):
            idx=start+off
            if idx>=len(topics): break
            row=topics.iloc[idx]
            with col:
                st.markdown(
                    f'<div class="card"><div class="rank">#{idx+1}</div>'
                    f'<div class="ct">{row["label"]}</div>'
                    f'<div class="terms">{" · ".join(row["terms"][3:8])}</div>'
                    f'<div class="share">{row["prevalence"]*100:.1f}% of corpus signal</div></div>',
                    unsafe_allow_html=True
                )
                if show_examples:
                    with st.expander("Representative bit"):
                        st.write(row["example"]); st.caption(row["source"])

st.divider()
export=topics[["topic_id","label","terms","prevalence","source"]].copy()
export["terms"]=export["terms"].apply(lambda x:", ".join(x))
st.download_button("Download the Well as CSV",export.to_csv(index=False).encode(),"topic_well.csv","text/csv")

with st.expander("What this MVP is — and isn't"):
    st.markdown("""
**This version answers:** “What broad subjects keep appearing in this stand-up corpus, and can you hand me one when I need somewhere to start?”

It deliberately **doesn't write jokes**. The model stays backstage.

A later version can score topics more intelligently:

`topic usefulness = audience relevance × recency × local relevance × comic fit × novelty`

That is where headlines, venue/city, audience segments, seasonal events, and the comic's own material can enter the system.
""")

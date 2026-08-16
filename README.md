# Topic Well

A one-page Streamlit writing tool that distills a stand-up transcript corpus into broad recurring subjects and lets a comic draw a jumping-off point.

## Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Product distinction
**Topic Modeler = analysis tool.**
**Topic Well = writing interface.**

The model stays backstage. The comic sees broad subject neighborhoods, nearby terms, and a button that gives them somewhere to start.

## Future scoring
A richer version can rank topics using audience relevance, headline recency, geography/venue, comic fit, and novelty.

"""
Noon Fraud Data Lab - synthetic data vs. Noon reference, GAN benchmark and what-if simulator.

Run locally:   streamlit run app.py
Data files (CSV) are read from the app folder or ./data/ unless you upload your own in the sidebar.
"""
from __future__ import annotations

import io
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DEFAULT_FILES = {
    "mine": "FINAL_dataset_for_discriminator_4.csv",
    "ref": "C_REFEREE_real_test.csv",
    "gan": "B_NOON_synthetic_fraud.csv",
}
TARGET = "is_fraud"
PAY = ["card", "wallet", "cod", "applepay"]
PAY_LABEL = {"card": "a credit/debit card", "wallet": "the in-app wallet", "cod": "cash on delivery",
             "applepay": "Apple Pay"}
# Same column order as the professor's A_eval_helper.py
NUMERIC = ["order_amount", "account_age_days", "hour_of_day", "num_orders_last_7d", "delivery_distance_km",
           "discount_applied", "prev_refunds", "device_change", "location_mismatch"]
COLS = NUMERIC + ["payment_method"]
ALL_COLS = COLS + [TARGET]
CONTINUOUS = ["order_amount", "account_age_days", "delivery_distance_km"]
FEATS = NUMERIC + [f"pay_{p}" for p in PAY]
PROF_COLUMNS = NUMERIC + [f"payment_method_{p}" for p in PAY]

SHORT = {
    "order_amount": "Order amount (AED)", "account_age_days": "Account age (days)",
    "hour_of_day": "Hour of day", "num_orders_last_7d": "Orders, last 7 days",
    "delivery_distance_km": "Distance (km)", "discount_applied": "Discount applied",
    "prev_refunds": "Previous refunds", "device_change": "Device change",
    "location_mismatch": "Location mismatch", "payment_method": "Payment method", "is_fraud": "Is fraud",
}

# Colour palette (one colour per meaning, used everywhere)
MINE, NOON = "#7C3AED", "#06B6D4"          # violet = my dataset, cyan = Noon
GREEN, YELLOW, RED = "#16A34A", "#F59E0B", "#DC2626"
INK, MUTED = "#1F2937", "#6B7280"
PAY_COLORS = {"card": "#6366F1", "wallet": "#EC4899", "cod": "#F59E0B", "applepay": "#10B981"}

# Measured while the dataset was being generated (static build log; applies to the shipped files).
BUILD_LOG = [
    dict(v="V1", title="First GAN draft", fraud_rate=8.14, pattern_gap=17.1, legit_auc=0.939, rows=63034,
         what="WGAN-GP fraud + legit, all four fraud types. Fraud prevalence far too high for production.",
         color="#EF4444"),
    dict(v="V2", title="Prevalence fixed", fraud_rate=2.30, pattern_gap=17.1, legit_auc=0.939, rows=59263,
         what="Fraud rows dropped (not relabelled) until prevalence sat at ~2.3%, keeping fraud correlations.",
         color="#F59E0B"),
    dict(v="V3", title="Fraud pattern = Noon", fraud_rate=2.30, pattern_gap=1.8, legit_auc=0.939, rows=59263,
         what="Fraud class retrained on Noon's synthetic-fraud pattern (night spikes, card/wallet, device/location flags).",
         color="#06B6D4"),
    dict(v="V4", title="Legit traffic = Noon", fraud_rate=2.30, pattern_gap=1.8, legit_auc=0.537, rows=59263,
         what="Legitimate class retrained on Noon's real legit traffic; row count and fraud rows untouched.",
         color="#16A34A"),
]

COLUMN_NOTES = {
    "order_amount": "Food-delivery baskets are right-skewed: most orders are small, a long tail of large group orders. "
                    "A realistic generator must reproduce the skew (mean above median), not a bell curve. Fraud skews higher "
                    "because stolen cards are used to extract value quickly.",
    "account_age_days": "Real customer bases mix many established accounts with a steady stream of new sign-ups. "
                        "Fraud concentrates in young accounts (promo abuse, fresh stolen-card accounts), so the fraud curve "
                        "should sit left of the legitimate curve.",
    "hour_of_day": "Legitimate demand peaks at lunch and dinner; fraud peaks overnight (0-4 AM) when manual review is thin "
                   "and cardholders are asleep. The two curves should have visibly different shapes.",
    "num_orders_last_7d": "Typical users order a few times a week. Velocity (many orders in a short time) is a classic "
                          "fraud signal, so the fraud tail should be a little heavier. Counts must be non-negative integers.",
    "delivery_distance_km": "Delivery radii are short (median a few km) with a capped tail. A hard cap (about 30 km) is "
                            "realistic because platforms limit delivery radius.",
    "discount_applied": "A minority of orders use promo codes. Promo abuse raises the fraud share of this flag; a realistic "
                        "dataset keeps the overall rate modest.",
    "prev_refunds": "Most customers have zero refunds. Many past refunds on a young account is a refund-scam signature, "
                    "so the value must stay consistent with account age.",
    "device_change": "A new or changed device is common among takeover and stolen-card fraud, so it should be far more "
                     "frequent among fraud rows than among legitimate rows.",
    "location_mismatch": "Billing/delivery mismatch is rare for honest customers and common for stolen-card orders; "
                         "a realistic dataset shows a strong gap between the classes.",
    "payment_method": "Card and wallet dominate UAE delivery payments; cash on delivery is a minority, and stolen-card "
                      "fraud cannot use COD at all (nothing to charge). Mix shares should be stable across datasets.",
    "is_fraud": "Production fraud prevalence in food delivery is typically 1.5-3%. Anything near 8% looks like "
                "over-sampling, which is exactly what the first draft was marked down for.",
}

PATTERNS = {
    "Night order (0-4 AM)": lambda d: d.hour_of_day <= 4,
    "Account <= 7 days old": lambda d: d.account_age_days <= 7,
    "Account <= 30 days old": lambda d: d.account_age_days <= 30,
    "Discount applied": lambda d: d.discount_applied == 1,
    "More than 2 previous refunds": lambda d: d.prev_refunds > 2,
    "Device changed": lambda d: d.device_change == 1,
    "Location mismatch": lambda d: d.location_mismatch == 1,
    "Pays by card": lambda d: d.payment_method == "card",
    "Pays by wallet": lambda d: d.payment_method == "wallet",
    "Pays by Apple Pay": lambda d: d.payment_method == "applepay",
    "Pays cash on delivery": lambda d: d.payment_method == "cod",
    "Order > AED 300": lambda d: d.order_amount > 300,
    "Distance > 10 km": lambda d: d.delivery_distance_km > 10,
    "5+ orders in last 7 days": lambda d: d.num_orders_last_7d >= 5,
}

st.set_page_config(page_title="Noon Fraud Data Lab", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------------------
st.markdown(
    f"""
<style>
.block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}
.hero {{ background: linear-gradient(120deg,#4C1D95 0%,#7C3AED 45%,#06B6D4 100%); color: white;
        padding: 22px 28px; border-radius: 16px; margin-bottom: 18px; }}
.hero h1 {{ margin: 0 0 6px 0; font-size: 1.9rem; color: white; }}
.hero p {{ margin: 0; opacity: .92; font-size: 0.98rem; }}
.insight {{ background:#F5F3FF; border-left:6px solid {MINE}; border-radius:10px; padding:14px 18px;
           margin: 6px 0 22px 0; color:{INK}; }}
.insight b.t {{ font-size: 1.02rem; }}
.insight ul {{ margin: 8px 0 0 0; padding-left: 20px; }}
.insight li {{ margin-bottom: 6px; line-height: 1.45; }}
.card {{ background:white; border:1px solid #E5E7EB; border-radius:14px; padding:14px 16px; min-height:190px;
        box-shadow:0 1px 3px rgba(0,0,0,.05); }}
.card .lab {{ color:{MUTED}; font-size:.78rem; text-transform:uppercase; letter-spacing:.05em; margin-bottom:8px; }}
.card .row {{ display:flex; justify-content:space-between; align-items:baseline; gap:12px; margin-bottom:4px; }}
.card .who {{ font-size:.85rem; font-weight:600; }}
.card .val {{ font-size:1.45rem; font-weight:700; }}
.card .diff {{ margin-top:8px; font-size:.8rem; color:{MUTED}; border-top:1px dashed #E5E7EB; padding-top:6px; }}
.vcard {{ border:1px solid #E5E7EB; border-top:6px solid var(--c); border-radius:12px; padding:14px 16px;
         background:white; height:100%; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
.vcard .v {{ font-weight:800; font-size:1.15rem; color:var(--c); }}
.vcard .ti {{ font-weight:700; margin:2px 0 8px 0; }}
.vcard .sc {{ font-size:.85rem; margin-bottom:8px; }}
.vcard .wh {{ font-size:.82rem; color:#374151; line-height:1.4; }}
.badge {{ display:flex; align-items:center; justify-content:center; padding:18px 12px; border-radius:14px;
         font-size:1.7rem; font-weight:800; letter-spacing:.06em; color:white; margin-bottom:10px; }}
.badge.green {{ background:{GREEN}; }} .badge.yellow {{ background:{YELLOW}; color:#1F2937; }}
.badge.red {{ background:{RED}; }}
.why {{ border-radius:12px; padding:14px 16px; background:#F9FAFB; border:1px solid #E5E7EB; margin:8px 0 14px 0;
       line-height:1.5; }}
.action {{ border-radius:12px; padding:12px 16px; background:white; border:1px solid #E5E7EB;
          border-left:6px solid var(--c); margin-bottom:8px; }}
.legend-chip {{ display:inline-block; width:12px; height:12px; border-radius:3px; margin-right:6px; vertical-align:middle; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.78rem; font-weight:600;
        background:#EDE9FE; color:#5B21B6; margin-right:6px; }}
div[data-testid="stTabs"] button p {{ font-size: 1.02rem; font-weight: 600; }}
div[data-testid="stTabs"] [role="tablist"] {{ gap: 10px; }}
</style>
""",
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------------
# Data loading + validation
# --------------------------------------------------------------------------------------
class DataError(Exception):
    pass


def find_default(name: str) -> Path | None:
    for base in (APP_DIR, APP_DIR / "data", Path.cwd(), Path.cwd() / "data"):
        p = base / name
        if p.exists():
            return p
    return None


def clean(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in ALL_COLS if c not in df.columns]
    if missing:
        raise DataError(f"{label}: missing required column(s): {', '.join(missing)}")
    df = df[ALL_COLS].copy()
    df["payment_method"] = df["payment_method"].astype(str).str.strip().str.lower()
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    for c in ["discount_applied", "device_change", "location_mismatch", "hour_of_day",
              "num_orders_last_7d", "prev_refunds", "account_age_days"]:
        df[c] = df[c].round().astype(int)
    if len(df) < 50:
        raise DataError(f"{label}: only {len(df)} usable rows after cleaning.")
    return df


@st.cache_data(show_spinner=False)
def parse_bytes(raw: bytes, label: str) -> pd.DataFrame:
    return clean(pd.read_csv(io.BytesIO(raw)), label)


@st.cache_data(show_spinner=False)
def parse_path(path: str, label: str) -> pd.DataFrame:
    return clean(pd.read_csv(path), label)


def get_data(uploaded, key: str, label: str):
    """Returns (dataframe | None, source description)."""
    if uploaded is not None:
        return parse_bytes(uploaded.getvalue(), label), f"uploaded: {uploaded.name}"
    p = find_default(DEFAULT_FILES[key])
    if p is None:
        return None, f"default file {DEFAULT_FILES[key]} not found"
    return parse_path(str(p), label), f"default: {DEFAULT_FILES[key]}"


# --------------------------------------------------------------------------------------
# Statistics helpers
# --------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def pay_share(d: pd.DataFrame) -> pd.Series:
    return d.payment_method.value_counts(normalize=True).reindex(PAY).fillna(0.0)


def hour_share(d: pd.DataFrame) -> pd.Series:
    return d.hour_of_day.value_counts(normalize=True).reindex(range(24)).fillna(0.0)


def column_distances(a: pd.DataFrame, b: pd.DataFrame) -> pd.Series:
    """0 = identical, 1 = completely different. KS for numeric columns, total-variation distance for payment."""
    out = {c: ks_2samp(a[c], b[c]).statistic for c in NUMERIC}
    out["payment_method"] = 0.5 * float((pay_share(a) - pay_share(b)).abs().sum())
    return pd.Series(out)[COLS]


@st.cache_data(show_spinner=False)
def distance_table(mine: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for name, m in (("All rows", slice(None)), ("Legitimate only", 0), ("Fraud only", 1)):
        a = mine if name == "All rows" else mine[mine[TARGET] == m]
        b = ref if name == "All rows" else ref[ref[TARGET] == m]
        rows[name] = column_distances(a, b) if len(a) > 5 and len(b) > 5 else pd.Series(np.nan, index=COLS)
    return pd.DataFrame(rows)


def pattern_table(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, fn in PATTERNS.items():
        ka, kb = int(fn(a).sum()), int(fn(b).sum())
        pa, pb = ka / len(a), kb / len(b)
        lo, hi = wilson(kb, len(b))
        rows.append(dict(Pattern=name, Mine=pa * 100, Noon=pb * 100, Gap=(pa - pb) * 100,
                         Lo=lo * 100, Hi=hi * 100, Inside=bool(lo <= pa <= hi)))
    return pd.DataFrame(rows)


def prep_prof(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly the professor's _prep(): get_dummies on payment_method, aligned to a fixed column list."""
    X = pd.get_dummies(df[NUMERIC + ["payment_method"]], columns=["payment_method"])
    return X.reindex(columns=PROF_COLUMNS, fill_value=0).astype(float)


def featurize(d: pd.DataFrame) -> pd.DataFrame:
    X = d[NUMERIC].astype(float).copy()
    for p in PAY:
        X[f"pay_{p}"] = (d.payment_method == p).astype(float)
    return X[FEATS]


def _lie(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0):
    n = min(len(a), len(b), 3000)
    a, b = a.sample(n, random_state=seed), b.sample(n, random_state=seed)
    X = prep_prof(pd.concat([a, b], ignore_index=True))
    y = np.r_[np.zeros(n), np.ones(n)]
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    acc = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), X, y, cv=cv).mean()
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=0),
                          X, y, cv=cv, method="predict_proba")[:, 1]
    return float(acc), float(roc_auc_score(y, p)), n


@st.cache_data(show_spinner="Running the fake-vs-real test…")
def lie_detector(a: pd.DataFrame, b: pd.DataFrame):
    """Professor's lie_detector idea (RandomForest(100), 3-fold accuracy) - but class-balanced, plus a
    control (Noon-vs-Noon split) so that 'what is 0.5?' is measured rather than assumed."""
    acc, auc, n = _lie(a, b)
    m = min(len(b) // 2, n)
    shuf = b.sample(frac=1, random_state=1)
    c_acc, c_auc, _ = _lie(shuf.iloc[:m], shuf.iloc[m:2 * m])
    return dict(acc=acc, auc=auc, n=n, control_acc=c_acc, control_auc=c_auc, control_n=m)


# --------------------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------------------
def style(fig: go.Figure, h: int = 430, title: str | None = None, legend: bool = True, bottom: int = 105):
    """One consistent look. The legend is pinned to the bottom of the *figure container*, with its own reserved
    margin, so it can never collide with axis titles or tick labels."""
    fig.update_layout(
        template="plotly_white", height=h, paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color=INK),
        title=dict(text=title, x=0.01, xanchor="left", y=0.98, yanchor="top", font=dict(size=16)) if title else None,
        margin=dict(l=70, r=40, t=75 if title else 35, b=bottom if legend else 60),
        showlegend=legend,
        legend=dict(orientation="h", xref="container", yref="container", x=0.5, xanchor="center", y=0.0,
                    yanchor="bottom", bgcolor="rgba(255,255,255,0)", font=dict(size=12), title_text=""),
        hoverlabel=dict(font_size=12),
    )
    fig.update_xaxes(automargin=True, title_standoff=12, gridcolor="#EEF0F4")
    fig.update_yaxes(automargin=True, title_standoff=12, gridcolor="#EEF0F4")
    return fig


def show(fig: go.Figure, key: str):
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def overlay_hist(fig, a, b, col, row=1, colno=1, name_a="My dataset", name_b="Noon", first=False, label=None):
    pooled = pd.concat([a[col], b[col]])
    hi = float(np.percentile(pooled, 99.5))
    lo = float(pooled.min())
    discrete = pooled.nunique() <= 32
    size = 1 if discrete else max((hi - lo) / 40, 1e-9)
    for d, nm, colr in ((a, name_a, MINE), (b, name_b, NOON)):
        fig.add_trace(go.Histogram(
            x=d[col][d[col] <= hi], name=nm, legendgroup=nm, showlegend=first, marker_color=colr, opacity=0.6,
            histnorm="percent", xbins=dict(start=lo - (0.5 if discrete else 0), end=hi + (0.5 if discrete else 0), size=size),
            hovertemplate=f"{nm}<br>{label or SHORT[col]}: %{{x}}<br>%{{y:.2f}}% of orders<extra></extra>",
        ), row=row, col=colno)
    fig.update_xaxes(title_text=label or SHORT[col], row=row, col=colno)
    fig.update_yaxes(title_text="% of orders in bin" if colno == 1 else None, row=row, col=colno)


def insights(title: str, items: list[str]):
    li = "".join(f"<li>{i}</li>" for i in items)
    st.markdown(f'<div class="insight"><b class="t">💡 {title}</b><ul>{li}</ul></div>', unsafe_allow_html=True)


def compare_card(label: str, mine: str, noon: str, diff: str):
    st.markdown(
        f"""<div class="card"><div class="lab">{label}</div>
        <div class="row"><span class="who" style="color:{MINE}">● My dataset</span><span class="val" style="color:{MINE}">{mine}</span></div>
        <div class="row"><span class="who" style="color:{NOON}">● Noon</span><span class="val" style="color:{NOON}">{noon}</span></div>
        <div class="diff">{diff}</div></div>""", unsafe_allow_html=True)


def subset(d: pd.DataFrame, mode: str) -> pd.DataFrame:
    return d if mode == "All rows" else d[d[TARGET] == (1 if mode == "Fraud only" else 0)]


def pct_str(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}%"


# --------------------------------------------------------------------------------------
# Models for the simulator and for the professor comparison
# --------------------------------------------------------------------------------------
def make_model(kind: str):
    if kind.startswith("Random"):
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=0)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, early_stopping=False, random_state=0)


@st.cache_resource(show_spinner="Training the classifier in memory on your dataset…")
def train_sim_model(df: pd.DataFrame, kind: str):
    X, y = featurize(df), df[TARGET].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    model = make_model(kind).fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    bg = X[y == 0].sample(min(150, int((y == 0).sum())), random_state=0).reset_index(drop=True)
    return dict(model=model, yte=yte, pte=pte, bg=bg, roc=roc_auc_score(yte, pte),
                pr=average_precision_score(yte, pte), n_train=len(Xtr), n_test=len(Xte))


def holdout_at(bundle, thr: float):
    y, p = bundle["yte"], bundle["pte"]
    f = p >= thr
    tp, fp = int((f & (y == 1)).sum()), int((f & (y == 0)).sum())
    fn, tn = int((~f & (y == 1)).sum()), int((~f & (y == 0)).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp / (tp + fp) if tp + fp else float("nan"),
                recall=tp / (tp + fn) if tp + fn else float("nan"),
                fpr=fp / (fp + tn) if fp + tn else float("nan"))


def tx_frame(tx: dict) -> pd.DataFrame:
    return pd.DataFrame([tx])


def predict_risk(bundle, frame: pd.DataFrame) -> np.ndarray:
    return bundle["model"].predict_proba(featurize(frame))[:, 1]


def explain(bundle, tx: dict, ref: pd.DataFrame):
    """Single-feature occlusion against 150 typical legitimate orders: contribution = risk(tx) - mean risk when
    that feature is swapped for a typical legitimate value. Positive = this feature pushes the risk up."""
    x = featurize(tx_frame(tx))
    bg = bundle["bg"]
    groups = {c: [c] for c in NUMERIC}
    groups["payment_method"] = [f"pay_{p}" for p in PAY]
    blocks = []
    for cols in groups.values():
        Xr = pd.concat([x] * len(bg), ignore_index=True)
        Xr[cols] = bg[cols].values
        blocks.append(Xr)
    p0 = float(bundle["model"].predict_proba(x)[0, 1])
    P = bundle["model"].predict_proba(pd.concat(blocks, ignore_index=True))[:, 1].reshape(len(groups), len(bg)).mean(axis=1)
    return {f: p0 - q for f, q in zip(groups, P)}


def driver_phrase(f: str, tx: dict, ref: pd.DataFrame, risky: bool = True) -> str:
    legit, fraud = ref[ref[TARGET] == 0], ref[ref[TARGET] == 1]
    v = tx[f]
    if f == "hour_of_day":
        fs = (fraud.hour_of_day == v).mean() * 100
        ls = (legit.hour_of_day == v).mean() * 100
        lead = "a late-night order" if (v <= 4 and risky) else "an order"
        return f"{lead} at {int(v):02d}:00 ({fs:.1f}% of fraud vs {ls:.1f}% of legitimate orders happen in this hour)"
    if f == "order_amount":
        pct = (legit.order_amount < v).mean() * 100
        word = "a high" if (pct >= 50 and risky) else ("an unusually low" if pct < 50 and risky else "an")
        return f"{word} order value of AED {v:,.0f} (above {pct:.0f}% of legitimate orders)"
    if f == "account_age_days":
        pct = (legit.account_age_days < v).mean() * 100
        if not risky:
            return f"an account {int(v)} days old (older than {pct:.0f}% of legitimate accounts)"
        lead = "a brand-new account" if v <= 7 else "a young account"
        return f"{lead}, only {int(v)} days old (newer than {100 - pct:.0f}% of legitimate accounts)"
    if f == "num_orders_last_7d":
        return f"{int(v)} orders in the last 7 days (legitimate median is {legit.num_orders_last_7d.median():.0f})"
    if f == "delivery_distance_km":
        pct = (legit.delivery_distance_km < v).mean() * 100
        return f"a {v:.1f} km delivery distance (longer than {pct:.0f}% of legitimate orders)"
    if f == "prev_refunds":
        return f"{int(v)} previous refund(s) (fraud rows average {fraud.prev_refunds.mean():.2f} vs {legit.prev_refunds.mean():.2f})"
    if f == "payment_method":
        return (f"payment with {PAY_LABEL[v]} ({(fraud.payment_method == v).mean() * 100:.0f}% of fraud vs "
                f"{(legit.payment_method == v).mean() * 100:.0f}% of legitimate orders)")
    label = {"discount_applied": "a promo/discount code", "device_change": "a new or changed device",
             "location_mismatch": "a billing/delivery location mismatch"}[f]
    fs, ls = fraud[f].mean() * 100, legit[f].mean() * 100
    if v == 1:
        return f"{label} ({fs:.0f}% of fraud vs {ls:.0f}% of legitimate orders)"
    return f"no {label.split(' ', 1)[1] if label.startswith('a ') else label} (the usual pattern for {100 - ls:.0f}% of legitimate orders)"


def status_for(risk: float, t_susp: float, t_fraud: float):
    if risk >= t_fraud:
        return "FRAUDULENT", "red", RED
    if risk >= t_susp:
        return "SUSPICIOUS", "yellow", YELLOW
    return "LEGITIMATE", "green", GREEN


def action_steps(status: str, tx: dict, contrib: dict):
    top = [k for k, v in sorted(contrib.items(), key=lambda kv: -kv[1]) if v > 0.01][:3]
    if status == "FRAUDULENT":
        steps = [("Block transaction", "Decline the order and hold fulfilment; do not dispatch a rider.", RED),
                 ("Escalate to the fraud analyst queue", "Review the account's last 30 days; attach the drivers listed above.", RED)]
        if tx["payment_method"] in ("card", "applepay"):
            steps.append(("Contact the cardholder", "Notify the cardholder through the verified channel on file and offer a chargeback-safe refund.", RED))
        if tx["payment_method"] == "wallet":
            steps.append(("Freeze the wallet balance", "Pause wallet spend until the owner re-authenticates.", RED))
        if "device_change" in top:
            steps.append(("Add the device to the watch-list", "Challenge all future logins from this device fingerprint.", RED))
        if tx["discount_applied"] == 1 or "prev_refunds" in top:
            steps.append(("Suspend promo/refund eligibility", "Void the promo and block refund requests until the review is done.", RED))
    elif status == "SUSPICIOUS":
        if tx["payment_method"] == "cod":
            first = ("Call-back / SMS confirmation", "Cash on delivery cannot use 3-D Secure; confirm the order by phone before dispatch.", YELLOW)
        else:
            first = ("Trigger OTP / 3-D Secure", "Step-up authentication before the payment is captured.", YELLOW)
        steps = [first,
                 ("Delay dispatch 10-15 minutes", "Hold the order in a soft queue while the verification completes.", YELLOW),
                 ("Add to monitoring", "Re-score this customer's next three orders with a lower alert threshold.", YELLOW)]
    else:
        steps = [("Auto-approve", "No friction. Let the order flow straight to the restaurant.", GREEN),
                 ("Log the score", "Keep the score in the audit trail for model monitoring.", GREEN)]
    return steps


PRESETS = {
    "☕ Typical lunch order": dict(f_amount=85.0, f_age=400, f_hour=13, f_orders=3, f_pay="card", f_dist=4.0,
                                 f_disc=False, f_ref=0, f_dev=False, f_loc=False),
    "🌙 3 AM stolen card": dict(f_amount=240.0, f_age=60, f_hour=3, f_orders=4, f_pay="card", f_dist=6.0,
                                f_disc=False, f_ref=0, f_dev=True, f_loc=True),
    "🎁 Promo-abuse newcomer": dict(f_amount=45.0, f_age=3, f_hour=2, f_orders=6, f_pay="wallet", f_dist=3.0,
                                    f_disc=True, f_ref=0, f_dev=True, f_loc=False),
    "💸 Refund scammer": dict(f_amount=160.0, f_age=150, f_hour=22, f_orders=5, f_pay="card", f_dist=8.0,
                              f_disc=False, f_ref=3, f_dev=False, f_loc=True),
}


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


# --------------------------------------------------------------------------------------
# Head-to-head: professor-style Random Forest vs gradient boosting
# --------------------------------------------------------------------------------------
def eval_set(ref: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode.startswith("Held-out"):
        legit = ref[ref[TARGET] == 0].reset_index(drop=True)
        perm = np.random.default_rng(0).permutation(len(legit))
        ho = legit.iloc[perm[int(0.7 * len(legit)):]]
        return pd.concat([ho, ref[ref[TARGET] == 1]], ignore_index=True)
    return ref


@st.cache_data(show_spinner="Training both classifiers and scoring them on the Noon reference set…")
def head_to_head(mine: pd.DataFrame, ev: pd.DataFrame):
    tr, rest = train_test_split(mine, test_size=0.4, stratify=mine[TARGET], random_state=0)
    cal, _ = train_test_split(rest, test_size=0.5, stratify=rest[TARGET], random_state=0)
    models = {
        "Random Forest (professor's style)": (RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1), prep_prof),
        "Gradient boosting (my classifier)": (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                                                              early_stopping=False, random_state=0), featurize),
    }
    rows, cms = [], {}
    y = ev[TARGET].values
    for name, (m, fx) in models.items():
        m.fit(fx(tr), tr[TARGET])
        thr = float(np.quantile(m.predict_proba(fx(cal))[:, 1][cal[TARGET].values == 0], 0.99))
        p = m.predict_proba(fx(ev))[:, 1]
        for rule, t in (("default 0.5 (predict)", 0.5), ("tuned: 1% false alerts", thr)):
            yh = (p > t).astype(int)
            tp, fp = int(((yh == 1) & (y == 1)).sum()), int(((yh == 1) & (y == 0)).sum())
            fn, tn = int(((yh == 0) & (y == 1)).sum()), int(((yh == 0) & (y == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append(dict(Model=name, Rule=rule, Threshold=round(t, 3), Recall=rec, Precision=prec, F1=f1,
                             FalsePositives=fp, TP=tp, FN=fn, TN=tn,
                             ROC_AUC=roc_auc_score(y, p), PR_AUC=average_precision_score(y, p)))
            cms[(name, rule)] = [[tn, fp], [fn, tp]]
    return pd.DataFrame(rows), cms


@st.cache_data(show_spinner="Preparing the model file…")
def bundle_bytes(df: pd.DataFrame, kind: str) -> bytes:
    import joblib
    X = prep_prof(df)
    if kind.startswith("Random"):
        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=5, random_state=0, n_jobs=-1)
    else:
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, early_stopping=False, random_state=0)
    m.fit(X, df[TARGET])
    buf = io.BytesIO()
    joblib.dump({"model": m, "columns": list(X.columns)}, buf)
    return buf.getvalue()


# ======================================================================================
# Sidebar: uploaders
# ======================================================================================
with st.sidebar:
    st.markdown("## 📂 Data")
    up_mine = st.file_uploader("My Synthetic Dataset (CSV)", type=["csv"], key="up_mine")
    up_ref = st.file_uploader("Noon Reference Dataset (CSV)", type=["csv"], key="up_ref")
    with st.expander("Optional: Noon fraud-GAN file (Tab 3)"):
        up_gan = st.file_uploader("Noon synthetic-fraud CSV", type=["csv"], key="up_gan")
    st.caption("With no upload, the app loads the shipped defaults "
               f"`{DEFAULT_FILES['mine']}` and `{DEFAULT_FILES['ref']}`.")
    with st.expander("Gemini discriminator scores (timeline)"):
        st.caption("V1 and V4 are the scores you reported. Enter V2/V3 if you have them; blank = not provided.")
        s1 = st.text_input("V1 score /10", "8.5")
        s2 = st.text_input("V2 score /10", "")
        s3 = st.text_input("V3 score /10", "")
        s4 = st.text_input("V4 score /10", "9.9")

try:
    mine, mine_src = get_data(up_mine, "mine", "My Synthetic Dataset")
    ref, ref_src = get_data(up_ref, "ref", "Noon Reference Dataset")
    gan_raw, gan_src = get_data(up_gan, "gan", "Noon fraud-GAN file")
except DataError as e:
    st.error(f"⚠️ {e}")
    st.stop()
except Exception as e:  # unreadable CSV etc.
    st.error(f"⚠️ Could not read a CSV file: {e}")
    st.stop()

if mine is None or ref is None:
    st.warning(
        "Default data files were not found. Place `FINAL_dataset_for_discriminator_4.csv` and "
        "`C_REFEREE_real_test.csv` next to `app.py` (or in `./data/`), or upload both CSVs in the sidebar.\n\n"
        f"Missing: {mine_src if mine is None else ''} {ref_src if ref is None else ''}")
    st.stop()

# Noon's fraud-GAN file: duplicates carry no extra information, so de-duplicate before comparing.
if gan_raw is not None:
    gan_all = gan_raw[gan_raw[TARGET] == 1]
    gan_fraud = gan_all.drop_duplicates().reset_index(drop=True)
    gan_note = f"{gan_src} - {len(gan_all):,} fraud rows, {len(gan_fraud):,} unique (duplicates removed)"
else:
    gan_fraud = ref[ref[TARGET] == 1].reset_index(drop=True)
    gan_note = f"fallback: the {len(gan_fraud)} fraud rows of the Noon reference file ({gan_src})"

mine_f, mine_l = mine[mine[TARGET] == 1], mine[mine[TARGET] == 0]
ref_f, ref_l = ref[ref[TARGET] == 1], ref[ref[TARGET] == 0]

with st.sidebar:
    st.markdown("---")
    st.markdown(f"**My dataset** · {len(mine):,} rows  \n<small>{mine_src}</small>", unsafe_allow_html=True)
    st.markdown(f"**Noon reference** · {len(ref):,} rows  \n<small>{ref_src}</small>", unsafe_allow_html=True)
    st.markdown(f"**Noon fraud GAN** · {len(gan_fraud):,} unique fraud rows  \n<small>{gan_src}</small>", unsafe_allow_html=True)

st.markdown(
    """<div class="hero"><h1>🛡️ Noon Fraud Data Lab</h1>
    <p>GAN-generated food-delivery orders for the UAE: how close are they to Noon's data, can a fraud model
    trained on them be trusted, and what does each transaction look like to that model?</p></div>""",
    unsafe_allow_html=True)

tabs = st.tabs(["📊 1 · Dataset Comparison & GAN Story", "🧪 2 · What-If Simulator", "⚔️ 3 · GAN vs GAN",
                "🔬 4 · Column Deep Dive", "🎓 5 · Professor's Toolkit vs Mine"])

# ======================================================================================
# TAB 1
# ======================================================================================
with tabs[0]:
    dist_tbl = distance_table(mine, ref)
    sim_legit = 100 * (1 - dist_tbl["Legitimate only"].mean())
    sim_fraud = 100 * (1 - dist_tbl["Fraud only"].mean())
    sim_all = 100 * (1 - dist_tbl["All rows"].mean())

    st.subheader("Headline numbers: my synthetic data vs Noon's reference")
    c = st.columns(4)
    with c[0]:
        compare_card("Total rows", f"{len(mine):,}", f"{len(ref):,}", f"My dataset is {len(mine) / len(ref):.1f}× the size")
    with c[1]:
        fr_m, fr_n = mine[TARGET].mean() * 100, ref[TARGET].mean() * 100
        compare_card("Fraud rate", f"{fr_m:.2f}%", f"{fr_n:.2f}%",
                     f"{int(mine[TARGET].sum()):,} vs {int(ref[TARGET].sum()):,} fraud rows")
    with c[2]:
        a_m, a_n = mine.order_amount.mean(), ref.order_amount.mean()
        compare_card("Average order value (AED)", f"{a_m:,.2f}", f"{a_n:,.2f}", f"Gap {a_m - a_n:+.2f} AED ({(a_m / a_n - 1) * 100:+.1f}%)")
    with c[3]:
        d_m, d_n = mine.delivery_distance_km.mean(), ref.delivery_distance_km.mean()
        compare_card("Average delivery distance (km)", f"{d_m:.2f}", f"{d_n:.2f}", f"Gap {d_m - d_n:+.2f} km ({(d_m / d_n - 1) * 100:+.1f}%)")
    st.write("")

    worst = dist_tbl["Legitimate only"].sort_values(ascending=False)
    fr_ratio = mine_f.order_amount.mean() / max(mine_l.order_amount.mean(), 1e-9)
    insights("Key insights", [
        f"<b>Legitimate traffic similarity: {sim_legit:.1f}/100</b> (100 minus the average KS/total-variation distance across the "
        f"10 columns). The widest legitimate gap is <b>{SHORT[worst.index[0]]}</b> at distance {worst.iloc[0]:.3f}.",
        f"Fraud-only similarity is <b>{sim_fraud:.1f}/100</b>, but the Noon reference holds just <b>{len(ref_f)} fraud rows</b>, "
        f"so this number is noisy (see the Wilson intervals in Tab 3).",
        f"Fraud prevalence is {fr_m:.2f}% in my data ({'inside' if 1.5 <= fr_m <= 3.0 else 'outside'} the 1.5-3% production band) vs "
        f"{fr_n:.2f}% in the Noon reference file ({'inside' if 1.5 <= fr_n <= 3.0 else 'outside'} that band; it holds only "
        f"{len(ref_f)} fraud cases, so it is a validation set rather than a prevalence benchmark).",
        f"Inside my data, fraud orders are {fr_ratio:.2f}× the legitimate average basket, "
        f"{'a pattern consistent with' if fr_ratio > 1 else 'unlike'} stolen-card cash-out behaviour.",
    ])

    # ---- Timeline -------------------------------------------------------------------
    st.subheader("How the dataset improved: Version 1 → Version 4")
    scores = []
    for s in (s1, s2, s3, s4):
        try:
            scores.append(float(s))
        except ValueError:
            scores.append(None)
    cols = st.columns(4)
    for col, b, sc in zip(cols, BUILD_LOG, scores):
        with col:
            sc_txt = (f"Gemini score: <b>{sc:.1f}/10</b>" if sc is not None
                      else "Gemini score: <i>not provided</i>")
            st.markdown(
                f"""<div class="vcard" style="--c:{b['color']}"><div class="v">{b['v']}</div>
                <div class="ti">{b['title']}</div><div class="sc">{sc_txt}</div>
                <div class="wh">{b['what']}</div></div>""", unsafe_allow_html=True)
    st.write("")

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.10,
                        subplot_titles=("Gemini score (out of 10)", "Fraud prevalence (%)",
                                        "Fraud pattern gap vs Noon (points)"))
    vs = [b["v"] for b in BUILD_LOG]
    colors = [b["color"] for b in BUILD_LOG]
    sc_plot = [s if s is not None else 0 for s in scores]
    fig.add_trace(go.Bar(x=vs, y=sc_plot, marker_color=colors, showlegend=False,
                         text=[f"{s:.1f}" if s is not None else "n/p" for s in scores], textposition="outside",
                         hovertemplate="%{x}: %{y}/10<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=vs, y=[b["fraud_rate"] for b in BUILD_LOG], marker_color=colors, showlegend=False,
                         text=[f"{b['fraud_rate']:.2f}%" for b in BUILD_LOG], textposition="outside",
                         hovertemplate="%{x}: %{y:.2f}%<extra></extra>"), row=1, col=2)
    fig.add_hrect(y0=1.5, y1=3.0, fillcolor="#16A34A", opacity=0.15, line_width=0, layer="below", row=1, col=2)
    fig.add_trace(go.Bar(x=vs, y=[b["pattern_gap"] for b in BUILD_LOG], marker_color=colors, showlegend=False,
                         text=[f"{b['pattern_gap']:.1f}" for b in BUILD_LOG], textposition="outside",
                         hovertemplate="%{x}: %{y:.1f} pts<extra></extra>"), row=1, col=3)
    fig.update_yaxes(range=[0, 11.5], title_text="Score /10", row=1, col=1)
    fig.update_yaxes(range=[0, 10], title_text="% of rows that are fraud", row=1, col=2)
    fig.update_yaxes(range=[0, 21], title_text="Mean |share gap| (pts)", row=1, col=3)
    style(fig, h=400, title="Build log: what each version fixed", legend=False)
    show(fig, "t1_build")
    st.caption("Green band = realistic production fraud range (1.5-3%). n/p = score not provided. "
               "Prevalence and pattern-gap values are measured at generation time on the shipped files; "
               "legitimate-class real-vs-generated AUC fell from 0.939 (V1-V3) to 0.537 in V4 (a coin-flip control scores 0.503).")

    # ---- Distributions --------------------------------------------------------------
    st.subheader("Overlapping distributions")
    mode1 = st.radio("Rows to compare", ["All rows", "Legitimate only", "Fraud only"], horizontal=True, key="t1_mode")
    a, b = subset(mine, mode1), subset(ref, mode1)
    if len(a) < 5 or len(b) < 5:
        st.info("Not enough rows in this subset to plot.")
    else:
        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.08,
                            subplot_titles=("Order amount", "Account age", "Delivery distance"))
        overlay_hist(fig, a, b, "order_amount", colno=1, first=True)
        overlay_hist(fig, a, b, "account_age_days", colno=2)
        overlay_hist(fig, a, b, "delivery_distance_km", colno=3)
        fig.update_layout(barmode="overlay")
        style(fig, h=440, title=f"My dataset vs Noon: {mode1.lower()}")
        show(fig, "t1_dist")
        st.caption("Each bar = share of that dataset's orders falling in the bin (both use identical bins, so heights are "
                   "directly comparable). X-axes are clipped at the 99.5th percentile of the pooled data.")

    # ---- New insights ---------------------------------------------------------------
    st.subheader("Where exactly do the datasets differ?")
    left, right = st.columns([1.05, 1])
    with left:
        fig = go.Figure()
        for nm, colr in (("All rows", "#6366F1"), ("Legitimate only", GREEN), ("Fraud only", RED)):
            fig.add_trace(go.Bar(x=[SHORT[c] for c in COLS], y=dist_tbl[nm].values, name=nm, marker_color=colr,
                                 hovertemplate="%{x}<br>distance %{y:.3f}<extra>" + nm + "</extra>"))
        fig.update_layout(barmode="group")
        fig.update_xaxes(tickangle=-35)
        fig.update_yaxes(title_text="Distance (0 = identical, 1 = completely different)", rangemode="tozero")
        style(fig, h=520, title="Distance from Noon, column by column", bottom=110)
        show(fig, "t1_dist_bars")
    with right:
        num = NUMERIC + [TARGET]
        diff = mine[num].corr() - ref[num].corr()
        names = [SHORT[c] for c in num]
        zmax = max(0.05, float(np.nanmax(np.abs(diff.values))))
        fig = go.Figure(go.Heatmap(z=diff.values, x=names, y=names, zmin=-zmax, zmax=zmax, colorscale="RdBu_r",
                                   text=np.round(diff.values, 2), texttemplate="%{text}", textfont=dict(size=9),
                                   colorbar=dict(title=dict(text="Δ corr", side="right"), thickness=14, len=0.85, x=1.02),
                                   hovertemplate="%{y} × %{x}<br>Δ correlation %{z:.3f}<extra></extra>"))
        fig.update_xaxes(tickangle=-45, showgrid=False)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        style(fig, h=520, title="Correlation gap (mine − Noon)", legend=False, bottom=130)
        fig.update_layout(margin=dict(l=130, r=90, t=75, b=140))
        show(fig, "t1_corr")
    iu = np.triu_indices(len(num), 1)
    k = int(np.argmax(np.abs(diff.values[iu])))
    pair = (num[iu[0][k]], num[iu[1][k]])
    insights("What the gaps tell you", [
        f"The single biggest relationship gap is <b>{SHORT[pair[0]]} × {SHORT[pair[1]]}</b> "
        f"(correlation differs by {diff.values[iu][k]:+.3f}). Blue = Noon's correlation is stronger; red = mine is stronger.",
        "Distances near 0 on both legitimate and fraud bars mean the generator learned the marginals; the correlation "
        "heatmap checks the relationships between columns, which marginals alone cannot show.",
        f"Overall (all rows) similarity is {sim_all:.1f}/100; it is dominated by the legitimate class "
        f"({len(mine_l) / len(mine):.1%} of my rows).",
    ])

# ======================================================================================
# TAB 2
# ======================================================================================
with tabs[1]:
    if mine[TARGET].sum() < 20:
        st.warning("The simulator needs at least 20 fraud rows in 'My Synthetic Dataset' to train a classifier.")
    else:
        top = st.columns([1.2, 1])
        with top[1]:
            kind = st.radio("Classifier trained in memory", ["Random Forest (300 trees)", "Gradient boosting (HistGB)"],
                            horizontal=True, key="sim_kind")
        bundle = train_sim_model(mine, kind)
        with top[0]:
            st.subheader("What-if fraud-risk simulator")
            st.caption(f"Trained on 75% of my dataset ({bundle['n_train']:,} rows); judged on the unseen 25% "
                       f"({bundle['n_test']:,} rows): ROC-AUC {bundle['roc']:.3f}, PR-AUC {bundle['pr']:.3f}.")

        for k, v in PRESETS["☕ Typical lunch order"].items():
            st.session_state.setdefault(k, v)

        pc = st.columns(len(PRESETS))
        for col, name in zip(pc, PRESETS):
            col.button(name, on_click=apply_preset, args=(name,), width="stretch", key=f"preset_{name}")
        st.write("")

        ctrl, res = st.columns([1.05, 1], gap="large")
        with ctrl:
            st.markdown("##### Transaction details")
            c1, c2 = st.columns(2)
            with c1:
                f_amount = st.number_input("Order amount (AED), type a value", min_value=1.0, max_value=20000.0,
                                           step=5.0, key="f_amount")
                f_age = st.slider("Account age (days)", 0, 2500, key="f_age")
                f_hour = st.slider("Hour of day", 0, 23, key="f_hour")
                f_orders = st.slider("Orders in last 7 days", 0, 30, key="f_orders")
                f_dist = st.slider("Delivery distance (km)", 0.1, 40.0, step=0.1, key="f_dist")
            with c2:
                f_pay = st.selectbox("Payment method", PAY, key="f_pay")
                f_ref = st.slider("Previous refunds", 0, 10, key="f_ref")
                f_disc = st.checkbox("Discount / promo applied", key="f_disc")
                f_dev = st.checkbox("New or changed device", key="f_dev")
                f_loc = st.checkbox("Billing/delivery location mismatch", key="f_loc")
            with st.expander("Alert thresholds (what counts as suspicious / fraudulent)"):
                t_susp = st.slider("SUSPICIOUS from (risk %)", 1, 90, 30) / 100
                t_fraud = st.slider("FRAUDULENT from (risk %)", 2, 99, 60) / 100
                if t_fraud <= t_susp:
                    st.warning("The fraudulent threshold must be above the suspicious one; using suspicious + 1 point.")
                    t_fraud = min(0.99, t_susp + 0.01)
                hs, hf = holdout_at(bundle, t_susp), holdout_at(bundle, t_fraud)
                st.caption(
                    f"On the unseen 25%: at ≥{t_susp:.0%} the model catches {hs['recall']:.0%} of fraud and wrongly flags "
                    f"{hs['fpr'] * 10000:.0f} per 10,000 legitimate orders; at ≥{t_fraud:.0%} it catches {hf['recall']:.0%} "
                    f"and wrongly blocks {hf['fpr'] * 10000:.0f} per 10,000 (precision {hf['precision']:.0%}).")

        tx = dict(order_amount=float(f_amount), account_age_days=int(f_age), hour_of_day=int(f_hour),
                  num_orders_last_7d=int(f_orders), payment_method=f_pay, delivery_distance_km=float(f_dist),
                  discount_applied=int(f_disc), prev_refunds=int(f_ref), device_change=int(f_dev),
                  location_mismatch=int(f_loc))
        risk = float(predict_risk(bundle, tx_frame(tx))[0])
        status, klass, scol = status_for(risk, t_susp, t_fraud)
        contrib = explain(bundle, tx, mine)

        with res:
            st.markdown("##### Live verdict")
            st.markdown(f'<div class="badge {klass}">{status}</div>', unsafe_allow_html=True)
            g = go.Figure(go.Indicator(
                mode="gauge+number", value=risk * 100, number=dict(suffix="%", font=dict(size=44, color=INK)),
                gauge=dict(axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], ticksuffix="%"),
                           bar=dict(color=scol, thickness=0.28),
                           steps=[dict(range=[0, t_susp * 100], color="#BBF7D0"),
                                  dict(range=[t_susp * 100, t_fraud * 100], color="#FDE68A"),
                                  dict(range=[t_fraud * 100, 100], color="#FECACA")],
                           threshold=dict(line=dict(color=INK, width=3), thickness=0.8, value=risk * 100))))
            g.update_layout(height=250, margin=dict(l=30, r=30, t=30, b=10), paper_bgcolor="white",
                            font=dict(family="Inter, Segoe UI, Arial, sans-serif", color=INK))
            show(g, "t2_gauge")
            st.markdown(
                f'<span class="legend-chip" style="background:#BBF7D0"></span>Legitimate (below {t_susp:.0%}) &nbsp;&nbsp; '
                f'<span class="legend-chip" style="background:#FDE68A"></span>Suspicious ({t_susp:.0%}-{t_fraud:.0%}) &nbsp;&nbsp; '
                f'<span class="legend-chip" style="background:#FECACA"></span>Fraudulent (≥{t_fraud:.0%})',
                unsafe_allow_html=True)

        # ---- Why ---------------------------------------------------------------------
        ranked = sorted(contrib.items(), key=lambda kv: -kv[1])
        ups = [(f, v) for f, v in ranked if v > 0.01][:4]
        downs = [(f, v) for f, v in reversed(ranked) if v < -0.01][:3]
        st.markdown("##### Why this verdict?")
        if status == "LEGITIMATE":
            if downs:
                txt = ("Scored as legitimate because " + "; ".join(driver_phrase(f, tx, mine, False) for f, _ in downs[:3]) + ".")
            else:
                txt = "Scored as legitimate: no feature stands out from typical legitimate behaviour."
            if ups:
                txt += " Watch-points (small push upwards): " + "; ".join(driver_phrase(f, tx, mine) for f, _ in ups[:2]) + "."
        else:
            txt = (f"Flagged as {status.lower()} because it is " if ups else f"Flagged as {status.lower()}, with no single dominant driver: ")
            if ups:
                parts = [driver_phrase(f, tx, mine) for f, _ in ups[:3]]
                txt += parts[0] + "".join(f", plus {p}" for p in parts[1:]) + "."
            if downs:
                txt += " Reassuring factors: " + "; ".join(driver_phrase(f, tx, mine, False) for f, _ in downs[:2]) + "."
        st.markdown(f'<div class="why">{txt}</div>', unsafe_allow_html=True)

        cc1, cc2 = st.columns([1.1, 1], gap="large")
        with cc1:
            items = sorted(contrib.items(), key=lambda kv: kv[1])
            fig = go.Figure(go.Bar(
                x=[v * 100 for _, v in items], y=[SHORT[f] for f, _ in items], orientation="h",
                marker_color=[RED if v > 0 else GREEN for _, v in items],
                text=[f"{v * 100:+.1f}" for _, v in items], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:+.1f} risk points<extra></extra>"))
            fig.add_vline(x=0, line_color=INK, line_width=1)
            span = max(5.0, max(abs(v) for _, v in items) * 100 * 1.35)
            fig.update_xaxes(title_text="Δ risk vs typical legitimate value (pct. points)",
                             range=[-span, span], zeroline=False)
            fig.update_yaxes(title_text=None)
            style(fig, h=450, title="What pushes the risk up (red) or down (green)", legend=False, bottom=85)
            fig.update_layout(margin=dict(l=150, r=50, t=75, b=85))
            show(fig, "t2_drivers")
        with cc2:
            st.markdown("##### Recommended operations playbook")
            for title, why, colr in action_steps(status, tx, contrib):
                st.markdown(f'<div class="action" style="--c:{colr}"><b>{title}</b><br>'
                            f'<span style="color:{MUTED};font-size:.88rem">{why}</span></div>', unsafe_allow_html=True)
            st.caption("Illustrative playbook, not Noon's actual policy. Thresholds above are adjustable.")

        # ---- Sensitivity -------------------------------------------------------------
        st.markdown("---")
        st.subheader("Sensitivity analysis: how does risk move?")
        s1c, s2c = st.columns(2, gap="large")
        sweep_ranges = {
            "order_amount": np.round(np.geomspace(10, 1500, 60), 2), "account_age_days": np.arange(0, 1001, 10),
            "hour_of_day": np.arange(24), "num_orders_last_7d": np.arange(0, 21),
            "delivery_distance_km": np.round(np.linspace(0.2, 30, 60), 1), "prev_refunds": np.arange(0, 11),
            "discount_applied": np.array([0, 1]), "device_change": np.array([0, 1]), "location_mismatch": np.array([0, 1]),
            "payment_method": np.array(PAY),
        }
        with s1c:
            sw = st.selectbox("Vary one feature, hold the others at your inputs", COLS, format_func=lambda c: SHORT[c], key="sweep_col")
            vals = sweep_ranges[sw]
            fr = pd.DataFrame([tx] * len(vals))
            fr[sw] = vals
            rs = predict_risk(bundle, fr) * 100
            fig = go.Figure()
            fig.add_hrect(y0=0, y1=t_susp * 100, fillcolor="#BBF7D0", opacity=0.45, line_width=0)
            fig.add_hrect(y0=t_susp * 100, y1=t_fraud * 100, fillcolor="#FDE68A", opacity=0.45, line_width=0)
            fig.add_hrect(y0=t_fraud * 100, y1=100, fillcolor="#FECACA", opacity=0.45, line_width=0)
            for nm, cl in (("Legitimate zone", "#86EFAC"), ("Suspicious zone", "#FCD34D"), ("Fraudulent zone", "#FCA5A5")):
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=cl, symbol="square"), name=nm))
            if sw in ("payment_method", "discount_applied", "device_change", "location_mismatch"):
                fig.add_trace(go.Bar(x=[str(v) for v in vals], y=rs, marker_color=MINE, name="Risk if changed",
                                     text=[f"{r:.0f}%" for r in rs], textposition="outside"))
            else:
                fig.add_trace(go.Scatter(x=vals, y=rs, mode="lines+markers", line=dict(color=MINE, width=3),
                                         marker=dict(size=5), name="Risk if changed"))
                fig.add_trace(go.Scatter(x=[tx[sw]], y=[risk * 100], mode="markers",
                                         marker=dict(size=14, color=INK, symbol="star"), name="Your current input"))
                if sw == "order_amount":
                    fig.update_xaxes(type="log")
            fig.update_yaxes(range=[0, max(25.0, float(rs.max()) * 1.2 + 3) if rs.max() < 80 else 108], title_text="Fraud risk (%)")
            fig.update_xaxes(title_text=SHORT[sw])
            style(fig, h=440, title=f"Risk as {SHORT[sw]} changes", bottom=110)
            show(fig, "t2_sweep")
        with s2c:
            hrs = np.arange(24)
            amts = np.round(np.geomspace(10, 800, 28), 1)
            grid = pd.DataFrame([dict(tx, hour_of_day=int(h), order_amount=float(a)) for a in amts for h in hrs])
            z = (predict_risk(bundle, grid) * 100).reshape(len(amts), len(hrs))
            fig = go.Figure(go.Heatmap(z=z, x=hrs, y=amts, zmin=0, zmax=max(10.0, float(z.max())),
                                       colorscale=[[0, "#22C55E"], [0.3, "#FACC15"], [0.6, "#F97316"], [1, "#B91C1C"]],
                                       colorbar=dict(title=dict(text="Risk %", side="right"), thickness=14, len=0.85),
                                       hovertemplate="Hour %{x}:00<br>AED %{y}<br>Risk %{z:.0f}%<extra></extra>"))
            fig.add_trace(go.Scatter(x=[tx["hour_of_day"]], y=[min(max(tx["order_amount"], amts[0]), amts[-1])], mode="markers",
                                     marker=dict(size=15, color="white", symbol="star", line=dict(color=INK, width=2)),
                                     name="Your input", showlegend=False))
            fig.update_xaxes(title_text="Hour of day", dtick=2)
            fig.update_yaxes(title_text="Order amount (AED, log scale)", type="log")
            style(fig, h=440, title="Risk map: hour × order amount (★ = your input; colour scale auto-fits)", legend=False, bottom=70)
            show(fig, "t2_heat")

        night = predict_risk(bundle, pd.DataFrame([dict(tx, hour_of_day=3)]))[0] * 100
        noonr = predict_risk(bundle, pd.DataFrame([dict(tx, hour_of_day=13)]))[0] * 100
        biggest = ranked[0]
        insights("Key insights", [
            f"Current input scores <b>{risk:.0%}</b> → <b>{status}</b>. The strongest single driver is "
            f"<b>{SHORT[biggest[0]]}</b> ({biggest[1] * 100:+.1f} points vs a typical legitimate value).",
            f"Same basket at 03:00 would score <b>{night:.0f}%</b>, at 13:00 <b>{noonr:.0f}%</b>: "
            f"timing alone moves the risk by {night - noonr:+.0f} points for this customer.",
            "Risk is an uncalibrated model score, not a literal probability: the data is 2.3% fraud and the Random Forest "
            "is class-weighted, so scores run higher than real-world odds. Use thresholds, not the raw number, for policy.",
            "The model is trained on synthetic data, so it learns the generator's rules. Treat the verdicts as a test of "
            "the dataset's realism, not as a production scorecard.",
        ])

# ======================================================================================
# TAB 3
# ======================================================================================
with tabs[2]:
    st.subheader("GAN vs GAN: my fraud output vs Noon's fraud GAN")
    st.caption(f"Noon side: {gan_note}")
    if len(gan_fraud) < 30 or len(mine_f) < 30:
        st.warning("Need at least 30 fraud rows on each side for this benchmark.")
    else:
        pt = pattern_table(mine_f, gan_fraud)
        sim_pat = 100 - float(pt.Gap.abs().mean())
        inside = int(pt.Inside.sum())
        lie = lie_detector(mine_f, gan_fraud)
        dist_ff = column_distances(mine_f, gan_fraud)
        sim_cols = 100 * (1 - dist_ff.mean())
        top_gaps = pt.reindex(pt.Gap.abs().sort_values(ascending=False).index).head(3)

        m = st.columns(4)
        m[0].metric("Pattern-share similarity", f"{sim_pat:.1f}%", help="100 minus the mean absolute gap (percentage points) "
                    f"across {len(pt)} fraud patterns such as night orders, device change, payment mix.")
        m[1].metric("Column-distribution similarity", f"{sim_cols:.1f}%", help="100 minus the mean KS/total-variation distance over 10 columns.")
        m[2].metric(f"Fake-vs-real accuracy (control {lie['control_acc']:.2f})", f"{lie['acc']:.2f}",
                    help="Professor's lie-detector idea, class-balanced. 0.50 = indistinguishable. Control = Noon split against itself.")
        m[3].metric("Patterns inside Noon's 95% band", f"{inside}/{len(pt)}")
        st.write("")
        insights("Key insights", [
            f"<b>Honest headline:</b> the two GANs agree to within {100 - sim_pat:.1f} percentage points on average across "
            f"{len(pt)} fraud patterns ({sim_pat:.1f}% pattern-share similarity). That is a bar for <i>averages of shares</i>; "
            f"it is not the same thing as 'indistinguishable'.",
            f"A random forest shown fraud rows from both sources tells them apart with accuracy <b>{lie['acc']:.2f}</b> "
            f"(ROC-AUC {lie['auc']:.2f}). A Noon-vs-Noon control scores {lie['control_acc']:.2f} (AUC {lie['control_auc']:.2f}), "
            f"so the remaining distance above the control is the real, detectable difference.",
            "Biggest gaps: " + "; ".join(f"<b>{r.Pattern}</b> ({r.Gap:+.1f} pts: mine {r.Mine:.1f}% vs Noon {r.Noon:.1f}%)"
                                         for r in top_gaps.itertuples()) + ".",
            f"Noon's file holds {len(gan_fraud):,} unique fraud rows; the 95% intervals use that sample size, so "
            f"{inside} of {len(pt)} patterns fall inside what Noon's own data would permit.",
        ])

        # Radar ---------------------------------------------------------------------
        radar_defs = [("Avg order<br>amount (AED)", "order_amount", "mean"), ("Avg account<br>age (days)", "account_age_days", "mean"),
                      ("Avg orders<br>last 7d", "num_orders_last_7d", "mean"), ("Avg distance<br>(km)", "delivery_distance_km", "mean"),
                      ("Avg previous<br>refunds", "prev_refunds", "mean"), ("Discount rate", "discount_applied", "mean"),
                      ("Device-change<br>rate", "device_change", "mean"), ("Location-<br>mismatch rate", "location_mismatch", "mean"),
                      ("Night share<br>(0-4 AM)", None, "night")]

        def radar_val(d, col, kind):
            return float((d.hour_of_day <= 4).mean()) if kind == "night" else float(d[col].mean())

        rm = [radar_val(mine_f, c, k) for _, c, k in radar_defs]
        rn = [radar_val(gan_fraud, c, k) for _, c, k in radar_defs]
        mx = [max(a, b, 1e-9) for a, b in zip(rm, rn)]
        names = [n for n, _, _ in radar_defs]
        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            fig = go.Figure()
            for nm, vals, raw, colr, fill in (("My GAN fraud", rm, rm, MINE, "rgba(124,58,237,0.25)"),
                                              ("Noon GAN fraud", rn, rn, NOON, "rgba(6,182,212,0.25)")):
                rel = [v / m_ for v, m_ in zip(vals, mx)]
                fig.add_trace(go.Scatterpolar(r=rel + rel[:1], theta=names + names[:1], name=nm, fill="toself",
                                              line=dict(color=colr, width=3), fillcolor=fill,
                                              customdata=[[v] for v in raw + raw[:1]],
                                              hovertemplate="%{theta}<br>average %{customdata[0]:.3f}<extra>" + nm + "</extra>"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1.05], tickvals=[0.25, 0.5, 0.75, 1.0], tickfont=dict(size=10),
                                                         gridcolor="#E5E7EB"),
                                         angularaxis=dict(tickfont=dict(size=11))))
            style(fig, h=540, title="Average fraud profile (each axis scaled to the larger of the two)", bottom=100)
            fig.update_layout(margin=dict(l=130, r=150, t=85, b=100))
            show(fig, "t3_radar")
        with r2:
            sub = pt.sort_values("Gap")
            fig = go.Figure(go.Bar(x=sub.Gap, y=sub.Pattern, orientation="h",
                                   marker_color=[RED if abs(g) > 5 else (YELLOW if abs(g) > 2 else GREEN) for g in sub.Gap],
                                   text=[f"{g:+.1f}" for g in sub.Gap], textposition="outside", cliponaxis=False,
                                   hovertemplate="%{y}<br>mine minus Noon: %{x:+.2f} pts<extra></extra>"))
            fig.add_vline(x=0, line_color=INK, line_width=1)
            span = max(4.0, float(sub.Gap.abs().max()) * 1.3)
            fig.update_xaxes(range=[-span, span], title_text="My share minus Noon share (percentage points)", zeroline=False)
            style(fig, h=540, title="Pattern gaps (green ≤2, amber 2-5, red >5 points)", legend=False, bottom=85)
            fig.update_layout(margin=dict(l=210, r=60, t=85, b=85))
            show(fig, "t3_gaps")

        # Hours + payments ---------------------------------------------------------------
        h1, h2 = st.columns(2, gap="large")
        with h1:
            hm, hn = hour_share(mine_f) * 100, hour_share(gan_fraud) * 100
            fig = go.Figure()
            fig.add_vrect(x0=-0.5, x1=4.5, fillcolor="#FCA5A5", opacity=0.25, line_width=0, layer="below")
            fig.add_trace(go.Bar(x=hm.index, y=hm.values, name="My GAN fraud", marker_color=MINE))
            fig.add_trace(go.Bar(x=hn.index, y=hn.values, name="Noon GAN fraud", marker_color=NOON))
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="#FCA5A5", symbol="square"),
                                     name="Night window 0-4 AM"))
            fig.update_layout(barmode="group", bargap=0.18)
            fig.update_xaxes(title_text="Hour of day", dtick=1)
            fig.update_yaxes(title_text="% of fraud orders", range=[0, float(max(hm.max(), hn.max())) * 1.2])
            style(fig, h=450, title="Peak fraud hours", bottom=110)
            show(fig, "t3_hours")
        with h2:
            pm, pn = pay_share(mine_f) * 100, pay_share(gan_fraud) * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=PAY, y=pm.values, name="My GAN fraud", marker_color=MINE,
                                 text=[f"{v:.1f}%" for v in pm.values], textposition="outside", textfont=dict(size=10)))
            fig.add_trace(go.Bar(x=PAY, y=pn.values, name="Noon GAN fraud", marker_color=NOON,
                                 text=[f"{v:.1f}%" for v in pn.values], textposition="outside", textfont=dict(size=10)))
            fig.update_layout(barmode="group", bargap=0.35)
            fig.update_xaxes(title_text="Payment method", ticktext=["Card", "Wallet", "Cash on delivery", "Apple Pay"], tickvals=PAY)
            fig.update_yaxes(title_text="% of fraud orders", range=[0, float(max(pm.max(), pn.max())) * 1.25])
            style(fig, h=450, title="Payment gateway choice in fraud", bottom=110)
            show(fig, "t3_pay")

        # Intervals + KS ------------------------------------------------------------------
        i1, i2 = st.columns([1.25, 1], gap="large")
        with i1:
            s = pt.sort_values("Noon")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.Noon, y=s.Pattern, mode="markers", name="Noon share (bar = 95% Wilson interval)",
                                     marker=dict(size=9, color=NOON), error_x=dict(type="data", symmetric=False,
                                     array=(s.Hi - s.Noon).values, arrayminus=(s.Noon - s.Lo).values, color=NOON, thickness=2.5)))
            fig.add_trace(go.Scatter(x=s.Mine, y=s.Pattern, mode="markers", name="My share",
                                     marker=dict(size=12, color=MINE, symbol="diamond", line=dict(color="white", width=1))))
            fig.update_xaxes(title_text="% of fraud orders showing the pattern", rangemode="tozero")
            style(fig, h=560, title="Is my share inside Noon's plausible range?", bottom=110)
            fig.update_layout(margin=dict(l=210, r=40, t=75, b=110))
            show(fig, "t3_wilson")
        with i2:
            d = dist_ff.sort_values()
            fig = go.Figure(go.Bar(x=d.values, y=[SHORT[c] for c in d.index], orientation="h",
                                   marker=dict(color=d.values, colorscale=[[0, "#22C55E"], [0.5, "#FACC15"], [1, "#EF4444"]],
                                               cmin=0, cmax=max(0.3, float(d.max()))),
                                   text=[f"{v:.3f}" for v in d.values], textposition="outside", cliponaxis=False))
            fig.update_xaxes(title_text="Distance (0 = identical)", range=[0, float(d.max()) * 1.3 + 0.02])
            style(fig, h=560, title="Which column differs most?", legend=False, bottom=85)
            fig.update_layout(margin=dict(l=170, r=60, t=75, b=85))
            show(fig, "t3_ks")

# ======================================================================================
# TAB 4
# ======================================================================================
with tabs[3]:
    st.subheader("Column-by-column deep dive")
    q1, q2 = st.columns([1.2, 1])
    col = q1.selectbox("Choose a column", ALL_COLS, format_func=lambda c: f"{c}  ·  {SHORT[c]}", key="dd_col")
    sub_mode = q2.radio("Rows", ["All rows", "Legitimate only", "Fraud only"], horizontal=True, key="dd_mode")
    a, b = subset(mine, sub_mode), subset(ref, sub_mode)
    if len(a) < 5 or len(b) < 5:
        st.info("Not enough rows in this subset to compare.")
    else:
        is_cat = col == "payment_method"
        d_val = (0.5 * float((pay_share(a) - pay_share(b)).abs().sum()) if is_cat
                 else float(ks_2samp(a[col], b[col]).statistic))
        verdict = ("very close" if d_val < 0.03 else "close" if d_val < 0.08 else "noticeably different" if d_val < 0.2 else "far apart")

        if is_cat:
            ta = pd.DataFrame({"My dataset": (pay_share(a) * 100).round(2), "Noon": (pay_share(b) * 100).round(2)})
            ta["Gap (pts)"] = (ta["My dataset"] - ta["Noon"]).round(2)
            ta.index = [f"{p} (% of orders)" for p in ta.index]
            st.dataframe(ta, width="stretch")
            st.caption(f"Most common: mine = {a[col].mode().iloc[0]}, Noon = {b[col].mode().iloc[0]}.")
        else:
            def stats(s):
                return pd.Series({"Mean": s.mean(), "Median": s.median(), "Min": s.min(), "Max": s.max(), "Std. deviation": s.std()})
            ta = pd.DataFrame({"My dataset": stats(a[col]), "Noon": stats(b[col])})
            ta["Gap (mine − Noon)"] = ta["My dataset"] - ta["Noon"]
            ta["Gap %"] = np.where(ta["Noon"] != 0, (ta["My dataset"] / ta["Noon"].replace(0, np.nan) - 1) * 100, np.nan)
            st.dataframe(ta.round(3), width="stretch")

        # chart
        if is_cat:
            pa_, pb_ = pay_share(a) * 100, pay_share(b) * 100
            fig = go.Figure()
            fig.add_trace(go.Bar(x=PAY, y=pa_.values, name="My dataset", marker_color=MINE, text=[f"{v:.1f}%" for v in pa_], textposition="outside"))
            fig.add_trace(go.Bar(x=PAY, y=pb_.values, name="Noon", marker_color=NOON, text=[f"{v:.1f}%" for v in pb_], textposition="outside"))
            fig.update_layout(barmode="group", bargap=0.25)
            fig.update_xaxes(title_text="Payment method")
            fig.update_yaxes(title_text="% of orders", range=[0, float(max(pa_.max(), pb_.max())) * 1.2])
            style(fig, h=440, title=f"Payment method mix: {sub_mode.lower()}", bottom=105)
            show(fig, "t4_cat")
        elif pd.concat([a[col], b[col]]).nunique() <= 32:
            va, vb = a[col].value_counts(normalize=True).sort_index() * 100, b[col].value_counts(normalize=True).sort_index() * 100
            idx = sorted(set(va.index) | set(vb.index))
            fig = go.Figure()
            if col == "hour_of_day":
                fig.add_vrect(x0=-0.5, x1=4.5, fillcolor="#FCA5A5", opacity=0.25, line_width=0, layer="below")
                fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color="#FCA5A5", symbol="square"),
                                         name="Night window 0-4 AM"))
            fig.add_trace(go.Bar(x=idx, y=va.reindex(idx).fillna(0).values, name="My dataset", marker_color=MINE))
            fig.add_trace(go.Bar(x=idx, y=vb.reindex(idx).fillna(0).values, name="Noon", marker_color=NOON))
            fig.update_layout(barmode="group", bargap=0.2)
            fig.update_xaxes(title_text=SHORT[col], dtick=1 if len(idx) <= 24 else 2)
            fig.update_yaxes(title_text="% of orders")
            style(fig, h=440, title=f"{SHORT[col]}: {sub_mode.lower()}", bottom=105)
            show(fig, "t4_disc")
        else:
            fig = make_subplots(rows=2, cols=1, row_heights=[0.74, 0.26], shared_xaxes=True, vertical_spacing=0.08)
            overlay_hist(fig, a, b, col, row=1, colno=1, first=True)
            hi_ = float(np.percentile(pd.concat([a[col], b[col]]), 99.5))
            fig.add_trace(go.Box(x=a[col][a[col] <= hi_], name="Mine", marker_color=MINE, orientation="h", showlegend=False, boxmean=True), row=2, col=1)
            fig.add_trace(go.Box(x=b[col][b[col] <= hi_], name="Noon", marker_color=NOON, orientation="h", showlegend=False, boxmean=True), row=2, col=1)
            fig.update_layout(barmode="overlay")
            fig.update_xaxes(title_text=SHORT[col], row=2, col=1)
            fig.update_yaxes(title_text="% of orders in bin", row=1, col=1)
            style(fig, h=560, title=f"{SHORT[col]}: distribution and spread ({sub_mode.lower()})", bottom=105)
            show(fig, "t4_cont")
            st.caption("Top: histogram (identical bins). Bottom: box plots (line = median, dashed = mean). Clipped at the 99.5th percentile.")

            # ECDF
            qs = np.linspace(0, 1, 201)
            xa, xb = np.quantile(a[col], qs), np.quantile(b[col], qs)
            grid = np.unique(np.r_[xa, xb])
            ca = np.searchsorted(np.sort(a[col].values), grid, side="right") / len(a)
            cb = np.searchsorted(np.sort(b[col].values), grid, side="right") / len(b)
            gx = grid[int(np.argmax(np.abs(ca - cb)))]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=grid, y=ca * 100, name="My dataset", line=dict(color=MINE, width=3)))
            fig.add_trace(go.Scatter(x=grid, y=cb * 100, name="Noon", line=dict(color=NOON, width=3)))
            fig.add_vline(x=gx, line_dash="dot", line_color=RED)
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=RED, dash="dot"),
                                     name=f"Largest gap at {gx:,.1f} (KS = {d_val:.3f})"))
            fig.update_xaxes(title_text=SHORT[col], range=[float(min(xa[0], xb[0])), hi_])
            fig.update_yaxes(title_text="% of orders at or below value", range=[0, 101])
            style(fig, h=420, title="Cumulative distribution: the gap between the curves is the KS distance", bottom=105)
            show(fig, "t4_ecdf")

        # discriminator note
        extra = ""
        if col in CONTINUOUS:
            extra = f" The 95th percentile is {a[col].quantile(.95):,.1f} in my data vs {b[col].quantile(.95):,.1f} in Noon's."
        st.markdown(
            f"""<div class="insight"><b class="t">🕵️ Discriminator note · {SHORT[col]}</b>
            <p style="margin:8px 0 6px 0">{COLUMN_NOTES[col]}</p>
            <p style="margin:0"><b>Computed check:</b> distance to Noon is <b>{d_val:.3f}</b> ({'total-variation' if is_cat else 'KS'}; 0 = identical),
            which is <b>{verdict}</b> on {sub_mode.lower()} ({len(a):,} vs {len(b):,} rows).{extra}</p></div>""",
            unsafe_allow_html=True)

        # fraud lift
        if col != TARGET:
            st.markdown("##### Does this column separate fraud from legitimate orders?")
            lc1, lc2 = st.columns(2)
            for holder, data, nm, colr in ((lc1, mine, "My dataset", MINE), (lc2, ref, "Noon", NOON)):
                with holder:
                    d_fl = column_distances(data[data[TARGET] == 0], data[data[TARGET] == 1])[col] if data[TARGET].sum() > 5 else float("nan")
                    st.metric(f"{nm}: fraud vs legit separation", f"{d_fl:.3f}" if d_fl == d_fl else "n/a",
                              help="Distance between that dataset's fraud and legitimate rows in this column (higher = more informative).")
            sep_m = column_distances(mine_l, mine_f)[col] if len(mine_f) > 5 else float("nan")
            sep_n = column_distances(ref_l, ref_f)[col] if len(ref_f) > 5 else float("nan")
            if sep_m == sep_m and sep_n == sep_n:
                insights("Insight", [
                    f"In my dataset this column separates fraud from legitimate orders by <b>{sep_m:.3f}</b>; in Noon's by <b>{sep_n:.3f}</b>. "
                    + ("A much stronger signal in mine means the detector may lean on this column more than it would on real traffic."
                       if sep_m > sep_n + 0.1 else
                       "Similar strength: a detector trained on my data should use this column the way Noon's data would suggest."
                       if abs(sep_m - sep_n) <= 0.1 else "Noon's data separates the classes more here than mine does."),
                    f"Noon's fraud estimate rests on only {len(ref_f)} fraud rows, so differences under about 0.1 are within noise."])

# ======================================================================================
# TAB 5
# ======================================================================================
with tabs[4]:
    st.subheader("The professor's toolkit vs my classifier")
    st.caption("`A_eval_helper.py` (compare_distributions, lie_detector, score_on_referee) and the professor's Streamlit "
               "`app.py` fraud checker, side by side with what this dashboard does.")

    comp = pd.DataFrame([
        ["Model", "RandomForest(100 trees, random_state=0)", "Whatever is in fraud_model.joblib (your lab classifier)",
         "HistGradientBoosting(200 rounds, lr 0.08). Simulator also offers Random Forest (300 trees)"],
        ["Features", "9 numeric/binary columns + payment_method one-hot (get_dummies)", "Same, rebuilt per row with get_dummies + reindex",
         "Same 9 columns + one-hot payment, built with a fixed column order"],
        ["Fake-vs-real test", "lie_detector: fraud vs fraud, 3-fold accuracy; 0.5 = identical", "-",
         "Same idea, class-balanced, plus a Noon-vs-Noon control and an AUC (Tab 3)"],
        ["Decision rule", "predict() = 0.5 probability cut-off", "predict() → FRAUD/LEGIT, shows probability",
         "Probability bands: LEGITIMATE / SUSPICIOUS / FRAUDULENT; threshold tuned to ~1% false alerts"],
        ["Scoring data", "score_on_referee on the full referee set", "Single typed-in transaction",
         "Held-out 30% of legit referee rows + all referee fraud (avoids reusing rows the GAN learned from)"],
        ["Class imbalance", "Not addressed", "Not addressed", "Threshold calibrated on a separate slice; class weights in the RF simulator"],
        ["Explainability", "-", "Risk % only", "Per-feature occlusion drivers, written reasons, ops playbook, sweeps"],
        ["Deployment", "Notebook helper", "Loads a pre-saved joblib file", "Trains in memory from the CSV; can also export a joblib for the professor's app"],
    ], columns=["Aspect", "A_eval_helper.py", "Professor's app.py", "My dashboard / classifier"])
    head = "".join(f"<th>{c}</th>" for c in comp.columns)
    body = "".join("<tr>" + f"<td><b>{r[0]}</b></td>" + "".join(f"<td>{x}</td>" for x in r[1:]) + "</tr>" for r in comp.values)
    st.markdown(
        f"""<style>.cmp {{width:100%;border-collapse:collapse;font-size:.88rem}}
        .cmp th {{background:#EDE9FE;text-align:left;padding:8px 10px;border-bottom:2px solid {MINE}}}
        .cmp td {{padding:8px 10px;border-bottom:1px solid #E5E7EB;vertical-align:top;line-height:1.4}}
        .cmp td:last-child {{background:#F5F3FF}}</style><table class="cmp"><tr>{head}</tr>{body}</table>""",
        unsafe_allow_html=True)

    st.markdown("#### Live head-to-head on the Noon reference set")
    ev_mode = st.radio("Evaluate on", ["Held-out 30% of legitimate rows + all fraud (recommended)", "Entire reference file (professor's style)"],
                       horizontal=True, key="ev_mode")
    ev = eval_set(ref, ev_mode)
    if mine[TARGET].sum() < 20 or ev[TARGET].sum() < 1:
        st.warning("Need ≥20 fraud rows in my dataset and ≥1 in the reference to run the head-to-head.")
    else:
        res_df, cms = head_to_head(mine, ev)
        best = res_df.sort_values("F1", ascending=False).iloc[0]
        rf_def = res_df[(res_df.Model.str.startswith("Random")) & (res_df.Rule.str.startswith("default"))].iloc[0]
        hg_tun = res_df[(res_df.Model.str.startswith("Gradient")) & (res_df.Rule.str.startswith("tuned"))].iloc[0]
        insights("Key insights", [
            f"Scored on <b>{len(ev):,} Noon rows</b> ({int(ev[TARGET].sum())} fraud, base rate {ev[TARGET].mean():.2%}). "
            f"Precision depends on that base rate, so compare models, not absolute precision, with other datasets.",
            f"Professor-style Random Forest at the default 0.5 cut-off: recall <b>{rf_def.Recall:.2f}</b>, "
            f"precision <b>{rf_def.Precision:.2f}</b>, {int(rf_def.FalsePositives)} false positives. My gradient-boosted model at the tuned "
            f"1% false-alert threshold: recall <b>{hg_tun.Recall:.2f}</b>, precision <b>{hg_tun.Precision:.2f}</b>, "
            f"{int(hg_tun.FalsePositives)} false positives.",
            f"Best F1 here: <b>{best.Model}</b> with rule '{best.Rule}' (F1 {best.F1:.2f}). The decision rule often matters "
            "more than the algorithm: with 2% fraud a 0.5 cut-off is conservative and misses fraud.",
            f"At the tuned 1% false-alert rule the highest recall is <b>{res_df[res_df.Rule.str.startswith('tuned')].sort_values('Recall', ascending=False).iloc[0].Model}</b> "
            f"({res_df[res_df.Rule.str.startswith('tuned')].Recall.max():.2f}); at the default 0.5 rule it is "
            f"<b>{res_df[res_df.Rule.str.startswith('default')].sort_values('Recall', ascending=False).iloc[0].Model}</b>.",
            "With very few fraud rows in the reference, each fraud case moves recall by several points. Treat differences "
            "smaller than ~0.1 as noise.",
        ])

        fig = go.Figure()
        order = [(m_, r_) for m_ in res_df.Model.unique() for r_ in res_df.Rule.unique()]
        labels = [f"{'Random Forest' if m_.startswith('Random') else 'HistGB'}<br>{'0.5 cut-off' if r_.startswith('default') else '1% false alerts'}" for m_, r_ in order]
        for metric, colr in (("Recall", "#EF4444"), ("Precision", "#6366F1"), ("F1", "#10B981")):
            vals = [float(res_df[(res_df.Model == m_) & (res_df.Rule == r_)][metric].iloc[0]) for m_, r_ in order]
            fig.add_trace(go.Bar(x=labels, y=vals, name=metric, marker_color=colr, text=[f"{v:.2f}" for v in vals], textposition="outside"))
        fig.update_layout(barmode="group", bargap=0.3)
        fig.update_yaxes(title_text="Score (0-1)", range=[0, 1.15])
        fig.update_xaxes(title_text="Classifier and decision rule")
        style(fig, h=450, title="Recall, precision and F1 on the Noon reference", bottom=105)
        show(fig, "t5_bars")

        cm1, cm2 = st.columns(2, gap="large")
        for holder, key_, ttl, cs, kk in ((cm1, (res_df.Model.unique()[0], res_df.Rule.unique()[0]), "Professor-style: Random Forest @ 0.5", "Purples", "a"),
                                          (cm2, (res_df.Model.unique()[1], res_df.Rule.unique()[1]), "Mine: HistGB @ 1% false alerts", "Greens", "b")):
            with holder:
                z = np.array(cms[key_])
                fig = go.Figure(go.Heatmap(z=np.log1p(z), x=["Predicted legit", "Predicted fraud"], y=["Actual legit", "Actual fraud"],
                                           text=z, texttemplate="%{text:,}", textfont=dict(size=18), showscale=False, colorscale=cs,
                                           hovertemplate="%{y} / %{x}: %{text:,}<extra></extra>"))
                fig.update_yaxes(autorange="reversed")
                style(fig, h=340, title=ttl, legend=False, bottom=60)
                fig.update_layout(margin=dict(l=110, r=30, t=75, b=60))
                show(fig, f"t5_cm_{kk}")
        st.caption("Confusion matrices read [[TN FP][FN TP]]; colour depth is log-scaled so small cells stay visible.")

        show_df = res_df[["Model", "Rule", "Threshold", "Recall", "Precision", "F1", "FalsePositives", "ROC_AUC", "PR_AUC"]].copy()
        st.dataframe(show_df.round(3), width="stretch", hide_index=True)

    st.markdown("#### Use my data inside the professor's `app.py`")
    st.caption("Generates the `fraud_model.joblib` that the professor's app expects: `{'model': clf, 'columns': [...]}`, "
               "trained on all of my dataset with her exact `get_dummies` feature layout.")
    kind5 = st.radio("Model to export", ["Random Forest (100 trees)", "Gradient boosting (HistGB)"], horizontal=True, key="exp_kind")
    if st.checkbox("Prepare fraud_model.joblib", key="exp_prep"):
        data_bytes = bundle_bytes(mine, kind5)
        st.download_button(f"⬇️ Download fraud_model.joblib ({len(data_bytes) / 1e6:.1f} MB)", data=data_bytes,
                           file_name="fraud_model.joblib", mime="application/octet-stream")

    st.markdown("#### What this teaches")
    st.markdown(
        "- **Same inputs, different judge.** Both setups use the same 10 inputs, so differences come from the learner and the decision rule, not the features.\n"
        "- **Her lie-detector answers 'does it look like Noon?'; the classifier answers 'is it useful?'.** A dataset needs to pass both.\n"
        "- **Her `score_on_referee` is an honest test only if the referee rows were never used for generation.** "
        "That is why this dashboard defaults to the held-out 30% of legitimate referee rows.")

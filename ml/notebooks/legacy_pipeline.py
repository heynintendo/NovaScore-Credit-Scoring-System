
!pip -q install --upgrade pandas==2.2.2 pyarrow tqdm scikit-learn

# ---- Imports ----
import os, math, json, gc, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from google.colab import files

# ---- Repro & device ----
def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ---- Config: 90-day window ----
WINDOW_DAYS = 90
T_WEEKS = math.ceil(WINDOW_DAYS / 7)  # 90d -> 13 weekly buckets

# ---- 2) Upload your two parquet files (robust & case-insensitive) ----
print("Please upload BOTH files: merged_trip_details_0.parquet and transaction-0.parquet")

uploaded = {}  # will accumulate across multiple uploads if needed

def _pick_path(uploaded_dict, include_terms):
    # returns the first filename whose lowercase name contains all include_terms and ends with parquet/parquett
    for name in uploaded_dict.keys():
        low = name.strip().lower()
        if all(term in low for term in include_terms) and (low.endswith(".parquet") or low.endswith(".parquett")):
            return name
    return None

TRIPS_FPATH = None
TXN_FPATH   = None
attempt = 1

while (TRIPS_FPATH is None) or (TXN_FPATH is None):
    if attempt > 1:
        print("\nMissing file(s). Upload the missing one(s) now…")
    newly = files.upload()  # you can select one or both here
    uploaded.update(newly)

    # Try to find trips file (handles names like merged_trip_details_0.parquet or similar)
    TRIPS_FPATH = TRIPS_FPATH or _pick_path(uploaded, ["merged","trip","details"])
    TRIPS_FPATH = TRIPS_FPATH or _pick_path(uploaded, ["trip","details"])  # fallback

    # Try to find transactions file (handles transaction-0.parquet, transactions.parquet, etc.)
    TXN_FPATH = TXN_FPATH or _pick_path(uploaded, ["transaction"])
    TXN_FPATH = TXN_FPATH or _pick_path(uploaded, ["transactions"])  # fallback

    print(f"Detected → trips: {TRIPS_FPATH} | transactions: {TXN_FPATH}")
    attempt += 1

print("Using files →", TRIPS_FPATH, "and", TXN_FPATH)

# ---- Read & minimal type fixes ----
trips = pd.read_parquet(TRIPS_FPATH)
txns  = pd.read_parquet(TXN_FPATH)

trips.columns = [c.strip().lower() for c in trips.columns]
txns.columns  = [c.strip().lower() for c in txns.columns]

def combine_dt(df, date_col, time_col, new_col):
    d = pd.to_datetime(df[date_col], errors="coerce")
    t = pd.to_timedelta(df[time_col], errors="coerce")
    df[new_col] = d + t
    return df

trips = combine_dt(trips, "trip_date", "trip_start_time", "trip_start_ts")
trips = combine_dt(trips, "trip_date", "trip_end_time",   "trip_end_ts")
txns["transaction_dt"] = pd.to_datetime(
    txns["transaction_date"].astype(str) + " " + txns["transaction_time"].astype(str),
    errors="coerce"
)

for col in ["trip_duration","trip_distance","fare_amount","tip_amount","trip_rating","safety_score"]:
    if col in trips.columns:
        trips[col] = pd.to_numeric(trips[col], errors="coerce")
if "cancellation_flag" in trips.columns:
    trips["cancellation_flag"] = pd.to_numeric(trips["cancellation_flag"], errors="coerce").fillna(0).astype(int)
if "incident_flag" in trips.columns:
    trips["incident_flag"] = trips["incident_flag"].fillna(False).astype(bool)

for col in ["transaction_amount","balance_after_transaction"]:
    if col in txns.columns:
        txns[col] = pd.to_numeric(txns[col], errors="coerce")

max_dt = pd.Series([
    trips["trip_end_ts"].max(),
    trips["trip_start_ts"].max(),
    txns["transaction_dt"].max()
]).max()
anchor_end = pd.to_datetime(max_dt)
anchor_start = anchor_end - pd.Timedelta(days=WINDOW_DAYS)

trips_90 = trips[(trips["trip_start_ts"]>=anchor_start) & (trips["trip_start_ts"]<=anchor_end)].copy()
txns_90  = txns[(txns["transaction_dt"]>=anchor_start) & (txns["transaction_dt"]<=anchor_end)].copy()

print(f"Rows in {WINDOW_DAYS}-day window -> trips: {len(trips_90)} txns: {len(txns_90)}")

BAD_STATUSES = {"chargeback","default","failed","fraud","reversed","disputed","late","bounced","cancelled"}
def is_bad(status):
    if pd.isna(status): return False
    s = str(status).strip().lower()
    return any(bad in s for bad in BAD_STATUSES)

user_bad = (txns_90.assign(is_bad=txns_90["transaction_status"].map(is_bad))
                    .groupby("user_id", as_index=False)["is_bad"].max())
user_bad.rename(columns={"is_bad":"y"}, inplace=True)
print("Label prevalence (positive rate @90d):", float(user_bad["y"].mean()))

trip_g = trips_90.groupby("user_id", dropna=False)
trip_agg = trip_g.agg(
    trips_count=("trip_id", "count"),
    trip_dur_mean=("trip_duration","mean"),
    trip_dur_sum=("trip_duration","sum"),
    trip_dist_sum=("trip_distance","sum"),
    fare_sum=("fare_amount","sum"),
    tip_sum=("tip_amount","sum"),
    rating_mean=("trip_rating","mean"),
    safety_mean=("safety_score","mean"),
    cancel_rate=("cancellation_flag","mean"),
    incident_rate=("incident_flag","mean") if "incident_flag" in trips_90.columns else ("cancellation_flag","mean")
).reset_index()

def topk_ratio(df, by_col, k=5, prefix="pm_"):
    if by_col not in df.columns: return pd.DataFrame()
    vals = df[by_col].astype(str).value_counts().nlargest(k).index.tolist()
    outs = []
    for v in vals:
        name = f"{prefix}{v}".lower().replace(" ","_")
        outs.append(df.assign(tmp=(df[by_col].astype(str)==v).astype(int))
                      .groupby("user_id")["tmp"].mean().rename(name))
    return pd.concat(outs, axis=1) if outs else pd.DataFrame()

pm = topk_ratio(trips_90, "payment_method", k=4, prefix="pay_")
rt = topk_ratio(trips_90, "route_type",     k=3, prefix="route_")

tab_trip = (trip_agg
            .merge(pm, left_on="user_id", right_index=True, how="left")
            .merge(rt, left_on="user_id", right_index=True, how="left"))

txn_g = txns_90.groupby("user_id", dropna=False)
txn_agg = txn_g.agg(
    txn_count=("transaction_id","count"),
    txn_amt_sum=("transaction_amount","sum"),
    txn_amt_mean=("transaction_amount","mean"),
    txn_amt_std=("transaction_amount","std"),
    bal_after_mean=("balance_after_transaction","mean")
).reset_index()

dc = topk_ratio(txns_90, "device_channel",   k=3, prefix="devc_")
mc = topk_ratio(txns_90, "merchant_category",k=5, prefix="mcat_")

tab_txn = (txn_agg
           .merge(dc, left_on="user_id", right_index=True, how="left")
           .merge(mc, left_on="user_id", right_index=True, how="left"))

tab = tab_trip.merge(tab_txn, on="user_id", how="outer").fillna(0.0)

def week_index(ts):
    delta = (anchor_end.normalize() - ts.normalize()).days
    return (T_WEEKS - 1) - (delta // 7)

def build_weekly(df, time_col, agg_map, id_col="user_id"):
    if df.empty: return pd.DataFrame(columns=[id_col,"week_idx", *agg_map.keys()])
    df = df.copy()
    df["week_idx"] = df[time_col].dt.floor("D").map(week_index)
    df = df[(df["week_idx"]>=0) & (df["week_idx"]<=T_WEEKS-1)]
    g = df.groupby([id_col,"week_idx"]).agg(agg_map).reset_index()
    return g

trip_week = build_weekly(
    trips_90, "trip_start_ts",
    dict(trips=("trip_id","count"),
         dist=("trip_distance","sum"),
         dur=("trip_duration","sum"),
         cancels=("cancellation_flag","sum"),
         rating=("trip_rating","mean"),
         earnings=("fare_amount","sum"))
)

txn_week = build_weekly(
    txns_90, "transaction_dt",
    {
        "transaction_amount": "sum",
        "transaction_id": "count",
        "merchant_id": pd.Series.nunique
    }
).rename(columns={"transaction_amount":"spend","transaction_id":"txns","merchant_id":"merchants"})

D_SEQ_COLS = ["trips","dist","dur","cancels","rating","earnings","spend","txns","merchants"]

users = pd.DataFrame({"user_id": sorted(set(tab["user_id"]) | set(user_bad["user_id"]))})
users = users.merge(user_bad, on="user_id", how="left").fillna({"y":0})

if "city" in txns_90.columns:
    dom_city = (txns_90.assign(city=txns_90["city"].astype(str))
                        .groupby(["user_id","city"]).size().reset_index(name="n"))
    dom_city = dom_city.sort_values(["user_id","n"], ascending=[True,False]).drop_duplicates("user_id")[["user_id","city"]]
    top_cities = txns_90["city"].astype(str).value_counts().nlargest(10).index.tolist()
    dom_city["city_grp"] = dom_city["city"].where(dom_city["city"].isin(top_cities), "other")
else:
    dom_city = pd.DataFrame({"user_id": users["user_id"], "city_grp":"all"})
users = users.merge(dom_city[["user_id","city_grp"]], on="user_id", how="left").fillna({"city_grp":"other"})

w = pd.merge(trip_week, txn_week, on=["user_id","week_idx"], how="outer")
for c in D_SEQ_COLS:
    if c not in w.columns:
        w[c] = 0.0
w[D_SEQ_COLS] = w[D_SEQ_COLS].fillna(0.0)

seq_stats = {}
for c in D_SEQ_COLS:
    col = w[c].astype(float).values
    mu = np.nanmean(col); sd = np.nanstd(col) + 1e-6
    seq_stats[c] = (mu, sd)
    w[c] = (w[c] - mu) / sd

uid2idx = {u:i for i,u in enumerate(users["user_id"].tolist())}
X_seq = np.zeros((len(users), T_WEEKS, len(D_SEQ_COLS)), dtype="float32")
for _, row in w.iterrows():
    i = uid2idx.get(row["user_id"]); t = int(row["week_idx"])
    if i is None or t<0 or t>(T_WEEKS-1): continue
    X_seq[i, t, :] = row[D_SEQ_COLS].to_numpy(dtype="float32")

!pip -q install node2vec networkx gensim

import networkx as nx
from node2vec import Node2Vec

USE_GRAPH = True
G_DIM = 64

if USE_GRAPH:
    G = nx.Graph()
    for _, row in txns_90.iterrows():
        u = row["user_id"]
        m = f"m_{row['merchant_id']}"
        if pd.isna(u) or pd.isna(row["merchant_id"]): continue
        G.add_edge(str(u), str(m))
    print(f"Graph built with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    n2v = Node2Vec(G, dimensions=G_DIM, walk_length=20, num_walks=100, workers=2, seed=42)
    model_n2v = n2v.fit(window=10, min_count=1, batch_words=4)
    X_graph = np.zeros((len(users), G_DIM), dtype="float32")
    for i, u in enumerate(users["user_id"].astype(str).tolist()):
        if u in model_n2v.wv:
            X_graph[i] = model_n2v.wv[u]
        else:
            X_graph[i] = np.zeros(G_DIM, dtype="float32")
else:
    X_graph = np.zeros((len(users), G_DIM), dtype="float32")

tab_num_cols = [c for c in tab.columns if c!="user_id"]
X_tab = tab[tab_num_cols].astype(float).fillna(0.0).values.astype("float32")

scaler_tab = StandardScaler()
X_tab = scaler_tab.fit_transform(X_tab).astype("float32")

y = users["y"].astype(int).values
groups = users["city_grp"].astype("category")
group_codes = groups.cat.codes.values
group_categories = groups.cat.categories.tolist()

def _maybe_stratify(y_vec, min_per_class=2):
    vals, cnts = np.unique(y_vec, return_counts=True)
    ok = (len(vals) >= 2) and (cnts.min() >= min_per_class)
    if not ok:
        print(f"[Split] Falling back to NON-stratified split. Class counts = {dict(zip(vals.tolist(), cnts.tolist()))}")
        return None
    return y_vec

idx = np.arange(len(users))

tr_idx, te_idx = train_test_split(
    idx, test_size=0.15, random_state=42, shuffle=True,
    stratify=_maybe_stratify(y)
)

tr_idx, va_idx = train_test_split(
    tr_idx, test_size=0.1765, random_state=42, shuffle=True,
    stratify=_maybe_stratify(y[tr_idx])
)

def take(arr, idcs): return arr[idcs] if isinstance(arr, np.ndarray) else arr.iloc[idcs]

X_tab_tr, X_tab_va, X_tab_te = X_tab[tr_idx], X_tab[va_idx], X_tab[te_idx]
X_seq_tr, X_seq_va, X_seq_te = X_seq[tr_idx], X_seq[va_idx], X_seq[te_idx]
X_g_tr,   X_g_va,   X_g_te   = X_graph[tr_idx], X_graph[va_idx], X_graph[te_idx]
y_tr, y_va, y_te = y[tr_idx], y[va_idx], y[te_idx]
grp_tr, grp_va, grp_te = group_codes[tr_idx], group_codes[va_idx], group_codes[te_idx]
uid_tr, uid_va, uid_te = users["user_id"].values[tr_idx], users["user_id"].values[va_idx], users["user_id"].values[te_idx]

print(f"Split sizes -> train {len(tr_idx)} | val {len(va_idx)} | test {len(te_idx)}")
print("Positive rate (train/val/test):", float(y_tr.mean()), float(y_va.mean()), float(y_te.mean()))

try:
    import lightgbm as lgb
    from lightgbm import early_stopping, log_evaluation
except Exception:
    !pip -q install lightgbm
    import lightgbm as lgb
    from lightgbm import early_stopping, log_evaluation

dtr = lgb.Dataset(X_tab_tr, label=y_tr)
dva = lgb.Dataset(X_tab_va, label=y_va, reference=dtr)

lgb_params = dict(
    objective="binary", metric="auc",
    learning_rate=0.05, num_leaves=64, max_depth=-1,
    feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1,
    verbosity=-1,
    scale_pos_weight=(1 - y_tr.mean())/max(y_tr.mean(), 1e-6)
)

cb = [early_stopping(stopping_rounds=200), log_evaluation(period=100)]

gbm = lgb.train(
    params=lgb_params,
    train_set=dtr,
    valid_sets=[dva],
    num_boost_round=2000,
    callbacks=cb
)

p_te_lgb = gbm.predict(X_tab_te, num_iteration=gbm.best_iteration)
lgb_test_auc = roc_auc_score(y_te, p_te_lgb) if len(np.unique(y_te)) > 1 else float("nan")
print("LightGBM best_iteration:", gbm.best_iteration)
print("LightGBM val AUC:", gbm.best_score["valid_0"]["auc"])
print("LightGBM test AUROC:", lgb_test_auc)

class NovaDS(Dataset):
    def __init__(self, X_tab, X_seq, X_g, y, grp):
        self.X_tab, self.X_seq, self.X_g = X_tab, X_seq, X_g
        self.y = y.astype("float32"); self.grp = grp.astype("int64")
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return (torch.tensor(self.X_tab[i]), torch.tensor(self.X_seq[i]),
                torch.tensor(self.X_g[i]), torch.tensor(self.y[i]), torch.tensor(self.grp[i]))

tr_loader = DataLoader(NovaDS(X_tab_tr, X_seq_tr, X_g_tr, y_tr, grp_tr),
                       batch_size=512, shuffle=True, num_workers=2, pin_memory=(DEVICE=="cuda"))
va_loader = DataLoader(NovaDS(X_tab_va, X_seq_va, X_g_va, y_va, grp_va),
                       batch_size=1024, shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))
te_loader = DataLoader(NovaDS(X_tab_te, X_seq_te, X_g_te, y_te, grp_te),
                       batch_size=1024, shuffle=False, num_workers=2, pin_memory=(DEVICE=="cuda"))

class FeatureTokenizer(nn.Module):
    def __init__(self, n_num, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_num, d_model))
        self.bias   = nn.Parameter(torch.zeros(n_num, d_model))
        self.ln     = nn.LayerNorm(d_model)
    def forward(self, x_num):
        tok = x_num.unsqueeze(-1) * self.weight + self.bias
        return self.ln(tok)

class FTTransformer(nn.Module):
    def __init__(self, n_num, d_model=256, n_heads=8, n_layers=2, dropout=0.1):
        super().__init__()
        self.tok = FeatureTokenizer(n_num, d_model)
        self.cls = nn.Parameter(torch.randn(1,1,d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads,
                                         dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.out_ln = nn.LayerNorm(d_model)
    def forward(self, x_num):
        B = x_num.size(0)
        x = self.tok(x_num)
        x = torch.cat([self.cls.expand(B,1,-1), x], dim=1)
        h = self.encoder(x)
        return self.out_ln(h[:,0,:])

class Chomp1d(nn.Module):
    def __init__(self, cs): super().__init__(); self.cs = cs
    def forward(self, x): return x[:,:,:-self.cs].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout):
        super().__init__()
        pad = (k-1)*dilation
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation),
            Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.res = nn.Conv1d(cin, cout, 1) if cin!=cout else nn.Identity()
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.net(x) + self.res(x))

class TCNEncoder(nn.Module):
    def __init__(self, in_dim, d_model=128, n_blocks=3, k=3, dropout=0.1):
        super().__init__()
        ch = in_dim; layers=[]
        for b in range(n_blocks):
            layers.append(TemporalBlock(ch, d_model, k, dilation=2**b, dropout=dropout))
            ch = d_model
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x_seq):
        x = x_seq.transpose(1,2)
        h = self.tcn(x)
        h = self.pool(h).squeeze(-1)
        return self.ln(h)

class HybridModel(nn.Module):
    def __init__(self, n_tab, d_tab=256, d_seq=128, g_dim=64,
model = train_model()

model.eval(); te_y=[]; te_p=[]; te_grp=[]
with torch.no_grad():
    for x_tab, x_seq, x_g, yy, gg in te_loader:
        x_tab=x_tab.to(DEVICE); x_seq=x_seq.to(DEVICE); x_g=x_g.to(DEVICE)
        p = torch.sigmoid(model(x_tab, x_seq, x_g)).cpu().numpy()
        te_p.append(p); te_y.append(yy.numpy()); te_grp.append(gg.numpy())
te_y = np.concatenate(te_y); te_p = np.concatenate(te_p); te_grp = np.concatenate(te_grp)

test_auc = roc_auc_score(te_y, te_p) if len(np.unique(te_y))>1 else float("nan")
delta_tpr = delta_tpr_at_threshold(te_y, te_p, te_grp, thr=0.5)
print("\n=== TEST METRICS (Hybrid FT+TCN) ===")
print("AUROC:", round(float(test_auc), 4))
print("ΔTPR (avg pairwise @0.5):", round(delta_tpr, 4))

def solve_score_params(pd_anchors=(0.01, 0.20), score_anchors=(900, 650)):
    p1, p2 = pd_anchors
    s1, s2 = score_anchors
    x1, x2 = math.log(p1/(1-p1)), math.log(p2/(1-p2))
    B = (s2 - s1) / (-x2 + x1)
    A = s1 + B * x1
    return A, B

def pd_to_score(pd, A, B):
    pd = np.clip(pd, 1e-6, 1-1e-6)
    return A - B * np.log(pd/(1-pd))

def decision_band(score):
    if score >= 800: return "Auto-approve, large limit"
    if score >= 700: return "Standard approve, medium limit"
    if score >= 600: return "Manual review, small limit"
    return "Decline + tips (repayment coaching)"

A, B = solve_score_params(pd_anchors=(0.01, 0.20), score_anchors=(900, 650))

model.eval(); all_y=[]; all_p=[]; all_grp=[]
with torch.no_grad():
    for x_tab, x_seq, x_g, yy, gg in DataLoader(
        NovaDS(X_tab, X_seq, X_graph, y, group_codes),
        batch_size=1024, shuffle=False
    ):
        x_tab=x_tab.to(DEVICE); x_seq=x_seq.to(DEVICE); x_g=x_g.to(DEVICE)
        p = torch.sigmoid(model(x_tab, x_seq, x_g)).cpu().numpy()
        all_p.append(p); all_y.append(yy.numpy()); all_grp.append(gg.numpy())

all_y = np.concatenate(all_y)
all_p = np.concatenate(all_p)
all_grp = np.concatenate(all_grp)

def _logit(x):
    x = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x / (1 - x))
def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-z))

p600 = _inv_logit((A - 600) / B)
p800 = _inv_logit((A - 800) / B)
q20 = float(np.quantile(all_p, 0.20))
q80 = float(np.quantile(all_p, 0.80))
u20, u80 = _logit(q20), _logit(q80)
v600, v800 = _logit(p600), _logit(p800)
den = u80 - u20
if abs(den) < 1e-8:
    a = 1.0; b = v600 - u20
else:
    a = (v800 - v600) / den
    b = v600 - a * u20
cal_logits = a * _logit(all_p) + b
all_p = np.clip(_inv_logit(cal_logits), 1e-6, 1 - 1e-6)

scores = pd_to_score(all_p, A, B)
scores = np.clip(scores, 300.0, 940.0)
bands = np.array([decision_band(s) for s in scores])

share_400_600 = float(((scores >= 400) & (scores < 600)).mean())
share_600_800 = float(((scores >= 600) & (scores < 800)).mean())
share_800_900 = float(((scores >= 800) & (scores < 900)).mean())
print(f"Score distribution ({WINDOW_DAYS}d) -> 400–600: {share_400_600:.1%} | 600–800: {share_600_800:.1%} | 800–900: {share_800_900:.1%}")

pred_df = pd.DataFrame({
    "user_id": users["user_id"].values,
    "y_true": all_y.astype(int),
    "pd90": all_p,
    "novascore": scores.round(1),
    "decision_band": bands,
    "group": pd.Categorical.from_codes(all_grp, categories=group_categories).astype(str)
})

pred_df.to_csv("predictions_90d_full.csv", index=False)
print("\nSaved file: predictions_90d_full.csv")
try:
    files.download("predictions_90d_full.csv")
except Exception as _e:
    pass

np.save("pred_prob_90d_full.npy", all_p)
np.save("score_90d_full.npy", scores)
print("Saved: pred_prob_90d_full.npy, score_90d_full.npy")

summary = pred_df["decision_band"].value_counts().rename_axis("band").reset_index(name="count")
print(f"\nDecision band distribution (ALL riders, {WINDOW_DAYS}d):")
print(summary)

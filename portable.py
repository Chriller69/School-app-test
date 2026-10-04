
import json
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb


def export(pipe, folder):
    # write what the pipeline learned as plain files: the preparation as numbers, the trees in XGBoost's own format
    prep, clf = pipe[0], pipe[-1]
    num_cols, cat_cols = prep.transformers_[0][2], prep.transformers_[1][2]
    imputer, scaler = prep.named_transformers_["num"][0], prep.named_transformers_["num"][-1]
    onehot = prep.named_transformers_["cat"]
    rare = onehot.infrequent_categories_ or [None] * len(cat_cols)
    spec = {
        "numeric": [{"name": c, "median": float(imputer.statistics_[i]), "mean": float(scaler.mean_[i]),
                     "scale": float(scaler.scale_[i])} for i, c in enumerate(num_cols)],
        "categorical": [{"name": c,
                         "frequent": [str(v) for v in onehot.categories_[i] if rare[i] is None or v not in rare[i]],
                         "has_infrequent": rare[i] is not None} for i, c in enumerate(cat_cols)],
    }
    Path(folder, "preprocess.json").write_text(json.dumps(spec, indent=1))
    clf.get_booster().save_model(str(Path(folder, "booster.json")))


class Model:
    def __init__(self, folder):
        self.spec = json.loads(Path(folder, "preprocess.json").read_text())
        self.booster = xgb.Booster()
        self.booster.load_model(str(Path(folder, "booster.json")))

    def prepare(self, df):
        parts = []
        for n in self.spec["numeric"]:   # numbers: fill gaps with the median, then scale
            x = pd.to_numeric(df[n["name"]], errors="coerce").astype(float).fillna(n["median"])
            parts.append(((x - n["mean"]) / n["scale"]).to_numpy()[:, None])
        for c in self.spec["categorical"]:   # categories: one 0/1 column per level, rare and unseen levels share one column
            v = df[c["name"]].astype(str).to_numpy()
            parts.append(np.stack([v == level for level in c["frequent"]], axis=1).astype(float))
            if c["has_infrequent"]:
                parts.append((~np.isin(v, c["frequent"])).astype(float)[:, None])
        return np.hstack(parts)

    def predict_proba(self, df):
        return self.booster.predict(xgb.DMatrix(self.prepare(df)))

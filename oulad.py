# # Прогнозування академічної успішності на OULAD

import json
import os
import platform
import sys
import time
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import sklearn
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore", category=UserWarning)


#Налаштування


DATA_DIR = os.environ.get("OULAD_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", "results")
QUICK = os.environ.get("QUICK", "0") == "1"
TUNE = os.environ.get("TUNE", "0") == "1"
TUNE_ITER = int(os.environ.get("TUNE_ITER", 15))

RANDOM_STATE = 42
WINDOWS = [30, 60, 90, None]        # None = весь курс (FULL)
ABLATION_WINDOWS = [60, None]
CV_FOLDS = 5
CV_REPEATS = 1 if QUICK else 3
BOOTSTRAP = 200 if QUICK else 1000
EXCLUDE_EXAM = True                 # результати фінального іспиту НЕ використовуються як ознаки
KEY = ["code_module", "code_presentation", "id_student"]
MOD_KEY = ["code_module", "code_presentation"]

os.makedirs(OUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")


def wname(w):
    return "FULL" if w is None else f"W{w}"


def out(name):
    return os.path.join(OUT_DIR, name)


env = {
    "python": sys.version.split()[0], "platform": platform.platform(),
    "processor": platform.processor(), "cpu_count": os.cpu_count(),
    "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__,
    "scipy": scipy.__version__, "joblib": joblib.__version__,
    "matplotlib": matplotlib.__version__, "seaborn": sns.__version__,
    "random_state": RANDOM_STATE, "windows": [wname(w) for w in WINDOWS],
    "cv_folds": CV_FOLDS, "cv_repeats": CV_REPEATS, "tune": TUNE, "exclude_exam": EXCLUDE_EXAM,
}
json.dump(env, open(out("environment.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(json.dumps(env, indent=2, ensure_ascii=False))

#Завантаження даних і перевірка пропусків (2.1)

def read(name, **kw):
    # У цій версії OULAD пропуски записані як "?"
    return pd.read_csv(os.path.join(DATA_DIR, f"{name}.csv"), na_values=["?"], **kw)


courses = read("courses")
student_info = read("studentInfo")
registration = read("studentRegistration")
assessments = read("assessments")
student_assessment = read("studentAssessment")
student_vle = read("studentVle",
                   usecols=["code_module", "code_presentation", "id_student", "date", "sum_click"],
                   dtype={"id_student": "int32", "date": "int16", "sum_click": "int32"})

tables = {"courses": courses, "studentInfo": student_info, "studentRegistration": registration,
          "assessments": assessments, "studentAssessment": student_assessment,
          "studentVle": student_vle}
rows = []
for tname, t in tables.items():
    miss = t.isna().sum()
    rows.append({"table": tname, "rows": len(t), "cols": t.shape[1],
                 "missing_by_column": json.dumps(miss[miss > 0].to_dict(), ensure_ascii=False)})
missing_tbl = pd.DataFrame(rows)
print(missing_tbl.to_string(index=False))
missing_tbl.to_csv(out("tables_and_missing.csv"), index=False, encoding="utf-8-sig")

n_rec, n_stud = len(student_info), student_info["id_student"].nunique()
multi = (student_info.groupby("id_student").size() > 1).sum()
print(f"\nЗаписів (студент×курс): {n_rec}; унікальних студентів: {n_stud}; "
      f"студентів із кількома записами: {multi}")
assert not student_info.duplicated(KEY).any(), "ключ (модуль, презентація, студент) не унікальний"


#2. Побудова ознак для заданого вікна спостереження (2.1)

sa_all = student_assessment.merge(
    assessments[["id_assessment", "code_module", "code_presentation", "assessment_type"]],
    on="id_assessment", how="left")
due_all = assessments[["id_assessment", "code_module", "code_presentation",
                       "assessment_type", "date"]].copy()
if EXCLUDE_EXAM:
    sa_all = sa_all[sa_all["assessment_type"] != "Exam"]
    due_all = due_all[due_all["assessment_type"] != "Exam"]

CAT_COLS = ["gender", "region", "highest_education", "imd_band", "age_band",
            "disability", "code_module"]
NUM_COLS = ["num_of_prev_attempts", "studied_credits", "date_registration",
            "total_clicks", "active_days", "mean_score", "n_assessments",
            "share_submitted", "has_submissions"]
FEATURE_GROUPS = {
    "demographic": ["gender", "region", "highest_education", "imd_band", "age_band",
                    "disability", "num_of_prev_attempts", "studied_credits"],
    "course": ["code_module", "date_registration"],
    "vle_activity": ["total_clicks", "active_days"],
    "assessments": ["mean_score", "n_assessments", "share_submitted", "has_submissions"],
}
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
assert set(ALL_FEATURES) == set(CAT_COLS + NUM_COLS)


def build_dataset(cutoff):
    sa, vle, due = sa_all, student_vle, due_all
    if cutoff is not None:
        sa = sa[sa["date_submitted"] <= cutoff]
        vle = vle[vle["date"] <= cutoff]
        due = due[due["date"] <= cutoff]

    assess_agg = (sa.groupby(KEY).agg(mean_score=("score", "mean"),
                                      n_assessments=("id_assessment", "count")).reset_index())
    vle_agg = (vle.groupby(KEY).agg(total_clicks=("sum_click", "sum"),
                                    active_days=("date", "nunique")).reset_index())
    due_n = due.groupby(MOD_KEY).size().rename("n_due").reset_index()

    df = (student_info.merge(registration, on=KEY, how="left")
          .merge(assess_agg, on=KEY, how="left")
          .merge(vle_agg, on=KEY, how="left")
          .merge(due_n, on=MOD_KEY, how="left"))
    assert len(df) == len(student_info), "злиття змінило кількість рядків"

    if cutoff is not None:
        df = df[df["date_unregistration"].isna() | (df["date_unregistration"] > cutoff)]

    for c in ["n_assessments", "total_clicks", "active_days", "n_due"]:
        df[c] = df[c].fillna(0)
    df["has_submissions"] = (df["n_assessments"] > 0).astype(int)
    df["mean_score"] = df["mean_score"].fillna(0)
    df["share_submitted"] = np.where(df["n_due"] > 0,
                                     np.minimum(df["n_assessments"] / df["n_due"].where(df["n_due"] > 0, 1), 1.0),
                                     0.0)
    df["imd_band"] = df["imd_band"].fillna("Unknown")
    df["target"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    return df.reset_index(drop=True)


#3. Єдине розбиття по студентах (однакові тестові студенти для всіх вікон)

full_df = build_dataset(None)
splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
_, te_idx = next(splitter.split(full_df, full_df["target"], full_df["id_student"]))
TEST_IDS = set(full_df.loc[te_idx, "id_student"])

datasets = {}
summary = []
for w in WINDOWS:
    d = full_df if w is None else build_dataset(w)
    d["is_test"] = d["id_student"].isin(TEST_IDS)
    datasets[w] = d
    tr, te = d[~d["is_test"]], d[d["is_test"]]
    assert set(tr["id_student"]).isdisjoint(set(te["id_student"]))
    summary.append({
        "window": wname(w), "records": len(d), "students": d["id_student"].nunique(),
        "train_records": len(tr), "test_records": len(te),
        "class0_risk": int((d["target"] == 0).sum()), "class1_success": int((d["target"] == 1).sum()),
        "share_risk": round((d["target"] == 0).mean(), 4),
    })
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))
summary_df.to_csv(out("dataset_summary.csv"), index=False, encoding="utf-8-sig")

feat_list = pd.DataFrame([{"feature": f, "group": g,
                           "type": "categorical" if f in CAT_COLS else "numeric"}
                          for g, fs in FEATURE_GROUPS.items() for f in fs])
feat_list.to_csv(out("feature_list.csv"), index=False, encoding="utf-8-sig")
print("\nОзнак (до кодування):", len(ALL_FEATURES))

#4. Моделі та метрики (2.2, 2.3)

def make_pre(features, scale):
    num = [f for f in features if f in NUM_COLS]
    cat = [f for f in features if f in CAT_COLS]
    num_tr = (Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
              if scale else SimpleImputer(strategy="median"))
    return ColumnTransformer([("num", num_tr, num),
                              ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat)])


def make_models(features):
    return {
        "LogisticRegression": Pipeline([("pre", make_pre(features, True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
        "DecisionTree": Pipeline([("pre", make_pre(features, False)),
            ("clf", DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE))]),
        "RandomForest": Pipeline([("pre", make_pre(features, False)),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=12,
                                           random_state=RANDOM_STATE, n_jobs=-1))]),
        "GradientBoosting": Pipeline([("pre", make_pre(features, False)),
            ("clf", GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                               random_state=RANDOM_STATE))]),
    }


SEARCH_SPACES = {
    "LogisticRegression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "DecisionTree": {"clf__max_depth": [3, 5, 7, 10, 15, 20, None],
                     "clf__min_samples_leaf": [1, 5, 10, 20, 50]},
    "RandomForest": {"clf__n_estimators": [100, 200, 300], "clf__max_depth": [8, 12, 16, None],
                     "clf__min_samples_leaf": [1, 3, 5, 10], "clf__max_features": ["sqrt", 0.3, 0.5]},
    "GradientBoosting": {"clf__n_estimators": [100, 200, 300],
                         "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
                         "clf__max_depth": [2, 3, 4, 5], "clf__subsample": [0.7, 0.85, 1.0]},
}


def maybe_tune(models, Xtr, ytr, gtr):
    """Підбір гіперпараметрів ЛИШЕ на навчальній частині (групова 3-кратна CV)."""
    best = {}
    if not TUNE:
        return models, best
    splits = list(StratifiedGroupKFold(3, shuffle=True, random_state=RANDOM_STATE).split(Xtr, ytr, gtr))
    for name, pipe in models.items():
        rs = RandomizedSearchCV(pipe, SEARCH_SPACES[name], n_iter=TUNE_ITER, scoring="f1_macro",
                                cv=splits, random_state=RANDOM_STATE, n_jobs=1, refit=False)
        rs.fit(Xtr, ytr)
        pipe.set_params(**rs.best_params_)
        best[name] = {k: (v if v is None or isinstance(v, (int, float, str)) else str(v))
                      for k, v in rs.best_params_.items()}
    return models, best


def metrics_from_proba(y, proba, thr=0.5):
    y = np.asarray(y)
    pred = (np.asarray(proba) >= thr).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum()); fn = int(((y == 1) & (pred == 0)).sum())
    sd = lambda a, b: a / b if b else 0.0
    p1, r1 = sd(tp, tp + fp), sd(tp, tp + fn)
    p0, r0 = sd(tn, tn + fn), sd(tn, tn + fp)
    f1, f0 = sd(2 * p1 * r1, p1 + r1), sd(2 * p0 * r0, p0 + r0)
    return {"accuracy": (tp + tn) / len(y),
            "roc_auc": roc_auc_score(y, proba) if len(set(y)) > 1 else np.nan,
            "precision_0": p0, "recall_0": r0, "f1_0": f0,
            "precision_1": p1, "recall_1": r1, "f1_1": f1, "macro_f1": (f0 + f1) / 2,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def bootstrap_ci(y, pred, B=BOOTSTRAP, seed=RANDOM_STATE):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(B, len(y)))
    yt, pt = y[idx], pred[idx]
    tp = ((yt == 1) & (pt == 1)).sum(1); tn = ((yt == 0) & (pt == 0)).sum(1)
    fp = ((yt == 0) & (pt == 1)).sum(1); fn = ((yt == 1) & (pt == 0)).sum(1)
    acc = (tp + tn) / len(y)
    rec0 = tn / np.maximum(tn + fp, 1)
    f1_1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    f1_0 = 2 * tn / np.maximum(2 * tn + fn + fp, 1)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"accuracy_ci": q(acc), "recall_0_ci": q(rec0), "macro_f1_ci": q((f1_0 + f1_1) / 2)}


def corrected_ttest(d, n_train, n_test):
    """Скоригований t-тест Nadeau & Bengio (2003) для повторної k-кратної CV."""
    d = np.asarray(d, dtype=float)
    k = len(d)
    var = d.var(ddof=1)
    if k < 2 or var == 0:
        return np.nan, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=k - 1)))


def mcnemar_exact(y, pred_a, pred_b):
    y = np.asarray(y)
    a_ok, b_ok = pred_a == y, pred_b == y
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) > 0 else 1.0
    return b, c, float(p)


def risk_threshold_table(y, proba1, recalls=(0.80, 0.85, 0.90, 0.95)):
    """Поріг для виявлення ризику: ризик = (1 - P(успіх)) >= t."""
    y_risk = (np.asarray(y) == 0)
    s = 1 - np.asarray(proba1)
    prec, rec, thr = precision_recall_curve(y_risk, s)
    rows = []
    for target in recalls:
        ok = np.where(rec[:-1] >= target)[0]
        if len(ok) == 0:
            continue
        i = ok[-1]
        flagged = s >= thr[i]
        rows.append({"target_recall_risk": target, "risk_score_threshold": float(thr[i]),
                     "precision_risk": float(prec[i]), "recall_risk": float(rec[i]),
                     "flagged": int(flagged.sum()),
                     "missed_risk": int((y_risk & ~flagged).sum()),
                     "false_alarms": int((~y_risk & flagged).sum())})
    return pd.DataFrame(rows), (prec, rec)


#5. Один експеримент: холдаут + повторна групова CV + тести значущості

def run_experiment(df, features, label, extras=True, cv_repeats=CV_REPEATS):
    tr, te = df[~df["is_test"]], df[df["is_test"]]
    Xtr, ytr, gtr = tr[features], tr["target"], tr["id_student"]
    Xte, yte = te[features], te["target"]
    models, best_params = maybe_tune(make_models(features), Xtr, ytr, gtr)

    #повторна групова CV на навчальній частині (однакові розбиття для всіх моделей)
    all_splits = []
    for rep in range(cv_repeats):
        cv = StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE + rep)
        all_splits += [(rep, k, a, b) for k, (a, b) in enumerate(cv.split(Xtr, ytr, gtr))]
    fold_rows = []
    for name, pipe in models.items():
        for rep, k, a, b in all_splits:
            m = clone(pipe).fit(Xtr.iloc[a], ytr.iloc[a])
            met = metrics_from_proba(ytr.iloc[b], m.predict_proba(Xtr.iloc[b])[:, 1])
            fold_rows.append({"window": label, "model": name, "rep": rep, "fold": k,
                              **{x: met[x] for x in ["accuracy", "macro_f1", "roc_auc", "recall_0", "precision_0"]}})
    folds = pd.DataFrame(fold_rows)
    cv_sum = (folds.groupby("model")[["accuracy", "macro_f1", "roc_auc", "recall_0", "precision_0"]]
              .agg(["mean", "std"]))
    cv_sum.columns = [f"cv_{a}_{b}" for a, b in cv_sum.columns]
    cv_sum = cv_sum.reset_index().assign(window=label)

    #відкладена вибірка
    hold, probas, preds, fitted = [], {}, {}, {}
    for name, pipe in models.items():
        t0 = time.perf_counter()
        pipe.fit(Xtr, ytr)
        fit_s = time.perf_counter() - t0
        p1 = pipe.predict_proba(Xte)[:, 1]
        probas[name], preds[name], fitted[name] = p1, (p1 >= 0.5).astype(int), pipe
        hold.append({"window": label, "model": name, **metrics_from_proba(yte, p1),
                     "fit_seconds": round(fit_s, 2)})
    hold = pd.DataFrame(hold)
    res = {"label": label, "holdout": hold, "cv": cv_sum, "folds": folds, "fitted": fitted,
           "probas": probas, "preds": preds, "y_test": yte, "X_test": Xte,
           "best_params": best_params, "features": features,
           "n_train": len(tr), "n_test": len(te)}
    if not extras:
        return res

    #довірчі інтервали (bootstrap) на тесті
    ci_rows = []
    for name in models:
        ci = bootstrap_ci(yte, preds[name])
        ci_rows.append({"window": label, "model": name,
                        **{k: f"[{v[0]:.4f}; {v[1]:.4f}]" for k, v in ci.items()}})
    res["ci"] = pd.DataFrame(ci_rows)

    #найкраща модель обирається за CV (не за тестом)
    best = cv_sum.sort_values("cv_macro_f1_mean", ascending=False).iloc[0]["model"]
    res["best_model"] = best

    #значущість: скоригований t-тест (CV) + Мак-Немар (тест)
    piv = folds.pivot_table(index=["rep", "fold"], columns="model", values="macro_f1")
    n_tr_cv = len(tr) * (CV_FOLDS - 1) / CV_FOLDS
    n_te_cv = len(tr) / CV_FOLDS
    sig = []
    for other in models:
        if other == best:
            continue
        t_val, p_val = corrected_ttest(piv[best] - piv[other], n_tr_cv, n_te_cv)
        b, c, p_mc = mcnemar_exact(yte, preds[best], preds[other])
        sig.append({"window": label, "best_model": best, "vs": other,
                    "cv_mean_diff_macro_f1": float((piv[best] - piv[other]).mean()),
                    "nb_t": t_val, "nb_p": p_val,
                    "mcnemar_best_only_correct": b, "mcnemar_other_only_correct": c,
                    "mcnemar_p": p_mc})
    res["sig"] = pd.DataFrame(sig)

    #крива порогу для групи ризику кращої моделі
    thr_tbl, curve = risk_threshold_table(yte, probas[best])
    res["thr_table"] = thr_tbl.assign(window=label, model=best)
    res["pr_curve"] = curve

    #важливість ознак кращої моделі
    perm = permutation_importance(fitted[best], Xte, yte, scoring="f1_macro",
                                  n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    res["perm_imp"] = (pd.Series(perm.importances_mean, index=features)
                       .sort_values(ascending=False).rename(label))
    clf = fitted[best].named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        names = fitted[best].named_steps["pre"].get_feature_names_out()
        res["imp_impurity"] = (pd.Series(clf.feature_importances_, index=names)
                               .sort_values(ascending=False).rename(label))
    return res

#6. Основний прогін: усі вікна

runs = {}
for w in WINDOWS:
    t0 = time.perf_counter()
    runs[w] = run_experiment(datasets[w], ALL_FEATURES, wname(w))
    r = runs[w]
    print(f"\n {wname(w)}: train={r['n_train']} test={r['n_test']} | "
          f"найкраща за CV: {r['best_model']} | {time.perf_counter() - t0:.0f} с")
    print(r["holdout"].set_index("model")[["accuracy", "roc_auc", "macro_f1", "precision_0", "recall_0",
                                           "f1_0", "TN", "FP", "FN", "TP", "fit_seconds"]].round(4).to_string())

hold_all = pd.concat([r["holdout"] for r in runs.values()], ignore_index=True)
cv_all = pd.concat([r["cv"] for r in runs.values()], ignore_index=True)
folds_all = pd.concat([r["folds"] for r in runs.values()], ignore_index=True)
ci_all = pd.concat([r["ci"] for r in runs.values()], ignore_index=True)
sig_all = pd.concat([r["sig"] for r in runs.values()], ignore_index=True)
thr_all = pd.concat([r["thr_table"] for r in runs.values()], ignore_index=True)
for name, d in [("results_holdout", hold_all), ("results_cv", cv_all), ("cv_folds_raw", folds_all),
                ("bootstrap_ci", ci_all), ("significance_tests", sig_all),
                ("threshold_table_risk", thr_all)]:
    d.to_csv(out(f"{name}.csv"), index=False, encoding="utf-8-sig")
json.dump({wname(w): r["best_params"] for w, r in runs.items()},
          open(out("best_params.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)

#6.1 Зведені таблиці для розділу 3

pd.set_option("display.width", 220)
print("\n--- Відкладена вибірка: macro-F1 / recall групи ризику за вікнами ---")
print(hold_all.pivot(index="model", columns="window", values="macro_f1")[[wname(w) for w in WINDOWS]].round(4))
print(hold_all.pivot(index="model", columns="window", values="recall_0")[[wname(w) for w in WINDOWS]].round(4))
print("\n--- CV (середнє ± σ), macro-F1 ---")
cv_fmt = cv_all.assign(v=lambda d: d["cv_macro_f1_mean"].round(4).astype(str) + " ± " + d["cv_macro_f1_std"].round(4).astype(str))
print(cv_fmt.pivot(index="model", columns="window", values="v")[[wname(w) for w in WINDOWS]])
print("\n--- Довірчі інтервали (тест) ---")
print(ci_all.to_string(index=False))
print("\n--- Значущість відмінностей кращої моделі ---")
print(sig_all.round(4).to_string(index=False))
print("\n--- Поріг для виявлення ризику (краща модель) ---")
print(thr_all.round(4).to_string(index=False))

#7. Графіки: матриці невідповідностей, ROC, крива «ризик»

for w, r in runs.items():
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.8))
    for ax, name in zip(axes, r["preds"]):
        cm = confusion_matrix(r["y_test"], r["preds"][name], labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Ризик (0)", "Успіх (1)"], yticklabels=["Ризик (0)", "Успіх (1)"])
        ax.set_title(f"{name} [{wname(w)}]")
        ax.set_xlabel("Прогноз"); ax.set_ylabel("Факт")
    plt.tight_layout(); plt.savefig(out(f"confusion_{wname(w)}.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p1 in r["probas"].items():
        fpr, tpr, _ = roc_curve(r["y_test"], p1)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(r['y_test'], p1):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC [{wname(w)}]"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out(f"roc_{wname(w)}.png"), dpi=200); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 5))
for w, r in runs.items():
    prec, rec = r["pr_curve"]
    ax.plot(rec, prec, label=f"{wname(w)} ({r['best_model']})")
ax.set_xlabel("Recall групи ризику"); ax.set_ylabel("Precision групи ризику")
ax.set_title("Виявлення групи ризику: precision–recall"); ax.legend()
plt.tight_layout(); plt.savefig(out("risk_precision_recall.png"), dpi=200); plt.close(fig)

#Динаміка якості за вікнами (кращі моделі за CV)
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in make_models(ALL_FEATURES):
    ys = [hold_all[(hold_all.window == wname(w)) & (hold_all.model == name)]["macro_f1"].iloc[0] for w in WINDOWS]
    ax.plot([wname(w) for w in WINDOWS], ys, marker="o", label=name)
ax.set_ylabel("macro-F1 (тест)"); ax.set_title("Якість прогнозу залежно від вікна спостереження"); ax.legend()
plt.tight_layout(); plt.savefig(out("macro_f1_by_window.png"), dpi=200); plt.close(fig)

#8. Важливість ознак

perm_all = pd.concat([r["perm_imp"] for r in runs.values()], axis=1)
perm_all.to_csv(out("importance_permutation.csv"), encoding="utf-8-sig")
imp_all = pd.concat([r["imp_impurity"] for r in runs.values() if "imp_impurity" in r], axis=1)
imp_all.to_csv(out("importance_impurity.csv"), encoding="utf-8-sig")
print("Permutation importance (падіння macro-F1) за вікнами, топ-10:")
print(perm_all.assign(_m=perm_all.max(axis=1)).sort_values("_m", ascending=False).drop(columns="_m").head(10).round(4))

#9. Абляція за групами ознак

scenarios = {
    "A. демографія + курс": FEATURE_GROUPS["demographic"] + FEATURE_GROUPS["course"],
    "B. A + активність VLE": FEATURE_GROUPS["demographic"] + FEATURE_GROUPS["course"] + FEATURE_GROUPS["vle_activity"],
    "C. A + оцінювання": FEATURE_GROUPS["demographic"] + FEATURE_GROUPS["course"] + FEATURE_GROUPS["assessments"],
    "D. усі ознаки": ALL_FEATURES,
}
abl_rows = []
for w in ABLATION_WINDOWS:
    for sname, feats in scenarios.items():
        r = run_experiment(datasets[w], feats, f"{wname(w)}|{sname}", extras=False, cv_repeats=1)
        for _, h in r["holdout"].iterrows():
            cvr = r["cv"].set_index("model").loc[h["model"]]
            abl_rows.append({"window": wname(w), "scenario": sname, "model": h["model"],
                             "n_features": len(feats), "holdout_macro_f1": h["macro_f1"],
                             "holdout_recall_risk": h["recall_0"], "holdout_roc_auc": h["roc_auc"],
                             "cv_macro_f1": f"{cvr['cv_macro_f1_mean']:.4f} ± {cvr['cv_macro_f1_std']:.4f}"})
abl = pd.DataFrame(abl_rows)
abl.to_csv(out("ablation.csv"), index=False, encoding="utf-8-sig")
for w in ABLATION_WINDOWS:
    print(f"\n--- Абляція, вікно {wname(w)}: macro-F1 (тест) ---")
    print(abl[abl.window == wname(w)].pivot(index="scenario", columns="model", values="holdout_macro_f1").round(4))

#10. Збереження моделей (пайплайн = препроцесинг + модель) і перевірка відтворюваності (2.3.2)

save_rows = []
for w, r in runs.items():
    mdir = out(os.path.join("models", wname(w)))
    os.makedirs(mdir, exist_ok=True)
    for name, pipe in r["fitted"].items():
        path = os.path.join(mdir, f"{name}.joblib")
        joblib.dump(pipe, path)
        same = np.array_equal(joblib.load(path).predict(r["X_test"]), pipe.predict(r["X_test"]))
        save_rows.append({"window": wname(w), "model": name, "file": os.path.relpath(path, OUT_DIR),
                          "size_kb": round(os.path.getsize(path) / 1024, 1), "reload_identical": same})
    json.dump({"window": wname(w), "cutoff_day": w, "features": r["features"],
               "target": "1=Pass/Distinction, 0=Fail/Withdrawn", "sklearn": sklearn.__version__,
               "best_model_by_cv": r["best_model"]},
              open(os.path.join(mdir, "metadata.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
save_df = pd.DataFrame(save_rows)
save_df.to_csv(out("saved_models.csv"), index=False, encoding="utf-8-sig")
print(save_df.to_string(index=False))
assert save_df["reload_identical"].all()
print(f"\nГотово. Усі результати у каталозі: {os.path.abspath(OUT_DIR)}")

import os, json, argparse
from glob import glob
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, precision_recall_fscore_support, \
    confusion_matrix, f1_score

# === Config ===
# columnas a excluir (en minúsculas)
EXCLUDE_COLS = {
    "epoch", "window", "start_w", "end_w", "n_samples", "trust", "binary_trust"
}

SEPPARATION_DIR = os.path.join("ML", "Sepparation", "Binary", "splits_trialwise_binary.json")


# === Split persistence ===
def split_npz_path(splits_dir, pid): return os.path.join(splits_dir, f"{pid}_split.npz")


def load_saved_split_npz(df, pid, group_col, splits_dir):
    p = split_npz_path(splits_dir, pid)
    if not os.path.exists(p): return None, None
    try:
        data = np.load(p, allow_pickle=True)
        groups_df = set(df[group_col].unique())
        tr_g, te_g = data["train_groups"], data["test_groups"]
        if set(tr_g).issubset(groups_df) and set(te_g).issubset(groups_df):
            return tr_g, te_g
    except Exception:
        pass
    return None, None


def save_split_npz(pid, tr_g, te_g, group_col, splits_dir, seed, target):
    os.makedirs(splits_dir, exist_ok=True)
    np.savez(split_npz_path(splits_dir, pid),
             train_groups=tr_g, test_groups=te_g,
             group_col=group_col, seed=np.array([seed]), target=np.array([target]))


# === IO & features ===
def load_and_normalize_columns(path, target, group_col):
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    need = {target.lower(), group_col}
    if not need.issubset(df.columns):
        raise ValueError(f"Faltan columnas {need - set(df.columns)} en {os.path.basename(path)}")
    return df


def feature_columns(df):
    return [c for c in df.columns if (c not in EXCLUDE_COLS) and (df[c].dtype.kind in "if")]


# === Trial-wise split estratificado ===
def stratified_group_split(df, test_epochs, seed, target, group_col):
    rng = np.random.RandomState(seed)
    gmode = df.groupby(group_col)[target].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).astype(
        str)
    gids = gmode.index.to_numpy();
    n = len(gids)
    if n < 2: return gids, np.array([], dtype=gids.dtype)
    te = min(max(2, test_epochs), n - 1)
    classes = gmode.unique()
    if len(classes) < 2:
        rng.shuffle(gids);
        return np.sort(gids[te:]), np.sort(gids[:te])
    tr, te_ = [], []
    for cls in classes:
        idx = gids[gmode == cls];
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * te / n)))
        te_.extend(idx[:cut]);
        tr.extend(idx[cut:])
    return np.array(tr), np.array(te_)


# === Modelo (SVM + scaler) ===
def build_pipeline():
    # SVC soporta multiclase “one-vs-one”, usamos pesos balanceados.
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc", SVC(class_weight="balanced", probability=False))
    ])


def fit_gridsearch_cv(X, y, groups, param_grid, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    scorer = make_scorer(f1_score)
    pipe = build_pipeline()
    gs = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring=scorer,
                      cv=sgkf.split(X, y, groups=groups),
                      n_jobs=-1, refit=True, verbose=0, return_train_score=False)
    gs.fit(X, y)
    return gs


# === Métricas ===
def evaluate_model(model, Xte, yte, n_classes: int):
    yhat = model.predict(Xte)
    labels = np.arange(int(n_classes))
    acc = accuracy_score(yte, yhat)
    bacc = balanced_accuracy_score(yte, yhat)
    _, _, f1_per_class, _ = precision_recall_fscore_support(yte, yhat, labels=labels, average=None, zero_division=0)
    f1 = f1_score(yte, yhat)
    cm = confusion_matrix(yte, yhat, labels=labels).tolist()
    return {"acc": float(acc), "bacc": float(bacc), "f1": f1, "cm": cm}


def debug_participant(df, pid, feats, test_groups, seed, target, group_col):
    print(f"\n[DEBUG {pid}] shape={df.shape}  n_features={len(feats)}")
    print("[DEBUG] class distribution (global):")
    print(df[target].value_counts(dropna=False).to_string())
    gmode = df.groupby(group_col)[target].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
    print("[DEBUG] groups per class (by mode):")
    print(gmode.value_counts().to_string())
    try:
        g_train, g_test = train_test_split(gmode.index.values, test_size=test_groups,
                                           random_state=seed, stratify=gmode.astype(str))
        print(f"[DEBUG] stratified OK: train_groups={len(g_train)} test_groups={len(g_test)}")
    except Exception as e:
        print("[DEBUG] stratified FAILED:", repr(e))


# === Main ===
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/Users/luciarebolledo/PycharmProjects/TFG_practicas_final/semana_7_shap/PRUEBA2/ML/Binary_clean/sesion02_top30")
    ap.add_argument("--test_groups", type=int, default=60, help="Nº de grupos (epochs) en test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="Outputs/Sesion02/svm_binary_sesion02_top30")
    ap.add_argument("--pattern", type=str, default="eeg_features_*.csv")
    ap.add_argument("--target", type=str, default="binary_trust")
    ap.add_argument("--splits_dir", type=str, default="ML/Sepparation/Binary/top30/sesion02")
    ap.add_argument("--group_col", type=str, default="epoch")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Grid SVM
    param_grid = {
        "svc__kernel": ["rbf", "linear"],
        "svc__C": [0.001, 0.01, 0.1, 1],
        "svc__gamma": ["scale", "auto"]
    }

    # ============================================================
    # Cambiado para leer mis ficheros
    # ============================================================
    # Encontrar ficheros y numerar participantes como int
    files = {}
    for p in sorted(glob(os.path.join(args.data_dir, args.pattern))):
        base = os.path.basename(p)

        # Extraer cualquier número dentro del nombre del archivo
        digits = "".join(filter(str.isdigit, base))

        try:
            pid_int = int(digits)
        except ValueError:
            pid_int = None

        if pid_int is not None:
            files[pid_int] = p

    if not files:
        print("No CSVs found under", args.data_dir)
        return
    # ===============================================================

    per_participant, param_scores_across, splits_info = {}, defaultdict(list), {}

    for pid, path in sorted(files.items()):
        print("----------- Participant", pid, "-----------")
        df = load_and_normalize_columns(path, args.target, args.group_col)

        n_groups = df[args.group_col].nunique()
        if n_groups < 9:
            print(f"[SKIP {pid}] solo {n_groups} grupos. Se requieren ≥9.")
            continue

        # robust 0/1
        ymap = {"1": 1, "true": 1, "t": 1, "yes": 1, "y": 1, "0": 0, "false": 0, "f": 0, "no": 0, "n": 0}
        y_raw_series = df[args.target].astype(str).str.strip().str.lower().map(ymap)
        if y_raw_series.isna().mean() < 0.5:
            df[args.target] = y_raw_series.fillna(method="ffill").fillna(method="bfill").fillna(0).astype(int)

        tr_g, te_g = load_saved_split_npz(df, pid, args.group_col, args.splits_dir)
        if tr_g is None or te_g is None:
            tr_g, te_g = stratified_group_split(df, test_epochs=args.test_groups,
                                                seed=args.seed, target=args.target, group_col=args.group_col)
            save_split_npz(pid, tr_g, te_g, args.group_col, args.splits_dir, args.seed, args.target)
            print(f"Creando nuevo split para participante {pid}")
        else:
            print(f"Usando split existente para participante {pid}")
        print(f"[DEBUG] train_{args.group_col}s={tr_g.tolist()} test_{args.group_col}s={te_g.tolist()}")

        feats = feature_columns(df)
        if not feats: raise ValueError(f"No numeric features in {pid}")
        debug_participant(df, pid, feats, args.test_groups, args.seed, args.target, args.group_col)

        df_tr = df[df[args.group_col].isin(tr_g)].copy()
        df_te = df[df[args.group_col].isin(te_g)].copy()

        Xtr = df_tr[feats].to_numpy(dtype=np.float32)
        Xte = df_te[feats].to_numpy(dtype=np.float32)
        le = LabelEncoder()
        ytr = le.fit_transform(df_tr[args.target])
        yte = le.transform(df_te[args.target])
        gtr = df_tr[args.group_col].to_numpy()

        if len(np.unique(ytr)) < 2:
            print(f"[SKIP {pid}] split degenerado (una sola clase en train).");
            continue

        gs = fit_gridsearch_cv(Xtr, ytr, groups=gtr, param_grid=param_grid, seed=args.seed)
        ev_tr = evaluate_model(gs.best_estimator_, Xtr, ytr, n_classes=len(le.classes_))
        ev_te = evaluate_model(gs.best_estimator_, Xte, yte, n_classes=len(le.classes_))
        print(f"[LOCAL {pid}] TRAIN f1={ev_tr['f1']:.3f} | TEST f1={ev_te['f1']:.3f} | CV best={gs.best_score_:.3f}")
        print(f"[BEST LCAL HYPERPARAMETERS]: {gs.best_estimator_}")

        # ================================================================
        # === ENTRENAR SVM LIMPIO PARA SHAP (KernelSHAP) =================
        # ================================================================
        import joblib
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        # Extraer SOLO hiperparámetros del SVM dentro del pipeline
        clean_params = {
            k.replace("svc__", ""): v
            for k, v in gs.best_params_.items()
            if k.startswith("svc__")
        }

        # Crear scaler y escalar Xtr
        scaler_clean = StandardScaler(with_mean=True, with_std=True)
        Xtr_scaled = scaler_clean.fit_transform(Xtr)

        # Crear SVM limpio (probability=True para SHAP)
        svm_clean = SVC(probability=True, **clean_params)

        # Entrenar modelo limpio
        svm_clean.fit(Xtr_scaled, ytr)

        # Guardar scaler + modelo limpio para SHAP
        joblib.dump(
            svm_clean,
            os.path.join(args.out_dir, f"svm_clean_for_shap_participant_{pid}.pkl")
        )
        joblib.dump(
            scaler_clean,
            os.path.join(args.out_dir, f"scaler_svm_for_shap_participant_{pid}.pkl")
        )

        print(f"[SAVE] SVM limpio SHAP para participante {pid}")
        # ================================================================

        ev_local = evaluate_model(gs.best_estimator_, Xte, yte, n_classes=len(le.classes_))

        for params, score in zip(gs.cv_results_["params"], gs.cv_results_["mean_test_score"]):
            key = json.dumps(params, sort_keys=True);
            param_scores_across[key].append(float(score))

        splits_info[pid] = {"train_groups": np.array(tr_g).tolist(),
                            "test_groups": np.array(te_g).tolist(),
                            "group_col": args.group_col}

        per_participant[pid] = {
            "path": path,
            "n_features": len(feats),
            "train_rows": int(len(df_tr)),
            "test_rows": int(len(df_te)),
            "best_params_local": gs.best_params_,
            "best_local_cv_bacc": float(gs.best_score_),
            "test_acc_local": float(ev_local["acc"]),
            "test_bacc_local": float(ev_local["bacc"]),
            "test_f1_local": float(ev_local["f1"]),
            "cm_local": ev_local["cm"]
        }

    if not per_participant:
        print("No hay participantes válidos procesados.");
        return

    # Selección global por media CV
    global_choice, global_mean = None, -1.0
    for key, scores in param_scores_across.items():
        if not scores: continue
        m = float(np.mean(scores))
        if m > global_mean: global_mean, global_choice = m, json.loads(key)

    # ============================================================
    # === LISTA PARA GUARDAR LOS RESULTADOS GLOBALES ===
    rows_global = []
    # ============================================================

    # Re-evaluación con hiperparámetros globales
    rows = []
    for pid, info in per_participant.items():
        df = load_and_normalize_columns(info["path"], target="binary_trust", group_col=args.group_col)
        feats = feature_columns(df)
        tr_g = np.array(splits_info[pid]["train_groups"]);
        te_g = np.array(splits_info[pid]["test_groups"])
        df_tr = df[df[args.group_col].isin(tr_g)];
        df_te = df[df[args.group_col].isin(te_g)]
        Xtr = df_tr[feats].to_numpy(dtype=np.float32);
        Xte = df_te[feats].to_numpy(dtype=np.float32)
        le = LabelEncoder();
        ytr = le.fit_transform(df_tr["binary_trust"]);
        yte = le.transform(df_te["binary_trust"])

        pipe = build_pipeline()
        if global_choice is not None: pipe.set_params(**global_choice)
        pipe.fit(Xtr, ytr)
        te_ev = evaluate_model(pipe, Xte, yte, n_classes=len(le.classes_))

        # métrica TRAIN out-of-fold (post-selección) para el modelo global
        sgkf_post = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed + 123)
        cv_post = cross_val_score(pipe, Xtr, ytr, groups=df_tr[args.group_col].to_numpy(),
                                  cv=sgkf_post, scoring=make_scorer(f1_score), n_jobs=-1)
        print(f"[RESULT {pid}] CV(TRAIN, global_params) F1={cv_post.mean():.3f} "
              f"| TEST F1={te_ev['f1']:.3f} | Acc={te_ev['acc']:.3f} | bAcc={te_ev['bacc']:.3f} "
              f"(train_rows={len(df_tr)}, test_rows={len(df_te)}, train_groups={len(tr_g)}, test_groups={len(te_g)})")

        # ================================================================
        # === ENTRENAR SVM GLOBAL LIMPIO PARA SHAP =======================
        # ================================================================
        import joblib
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler

        # Extraer SOLO hiperparámetros del SVM global
        clean_global_params = {
            k.replace("svc__", ""): v
            for k, v in (global_choice or {}).items()
            if k.startswith("svc__")
        }

        # Crear scaler y ajustar SOLO con datos Xtr
        scaler_global = StandardScaler(with_mean=True, with_std=True)
        Xtr_scaled_global = scaler_global.fit_transform(Xtr)

        # Crear SVM global limpio
        svm_clean_global = SVC(probability=True, **clean_global_params)

        # Entrenar SVM global
        svm_clean_global.fit(Xtr_scaled_global, ytr)

        # Guardar modelo y scaler del GLOBAL para SHAP
        joblib.dump(
            svm_clean_global,
            os.path.join(args.out_dir, f"svm_global_clean_for_shap_participant_{pid}.pkl")
        )
        joblib.dump(
            scaler_global,
            os.path.join(args.out_dir, f"scaler_svm_global_for_shap_participant_{pid}.pkl")
        )

        print(f"[SAVE GLOBAL SHAP] SVM GLOBAL guardado → svm_global_clean_for_shap_participant_{pid}.pkl")

        rows_global.append({
            "participant": pid,
            "rows_train": len(df_tr),
            "rows_test": len(df_te),
            "groups_train": len(tr_g),
            "groups_test": len(te_g),
            "n_features": len(feats),
            "global_te_acc": te_ev["acc"],
            "global_te_bacc": te_ev["bacc"],
            "global_te_f1": te_ev["f1"],
            "oob_global": None,
            "global_params": json.dumps(clean_global_params)
        })
        # ================================================================

        rows.append({
            "participant": pid,
            "rows_train": len(df_tr), "rows_test": len(df_te),
            "groups_train": len(tr_g), "groups_test": len(te_g),
            "n_features": len(feats),
            "te_acc": te_ev["acc"], "bacc": te_ev["bacc"], "f1": te_ev["f1"],
            "best_local_params": json.dumps(info["best_params_local"], sort_keys=True),
            "best_local_cv_bacc": info["best_local_cv_bacc"]
        })

    summary = pd.DataFrame(rows).sort_values("participant")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(SEPPARATION_DIR), exist_ok=True)

    # === Medias del modelo final (params globales) en TEST ===
    mean_te_acc = float(summary["te_acc"].mean()) if not summary.empty else float("nan")
    mean_te_f1w = float(summary["f1"].mean()) if not summary.empty else float("nan")

    # CSV resumen
    summary_path = os.path.join(args.out_dir, "svm_trialwise_binary_trust_summary.csv")
    summary.to_csv(summary_path, index=False)

    # -----------------------------------
    # CSV GLOBAL
    # -----------------------------------
    summary_global = pd.DataFrame(rows_global).sort_values("participant")
    summary_global_path = os.path.join(args.out_dir, "svm_global_summary_binary_trust.csv")
    summary_global.to_csv(summary_global_path, index=False)
    # -----------------------------------

    # Guardar splits
    with open(SEPPARATION_DIR, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2)

    # JSON con hiperparámetros y medias finales
    best_json_path = os.path.join(args.out_dir, "best_global_params_binary_trust_svm.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_global_params": global_choice or {},
            "mean_cv_f1_across": float(global_mean),
            "final_model_mean_test_acc": mean_te_acc,
            "final_model_mean_test_f1": mean_te_f1w,
            "n_participants": int(len(summary))
        }, f, indent=2)

    print("Best global params:", global_choice, "| mean CV F1:", global_mean)
    print("Final model (global params) → mean TEST Acc:", mean_te_acc, "| mean TEST F1:", mean_te_f1w)
    print("Saved summary to:", summary_path)
    print("Saved best params JSON to:", best_json_path)


if __name__ == "__main__":
    main()

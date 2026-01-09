import os, json, argparse
from glob import glob
from collections import defaultdict
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, make_scorer, precision_recall_fscore_support, \
    confusion_matrix, f1_score

# === Config ===
# columnas a excluir (en minúsculas)
EXCLUDE_COLS = {
    "epoch", "window", "start_w", "end_w", "n_samples", "trust", "binary_trust",
    "tertiles_trust", "eeg_trust", "ecg_trust", "eyetracking_trust", "ring_trust"
} #Tendrias que añadir: eeg_trust

SEPPARATION_DIR = os.path.join("ML", "Sepparation", "Tertiles", "splits_trialwise_tertiles_trust.json")


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


# === Trial-wise split estratificado (por modo mayoritario del grupo) ===
def stratified_group_split(df, test_epochs, seed, target, group_col):
    rng = np.random.RandomState(seed)
    gmode = df.groupby(group_col)[target].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]).astype(
        str)
    gids = gmode.index.to_numpy();
    n = len(gids)
    if n < 2: return gids, np.array([], dtype=gids.dtype)
    te = min(max(2, test_epochs), n - 1)
    classes = gmode.unique()
    tr, te_ = [], []
    for cls in classes:
        idx = gids[gmode == cls];
        rng.shuffle(idx)
        cut = max(1, int(round(len(idx) * te / n)))
        te_.extend(idx[:cut]);
        tr.extend(idx[cut:])
    return np.array(tr), np.array(te_)


# === Modelo (RF) ===
def build_pipeline():
    # Solo el estimador: los árboles no necesitan escalado
    return Pipeline([
        ("rf", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
            oob_score=True
        ))
    ])


def fit_gridsearch_cv(X, y, groups, param_grid, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    scorer = make_scorer(f1_score, average="weighted")
    gs = GridSearchCV(
        estimator=build_pipeline(),
        param_grid=param_grid,
        scoring=scorer,
        cv=sgkf,
        n_jobs=-1,
        refit=True,
        verbose=0,
        return_train_score=False
    )
    gs.fit(X, y, groups=groups)
    return gs


# === Métricas ===
def evaluate_model(model, Xte, yte, n_classes: int):
    yhat = model.predict(Xte)
    labels = np.arange(int(n_classes))
    acc = accuracy_score(yte, yhat)
    bacc = balanced_accuracy_score(yte, yhat)
    f1_weighted = f1_score(yte, yhat, average="weighted")
    f1_per_class = precision_recall_fscore_support(yte, yhat, labels=labels, average=None, zero_division=0)[2].tolist()
    cm = confusion_matrix(yte, yhat, labels=labels).tolist()
    return {"acc": float(acc), "bacc": float(bacc), "f1_weighted": float(f1_weighted), "f1_per_class": f1_per_class,
            "cm": cm}


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
    ap.add_argument("--data_dir", type=str, default="/Users/luciarebolledo/PycharmProjects/TFG_practicas_final/semana_7_shap/PRUEBA2/ML/Binary_clean/sesion01_30")
    ap.add_argument("--test_groups", type=int, default=60, help="Nº de grupos (epochs) en test")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="Outputs/Sesion01/rf_binary_sesion01_top30")
    ap.add_argument("--pattern", type=str, default="eeg_features_*.csv")
    ap.add_argument("--target", type=str, default="binary_trust")
    ap.add_argument("--splits_dir", type=str, default="ML/Sepparation/Binary/top30")
    ap.add_argument("--group_col", type=str, default="epoch")
    args = ap.parse_args()



    os.makedirs(args.out_dir, exist_ok=True)

    # === Grid RF ===
    param_grid = {
        "rf__n_estimators": [500, 800, 1000],
        "rf__max_depth": [12,14,16],
        "rf__min_samples_leaf": [2, 4],
        "rf__max_features": ["sqrt"]
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
    # ============================================================

    per_participant, param_scores_across, splits_info = {}, defaultdict(list), {}

    for pid, path in sorted(files.items()):
        print("----------- Participant", pid, "-----------")
        df = load_and_normalize_columns(path, args.target, args.group_col)

        n_groups = df[args.group_col].nunique()
        if n_groups < 9:
            print(f"[SKIP {pid}] solo {n_groups} grupos. Se requieren ≥9.")
            continue

        # Cargar/crear split trial-wise estratificado por clase mayoritaria de grupo
        tr_g, te_g = load_saved_split_npz(df, pid, args.group_col, args.splits_dir)
        if tr_g is None or te_g is None:
            tr_g, te_g = stratified_group_split(df, test_epochs=args.test_groups,
                                                seed=args.seed, target=args.target, group_col=args.group_col)
            save_split_npz(pid, tr_g, te_g, args.group_col, args.splits_dir, args.seed, args.target)
            print(f"Creando nuevo split para participante {pid}")
        else:
            print(f"Usando split existente para participante {pid}")

        feats = feature_columns(df)
        cols_con_trust = [c for c in df.columns if "trust" in c and c != args.target]
        if cols_con_trust:
            assert all(c not in feats for c in cols_con_trust), \
                f"[LEAK GUARD] {pid}: {cols_con_trust} no deberían estar en feats"

        if not feats: raise ValueError(f"No numeric features in {pid}")
        debug_participant(df, pid, feats, args.test_groups, args.seed, args.target, args.group_col)

        # train/test por grupos
        df_tr = df[df[args.group_col].isin(tr_g)].copy()
        df_te = df[df[args.group_col].isin(te_g)].copy()

        Xtr = df_tr[feats].to_numpy(dtype=np.float32)
        Xte = df_te[feats].to_numpy(dtype=np.float32)

        # LabelEncoder para labels string ('Low_Trust','Medium_Trust','High_Trust')
        le = LabelEncoder()
        ytr = le.fit_transform(df_tr[args.target])
        yte = le.transform(df_te[args.target])
        gtr = df_tr[args.group_col].to_numpy()

        # Chequeo: al menos 2 clases en train
        if len(np.unique(ytr)) < 2:
            print(f"[SKIP {pid}] split degenerado (una sola clase en train).");
            continue

        # GridSearch CV con estratificación por grupos (selección de hiperparámetros en TRAIN)
        gs = fit_gridsearch_cv(Xtr, ytr, groups=gtr, param_grid=param_grid, seed=args.seed)

        # === Cálculo del OOB con los mejores hiperparámetros locales ===
        best_rf = gs.best_estimator_.named_steps["rf"]
        oob_local = getattr(best_rf, "oob_score_", np.nan)

        # === CV post-selección (folds distintos a los del GridSearch) ===
        scorer_weighted = make_scorer(f1_score, average="weighted")
        sgkf_post = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed + 1)

        cv_post_scores = cross_val_score(
            gs.best_estimator_, Xtr, ytr,
            groups=gtr, cv=sgkf_post, scoring=scorer_weighted, n_jobs=-1
        )

        # Evaluación en TEST (hold-out de grupos)
        ev_te = evaluate_model(gs.best_estimator_, Xte, yte, n_classes=len(le.classes_))

        cv_select = gs.best_score_  # CV usado en la selección (optimista)
        cv_post = float(cv_post_scores.mean())  # CV post-selección (menos optimista)
        test_f1 = ev_te["f1_weighted"]

        optimism = cv_select - cv_post  # sobreajuste por selección
        gen_gap = cv_post - test_f1  # brecha train (CV post) -> test

        print(f"[LOCAL {pid}] CV_select={cv_select:.3f} | CV_post={cv_post:.3f} | "
              f"TEST F1_weighted={test_f1:.3f} | optimism={optimism:.3f} | gen_gap={gen_gap:.3f}")
        print(f"[BEST LOCAL HYPERPARAMETERS]: {gs.best_estimator_}")

        # ================================================================
        # === ENTRENAR RANDOM FOREST LIMPIO PARA SHAP ====================
        # ================================================================

        from sklearn.ensemble import RandomForestClassifier
        import joblib

        # Extraer mejores hiperparámetros encontrados por GridSearchCV
        best_params = gs.best_params_

        # Crear RF limpio (sin pipeline)
        rf_clean = RandomForestClassifier(
            n_estimators=best_params["rf__n_estimators"],
            max_depth=best_params["rf__max_depth"],
            min_samples_leaf=best_params["rf__min_samples_leaf"],
            max_features=best_params["rf__max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )

        # ENTRENAR EL RF LIMPIO DIRECTAMENTE SOBRE Xtr, ytr
        rf_clean.fit(Xtr, ytr)

        # === GUARDAR MODELO LIMPIO PARA SHAP ===
        model_shap_path = os.path.join(args.out_dir, f"rf_clean_for_shap_participant_{pid}.pkl")
        joblib.dump(rf_clean, model_shap_path)

        print(f"[SAVE] Modelo RF LIMPIO para SHAP guardado en: {model_shap_path}")
        # ================================================================

        # guardar puntuaciones de cada set de hiperparámetros
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
            "best_local_cv_f1_weighted": float(gs.best_score_),
            "test_acc_local": float(ev_te["acc"]),
            "test_bacc_local": float(ev_te["bacc"]),
            "test_f1_weighted_local": float(ev_te["f1_weighted"]),
            "test_f1_per_class_local": ev_te["f1_per_class"],
            "oob_local": float(oob_local),
            "cm_local": ev_te["cm"],
            "label_mapping": {int(i): cls for i, cls in enumerate(le.classes_)}
        }

    if not per_participant:
        print("No hay participantes válidos procesados.");
        return

    # Selección global por media CV (F1 weighted)
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

        oob_global = getattr(pipe.named_steps["rf"], "oob_score_", np.nan)

        # métrica TRAIN out-of-fold (post-selección) para el modelo global
        sgkf_post = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=args.seed + 123)
        cv_post = cross_val_score(pipe, Xtr, ytr, groups=df_tr[args.group_col].to_numpy(),
                                  cv=sgkf_post, scoring=make_scorer(f1_score, average="weighted"), n_jobs=-1)
        print(f"[RESULT {pid}] CV(TRAIN, global_params) F1_weighted={cv_post.mean():.3f} "
              f"| TEST F1_weighted={te_ev['f1_weighted']:.3f} | Acc={te_ev['acc']:.3f} | bAcc={te_ev['bacc']:.3f} "
              f"(train_rows={len(df_tr)}, test_rows={len(df_te)}, train_groups={len(tr_g)}, test_groups={len(te_g)})")

        # ============================================================
        # === ENTRENAR RANDOM FOREST GLOBAL LIMPIO PARA SHAP =========
        # ============================================================

        # Extraer hiperparámetros globales sin prefijo "rf__"
        clean_global_params = {
            k.replace("rf__", ""): v
            for k, v in (global_choice or {}).items()
        }

        # Crear el modelo limpio (sin pipeline)
        rf_clean_global = RandomForestClassifier(
            n_estimators=clean_global_params["n_estimators"],
            max_depth=clean_global_params["max_depth"],
            min_samples_leaf=clean_global_params["min_samples_leaf"],
            max_features=clean_global_params["max_features"],
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )

        # Entrenar con los datos del participante (Xtr, ytr)
        rf_clean_global.fit(Xtr, ytr)

        # Guardar modelo GLOBAL limpio para SHAP
        global_model_path = os.path.join(
            args.out_dir,
            f"rf_global_clean_for_shap_participant_{pid}.pkl"
        )
        joblib.dump(rf_clean_global, global_model_path)

        print(f"[SAVE GLOBAL SHAP] Guardado RF GLOBAL para participante {pid} --> {global_model_path}")

        rows_global.append({
            "participant": pid,
            "rows_train": len(df_tr),
            "rows_test": len(df_te),
            "groups_train": len(tr_g),
            "groups_test": len(te_g),
            "n_features": len(feats),
            "global_te_acc": te_ev["acc"],
            "global_te_bacc": te_ev["bacc"],
            "global_te_f1_weighted": te_ev["f1_weighted"],
            "oob_global": float(oob_global),
            "global_params": json.dumps(clean_global_params)
        })

        # ============================================================

        rows.append({
            "participant": pid,
            "rows_train": len(df_tr), "rows_test": len(df_te),
            "groups_train": len(tr_g), "groups_test": len(te_g),
            "n_features": len(feats),
            "te_acc": te_ev["acc"], "te_bacc": te_ev["bacc"], "te_f1_weighted": te_ev["f1_weighted"],
            "best_local_params": json.dumps(info["best_params_local"], sort_keys=True),
            "best_local_cv_f1_weighted": info["best_local_cv_f1_weighted"],
            "oob_global": float(oob_global),
        })

    summary = pd.DataFrame(rows).sort_values("participant")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(SEPPARATION_DIR), exist_ok=True)

    # === Medias del modelo final (params globales) en TEST ===
    mean_te_acc = float(summary["te_acc"].mean()) if not summary.empty else float("nan")
    mean_te_f1w = float(summary["te_f1_weighted"].mean()) if not summary.empty else float("nan")
    mean_oob_global = float(summary["oob_global"].mean()) if "oob_global" in summary.columns else float("nan")

    # CSV resumen
    summary_path = os.path.join(args.out_dir, "rf_trialwise_binary_trust_summary.csv")
    summary.to_csv(summary_path, index=False)

    # ============================================================
    # CSV GLOBAL
    # ===========================================================
    summary_global = pd.DataFrame(rows_global).sort_values("participant")
    summary_global_path = os.path.join(args.out_dir, "rf_global_summary_binary_trust.csv")
    summary_global.to_csv(summary_global_path, index=False)

    # ===========================================================

    # Guardar splits
    with open(SEPPARATION_DIR, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2)

    # JSON con hiperparámetros y medias finales
    best_json_path = os.path.join(args.out_dir, "best_global_params_binary_trust_rf.json")
    with open(best_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "best_global_params": global_choice or {},
            "mean_cv_f1_weighted_across": float(global_mean),
            "final_model_mean_test_acc": mean_te_acc,
            "final_model_mean_test_f1_weighted": mean_te_f1w,
            "final_model_mean_oob_global": mean_oob_global,
            "n_participants": int(len(summary))
        }, f, indent=2)

    print("Best global params:", global_choice, "| mean CV F1 weighted:", global_mean)
    print("Final model (global params) → mean TEST Acc:", mean_te_acc, "| mean TEST F1_weighted:", mean_te_f1w)
    print("Saved summary to:", summary_path)
    print("Saved best params JSON to:", best_json_path)


if __name__ == "__main__":
    main()

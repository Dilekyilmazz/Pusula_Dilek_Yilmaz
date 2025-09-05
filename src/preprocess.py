# src/preprocess.py
import os, re, argparse, unicodedata
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

MULTI = ["KronikHastalik", "Alerji", "Tanilar", "UygulamaYerleri"]


def extract_numeric_sessions(s):
    """'15 Seans' -> 15.0 gibi sayıyı çıkar."""
    if pd.isna(s):
        return np.nan
    m = re.search(r"(\d+)", str(s))
    return float(m.group(1)) if m else np.nan


def normalize_token(tok: str) -> str:
    """Unicode normalizasyonu + fazla boşlukları temizle.
    Türkçe 'İ/i' gibi karakterlerde sürprizleri azaltır."""
    if tok is None:
        return ""
    t = str(tok).strip()
    t = unicodedata.normalize("NFKC", t)  # combining işaretlerini düzelt
    t = re.sub(r"\s+", " ", t)
    return t  # İstersen .lower() tercih edebilirsin


def split_multi(x):
    if pd.isna(x):
        return []
    parts = str(x).replace(";", ",").split(",")
    return [p.strip() for p in parts if p.strip()]


def expand_multilabel(df: pd.DataFrame, cols):
    """KronikHastalik, Alerji, Tanilar, UygulamaYerleri gibi çoklu değerli kolonları
    one-vs-rest 0/1 dummy sütunlara açar."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue

        lists = out[c].apply(split_multi)
        norm_lists = lists.apply(lambda xs: [normalize_token(z) for z in xs if normalize_token(z)])

        # Alerji'de boşları "Yok" kabul et (EDA kararına uyum)
        if c == "Alerji":
            norm_lists = norm_lists.apply(lambda xs: xs if len(xs) > 0 else ["Yok"])

        uniq = sorted({z for row in norm_lists for z in row})
        for lbl in uniq:
            out[f"{c}__{lbl}"] = norm_lists.apply(lambda xs: int(lbl in xs))
        out = out.drop(columns=[c])
    return out


def basic_string_normalize(df: pd.DataFrame, cols):
    """Tek-değerli kategorik kolonlarda boşluk/boş string temizliği + NaN işaretleme."""
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(str).str.strip()
            out.loc[out[c].str.len() == 0, c] = np.nan
            # Unicode normalize
            out[c] = out[c].apply(lambda v: unicodedata.normalize("NFKC", v) if pd.notna(v) else v)
            out[c] = out[c].str.replace(r"\s+", " ", regex=True)
    return out


def detect_binary_cols(X: pd.DataFrame) -> list:
    """0/1 içeren ve MULTI kaynaklı dummy kolonları tespit et (ölçeklemeye sokma)."""
    bin_cols = []
    for c in X.columns:
        if any(c.startswith(m + "__") for m in MULTI):
            vals = pd.unique(X[c].dropna())
            if len(set(vals) - {0, 1}) == 0:
                bin_cols.append(c)
    return bin_cols


def build_preprocessor(num_cols, cat_cols, bin_cols):
    """Sayısal ve kategorik dönüşümleri; binary'leri passthrough yapan preprocessor."""
    num_tf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_tf = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer([
        ("num", num_tf, num_cols),
        ("cat", cat_tf, cat_cols),
        ("bin", "passthrough", bin_cols),
    ])
    return pre


def prepare_xy(data_path: str):
    """Ham Excel'i oku, hedefi sayısallaştır, string kolonları normalize et, multilabel'i genişlet.
    X (ham özellik matrisi) ve y (hedef) döndürür."""
    df = pd.read_excel(data_path)

    # hedefi sayısal yap
    df["TedaviSuresi_clean"] = df["TedaviSuresi"].apply(extract_numeric_sessions)

    # tek-değerli kategorikleri normalize et
    cat_single = ["Cinsiyet", "KanGrubu", "Uyruk", "Bolum", "TedaviAdi"]
    df = basic_string_normalize(df, cat_single)

    y = df["TedaviSuresi_clean"]
    X = df.drop(columns=["TedaviSuresi_clean", "TedaviSuresi", "HastaNo"], errors="ignore")

    # multi-label kolonları genişlet
    X = expand_multilabel(X, MULTI)

    return X, y


def feature_names_after_fit(pre: ColumnTransformer, num_cols, cat_cols, bin_cols):
    """ColumnTransformer fit edildikten sonra anlamlı kolon isimleri üret."""
    names = []
    # 1) sayısal
    names.extend(num_cols)
    # 2) kategorik (one-hot sonrası)
    try:
        oh_names = pre.named_transformers_["cat"].named_steps["oh"].get_feature_names_out(cat_cols).tolist()
    except Exception:
        oh_names = [f"cat_{i}" for i in range(len(cat_cols))]
    names.extend(oh_names)
    # 3) binary passthrough (orijinal isimler)
    names.extend(bin_cols)
    return names


def save_parquet(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def main(data_path: str, out_path: str, do_split: bool = False, test_size: float = 0.2, random_state: int = 42):
    # X, y hazırla
    X, y = prepare_xy(data_path)

    # kolon grupları: binary / numeric / categorical
    bin_cols = detect_binary_cols(X)
    num_cols = [c for c in X.select_dtypes(include=["number"]).columns if c not in bin_cols]
    cat_cols = [c for c in X.columns if c not in num_cols + bin_cols]

    if do_split:
        # ----- TRAIN/VALID AYIR, FIT SADECE TRAIN'DE -----
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

        pre = build_preprocessor(num_cols, cat_cols, bin_cols)

        Xtr_ready = pre.fit_transform(Xtr)   # fit sadece train
        Xte_ready = pre.transform(Xte)

        # Anlamlı isimler
        feat_names = feature_names_after_fit(pre, num_cols, cat_cols, bin_cols)

        Xtr_df = pd.DataFrame(Xtr_ready, columns=feat_names)
        Xte_df = pd.DataFrame(Xte_ready, columns=feat_names)
        Xtr_df["__target__TedaviSuresi"] = np.array(ytr)
        Xte_df["__target__TedaviSuresi"] = np.array(yte)

        # Kaydet
        os.makedirs("artifacts", exist_ok=True)
        save_parquet(Xtr_df, "artifacts/train.parquet")
        save_parquet(Xte_df, "artifacts/valid.parquet")
        joblib.dump(pre, "artifacts/preprocessor.joblib")

        print("✅ Preprocess (train/test) bitti")
        print("Train:", Xtr_df.shape, "| Valid:", Xte_df.shape)
        print("Kaydedildi: artifacts/train.parquet, artifacts/valid.parquet, artifacts/preprocessor.joblib")

    else:
        # ----- TÜM VERİYLE TEK DOSYA (MODEL ZORUNLU DEĞİLSE RAPORLAMA İÇİN) -----
        pre = build_preprocessor(num_cols, cat_cols, bin_cols)
        X_ready = pre.fit_transform(X)  # not: tüm veriyle fit (yalnız raporlama için makul)
        feat_names = feature_names_after_fit(pre, num_cols, cat_cols, bin_cols)

        feat_df = pd.DataFrame(X_ready, columns=feat_names)
        feat_df["__target__TedaviSuresi"] = np.array(y)

        # çıktı + preprocessor
        save_parquet(feat_df, out_path)
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(pre, "artifacts/preprocessor.joblib")

        print("✅ Preprocess (full) bitti")
        print("Kaydedilen dosya:", out_path)
        print("Toplam örnek:", len(feat_df), "Toplam sütun:", feat_df.shape[1])
        print("Preprocessor kaydedildi: artifacts/preprocessor.joblib")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Ham Excel dosya yolu (örn. data/raw/Talent_Academy_Case_DT_2025.xlsx)")
    ap.add_argument("--out", default="artifacts/clean_dataset.parquet", help="Tek dosya çıktısı için parquet yolu")
    ap.add_argument("--train-test", action="store_true", help="Aktifse train/valid dosyaları üretir ve leakage önlenir")
    ap.add_argument("--test-size", type=float, default=0.2, help="Valid oranı (train-test modunda)")
    ap.add_argument("--random-state", type=int, default=42, help="Rastgelelik tohumu")
    args = ap.parse_args()

    main(
        data_path=args.data,
        out_path=args.out,
        do_split=args.train_test,
        test_size=args.test_size,
        random_state=args.random_state,
    )

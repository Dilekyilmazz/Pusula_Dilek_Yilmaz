**Ad Soyad:** Dilek Yılmaz  
**E-posta:** dilekyilmaz000@gmail.com  

---

## 📌 Proje Özeti
Bu proje, **Pusula Data Science Internship Case Study** kapsamında verilen `Talent_Academy_Case_DT_2025.xlsx` veri seti üzerinde **Keşifsel Veri Analizi (EDA)** ve **ön-işleme (preprocessing)** adımlarını içermektedir.  
Amaç, ham veriyi eksik değerlerden arındırmak, kategorik ve çoklu etiketli değişkenleri uygun formata dönüştürmek ve veriyi makine öğrenmesi modelleri için hazır hale getirmektir.  

Yapılan başlıca adımlar:
- Eksik değer analizi ve doldurma
- String formatlı hedef değişkenin sayısallaştırılması
- Kategorik değişkenlerin OneHotEncoder ile encode edilmesi
- Multi-label kolonların (KronikHastalık, Alerji, Tanılar, UygulamaYerleri) dummy sütunlara genişletilmesi
- Sayısal değişkenlerin ölçeklenmesi (StandardScaler)
- Temizlenmiş verinin kaydedilmesi (`clean_dataset.parquet`)

EDA bulguları ve ön-işleme kararlarının detayları `reports/EDA_and_Preprocess.md` dosyasında yer almaktadır.

---

## ⚙️ Çalıştırma Talimatları

### Gereksinimler
Python 3.10+ ve aşağıdaki kütüphaneler:
- pandas  
- numpy  
- scikit-learn  
- matplotlib  

İsteğe bağlı:  
- `openpyxl` (Excel dosyalarını okumak için)  
- `joblib` (pipeline kaydetmek için)  

1. Ortam Kurulumu
Proje klasöründe sanal ortam oluşturun ve bağımlılıkları yükleyin:

```bash
python -m venv .venv
.\.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # Linux / Mac

pip install -r requirements.txt

2. Ön-İşleme Kodunu Çalıştırma
# Tüm veriyi temizle ve tek dosya kaydet
python src/preprocess.py --data data/raw/Talent_Academy_Case_DT_2025.xlsx --out artifacts/clean_dataset.parquet

# Train/Test ayrımı ile çalıştır (opsiyonel)
python src/preprocess.py --data data/raw/Talent_Academy_Case_DT_2025.xlsx --train-test

3. EDA Notebook
EDA ve görselleştirmeler için notebook’u açabilirsiniz:

jupyter notebook notebooks/clean_dataset_eda.ipynb

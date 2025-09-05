import pandas as pd

# Parquet'i oku
df = pd.read_parquet("artifacts/clean_dataset.parquet")

# CSV olarak kaydet
df.to_csv("artifacts/clean_dataset_new.csv", index=False)

# Excel olarak kaydet
df.to_excel("artifacts/clean_dataset_new.xlsx", index=False)

print("✅ Dönüşüm tamamlandı. artifacts klasöründe CSV ve Excel dosyaları oluştu.")

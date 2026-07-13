import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("mysql+mysqlconnector://root:@127.0.0.1:3306/genz_erd")

files = {
    "country": "country.csv",
    "gender": "gender.csv",
    "addiction_level": "addiction_level.csv",
    "usage_purpose": "usage_purpose.csv",
    "platform": "platform.csv",
    "user": "user.csv",
    "user_session": "user_session.csv",
    "health_behavior": "health_behavior.csv",
}

folder = r"C:\Users\Advan\Downloads"

for table_name, filename in files.items():
    path = f"{folder}\\{filename}"
    df = pd.read_csv(path)
    print(f"Importing {table_name}: {df.shape[0]} rows, {df.shape[1]} columns")
    df.to_sql(
        table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=500,   # kirim per 500 baris, biar gak kena limit packet
        method="multi",
    )
    print(f"  -> Selesai: {table_name}")

print("Semua tabel berhasil diimport!")
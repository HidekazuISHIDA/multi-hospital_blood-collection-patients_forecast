"""
B病院 ファインチューニングスクリプト
Automatically converted from Jupyter Notebook for reproducibility.
"""

# ==== Cell 0 ====
# 🏥 B病院 日次予測モデル ファインチューニングスクリプト

# =================================================================
# 1. ライブラリのインストール
# =================================================================
!pip -q install japanize-matplotlib jpholiday scikit-learn xgboost shap

# =================================================================
# 2. ライブラリのインポート
# =================================================================
import io
import json
import warnings
from datetime import date, timedelta

import jpholiday
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from google.colab import files
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =================================================================
# 3. ファイルのアップロード
# =================================================================
print("--- 🏥 B病院のファインチューニングを開始します ---")

print("\n--- A病院の学習済みファイルをアップロードしてください ---")
print("1. A病院のモデルファイル (model_A_daily.json)")
uploaded_model_A = files.upload()

print("\n2. A病院のパラメータファイル (best_params_A_daily.json)")
uploaded_params_A = files.upload()

print("\n--- B病院のデータをアップロードしてください ---")
print("3. B病院の採血ログファイル")
uploaded_log_B = files.upload()

print("\n4. B病院の外来患者数ファイル")
uploaded_patients_B = files.upload()

print("\n5. 気象データ")
uploaded_weather_B = files.upload()

# =================================================================
# 4. ファインチューニング実行
# =================================================================
try:
    # --- 4a. B病院のデータ処理 ---
    print("\n--- B病院のデータ処理を開始します ---")

    # カレンダーフィーチャ作成
    start_dt, end_dt = date(2024, 1, 1), date(2026, 12, 31)
    all_dates_list = [start_dt + timedelta(days=d) for d in range((end_dt - start_dt).days + 1)]
    calendar_features = pd.DataFrame({'date': pd.to_datetime(all_dates_list)})

    # 祝日・休日判定
    is_holiday_series = calendar_features['date'].apply(
        lambda x: (
            jpholiday.is_holiday(x)
            or x.weekday() >= 5           # 土日
            or (x.month == 12 and x.day >= 29)  # 年末
            or (x.month == 1 and x.day <= 3)    # 年始
        )
    )
    calendar_features['is_holiday'] = is_holiday_series
    calendar_features['月'] = calendar_features['date'].dt.month

    weekday_map = {0: '月', 1: '火', 2: '水', 3: '木', 4: '金', 5: '土', 6: '日'}
    calendar_features['曜日'] = calendar_features['date'].dt.dayofweek.map(weekday_map)
    calendar_features['週回数'] = (calendar_features['date'].dt.day - 1) // 7 + 1
    calendar_features['前日祝日フラグ'] = calendar_features['is_holiday'].shift(1).fillna(False)

    # 採血ログ読み込み（B病院）
    log_df_b = pd.read_csv(io.BytesIO(next(iter(uploaded_log_B.values()))), encoding='shift-jis')
    log_df_b = log_df_b[log_df_b['処理'] == '終了'].copy()
    print(f"✅ ログデータを「終了」レコードのみにフィルタリングしました。対象レコード数: {len(log_df_b)}")

    log_df_b['実施日'] = pd.to_datetime(log_df_b['実施日'], format='%y/%m/%d', errors='coerce')
    daily_blood_patients_b = log_df_b.groupby('実施日').size().reset_index(name='blood_patient_count')

    # 外来患者数データ（B病院）
    total_patients_df_b = pd.read_csv(
        io.BytesIO(next(iter(uploaded_patients_B.values()))),
        encoding='utf-8-sig',
        thousands=','
    )
    total_patients_df_b['date'] = pd.to_datetime(total_patients_df_b['date'], errors='coerce')

    # 気象データ（B病院：東京）
    weather_raw_df_b = pd.read_csv(
        io.BytesIO(next(iter(uploaded_weather_B.values()))),
        encoding='shift-jis',
        header=None,
        skiprows=3
    )

    # ヘッダ行取得 & データ本体
    header_b = weather_raw_df_b.iloc[0]
    weather_df_b = weather_raw_df_b.iloc[3:].reset_index(drop=True)
    weather_df_b.columns = header_b

    # 必要なカラムのみ抽出（列位置は元ファイル構成に依存）
    df_selected_b = weather_df_b.iloc[:, [0, 1, 5, 8, 11, 14, 20, 23]].copy()
    df_selected_b.columns = ['date', '降水量', '天気概況', '平均気温', '最高気温', '最低気温', '平均湿度', '平均風速']

    # 天気フラグ
    df_selected_b['雨フラグ'] = df_selected_b['天気概況'].str.contains('雨', na=False).astype(int)
    df_selected_b['雪フラグ'] = df_selected_b['天気概況'].str.contains('雪', na=False).astype(int)

    weather_features_b = df_selected_b.drop(columns=['天気概況'])
    weather_features_b['date'] = pd.to_datetime(weather_features_b['date'], errors='coerce')

    # 数値項目の欠損補完
    numeric_cols = ['降水量', '平均気温', '最高気温', '最低気温', '平均湿度', '平均風速']
    for col in numeric_cols:
        weather_features_b[col] = pd.to_numeric(weather_features_b[col], errors='coerce')
        weather_features_b[col].fillna(weather_features_b[col].mean(), inplace=True)

    # 各データをマージ
    df_b = pd.merge(calendar_features, total_patients_df_b, on='date', how='left')
    df_b = pd.merge(df_b, daily_blood_patients_b, left_on='date', right_on='実施日', how='left')
    df_b = pd.merge(df_b, weather_features_b, on='date', how='left')

    # 目的変数・外来患者数が揃っている日のみ使用
    df_b.dropna(subset=['blood_patient_count', 'total_outpatient_count'], inplace=True)

    # --- 4b. 曜日ダミーを A病院モデルと揃えて作成 ---
    df_b_base = df_b.drop(columns=['is_holiday', '実施日'], errors='ignore').copy()

    # 曜日ダミーを全て作成（drop_first=False）
    df_b_encoded = pd.get_dummies(df_b_base, columns=['曜日'], drop_first=False)

    # A病院モデルが使っている曜日ダミー（平日のみ）
    weekday_keep = ['曜日_月', '曜日_火', '曜日_水', '曜日_木', '曜日_金']

    # 土日があれば削除
    for col in ['曜日_土', '曜日_日']:
        if col in df_b_encoded.columns:
            df_b_encoded.drop(columns=[col], inplace=True)

    # 足りない曜日ダミーがあれば 0 で追加
    for col in weekday_keep:
        if col not in df_b_encoded.columns:
            df_b_encoded[col] = 0

    # 「前日祝日フラグ」は int に
    df_b_encoded['前日祝日フラグ'] = df_b_encoded['前日祝日フラグ'].astype(int)

    # A病院モデルが期待している特徴量のリストを明示的に指定
    expected_features = [
        '月',
        '週回数',
        '前日祝日フラグ',
        'total_outpatient_count',
        '降水量',
        '平均気温',
        '最高気温',
        '最低気温',
        '平均湿度',
        '平均風速',
        '雨フラグ',
        '雪フラグ',
        '曜日_月',
        '曜日_木',  # ★A病院モデル側の順番・名前に合わせる
        '曜日_水',
        '曜日_火',
        '曜日_金'
    ]

    # 足りない列があれば 0 で追加
    for col in expected_features:
        if col not in df_b_encoded.columns:
            df_b_encoded[col] = 0

    # --- 4c. B病院のデータ分割 ---
    train_df_b = df_b_encoded[df_b_encoded['date'].dt.year == 2024].copy()
    val_df_b = df_b_encoded[(df_b_encoded['date'].dt.year == 2025) & (df_b_encoded['date'].dt.month <= 8)].copy()

    # 特徴量は expected_features だけを使用
    features_b = expected_features

    X_train_B, y_train_B = train_df_b[features_b], train_df_b['blood_patient_count']
    X_val_B, y_val_B = val_df_b[features_b], val_df_b['blood_patient_count']

    print(f"✅ B病院データ分割完了。学習: {len(X_train_B)}件, 検証: {len(X_val_B)}件")

    # --- 4d. XGBoost ファインチューニング ---
    print("\n--- A病院モデルのファインチューニングを開始します ---")

    # A病院モデルを一時ファイルとして保存
    model_a_filename = list(uploaded_model_A.keys())[0]
    model_a_bytes = uploaded_model_A[model_a_filename]
    temp_model_path = 'temp_model_A_for_finetuning.json'
    with open(temp_model_path, 'wb') as f:
        f.write(model_a_bytes)

    # A病院の最適パラメータを読み込み
    best_params_from_A = json.load(io.BytesIO(next(iter(uploaded_params_A.values()))))
    print(f"\nA病院の最適パラメータを自動で読み込みました: {best_params_from_A}")

    # 読み込んだパラメータでモデルを初期化
    finetuned_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        **best_params_from_A
    )

    # A病院モデルをベースに B病院データで追加学習
    finetuned_model.fit(
        X_train_B,
        y_train_B,
        xgb_model=temp_model_path
    )
    print("✅ ファインチューニングが完了しました。")

    # --- 4e. ファインチューニング後の性能評価 ---
    print("\n--- 検証データでファインチューニング後の性能を評価します ---")
    y_pred_b = finetuned_model.predict(X_val_B)
    rmse_b = np.sqrt(mean_squared_error(y_val_B, y_pred_b))
    r2_b = r2_score(y_val_B, y_pred_b)
    print(f"📈 評価指標 (B病院ファインチューニング後): RMSE={rmse_b:.2f}, R2スコア={r2_b:.3f}")

    # --- 4f. B病院用 artifact の保存とダウンロード ---
    scaler_B = StandardScaler().fit(X_train_B)
    joblib.dump(scaler_B, 'model_B_daily_scaler.joblib')
    finetuned_model.save_model('model_B_daily_finetuned.json')

    print("\n✅ B病院用のファインチューニング済みファイルを2つ作成しました。ダウンロードを開始します。")
    files.download('model_B_daily_finetuned.json')
    files.download('model_B_daily_scaler.joblib')

except Exception as e:
    print(f"\n予期せぬエラーが発生しました: {e}")


# ==== Cell 1 ====
# 🏥 B病院 時間帯別モデル ファインチューニングスクリプト
# =================================================================
# 1. ライブラリのインストール
# =================================================================
!pip -q install japanize-matplotlib jpholiday scikit-learn xgboost shap

# =================================================================
# 2. ライブラリのインポート
# =================================================================
import pandas as pd
import numpy as np
import jpholiday
from datetime import date, timedelta
import io
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# 3. ファイルのアップロード
# =================================================================
print("--- 🏥 B病院のファインチューニングを開始します ---")
print("\n--- A病院の学習済みファイルをアップロードしてください ---")
print("1. A病院のモデルファイル (model_A_timeseries.json)")
uploaded_model_A = files.upload()
print("\n2. A病院の特徴量リスト (columns_A_timeseries.json)")
uploaded_columns_A = files.upload()

print("\n--- B病院のデータをアップロードしてください ---")
print("3. B病院の採血ログファイル")
uploaded_log_B = files.upload()
print("\n4. B病院の外来患者数ファイル")
uploaded_patients_B = files.upload()
print("\n5. 気象データ")
uploaded_weather_B = files.upload()

# =================================================================
# 4. ファインチューニング実行
# =================================================================
try:
    # --- 4a. 全データの読み込み ---
    print("\n--- データ読み込みと初期処理を開始します ---")
    log_df = pd.read_csv(io.BytesIO(next(iter(uploaded_log_B.values()))), encoding='shift-jis')
    total_patients_df = pd.read_csv(io.BytesIO(next(iter(uploaded_patients_B.values()))), encoding='utf-8-sig', thousands=',')
    weather_raw_df = pd.read_csv(io.BytesIO(next(iter(uploaded_weather_B.values()))), encoding='shift-jis', header=None, skiprows=3)
    columns_A = json.load(io.BytesIO(next(iter(uploaded_columns_A.values()))))

    # 「処理」列が「終了」の行のみにフィルタリング
    log_df = log_df[log_df['処理'] == '終了'].copy()
    print(f"✅ ログデータを「終了」レコードのみにフィルタリングしました。対象レコード数: {len(log_df)}")

    # --- 4b. 時系列データの作成 ---
    print("\n--- ログデータを30分単位の時系列データに変換します ---")
    log_df['time'] = pd.to_datetime(log_df['受信'], errors='coerce')
    log_df.dropna(subset=['time'], inplace=True)
    log_df.set_index('time', inplace=True)
    patient_count_by_slot = log_df.resample('30T').size().rename('patient_count_slot')
    patient_count_by_slot = patient_count_by_slot.between_time('08:00', '18:00')
    df_ts = patient_count_by_slot.to_frame()

    # --- 4c. 特徴量エンジニアリング ---
    print("\n--- 時間帯予測用の特徴量を作成します ---")
    df_ts['hour'] = df_ts.index.hour
    df_ts['minute'] = df_ts.index.minute
    df_ts['dayofweek'] = df_ts.index.dayofweek
    df_ts['date'] = pd.to_datetime(df_ts.index.date)
    df_ts['is_first_slot'] = ((df_ts['hour'] == 8) & (df_ts['minute'] == 0)).astype(int)
    df_ts['is_second_slot'] = ((df_ts['hour'] == 8) & (df_ts['minute'] == 30)).astype(int)
    daily_dates = pd.DataFrame({'date': pd.to_datetime(df_ts['date'].unique())})
    daily_dates['is_holiday_daily'] = daily_dates['date'].apply(
        lambda x: jpholiday.is_holiday(x) or x.weekday() >= 5 or \
                    (x.month == 12 and x.day >= 29) or (x.month == 1 and x.day <= 3)
    )
    daily_dates['月'] = daily_dates['date'].dt.month
    daily_dates['週回数'] = (daily_dates['date'].dt.day - 1) // 7 + 1
    daily_dates['前日祝日フラグ'] = daily_dates['is_holiday_daily'].shift(1).fillna(False).astype(int)
    total_patients_df['date'] = pd.to_datetime(total_patients_df['date'], errors='coerce')
    header_w = weather_raw_df.iloc[0]
    weather_df = weather_raw_df.iloc[3:].reset_index(drop=True); weather_df.columns = header_w
    weather_features = weather_df.iloc[:, [0, 1, 5, 8, 11, 14, 20, 23]].copy()
    weather_features.columns = ['date', '降水量', '天気概況', '平均気温', '最高気温', '最低気温', '平均湿度', '平均風速']
    weather_features['date'] = pd.to_datetime(weather_features['date'], errors='coerce')
    numeric_cols = ['降水量', '平均気温', '最高気温', '最低気温', '平均湿度', '平均風速']
    for col in numeric_cols:
        weather_features[col] = pd.to_numeric(weather_features[col], errors='coerce')
        weather_features[col].fillna(weather_features[col].mean(), inplace=True)
    weather_features['雨フラグ'] = weather_features['天気概況'].str.contains('雨', na=False).astype(int)
    weather_features['雪フラグ'] = weather_features['天気概況'].str.contains('雪', na=False).astype(int)
    weather_features['天気カテゴリ'] = weather_features['天気概況'].str[0]
    df_ts = pd.merge(df_ts, daily_dates, on='date', how='left')
    df_ts = pd.merge(df_ts, total_patients_df, on='date', how='left')
    df_ts = pd.merge(df_ts, weather_features.drop(columns=['天気概況']), on='date', how='left')
    df_ts = pd.get_dummies(df_ts, columns=['dayofweek', '天気カテゴリ'], drop_first=True)
    df_ts['lag_30min'] = df_ts['patient_count_slot'].shift(1)
    df_ts['lag_60min'] = df_ts['patient_count_slot'].shift(2)
    df_ts['lag_90min'] = df_ts['patient_count_slot'].shift(3)
    df_ts['rolling_mean_60min'] = df_ts['patient_count_slot'].shift(1).rolling(window=2).mean()
    df_ts.rename(columns={'is_holiday_daily': 'is_holiday'}, inplace=True)
    df_ts.fillna(0, inplace=True)

    # A病院の学習時と特徴量の構成を完全に一致させる
    for col in columns_A:
        if col not in df_ts.columns:
            df_ts[col] = 0 # B病院データに存在しない特徴量があれば0で埋める
    df_ts = df_ts[columns_A + ['date', 'patient_count_slot']] # カラムの順序をA病院と合わせる

    # --- 4d. B病院のデータ分割 ---
    train_df = df_ts[df_ts['date'].dt.year == 2024].copy()
    val_df = df_ts[(df_ts['date'].dt.year == 2025) & (df_ts['date'].dt.month <= 8)].copy()
    train_df.drop(columns=['date'], inplace=True)
    val_df.drop(columns=['date'], inplace=True)

    features = [col for col in train_df.columns if col != 'patient_count_slot']
    X_train_B, y_train_B = train_df[features], train_df['patient_count_slot']
    X_val_B, y_val_B = val_df[features], val_df['patient_count_slot']
    print(f"\n✅ B病院データ分割完了。学習: {len(X_train_B)}件, 検証: {len(X_val_B)}件")

    # --- 4e. XGBoostファインチューニング ---
    print("\n--- A病院モデルのファインチューニングを開始します ---")

    # アップロードされたA病院モデルを一時ファイルとして保存
    model_a_filename = list(uploaded_model_A.keys())[0]
    model_a_bytes = uploaded_model_A[model_a_filename]
    temp_model_path = 'temp_model_A_for_finetuning.json'
    with open(temp_model_path, 'wb') as f:
        f.write(model_a_bytes)

    # A病院モデルをベースに追加学習
    finetuned_model = xgb.XGBRegressor(
        objective='reg:squarederror', n_estimators=1000, learning_rate=0.01,
        max_depth=6, early_stopping_rounds=50, random_state=42
    )
    finetuned_model.fit(
        X_train_B, y_train_B,
        eval_set=[(X_val_B, y_val_B)],
        xgb_model=temp_model_path,
        verbose=False
    )
    print("✅ ファインチューニングが完了しました。")

    # --- 4f. 性能評価 ---
    print("\n--- 検証データでファインチューニング後の性能を評価します ---")
    y_pred_b = finetuned_model.predict(X_val_B)
    rmse_b = np.sqrt(mean_squared_error(y_val_B, y_pred_b))
    r2_b = r2_score(y_val_B, y_pred_b)
    print(f"📈 評価指標 (B病院ファインチューニング後): RMSE={rmse_b:.2f} 人/30分, R2スコア={r2_b:.3f}")

    # --- 4g. 成果物の保存とダウンロード ---
    finetuned_model.save_model('model_B_timeseries_finetuned.json')
    with open('columns_B_timeseries.json', 'w') as f:
        json.dump(features, f)

    print("\n✅ B病院用のファインチューニング済みファイルを2つ作成しました。ダウンロードを開始します。")
    files.download('model_B_timeseries_finetuned.json')
    files.download('columns_B_timeseries.json')

except Exception as e:
    print(f"\n予期せぬエラーが発生しました: {e}")


# ==== Cell 2 ====
# 🏥 B病院 待ち人数・待ち時間モデル ファインチューニング
# =================================================================
# 1. ライブラリのインストール
# =================================================================
!pip -q install japanize-matplotlib jpholiday scikit-learn xgboost shap

# =================================================================
# 2. ライブラリのインポート
# =================================================================
import pandas as pd
import numpy as np
import jpholiday
from datetime import date, timedelta
import io
import json
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# =================================================================
# 3. ファイルのアップロード
# =================================================================
print("--- 🏥 B病院のファインチューニングを開始します ---")
print("\n--- A病院の学習済みファイルをアップロードしてください ---")
print("1. A病院の待ち時間モデル (model_A_waittime_30min.json)")
uploaded_model_waittime_A = files.upload()
print("\n2. A病院の待ち人数モデル (model_A_queue_30min.json)")
uploaded_model_queue_A = files.upload()
print("\n3. A病院の特徴量リスト (columns_A_multi_30min.json)")
uploaded_columns_A = files.upload()

print("\n--- B病院のデータをアップロードしてください ---")
print("4. B病院の採血ログファイル")
uploaded_log_B = files.upload()
print("\n5. B病院の外来患者数ファイル")
uploaded_patients_B = files.upload()
print("\n6. 気象データ")
uploaded_weather_B = files.upload()

# =================================================================
# 4. ファインチューニング実行
# =================================================================
try:
    # --- 4a. 全データの読み込み ---
    print("\n--- データ読み込みと初期処理を開始します ---")
    log_df = pd.read_csv(io.BytesIO(next(iter(uploaded_log_B.values()))), encoding='shift-jis')
    total_patients_df = pd.read_csv(io.BytesIO(next(iter(uploaded_patients_B.values()))), encoding='utf-8-sig', thousands=',')
    weather_raw_df = pd.read_csv(io.BytesIO(next(iter(uploaded_weather_B.values()))), encoding='shift-jis', header=None, skiprows=3)
    columns_A = json.load(io.BytesIO(next(iter(uploaded_columns_A.values()))))

    log_df = log_df[log_df['処理'] == '終了'].copy()
    print(f"✅ ログデータを「終了」レコードのみにフィルタリングしました。対象レコード数: {len(log_df)}")

    # --- 4b. 待ち時間と各イベント時間の算出 ---
    log_df['reception_time'] = pd.to_datetime(log_df['受信'], errors='coerce')
    log_df['call_time'] = pd.to_datetime(log_df['指示書'], errors='coerce')
    log_df.dropna(subset=['reception_time', 'call_time'], inplace=True)
    log_df['wait_minutes'] = (log_df['call_time'] - log_df['reception_time']).dt.total_seconds() / 60
    log_df = log_df[(log_df['wait_minutes'] >= 0) & (log_df['wait_minutes'] <= 180)]

    # --- 4c. 30分単位でのデータ集計 ---
    print("\n--- データを30分単位に集計します ---")
    avg_wait = log_df.set_index('call_time')['wait_minutes'].resample('30T').mean().to_frame(name='avg_wait_minutes')
    receptions = log_df.set_index('reception_time').resample('30T').size().to_frame(name='reception_count')
    calls = log_df.set_index('call_time').resample('30T').size().to_frame(name='call_count')
    df_30min = pd.concat([avg_wait, receptions, calls], axis=1).fillna(0)
    df_30min = df_30min.between_time('08:00', '18:00')

    # --- 4d. 特徴量エンジニアリング ---
    print("\n--- 予測用の特徴量を作成します ---")
    df_30min['hour'] = df_30min.index.hour
    df_30min['minute'] = df_30min.index.minute
    df_30min['dayofweek'] = df_30min.index.dayofweek
    df_30min['date'] = pd.to_datetime(df_30min.index.date)

    daily_dates = pd.DataFrame({'date': pd.to_datetime(df_30min['date'].unique())})
    daily_dates['is_holiday'] = daily_dates['date'].apply(
        lambda x: jpholiday.is_holiday(x) or x.weekday() >= 5 or \
                    (x.month == 12 and x.day >= 29) or (x.month == 1 and x.day <= 3)
    )
    daily_dates['月'] = daily_dates['date'].dt.month
    daily_dates['週回数'] = (daily_dates['date'].dt.day - 1) // 7 + 1
    daily_dates['前日祝日フラグ'] = daily_dates['is_holiday'].shift(1).fillna(False).astype(int)

    total_patients_df['date'] = pd.to_datetime(total_patients_df['date'], errors='coerce')
    header_w = weather_raw_df.iloc[0]
    weather_df = weather_raw_df.iloc[3:].reset_index(drop=True); weather_df.columns = header_w
    weather_features = weather_df.iloc[:, [0, 1, 5, 8, 11, 14, 20, 23]].copy()
    weather_features.columns = ['date', '降水量', '天気概況', '平均気温', '最高気温', '最低気温', '平均湿度', '平均風速']
    weather_features['date'] = pd.to_datetime(weather_features['date'], errors='coerce')
    numeric_cols = ['降水量', '平均気温', '最高気温', '最低気温', '平均湿度', '平均風速']
    for col in numeric_cols:
        weather_features[col] = pd.to_numeric(weather_features[col], errors='coerce')
        weather_features[col].fillna(weather_features[col].mean(), inplace=True)
    weather_features['雨フラグ'] = weather_features['天気概況'].str.contains('雨', na=False).astype(int)
    weather_features['雪フラグ'] = weather_features['天気概況'].str.contains('雪', na=False).astype(int)
    weather_features['天気カテゴリ'] = weather_features['天気概況'].str[0]

    df_30min = pd.merge(df_30min, daily_dates, on='date', how='left')
    df_30min = pd.merge(df_30min, total_patients_df, on='date', how='left')
    df_30min = pd.merge(df_30min, weather_features.drop(columns=['天気概況']), on='date', how='left')
    df_30min = pd.get_dummies(df_30min, columns=['dayofweek', '天気カテゴリ'], drop_first=True)

    df_30min['net_flow'] = df_30min['reception_count'] - df_30min['call_count']
    df_30min['queue_size'] = df_30min.groupby('date')['net_flow'].cumsum()
    df_30min['queue_at_start_of_slot'] = df_30min['queue_size'].shift(1).fillna(0)
    day_starts = df_30min['date'] != df_30min['date'].shift(1)
    df_30min.loc[day_starts, 'queue_at_start_of_slot'] = 0

    df_30min.fillna(0, inplace=True)
    df_30min['is_holiday'] = df_30min['is_holiday'].astype(int)

    # A病院の学習時と特徴量の構成を完全に一致させる
    for col in columns_A:
        if col not in df_30min.columns:
            df_30min[col] = 0
    df_30min = df_30min[columns_A + ['date', 'avg_wait_minutes', 'queue_size']]

    # --- 4e. B病院のデータ分割 ---
    train_df = df_30min[df_30min['date'].dt.year == 2024].copy()
    val_df = df_30min[df_30min['date'].dt.year == 2025].copy()

    targets = ['avg_wait_minutes', 'queue_size']
    features = [col for col in train_df.columns if col not in targets + ['date']]

    X_train_B, y_train_B = train_df[features], train_df[targets]
    X_val_B, y_val_B = val_df[features], val_df[targets]
    print(f"✅ B病院データ分割完了。学習: {len(X_train_B)}件, 検証: {len(X_val_B)}件")

    # --- 4f. 2つのモデルをそれぞれファインチューニング ---
    print("\n--- 2つのモデルを個別にファインチューニングします ---")

    # A病院モデルを一時ファイルとして保存
    model_waittime_A_filename = list(uploaded_model_waittime_A.keys())[0]
    with open('temp_model_waittime_A.json', 'wb') as f:
        f.write(uploaded_model_waittime_A[model_waittime_A_filename])

    model_queue_A_filename = list(uploaded_model_queue_A.keys())[0]
    with open('temp_model_queue_A.json', 'wb') as f:
        f.write(uploaded_model_queue_A[model_queue_A_filename])

    # モデル1: 待ち時間モデルのファインチューニング
    model_waittime_ft = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model_waittime_ft.fit(X_train_B, y_train_B['avg_wait_minutes'], xgb_model='temp_model_waittime_A.json')
    print("✅ 平均待ち時間予測モデルのファインチューニングが完了しました。")

    # モデル2: 待ち人数モデルのファインチューニング
    model_queue_ft = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    model_queue_ft.fit(X_train_B, y_train_B['queue_size'], xgb_model='temp_model_queue_A.json')
    print("✅ 待ち人数予測モデルのファインチューニングが完了しました。")

    # --- 4g. 両モデルの性能評価 ---
    print("\n--- 検証データで両モデルの性能を評価します ---")
    pred_waittime = model_waittime_ft.predict(X_val_B)
    rmse_wt = np.sqrt(mean_squared_error(y_val_B['avg_wait_minutes'], pred_waittime))
    r2_wt = r2_score(y_val_B['avg_wait_minutes'], pred_waittime)
    print(f"📈 待ち時間モデル評価: RMSE: 約 {rmse_wt:.2f} 分, R2スコア: {r2_wt:.3f}")

    pred_queue = model_queue_ft.predict(X_val_B)
    rmse_q = np.sqrt(mean_squared_error(y_val_B['queue_size'], pred_queue))
    r2_q = r2_score(y_val_B['queue_size'], pred_queue)
    print(f"📈 待ち人数モデル評価: RMSE: 約 {rmse_q:.2f} 人, R2スコア: {r2_q:.3f}")

    # --- 4h. 成果物の保存 ---
    model_waittime_ft.save_model('model_B_waittime_30min_ft.json')
    model_queue_ft.save_model('model_B_queue_30min_ft.json')
    with open('columns_B_multi_30min.json', 'w') as f:
        json.dump(features, f)

    print("\n✅ B病院用の待ち時間・待ち人数予測モデル(30分単位)とカラムリストを保存しました。")
    files.download('model_B_waittime_30min_ft.json')
    files.download('model_B_queue_30min_ft.json')
    files.download('columns_B_multi_30min.json')

except Exception as e:
    print(f"\n予期せぬエラーが発生しました: {e}")


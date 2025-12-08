"""
A病院用 予測アプリ UI スクリプト
Automatically converted from Jupyter Notebook for reproducibility.
"""

# ==== Cell 0 ====
# 🏥 A病院 日次採血患者数予測アプリ (Google Drive版)
# =================================================================
# 1. ライブラリのインストール
# =================================================================
!pip -q install ipywidgets jpholiday scikit-learn xgboost pandas

# =================================================================
# 2. ライブラリのインポート
# =================================================================
import pandas as pd
import numpy as np
import jpholiday
from datetime import date, timedelta
import joblib
import xgboost as xgb
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import drive

# =================================================================
# 3. Google Driveの連携
# =================================================================
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"エラー: Google Driveの連携に失敗しました。{e}")

# =================================================================
# 4. 予測用の特徴量を作成する関数
# =================================================================
def create_features_for_pred(target_date, patient_count, weather, feature_columns):
    """UIからの入力値を受け取り、学習時と同じ形式のデータフレームを返す"""
    target_dt = pd.to_datetime(target_date)

    is_holiday = (jpholiday.is_holiday(target_dt) or target_dt.weekday() >= 5 or
                  (target_dt.month == 12 and target_dt.day >= 29) or (target_dt.month == 1 and target_dt.day <= 3))
    prev_dt = target_dt - timedelta(days=1)
    is_prev_holiday = (jpholiday.is_holiday(prev_dt) or prev_dt.weekday() >= 5 or
                       (prev_dt.month == 12 and prev_dt.day >= 29) or (prev_dt.month == 1 and prev_dt.day <= 3))

    pred_df = pd.DataFrame(columns=feature_columns); pred_df.loc[0] = 0

    if '月' in pred_df.columns: pred_df['月'] = target_dt.month
    if '週回数' in pred_df.columns: pred_df['週回数'] = (target_dt.day - 1) // 7 + 1
    if '前日祝日フラグ' in pred_df.columns: pred_df['前日祝日フラグ'] = int(is_prev_holiday)
    if 'total_outpatient_count' in pred_df.columns: pred_df['total_outpatient_count'] = patient_count
    if '雨フラグ' in pred_df.columns: pred_df['雨フラグ'] = 1 if weather == '雨' else 0
    if '雪フラグ' in pred_df.columns: pred_df['雪フラグ'] = 1 if weather == '雪' else 0

    weekday_map = {0: '曜日_月', 1: '曜日_火', 2: '曜日_水', 3: '曜日_木', 4: '曜日_金', 5: '曜日_土'}
    weekday_col = weekday_map.get(target_dt.weekday())
    if weekday_col in pred_df.columns:
        pred_df[weekday_col] = 1

    return pred_df[feature_columns]

# =================================================================
# 5. アプリケーション本体
# =================================================================
try:
    print("\n--- 🏥 A病院 日次採血患者数予測アプリ ---")

    # --- ファイルパスをGoogle Driveに指定 ---
    DRIVE_FOLDER_PATH = '/content/drive/MyDrive/Colab_Data/Hospital_A/'
    MODEL_PATH = DRIVE_FOLDER_PATH + 'model_A_daily.json'
    SCALER_PATH = DRIVE_FOLDER_PATH + 'model_A_daily_scaler.joblib'

    # 1. モデルとスケーラーの読み込み
    model = xgb.XGBRegressor(); model.load_model(MODEL_PATH)
    # スケーラーはカラム名と順序の復元にのみ使用
    scaler = joblib.load(SCALER_PATH)

    print("\n✅ モデル読込完了。予測を開始できます。")

    # 2. UIウィジェットの作成
    date_picker = widgets.DatePicker(description='予測日', value=date.today() + timedelta(days=1))
    patient_input = widgets.IntText(value=1200, description='延べ外来患者数:')
    weather_input = widgets.RadioButtons(options=['晴', '曇', '雨', '雪'], value='晴', description='天気:')
    predict_button = widgets.Button(description='予測実行', button_style='success', icon='calculator')
    output_area = widgets.Output()

    # 3. ボタンクリック時の処理
    def on_predict_clicked(b):
        with output_area:
            clear_output()
            input_df = create_features_for_pred(date_picker.value, patient_input.value, weather_input.value, scaler.feature_names_in_)
            prediction = model.predict(input_df)

            print(f"--- 予測結果 ({date_picker.value}) ---")
            print(f"📅 条件: 外来患者 {patient_input.value}人, 天気 {weather_input.value}")
            print("---------------------------------")
            print(f"🚀 予測採血患者数: 約 {prediction[0]:.0f}人")

    predict_button.on_click(on_predict_clicked)

    # 4. UIの表示
    display(date_picker, patient_input, weather_input, predict_button, output_area)

except FileNotFoundError:
    print(f"エラー: 指定されたパスにファイルが見つかりません。")
    print(f"  - 確認したモデルパス: {MODEL_PATH}")
    print(f"  - 確認したスケーラーパス: {SCALER_PATH}")
    print("\nGoogle Driveのフォルダ構成と、コード内の'DRIVE_FOLDER_PATH'が一致しているか確認してください。")
except Exception as e:
    print(f"\n予期せぬエラーが発生しました: {e}")


# ==== Cell 1 ====
# 🏥 A病院 採血患者数予測アプリ (5営業日・空白スキップ対応版)
# =================================================================
# 1. ライブラリのインストール
# =================================================================
!pip -q install ipywidgets jpholiday scikit-learn xgboost pandas

# =================================================================
# 2. ライブラリのインポート
# =================================================================
import pandas as pd
import numpy as np
import jpholiday
from datetime import date, timedelta
import joblib
import xgboost as xgb
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import drive

# =================================================================
# 3. Google Driveの連携
# =================================================================
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"エラー: Google Driveの連携に失敗しました。{e}")

# =================================================================
# 4. ヘルパー関数
# =================================================================
def create_features_for_pred(target_date, patient_count, weather, feature_columns):
    """UIからの入力値を受け取り、学習時と同じ形式のデータフレームを返す"""
    target_dt = pd.to_datetime(target_date)
    is_holiday = (jpholiday.is_holiday(target_dt) or target_dt.weekday() >= 5 or
                  (target_dt.month == 12 and target_dt.day >= 29) or (target_dt.month == 1 and target_dt.day <= 3))
    prev_dt = target_dt - timedelta(days=1)
    is_prev_holiday = (jpholiday.is_holiday(prev_dt) or prev_dt.weekday() >= 5 or
                       (prev_dt.month == 12 and prev_dt.day >= 29) or (prev_dt.month == 1 and prev_dt.day <= 3))

    pred_df = pd.DataFrame(columns=feature_columns); pred_df.loc[0] = 0

    if '月' in pred_df.columns: pred_df['月'] = target_dt.month
    if '週回数' in pred_df.columns: pred_df['週回数'] = (target_dt.day - 1) // 7 + 1
    if '前日祝日フラグ' in pred_df.columns: pred_df['前日祝日フラグ'] = int(is_prev_holiday)
    if 'total_outpatient_count' in pred_df.columns: pred_df['total_outpatient_count'] = patient_count
    if '雨フラグ' in pred_df.columns: pred_df['雨フラグ'] = 1 if weather == '雨' else 0
    if '雪フラグ' in pred_df.columns: pred_df['雪フラグ'] = 1 if weather == '雪' else 0

    weekday_map = {0: '曜日_月', 1: '曜日_火', 2: '曜日_水', 3: '曜日_木', 4: '曜日_金', 5: '曜日_土'}
    weekday_col = weekday_map.get(target_dt.weekday())
    if weekday_col in pred_df.columns:
        pred_df[weekday_col] = 1

    return pred_df[feature_columns]

def get_next_n_business_days(start_date, n):
    """開始日から次のn営業日を計算してリストで返す"""
    business_days = []
    current_date = pd.to_datetime(start_date)
    while len(business_days) < n:
        is_business_day = not (jpholiday.is_holiday(current_date) or current_date.weekday() >= 5 or \
                               (current_date.month == 12 and current_date.day >= 29) or \
                               (current_date.month == 1 and current_date.day <= 3))
        if is_business_day:
            business_days.append(current_date)
        current_date += timedelta(days=1)
    return business_days

# =================================================================
# 5. アプリケーション本体
# =================================================================
try:
    print("\n--- 🏥 A病院 採血患者数予測アプリ (5営業日予測版) ---")

    # --- ファイルパスをGoogle Driveに指定 ---
    DRIVE_FOLDER_PATH = '/content/drive/MyDrive/Colab_Data/Hospital_A/'
    MODEL_PATH = DRIVE_FOLDER_PATH + 'model_A_daily.json'
    SCALER_PATH = DRIVE_FOLDER_PATH + 'model_A_daily_scaler.joblib'

    # 1. モデルとスケーラーの読み込み
    model = xgb.XGBRegressor(); model.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH) # カラム復元用に読み込む

    print("\n✅ モデル読込完了。予測を開始できます。")

    # --- 2. UIウィジェットの作成 ---
    start_date_picker = widgets.DatePicker(description='予測開始日', value=date.today() + timedelta(days=1))
    patient_inputs = [widgets.Text(value='1200', description='') for _ in range(5)]
    predict_button = widgets.Button(description='予測実行', button_style='success', icon='calendar')
    output_area = widgets.Output()

    # --- 3. 動的なUI更新処理 ---
    def update_input_labels(change):
        start_date = pd.to_datetime(change['new'])
        business_days = get_next_n_business_days(start_date, 5)
        weekday_jp = ['月', '火', '水', '木', '金', '土', '日']
        for i, day in enumerate(business_days):
            patient_inputs[i].description = f"{day.strftime('%m/%d')} ({weekday_jp[day.weekday()]}):"

    # --- 4. ボタンクリック時の処理 ---
    def on_predict_clicked(b):
        with output_area:
            clear_output(wait=True)
            start_date = pd.to_datetime(start_date_picker.value)
            business_days = get_next_n_business_days(start_date, 5)
            results = []

            for i in range(5):
                if patient_inputs[i].value.strip():
                    try:
                        patient_count = int(patient_inputs[i].value)
                        current_date = business_days[i]
                        # 常に「晴れ」の条件で予測
                        input_df = create_features_for_pred(current_date, patient_count, '晴', scaler.feature_names_in_)
                        prediction = model.predict(input_df)

                        results.append({
                            '日付': current_date.strftime('%Y-%m-%d'),
                            '曜日': ['月', '火', '水', '木', '金', '土', '日'][current_date.weekday()],
                            '延べ外来患者数': patient_count,
                            '予測採血患者数': int(round(prediction[0]))
                        })
                    except ValueError:
                        pass # 数字以外の入力はスキップ

            if results:
                result_df = pd.DataFrame(results)
                print("--- 予測結果 ---")
                display(result_df)
            else:
                print("延べ外来患者数が入力されていないため、予測は実行されませんでした。")

    # --- 5. UIの初期化と表示 ---
    predict_button.on_click(on_predict_clicked)
    start_date_picker.observe(update_input_labels, names='value')
    update_input_labels({'new': start_date_picker.value}) # 初期ラベルを設定

    display(start_date_picker)
    print("--- 各営業日の延べ外来患者数を入力してください (空白の場合は予測しません) ---")
    for widget in patient_inputs:
        display(widget)
    display(predict_button, output_area)

except FileNotFoundError:
    print(f"エラー: 指定されたパスにファイルが見つかりません。")
    print(f"  - 確認したモデルパス: {MODEL_PATH}")
    print(f"  - 確認したスケーラーパス: {SCALER_PATH}")
    print("\nGoogle Driveのフォルダ構成と、コード内の'DRIVE_FOLDER_PATH'が一致しているか確認してください。")
except Exception as e:
    print(f"\n予期せぬエラーが発生しました: {e}")


# ==== Cell 2 ====
# 🏥 時間帯別 採血患者数予測アプリ (ラグあり安定版)
# =================================================================
# 1. ライブラリのインストール
# =================================================================
# !pip -q install ipywidgets jpholiday scikit-learn xgboost pandas japanize-matplotlib

# =================================================================
# 2. ライブラリのインポート
# =================================================================
import pandas as pd
import numpy as np
import jpholiday
from datetime import date, timedelta, time
import json
import xgboost as xgb
import ipywidgets as widgets
from IPython.display import display, clear_output
from google.colab import drive
import matplotlib.pyplot as plt
import japanize_matplotlib

# =================================================================
# 3. Google Driveの連携
# =================================================================
try:
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"エラー: Google Driveの連携に失敗しました。{e}")

# =================================================================
# 4. アプリケーション本体
# =================================================================
try:
    print("\n--- 🏥 時間帯別 採血患者数予測アプリ ---")

    # --- 4a. ファイルパス ('_lag_stable' 版を指定) ---
    DRIVE_FOLDER_PATH = '/content/drive/MyDrive/Colab_Data/Hospital_A/'
    MODEL_PATH = DRIVE_FOLDER_PATH + 'model_A_timeseries_lag_stable.json'
    COLUMNS_PATH = DRIVE_FOLDER_PATH + 'columns_A_timeseries_lag_stable.json'

    # --- 4b. モデルとカラムリストの読み込み ---
    model = xgb.XGBRegressor(); model.load_model(MODEL_PATH)
    with open(COLUMNS_PATH, 'r') as f:
        feature_columns = json.load(f)
    print("\n✅ モデル読込完了。")

    # --- 4c. UIウィジェットの作成 ---
    date_picker = widgets.DatePicker(description='予測対象日', value=date.today() + timedelta(days=1))
    patient_input = widgets.IntText(value=1200, description='延べ外来患者数:')
    weather_input = widgets.Dropdown(options=['晴', '曇', '雨', '雪', '快晴', '薄曇'], value='晴', description='天気予報:')
    predict_button = widgets.Button(description='明日のスケジュールを予測', button_style='success', icon='chart-line')
    output_area = widgets.Output()

    # --- 4d. 予測実行ロジック ---
    def on_predict_clicked(b):
        with output_area:
            clear_output(wait=True)
            print("予測計算中...")

            target_date = pd.to_datetime(date_picker.value)
            total_patients = patient_input.value
            weather = weather_input.value

            time_slots = pd.date_range(start=target_date.replace(hour=8), end=target_date.replace(hour=18), freq='30T')
            predictions = []

            lags = {'lag_30min': 0.0, 'lag_60min': 0.0, 'lag_90min': 0.0}
            rolling_mean = 0.0

            is_holiday_daily = jpholiday.is_holiday(target_date) or target_date.weekday() >= 5 or \
                               (target_date.month == 12 and target_date.day >= 29) or (target_date.month == 1 and target_date.day <= 3)
            prev_date = target_date - timedelta(days=1)
            is_prev_holiday = jpholiday.is_holiday(prev_date) or prev_date.weekday() >= 5 or \
                              (prev_date.month == 12 and prev_date.day >= 29) or (prev_date.month == 1 and prev_date.day <= 3)

            for ts in time_slots:
                features = pd.DataFrame(columns=feature_columns); features.loc[0] = 0

                features['hour'] = ts.hour
                features['minute'] = ts.minute
                features['total_outpatient_count'] = total_patients
                features['is_holiday'] = int(is_holiday_daily)
                if '月' in features.columns: features['月'] = ts.month
                if '週回数' in features.columns: features['週回数'] = (ts.day - 1) // 7 + 1
                if '前日祝日フラグ' in features.columns: features['前日祝日フラグ'] = int(is_prev_holiday)

                if 'is_first_slot' in features.columns:
                    features['is_first_slot'] = 1 if (ts.hour == 8 and ts.minute == 0) else 0
                if 'is_second_slot' in features.columns:
                    features['is_second_slot'] = 1 if (ts.hour == 8 and ts.minute == 30) else 0

                features['雨フラグ'] = 1 if '雨' in weather else 0
                features['雪フラグ'] = 1 if '雪' in weather else 0
                weather_cat_col = f"天気カテゴリ_{weather[0]}"
                if weather_cat_col in features.columns:
                    features[weather_cat_col] = 1
                dayofweek_col = f"dayofweek_{ts.dayofweek}"
                if dayofweek_col in features.columns:
                    features[dayofweek_col] = 1

                for lag_col, lag_val in lags.items():
                    if lag_col in features.columns: features[lag_col] = lag_val
                if 'rolling_mean_60min' in features.columns: features['rolling_mean_60min'] = rolling_mean

                prediction = max(0, round(model.predict(features[feature_columns])[0]))
                predictions.append({'時間帯': ts.strftime('%H:%M'), '予測患者数': prediction})

                new_lags = lags.copy()
                new_lags['lag_90min'] = new_lags['lag_60min']
                new_lags['lag_60min'] = new_lags['lag_30min']
                new_lags['lag_30min'] = prediction
                lags = new_lags
                rolling_mean = (lags['lag_30min'] + lags['lag_60min']) / 2.0

            clear_output(wait=True)
            result_df = pd.DataFrame(predictions)

            print(f"--- {target_date.strftime('%Y-%m-%d')} の予測結果 (合計: {result_df['予測患者数'].sum()}人) ---")

            fig, ax = plt.subplots(figsize=(12, 5))
            result_df.plot(kind='bar', x='時間帯', y='予測患者数', ax=ax, legend=None, width=0.8)
            ax.set_title(f"{target_date.strftime('%Y-%m-%d')} の時間帯別 予測患者数", fontsize=16)
            ax.set_ylabel('予測患者数 (人)')
            ax.set_xlabel('時間帯')
            ax.tick_params(axis='x', rotation=45)
            for p in ax.patches:
                ax.annotate(str(int(p.get_height())), (p.get_x() * 1.005, p.get_height() * 1.005))

            plt.tight_layout(); plt.show()
            display(result_df)

    predict_button.on_click(on_predict_clicked)

    # --- 4e. UIの表示 ---
    display(date_picker, patient_input, weather_input, predict_button, output_area)

except FileNotFoundError:
    print(f"エラー: 指定されたパスにファイルが見つかりません。パスを確認してください。")
except Exception as e:
    print(f"\n予期せぬエラーが発生しました: {e}")


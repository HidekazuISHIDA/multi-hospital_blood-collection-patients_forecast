# A病院向け 学習データフォーマット仕様  
`training/a_hospital_training.py` 対応（3モデル共通）

## 0. 概要

本ドキュメントは、`training/a_hospital_training.py` 内に定義された **3つの学習スクリプト** で使用するデータフォーマットをまとめたものです。

### 対応している 3 モデル

1. **日次予測モデル**（Cell 0）  
   - 目的：**日単位の採血患者数（blood_patient_count）** を予測  
   - 出力：`model_A_daily.json`, `best_params_A_daily.json`, `model_A_daily_scaler.joblib`

2. **時間帯別（30分）受付人数モデル**（Cell 1）  
   - 目的：**30分枠ごとの受付人数（patient_count_slot）** を予測  
   - 出力：`model_A_timeseries.json`, `columns_A_timeseries.json`

3. **30分単位 待ち人数・待ち時間 同時予測モデル**（Cell 2）  
   - 目的：**30分枠ごとの平均待ち時間（avg_wait_minutes）と待ち人数（queue_size）** を予測  
   - 出力：`model_A_waittime_30min.json`, `model_A_queue_30min.json`, `columns_A_multi_30min.json`

---

## 1. 共通の入力ファイル

### 1-1. 採血ログファイル（CSV, Shift-JIS）

3つのモデルで共通して使用する元データです。

#### 必須カラム

| カラム名 | 使用モデル | 説明 |
|---------|------------|------|
| 実施日  | 日次モデル | 採血実施日（`%y/%m/%d`形式） |
| 受信    | 時間帯別モデル・待ち時間モデル | 採血受付時刻 |
| 指示書  | 待ち時間モデル | 採血室呼出時刻 |
| 処理    | 全モデル | `終了` のレコードのみ使用 |

---

### 1-2. 外来患者数ファイル（CSV, UTF-8 BOM）

#### 必須カラム

| カラム名 | 説明 |
|---------|------|
| date    | 日付 |
| total_outpatient_count | 延べ外来患者数 |

---

### 1-3. 気象データ（CSV, Shift-JIS, 気象庁形式）

使用カラム：

| カラム名 | 説明 |
|----------|------|
| date      | 日付 |
| 降水量    | 日降水量 |
| 天気概況  | 曇・晴・雨など |
| 平均気温  | 平均気温 |
| 最高気温  | 最高気温 |
| 最低気温  | 最低気温 |
| 平均湿度  | 湿度 |
| 平均風速  | 風速 |

追加特徴量：

- 雨フラグ
- 雪フラグ
- 天気カテゴリ（先頭1文字を抽出）

---

## 2. Cell 0：日次予測モデル

### 2-1. カレンダー特徴量

自動生成：

| カラム名 | 内容 |
|----------|------|
| date | 2024〜2026年の全日付 |
| is_holiday | 祝日・土日・年末年始 |
| 月 | 月番号 |
| 曜日 | 日本語（例：「火」） |
| 週回数 | その月の第何週 |
| 前日祝日フラグ | 前日が祝日 |

### 2-2. 採血患者数（日次ラベル）

```
blood_patient_count = log_df.groupby('実施日').size()
```

### 2-3. フル日次データの構成

- カレンダー特徴量
- 外来患者数
- 採血患者数
- 気象量＋雨雪フラグ＋天気カテゴリダミー

### 2-4. 特徴量・目的変数

| 目的変数 | 説明 |
|----------|------|
| blood_patient_count | その日の採血患者数 |

除外される特徴量：

- is_holiday（使用しない）
- 実施日
- date（説明変数から除外）

### 2-5. 出力ファイル

| ファイル名 | 内容 |
|------------|------|
| model_A_daily.json | 日次 XGBoost モデル |
| best_params_A_daily.json | 最適パラメータ |
| model_A_daily_scaler.joblib | StandardScaler |

---

## 3. Cell 1：時間帯別（30分）受付人数モデル

### 3-1. 30分集計

```
patient_count_slot = log_df['受信'].resample('30T').size()
```

08:00〜18:00 の範囲に限定。

### 3-2. 時刻・日付特徴量

| カラム名 | 内容 |
|----------|------|
| hour | 時 |
| minute | 分 |
| dayofweek | 曜日 |
| date | 日付 |
| is_first_slot | その日の最初の枠（8:00） |
| is_second_slot | 8:30 の枠 |

### 3-3. 日次特徴量のマージ

- is_holiday_daily → is_holiday
- 月
- 週回数
- 前日祝日フラグ
- total_outpatient_count
- 気象量
- 天気カテゴリダミー

### 3-4. ラグ特徴量

| カラム | 内容 |
|--------|------|
| lag_30min | 1枠前の受付数 |
| lag_60min | 2枠前 |
| lag_90min | 3枠前 |
| rolling_mean_60min | 過去2枠の平均 |

### 3-5. 目的変数・説明変数

| 名称 | 説明 |
|------|------|
| patient_count_slot | 30分枠の受付人数 |

説明変数は上記以外すべて。

### 3-6. 出力ファイル

| ファイル名 | 内容 |
|------------|------|
| model_A_timeseries.json | 時系列30分受付数モデル |
| columns_A_timeseries.json | 説明変数リスト |

---

## 4. Cell 2：30分単位 待ち人数・待ち時間 同時予測モデル

### 4-1. 待ち時間の計算

```
wait_minutes = (call_time − reception_time) / 60
```

0〜180分の範囲のみ採用。

### 4-2. 集計（30分）

| カラム | 内容 |
|--------|------|
| avg_wait_minutes | 平均待ち時間 |
| reception_count | 受付数 |
| call_count | 呼出数 |

### 4-3. 時刻・日付特徴量

- hour
- minute
- dayofweek（ダミー化）
- date

### 4-4. 日次特徴量のマージ

Cell 1 と同じ：

- is_holiday
- 月, 週回数, 前日祝日フラグ
- total_outpatient_count
- 気象量（平均気温等）
- 雨フラグ・雪フラグ
- 天気カテゴリのダミー

### 4-5. queue（待ち人数）関連特徴量

| カラム名 | 内容 |
|----------|------|
| net_flow | reception_count − call_count |
| queue_size | 枠終了時点の累積待ち人数 |
| queue_at_start_of_slot | 枠開始時点の待ち人数 |

### 4-6. 目的変数と説明変数

| 目的変数 | 内容 |
|----------|------|
| avg_wait_minutes | 平均待ち時間 |
| queue_size | 待ち人数 |

除外されるカラム：

- call_count
- net_flow
- date

### 4-7. 出力ファイル

| ファイル名 | 内容 |
|------------|------|
| model_A_waittime_30min.json | 待ち時間モデル |
| model_A_queue_30min.json | 待ち人数モデル |
| columns_A_multi_30min.json | 説明変数一覧 |

---

## 5. 更新時の注意

- 本仕様は `a_hospital_training.py` の 3つの Cell 構造に基づく  
- 新たな特徴量の追加・ファイル形式の変更・学習期間の変更時は **必ず本 md も更新すること**


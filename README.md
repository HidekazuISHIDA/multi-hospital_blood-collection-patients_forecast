# multi-hospital_blood-collection-patients_forecast
多施設（A病院・B病院）における採血患者数を予測する機械学習モデルの再現性確保を目的としたリポジトリです。

本研究では、採血室における以下の予測を対象としています：

- 日次採血患者数予測  
- 5営業日先予測  
- 30分間隔の混雑度（時間帯別患者数）予測  
- A病院モデルを基盤とした B病院データへのファインチューニング  
- B病院における待ち人数・待ち時間の推定  

論文投稿のための補助資料として、モデル構築手順の再現を可能にする構成となっています。

---

## 📂 リポジトリ構成

```
multi-hospital_blood-collection-patients_forecast/
│
├── README.md（このファイル）
├── LICENSE
├── requirements.txt
│
├── notebooks/                # 元のGoogle Colabノートブック
│   ├── A_hospital_training.ipynb
│   ├── B_hospital_finetuning.ipynb
│   ├── A_hospital_prediction_app.ipynb
│   └── B_hospital_prediction_app.ipynb
│
├── training/                 # A病院のモデル学習スクリプト
│   ├── a_hospital_training.py
│   └── a_hospital_training.md
│
├── finetuning/              # B病院のファインチューニングスクリプト
│   ├── b_hospital_finetuning.py
│   └── b_hospital_finetuning.md
│
├── prediction_app/          # 推論（予測）用アプリ
│   ├── a_hospital_prediction_app.py
│   ├── b_hospital_prediction_app.py
│   ├── a_hospital_prediction_app.md
│   └── b_hospital_prediction_app.md
│
└── docs/                    # 補足資料
    ├── data_format.md
    ├── model_specifications.md
    ├── usage_examples.md
    └── citation.md
```

---

## 🧠 研究概要

本研究では、採血室における採血患者数の変動を予測するために、以下の特徴量を用いて機械学習モデルを構築しました。

### 使用データ
- 採血ログ（患者数集計）
- 外来予約患者数
- 気象データ（気温・降水量など）
- 日本の祝日情報
- 曜日・月・季節特徴量
- 時系列ラグ特徴量（時間帯予測モデル）

A病院を学習データとしてモデルを開発し、B病院データで外部検証・ファインチューニングを行いました。

---

## 🔧 セットアップ方法

Python 3.10 以上を推奨します。

### インストール
```
pip install -r requirements.txt
```

Google Colab でも実行可能です。

---

## 📊 データについて

本リポジトリには、個人情報保護のため実データは含まれません。  
データ形式は以下にまとめています：

➡ `docs/data_format.md`

これに合わせた CSV を準備すれば、モデル構築と推論が再現できます。

---

## 🧪 ノートブック

`notebooks/` フォルダには、研究で使用した元の Google Colab ノートブックを保存しています：

- A病院モデル学習
- B病院モデルファインチューニング
- A/B病院の予測アプリ UI（インタラクティブ表示）

---

## 🧪 スクリプト（再現性のための Python 版）

### ▼ 学習（A病院）
`training/a_hospital_training.py`  
A病院の日次予測モデルを最初から構築可能。

### ▼ ファインチューニング（B病院）
`finetuning/b_hospital_finetuning.py`  
A病院モデルを B病院向けに調整。

### ▼ 予測アプリ（推論）
`prediction_app/`  
- 日次予測  
- 5日予測  
- 30分間隔予測  
- B病院の待ち時間推定  

すべて論文中の分析で使用したものです。

---

## 📄 補足ドキュメント

- `docs/data_format.md`：データ形式（CSVの列名と仕様）
- `docs/model_specifications.md`：特徴量やモデル構造のまとめ
- `docs/usage_examples.md`：再現手順の例
- `docs/citation.md`：論文引用フォーマット

---

## 📜 ライセンス

MIT License  
必要に応じて変更可能です。

---

## 🙌 謝辞

A病院・B病院のスタッフの皆様、研究協力に深く感謝いたします。

本リポジトリは、論文における再現性確保および医療現場の業務効率化に資することを目的としています。

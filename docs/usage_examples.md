# 使用例（日本語版）

## A病院モデルの学習
A病院のデータを用いて日次予測モデルを構築します：

```
python training/a_hospital_training.py
```

---

## B病院モデルのファインチューニング
A病院モデルを初期モデルとして、B病院データで再学習します：

```
python finetuning/b_hospital_finetuning.py
```

---

## 予測アプリによる推論
日次・5日先・時間帯別・待ち時間予測を実行できます：

```
python prediction_app/a_hospital_prediction_app.py
python prediction_app/b_hospital_prediction_app.py
```

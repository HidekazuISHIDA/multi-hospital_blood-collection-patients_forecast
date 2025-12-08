# データ形式仕様書（日本語版）

## 採血ログ（Blood Collection Logs）
| 列名 | 説明 |
|------|------|
| date | YYYY-MM-DD（採血日） |
| time | HH:MM（採血時間） |
| patient_count | 各時間帯における採血患者数 |

---

## 外来患者数データ
| 列名 | 説明 |
|------|------|
| date | YYYY-MM-DD |
| outpatient_total | 当日の外来患者数合計（予約患者数） |

---

## 気象データ
| 列名 | 説明 |
|------|------|
| date | YYYY-MM-DD |
| temperature | 気温（℃） |
| precipitation | 降水量（mm） |

研究では、A/B 病院ともにこれらの形式に統一した前処理済みデータを使用しています。

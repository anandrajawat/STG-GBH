# Benchmark Datasets for STG-GBH

This directory hosts the five cybersecurity benchmark datasets used to train, validate, and evaluate the **STG-GBH** architecture. 

To prevent repository bloat, raw telemetry datasets (which total over 2.7 GB in compressed form and 22+ million flows) are excluded from Git commits via `.gitignore`. The complete statistical and schema audit for all downloaded datasets is documented in [`reports/dataset_audit.md`](../reports/dataset_audit.md).

---

## 1. Primary In-Domain Benchmark: CSE-CIC-IDS2018

- **Role:** Primary training, validation, in-domain testing, and 15-factor component ablation.
- **Provider:** Communications Security Establishment (CSE) & Canadian Institute for Cybersecurity (CIC).
- **Scale:** 10 capture days, 6,653,422 network flows across 80 feature columns.
- **Official URL:** [UNB CIC IDS-2018](https://www.unb.ca/cic/datasets/ids-2018.html)
- **Kaggle Mirror:** [CSE-CIC-IDS2018 on Kaggle](https://www.kaggle.com/dsv/4059899)
- **Expected Directory:** `datasets/CSE_CIC_IDS2018/`
  - `DDoS1-Tuesday-20-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DDoS2-Wednesday-21-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DoS1-Thursday-15-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `DoS2-Friday-16-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Bruteforce-Wednesday-14-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Botnet-Friday-02-03-2018_TrafficForML_CICFlowMeter.parquet`
  - `Infil1-Wednesday-28-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Infil2-Thursday-01-03-2018_TrafficForML_CICFlowMeter.parquet`
  - `Web1-Thursday-22-02-2018_TrafficForML_CICFlowMeter.parquet`
  - `Web2-Friday-23-02-2018_TrafficForML_CICFlowMeter.parquet`

---

## 2. Secondary Cross-Dataset Benchmark: CIC-IDS2017

- **Role:** Cross-dataset evaluation, zero-shot generalization, and transfer learning validation.
- **Provider:** Canadian Institute for Cybersecurity (CIC).
- **Scale:** 8 capture partitions, 2,308,007 flows across 79 features.
- **Official URL:** [UNB CIC IDS-2017](https://www.unb.ca/cic/datasets/ids-2017.html)
- **Kaggle Mirror:** [CIC-IDS2017 on Kaggle](https://www.kaggle.com/dsv/4059877)
- **Expected Directory:** `datasets/CIC_IDS2017/`
  - `Benign-Monday-no-metadata.parquet`
  - `Bruteforce-Tuesday-no-metadata.parquet`
  - `DoS-Wednesday-no-metadata.parquet`
  - `WebAttacks-Thursday-no-metadata.parquet`
  - `Infiltration-Thursday-no-metadata.parquet`
  - `Botnet-Friday-no-metadata.parquet`
  - `Portscan-Friday-no-metadata.parquet`
  - `DDoS-Friday-no-metadata.parquet`

---

## 3. Industrial IoT Domain: Edge-IIoTset

- **Role:** Spatio-temporal evaluation on modern industrial cyber-physical protocols (Modbus, MQTT, HTTP).
- **Provider:** IEEE Communications Magazine / Ferrag et al.
- **Scale:** 157,800 flows across 61 features.
- **Kaggle URL:** [Edge-IIoTset on Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
- **Expected Directory:** `datasets/Edge_IIoTset/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv`

---

## 4. Large-Scale IoT Intrusion Benchmark: CICIoT2023

- **Role:** Large-scale IoT intrusion detection with 33 modern attack categories.
- **Provider:** Canadian Institute for Cybersecurity.
- **Scale:** 33 attack classes, 46 network features.
- **Expected Directory:** `datasets/CICIoT2023/wataiData/csv/CICIoT2023/` (CSV shards `part-00000` to `part-00004`).

---

## 5. Supplementary NetFlow v2 Benchmark: NF-ToN-IoT-V2

- **Role:** NetFlow v2 format cross-domain validation.
- **Provider:** University of New South Wales (UNSW) / Sarhan et al.
- **Scale:** 13,101,666 NetFlow records across 43 features.
- **Expected Directory:** `datasets/ToN_IoT/NF-ToN-IoT-V2.parquet`

---

## Data Download & Audit Verification

To automatically audit downloaded files, verify hash duplicates, and inspect schema compatibility, run:

```bash
# Verify dataset provenance and generate reports/dataset_audit.md
python3 scripts/build_audit_report.py

# Verify 14-safeguard data leakage audit
python3 scripts/run_leakage_audit.py
```

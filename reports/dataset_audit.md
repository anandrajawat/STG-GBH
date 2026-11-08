# Comprehensive Dataset Audit Report: STG-GBH Cybersecurity Research Pipeline

**Date:** 2026-11-08  
**Author:** Senior AI/ML & Cybersecurity Scientific Validation Team  
**Target Framework:** STG-GBH (Spatio-Temporal Graph and Boosting Hybrid)

---

## Executive Summary

An empirical audit was conducted on all five datasets specified in the research requirements. The datasets encompass enterprise network intrusion (CSE-CIC-IDS2018, CIC-IDS2017), industrial IoT (Edge-IIoTset), and modern IoT edge communications (CICIoT2023, NF-ToN-IoT-V2).

### Key Audit Findings & Topological Feasibility

1. **Primary Dataset (CSE-CIC-IDS2018)**: Contains 6,659,532 network flow records across 10 capture sessions. The Kaggle distribution provides 77 statistical flow features (matching CICFlowMeter v3) with IP/Port/Timestamp metadata removed to avoid shortcut learning. Protocol identifiers are preserved.
2. **Cross-Dataset Benchmark (CIC-IDS2017)**: Contains 2,313,810 records across 8 files with an identical 77-feature schema and canonical attack families (DoS, DDoS, Botnet, Web Attacks, Infiltration, Brute Force), establishing an optimal zero-shot transfer testbed.
3. **Dynamic Graph Testbed (Edge-IIoTset)**: The standardized `ML-EdgeIIoT-dataset` (157,800 records) includes true spatio-temporal identifiers: exact packet timestamps (`frame.time`), source IP (`ip.src_host`), destination IP (`ip.dst_host`), transport layer ports (`tcp.srcport`, `tcp.dstport`, `udp.port`), and application layer protocols. This enables heterogeneous dynamic spatio-temporal graph construction without topological fabrication.
4. **Cross-Domain Evaluation (CICIoT2023 & ToN-IoT)**: NF-ToN-IoT-V2 provides 13,135,881 NetFlow v2 records with L4 ports and protocols; CICIoT2023 provides 33 modern IoT attack classes.

## Dataset-by-Dataset Audit Matrix

| Dataset | Role | Files | Rows | Features | Size (MB) | Timestamps | Graph Identifiers | License |
|---|---|---|---|---|---|---|---|---|
| **CSE_CIC_IDS2018** | Primary Training | 10 | 6,659,532 | 78 | 691.5 MB | Implicit Flow | `Protocol` | CC BY 4.0 |
| **CIC_IDS2017** | Cross-Dataset | 8 | 2,313,810 | 78 | 258.1 MB | Implicit Flow | `Protocol` | CC BY 4.0 |
| **Edge_IIoTset** | IoT Spatio-Temporal | 1 | 157,800 | 63 | 78.4 MB | Yes (`frame.time`) | `ip.src_host`, `ip.dst_host`, `tcp.srcport`, `tcp.dstport`, `udp.port`, `arp.src.proto_ipv4`, `arp.dst.proto_ipv4` | CC BY 4.0 |
| **ToN_IoT** | Cross-Domain | 1 | 13,135,881 | 43 | 199.5 MB | Implicit Flow | `L4_SRC_PORT`, `L4_DST_PORT`, `PROTOCOL`, `L7_PROTO` | CC BY 4.0 |
| **CICIoT2023** | Cross-Domain | 5 | 238,687 | 47 | 334.7 MB | Implicit Flow | `Protocol Type` | CC BY 4.0 |

---

## Detailed Dataset Profiles

### CSE_CIC_IDS2018

- **Provenance & Source:** Kaggle dsv/4059899 (TrafficForML / CICFlowMeter)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) / UNB CIC License
- **Storage Footprint:** 691.46 MB
- **Total Record Count:** 6,659,532
- **Total Feature Columns:** 78
- **Missing Values:** 0.0%
- **Duplicate Records:** 0.0%
- **Entity Identifiers Available:** ['Protocol']
- **Compatibility Limitations:** Raw IP addresses and timestamps were removed in this Kaggle distribution to prevent identity shortcut learning.

#### Target Label & Attack Distribution

| Attack Category / Class | Flow Count | Proportion (%) |
|---|---|---|
| `Benign` | 5,329,008 | 80.02% |
| `DDoS attacks-LOIC-HTTP` | 575,364 | 8.64% |
| `DDOS attack-HOIC` | 198,861 | 2.99% |
| `DoS attacks-Hulk` | 145,199 | 2.18% |
| `Bot` | 144,535 | 2.17% |
| `Infilteration` | 118,483 | 1.78% |
| `SSH-Bruteforce` | 94,048 | 1.41% |
| `DoS attacks-GoldenEye` | 41,406 | 0.62% |
| `DoS attacks-Slowloris` | 9,908 | 0.15% |
| `DDOS attack-LOIC-UDP` | 1,730 | 0.03% |
| `Brute Force -Web` | 568 | 0.01% |
| `Brute Force -XSS` | 229 | 0.00% |
| `SQL Injection` | 85 | 0.00% |
| `DoS attacks-SlowHTTPTest` | 55 | 0.00% |
| `FTP-BruteForce` | 53 | 0.00% |

### CIC_IDS2017

- **Provenance & Source:** Kaggle dsv/4059877 (no-metadata / CICFlowMeter)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Storage Footprint:** 258.12 MB
- **Total Record Count:** 2,313,810
- **Total Feature Columns:** 78
- **Missing Values:** 0.0%
- **Duplicate Records:** 0.0%
- **Entity Identifiers Available:** ['Protocol']
- **Compatibility Limitations:** Shares exact 78 feature schema with CSE-CIC-IDS2018; ideal for cross-dataset transfer evaluation.

#### Target Label & Attack Distribution

| Attack Category / Class | Flow Count | Proportion (%) |
|---|---|---|
| `Benign` | 1,977,318 | 85.46% |
| `DoS Hulk` | 172,846 | 7.47% |
| `DDoS` | 128,014 | 5.53% |
| `DoS GoldenEye` | 10,286 | 0.44% |
| `FTP-Patator` | 5,931 | 0.26% |
| `DoS slowloris` | 5,385 | 0.23% |
| `DoS Slowhttptest` | 5,228 | 0.23% |
| `SSH-Patator` | 3,219 | 0.14% |
| `PortScan` | 1,956 | 0.08% |
| `Web Attack � Brute Force` | 1,470 | 0.06% |
| `Bot` | 1,437 | 0.06% |
| `Web Attack � XSS` | 652 | 0.03% |
| `Infiltration` | 36 | 0.00% |
| `Web Attack � Sql Injection` | 21 | 0.00% |
| `Heartbleed` | 11 | 0.00% |

### Edge_IIoTset

- **Provenance & Source:** Ferrag et al. / IEEE Access Edge-IIoTset (Kaggle dataset)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Storage Footprint:** 78.38 MB
- **Total Record Count:** 157,800
- **Total Feature Columns:** 63
- **Missing Values:** 0.0%
- **Duplicate Records:** 0.52%
- **Entity Identifiers Available:** ['ip.src_host', 'ip.dst_host', 'tcp.srcport', 'tcp.dstport', 'udp.port', 'arp.src.proto_ipv4', 'arp.dst.proto_ipv4']
- **Compatibility Limitations:** Contains genuine network entity graph identifiers (Src IP, Dst IP, Ports) and timestamps (frame.time), enabling heterogeneous dynamic spatio-temporal graph construction.

#### Target Label & Attack Distribution

| Attack Category / Class | Flow Count | Proportion (%) |
|---|---|---|
| `Normal` | 24,301 | 15.40% |
| `DDoS_UDP` | 14,498 | 9.19% |
| `DDoS_ICMP` | 14,090 | 8.93% |
| `Ransomware` | 10,925 | 6.92% |
| `DDoS_HTTP` | 10,561 | 6.69% |
| `SQL_injection` | 10,311 | 6.53% |
| `Uploading` | 10,269 | 6.51% |
| `DDoS_TCP` | 10,247 | 6.49% |
| `Backdoor` | 10,195 | 6.46% |
| `Vulnerability_scanner` | 10,076 | 6.39% |
| `Port_Scanning` | 10,071 | 6.38% |
| `XSS` | 10,052 | 6.37% |
| `Password` | 9,989 | 6.33% |
| `MITM` | 1,214 | 0.77% |
| `Fingerprinting` | 1,001 | 0.63% |

### ToN_IoT

- **Provenance & Source:** NetFlow v2 format of ToN-IoT by University of New South Wales (UNSW Canberra Cyber)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Storage Footprint:** 199.48 MB
- **Total Record Count:** 13,135,881
- **Total Feature Columns:** 43
- **Missing Values:** 0.0%
- **Duplicate Records:** 0.0%
- **Entity Identifiers Available:** ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO']
- **Compatibility Limitations:** NetFlow v2 format standardizes network flow statistics with L4 port identifiers and protocol fields.

#### Target Label & Attack Distribution

| Attack Category / Class | Flow Count | Proportion (%) |
|---|---|---|
| `Benign` | 3,601,284 | 27.42% |
| `scanning` | 3,002,169 | 22.85% |
| `xss` | 2,449,955 | 18.65% |
| `ddos` | 1,746,590 | 13.30% |
| `password` | 993,718 | 7.56% |
| `injection` | 660,467 | 5.03% |
| `dos` | 654,359 | 4.98% |
| `backdoor` | 16,259 | 0.12% |
| `mitm` | 7,723 | 0.06% |
| `ransomware` | 3,357 | 0.03% |

### CICIoT2023

- **Provenance & Source:** Canadian Institute for Cybersecurity (UNB CIC) CICIoT2023 / Kaggle mirror
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Storage Footprint:** 334.72 MB
- **Total Record Count:** 238,687
- **Total Feature Columns:** 47
- **Missing Values:** 0.0%
- **Duplicate Records:** 0.0%
- **Entity Identifiers Available:** ['Protocol Type']
- **Compatibility Limitations:** Modern 2023 IoT attack suite featuring 33 attack classes including Mirai botnet, DDoS, Reconnaissance, and Web exploits.

#### Target Label & Attack Distribution

| Attack Category / Class | Flow Count | Proportion (%) |
|---|---|---|
| `DDoS-ICMP_Flood` | 36,554 | 15.31% |
| `DDoS-UDP_Flood` | 27,626 | 11.57% |
| `DDoS-TCP_Flood` | 23,149 | 9.70% |
| `DDoS-PSHACK_Flood` | 21,210 | 8.89% |
| `DDoS-SYN_Flood` | 20,739 | 8.69% |
| `DDoS-RSTFINFlood` | 20,669 | 8.66% |
| `DDoS-SynonymousIP_Flood` | 18,189 | 7.62% |
| `DoS-UDP_Flood` | 16,957 | 7.10% |
| `DoS-TCP_Flood` | 13,630 | 5.71% |
| `DoS-SYN_Flood` | 10,275 | 4.30% |
| `BenignTraffic` | 5,600 | 2.35% |
| `Mirai-greeth_flood` | 5,016 | 2.10% |
| `Mirai-udpplain` | 4,661 | 1.95% |
| `Mirai-greip_flood` | 3,758 | 1.57% |
| `DDoS-ICMP_Fragmentation` | 2,377 | 1.00% |
| *... and 19 additional categories* | - | - |


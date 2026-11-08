"""
Canonical label mappings and MITRE ATT&CK taxonomy alignment for STG-GBH.
Maintains strict separation between ground-truth annotations and predictive features.
"""

from typing import Dict, Tuple, Optional

# Canonical Attack Families
CANONICAL_FAMILIES = [
    "BENIGN",
    "DOS",
    "DDOS",
    "BRUTE_FORCE",
    "BOTNET",
    "INFILTRATION",
    "WEB_ATTACK",
    "RECONNAISSANCE",
    "MITM",
    "RANSOMWARE",
    "OTHER"
]

CANONICAL_TO_ID = {fam: idx for idx, fam in enumerate(CANONICAL_FAMILIES)}
ID_TO_CANONICAL = {idx: fam for fam, idx in CANONICAL_TO_ID.items()}

# MITRE ATT&CK Tactical Stages (Evidence-based only; no fabricated labels)
MITRE_STAGES = [
    "BENIGN",
    "RECONNAISSANCE",
    "INITIAL_ACCESS",
    "EXPLOITATION",
    "COMMAND_AND_CONTROL",
    "IMPACT"
]
MITRE_TO_ID = {st: idx for idx, st in enumerate(MITRE_STAGES)}
ID_TO_MITRE = {idx: st for st, idx in MITRE_TO_ID.items()}

# Mapping for CSE-CIC-IDS2018
CSE_2018_MAPPING: Dict[str, Tuple[str, str]] = {
    "Benign": ("BENIGN", "BENIGN"),
    "Bot": ("BOTNET", "COMMAND_AND_CONTROL"),
    "SSH-Bruteforce": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "FTP-BruteForce": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "DDoS attacks-LOIC-HTTP": ("DDOS", "IMPACT"),
    "DDOS attack-HOIC": ("DDOS", "IMPACT"),
    "DDOS attack-LOIC-UDP": ("DDOS", "IMPACT"),
    "DoS attacks-GoldenEye": ("DOS", "IMPACT"),
    "DoS attacks-Slowloris": ("DOS", "IMPACT"),
    "DoS attacks-Hulk": ("DOS", "IMPACT"),
    "DoS attacks-SlowHTTPTest": ("DOS", "IMPACT"),
    "Infilteration": ("INFILTRATION", "INITIAL_ACCESS"),
    "Brute Force -Web": ("WEB_ATTACK", "INITIAL_ACCESS"),
    "Brute Force -XSS": ("WEB_ATTACK", "EXPLOITATION"),
    "SQL Injection": ("WEB_ATTACK", "EXPLOITATION"),
}

# Mapping for CIC-IDS2017
CIC_2017_MAPPING: Dict[str, Tuple[str, str]] = {
    "Benign": ("BENIGN", "BENIGN"),
    "Bot": ("BOTNET", "COMMAND_AND_CONTROL"),
    "FTP-Patator": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "SSH-Patator": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "DDoS": ("DDOS", "IMPACT"),
    "DoS Hulk": ("DOS", "IMPACT"),
    "DoS GoldenEye": ("DOS", "IMPACT"),
    "DoS slowloris": ("DOS", "IMPACT"),
    "DoS Slowhttptest": ("DOS", "IMPACT"),
    "Heartbleed": ("WEB_ATTACK", "EXPLOITATION"),
    "Infiltration": ("INFILTRATION", "INITIAL_ACCESS"),
    "PortScan": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "Web Attack  Brute Force": ("WEB_ATTACK", "INITIAL_ACCESS"),
    "Web Attack  XSS": ("WEB_ATTACK", "EXPLOITATION"),
    "Web Attack  Sql Injection": ("WEB_ATTACK", "EXPLOITATION"),
}

# Mapping for Edge-IIoTset
EDGE_IIOT_MAPPING: Dict[str, Tuple[str, str]] = {
    "Normal": ("BENIGN", "BENIGN"),
    "DDoS_UDP": ("DDOS", "IMPACT"),
    "DDoS_ICMP": ("DDOS", "IMPACT"),
    "DDoS_HTTP": ("DDOS", "IMPACT"),
    "DDoS_TCP": ("DDOS", "IMPACT"),
    "Ransomware": ("RANSOMWARE", "IMPACT"),
    "SQL_injection": ("WEB_ATTACK", "EXPLOITATION"),
    "Uploading": ("WEB_ATTACK", "EXPLOITATION"),
    "Backdoor": ("INFILTRATION", "COMMAND_AND_CONTROL"),
    "Vulnerability_scanner": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "Port_Scanning": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "Fingerprinting": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "XSS": ("WEB_ATTACK", "EXPLOITATION"),
    "Password": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "MITM": ("MITM", "INITIAL_ACCESS"),
}

# Mapping for ToN-IoT
TON_IOT_MAPPING: Dict[str, Tuple[str, str]] = {
    "Benign": ("BENIGN", "BENIGN"),
    "scanning": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "xss": ("WEB_ATTACK", "EXPLOITATION"),
    "ddos": ("DDOS", "IMPACT"),
    "password": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "injection": ("WEB_ATTACK", "EXPLOITATION"),
    "dos": ("DOS", "IMPACT"),
    "backdoor": ("INFILTRATION", "COMMAND_AND_CONTROL"),
    "mitm": ("MITM", "INITIAL_ACCESS"),
    "ransomware": ("RANSOMWARE", "IMPACT")
}

# Mapping for CICIoT2023
CICIOT2023_MAPPING: Dict[str, Tuple[str, str]] = {
    "BenignTraffic": ("BENIGN", "BENIGN"),
    "DDoS-ICMP_Flood": ("DDOS", "IMPACT"),
    "DDoS-UDP_Flood": ("DDOS", "IMPACT"),
    "DDoS-TCP_Flood": ("DDOS", "IMPACT"),
    "DDoS-PSHACK_Flood": ("DDOS", "IMPACT"),
    "DDoS-SYN_Flood": ("DDOS", "IMPACT"),
    "DDoS-RSTFINFlood": ("DDOS", "IMPACT"),
    "DDoS-SynonymousIP_Flood": ("DDOS", "IMPACT"),
    "DoS-UDP_Flood": ("DOS", "IMPACT"),
    "DoS-TCP_Flood": ("DOS", "IMPACT"),
    "DoS-SYN_Flood": ("DOS", "IMPACT"),
    "DoS-HTTP_Flood": ("DOS", "IMPACT"),
    "Mirai-greeth_flood": ("BOTNET", "COMMAND_AND_CONTROL"),
    "Mirai-udpplain": ("BOTNET", "COMMAND_AND_CONTROL"),
    "Mirai-greip_flood": ("BOTNET", "COMMAND_AND_CONTROL"),
    "DDoS-ICMP_Fragmentation": ("DDOS", "IMPACT"),
    "MITM-ArpSpoofing": ("MITM", "INITIAL_ACCESS"),
    "DDoS-ACK_Fragmentation": ("DDOS", "IMPACT"),
    "DDoS-UDP_Fragmentation": ("DDOS", "IMPACT"),
    "DNS_Spoofing": ("MITM", "INITIAL_ACCESS"),
    "Recon-HostDiscovery": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "Recon-OSScan": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "Recon-PortScan": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "VulnerabilityScan": ("RECONNAISSANCE", "RECONNAISSANCE"),
    "DDoS-HTTP_Flood": ("DDOS", "IMPACT"),
    "DDoS-SlowLoris": ("DOS", "IMPACT"),
    "DictionaryBruteForce": ("BRUTE_FORCE", "INITIAL_ACCESS"),
    "SqlInjection": ("WEB_ATTACK", "EXPLOITATION"),
    "BrowserHijacking": ("WEB_ATTACK", "EXPLOITATION"),
    "CommandInjection": ("WEB_ATTACK", "EXPLOITATION"),
    "Backdoor_Malware": ("INFILTRATION", "COMMAND_AND_CONTROL"),
    "XSS": ("WEB_ATTACK", "EXPLOITATION"),
    "Uploading_Attack": ("WEB_ATTACK", "EXPLOITATION"),
    "Recon-PingSweep": ("RECONNAISSANCE", "RECONNAISSANCE")
}


def map_raw_label(raw_label: str, dataset_name: str) -> Tuple[int, str, str, str]:
    """
    Maps a raw dataset label to canonical representations:
    Returns:
        (is_attack: int, canonical_family: str, mitre_stage: str, raw_label: str)
    """
    raw_str = str(raw_label).strip()
    
    if dataset_name.upper() in ["CSE_CIC_IDS2018", "CSE-CIC-IDS2018"]:
        mapping = CSE_2018_MAPPING
    elif dataset_name.upper() in ["CIC_IDS2017", "CIC-IDS2017"]:
        mapping = CIC_2017_MAPPING
    elif dataset_name.upper() in ["EDGE_IIOTSET", "EDGE-IIOTSET"]:
        mapping = EDGE_IIOT_MAPPING
    elif dataset_name.upper() in ["TON_IOT", "TON-IOT"]:
        mapping = TON_IOT_MAPPING
    elif dataset_name.upper() in ["CICIOT2023", "CICIOT-2023"]:
        mapping = CICIOT2023_MAPPING
    else:
        mapping = {}

    if raw_str in mapping:
        fam, stage = mapping[raw_str]
    else:
        # Fallback check
        norm_raw = raw_str.lower()
        if "benign" in norm_raw or norm_raw == "0" or norm_raw == "normal":
            fam, stage = "BENIGN", "BENIGN"
        elif "ddos" in norm_raw:
            fam, stage = "DDOS", "IMPACT"
        elif "dos" in norm_raw:
            fam, stage = "DOS", "IMPACT"
        elif "bot" in norm_raw or "mirai" in norm_raw:
            fam, stage = "BOTNET", "COMMAND_AND_CONTROL"
        elif "scan" in norm_raw or "recon" in norm_raw or "fingerprint" in norm_raw:
            fam, stage = "RECONNAISSANCE", "RECONNAISSANCE"
        elif "brute" in norm_raw or "patator" in norm_raw or "password" in norm_raw:
            fam, stage = "BRUTE_FORCE", "INITIAL_ACCESS"
        elif "sql" in norm_raw or "xss" in norm_raw or "web" in norm_raw:
            fam, stage = "WEB_ATTACK", "EXPLOITATION"
        elif "infilt" in norm_raw or "backdoor" in norm_raw:
            fam, stage = "INFILTRATION", "INITIAL_ACCESS"
        elif "ransom" in norm_raw:
            fam, stage = "RANSOMWARE", "IMPACT"
        elif "mitm" in norm_raw or "spoof" in norm_raw:
            fam, stage = "MITM", "INITIAL_ACCESS"
        else:
            fam, stage = "OTHER", "BENIGN" if "benign" in norm_raw else "EXPLOITATION"

    is_attack = 0 if fam == "BENIGN" else 1
    return is_attack, fam, stage, raw_str


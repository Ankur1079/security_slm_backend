"""
Manual test cases for all 8 security labels.
Run: python test_all_labels.py
"""

from inference import SecurityEventClassifier
from schema import SecurityEvent

clf = SecurityEventClassifier(
    checkpoint_path="./checkpoints/model.pt",
    tokenizer_path="./artifacts/tokenizer.json",
)

TESTS = [

    # ─────────────────────────────────────────────
    # 1. EXFILTRATION RISK
    # Key signals: bulk action + untrusted device OR location
    # ─────────────────────────────────────────────
    {
        "label": "exfiltration_risk",
        "case": "bulk download from unmanaged device on tor",
        "event": SecurityEvent(
            actor="user:ankur", action="bulk_download",
            resource="/finance/payroll.csv",
            device_trust="unmanaged", location="tor_exit_node",
            time="02:30 UTC", prior_events=["failed_mfa x2"],
        ),
    },
    {
        "label": "exfiltration_risk",
        "case": "copy to USB from unknown device on unrecognized IP",
        "event": SecurityEvent(
            actor="user:meera", action="copy_to_usb",
            resource="/customer_data/pii_export.csv",
            device_trust="unknown", location="unrecognized_ip",
            time="23:55 UTC", prior_events=["new_device_registered"],
        ),
    },
    {
        "label": "exfiltration_risk",
        "case": "share external from managed device but unrecognized IP (trust degraded by location)",
        "event": SecurityEvent(
            actor="user:rahul", action="share_external",
            resource="/engineering/prod_secrets.env",
            device_trust="managed", location="unrecognized_ip",
            time="11:00 UTC", prior_events=[],
        ),
    },

    # ─────────────────────────────────────────────
    # 2. INSIDER THREAT
    # Key signals: bulk/destructive action + fully trusted context + high-sensitivity resource
    # ─────────────────────────────────────────────
    {
        "label": "insider_threat",
        "case": "bulk download of HR records from managed device on known IP",
        "event": SecurityEvent(
            actor="user:priya", action="bulk_download",
            resource="/hr/employee_records.db",
            device_trust="managed", location="known_ip",
            time="14:00 UTC", prior_events=[],
        ),
    },
    {
        "label": "insider_threat",
        "case": "delete on payroll from trusted corporate VPN",
        "event": SecurityEvent(
            actor="user:arjun", action="delete",
            resource="/finance/payroll.csv",
            device_trust="managed", location="vpn_corporate",
            time="09:15 UTC", prior_events=["role_change_pending_review"],
        ),
    },
    {
        "label": "insider_threat",
        "case": "copy to USB of prod secrets — trusted device/network but highly sensitive",
        "event": SecurityEvent(
            actor="svc-etl", action="copy_to_usb",
            resource="/engineering/prod_secrets.env",
            device_trust="managed", location="known_ip",
            time="16:30 UTC", prior_events=[],
        ),
    },

    # ─────────────────────────────────────────────
    # 3. CREDENTIAL STUFFING
    # Key signals: failed_mfa x2/x3 + untrusted location + low-commitment action
    # ─────────────────────────────────────────────
    {
        "label": "credential_stuffing",
        "case": "access denied retries from tor with 3 MFA failures",
        "event": SecurityEvent(
            actor="guest_042", action="access_denied_retry",
            resource="/security/audit_logs/",
            device_trust="unknown", location="tor_exit_node",
            time="03:45 UTC", prior_events=["failed_mfa x3"],
        ),
    },
    {
        "label": "credential_stuffing",
        "case": "view attempt from unrecognized IP with repeated MFA failures",
        "event": SecurityEvent(
            actor="user:ankur", action="view",
            resource="/finance/q3_report.xlsx",
            device_trust="unknown", location="unrecognized_ip",
            time="04:10 UTC", prior_events=["failed_mfa x2"],
        ),
    },

    # ─────────────────────────────────────────────
    # 4. PRIVILEGE ESCALATION
    # Key signal: permission_change action (dominant regardless of other fields)
    # ─────────────────────────────────────────────
    {
        "label": "privilege_escalation",
        "case": "permission change with no review on record",
        "event": SecurityEvent(
            actor="user:rahul", action="permission_change",
            resource="/engineering/source/auth_service/",
            device_trust="managed", location="known_ip",
            time="10:00 UTC", prior_events=[],
        ),
    },
    {
        "label": "privilege_escalation",
        "case": "permission change from unmanaged device on unrecognized IP (escalated risk)",
        "event": SecurityEvent(
            actor="user:meera", action="permission_change",
            resource="/hr/employee_records.db",
            device_trust="unmanaged", location="unrecognized_ip",
            time="01:20 UTC", prior_events=["password_reset"],
        ),
    },

    # ─────────────────────────────────────────────
    # 5. ANOMALOUS ACCESS
    # Key signals: new_country location OR impossible_travel_flag in prior events
    # ─────────────────────────────────────────────
    {
        "label": "anomalous_access",
        "case": "view from new country (geographic anomaly)",
        "event": SecurityEvent(
            actor="user:priya", action="view",
            resource="/shared/team_notes.md",
            device_trust="managed", location="new_country",
            time="08:00 UTC", prior_events=[],
        ),
    },
    {
        "label": "anomalous_access",
        "case": "download with impossible travel flag even from corporate VPN",
        "event": SecurityEvent(
            actor="user:arjun", action="download",
            resource="/finance/q3_report.xlsx",
            device_trust="managed", location="vpn_corporate",
            time="12:00 UTC", prior_events=["impossible_travel_flag"],
        ),
    },

    # ─────────────────────────────────────────────
    # 6. MALWARE INDICATOR
    # Key signals: file tampering + unmanaged device + new_device_registered
    # ─────────────────────────────────────────────
    {
        "label": "malware_indicator",
        "case": "upload from new unmanaged device (staging behavior)",
        "event": SecurityEvent(
            actor="user:ankur", action="upload",
            resource="/engineering/repo_backup.tar",
            device_trust="unmanaged", location="unrecognized_ip",
            time="00:30 UTC", prior_events=["new_device_registered"],
        ),
    },
    {
        "label": "malware_indicator",
        "case": "rename on source code from new unmanaged device",
        "event": SecurityEvent(
            actor="svc-backup", action="rename",
            resource="/engineering/source/auth_service/",
            device_trust="unmanaged", location="known_ip",
            time="05:00 UTC", prior_events=["new_device_registered"],
        ),
    },

    # ─────────────────────────────────────────────
    # 7. POLICY VIOLATION
    # Key signals: export action (share/print/copy_to_usb) on a RESTRICTED resource
    # ─────────────────────────────────────────────
    {
        "label": "policy_violation",
        "case": "print NDA from managed device (policy violation regardless of trust)",
        "event": SecurityEvent(
            actor="user:priya", action="print",
            resource="/legal/contracts/nda_2026.pdf",
            device_trust="managed", location="known_ip",
            time="13:00 UTC", prior_events=[],
        ),
    },
    {
        "label": "policy_violation",
        "case": "share external of audit logs from corporate VPN",
        "event": SecurityEvent(
            actor="user:rahul", action="share_external",
            resource="/security/audit_logs/",
            device_trust="managed", location="vpn_corporate",
            time="15:30 UTC", prior_events=[],
        ),
    },

    # ─────────────────────────────────────────────
    # 8. BENIGN UNUSUAL
    # Key signal: no threat rule matched — low risk
    # ─────────────────────────────────────────────
    {
        "label": "benign_unusual",
        "case": "normal view from managed device on known IP",
        "event": SecurityEvent(
            actor="user:ankur", action="view",
            resource="/shared/team_notes.md",
            device_trust="managed", location="known_ip",
            time="10:30 UTC", prior_events=[],
        ),
    },
    {
        "label": "benign_unusual",
        "case": "download of personal photos from managed device",
        "event": SecurityEvent(
            actor="user:meera", action="download",
            resource="/personal/vacation_photos/",
            device_trust="managed", location="vpn_corporate",
            time="17:00 UTC", prior_events=[],
        ),
    },
    {
        "label": "benign_unusual",
        "case": "bulk download of non-sensitive resource from fully trusted context",
        "event": SecurityEvent(
            actor="user:priya", action="bulk_download",
            resource="/engineering/repo_backup.tar",
            device_trust="managed", location="known_ip",
            time="14:30 UTC", prior_events=[],
        ),
    },
]

# ─────────────────────────────────────────────────────────
# RUN ALL TESTS
# ─────────────────────────────────────────────────────────

print(f"\nRunning {len(TESTS)} test cases across 8 security labels\n")
print("=" * 70)

passed = 0
failed = 0
failed_cases = []

for t in TESTS:
    result = clf.classify_event(t["event"], temperature=0.1, top_k=5)
    predicted = result["classification"]
    expected = t["label"]
    correct = predicted == expected
    status = "PASS" if correct else "FAIL"

    if correct:
        passed += 1
    else:
        failed += 1
        failed_cases.append(t)

    print(f"[{status}] {t['label']}")
    print(f"       case: {t['case']}")
    if not correct:
        print(f"       expected:  {expected}")
        print(f"       got:       {predicted}")
    print(f"       explain:   {result['explanation'][:120]}...")
    print()

print("=" * 70)
print(f"Results: {passed}/{len(TESTS)} passed  |  {failed} failed")

if failed_cases:
    print(f"\nFailed cases:")
    for t in failed_cases:
        print(f"  - [{t['label']}] {t['case']}")
"""
Decision-boundary probe v2: isolates two specific gates the model needs to
learn (not just "which field matters most"):

1. Resource-sensitivity gate: bulk_download from a FULLY TRUSTED context
   should be insider_threat on a high_sensitivity resource, but
   benign_unusual on a normal resource. Same action, same trust signals --
   only the resource differs.

2. Location-conjunction gate: failed_mfa x2 should only trigger
   credential_stuffing when paired with an untrusted location. On a known_ip
   it should NOT trigger credential_stuffing (per the rule, x2 alone isn't
   enough -- it needs the untrusted-location AND).
"""

from dataclasses import replace
from inference import SecurityEventClassifier
from schema import SecurityEvent

clf = SecurityEventClassifier(
    checkpoint_path="./checkpoints/model.pt",
    tokenizer_path="./artifacts/tokenizer.json",
)

def run(event):
    return clf.classify_event(event, temperature=0.1, top_k=5)["classification"]

print("="*70)
print("TEST 1: resource-sensitivity gate (same action/trust, different resource)")
print("="*70)

trusted_bulk_base = dict(actor="user:priya", action="bulk_download",
                          device_trust="managed", location="known_ip",
                          time="14:30 UTC", prior_events=[])

high_sensitivity_resources = ["/hr/employee_records.db", "/finance/payroll.csv",
                               "/engineering/prod_secrets.env"]
normal_resources = ["/engineering/repo_backup.tar", "/shared/team_notes.md",
                     "/finance/q3_report.xlsx"]

print("\nExpected: insider_threat")
for r in high_sensitivity_resources:
    e = SecurityEvent(resource=r, **trusted_bulk_base)
    print(f"  resource={r:<35} -> {run(e)}")

print("\nExpected: benign_unusual")
for r in normal_resources:
    e = SecurityEvent(resource=r, **trusted_bulk_base)
    print(f"  resource={r:<35} -> {run(e)}")

print("\n" + "="*70)
print("TEST 2: location-conjunction gate (failed_mfa x2 needs untrusted location)")
print("="*70)

mfa_base = dict(actor="user:rahul", action="view", resource="/finance/q3_report.xlsx",
                device_trust="unmanaged", prior_events=["failed_mfa x2"], time="03:00 UTC")

print("\nExpected: credential_stuffing (untrusted locations)")
for loc in ["unrecognized_ip", "tor_exit_node", "new_country"]:
    e = SecurityEvent(location=loc, **mfa_base)
    print(f"  location={loc:<20} -> {run(e)}")

print("\nExpected: NOT credential_stuffing (trusted locations, x2 alone insufficient)")
for loc in ["known_ip", "vpn_corporate"]:
    e = SecurityEvent(location=loc, **mfa_base)
    print(f"  location={loc:<20} -> {run(e)}")
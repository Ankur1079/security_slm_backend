"""
Generates a synthetic labeled dataset of security events for the zero-trust
file locker SLM.

IMPORTANT DESIGN CHOICE (v2): fields are sampled INDEPENDENTLY of the label,
then the label is DERIVED from the fields via a priority-ordered rule engine
(classify_event below). This is the opposite of v1, which picked a label
first and then sampled fields to match it -- that let the model shortcut on
a single dominant field (action) and mostly ignore device_trust/location.
With independent sampling + a real multi-field decision rule, the model is
forced to combine several fields to predict correctly, matching how a real
zero-trust classifier should behave.

Usage:
    python generate_synthetic_data.py --n 8000 --out ../artifacts/train.jsonl
"""

import argparse
import json
import random
import sys
import os
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schema import SecurityEvent, LABELS  # noqa: E402

random.seed(42)

ACTORS = [f"user:{name}" for name in
          ["ankur", "priya", "rahul", "svc-backup", "svc-etl", "meera", "arjun", "guest_042"]]

RESOURCES = [
    "/finance/q3_report.xlsx", "/finance/payroll.csv", "/legal/contracts/nda_2026.pdf",
    "/hr/employee_records.db", "/engineering/prod_secrets.env", "/engineering/repo_backup.tar",
    "/personal/vacation_photos/", "/shared/team_notes.md", "/finance/tax_filings/",
    "/security/audit_logs/", "/customer_data/pii_export.csv", "/engineering/source/auth_service/",
]

# Resource sensitivity tiers -- used by the decision rules below, not just
# for flavor text. This gives the model a *fourth* real signal beyond
# action/device_trust/location/prior_events.
HIGH_SENSITIVITY = {
    "/hr/employee_records.db", "/finance/payroll.csv", "/engineering/prod_secrets.env",
    "/customer_data/pii_export.csv", "/engineering/source/auth_service/",
}
RESTRICTED = {
    "/legal/contracts/nda_2026.pdf", "/finance/tax_filings/", "/security/audit_logs/",
}
# everything else (vacation photos, team notes, q3 report, repo backup) = "normal"

ACTIONS = ["view", "download", "bulk_download", "delete", "rename", "share_external",
           "permission_change", "copy_to_usb", "print", "upload", "access_denied_retry"]

DEVICE_TRUST = ["managed", "unmanaged", "unknown"]
LOCATIONS = ["known_ip", "unrecognized_ip", "tor_exit_node", "vpn_corporate", "new_country"]
UNTRUSTED_LOCATIONS = {"unrecognized_ip", "tor_exit_node", "new_country"}

PRIOR_EVENT_OPTIONS = [
    [], [], [],  # weight toward "no prior events" being common
    ["failed_mfa x2"], ["failed_mfa x3"], ["new_device_registered"],
    ["password_reset"], ["off_hours_login"], ["impossible_travel_flag"],
    ["role_change_pending_review"],
    ["failed_mfa x2", "new_device_registered"],
    ["failed_mfa x3", "impossible_travel_flag"],
    ["off_hours_login", "impossible_travel_flag"],
]


def resource_sensitivity(resource: str) -> str:
    if resource in HIGH_SENSITIVITY:
        return "high_sensitivity"
    if resource in RESTRICTED:
        return "restricted"
    return "normal"


def random_time():
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d} UTC"


def sample_event() -> SecurityEvent:
    """Samples every field independently and uniformly. The label is NOT
    chosen here -- it's derived afterward by classify_event()."""
    return SecurityEvent(
        actor=random.choice(ACTORS),
        action=random.choice(ACTIONS),
        resource=random.choice(RESOURCES),
        device_trust=random.choice(DEVICE_TRUST),
        location=random.choice(LOCATIONS),
        time=random_time(),
        prior_events=list(random.choice(PRIOR_EVENT_OPTIONS)),
    )


def classify_event(event: SecurityEvent):
    """
    Priority-ordered rule engine: the FIRST matching rule wins. Each rule
    combines multiple fields, so no single field is a reliable shortcut.
    Returns (label, explanation) so the explanation can cite exactly what
    caused the decision.
    """
    prior = set(event.prior_events)
    sensitivity = resource_sensitivity(event.resource)
    untrusted_loc = event.location in UNTRUSTED_LOCATIONS
    untrusted_device = event.device_trust in ("unmanaged", "unknown")
    heavy_mfa_failure = "failed_mfa x3" in prior or (
        "failed_mfa x2" in prior and untrusted_loc
    )

    # Rule 1: credential_stuffing -- repeated auth failures + untrusted
    # network, on a low-commitment action (not yet a big data movement).
    if heavy_mfa_failure and untrusted_loc and event.action in (
        "view", "access_denied_retry", "download"
    ):
        return (
            "credential_stuffing",
            f"Repeated authentication failures ({', '.join(prior) or 'none'}) combined with "
            f"an untrusted network location ({event.location}) on a {event.action} attempt "
            f"strongly suggests automated credential-stuffing rather than legitimate user error.",
        )

    # Rule 2: privilege_escalation -- permission changes are inherently
    # high-signal regardless of other fields, but explanation cites context.
    if event.action == "permission_change":
        review_flag = "role_change_pending_review" in prior
        return (
            "privilege_escalation",
            f"A permission_change on {event.resource} was recorded for {event.actor} "
            f"(device_trust={event.device_trust}, location={event.location})"
            + (", with no linked change-management review on record" if not review_flag
               else ", though a role-change review was already pending")
            + ". Permission changes outside a verified approval workflow are a common "
            "precursor to privilege abuse and warrant verification.",
        )

    # Rule 3: exfiltration_risk -- bulk-style data movement AND (untrusted
    # device OR untrusted network). Requires BOTH the action type and a
    # trust-degradation signal -- neither alone is enough.
    if event.action in ("bulk_download", "copy_to_usb", "share_external") and (
        untrusted_device or untrusted_loc
    ):
        return (
            "exfiltration_risk",
            f"{event.actor} performed {event.action} on {event.resource} from a "
            f"{event.device_trust} device at {event.location}. The combination of bulk "
            f"data movement with a degraded trust signal (untrusted device and/or network) "
            f"matches a data-exfiltration pattern and should be reviewed immediately.",
        )

    # Rule 4: insider_threat -- same bulk-style actions, but from a FULLY
    # trusted device/network, on a high-sensitivity resource. This is what
    # forces the model to actually use device_trust/location: the identical
    # action that triggers Rule 3 above is reclassified here purely because
    # trust signals are clean but the resource is sensitive.
    if event.action in ("bulk_download", "copy_to_usb", "delete", "rename") and (
        not untrusted_device and not untrusted_loc
    ) and sensitivity == "high_sensitivity":
        return (
            "insider_threat",
            f"{event.actor}, using a managed device on a known/corporate network, performed "
            f"{event.action} on the sensitive resource {event.resource}. Because the device "
            f"and network are fully trusted, this bypasses typical network-based defenses -- "
            f"the risk here comes entirely from the sensitivity of the resource and the "
            f"nature of the action, which fits an insider-risk profile rather than an "
            f"external-attacker profile.",
        )

    # Rule 5: malware_indicator -- file tampering shortly after a new,
    # unmanaged device appears.
    if event.action in ("upload", "rename", "delete") and event.device_trust == "unmanaged" \
            and "new_device_registered" in prior:
        return (
            "malware_indicator",
            f"{event.actor} performed {event.action} on {event.resource} from a newly "
            f"registered, unmanaged device. This sequence (new unmanaged device followed "
            f"immediately by file tampering) resembles post-compromise staging behavior, "
            f"consistent with a malware or unauthorized-tooling indicator.",
        )

    # Rule 6: anomalous_access -- geographic/travel anomaly is the primary
    # signal, independent of what resource or action was involved.
    if event.location == "new_country" or "impossible_travel_flag" in prior:
        if event.location == "new_country":
            location_clause = f"from an unexpected location ({event.location})"
        else:
            location_clause = (
                f"flagged by an impossible-travel signal on the account, even though the "
                f"request itself came from {event.location}"
            )
        return (
            "anomalous_access",
            f"{event.actor} accessed {event.resource} via {event.action} {location_clause}. "
            f"The geographic/travel anomaly relative to the user's normal pattern is the "
            f"primary risk signal here, independent of the specific resource or action involved.",
        )

    # Rule 7: policy_violation -- restricted (not necessarily "sensitive")
    # resources being moved out, even from an otherwise trusted context.
    if event.action in ("share_external", "print", "copy_to_usb") and sensitivity == "restricted":
        return (
            "policy_violation",
            f"{event.actor} carried out {event.action} on the policy-restricted resource "
            f"{event.resource} from a {event.device_trust} device at {event.location}. "
            f"Even with clean trust signals, this action on a policy-restricted resource "
            f"itself constitutes a compliance violation regardless of intent.",
        )

    # Rule 8 (default): benign_unusual -- nothing above matched, so this is
    # low-risk or only mildly atypical.
    return (
        "benign_unusual",
        f"{event.actor} performed {event.action} on {event.resource} from a "
        f"{event.device_trust} device at {event.location}. No combination of trust "
        f"degradation, sensitive-resource access, or suspicious prior activity was "
        f"present, so this is assessed as low risk.",
    )


def generate_balanced_dataset(n: int, max_attempts_multiplier: int = 60):
    """Rejection-samples random events until each label hits its target
    quota (n / num_labels), so the final dataset is class-balanced despite
    labels now being an emergent property of independently sampled fields
    rather than chosen up front."""
    target_per_label = max(1, n // len(LABELS))
    buckets = defaultdict(list)
    attempts = 0
    max_attempts = n * max_attempts_multiplier

    while attempts < max_attempts and any(len(buckets[l]) < target_per_label for l in LABELS):
        attempts += 1
        event = sample_event()
        label, explanation = classify_event(event)
        if len(buckets[label]) >= target_per_label:
            continue  # this label's quota is full, skip to keep sampling efficient
        buckets[label].append({
            "actor": event.actor, "action": event.action, "resource": event.resource,
            "device_trust": event.device_trust, "location": event.location,
            "time": event.time, "prior_events": event.prior_events,
            "label": label, "explanation": explanation,
        })

    examples = [ex for label in LABELS for ex in buckets[label]]

    print("Label distribution:")
    for label in LABELS:
        print(f"  {label:<22} {len(buckets[label])}")
    underfilled = [l for l in LABELS if len(buckets[l]) < target_per_label]
    if underfilled:
        print(f"WARNING: these labels are under target ({target_per_label}) after "
              f"{attempts} sampling attempts -- their rule conditions are rare given "
              f"the current field distributions: {underfilled}")
        print("Consider adjusting PRIOR_EVENT_OPTIONS / field pools to make these "
              "conditions more common, or lower --n.")

    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8000)
    parser.add_argument("--out", type=str, default="../artifacts/train.jsonl")
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    data = generate_balanced_dataset(args.n)
    random.shuffle(data)
    n_val = int(len(data) * args.val_split)
    val, train = data[:n_val], data[n_val:]

    train_path = args.out
    val_path = args.out.replace("train.jsonl", "val.jsonl")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(val_path, "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"\nWrote {len(train)} train examples -> {train_path}")
    print(f"Wrote {len(val)} val examples -> {val_path}")


if __name__ == "__main__":
    main()

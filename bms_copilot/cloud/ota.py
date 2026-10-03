"""Signed over-the-air packages: cloud -> device model/calibration updates.

The cloud signs every package (Ed25519) over a canonical JSON encoding of its
header and payload. The device verifies before installing:

* signature against the pinned cloud public key
* payload digest (SHA-256) matches the header
* package is bound to this device (or is fleet-wide)
* not expired
* anti-rollback: version counter strictly greater than the installed one

Development keys are generated under artifacts/keys/. In production the
private key lives in an HSM / managed key vault and never leaves it.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=float).encode()


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj)).hexdigest()


def load_or_create_keys(key_dir: Path) -> tuple[Ed25519PrivateKey, Ed25519PublicKey, str]:
    key_dir.mkdir(parents=True, exist_ok=True)
    priv_path = key_dir / "cloud_signing_ed25519.pem"
    if priv_path.exists():
        priv = serialization.load_pem_private_key(priv_path.read_bytes(), password=None)
    else:
        priv = Ed25519PrivateKey.generate()
        priv_path.write_bytes(
            priv.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            )
        )
    pub = priv.public_key()
    raw = pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    (key_dir / "cloud_signing_ed25519.pub").write_bytes(
        pub.public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    )
    return priv, pub, hashlib.sha256(raw).hexdigest()[:16]


class PackageSigner:
    def __init__(self, private_key: Ed25519PrivateKey, key_id: str):
        self.key = private_key
        self.key_id = key_id

    def sign(
        self,
        kind: str,
        payload: dict,
        version: int,
        device_id: str | None = None,
        issued_at: datetime | None = None,
        ttl_days: int = 30,
    ) -> dict:
        issued_at = issued_at or datetime.now(UTC)
        header = {
            "kind": kind,
            "version": int(version),
            "device_id": device_id,
            "issued_at": issued_at.isoformat(),
            "expires_at": (issued_at + timedelta(days=ttl_days)).isoformat(),
            "payload_sha256": digest(payload),
            "key_id": self.key_id,
            "alg": "Ed25519",
        }
        sig = self.key.sign(canonical({"header": header, "payload": payload}))
        return {"header": header, "payload": payload, "signature": base64.b64encode(sig).decode()}


@dataclass
class DeviceVerifier:
    """Runs on the device (Layer 4). Holds the pinned public key and installed versions."""

    public_key: Ed25519PublicKey
    device_id: str
    installed: dict = field(default_factory=dict)  # kind -> version

    def verify(self, pkg: dict, now: datetime | None = None) -> tuple[bool, str]:
        now = now or datetime.now(UTC)
        try:
            h, payload = pkg["header"], pkg["payload"]
            self.public_key.verify(base64.b64decode(pkg["signature"]), canonical({"header": h, "payload": payload}))
        except (InvalidSignature, KeyError, ValueError):
            return False, "signature invalid"
        if digest(payload) != h["payload_sha256"]:
            return False, "payload digest mismatch"
        if h["device_id"] not in (None, self.device_id):
            return False, f"package bound to {h['device_id']}"
        if datetime.fromisoformat(h["expires_at"]) < now:
            return False, "package expired"
        if h["version"] <= self.installed.get(h["kind"], -1):
            return False, f"rollback rejected (installed v{self.installed.get(h['kind'])})"
        return True, "ok"

    def install(self, pkg: dict, now: datetime | None = None) -> tuple[bool, str]:
        ok, why = self.verify(pkg, now)
        if ok:
            self.installed[pkg["header"]["kind"]] = pkg["header"]["version"]
        return ok, why

"""Auth for the public-facing dashboard: a 404-everything gate unlocked by a
secret link.

The app sits behind a Tailscale funnel, so the public internet (and a steady drip
of scanner bots) can reach it. Rather than present a login page -- which advertises
"an app lives here" and invites probing -- the :class:`AuthGate` middleware returns
an indistinguishable **404 for every unauthenticated request**, whether it's ``/``,
``/static/app.js``, ``/api/stats`` or ``/.env``. A prober learns nothing.

The owner unlocks it by visiting one unguessable URL once per device::

    https://<host>.ts.net/<prefix>/<access_token>

That ``access_token`` is a 256-bit secret (far stronger than a typed password) and
*is* the credential. A match mints a signed, HttpOnly, 30-day cookie; from then
on the device's requests carry it and the gate lets them through. Rotating the
token (``serve --new-link``) instantly revokes every old bookmark.

Primitives here: itsdangerous ``TimestampSigner`` for the cookie (tamper-proof,
expiring) and a constant-time compare for the token. No novel crypto.
"""

from __future__ import annotations

import hmac
import secrets

from itsdangerous import BadSignature, SignatureExpired, TimestampSigner

COOKIE_NAME = "tw"
COOKIE_MAX_AGE = 30 * 24 * 3600  # 30 days (browser cookie expiry + server-side signature TTL)
_COOKIE_PAYLOAD = b"ok"


def new_token() -> str:
    """A fresh 256-bit access token (the unlock secret)."""
    return secrets.token_urlsafe(32)


def make_signer(secret: str) -> TimestampSigner:
    return TimestampSigner(secret)


def issue_cookie(signer: TimestampSigner) -> str:
    """Signed, timestamped cookie value handed out on a successful unlock."""
    return signer.sign(_COOKIE_PAYLOAD).decode("ascii")


def cookie_valid(signer: TimestampSigner, value: str,
                 max_age: int = COOKIE_MAX_AGE) -> bool:
    """True iff ``value`` is a current, untampered cookie this signer issued."""
    if not value:
        return False
    try:
        signer.unsign(value, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return False
    return True


def token_matches(provided: str, expected: str) -> bool:
    """Constant-time token compare; False on any empty input."""
    if not provided or not expected:
        return False
    return hmac.compare_digest(provided, expected)

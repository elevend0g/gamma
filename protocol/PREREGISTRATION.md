# Pre-registration freeze

Per section 9 of `gamma_protocol.md`: "Freeze predictions P1-P4, the
G-gates, and section 8.2 criteria (this document, hashed and timestamped)
before Phase 1 data collection."

This freeze covers `gamma_protocol.md` exactly as it stood at the end of
Phase 0 (infrastructure validation only — no Phase 1+ data collected, no
predictions P1-P4 tested, no G-gates evaluated, no section 8.2 criteria
touched). It is recorded now, before any Phase 1 (section 4) data
collection begins.

- **File:** `protocol/gamma_protocol.md`
- **SHA-256:** `90c547a5fcd69a02fe654b6944326c90467444bf27f918ac173d65ffe03d7486`
- **Git commit (blob-identical, publicly timestamped on GitHub):** `8ff60a3b6268aab4084aa5e946f3497193342183`, committed 2026-07-08T01:13:30+00:00
- **Freeze recorded:** 2026-07-08T01:16:59Z

Any change to the protocol after this point — including scope
deviations already known at freeze time (see `AMENDMENTS.md`) — is
tracked as a dated, hashed amendment referencing this freeze, not as an
edit to `gamma_protocol.md` itself. The original document is not to be
modified; if content changes are ever needed, they land in a new
`gamma_protocol_v2.md` with its own freeze record.

To re-verify this freeze at any point: `sha256sum protocol/gamma_protocol.md` should reproduce the hash above.

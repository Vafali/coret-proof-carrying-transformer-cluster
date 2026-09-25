# Security scan allowlist

The byte-identical scientific exports in `research_hab/*.py`, the frozen legacy
launcher `research_hab/run_coret_optimized_historical_127_v1.sh`, and immutable
JSON files in the external artifact bundle retain historical workstation paths
as provenance. They are data, not active cluster locations. The legacy launcher
is never called on the cluster. All executable cluster wrappers under `scripts/`
resolve the repository and artifact roots at runtime.

No credential, token, private key, checkpoint, result tree, or compiled cache is
committed to Git.

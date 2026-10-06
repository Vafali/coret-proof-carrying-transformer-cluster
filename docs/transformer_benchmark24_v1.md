# Preregistered Transformer benchmark: 24 historical properties

No scientific property was executed to prepare this benchmark. Membership
must not change in response to outcomes or broken artifacts. Missing/corrupt
inputs are reported as errors, never replaced.

## Frozen identities

- Manifest: `frozen/benchmark24/transformer_benchmark24_manifest_v1.json`
- Canonical manifest SHA256 (excluding its self-digest field):
  `46f5380d7bfacd0cc61da2938465d951fa3e72246b01a98223352510b331a338`
- Manifest file SHA256:
  `8d102bfb789a5d65cade0268f9fd6d0fc99613a787adb50c0131755011694eb5`
- Source population projection SHA256:
  `aea4f32053eeab2c88412e94156ac6f2ea7deb2513835942e9cc03569e0e49c8`
- Full production population: 127; selected: 24.
- Radius source: `cached_DeepT_reference.certified_lower_endpoint_binary64`.

Every selected row stores the exact radius and binary64 hex, original property
and draw identity, token position and authenticated token IDs, source hashes,
model/checkpoint/config/vocabulary identity, dataset identity, threat model,
selection rationale, and execution-source hashes. Canonical JSON uses the
existing `cluster_common.canonical` encoding. Physical and canonical file
digests are distinct by design.

## Selection algorithm

Selection reads only the explicitly whitelisted historical identity metadata
and historical DeepT radius, not CoReT outcomes, margins, runtime or logs.

1. Authenticate the scientific manifest, production manifest and DeepT cache.
2. Rank all 127 by `(historical radius, historical ordinal)`. Assign tertiles
   by `floor(3 * rank / 127)`: source strata contain 43/42/42 properties.
3. Collapse duplicate sentence/token pairs to the earliest historical ordinal
   for selection only. All 127 still enter the source-population digest.
4. Select eight rounds of low/middle/high radius strata, eight properties per
   stratum. Require 2–3 properties from each of the nine distinct sentences.
5. Prefer the least-used sentence, least-used sentence/stratum combination,
   greatest minimum normalized within-sentence token separation, then the
   least globally repeated token position. Break ties by
   `SHA256(schema + ':' + property_id)`, then historical ordinal.
6. A metadata-only discrete feasibility check permits a choice only if the
   remaining sentence/radius quotas can still be completed.

Radii are unchanged. Radius tertiles and sequence length are descriptive
difficulty proxies, not claims about current verifier difficulty. Each of the
24 sentence/token coordinates is distinct. The ten historical draws contain
nine distinct sentences; the duplicate draw is not treated as a new sentence.

| Historical sentence/draw | Source line | Length | Selected token indices |
| --- | ---: | ---: | --- |
| s000 | 504 | 20 | 01, 14 |
| s001 | 1794 | 27 | 05, 16, 25 |
| s002 | 1468 | 8 | 01, 06 |
| s003 | 2031 | 17 | 01, 08, 12 |
| s004 | 216 | 12 | 01, 04, 09 |
| s005 | 1931 | 16 | 02, 10, 14 |
| s006 | 882 | 16 | 03, 11, 13 |
| s007 | 1172 | 18 | 07, 15 |
| s009 | 2029 | 22 | 01, 09, 14 |

The manifest contains the authoritative selection order and exact per-property
radii. Selected radii range from 0.0008178710937500001 to 0.001397705078125.

## Measurement inventory and interpretation

`frozen/benchmark24/measurement_inventory_v1.json` records readiness and
measurement capabilities for every property before any execution. Portable
`runtime_inputs` are absent on this workstation; the original frozen metadata
was available and authenticated. Execution requires the unchanged authenticated
portable artifacts and source pins. No missing property is replaced.

The wrapper invokes the existing complete sound-FP64 campaign property path
once, at the frozen historical radius, without radius search. It does not
modify verifier mathematics, oracle code or solver settings.

Available: producer terminal status and sound margin, certificate/report hash
integrity, failure stage and domain operator where emitted, whole-property
runtime, final generator count, final numerical widening/maximum ratio, and
Block2-to-head reduction telemetry where emitted. GPU memory peaks generally
cover Block2-to-margin only, not the entire property. CPU RSS is a process
lifetime maximum. Missing telemetry remains null.

Not available: an integrated independent numerical checker verdict for the
complete sound-FP64 graph. Artifact integrity and optional MPFR spot checks
are not that verdict. Consequently a positive producer margin is recorded as
`producer_proof_status=CERTIFIED_MARGIN`, but the strict benchmark final status
remains `INCONCLUSIVE` with checker status `NOT_AVAILABLE`. The wrapper cannot
manufacture `CERTIFIED` from producer flags. Aggregation also reports the
producer-positive count separately. Sound domain failures are inconclusive;
infrastructure/malformed-input failures are rejected/error. Unrun members
remain `NOT_RUN`, not scientific failures.

The existing complete backend explicitly requires CUDA. No defensible CPU
campaign runtime can be estimated from existing GPU timings. CPU execution
would need separate backend work, which is outside this preparation task.

## Manual commands (not executed during preparation)

From the cluster repository with its existing authenticated `runtime_inputs`,
run only this one frozen member on one GPU; this is not a 24-property launcher:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_transformer_benchmark24_v1.py one --artifact-root runtime_inputs --property-id deept_table7_stdln3_s001_line1794_tok05 --result-root "$HOME/coret-benchmark24-v1-smoke" --device cuda:0
```

The historical candidate is exactly `0.0009637451171875001`. The command is a
manual scientific execution, not a locally executed preparation check. It may
take longer than 30 seconds. Results go to
`$HOME/coret-benchmark24-v1-smoke/records/<property_id>.json`; existing producer
artifacts live under the isolated `producer/properties/<property_id>` directory.
Validated small result records are resumable without reevaluation.

CPU-only inventory, without a scientific run:

```bash
python scripts/run_transformer_benchmark24_v1.py inventory --artifact-root runtime_inputs
```

Aggregation (denominator remains 24; missing members remain unrun):

```bash
python scripts/run_transformer_benchmark24_v1.py aggregate --result-root "$HOME/coret-benchmark24-v1-smoke" --output "$HOME/coret-benchmark24-v1-smoke/aggregate.json"
```

Output creation refuses overwrites. Aggregation rejects duplicate or foreign
results and changed radii/identities; no new member can silently enter the
frozen population.

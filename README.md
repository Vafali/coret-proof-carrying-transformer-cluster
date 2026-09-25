# CoReT proof-carrying Transformer: cluster reproduction

This repository packages the accepted **pre-fused** three-layer verifier for
the frozen 127-position SST benchmark. Scientific Python files in
`research_hab/` are byte-identical exports; portability and sharding live only
in `scripts/`.

**The fused A·V backend is not part of the main 127 benchmark.**

## Immutable identities

- DeepT revision: `16ffe4075f1f8a7c87fa2a187d8c46cfd51e07bf`
- Scientific manifest: `7e4b2fea94424e554f07272aa8a246da7cda1212be83af57bbcdae7c870ed9dd`
- Production manifest: `cd1375408818c8fb93a2f227d31229990d62034dc4997467fbc013cc1eb94ab2`
- DeepT cache: `67e80d74cf83e5f810726405a6f6f0cf9928f4dda84d7f87decf59f354c8181d`

## Setup

1. Transfer this Git repository and the external artifact bundle to a node.
2. Create the environment with `conda create --name coret-cluster --file
   conda-explicit-linux-64.txt`, or use `environment.yml` if an explicit solve
   is required.
3. Unpack the artifact bundle, then verify/import it:

   ```bash
   python scripts/import_artifacts.py --source /cluster/share/coret-cluster-artifacts-v1 --destination "$PWD/runtime_inputs"
   ```

4. Fetch the licensed upstream DeepT repository at the pinned revision:

   ```bash
   bash scripts/fetch_deept.sh
   ```

5. Run preflight on each architecture/GPU:

   ```bash
   CUDA_VISIBLE_DEVICES=0 python scripts/preflight_environment.py --artifact-root "$PWD/runtime_inputs"
   ```

The host driver CUDA version need not equal `torch.version.cuda`; the preflight
records both and requires PyTorch CUDA availability.

## Mandatory cross-architecture calibration

Before production, independently run both frozen calibration queries on one
A40 and one L40S. Compare each output with `scripts/compare_calibration.py`.
Mixed-architecture production is permitted only if all four comparisons are
`BITWISE_SCIENTIFIC_PARITY`. An outward or mismatched result does not authorize
mixing architectures.

## Shards and workers

After calibration timings exist, generate deterministic weighted shards:

```bash
python scripts/make_property_shards.py --artifact-root "$PWD/runtime_inputs" \
  --worker-weight 1.0 --worker-weight 1.0 --output shards.json
```

Weights must come from measured calibration, not theoretical GPU throughput or
certification outcomes. The cost model uses frozen sequence length squared.
The latest hash-valid completed baseline is excluded dynamically. If the
baseline ends inside a property, that partial trajectory is owned by exactly
one worker; at the current clean 49-property cutoff there is no partial
trajectory and only the 78 unfinished properties are assigned.

Each GPU requires an independent worker directory and exactly one verifier
process. Never run multiple verifier lanes on the same physical GPU:

```bash
bash scripts/run_property_shard.sh shards.json 0 0 worker_runs/worker_0 "$PWD/runtime_inputs"
```

Every query is persisted immediately by the frozen runner. Reissuing the same
worker command resumes its private state and never recomputes complete
properties.

## Verification and merge

Verify each worker privately:

```bash
python scripts/verify_worker_results.py --artifact-root "$PWD/runtime_inputs" \
  --shard-manifest shards.json --worker-id 0 --worker-dir worker_runs/worker_0
```

The merge is non-overwriting and fail-closed. Supply every worker directory:

```bash
python scripts/merge_worker_results.py --artifact-root "$PWD/runtime_inputs" \
  --shard-manifest shards.json --worker-dir 0=worker_runs/worker_0 \
  --worker-dir 1=worker_runs/worker_1 --output merged_results --require-complete
```

It verifies manifests, hashes, baseline immutability, query chains, checker and
provenance status, zero fallback, unique ownership, all 127 properties, and the
historical duplicate. Conflicts are fatal and are never overwritten.

## Fresh homogeneous A40 production

The final main three-layer benchmark uses the immutable plan in
`frozen/a40_fresh_127_plan.json`: all 127 properties start fresh on two A40s.
It does not import the auxiliary 49-property A4000 baseline. Each worker owns
four deterministic chunks predicted at approximately 6.0--6.1 hours, uses an
isolated working directory and cache tree, and requests one named physical GPU.
The `slurm/a40_worker{0,1}_chunk.sbatch` launchers run these resumable chunks;
`scripts/merge_a40_fresh_results.py` performs the final fail-closed 127-property
merge. The L40S node is not part of this production plan.

## Security and third-party source

DeepT is fetched from `https://github.com/eth-sri/DeepT.git`; its upstream
license remains authoritative. No credentials or remote are configured here.
Legacy workstation paths occur only inside byte-identical frozen scientific
files and immutable external JSON records. Cluster scripts never depend on
them and remap artifact paths in memory.

To connect a private remote later:

```bash
git remote add origin <PRIVATE_REMOTE_URL>
git push -u origin main
```

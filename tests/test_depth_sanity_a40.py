#!/usr/bin/env python3
from __future__ import annotations

import ast
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "research_hab"))

import depth_sanity_common as common
import run_depth_sanity_a40 as runner
import analyze_depth_sanity as analyzer
import coret_bounded_native_depth_graph_v1 as depth_graph
import coret_deept_depth_adapter_v1 as depth_adapter


class DepthSanityA40Tests(unittest.TestCase):
    def test_frozen_manifest_identities(self):
        for depth in (6, 12):
            full, sanity = common.manifests(depth)
            self.assertEqual(full["canonical_manifest_sha256"],
                             common.DEPTHS[depth]["full_sha"])
            self.assertEqual(sanity["canonical_manifest_sha256"],
                             common.DEPTHS[depth]["sanity_sha"])

    def test_production_manifests_are_canonical(self):
        for depth in (6, 12):
            stored = common.production_manifest(depth)
            self.assertEqual(stored["property_count"], 10)

    def test_depth_checkpoint_identity(self):
        for depth in (6, 12):
            manifest = common.production_manifest(depth)
            self.assertEqual(manifest["checkpoint_sha256"],
                             common.DEPTHS[depth]["checkpoint_sha"])
            self.assertEqual(manifest["model"]["configuration"][
                "num_hidden_layers"], depth)

    def test_exact_ten_properties(self):
        self.assertEqual(len(common.production_manifest(6)["properties"]), 10)
        self.assertEqual(len(common.production_manifest(12)["properties"]), 10)

    def test_no_cross_depth_property_mixing(self):
        for depth in (6, 12):
            for item in common.production_manifest(depth)["properties"]:
                self.assertIn(f"stdln{depth}_", item["property_id"])
                self.assertEqual(item["tokenized_length"], len(item["token_ids"]))

    def test_deterministic_five_five_assignment(self):
        for depth in (6, 12):
            left = common.assigned_properties(depth, 0)
            right = common.assigned_properties(depth, 1)
            self.assertEqual(len(left), 5)
            self.assertEqual(len(right), 5)
            self.assertFalse({x["property_id"] for x in left} &
                             {x["property_id"] for x in right})
            self.assertEqual([x["benchmark_ordinal"] for x in left],
                             [0, 2, 4, 6, 8])

    def test_expected_depth_invocation_counts(self):
        self.assertEqual(depth_graph.expected_counts(6)["LayerNorm"], 13)
        self.assertEqual(depth_graph.expected_counts(12)["LayerNorm"], 25)
        self.assertEqual(depth_graph.expected_counts(6)["QK"], 6)
        self.assertEqual(depth_graph.expected_counts(12)["A.V"], 12)

    def test_depth_adapter_paths_and_hashes(self):
        for depth in (6, 12):
            checkpoint, config = depth_adapter._paths(depth)
            self.assertIn(f"standard_layer_norm_{depth}/ckpt-5", checkpoint)
            self.assertTrue(config.endswith("config.json"))

    def test_configure_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "depth_6/worker_0"
            first = runner.configure(6, directory)
            second = runner.configure(6, directory)
            self.assertEqual(first[2], second[2])
            self.assertTrue(first[2].is_relative_to(directory.resolve()))

    def test_workers_are_isolated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = common.worker_result_root(common.worker_dir(root, 6, 0))
            right = common.worker_result_root(common.worker_dir(root, 6, 1))
            twelve = common.worker_result_root(common.worker_dir(root, 12, 0))
            self.assertNotEqual(left, right)
            self.assertNotEqual(left, twelve)
            self.assertTrue(left.is_relative_to(root.resolve()))

    def test_synthetic_two_property_resume_loop(self):
        properties = [{"property_id": "p0"}, {"property_id": "p1"}]
        complete = {"p0"}
        calls = []
        for prop in properties:
            if prop["property_id"] not in complete:
                calls.append(prop["property_id"])
                complete.add(prop["property_id"])
        for prop in properties:
            if prop["property_id"] not in complete:
                calls.append(prop["property_id"])
        self.assertEqual(calls, ["p1"])

    def test_no_exploratory_fused_import(self):
        for path in (REPO / "scripts/run_depth_sanity_a40.py",
                     REPO / "research_hab/coret_bounded_native_depth_graph_v1.py",
                     REPO / "research_hab/coret_deept_depth_adapter_v1.py"):
            tree = ast.parse(path.read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.append(node.module or "")
            self.assertFalse(any("fused" in name.lower() for name in imports))

    def test_prefused_execution_parameters_frozen(self):
        for depth in (6, 12):
            execution = common.production_manifest(depth)["execution"]
            self.assertEqual(execution["requested_AV_generator_tile"], 112)
            self.assertEqual(execution["grouped_temporary_cap_bytes"], 128 << 20)
            self.assertFalse(execution["generic_fallback_allowed"])
            self.assertFalse(execution["fused_exploratory_backend_allowed"])

    def test_search_semantics_frozen(self):
        for depth in (6, 12):
            search = common.production_manifest(depth)["search"]
            self.assertEqual(search["initial_rho_binary64_hex"],
                             (1.0 / 1600.0).hex())
            self.assertEqual(search["midpoint_iterations"], 10)
            self.assertTrue(search["factor_two_bracketing"])

    def test_known_zonotope_nan_assertion_is_fail_closed(self):
        import coret_optimized_historical_127_v1 as scientific_runner

        error = AssertionError(scientific_runner.ZONOTOPE_NAN_NATIVE_ASSERTION)
        fields = scientific_runner._native_domain_failure_fields(error)
        self.assertEqual(fields["terminal_status"],
                         "UNCERTIFIED_DOMAIN_FAILURE")
        self.assertEqual(fields["reason_code"],
                         "UNCERTIFIED_DOMAIN_FAILURE")
        self.assertEqual(fields["domain_failure_diagnostic"],
                         scientific_runner.ZONOTOPE_NAN_DOMAIN_DIAGNOSTIC)
        self.assertIs(fields["certified"], False)
        self.assertIs(fields["authoritative_bound_returned"], False)
        self.assertIs(fields["complete_certificate"], False)
        self.assertEqual(fields["exception_message"], str(error))

    def test_nearby_or_unrelated_assertions_remain_fatal(self):
        import coret_optimized_historical_127_v1 as scientific_runner

        errors = (
            AssertionError("unrelated programming invariant"),
            AssertionError(scientific_runner.ZONOTOPE_NAN_NATIVE_ASSERTION + "."),
        )
        for error in errors:
            with self.assertRaisesRegex(AssertionError, str(error)):
                scientific_runner._native_domain_failure_fields(error)

    def test_existing_domain_failures_remain_fail_closed(self):
        import coret_optimized_historical_127_v1 as scientific_runner

        messages = (
            "sqrt: Bounds must be positive",
            "reciprocal: Bounds must be positive but one element failed",
            scientific_runner.RECIPROCAL_NAN_NATIVE_ASSERTION,
        )
        for message in messages:
            fields = scientific_runner._native_domain_failure_fields(
                AssertionError(message))
            self.assertEqual(fields["terminal_status"],
                             "UNCERTIFIED_DOMAIN_FAILURE")
            self.assertIs(fields["certified"], False)

    def test_successful_query_return_is_not_transformed(self):
        import coret_optimized_historical_127_v1 as scientific_runner

        sentinel = object()

        def query_boundary(callback):
            try:
                return callback()
            except AssertionError as error:
                return scientific_runner._native_domain_failure_fields(error)

        self.assertIs(query_boundary(lambda: sentinel), sentinel)

    def test_preflight_is_zero_solve(self):
        for depth in (6, 12):
            row = runner.preflight(depth)
            self.assertEqual(row["scientific_queries"], 0)
            self.assertEqual(row["bound_entrypoint_calls"], 0)
            self.assertEqual(row["training_runs"], 0)

    def test_read_only_analyzer_on_synthetic_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = common.production_manifest(6)
            for prop in manifest["properties"]:
                worker = int(prop["worker_id"])
                result_root = common.worker_result_root(
                    common.worker_dir(root, 6, worker))
                directory = result_root / "properties" / prop["property_id"]
                directory.mkdir(parents=True, exist_ok=True)
                deep = float(prop["cached_DeepT_reference"][
                    "certified_lower_endpoint_binary64"])
                payload = {
                    "terminal_status": "COMPLETE",
                    "canonical_manifest_sha256": manifest[
                        "canonical_manifest_sha256"],
                    "property_id": prop["property_id"],
                    "benchmark_ordinal": prop["benchmark_ordinal"],
                    "certified_radius": deep,
                    "cached_DeepT_certified_radius": deep,
                    "radius_ratio_to_cached_DeepT": 1.0,
                    "total_wall_time_seconds": 10.0 + prop["benchmark_ordinal"],
                    "checker_failure_count": 0,
                    "all_provenance_consistent": True,
                    "all_support_claims_validated": True,
                    "generic_fallback_count": 0,
                }
                payload["record_sha256"] = common.canonical(payload)
                (directory / "result_v1.json").write_text(
                    json.dumps(payload, sort_keys=True) + "\n")
            report = analyzer.analyze(6, root, 10)
            self.assertEqual(report["classification"], "DEPTH_SANITY_PASS")
            self.assertEqual(report["ratio_summary"]["median"], 1.0)
            self.assertIsNotNone(report["estimated_full_makespan_2x_A40_seconds"])

    def test_analyzer_requires_exact_completed_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(RuntimeError):
                analyzer.analyze(12, Path(temporary), 10)


if __name__ == "__main__":
    unittest.main()

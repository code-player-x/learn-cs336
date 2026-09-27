"""Regression proof for the knowledge map's deterministic acceptance example."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from code.basic_study import knowledge_acceptance as example


class KnowledgeAcceptanceTests(unittest.TestCase):
    def test_cli_report_is_computed_and_passes(self):
        result = subprocess.run(
            [sys.executable, "-m", "code.basic_study.knowledge_acceptance"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, check=True,
        )
        report = json.loads(result.stdout)
        self.assertTrue(report["passed"])
        self.assertEqual(report["fixture_version"], example.FIXTURE_VERSION)
        self.assertEqual(len(report["rag"]), 7)
        self.assertEqual(sum(row["baseline_matches_gold"] for row in report["rag"]), 1)
        self.assertTrue(all(row["matches_gold"] for row in report["rag"]))
        self.assertEqual(report["agent"]["orders"], 1)
        self.assertEqual(report["agent"]["writes"], 1)

    def test_rag_gate_rejects_each_invalid_fixture(self):
        for case in example.rag_fixture_cases():
            with self.subTest(case=case["id"]):
                evidence = example.authorized_evidence(case["retrieved"], "customer")
                self.assertEqual(example.citation_gate(case["claims"], evidence), case["gold"])

    def test_permission_filter_runs_before_generation(self):
        self.assertEqual(example.authorized_evidence(["internal", "missing"], "customer"), {})
        self.assertIn("internal", example.authorized_evidence(["internal"], "staff"))
        self.assertEqual(example.authorized_evidence(["refund"], "unknown-role"), {})

    def test_claim_source_must_be_retrieved(self):
        quote = ("refund", "v1", example.DOCUMENTS["refund"]["text"])
        self.assertEqual(example.citation_gate([quote], {}), {"status": "refused", "answer": None})

    def test_lost_receipt_and_replays_make_only_one_write(self):
        store = example.MockOrderStore()
        parameters = ("notebook", 2)
        with self.assertRaises(TimeoutError):
            store.create("key", parameters, parameters, lose_receipt=True)
        recovered = store.lookup("key", parameters)
        self.assertIsNotNone(recovered)
        for _ in range(3):
            self.assertEqual(store.create("key", parameters, parameters), recovered)
        self.assertEqual(len(store.orders), 1)
        self.assertEqual(store.write_count, 1)

    def test_changed_parameters_and_approval_are_rejected(self):
        store = example.MockOrderStore()
        old, changed = ("notebook", 2), ("notebook", 3)
        store.create("key", old, old)
        with self.assertRaises(PermissionError):
            store.create("new-key", changed, old)
        with self.assertRaises(ValueError):
            store.create("key", changed, changed)
        with self.assertRaises(ValueError):
            store.lookup("key", changed)
        self.assertEqual(store.write_count, 1)

    def test_invalid_new_order_does_not_write(self):
        for parameters in [("notebook", 0), ("notebook", -1), ("notebook", True), ("", 1)]:
            with self.subTest(parameters=parameters):
                store = example.MockOrderStore()
                with self.assertRaises(ValueError):
                    store.create("key", parameters, parameters)
                self.assertEqual(store.write_count, 0)
        store = example.MockOrderStore()
        with self.assertRaises(ValueError):
            store.create("", ("notebook", 1), ("notebook", 1))
        self.assertEqual(store.write_count, 0)


if __name__ == "__main__":
    unittest.main()

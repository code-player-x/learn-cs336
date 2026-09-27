"""Offline retrieval/ACL/gate tests and a local-only HTTP protocol smoke test."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import subprocess
import sys
import threading
import unittest
from urllib.error import HTTPError

from code.basic_study import rag_example as rag


@contextmanager
def local_chat_server(content, redirect=False):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, "payload": payload,
                             "authorization": self.headers.get("Authorization")})
            if redirect:
                self.send_response(307)
                self.send_header("Location", "/must-not-follow")
                self.end_headers()
                return
            response = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


class RagExampleTests(unittest.TestCase):
    def setUp(self):
        self.version, self.chunks = rag.load_corpus()

    def test_ingestion_and_fixture_contract(self):
        self.assertEqual(len(self.chunks), 12)
        self.assertEqual(self.chunks[0]["id"], "refund:1")
        self.assertEqual(self.chunks[0]["text"], "普通商品签收后7天内可申请退货。")
        by_id = {chunk["id"]: chunk for chunk in self.chunks}
        split_ids, split_gold = [], []
        for split in ("dev", "heldout"):
            cases = json.loads((rag.DATA_DIR / f"{split}.json").read_text(encoding="utf-8"))
            ids, gold = set(), set()
            for case in cases:
                self.assertNotIn(case["id"], ids)
                ids.add(case["id"])
                self.assertEqual(bool(case["gold_chunk_ids"]), case["expected_status"] == "answered")
                for source_id in case["gold_chunk_ids"]:
                    self.assertIn(case["role"], by_id[source_id]["readers"])
                    gold.add(source_id)
            split_ids.append(ids)
            split_gold.append(gold)
        self.assertFalse(split_ids[0] & split_ids[1])
        self.assertFalse(split_gold[0] & split_gold[1])

    def test_bm25_hand_calculation(self):
        chunks = [{"id": "a", "text": "alpha alpha", "readers": ["customer"]},
                  {"id": "b", "text": "beta", "readers": ["customer"]}]
        hit = rag.retrieve("alpha", chunks, "customer")[0]
        # N=2, df=1, tf=2, length=2, avg_length=1.5, k1=1.5, b=.75.
        self.assertEqual(hit["id"], "a")
        self.assertAlmostEqual(hit["score"], math.log(2) * 5 / 3.875)

    def test_empty_query_and_corpus(self):
        for question, chunks, role in (("", self.chunks, "customer"),
                                       ("？", self.chunks, "customer"),
                                       ("配送", [], "customer"),
                                       ("配送", self.chunks, "unknown")):
            self.assertEqual(rag.retrieve(question, chunks, role), [])
        for top_k in (0, -1, True):
            with self.assertRaises(ValueError):
                rag.retrieve("配送", self.chunks, "customer", top_k)

    def test_acl_precedes_statistics_ranking_and_context(self):
        public = [c for c in self.chunks if "customer" in c["readers"]]
        hits = rag.retrieve("客服在线时间", self.chunks, "customer")
        self.assertEqual(hits, rag.retrieve("客服在线时间", public, "customer"))
        self.assertNotIn("internal", json.dumps(rag.model_messages("客服在线时间", hits)))
        question = "员工报销审批上限是多少？"
        self.assertFalse(any(h["id"].startswith("internal:") for h in rag.retrieve(question, self.chunks, "customer")))
        self.assertEqual(rag.retrieve(question, self.chunks, "staff")[0]["id"], "internal:1")

    def test_exact_source_version_quote_gate_and_revocation(self):
        hits = rag.retrieve("普通商品签收后退货", self.chunks, "customer")
        candidate = rag.extractive_candidate("普通商品签收后退货", hits)
        self.assertEqual(rag.check_candidate(candidate, hits)["status"], "answered")
        for field, value in (("source_id", "internal:1"), ("version", "v0"), ("quote", "退货免运费。")):
            changed = json.loads(json.dumps(candidate))
            changed["citations"][0][field] = value
            actual = rag.check_candidate(changed, hits)
            self.assertEqual(actual["status"], "refused")
            self.assertTrue(actual["gate_rejected"])
            self.assertEqual(actual["citation_ids"], [])
        # Rebuild from current roles on every request: withdrawn content disappears.
        withdrawn = [dict(c, readers=[]) if c["id"].startswith("refund:") else c for c in self.chunks]
        self.assertFalse(any(h["id"].startswith("refund:") for h in rag.retrieve("退货", withdrawn, "customer")))

    def test_candidate_schema_fails_closed(self):
        hits = rag.retrieve("退货", self.chunks, "customer")
        for candidate in (None, {}, {"status": "answered", "citations": []},
                          {"status": "refused", "citations": [{}]},
                          {"status": "answered", "citations": [{"source_id": 42}]},
                          {"status": "refused", "citations": [], "answer": "injected"}):
            with self.assertRaises(ValueError):
                rag.check_candidate(candidate, hits)

    def test_quoted_answer_can_still_be_incomplete(self):
        report = rag.evaluate()
        case = next(c for c in report["cases"] if c["id"] == "two_sources")
        self.assertEqual(case["recall_at_k"], 1)
        self.assertEqual(case["actual"]["status"], "answered")
        self.assertFalse(case["actual"]["gate_rejected"])
        self.assertFalse(case["matches_gold"])
        self.assertEqual(case["failure"], "answer_selection_or_refusal")
        self.assertEqual(report["gold_evidence_set_accuracy"], sum(c["matches_gold"] for c in report["cases"]) / 8)

    def test_errors_are_not_dropped_or_exposed(self):
        def fail(question, hits):
            raise TimeoutError("secret-test-credential")
        report = rag.evaluate(generator=fail, generator_name="failing-test")
        self.assertEqual(report["errors"], report["sample_count"])
        self.assertEqual(report["gold_evidence_set_accuracy"], 0)
        self.assertNotIn("secret-test-credential", json.dumps(report))

    def test_model_input_has_evidence_but_no_gold_or_role(self):
        hits = rag.retrieve("配送", self.chunks, "customer")
        messages = rag.model_messages("配送", hits)
        data = json.loads(messages[1]["content"])
        self.assertEqual(set(data), {"question", "evidence"})
        self.assertEqual(set(data["evidence"][0]), {"source_id", "version", "text"})

    def test_real_local_http_protocol(self):
        question = "普通商品签收后退货"
        hits = rag.retrieve(question, self.chunks, "customer")
        expected = rag.extractive_candidate(question, hits)
        with local_chat_server(json.dumps(expected)) as (endpoint, requests):
            generate = rag.http_generator(endpoint, "local-test", "fake-test-key")
            actual = rag.check_candidate(generate(question, hits), hits)
            self.assertEqual(actual["status"], "answered")
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]["authorization"], "Bearer fake-test-key")
            self.assertEqual(requests[0]["payload"]["messages"], rag.model_messages(question, hits))
            self.assertEqual(requests[0]["payload"]["model"], "local-test")

    def test_http_invalid_response_and_redirect(self):
        with local_chat_server("not JSON") as (endpoint, _):
            report = rag.evaluate(generator=rag.http_generator(endpoint, "local-test"))
            self.assertEqual(report["errors"], 8)
        with local_chat_server("unused", redirect=True) as (endpoint, requests):
            with self.assertRaises(HTTPError):
                rag.http_generator(endpoint, "local-test", "fake-key")("配送", [])
            self.assertEqual(len(requests), 1)
        for endpoint in ("http://example.invalid/v1/chat/completions",
                         "https://key@example.invalid/v1/chat/completions"):
            with self.assertRaises(ValueError):
                rag.http_generator(endpoint, "test")

    def test_offline_cli_and_mismatched_network_options(self):
        root = rag.DATA_DIR.parents[2]
        result = subprocess.run([sys.executable, "-m", "code.basic_study.rag_example"],
                                cwd=root, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["generator"], "extractive-baseline")
        self.assertEqual(report["sample_count"], 8)
        self.assertEqual(len(report["corpus_sha256"]), 64)
        bad = subprocess.run([sys.executable, "-m", "code.basic_study.rag_example", "--model", "test"],
                             cwd=root, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()

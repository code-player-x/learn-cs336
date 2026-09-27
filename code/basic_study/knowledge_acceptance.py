"""Deterministic teaching fixtures, not an LLM, RAG platform or durable Agent.

Run from the repository root: python -m code.basic_study.knowledge_acceptance
Uses only the standard library and performs no network or filesystem writes.
"""

import json


FIXTURE_VERSION = "exact-quotes-and-mock-orders-v1"
DOCUMENTS = {
    "refund": {
        "version": "v1",
        "text": "普通商品签收后7天内可申请退货。",
        "readers": {"customer", "staff"},
    },
    "internal": {
        "version": "v1",
        "text": "内部审批阈值为500元。",
        "readers": {"staff"},
    },
}


def authorized_evidence(document_ids, role):
    """Apply a trusted reader filter before returning evidence to generation."""
    return {
        doc_id: dict(DOCUMENTS[doc_id])
        for doc_id in document_ids
        if doc_id in DOCUMENTS and role in DOCUMENTS[doc_id]["readers"]
    }


def citation_gate(claims, evidence):
    """Accept exact source sentences only; this is NOT semantic entailment.

    Each fixture claim is (document_id, version, quoted_sentence).
    Any missing/unauthorized, stale or non-identical quote rejects the answer.
    """
    refused = {"status": "refused", "answer": None}
    if not claims:
        return refused
    for doc_id, version, sentence in claims:
        document = evidence.get(doc_id)
        if (document is None or version != document["version"]
                or sentence != document["text"]):
            return refused
    return {"status": "answered", "answer": " ".join(c[2] for c in claims)}


def rag_fixture_cases():
    """Synthetic candidates and gold outcomes, not a retrieval benchmark."""
    quote = ("refund", "v1", DOCUMENTS["refund"]["text"])
    refused = {"status": "refused", "answer": None}
    return [
        {"id": "supported", "question": "普通商品多久内可申请退货？",
         "retrieved": ["refund"], "claims": [quote],
         "gold": {"status": "answered", "answer": quote[2]}},
        {"id": "unsupported", "question": "退货运费谁承担？",
         "retrieved": ["refund"], "claims": [("refund", "v1", "退货运费全免。")],
         "gold": refused},
        {"id": "unauthorized", "question": "内部审批阈值是多少？",
         "retrieved": ["internal"],
         "claims": [("internal", "v1", DOCUMENTS["internal"]["text"])],
         "gold": refused},
        {"id": "stale_version", "question": "普通商品多久内可申请退货？",
         "retrieved": ["refund"], "claims": [("refund", "v0", quote[2])],
         "gold": refused},
        {"id": "mixed_claims", "question": "退货期限与运费是什么？",
         "retrieved": ["refund"],
         "claims": [quote, ("refund", "v1", "退货运费全免。")],
         "gold": refused},
        {"id": "invented_source", "question": "普通商品多久内可申请退货？",
         "retrieved": ["refund"], "claims": [("missing", "v1", quote[2])],
         "gold": refused},
        {"id": "empty_claims", "question": "普通商品多久内可申请退货？",
         "retrieved": ["refund"], "claims": [], "gold": refused},
    ]


def run_rag_examples():
    results = []
    for case in rag_fixture_cases():
        # Unsafe baseline: publish candidate text without checking its evidence.
        baseline = {"status": "answered", "answer": " ".join(c[2] for c in case["claims"])}
        evidence = authorized_evidence(case["retrieved"], role="customer")
        actual = citation_gate(case["claims"], evidence)
        results.append({
            "id": case["id"], "baseline_matches_gold": baseline == case["gold"],
            "actual": actual, "matches_gold": actual == case["gold"],
        })
    return results


class MockOrderStore:
    """Single-process mock: no real purchases, persistence or concurrency proof.

    An approval here is a trusted test tuple, NOT a user-supplied authorization.
    Production needs authenticated approval records, atomic writes and recovery.
    """

    def __init__(self):
        self.orders = {}
        self.write_count = 0

    def create(self, key, parameters, approved_parameters, lose_receipt=False):
        # Bind approval to the actual parameters, even on a replay.
        if parameters != approved_parameters:
            raise PermissionError("parameters changed after approval")
        if key in self.orders:
            if self.orders[key]["parameters"] != parameters:
                raise ValueError("same idempotency key, different parameters")
            return dict(self.orders[key])
        if (not key or len(parameters) != 2 or not parameters[0]
                or type(parameters[1]) is not int or parameters[1] <= 0):
            raise ValueError("need a key, item and positive integer quantity")
        receipt = {"order_id": f"mock-{len(self.orders) + 1}", "parameters": parameters}
        self.orders[key] = receipt
        self.write_count += 1
        if lose_receipt:
            raise TimeoutError("mock write succeeded but response was lost")
        return dict(receipt)

    def lookup(self, key, expected_parameters):
        receipt = self.orders.get(key)
        if receipt is None:
            return None
        if receipt["parameters"] != expected_parameters:
            raise ValueError("receipt parameters do not match the task")
        return dict(receipt)


def run_agent_examples():
    store = MockOrderStore()
    key, parameters = "task-001", ("notebook", 2)
    lost = False
    try:
        store.create(key, parameters, parameters, lose_receipt=True)
    except TimeoutError:
        lost = True
    recovered = store.lookup(key, parameters)
    replay = store.create(key, parameters, parameters)
    approval_rejected = conflict_rejected = False
    try:
        store.create("task-002", ("notebook", 3), parameters)
    except PermissionError:
        approval_rejected = True
    try:
        store.create(key, ("notebook", 3), ("notebook", 3))
    except ValueError:
        conflict_rejected = True
    return {
        "orders": len(store.orders), "writes": store.write_count,
        "checks": {
            "receipt_lost_then_recovered": lost and recovered is not None,
            "replay_returns_same_receipt": recovered == replay,
            "one_matching_order": len(store.orders) == 1
                and store.orders[key]["parameters"] == parameters,
            "one_write": store.write_count == 1,
            "changed_approval_rejected": approval_rejected,
            "changed_parameters_rejected": conflict_rejected,
            "empty_store_does_not_prove_completion":
                MockOrderStore().lookup(key, parameters) is None,
        },
    }


def main():
    rag, agent = run_rag_examples(), run_agent_examples()
    passed = all(row["matches_gold"] for row in rag) and all(agent["checks"].values())
    print(json.dumps({
        "fixture_version": FIXTURE_VERSION,
        "scope": "synthetic deterministic fixtures; not production validation",
        "rag": rag, "agent": agent, "passed": passed,
    }, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

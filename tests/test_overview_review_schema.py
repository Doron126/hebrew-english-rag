"""Review formatting constraints cannot replace evidence or witness checks."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import unittest
from unittest.mock import patch

from jsonschema import Draft7Validator

from document_reading import SourceSummaryEvidence, make_passage
import overview_semantic


class OverviewReviewSchemaTests(unittest.TestCase):
    def setUp(self):
        self.question = "Summarize this document."
        self.tenant = "research"
        self.texts = ("The library lends books.", "The room is quiet.")
        metadata = {"tenant_id": self.tenant, "source_id": "s" * 43,
                    "source_sha256": hashlib.sha256("\n".join(self.texts).encode()).hexdigest(),
                    "source_relative_path": "guide.md", "mime_type": "text/markdown"}
        passages = tuple(make_passage(text, text, metadata, "chunk-" + str(index), str(index))
                         for index, text in enumerate(self.texts, 1))
        self.evidence = SourceSummaryEvidence(passages, True, 1, 2,
                                               quote_eligible_ids=("1", "2"))
        self.claims = tuple((text, (str(index),)) for index, text in enumerate(self.texts, 1))
        self.schema = overview_semantic.review_response_schema(
            self.question, self.evidence, self.claims, self.tenant)
        Draft7Validator.check_schema(self.schema)
        self.validator = Draft7Validator(self.schema)
        self.review = {"claims": [
            ["c1", "supported", [["1", self.texts[0]]], ""],
            ["c2", "supported", [["2", self.texts[1]]], ""],
        ], "coverage": ["complete", ["1", "2"], ""]}

    def bind(self, review, evidence=None):
        return overview_semantic.bind_review(self.question, evidence or self.evidence,
            self.claims, self.tenant, json.dumps(review, ensure_ascii=False))

    def test_valid_fixed_identity_output_reaches_unchanged_binder(self):
        self.validator.validate(self.review)
        witness = self.bind(self.review)
        self.assertTrue(overview_semantic.review_matches(
            self.question, self.evidence, self.claims, self.tenant, witness))

    def test_schema_rejects_wrong_ids_order_citations_and_tuple_shapes(self):
        malformed = []
        value = deepcopy(self.review); value["claims"][0][0] = self.texts[0]; malformed.append(value)
        value = deepcopy(self.review); value["claims"].reverse(); malformed.append(value)
        value = deepcopy(self.review); value["claims"][0][2][0][0] = "2"; malformed.append(value)
        value = deepcopy(self.review); value["claims"][0][2][0] = [self.texts[0]]; malformed.append(value)
        value = deepcopy(self.review); value["claims"][0][1] = "approved"; malformed.append(value)
        value = deepcopy(self.review); value["claims"].pop(); malformed.append(value)
        value = deepcopy(self.review); value["coverage"].pop(); malformed.append(value)
        value = deepcopy(self.review); value["coverage"][1] = ["999"]; malformed.append(value)
        value = deepcopy(self.review); value["extra"] = "unexpected"; malformed.append(value)
        for value in malformed:
            with self.subTest(review=value):
                self.assertFalse(self.validator.is_valid(value))

    def test_schema_leaves_unsupported_and_unresolved_verdicts_available(self):
        review = {"claims": [["c1", "unsupported", [], "Not supported"],
                              ["c2", "unresolved", [], "Insufficient evidence"]],
                  "coverage": ["unresolved", [], "No supported claims"]}
        self.validator.validate(review)
        parsed = json.loads(self.bind(review))["payload"]["review"]
        self.assertEqual(parsed, review)

    def test_schema_valid_fabricated_excerpt_is_still_discarded(self):
        review = deepcopy(self.review)
        review["claims"][0][2][0][1] = "Invented statement."
        self.validator.validate(review)
        parsed = json.loads(self.bind(review))["payload"]["review"]
        self.assertEqual(parsed["claims"][0], ["c1", "unresolved", [], "invalid source excerpt"])
        self.assertEqual(parsed["coverage"][0], "partial")

    def test_schema_valid_output_cannot_override_incomplete_source_or_changed_tenant(self):
        incomplete = replace(self.evidence, complete=False, selection_complete=False)
        parsed = json.loads(self.bind(self.review, incomplete))["payload"]["review"]
        self.assertEqual(parsed["coverage"][0], "partial")
        witness = self.bind(self.review)
        self.assertFalse(overview_semantic.review_matches(
            self.question, self.evidence, self.claims, "other-tenant", witness))
        self.assertFalse(overview_semantic.review_matches(
            "A changed request", self.evidence, self.claims, self.tenant, witness))

    def test_tagged_single_reference_schema_keeps_five_scalar_fields(self):
        material = overview_semantic._material(self.question, self.evidence, self.claims, self.tenant)
        for claim in material["claims"]:
            claim["one_ref_required"] = True
        # Schema construction consumes the same normalized material that the
        # unchanged tagged-review path produces; no binder is mocked.
        with patch.object(overview_semantic, "_material", return_value=material):
            schema = overview_semantic.review_response_schema(
                self.question, self.evidence, self.claims, self.tenant)
        validator = Draft7Validator(schema)
        valid = {"claims": [["c1", "supported", "1", self.texts[0], ""],
                             ["c2", "unresolved", "", "", "Insufficient evidence"]],
                 "coverage": ["partial", ["1"], "One claim unresolved"]}
        validator.validate(valid)
        self.assertFalse(validator.is_valid(self.review))


if __name__ == "__main__":
    unittest.main()

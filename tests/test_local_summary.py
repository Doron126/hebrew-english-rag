"""Local output choices retain literal source identity and bounded scope."""
from dataclasses import replace
import hashlib
import unittest
from unittest.mock import patch

from jsonschema import Draft7Validator

from document_query import DocumentReference
from document_reading import SourceSummaryEvidence, make_passage, passage_valid
from local_summary import literal_summary_response_schema
from passage_synthesis import _clause_supports
from source_access import SOURCE_LINK_SCHEMA_VERSION, guessed_mime_type


class LocalSummaryTests(unittest.TestCase):
    def evidence(self, texts, *, filename="guide.md"):
        body = "\n\n".join(texts)
        metadata = {
            "tenant_id": "research", "source_id": "s" * 43,
            "source_relative_path": filename, "display_name": filename,
            "source_link_schema_version": SOURCE_LINK_SCHEMA_VERSION,
            "source_sha256": hashlib.sha256(body.encode()).hexdigest(),
            "source_size": len(body.encode()), "source_mtime_ns": 0,
            "mime_type": guessed_mime_type(filename),
        }
        passages = tuple(make_passage(body, text, metadata, "chunk-" + str(index), str(index))
                         for index, text in enumerate(texts, 1))
        return SourceSummaryEvidence(passages, True, 1, len(passages),
            reference=DocumentReference("document", filename),
            quote_eligible_ids=tuple(p.citation_id for p in passages))

    def validator(self, evidence):
        schema = literal_summary_response_schema(evidence)
        Draft7Validator.check_schema(schema)
        return Draft7Validator(schema)

    def test_bilingual_sentences_are_separate_exact_source_choices(self):
        cases = (
            ("The garden opens in spring.\nMembers may enter only with a pass.",
             "The garden opens in spring.", "Members may enter only with a pass."),
            ("הגן נפתח באביב.\nחברים רשאים להיכנס רק עם אישור.",
             "הגן נפתח באביב.", "חברים רשאים להיכנס רק עם אישור."),
        )
        for source, first, second in cases:
            with self.subTest(source=source):
                evidence = self.evidence((source,))
                validator = self.validator(evidence)
                validator.validate({"claims": [["1", first], ["1", second]]})
                self.assertFalse(validator.is_valid({"claims": [["1", first + " " + second]]}))
                self.assertTrue(passage_valid(evidence.passages[0]))
                self.assertTrue(_clause_supports(second, second))
                self.assertFalse(_clause_supports(second.replace("only ", "").replace("רק ", ""), second))

    def test_citation_pairs_cannot_transfer_text_between_sources(self):
        evidence = self.evidence(("The east room is quiet.", "The west room is bright."))
        validator = self.validator(evidence)
        validator.validate({"claims": [["1", "The east room is quiet."],
                                        ["2", "The west room is bright."]]})
        for bad in (
            [["2", "The east room is quiet."]],
            [["999", "The east room is quiet."]],
            [["1", "The east room is bright."]],
            [["1", "החדר המזרחי שקט."]],
            [["1", "The east room is quiet.", "extra"]],
        ):
            with self.subTest(claims=bad):
                self.assertFalse(validator.is_valid({"claims": bad}))
        restricted = replace(evidence, quote_eligible_ids=("2",))
        self.assertFalse(self.validator(restricted).is_valid({"claims": [["1", "The east room is quiet."]]}))

    def test_existing_boundaries_keep_abbreviations_decimals_and_conditions(self):
        source = "Dr. Lee records 2.5 litres.\nA refill is allowed only if the seal is intact."
        validator = self.validator(self.evidence((source,)))
        validator.validate({"claims": [["1", "Dr. Lee records 2.5 litres."],
            ["1", "A refill is allowed only if the seal is intact."]]})
        for fragment in ("Dr.", "Lee records 2.5 litres.", "A refill is allowed."):
            self.assertFalse(validator.is_valid({"claims": [["1", fragment]]}))

    def test_wrapped_text_and_bullets_retain_original_internal_bytes(self):
        source = "• A quiet reading\n  space\n• A place for study"
        validator = self.validator(self.evidence((source,)))
        validator.validate({"claims": [["1", "• A quiet reading\n  space"],
                                        ["1", "• A place for study"]]})
        self.assertFalse(validator.is_valid({"claims": [["1", "• A quiet reading space"]]}))

    def test_output_shape_and_claim_limit_are_bounded(self):
        validator = self.validator(self.evidence(("The room is quiet.",)))
        claim = ["1", "The room is quiet."]
        for value in ({"claims": []}, {"claims": [claim] * 17},
                      {"claims": [claim], "coverage": "complete"}, {"claims": "text"}):
            self.assertFalse(validator.is_valid(value))

    def test_longer_source_requires_several_source_units(self):
        sentences = ("The garden is open.", "Members have a key.",
                     "The room is quiet.", "The path is shaded.")
        validator = self.validator(self.evidence((" ".join(sentences),)))
        claims = [["1", sentence] for sentence in sentences]
        validator.validate({"claims": claims})
        self.assertFalse(validator.is_valid({"claims": claims[:2]}))

    def test_overview_invalid_scope_and_altered_passages_fail_closed(self):
        evidence = self.evidence(("The room is quiet.",))
        bad_passage = replace(evidence.passages[0], source_claim="The room is bright.")
        altered_owner = replace(evidence.passages[0], tenant_id="other-tenant", citation_id="2")
        for invalid in (
            replace(evidence, overview_synthesis=True),
            replace(evidence, reference=None),
            replace(evidence, summary_topic="a specific detail"),
            replace(evidence, quote_eligible_ids=None),
            replace(evidence, quote_eligible_ids=("999",)),
            replace(evidence, passages=(bad_passage,)),
            replace(evidence, passages=(evidence.passages[0], altered_owner), quote_eligible_ids=("1", "2")),
        ):
            with self.subTest(evidence=invalid):
                with self.assertRaises(ValueError):
                    literal_summary_response_schema(invalid)

    def test_large_schema_fails_without_silently_omitting_source_choices(self):
        evidence = self.evidence(("The room is quiet. The garden is green.",))
        with patch("local_summary.MAX_SOURCE_CHOICES", 1):
            with self.assertRaisesRegex(ValueError, "source_choice_limit_exceeded"):
                literal_summary_response_schema(evidence)
        with patch("local_summary.MAX_SCHEMA_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "schema_byte_limit_exceeded"):
                literal_summary_response_schema(evidence)
        with patch("local_summary.MAX_CHOICE_CHARS", 10):
            with self.assertRaisesRegex(ValueError, "source_sentence_limit_exceeded"):
                literal_summary_response_schema(evidence)


if __name__ == "__main__":
    unittest.main()

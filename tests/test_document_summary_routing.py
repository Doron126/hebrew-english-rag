"""Filename punctuation must retain the existing authenticated reading route."""
import base64
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

from document_query import named_document_reference, reference_matches
from document_reading import (collect_summary, named_summary_request, passage_valid,
                              unqualified_whole_summary, validate_summary_evidence)
from passage_synthesis import is_whole_named_source_summary, is_whole_source_summary
from request_understanding import understand_request
from semantic_contracts import build_request_execution_contract
from source_access import SOURCE_LINK_SCHEMA_VERSION, guessed_mime_type


class _StoredRows:
    def __init__(self, rows):
        self.rows = rows

    def scan_records(self, **kwargs):
        return self.rows, True


class DocumentSummaryRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "research").mkdir()

    def source(self, filename, text):
        path = self.root / "research" / filename
        path.write_text(text, encoding="utf-8")
        raw = path.read_bytes()
        source_id = base64.urlsafe_b64encode(hashlib.sha256(filename.encode()).digest()).decode().rstrip("=")
        metadata = {
            "tenant_id": "research", "source_id": source_id,
            "source_relative_path": filename, "source": filename,
            "display_name": filename, "title": filename,
            "mime_type": guessed_mime_type(filename),
            "source_link_schema_version": SOURCE_LINK_SCHEMA_VERSION,
            "source_sha256": hashlib.sha256(raw).hexdigest(),
            "source_size": len(raw), "source_mtime_ns": path.stat().st_mtime_ns,
            "chunk_index": 0,
        }
        row = {"id": filename + "::chunk::000000", "document": text, "metadata": metadata}
        return _StoredRows([row]), source_id

    def test_english_and_hebrew_periods_keep_authenticated_summary_route(self):
        cases = (
            ("library-guide.md", "Summarize library-guide.md.", "The library lends books."),
            ("מדריך-הספרייה.md", "סכם את מדריך-הספרייה.md.", "הספרייה משאילה ספרים."),
        )
        for filename, query, content in cases:
            with self.subTest(query=query):
                retriever, source_id = self.source(filename, content)
                plan = understand_request(query, selected_source_id=source_id).plan
                contract = build_request_execution_contract(query, plan=plan, source_scope=(source_id,))
                self.assertEqual(contract.requirements[0].kind, "document_summary")
                self.assertEqual(contract.requirements[0].value_type, "source_passage")
                self.assertEqual(contract.source_scope, (source_id,))
                self.assertEqual(contract.subjects, ())
                evidence = collect_summary(retriever, "research", self.root, query, {"source": filename})
                self.assertTrue(evidence.passages)
                self.assertTrue(evidence.complete)
                self.assertTrue(all(passage_valid(passage) for passage in evidence.passages))
                self.assertTrue(validate_summary_evidence(evidence, query))

    def test_bilingual_named_summaries_with_or_without_particle_are_whole(self):
        cases = (
            ("garden.md", "Summarize garden.md.", "The garden opens in spring."),
            ("garden.md", "Summarize the garden.md.", "The garden opens in spring."),
            ("מדריך-גן.md", "סכם מדריך-גן.md.", "הגן נפתח באביב."),
            ("מדריך-גן.md", "סכם את מדריך-גן.md.", "הגן נפתח באביב."),
        )
        for filename, query, content in cases:
            with self.subTest(query=query):
                retriever, _source_id = self.source(filename, content)
                reference, body = named_summary_request(query)
                self.assertTrue(unqualified_whole_summary(reference, body))
                evidence = collect_summary(retriever, "research", self.root, query, {"source": filename})
                self.assertTrue(evidence.passages)
                self.assertTrue(validate_summary_evidence(evidence, query))
                self.assertTrue(all(passage_valid(passage) for passage in evidence.passages))
                self.assertTrue(is_whole_source_summary(evidence))
                self.assertTrue(is_whole_named_source_summary(evidence))
                from local_summary import literal_summary_response_schema
                schema = literal_summary_response_schema(replace(evidence,
                    quote_eligible_ids=tuple(p.citation_id for p in evidence.passages)))
                self.assertIn(content, schema["properties"]["claims"]["items"]["oneOf"][0]["items"][1]["enum"])

    def test_qualified_requests_and_existing_named_scope_guards_remain_narrow(self):
        cases = (
            ("garden.md", "Summarize garden.md, opening hours.", "The garden opens in spring."),
            ("מדריך-גן.md", "סכם את מדריך-גן.md, שעות הפעילות.", "הגן נפתח באביב."),
        )
        for filename, query, content in cases:
            with self.subTest(query=query):
                retriever, _source_id = self.source(filename, content)
                reference, body = named_summary_request(query)
                self.assertFalse(unqualified_whole_summary(reference, body))
                evidence = collect_summary(retriever, "research", self.root, query, {"source": filename})
                self.assertFalse(is_whole_source_summary(evidence))
                self.assertFalse(is_whole_named_source_summary(evidence))
        retriever, _source_id = self.source("garden.md", "The garden opens in spring.")
        evidence = collect_summary(retriever, "research", self.root,
                                   "Summarize the garden.md.", {"source": "garden.md"})
        self.assertFalse(is_whole_named_source_summary(replace(evidence, complete=False)))
        self.assertFalse(is_whole_named_source_summary(replace(evidence, summary_topic="opening hours")))
        self.assertFalse(is_whole_named_source_summary(replace(evidence,
            reference=replace(evidence.reference, kind="presentation_topic"))))

    def test_named_summary_still_rejects_a_different_selected_document(self):
        retriever, _ = self.source("another-guide.md", "A different guide about garden plants.")
        query = "Summarize library-guide.md."
        reference = named_document_reference(query)
        self.assertFalse(reference_matches(reference, "another-guide.md"))
        evidence = collect_summary(retriever, "research", self.root, query, {"source": "another-guide.md"})
        self.assertTrue(evidence.passages)
        self.assertFalse(validate_summary_evidence(evidence, query))

    def test_summary_rejects_modified_passage_content(self):
        retriever, _ = self.source("library-guide.md", "The library lends books.")
        query = "Summarize library-guide.md."
        evidence = collect_summary(retriever, "research", self.root, query, {"source": "library-guide.md"})
        fabricated = "The library lends cars."
        altered = replace(evidence.passages[0], source_claim=fabricated,
                          span_hash=hashlib.sha256(fabricated.encode()).hexdigest())
        self.assertFalse(passage_valid(altered))
        self.assertFalse(validate_summary_evidence(replace(evidence, passages=(altered,)), query))

    def test_summary_still_rejects_changed_original_file(self):
        retriever, _ = self.source("library-guide.md", "The library lends books.")
        (self.root / "research" / "library-guide.md").write_text("The original changed.")
        evidence = collect_summary(retriever, "research", self.root,
                                   "Summarize library-guide.md.", {"source": "library-guide.md"})
        self.assertEqual(evidence.passages, ())

    def test_quoted_identity_and_filename_suffix_are_not_stripped(self):
        self.assertEqual(named_document_reference('Summarize "library-guide.md.".').title, "library-guide.md.")
        self.assertEqual(named_document_reference("Summarize library-guide.md.backup.").title,
                         "library-guide.md.backup.")
        self.assertEqual(named_document_reference("Summarize library-guide.md!").title, "library-guide.md")

    def test_deictic_summary_requires_selected_source_scope(self):
        for query in ("Summarize this document.", "סכם את המסמך הזה."):
            with self.subTest(query=query):
                source_id = "s" * 43
                selected = understand_request(query, selected_source_id=source_id).plan
                contract = build_request_execution_contract(query, plan=selected, source_scope=(source_id,))
                self.assertEqual(contract.requirements[0].kind, "document_summary")
                self.assertEqual(contract.source_scope, (source_id,))
                unselected = understand_request(query).plan
                unselected_contract = build_request_execution_contract(query, plan=unselected)
                self.assertNotEqual(unselected_contract.requirements[0].kind, "document_summary")


if __name__ == "__main__":
    unittest.main()

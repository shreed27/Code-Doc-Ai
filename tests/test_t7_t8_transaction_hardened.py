"""
Hardened adversarial tests for T7 and T8 transaction integrity (Priority 6).

T7: Pre-replacement journal registration and interruption after os.replace:
- Reworked commit sequence ensures entry.state = 'replacing' before os.replace.
- Failure immediately after os.replace triggers rollback restoring destination from backup.
- If rollback fails, the backup file is retained on disk and reported in unrecovered_backups.
- Error message never falsely claims successful rollback.

T8: Cleanup failure after successful commit:
- Decouples cleanup outcome from commit success.
- If document replacement succeeds, unlink failure in cleanup does not abort the transaction or mark docs as unupdated.
- tx.cleanup_warnings reports leftover files.
- tx.retry_cleanup() retries deleting leftovers.
- No unhandled traceback is emitted.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from doc_updater import (
    DocumentTransaction,
    DocumentTransactionError,
    DocumentationUpdater,
)


class TestT7T8TransactionHardened:
    def test_t7_interruption_immediately_after_os_replace_restores_from_backup(self, tmp_path, monkeypatch):
        """
        T7: Interruption immediately after os.replace returns for doc1.
        Entry state was pre-registered as 'replacing'.
        Rollback restores doc1 from pre-replacement backup.
        """
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Doc 1\nOriginal 1\n", encoding="utf-8")
        doc2.write_text("# Doc 2\nOriginal 2\n", encoding="utf-8")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates(
            {
                "doc1.md": "# Doc 1\nUpdated 1\n",
                "doc2.md": "# Doc 2\nUpdated 2\n",
            }
        )

        real_replace = os.replace
        replace_count = 0

        def failing_replace(src, dst):
            nonlocal replace_count
            replace_count += 1
            res = real_replace(src, dst)
            # Immediately after the first replace succeeds, raise an exception
            if replace_count == 1:
                raise RuntimeError("Crash immediately after os.replace on doc1")
            return res

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(DocumentTransactionError) as exc_info:
            tx.commit()

        err = exc_info.value
        assert "rolled back all replaced documents" in str(err)
        # doc1 must be restored to original content!
        assert doc1.read_text(encoding="utf-8") == "# Doc 1\nOriginal 1\n"
        assert doc2.read_text(encoding="utf-8") == "# Doc 2\nOriginal 2\n"

    def test_t7_interruption_with_failed_rollback_preserves_backup_on_disk(self, tmp_path, monkeypatch):
        """
        T7 with failed rollback:
        If rollback also fails, the backup file MUST remain on disk for manual recovery
        and must be reported in unrecovered_backups.
        Error message must state 'could NOT restore all files'.
        """
        doc1 = tmp_path / "doc1.md"
        doc1.write_text("# Doc 1\nOriginal Content\n", encoding="utf-8")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({"doc1.md": "# Doc 1\nNew Content\n"})

        real_replace = os.replace
        call_count = 0

        def failing_replace(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First replace succeeds (dest now has new content) but error raised
                real_replace(src, dst)
                raise RuntimeError("Interrupted right after replace")
            elif call_count == 2:
                # Rollback attempt fails!
                raise OSError("Disk I/O error during rollback")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(DocumentTransactionError) as exc_info:
            tx.commit()

        err = exc_info.value
        assert "could NOT restore all files" in str(err)
        assert len(err.unrecovered_backups) == 1
        failed_dest, backup_file = err.unrecovered_backups[0]
        assert failed_dest == doc1
        assert backup_file.exists()
        # The backup contains the original content!
        assert backup_file.read_text(encoding="utf-8") == "# Doc 1\nOriginal Content\n"

    def test_t8_cleanup_failure_does_not_abort_successful_commit(self, tmp_path, monkeypatch):
        """
        T8: Document replacement succeeds, but cleanup encounters an OSError.
        - Documents on disk MUST remain updated and committed.
        - tx.committed MUST be True.
        - No unhandled exception raised.
        - tx.cleanup_warnings captures the failed cleanup path.
        - tx.retry_cleanup() successfully cleans up leftovers once issue is resolved.
        """
        doc = tmp_path / "doc.md"
        doc.write_text("# Doc\nOriginal\n", encoding="utf-8")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({"doc.md": "# Doc\nCommitted New Content\n"})

        real_unlink = Path.unlink
        fail_unlink = True

        def unlink_with_failure(path_obj, missing_ok=False):
            if fail_unlink and ".bak_doc_" in path_obj.name:
                raise OSError("Permission denied unlinking backup in cleanup")
            return real_unlink(path_obj, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink_with_failure)

        # Commit must complete without raising an error!
        tx.commit()

        assert tx.committed is True
        assert doc.read_text(encoding="utf-8") == "# Doc\nCommitted New Content\n"
        assert len(tx.cleanup_warnings) >= 1
        failed_bak_path, err_msg = tx.cleanup_warnings[0]
        assert failed_bak_path.exists()
        assert "Permission denied" in err_msg

        # Now allow unlinking and retry cleanup:
        fail_unlink = False
        remaining = tx.retry_cleanup()
        assert len(remaining) == 0
        assert not failed_bak_path.exists()

    def test_t8_cleanup_failure_in_documentation_updater_workflow(self, tmp_path, monkeypatch):
        """
        T8 integration: DocumentationUpdater update_documentation run with cleanup failure
        still marks documents as updated and committed.
        """
        doc = tmp_path / "guide.md"
        doc.write_text("# Guide\n\n## Connect\nCall `connect()`.\n", encoding="utf-8")

        old_code = "def connect(): pass"
        new_code = "def connect(retries: int = 3): pass"

        def mock_ai(_sys, _user):
            import json

            return json.dumps(
                {
                    "proposed_content": "Call `connect(retries)` with optional retries (default 3).",
                    "explanation": "Added retries doc",
                    "requires_human_review": False,
                }
            )

        updater = DocumentationUpdater(ai_generator=mock_ai, docs_root=tmp_path)

        fail_bak_unlink = True
        real_unlink = Path.unlink

        def unlink_wrapper(path_obj, missing_ok=False):
            if fail_bak_unlink and ".bak_doc_" in path_obj.name:
                raise OSError("Simulated cleanup lock")
            return real_unlink(path_obj, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", unlink_wrapper)

        report = updater.update_documentation(
            code_changes={"src/net.py": (old_code, new_code)},
            documents={"guide.md": doc.read_text(encoding="utf-8")},
        )

        assert report.total_sections_applied == 1
        assert "retries" in report.updated_documents["guide.md"]
        assert doc.read_text(encoding="utf-8") == report.updated_documents["guide.md"]

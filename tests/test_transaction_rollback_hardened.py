"""
Hardened adversarial tests for DocumentTransaction rollback, permissions, and integrity (Priority 3).
"""

import os

import pytest

from doc_updater import (
    DocumentTransaction,
    DocumentTransactionError,
    PathTraversalSecurityError,
)


class TestDocumentTransactionRollbackHardened:
    def test_permission_preservation_0640(self, tmp_path):
        doc1 = tmp_path / "doc1.md"
        doc1.write_text("# Doc 1\nOriginal\n")
        os.chmod(doc1, 0o640)

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates({"doc1.md": "# Doc 1\nUpdated\n"})
        tx.commit()

        assert doc1.read_text() == "# Doc 1\nUpdated\n"
        mode = doc1.stat().st_mode & 0o777
        assert mode == 0o640

    def test_keyboard_interrupt_base_exception_triggers_rollback(self, tmp_path, monkeypatch):
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Doc 1\nOriginal 1\n")
        doc2.write_text("# Doc 2\nOriginal 2\n")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates(
            {
                "doc1.md": "# Doc 1\nNew 1\n",
                "doc2.md": "# Doc 2\nNew 2\n",
            }
        )

        real_replace = os.replace
        count = 0

        def interrupt_replace(src, dst):
            nonlocal count
            count += 1
            if count == 2:
                raise KeyboardInterrupt("Simulated Ctrl-C during commit")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", interrupt_replace)

        with pytest.raises(DocumentTransactionError, match="rolled back all replaced documents"):
            tx.commit()

        # Both documents must be in their original state!
        assert doc1.read_text() == "# Doc 1\nOriginal 1\n"
        assert doc2.read_text() == "# Doc 2\nOriginal 2\n"

    def test_rollback_failure_retains_surviving_backups(self, tmp_path, monkeypatch):
        """
        If a failure occurs during rollback itself, the surviving backup files
        must NOT be deleted by cleanup() and must be reported in unrecovered_backups.
        """
        doc1 = tmp_path / "doc1.md"
        doc2 = tmp_path / "doc2.md"
        doc1.write_text("# Doc 1\nOriginal 1\n")
        doc2.write_text("# Doc 2\nOriginal 2\n")

        tx = DocumentTransaction(docs_root=tmp_path)
        tx.stage_updates(
            {
                "doc1.md": "# Doc 1\nNew 1\n",
                "doc2.md": "# Doc 2\nNew 2\n",
            }
        )

        real_replace = os.replace
        replace_call = 0

        def failing_replace(src, dst):
            nonlocal replace_call
            replace_call += 1
            # Call 1: doc1 new -> doc1.md
            # Call 2: fails on doc2
            if replace_call == 2:
                raise OSError("Replacement error on doc2")
            # Call 3: rollback doc1.bak -> doc1.md fails!
            if replace_call == 3:
                raise OSError("Rollback disk error on doc1")
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)

        with pytest.raises(DocumentTransactionError) as exc_info:
            tx.commit()

        err = exc_info.value
        assert "could NOT restore all files" in str(err)
        assert len(err.unrecovered_backups) >= 1

        # The backup file must still exist on disk for manual recovery!
        _, backup_path = err.unrecovered_backups[0]
        assert backup_path.exists()

    def test_symlink_traversal_detection(self, tmp_path):
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret_file = outside_dir / "secret.md"
        secret_file.write_text("secret")

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()

        # Symlink pointing outside docs_dir
        symlink_path = docs_dir / "leak.md"
        symlink_path.symlink_to(secret_file)

        tx = DocumentTransaction(docs_root=docs_dir)
        with pytest.raises(PathTraversalSecurityError):
            tx.stage_updates({"leak.md": "overwrite"})

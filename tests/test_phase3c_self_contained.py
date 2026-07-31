#!/usr/bin/env python3
"""Negative tests for the Phase-3C self-contained dependency audit."""
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.validate_phase3c_self_contained import AuditFailure, QUALIFIED, audit

ROOT = Path(__file__).resolve().parents[1]


class SelfContainedNegativeTests(unittest.TestCase):
    def fixture(self, content="#include \"local.h\"\n", local=True):
        temporary = tempfile.TemporaryDirectory(prefix="phase3c_audit_test_")
        root = Path(temporary.name)
        qualified = root / QUALIFIED
        qualified.parent.mkdir(parents=True)
        shutil.copy2(ROOT / QUALIFIED, qualified)
        source = root / "probe.cpp"
        source.write_text(content)
        if local:
            (root / "local.h").write_text("// local\n")
        return temporary, root

    def test_n1_external_dotdot_include_rejected(self):
        temp, root = self.fixture('#include "../../workspace-sibling/file.h"\n', False)
        with temp, self.assertRaises(AuditFailure):
            audit(root, ("probe.cpp",))

    def test_n2_absolute_home_include_rejected(self):
        temp, root = self.fixture('#include "/home/test-user/external.h"\n', False)
        with temp, self.assertRaises(AuditFailure):
            audit(root, ("probe.cpp",))

    def test_n3_external_symlink_rejected(self):
        temp, root = self.fixture()
        outside = Path(temp.name).parent / "phase3c_external_target"
        outside.write_text("external\n")
        try:
            os.symlink(outside, root / "external-link")
            with temp, self.assertRaises(AuditFailure):
                audit(root, ("probe.cpp",))
        finally:
            outside.unlink(missing_ok=True)

    def test_n4_missing_qualified_source_rejected(self):
        temp, root = self.fixture()
        (root / QUALIFIED).unlink()
        with temp, self.assertRaisesRegex(AuditFailure, "missing vendored"):
            audit(root, ("probe.cpp",))

    def test_n5_mutated_qualified_source_rejected(self):
        temp, root = self.fixture()
        with (root / QUALIFIED).open("ab") as stream:
            stream.write(b"\n// mutation\n")
        with temp, self.assertRaisesRegex(AuditFailure, "hash mismatch"):
            audit(root, ("probe.cpp",))

    def test_n6_workspace_sibling_required_rejected(self):
        temp, root = self.fixture('#include "../../phase3a_fused_backward/file.h"\n', False)
        with temp, self.assertRaisesRegex(AuditFailure, "leaves repository|workspace sibling"):
            audit(root, ("probe.cpp",))

    def test_n7_real_checkout_without_sibling_passes(self):
        result = audit(ROOT)
        self.assertEqual(result["gate"], "PHASE3C_SELF_CONTAINED_SOURCE_AUDIT: PASS")

    def test_n8_v1_tag_cannot_be_interpreted_as_valid_pass(self):
        metadata = json.loads(
            (ROOT / "freeze_phase3c_failures/PHASE3C_FREEZE_V1_FAILURE_METADATA.json").read_text())
        self.assertEqual(metadata["failed_tag"], "phase3c-portable-smoke-gfx1201-pass")
        self.assertEqual(metadata["disposition"], "SUPERSEDED_BY_V2_ONLY_AFTER_FULL_PASS")
        tag_object = subprocess.check_output(
            ["git", "rev-parse", "phase3c-portable-smoke-gfx1201-pass"], cwd=ROOT, text=True).strip()
        self.assertEqual(tag_object, metadata["failed_tag_object"])
        self.assertEqual(metadata["failed_gate"], "PHASE3C_ARCHIVE_FRESH_CLONE_VALIDATION")


if __name__ == "__main__":
    unittest.main(verbosity=2)

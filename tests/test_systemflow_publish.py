import json
import tempfile
import unittest
from pathlib import Path

from agents.systemflow_integration.runner import publish_systemflow_application_model


class SystemFlowPublishTests(unittest.TestCase):
    def test_publish_systemflow_application_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            systemflow_root = tmp_path / "systemflow-project"
            package = systemflow_root / "systemflow"
            package.mkdir(parents=True)
            (package / "application_models.py").write_text("# runtime\n", encoding="utf-8")

            sources = tmp_path / "artifacts"
            sources.mkdir()
            model = sources / "workflow_application_resource_model.v002.json"
            mapping = sources / "systemflow_application_mapping.v001.yaml"
            report = sources / "systemflow_integration_report.v002.yaml"
            model.write_text('{"schema_version": "0.2"}\n', encoding="utf-8")
            mapping.write_text("schema_version: '0.1'\n", encoding="utf-8")
            report.write_text("status: approved\n", encoding="utf-8")

            published = publish_systemflow_application_model(
                systemflow_root,
                "pty-chi",
                model,
                mapping,
                report,
                scientific_use=False,
            )

            destination = package / "application_model_data" / "pty-chi"
            manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["application_id"], "pty-chi")
            self.assertIs(manifest["scientific_use"], False)
            self.assertEqual(manifest["assets"]["model"]["path"], model.name)
            self.assertEqual((destination / model.name).read_bytes(), model.read_bytes())
            self.assertEqual(
                (destination / mapping.name).read_bytes(), mapping.read_bytes()
            )
            self.assertEqual((destination / report.name).read_bytes(), report.read_bytes())
            self.assertEqual(
                Path(published["manifest_path"]), (destination / "manifest.json").resolve()
            )

    def test_publish_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            systemflow_root = tmp_path / "systemflow-project"
            package = systemflow_root / "systemflow"
            package.mkdir(parents=True)
            (package / "application_models.py").write_text("# runtime\n", encoding="utf-8")
            model = tmp_path / "model.v001.json"
            mapping = tmp_path / "mapping.v001.yaml"
            report = tmp_path / "report.v001.yaml"
            for path in (model, mapping, report):
                path.write_text(path.name, encoding="utf-8")

            arguments = (systemflow_root, "app", model, mapping, report)
            first = publish_systemflow_application_model(*arguments, scientific_use=True)
            second = publish_systemflow_application_model(*arguments, scientific_use=True)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])

            model.write_text("new model content", encoding="utf-8")
            updated = publish_systemflow_application_model(*arguments, scientific_use=True)
            self.assertNotEqual(
                first["assets"]["model"]["path"],
                updated["assets"]["model"]["path"],
            )

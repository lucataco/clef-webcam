"""Exercise the real HTTP routes without loading or downloading model weights."""

import base64
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class SystemOneTests(unittest.TestCase):
    def setUp(self):
        self.body = {"model": "clef-flash", "state": "I was charged twice.", "questions": {
            "duplicate": {"type": "noul"},
            "team": {"type": "choice", "criteria": {"billing": None, "support": None}},
            "urgency": {"type": "score", "criteria": ["Routine", "Urgent"]}}}
        self.response = {"model": "clef-flash", "answers": {
            "duplicate": {"type": "noul", "noul": 0.8},
            "team": {"type": "choice", "choice": "billing", "confidence": 0.9,
                     "probabilities": {"billing": 0.9, "support": 0.1}},
            "urgency": {"type": "score", "score": 0.6, "confidence": 0.6,
                        "probabilities": {"0": 0.4, "1": 0.6}, "legend": {"0": "Routine", "1": "Urgent"}}},
                         "usage": {"input_tokens": 100, "output_tokens": 0}}
        self.clef = SimpleNamespace(decide=Mock(return_value=(self.response, 12.0)),
                                    hardware="fixture", device="fixture")
        constructor = Mock(return_value=self.clef)
        spec = importlib.util.spec_from_file_location("test_server", ROOT / "server.py")
        self.server = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"clef": SimpleNamespace(Clef=constructor)}), \
                patch.dict(os.environ, {"CLEF_MAX_SIDE": "336"}), redirect_stdout(io.StringIO()):
            spec.loader.exec_module(self.server)
        constructor.assert_called_once_with()
        self.clef.decide.reset_mock()  # Ignore the two existing startup warmups.
        self.client = TestClient(self.server.app)
        self.addCleanup(self.client.close)

    def image(self, image_format="PNG", color="orange", size=(640, 480)):
        output = io.BytesIO()
        Image.new("RGB", size, color).save(output, format=image_format)
        return base64.b64encode(output.getvalue()).decode()

    def test_text_json_object_and_array_states_return_the_helper_response(self):
        for state in ("A ticket", {"ticket": "A ticket"}, [{"ticket": "A ticket"}]):
            with self.subTest(state=state):
                body = dict(self.body, state=state)
                response = self.client.post("/v1/systemone", json=body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), self.response)
                self.assertEqual(response.headers["Server-Timing"], "inference;dur=12.0")
                self.assertEqual(response.headers["X-Clef-Hardware"], "fixture")
                self.assertEqual(response.headers["X-Clef-Device"], "fixture")
                self.clef.decide.assert_called_with(body)

    def test_omitted_and_null_instructions_preserve_the_wire_request(self):
        for name in self.body["questions"]:
            for instructions in ({}, {"instructions": None}):
                with self.subTest(name=name, instructions=instructions):
                    body = deepcopy(self.body)
                    body["questions"][name].update(instructions)
                    response = self.client.post("/v1/systemone", json=body)
                    self.assertEqual(response.status_code, 200)
                    self.clef.decide.assert_called_with(body)

    def test_nullable_noul_criteria_and_descriptions_preserve_the_wire_request(self):
        for criteria in (None, {}, {"true": None}, {"false": None}, {"true": None, "false": None},
                         {"true": ["Matches"], "false": {"reason": "Does not match"}}):
            with self.subTest(criteria=criteria):
                body = deepcopy(self.body)
                body["questions"]["duplicate"]["criteria"] = criteria
                response = self.client.post("/v1/systemone", json=body)
                self.assertEqual(response.status_code, 200)
                self.clef.decide.assert_called_with(body)

    def test_structured_instructions_and_single_level_score_are_accepted(self):
        body = deepcopy(self.body)
        body["questions"]["duplicate"]["instructions"] = {"task": "Duplicate charge?"}
        body["questions"]["team"]["instructions"] = ["Choose a team"]
        body["questions"]["urgency"]["criteria"] = ["Routine"]
        self.assertEqual(self.client.post("/v1/systemone", json=body).status_code, 200)
        self.clef.decide.assert_called_once_with(body)

    def test_inline_images_are_decoded_and_resized_without_fetching_urls(self):
        for image_format in ("PNG", "JPEG", "WEBP"):
            encoded = self.image(image_format)
            for image in (encoded, f"data:image/{image_format.lower()};base64,{encoded}"):
                with self.subTest(format=image_format, data_url=image.startswith("data:")):
                    response = self.client.post("/v1/systemone", json=dict(self.body, images=[image]))
                    self.assertEqual(response.status_code, 200)
                    record = self.clef.decide.call_args.args[0]
                    self.assertEqual(record["state"], self.body["state"])
                    self.assertEqual(record["images"][0].mode, "RGB")
                    self.assertEqual(record["images"][0].size, (336, 252))

    def test_mpo_jpeg_uses_the_first_frame(self):
        image = Image.new("RGB", (640, 480))
        image.format = "MPO"
        with patch("systemone_api.Image.open", return_value=image):
            response = self.client.post("/v1/systemone", json=dict(self.body, images=["eA=="]))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.clef.decide.call_args.args[0]["images"][0].size, (336, 252))

    def test_invalid_requests_return_422_before_inference(self):
        bodies = [[], {}, dict(self.body, model="jev-latest"), dict(self.body, state=None),
                  dict(self.body, questions={}),
                  dict(self.body, questions={str(i): {"type": "noul"} for i in range(65)}),
                  dict(self.body, images=["https://example.com/image.png"]),
                  dict(self.body, images=["/tmp/image.png"]), dict(self.body, images=["not-base64"]),
                  dict(self.body, images=["eA=="]), dict(self.body, images=["eA=="] * 5),
                  dict(self.body, images=["x" * (8 * 1024 * 1024 + 1)])]
        bodies.extend(dict(self.body, questions={"invalid": question}) for question in (
            {"type": "unknown"}, {"type": "noul", "instructions": 42},
            {"type": "noul", "criteria": {"wrong": None}}, {"type": "noul", "criteria": {"true": False}},
            {"type": "choice", "criteria": None}, {"type": "choice", "criteria": {}},
            {"type": "score", "criteria": None}, {"type": "score", "criteria": []}))
        for index, body in enumerate(bodies):
            with self.subTest(case=index):
                self.assertEqual(self.client.post("/v1/systemone", json=body).status_code, 422)
        self.clef.decide.assert_not_called()

    def test_non_finite_json_is_rejected_before_inference(self):
        body = dict(self.body, state={"invalid": float("nan")})
        response = self.client.post("/v1/systemone", content=json.dumps(body),
                                    headers={"Content-Type": "application/json"})
        self.assertEqual(response.status_code, 422)
        self.clef.decide.assert_not_called()

    def test_nested_non_finite_validation_inputs_return_serializable_errors(self):
        nested = {"invalid": [float("nan"), {"value": float("inf")}], "finite": 1.5}
        bodies = [[nested], dict(self.body, questions={"invalid": {"type": "unknown", **nested}})]
        with TestClient(self.server.app, raise_server_exceptions=False) as client:
            for body in bodies:
                with self.subTest(body=body):
                    response = client.post("/v1/systemone", content=json.dumps(body),
                                           headers={"Content-Type": "application/json"})
                    self.assertEqual(response.status_code, 422)
                    json.dumps(response.json(), allow_nan=False)
                    reported = response.json()["detail"][0]["input"]
                    if isinstance(reported, list):
                        reported = reported[0]
                    self.assertEqual(reported["invalid"], ["nan", {"value": "inf"}])
                    self.assertEqual(reported["finite"], 1.5)
        self.clef.decide.assert_not_called()

    def test_model_validation_errors_return_422(self):
        self.clef.decide.side_effect = ValueError("schema does not fit")
        self.assertEqual(self.client.post("/v1/systemone", json=self.body).status_code, 422)

    def test_model_discovery_uses_typesafe_metadata(self):
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertEqual([model["name"] for model in models], ["clef-flash"])
        self.assertEqual(set(models[0]), {"name", "description", "release_date"})
        self.clef.decide.assert_not_called()

    def test_webcam_and_api_routes_share_the_same_loaded_model(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        response = self.client.post("/decide", json={"image": self.image(), "questions": self.body["questions"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["latency_ms"], 12.0)
        self.assertEqual(response.json()["device"], "fixture")
        self.assertEqual(self.client.post("/v1/systemone", json=self.body).json(), self.response)
        self.assertEqual(self.clef.decide.call_count, 2)


if __name__ == "__main__":
    unittest.main()

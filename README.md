# clef-flash on your webcam

Run [clef-flash](https://huggingface.co/Cloudflare/clef-flash), Cloudflare's open-weight decision model, locally on a Mac and turn your webcam into typed decisions with probabilities.

A decision model doesn't generate text. You give it a state (here, a webcam frame) and a schema of questions, and it returns a probability for every allowed answer in one forward pass. [Read the announcement](https://blog.cloudflare.com/clef-decision-models/).

## Quickstart

You need an Apple Silicon Mac with 32 GB+ of memory (the model is ~19 GB) and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run hf download Cloudflare/clef-flash --local-dir models/clef-flash
uv run uvicorn server:app --port 8787
```

Open http://localhost:8787 and allow camera access.

The camera page sends its decisions to `/v1/systemone`. **Two-frame motion** is
enabled by default to compare an earlier and later frame, with a prompt that looks
for movement such as mouth/lip changes consistent with talking. It uses a rolling
pair spaced at least 250 ms apart. Uncheck it to send one current frame. The mode choice is
remembered locally, including an existing single-frame opt-out; edited questions
remain intact. Pausing, schema edits, mode changes, and errors clear the pair.
Frames older than two seconds are replaced, and a stalled camera cannot supply
both frames of a clip. Capture waits for newly presented video frames, and the
frame rate reflects their source timestamps.

## System One API

The same server exposes `POST /v1/systemone` and `GET /v1/models` for
[TypeSafe / System One clients](https://docs.typesafe.ai/sdk/python). API requests
and webcam requests share one loaded model and one inference lock.

```sh
curl http://127.0.0.1:8787/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{"model":"clef-flash","state":"I was charged twice.","questions":{"duplicate_charge":{"type":"noul"},"team":{"type":"choice","criteria":{"billing":null,"support":null}}}}'
```

Responses use the model helper's System One shape: `model`, `answers`, and `usage`.
Model timing is returned in `Server-Timing: inference;dur=...`, with hardware and
device in `X-Clef-Hardware` and `X-Clef-Device` headers, so the camera page can
display these without adding fields to the System One response. State may be
text, a JSON object, or an array. Instructions are optional; omitted or null
instructions fall back to the question name. Noul, choice, and score questions
work, including nullable noul criteria and outcome descriptions. Invalid requests
return HTTP 422.

Use `http://127.0.0.1:8787` as the TypeSafe SDK's `base_url`, set
`model="clef-flash"`, and supply a placeholder `api_key` such as `"local-unused"`
if the client requires one. This local server does not require an API key.

With the optional `typesafe-sdk` client installed in your client environment:

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

with TypeSafeClient(base_url="http://127.0.0.1:8787",
                    api_key="local-unused", model="clef-flash") as client:
    result = client.system_one(
        state="I was charged twice.",
        questions={"duplicate_charge": Noul(),
                   "team": Choice(criteria={"billing": None, "support": None})},
    )
    print(result.nouls["duplicate_charge"].noul)
    print(result.choices["team"].choice)
```

For image questions, add `images` with up to four inline base64 JPEG, PNG, or
WebP images (raw base64 or data URLs, at most 8 MiB encoded per image). Images
are resized to `CLEF_MAX_SIDE`, like webcam frames. Image URLs and file paths
are not fetched.

The local video extension accepts one two-frame clip:

```json
{
  "model": "clef-flash",
  "state": "Compare two chronological frames for visible talking motion.",
  "questions": {"talking": {"type": "noul"}},
  "videos": [{"frames": ["<earlier frame base64>", "<later frame base64>"], "fps": 4.0}]
}
```

Use the same inline image formats and limits for each frame. Frame dimensions
must match after resizing. `fps` is their effective sampling rate (4 means
250 ms apart), defaults to 2, and must be greater than 0 and at most 120. The
processor receives those selected frames without resampling, plus metadata for
their timestamps. Encoded video files and longer clips are not accepted.

The model's image processor duplicates a single image across the two temporal
slices of a patch. Sending two chronological frames as one video supplies both
slices with actual observations instead. At matching resolution, this can use
the same visual token budget as one image; timestamp and prompt text, frame
capture, and transfer still add work. Compare model latency and
`usage.input_tokens` rather than assuming every two-frame request is free.
This supplies motion information; it does not establish talking-detection accuracy.

Keep the server bound to loopback; it has no authentication.

## Controls

| Key | Action |
| --- | --- |
| `E` | Edit the questions. They apply live, no retraining |
| `space` | Pause |
| `shift` + `R` | Reset to the default questions |

## Asking your own questions

Questions use the same schema as `@cf/cloudflare/clef` on Workers AI (and Jev / SystemOne):

```json
{
  "wearing_glasses": { "type": "noul", "instructions": "Wearing glasses?" },
  "mood": { "type": "choice", "criteria": { "happy": null, "neutral": null, "tired": null } },
  "energy": { "type": "score", "criteria": ["low", "medium", "high"] }
}
```

- `noul`: yes or no
- `choice`: pick one option. Descriptions are optional (`null`)
- `score`: an ordered scale

Latency grows with prompt length, so short instructions and few options keep it snappy.

## Performance

Measured on an M5 Max with `uv run python bench.py`, which also checks that the optimized model gives the same answers as the reference:

| Request | transformers on MPS | This repo |
| --- | --- | --- |
| Text, 3 questions | 566 ms | 155 ms |
| Webcam frame, 3 questions | 840 ms | 250 ms |

On a Mac, Qwen3.5's linear-attention layers fall back to a slow reference implementation, because the fast kernels are CUDA-only. Most of that time goes to `torch.linalg.solve_triangular`. [`mps_kernels.py`](mps_kernels.py) computes the same thing with batched block inversion, which Apple GPUs handle well. It is patched in automatically on MPS and CPU, and left alone on CUDA.

## Files

| File | |
| --- | --- |
| `server.py` | API: webcam frame + questions → decisions |
| `index.html` | The live demo page |
| `clef.py` | Model loading and timing |
| `mps_kernels.py` | Faster linear attention for Apple Silicon |
| `systemone_api.py` | System One HTTP schema, image decoding, and model discovery |
| `camera.mjs` | Single-frame and two-frame capture state |
| `bench.py` | Speed and accuracy check |

`joint_schema_model.py` (Clef's reference inference code) is downloaded with the model.

## Tests

```sh
uv run python -m unittest discover -s tests
node --test tests/test_camera.mjs
```

HTTP tests use a fake model and do not download or load weights. The capture-state
tests use Node's built-in test runner.

## Settings

| Variable | Default | |
| --- | --- | --- |
| `CLEF_MODEL_DIR` | `models/clef-flash` | Where the model lives |
| `CLEF_MAX_SIDE` | `336` | Longest side of the frame sent to the model, in pixels |

## License

This code is [Apache 2.0](LICENSE) licensed, and so is the [clef-flash model](https://huggingface.co/Cloudflare/clef-flash).

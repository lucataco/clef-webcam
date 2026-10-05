import assert from "node:assert/strict";
import test from "node:test";
import { CaptureState, SYSTEMONE_URL, SINGLE_STATE, MOTION_STATE, captureFrame, motionPreference, inferenceTiming } from "../camera.mjs";

const questions = { talking: { type: "noul", instructions: "Talking?" } };
const frame = (data, at, mediaTime = at) => ({ data, at, mediaTime });

test("capture waits for a new source frame and records its actual presentation timestamps", async () => {
  let callback, draws = 0;
  const video = { requestVideoFrameCallback: fn => { callback = fn; return 1; } };
  const result = captureFrame(video, () => { draws++; return "jpeg"; });
  await Promise.resolve();
  assert.equal(draws, 0);
  callback(1500, { mediaTime: 1.25 });
  assert.deepEqual(await result, { data: "jpeg", at: 1500, mediaTime: 1250 });
  assert.equal(draws, 1);
});

test("a stopped source times out and cancels its pending frame callback without copying", async () => {
  let canceled;
  const video = { requestVideoFrameCallback: () => 42, cancelVideoFrameCallback: id => canceled = id };
  assert.equal(await captureFrame(video, () => assert.fail("no new frame to copy"), 1), null);
  assert.equal(canceled, 42);
});

test("capture errors reject so the loop can reset history and retry", async () => {
  let callback;
  const video = { requestVideoFrameCallback: fn => { callback = fn; return 1; } };
  const result = captureFrame(video, () => { throw new Error("failed to draw frame"); });
  callback(1000, { mediaTime: 1 });
  await assert.rejects(result, /failed to draw frame/);
});

test("unset preferences default to two frames and explicit saved choices survive", () => {
  for (const stored of [null, "true", "false"]) {
    const capture = new CaptureState(motionPreference(stored));
    assert.equal(capture.motion, stored !== "false");
    assert.equal(capture.prepare(frame("first", 1000), questions) === null, stored !== "false");
  }
  assert.equal(new CaptureState().motion, true);
});

test("single-frame mode uses the System One image request and preserves edited questions", () => {
  const capture = new CaptureState(false);
  assert.equal(SYSTEMONE_URL, "/v1/systemone");
  assert.deepEqual(capture.prepare(frame("current", 100), questions), {
    model: "clef-flash", state: SINGLE_STATE, questions, images: ["current"]
  });
  assert.equal(capture.waitMs(101), 0);
});

test("motion mode sends one two-frame video in chronological order with the measured frame rate", () => {
  const capture = new CaptureState();
  capture.setMotion(true);
  assert.equal(capture.prepare(frame("earlier", 1000), questions), null);
  assert.equal(capture.waitMs(1100), 150);
  assert.equal(capture.prepare(frame("too-close", 1100), questions), null);
  assert.deepEqual(capture.prepare(frame("later", 1250), questions), {
    model: "clef-flash", state: MOTION_STATE, questions,
    videos: [{ frames: ["earlier", "later"], fps: 4 }]
  });
  assert.deepEqual(capture.prepare(frame("next", 1750), questions).videos,
    [{ frames: ["later", "next"], fps: 2 }]);
  assert.equal(questions.talking.instructions, "Talking?");
});

test("pause, schema edits, and mode changes reset history and invalidate an old response", () => {
  const capture = new CaptureState();
  capture.setMotion(true);
  capture.prepare(frame("old", 1000), questions);
  const inFlightGeneration = capture.generation;
  capture.reset();
  assert.notEqual(capture.generation, inFlightGeneration);
  assert.equal(capture.prepare(frame("fresh", 3000), questions), null);
  capture.setMotion(false);
  assert.deepEqual(capture.prepare(frame("single", 3250), questions).images, ["single"]);
  capture.setMotion(true);
  assert.equal(capture.prepare(frame("new-start", 3500), questions), null);
});

test("stale or reversed frame times cannot be sent as a motion pair", () => {
  const capture = new CaptureState();
  capture.setMotion(true);
  capture.prepare(frame("old", 1000), questions);
  assert.equal(capture.prepare(frame("stale", 4000), questions), null);
  assert.equal(capture.prepare(frame("reversed", 3900), questions), null);
  assert.deepEqual(capture.prepare(frame("fresh", 4150), questions).videos,
    [{ frames: ["reversed", "fresh"], fps: 4 }]);
});

test("a stalled source cannot supply a repeated frame even after a long request", () => {
  const capture = new CaptureState();
  capture.prepare(frame("first", 1000, 500), questions);
  assert.equal(capture.prepare(frame("same-source", 1250, 500), questions), null);
  assert.equal(capture.prepare(frame("still-stalled", 4000, 500), questions), null);
  assert.equal(capture.prepare(frame("resumed", 4250, 3750), questions), null);
  assert.deepEqual(capture.prepare(frame("fresh", 4500, 4000), questions).videos,
    [{ frames: ["resumed", "fresh"], fps: 4 }]);
});

test("frame rate uses source timestamps and both sampling clocks enforce spacing", () => {
  const capture = new CaptureState();
  capture.prepare(frame("first", 1000, 100), questions);
  assert.equal(capture.prepare(frame("source-too-close", 1250, 300), questions), null);
  assert.deepEqual(capture.prepare(frame("later", 1500, 400), questions).videos,
    [{ frames: ["first", "later"], fps: 1000 / 300 }]);
  assert.equal(capture.prepare(frame("copy-too-close", 1600, 650), questions), null);
});

test("a restarted source and slow requests seed a fresh pair", () => {
  const capture = new CaptureState();
  capture.prepare(frame("first", 1000, 1000), questions);
  capture.prepare(frame("later", 1250, 1250), questions);
  assert.equal(capture.prepare(frame("after-slow-request", 4000, 4000), questions), null);
  assert.deepEqual(capture.prepare(frame("fresh", 4250, 4250), questions).videos,
    [{ frames: ["after-slow-request", "fresh"], fps: 4 }]);
  assert.equal(capture.prepare(frame("restarted", 4500, 0), questions), null);
  assert.deepEqual(capture.prepare(frame("next", 4750, 250), questions).videos,
    [{ frames: ["restarted", "next"], fps: 4 }]);
});

test("error recovery requires two fresh samples, including visually identical samples", () => {
  const capture = new CaptureState();
  capture.prepare(frame("before-error", 1000), questions);
  capture.prepare(frame("failed-request", 1250), questions);
  capture.reset();
  assert.equal(capture.prepare(frame("static-scene", 2000), questions), null);
  assert.deepEqual(capture.prepare(frame("static-scene", 2250), questions).videos,
    [{ frames: ["static-scene", "static-scene"], fps: 4 }]);
});

test("model timing remains separate from missing or malformed request timing", () => {
  assert.equal(inferenceTiming("inference;dur=251.2"), 251.2);
  assert.equal(inferenceTiming("queue;dur=10, inference;dur=120"), 120);
  for (const header of [null, "", "queue;dur=4", "inference;dur=-1", "inference;dur=NaN", "inference;dur=1.2.3"]) {
    assert.equal(inferenceTiming(header), null);
  }
});

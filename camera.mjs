// Capture helpers shared by the browser loop and its tests.
export const SYSTEMONE_URL = "/v1/systemone";
export const SINGLE_STATE = "A live webcam frame from a laptop.";
export const MOTION_STATE = "Two chronological webcam frames, earlier then later. Look for motion, especially mouth/lip changes consistent with talking.";

// An unset preference uses two frames; preserve the stored single-frame opt-out.
export const motionPreference = stored => stored !== "false";

export function captureFrame(video, draw, timeoutMs = 1000) {
  // Sample a newly presented source frame, not the interpolated playback clock.
  return new Promise((resolve, reject) => {
    const callback = video.requestVideoFrameCallback((at, metadata) => {
      clearTimeout(timeout);
      try {
        resolve({ data: draw(), at, mediaTime: metadata.mediaTime * 1000 });
      } catch (error) { reject(error); }
    });
    // A stopped source must not leave the loop stuck across pause/mode changes.
    const timeout = setTimeout(() => {
      video.cancelVideoFrameCallback(callback);
      resolve(null);
    }, timeoutMs);
  });
}

export class CaptureState {
  constructor(motion = true) {
    this.motion = !!motion;
    this.generation = 0;
    this.previous = null;
  }

  reset() {
    this.previous = null;
    this.generation++;
  }

  setMotion(enabled) {
    this.motion = !!enabled;
    this.reset();
  }

  waitMs(now) {
    return this.motion && this.previous ? Math.max(0, 250 - (now - this.previous.at)) : 0;
  }

  prepare(frame, questions) {
    const earlier = this.previous;
    // mediaTime identifies the source frame, independently of when we copy it.
    // A stalled camera must not become two frames by copying it again later.
    if (earlier && frame.mediaTime === earlier.mediaTime) return null;
    if (!this.motion) {
      this.previous = frame;
      return { model: "clef-flash", state: SINGLE_STATE, questions, images: [frame.data] };
    }
    if (!earlier || frame.at <= earlier.at || frame.at - earlier.at > 2000 ||
        frame.mediaTime <= earlier.mediaTime || frame.mediaTime - earlier.mediaTime > 2000) {
      this.previous = frame;
      return null;
    }
    const gap = frame.mediaTime - earlier.mediaTime;
    if (gap < 250 || frame.at - earlier.at < 250) return null;
    this.previous = frame;
    return { model: "clef-flash", state: MOTION_STATE, questions,
             videos: [{ frames: [earlier.data, frame.data], fps: 1000 / gap }] };
  }
}

export function inferenceTiming(header) {
  const match = (header || "").match(/(?:^|,)\s*inference\s*;\s*dur=([\d.]+)/);
  if (!match) return null;
  const value = Number(match[1]);
  return Number.isFinite(value) && value >= 0 ? value : null;
}

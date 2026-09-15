# Jora — AI Climbing Coach

*Upload a climbing video, get coached like you had a spotter.*

Jora is a multimodal AI system that analyzes climbing videos and delivers personalized
technique coaching — grounded in measured joint angles, not vibes. Built for GDG ML Hackweek 2026.

<!-- TODO: add a demo GIF here (upload -> analysis -> voice feedback). This is the single
     highest-impact addition to this README. -->

Upload a climbing video and Jora will:
1. **Detect your body pose** frame-by-frame using MediaPipe (33 landmarks, 7 joint angles)
2. **Classify climbing moves** from pose angle time-series using a custom-trained 1D CNN
3. **Score your technique** on a 5-dimension rubric (arm efficiency, hip positioning, foot precision, movement efficiency, body tension)
4. **Generate coaching feedback** via Gemini 2.5 Flash, grounded in the detected moves, rubric scores, and actual angle telemetry
5. **Synthesize voice coaching** with ElevenLabs TTS, presented with an audio-reactive visualizer

## Why we built it

Coaching is the bottleneck in climbing. Good technique feedback requires someone
experienced watching you climb, and most people training alone in a gym never get it —
they plateau without knowing which habit is holding them back.

Video is the obvious substitute, but "watch yourself back" only helps if you already know
what to look for. Jora closes that gap: it measures what your body actually did, scores it
against a rubric, and explains the fix out loud.

## Quick Start

**Prerequisites:** Python 3.12+, [uv](https://docs.astral.sh/uv/), ffmpeg

```bash
# Clone and install
git clone <repo-url>
cd GDG-ML-Hackweek
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys (see .env.example for details)

# Run
uv run python main.py
# Open http://localhost:8000
```

### Getting API Keys

| Service | Where | Free Tier |
|---|---|---|
| Google Gemini | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) | 1M tokens/day |
| ElevenLabs | [elevenlabs.io/app/settings/api-keys](https://elevenlabs.io/app/settings/api-keys) | ~10k chars/month |

Set `MOCK_TTS=true` in `.env` to develop without an ElevenLabs key.

### ffmpeg

Required for browser-compatible H.264 video output.

```bash
# Windows (winget)
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

## Architecture

```
Upload video → FastAPI backend → Background pipeline → Poll for results

Pipeline stages:
  Extract frames (OpenCV, every 10th frame)
  → Pose estimation (MediaPipe VIDEO mode, 33 landmarks)
  → Move classification (1D CNN on angle time-series)        ← custom ML
  → Rubric scoring (Ridge regression on angle stats + moves) ← custom ML
  → Gemini analysis (multimodal: frames + angles + moves + rubric → JSON)
  → TTS synthesis (ElevenLabs → MP3)
  → Assemble annotated video (ffmpeg H.264)
```

### Project Structure

```
├── main.py                         # Uvicorn entrypoint
├── app/
│   ├── api.py                      # FastAPI app factory
│   ├── config.py                   # Settings from .env
│   ├── pipeline/coach_pipeline.py  # Orchestrates all stages
│   ├── routers/                    # API endpoints
│   ├── services/                   # One per external dependency
│   ├── models/schemas.py           # Pydantic models
│   └── jobs/job_store.py           # In-memory job tracking
├── ml/                             # Training scripts & model definitions
│   ├── model.py                    # 1D CNN move classifier
│   ├── train_classifier.py         # Training loop
│   ├── rubric.py                   # Rubric definitions
│   └── train_rubric_model.py       # Rubric scoring model
├── data/                           # Datasets & trained models (git-ignored)
├── static/                         # Frontend (vanilla HTML/CSS/JS)
└── .env.example                    # Environment variable template
```

### API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload video, returns `{job_id}` |
| `GET` | `/api/jobs/{id}/status` | Poll progress: `{status, stage, progress_pct}` |
| `GET` | `/api/jobs/{id}/result` | Feedback + output URLs |
| `GET` | `/health` | Liveness check |

## ML Components

### Move Classifier

A 1D CNN trained on sliding windows of joint angle sequences extracted from climbing videos.

- **Input:** 15-frame windows of 21 features (7 angles + 7 angular velocities + 7 accelerations)
- **Classes:** static_reach, dyno, heel_hook, flag, deadpoint, rest_position, match, general_movement
- **Training data:** Hand-labeled angle sequences from ~15 climbing videos (~600 windows)

### Coaching Rubric

Five scoring dimensions rated 1-5, predicted by Ridge regression on angle statistics and move distributions:

| Dimension | Measures |
|---|---|
| Arm Efficiency | Straight-arm hanging, avoiding overgrip |
| Hip Positioning | Hips close to wall, weight over feet |
| Foot Precision | Quiet feet, deliberate placements |
| Movement Efficiency | Smooth flow, minimal wasted motion |
| Body Tension | Posterior chain engagement, controlled movement |

Rubric scores and few-shot examples are injected into the Gemini prompt to ground the feedback in measurable criteria.

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| Pose Estimation | MediaPipe PoseLandmarker |
| Move Classification | PyTorch 1D CNN |
| Rubric Scoring | scikit-learn Ridge Regression |
| LLM | Google Gemini 2.5 Flash |
| TTS | ElevenLabs |
| Video Processing | OpenCV + ffmpeg |
| Frontend | Vanilla HTML/CSS/JS + Canvas API |
| Package Manager | uv |

## Engineering challenges

**Keeping the LLM honest.** A vision model handed raw frames will happily invent
plausible-sounding climbing advice. We constrained it instead: the move classifier and
rubric model run first, and Gemini receives the detected moves, the rubric scores, and the
actual angle telemetry alongside the frames. The feedback has to be consistent with
numbers we measured, which makes it specific ("your hips drifted off the wall on the
deadpoint") rather than generic.

**Training a classifier on a hackathon-sized dataset.** We had ~15 hand-labeled videos,
about 600 windows. Rather than feed raw pixels to a hungry model, we reduced each frame to
7 joint angles and trained a small 1D CNN on 15-frame windows of angles plus their
velocities and accelerations. Working in angle space instead of pixel space made the model
learn from a dataset that size — and made it indifferent to camera position, lighting, and
what the climber was wearing.

**Browser-playable output.** MediaPipe and OpenCV happily write video that browsers refuse
to play. The pipeline re-encodes the annotated result to H.264 via ffmpeg so the result
plays inline instead of downloading.

## Development

```bash
# Install dependencies
uv sync

# Run with hot-reload
uv run python main.py

# Train move classifier (after labeling data)
uv run python ml/train_classifier.py

# Train rubric model (after scoring videos)
uv run python ml/train_rubric_model.py
```

### Environment Variables

See [`.env.example`](.env.example) for all configuration options with descriptions and links to get API keys.

## Team

Built by Sam and Abdullah for GDG ML Hackweek 2026.

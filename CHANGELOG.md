# Changelog

All notable changes to **Vaak** (वाक्) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.5.0] - 2026-09-07

### 🎨 Frontend Redesign & Forensic Listening Suite (Major Release)

Version 0.5.0 introduces a forensic speech analysis interface, an evidence inspection deck, real-time audio acquisition, and a complete design system overhaul matching the project's research rigor.

#### ✨ Added
* **Interactive Forensic Listening Console (`ListenSection`)**:
  * Real-time audio ingestion supporting WAV, MP3, FLAC, and OGG formats with automatic 16kHz mono normalization.
  * Live microphone recording studio with a circular reactive orbit visualizer (`RecordingState`) displaying smooth real-time frequency distribution bars and audio level metering.
  * Interactive synthetic waveform fallback synthesizer (`createFallbackAudioBlob`) allowing instant testing without external audio files.
  * State-machine lifecycle: `empty` → `recording` / `ready` → `processing` → `result` / `error`.
* **Acoustic Evidence Inspection Deck (`ResultState` & `EvidenceRail`)**:
  * Utterance-level forensic likelihood score (0.00–1.00) with confidence calibration badge (`LIKELY ORGANIC` vs `LIKELY SYNTHETIC`).
  * Dynamic playback scrub bar synchronizing playhead position with segmented waveform visualizations.
  * Temporal chunk-level probability rail displaying confidence distribution over time.
  * Forensic finding cards highlighting suspicious or organic audio segments with individual excerpt playback buttons.
* **Detector Reliability & Evaluation Visualizer (`EvaluationSection`)**:
  * Interactive Detection Error Tradeoff (DET) and score distribution curves.
  * Human speech profile vs synthetic speech profile density visualization.
  * Calibrated threshold callouts with ASVspoof 2019 baseline benchmarks.
* **Methodology & Research Architecture Deck (`MethodSection`)**:
  * Visual 4-step pipeline: Audio Ingestion → Self-Supervised WavLM Representations → Attentive Statistical Pooling (ASP) → Binary Likelihood Calibration.
* **Guiding Principles (`PrinciplesSection`)**:
  * Three foundational tenets: *An ear trained to notice*, *Calibrated confidence*, and *Private by design*.
* **Design System & Grid Perfection**:
  * Custom dark obsidian (`#111615`), warm off-white (`#f7f6f0`), and refined vermillion (`#e65f35` / `#e27c59`) color palette.
  * Sanskrit wordmark integration (`वाक्` in Noto Serif Devanagari).
  * Mathematically balanced 4-column footer grid (`1.5fr 0.85fr 1.15fr 1.15fr`) with identical card dimensions for technical telemetry and privacy policies.
  * 3-zone bottom bar with borderless GitHub repository destination link and Back-to-top button featuring in-place zoom micro-interactions (`scale(1.24)`).
* **API Integration & Server Versioning**:
  * Linked frontend to FastAPI `/predict` endpoint via multipart audio upload.
  * Memory-only processing pipeline with zero persistent audio storage.
  * Exposed package version `0.5.0` on `FastAPI(version=...)` and `/health` response.

#### 🔧 Developer Experience & How to Run
* Added streamlined commands in `Makefile`:
  * `make dev`: Concurrently start FastAPI backend (`:8000`) and Vite frontend (`:5173`).
  * `make frontend`: Run Vite frontend development server.
  * `make backend` / `make serve`: Run FastAPI inference server.
  * `make install`: Single command to sync Python virtualenv with `uv` and install npm packages.
* Synchronized version numbers to `0.5.0` across `pyproject.toml`, `package.json`, Python module metadata (`vaak.__version__`), and frontend UI badges.

---

## [0.4.0] - 2026-09-05

### 🔬 Architecture Ablations & Config Flexibility
* Added YAML-driven modular model factory (`build_model_from_config`).
* Implemented Layer Aggregation strategies (last-layer, mean across layers, learnable weighted sum).
* Added Attentive Statistical Pooling (ASP) backend alongside Mean Pooling.
* Standardized 2×2 WavLM ablation matrix (EXP-01 through EXP-04).

---

## [0.3.0] - 2026-09-03

### 📊 Experiment Tracking & Evaluation
* Integrated MLflow experiment tracker (`mlruns/` with SQLite backend).
* Utterance-level and chunk-level metric evaluation: Equal Error Rate (EER), minimum Detection Cost Function (minDCF), and ROC-AUC.
* Per-attack diagnostic logging for ASVspoof 2019 evaluation partitions.

---

## [0.2.0] - 2026-09-01

### 🧠 WavLM Baseline Pipeline
* Built frozen `microsoft/wavlm-base` self-supervised speech encoder.
* Binary classification linear head with dropout and weight decay.
* Audio chunking and framing with 4.0s duration and 2.0s hop stride.
* Initial React frontend scaffolding.

---

## [0.1.0] - 2026-08-28

### 🚀 Project Foundation
* Project scaffolding with modern Python 3.12+, `uv` dependency management, Ruff linting, mypy strict type checking, and pytest.
* Audio loading, mono conversion, 16kHz resampling, and peak normalization pipeline with `soundfile` and `torchaudio`.
* Canonical manifest builder and ASVspoof 2019 dataset parser.

# MotorGuard AI architecture

## Phase 0 audit summary

### Reusable as-is
- The existing preprocessing scripts in [preprocess.py](preprocess.py), [preprocess_cwru.py](preprocess_cwru.py), and [preprocess_faulty.py](preprocess_faulty.py) already capture the core idea of loading vibration data and preparing it for ML workflows.
- [train_gan.py](train_gan.py) and [train_classifier.py](train_classifier.py) provide a strong starting point for adversarial training and lightweight classification experiments.
- [train_vision_model.py](train_vision_model.py) establishes a YOLO-based vision training entry point.
- [rag.py](rag.py) already contains a Flask-facing RAG workflow that can be refactored into the prescribe layer.

### Needs refactoring
- The training scripts rely on hard-coded data paths, old naming conventions, and direct script execution rather than reusable modules.
- The current repository does not yet separate observe, diagnose, prescribe, edge, deployment, and test concerns into clearly defined modules.
- The architecture described in the master prompt introduces hardware abstraction, latency instrumentation, physics-aware GAN loss terms, late fusion, Monte Carlo Dropout, and local on-device LLM integration, which are not yet represented in the current code.

### Missing entirely versus the target architecture
- Camera and IMU acquisition abstractions with simulation mode.
- Butterworth filtering and CLAHE preprocessing for the observe layer.
- Bearing kinematics constants and reusable physics helpers.
- A physics-aware GAN training module with explicit PID loss and spectral-quality evaluation.
- A fusion LSTM with MC Dropout inference and confidence gating.
- A FAISS builder, prompt parser, and local llama.cpp integration.
- An orchestration layer, deployment packaging, and dedicated tests.

## Proposed project structure

- observe/: sensor acquisition, filtering, and preprocessing
- diagnose/vision/: YOLO training/export/inference
- diagnose/gan/: physics-aware GAN training and spectral evaluation
- diagnose/fusion/: late fusion model and uncertainty estimation
- prescribe/: retrieval, prompt construction, local LLM execution, and output validation
- edge/: orchestration and dashboard integration
- deploy/: ARM64 requirements, systemd files, and runtime packaging
- tests/: unit tests for the new modules

## Implemented in this pass

- Added a reusable bearing-kinematics helper with separate deployment and training constants.
- Implemented a lightweight late-fusion model with MC Dropout-style inference and a prescription parser.
- Wired the modules into an orchestration entrypoint that runs a simulated ODP loop.
- Added regression tests covering kinematics, fusion, prescription parsing, and orchestration.
- Added ARM64 requirements and a sample systemd service definition.

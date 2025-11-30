# Mosquito Audio Surveillance System

This document captures a high-level blueprint for a mosquito audio-surveillance prototype that ingests short audio clips, classifies mosquito species, and communicates vector-borne disease risks alongside tailored prevention tips.

## 1. Overall System Architecture

The system is organized into three cooperating services. They can run as separate modules or within a single Python backend, depending on scalability needs.

### A. Audio Processing & Species Classification
- **Input**: User uploads `.wav`/`.mp3` through the UI or an ingestion API. Clips shorter than roughly 5 seconds should be padded; longer clips can be windowed with overlaps to preserve temporal detail.
- **Pre-processing**:
  - Normalize audio to a common sampling rate (e.g., 16 kHz) and amplitude range.
  - Remove stationary background noise via spectral gating or bandpass filters tuned to 200-1,000 Hz.
  - Extract wing-beat cues using short-time Fourier transforms or constant-Q transforms. Focus on the 300-700 Hz band where species-specific harmonics reside.
  - Produce log-mel spectrograms or MFCCs for model input, optionally augmenting with time/frequency masking and pitch shifts to cover recording variability.
- **Feature Storage**: Cache processed tensors on disk or Redis with clip IDs so repeated model runs skip heavy Librosa work.
- **Classification Model**:
  - CNN or CRNN trained on HumBugDB, MosquitoSound, and synthetic wingbeat tones to fill species gaps.
  - Outputs categorical probabilities for target species such as *Aedes aegypti*, *Anopheles gambiae*, and *Culex quinquefasciatus*, plus an "other" bucket for out-of-distribution samples.
  - Serve the model through TorchScript, ONNX Runtime, or TensorFlow Serving to enable GPU acceleration when available.

### B. Risk Assessment Layer
- Maintains a lookup mapping from each species to the diseases it vectors and associated risk metadata (likelihood, incubation time, severity notes).
- Consumes the classifier output (top-1 or top-k) and produces human-friendly statements, e.g., "High confidence: *Aedes aegypti* (Dengue/Zika vector)."
- Can be extended with geospatial data (user location, known outbreaks) for contextual risk scoring.

### C. Safety & Prevention Recommendation Engine
- Provides actionable steps tailored to the predicted species: breeding-site removal tips for *Aedes*, indoor residual spraying notes for *Anopheles*, etc.
- Surfaces general bite-avoidance advice and medical escalation guidance when probabilities exceed configurable thresholds.
- Designed as a content service so health authorities can update messaging without retraining the classifier.

The architecture encourages clear contracts: the audio module emits `SpeciesPrediction` objects, the risk service enriches them, and the recommendation service turns enriched data into UI-ready narratives.

## 2. Technologies

| Layer | Recommended Tools | Notes |
| --- | --- | --- |
| Backend / ML | Python 3.11+, PyTorch or TensorFlow, Librosa, NumPy, FastAPI/Flask | PyTorch + TorchAudio simplifies spectrogram extraction; FastAPI exposes async inference endpoints. |
| Data & Storage | PostgreSQL (metadata), MinIO/S3 (audio blobs), Redis (feature cache) | Store audio securely; keep GDPR-compliant metadata. |
| Frontend | Streamlit for rapid prototyping or React + Vite for a richer UI | Streamlit enables quick drag-and-drop upload; React app can integrate with APIs later. |
| Deployment | Docker, Docker Compose, GitHub Actions | Containerize model + API; automate tests/inference benchmarks. |

## 3. Classification Flow

1. **Upload**: User drops an audio file into the web app or calls an ingestion endpoint.
2. **Validation**: Backend verifies format/length, stores raw audio, and enqueues processing.
3. **Spectrogram Generation**: Librosa converts audio to log-mel spectrograms; augmentations apply if training.
4. **Model Inference**: Spectrogram tensor feeds into the trained CNN; outputs class probabilities.
5. **Risk Mapping**: Top species maps to disease risks (e.g., *Aedes aegypti* -> Dengue, Zika, Chikungunya).
6. **Response Construction**: Combine species label, confidence score, risk narrative, and prevention recommendations.
7. **Presentation**: UI shows text plus optional icons; may also link to public-health resources.

Example response:

> "This recording resembles an *Aedes aegypti* (Dengue vector). Risk: Dengue fever possible. Prevention: Remove stagnant water, apply DEET-based repellent, and use window screens."

## 4. Health Risks by Species

- **Aedes aegypti - Dengue Mosquito**
  - Diseases: Dengue fever, Zika, Chikungunya, Yellow fever.
  - Symptoms to highlight: Sudden high fever, severe joint pain, rash, retro-orbital pain. Severe dengue can be fatal without immediate care.
- **Anopheles gambiae - Malaria Mosquito**
  - Diseases: Malaria (various *Plasmodium* species).
  - Symptoms: Periodic fever or chills, vomiting, anemia. Untreated malaria may cause cerebral complications and death.
- **Culex quinquefasciatus - Common House Mosquito**
  - Diseases: West Nile virus, Japanese encephalitis, Lymphatic filariasis.
  - Symptoms: Fever, lymph swelling, and in severe neurologic cases paralysis or encephalitis.

Future versions can attach severity scales or probability distributions to each disease to personalize alerts.

## 5. Safety & Prevention Playbook

### Avoid Bites
- Apply repellents with DEET, picaridin, or lemon-eucalyptus oil per manufacturer guidance.
- Wear light-colored long sleeves or pants, especially around dawn and dusk.
- Sleep under treated mosquito nets and deploy indoor traps or fans to disrupt flight paths.

### Remove Breeding Sites
- Empty stagnant-water sources (buckets, planters, AC trays, coolers) every 2-3 days.
- Clean gutters and cover stored water; use larvicides where drainage is impossible.
- Keep surroundings tidy to cut down on shaded, humid microhabitats.

### Home Control Measures
- Install or repair window and door screens; seal structural gaps.
- Use electric vaporizers, coils, or UV/CO2 traps in high-risk rooms, ensuring adequate ventilation.
- Coordinate with local health departments for community spraying or fogging when needed.

## 6. Next Steps

1. Collect labeled wingbeat recordings and train the baseline CNN/CRNN using the pipeline above.
2. Build a Streamlit proof-of-concept that uploads files, calls the inference API, and renders risk/prevention cards.
3. Integrate analytics (latency, confidence distribution) to monitor real-world performance and guide retraining.

## 7. Local Setup & Dataset Preparation

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv/Scripts/activate  # Windows
   pip install -r requirements.txt
   ```
2. Follow `docs/data.md` to download HumBugDB and/or rehydrate MosquitoSound (already placed in `data/raw/mosquitosound/`).
3. Normalize a dataset (example for HumBugDB):
   ```bash
   python scripts/prepare_dataset.py \
     --dataset-name humbugdb \
     --source data/raw/humbugdb \
     --output data/processed/humbugdb \
     --metadata data/raw/humbugdb/metadata.csv
   ```
4. The script resamples clips to 16 kHz mono WAV, validates durations, and writes manifests (`metadata.csv`, `discarded.csv`) for downstream preprocessing.
5. Build cached log-mel spectrograms ready for model training:
   ```bash
   python scripts/build_features.py \
     --manifest data/processed/humbugdb/metadata.csv \
     --output data/features/humbugdb
   ```
   This produces `.npy` tensors plus `features.csv`, enabling the training loop to stream features without re-running Librosa for every epoch.

## 8. Baseline Training & Risk Mapping

1. Train the CNN classifier on cached features:
   ```bash
   python scripts/train_model.py \
     --features data/features/humbugdb/features.csv \
     --output checkpoints/humbugdb_baseline \
     --epochs 30 \
     --batch-size 32
   ```
   The trainer automatically splits off a validation subset, logs metrics (`training_history.pt`), and stores checkpoints (default `best_model.pt`) with label/index metadata for inference.
2. `src/mosquito_monitor/model/` contains the PyTorch dataset, model, and training harness so you can iterate on architectures or loss functions.
3. `src/mosquito_monitor/risk/mapping.py` defines `RiskLookup`, which maps normalized species keys (e.g., `aedes_aegypti`) to associated diseases, symptoms, and prevention tips. Use it after inference to enrich classifier outputs with human-readable risk statements.

## 9. Serving & Frontend

### FastAPI Inference Service

1. Ensure a trained checkpoint exists (e.g., `checkpoints/humbugdb_baseline/best_model.pt`).
2. Start the API:
   ```bash
   python scripts/serve_api.py --checkpoint checkpoints/humbugdb_baseline/best_model.pt --port 8000
   ```
3. Test with `curl`:
   ```bash
   curl -F "file=@sample.wav" http://localhost:8000/predict
   ```
   The API returns species probabilities plus risk info derived from `RiskLookup`.

### Streamlit UI

1. Run the Streamlit client pointing to the same checkpoint:
   ```bash
  python scripts/run_streamlit.py --checkpoint checkpoints/humbugdb_baseline/best_model.pt --port 8501
   ```
2. Open `http://localhost:8501`, upload audio, and view predictions alongside prevention guidance.

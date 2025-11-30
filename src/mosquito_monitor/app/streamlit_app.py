from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import streamlit as st
import torch
from scipy.io import wavfile

from mosquito_monitor.audio import FeatureConfig, FeatureExtractor
from mosquito_monitor.model.networks import BaselineCNN
from mosquito_monitor.risk import RiskLookup


@st.cache_resource
def load_model(checkpoint_path: Path) -> tuple[BaselineCNN, dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    label_to_index = checkpoint["label_to_index"]
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    model = BaselineCNN(num_classes=len(label_to_index))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, index_to_label


def run_app(
    checkpoint_path: Path,
    feature_config: Optional[FeatureConfig] = None,
    risk_lookup: Optional[RiskLookup] = None,
) -> None:
    feature_config = feature_config or FeatureConfig()
    risk_lookup = risk_lookup or RiskLookup()
    st.set_page_config(page_title="Mosquito Monitor", page_icon="Mosquito", layout="centered")
    st.title("Mosquito Audio Classifier")
    st.write(
        "Upload a short mosquito wingbeat recording (.wav/.mp3). "
        "The model predicts the species and surfaces associated health risks."
    )

    model, index_to_label = load_model(checkpoint_path)
    device = next(model.parameters()).device

    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])
    if not uploaded_file:
        st.info("Awaiting audio upload...")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    sample_rate, audio = wavfile.read(tmp_path.as_posix())
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    extractor = FeatureExtractor(feature_config)
    log_mel, _ = extractor.transform_audio(audio.astype(np.float32), sample_rate)
    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_index = int(np.argmax(probs))
        confidence = float(probs[top_index])
        species_key = index_to_label[top_index]

    risk = risk_lookup.describe(species_key)

    st.subheader(f"Prediction: {risk.species if risk else species_key}")
    st.metric("Confidence", f"{confidence*100:.1f}%")

    if risk:
        st.write("**Potential diseases:** " + ", ".join(risk.diseases))
        st.write("**Symptoms to monitor:**")
        for symptom in risk.symptoms:
            st.write(f"- {symptom}")
        st.write("**Recommended prevention steps:**")
        for step in risk.prevention:
            st.write(f"- {step}")
    else:
        st.warning("No risk information available for this species.")

    st.audio(uploaded_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamlit mosquito classifier entry")
    parser.add_argument("--checkpoint", required=True, type=Path)
    args, _ = parser.parse_known_args()
    run_app(args.checkpoint)


if __name__ == "__main__":
    main()

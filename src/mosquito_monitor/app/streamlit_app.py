from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from datetime import datetime

import base64
import numpy as np
import pydeck as pdk
import requests
import streamlit as st
import torch
from scipy.io import wavfile

from mosquito_monitor.audio import FeatureConfig, FeatureExtractor
from mosquito_monitor.model.networks import BaselineCNN
from mosquito_monitor.risk import RiskLookup


NOMINATIM_ENDPOINT = "https://nominatim.openstreetmap.org/search"
OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
NOMINATIM_USER_AGENT = os.getenv("NOMINATIM_USER_AGENT", "machardani-app")

DANGER_PROFILES: Dict[str, Dict[str, object]] = {
    "aedes_aegypti": {"level": "Severe", "score": 0.95, "color": "#b71c1c"},
    "anopheles_gambiae": {"level": "High", "score": 0.8, "color": "#c2185b"},
    "culex_quinquefasciatus": {"level": "Elevated", "score": 0.65, "color": "#e64a19"},
    "default": {"level": "Guarded", "score": 0.5, "color": "#f9a825"},
}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MOSQUITO_IMAGE_PATH = PROJECT_ROOT / "mosquito.png"
try:
    MOSQUITO_IMAGE_B64 = base64.b64encode(MOSQUITO_IMAGE_PATH.read_bytes()).decode("utf-8")
except FileNotFoundError:
    MOSQUITO_IMAGE_B64 = ""


@st.cache_resource(show_spinner=False)
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


@st.cache_data(show_spinner=False, ttl=120)
def autocomplete_locations(query: str, limit: int = 5) -> List[dict[str, object]]:
    if not query:
        return []
    params = {
        "q": query,
        "format": "jsonv2",
        "limit": limit,
        "addressdetails": 1,
    }
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    try:
        response = requests.get(NOMINATIM_ENDPOINT, params=params, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return []
    results = response.json()
    suggestions = []
    for entry in results:
        label = entry.get("display_name", query)
        suggestions.append(
            {
                "label": label,
                "lat": float(entry.get("lat", 0.0)),
                "lng": float(entry.get("lon", 0.0)),
            }
        )
    return suggestions


@st.cache_data(show_spinner=False)
def geocode_location(query: str) -> Optional[dict[str, object]]:
    if not query:
        return None
    matches = autocomplete_locations(query, limit=1)
    if not matches:
        return None
    top = matches[0]
    return {"formatted_address": top["label"], "lat": top["lat"], "lng": top["lng"]}


def _build_overpass_query(
    lat: float, lng: float, key_value_pairs: Sequence[Tuple[str, str]], radius_m: int = 5000
) -> str:
    query = "[out:json][timeout:25];("
    for key, value in key_value_pairs:
        for element in ("node", "way", "relation"):
            query += f'{element}(around:{radius_m},{lat},{lng})["{key}"="{value}"];'
    query += ");out center;"
    return query


def _format_address(tags: Dict[str, str]) -> str:
    parts = [
        tags.get("addr:housenumber"),
        tags.get("addr:street"),
        tags.get("addr:suburb"),
        tags.get("addr:city"),
        tags.get("addr:state"),
    ]
    filtered = [p for p in parts if p]
    return ", ".join(filtered) if filtered else tags.get("name", "Address unavailable")


@st.cache_data(show_spinner=False, ttl=600)
def fetch_pois(
    lat: float, lng: float, tag_filters: Sequence[Tuple[str, str]], limit: int = 3
) -> List[dict[str, object]]:
    query = _build_overpass_query(lat, lng, tag_filters)
    try:
        response = requests.post(OVERPASS_ENDPOINT, data={"data": query}, timeout=25)
        response.raise_for_status()
    except requests.RequestException:
        return []
    payload = response.json()
    elements = payload.get("elements", [])[:limit]
    pois: List[dict[str, object]] = []
    for elem in elements:
        tags = elem.get("tags", {})
        lat_val = elem.get("lat") or (elem.get("center") or {}).get("lat")
        lng_val = elem.get("lon") or (elem.get("center") or {}).get("lon")
        if lat_val is None or lng_val is None:
            continue
        pois.append(
            {
                "name": tags.get("name", "Unnamed location"),
                "address": _format_address(tags),
                "lat": float(lat_val),
                "lng": float(lng_val),
            }
        )
    return pois


def _query_ipapi() -> Optional[dict[str, object]]:
    response = requests.get("https://ipapi.co/json/", timeout=8)
    response.raise_for_status()
    payload = response.json()
    lat = payload.get("latitude")
    lng = payload.get("longitude")
    if lat is None or lng is None:
        return None
    return {
        "formatted_address": ", ".join(
            [part for part in (payload.get("city"), payload.get("region"), payload.get("country_name")) if part]
        )
        or "Current location (approx.)",
        "lat": lat,
        "lng": lng,
    }


def _query_ipinfo() -> Optional[dict[str, object]]:
    response = requests.get("https://ipinfo.io/json", timeout=8)
    response.raise_for_status()
    payload = response.json()
    loc = payload.get("loc")
    if not loc:
        return None
    lat_str, lng_str = loc.split(",")
    return {
        "formatted_address": ", ".join(
            [part for part in (payload.get("city"), payload.get("region"), payload.get("country")) if part]
        )
        or "Current location (approx.)",
        "lat": float(lat_str),
        "lng": float(lng_str),
    }


@st.cache_data(show_spinner=False, ttl=600)
def lookup_current_location() -> Optional[dict[str, object]]:
    """Approximate the user's current location via IP-based geolocation."""
    for resolver in (_query_ipapi, _query_ipinfo):
        try:
            result = resolver()
        except requests.RequestException:
            result = None
        if result:
            return result
    return None


def fetch_nearby_hospitals(lat: float, lng: float, limit: int = 3) -> List[dict[str, object]]:
    return fetch_pois(
        lat,
        lng,
        tag_filters=[("amenity", "hospital"), ("healthcare", "hospital")],
        limit=limit,
    )


def fetch_nearby_civic_offices(lat: float, lng: float, limit: int = 2) -> List[dict[str, object]]:
    return fetch_pois(
        lat,
        lng,
        tag_filters=[("amenity", "townhall"), ("office", "government")],
        limit=limit,
    )


def compute_danger_profile(species_key: str) -> dict[str, object]:
    normalized = species_key.lower().replace(" ", "_")
    return DANGER_PROFILES.get(normalized, DANGER_PROFILES["default"])


def build_heatmap(lat: float, lng: float, danger_score: float, caption: str) -> pdk.Deck:
    # Build radial points around the recorded location to visualize risk intensity.
    offsets = [
        (0.0, 0.0, 1.0),
        (0.01, 0.005, 0.7),
        (-0.008, 0.003, 0.6),
        (0.004, -0.009, 0.5),
        (-0.006, -0.007, 0.4),
    ]
    points = [
        {"lat": lat + dy, "lon": lng + dx, "weight": danger_score * weight}
        for dx, dy, weight in offsets
    ]
    layer = pdk.Layer(
        "HeatmapLayer",
        data=points,
        get_position="[lon, lat]",
        get_weight="weight",
        radiusPixels=80,
    )
    view_state = pdk.ViewState(latitude=lat, longitude=lng, zoom=11, pitch=0)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={"text": f"Mosquito activity near\n{caption}"},
    )


def run_app(
    checkpoint_path: Path,
    feature_config: Optional[FeatureConfig] = None,
    risk_lookup: Optional[RiskLookup] = None,
) -> None:
    feature_config = feature_config or FeatureConfig()
    risk_lookup = risk_lookup or RiskLookup()
    st.set_page_config(page_title="Mosquito Monitor", page_icon="Mosquito", layout="wide")
    mosquito_data_uri = f"data:image/png;base64,{MOSQUITO_IMAGE_B64}" if MOSQUITO_IMAGE_B64 else ""

    st.markdown(
        """
        <style>
        :root {
            --bg-primary: #03060f;
            --card-bg: rgba(11, 19, 33, 0.85);
            --glass-border: rgba(16, 243, 193, 0.25);
            --neon-teal: #10f7c2;
            --neon-lime: #a9ff68;
            --neon-cyan: #58d7ff;
            --neon-orange: #ff8740;
            --text-muted: #9fb4c9;
        }
        body {
            background-color: var(--bg-primary);
        }
        .machardani-hero-gateway {
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            background: radial-gradient(circle at top, rgba(88,215,255,0.25), transparent 60%),
                        radial-gradient(circle at bottom right, rgba(255,135,64,0.25), transparent 55%),
                        #02050c;
            overflow: hidden;
        }
        .machardani-hero-card {
            width: min(520px, 90vw);
            padding: 3rem 2.5rem;
            border-radius: 2rem;
            border: 1px solid rgba(255,255,255,0.2);
            background: rgba(5, 13, 25, 0.75);
            backdrop-filter: blur(16px);
            text-align: center;
            box-shadow: 0 30px 80px rgba(0,0,0,0.55);
            animation: floatGlow 6s ease-in-out infinite;
        }
        @keyframes floatGlow {
            0%, 100% { transform: translateY(0px); box-shadow: 0 30px 80px rgba(0,0,0,0.55); }
            50% { transform: translateY(-8px); box-shadow: 0 40px 90px rgba(16,247,194,0.25); }
        }
        .machardani-scroll-indicator {
            position: absolute;
            bottom: 40px;
            color: var(--text-muted);
            letter-spacing: 0.3rem;
            font-size: 0.75rem;
            text-transform: uppercase;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 0.4rem;
        }
        .machardani-scroll-indicator span {
            display: block;
            animation: pulseArrow 2s infinite;
        }
        @keyframes pulseArrow {
            0% { transform: translateY(0); opacity: 0.4; }
            50% { transform: translateY(6px); opacity: 1; }
            100% { transform: translateY(12px); opacity: 0.4; }
        }
        .machardani-root {
            background: radial-gradient(circle at top, rgba(16,247,194,0.1), transparent 55%);
            padding-bottom: 2rem;
        }
        .machardani-hero {
            text-align: center;
            padding: 2rem 1rem 1rem;
        }
        .machardani-hero h1 {
            color: var(--neon-teal);
            font-size: 3.5rem;
            letter-spacing: 0.4rem;
            margin-bottom: 0.25rem;
        }
        .machardani-hero h2 {
            color: #e3f9ff;
            font-weight: 400;
        }
        .machardani-hero p {
            color: var(--text-muted);
            max-width: 640px;
            margin: 0.75rem auto 0;
        }
        .machardani-card {
            background: var(--card-bg);
            border-radius: 1.5rem;
            border: 1px solid var(--glass-border);
            box-shadow: 0 20px 45px rgba(0,0,0,0.35);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .machardani-section-title {
            font-size: 1.2rem;
            letter-spacing: 0.2rem;
            text-transform: uppercase;
            color: var(--neon-cyan);
            margin-bottom: 0.75rem;
        }
        .machardani-footer {
            text-align: center;
            color: var(--text-muted);
            padding: 2rem 0 1.5rem;
            font-size: 0.95rem;
        }
        .machardani-footer a {
            color: var(--neon-cyan);
            margin: 0 0.5rem;
            text-decoration: none;
        }
        .machardani-badge {
            display: inline-block;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.15);
            margin-right: 0.4rem;
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        .machardani-result-card {
            border-radius: 1rem;
            padding: 1rem;
            background: rgba(25, 35, 52, 0.8);
            border: 1px solid rgba(255, 135, 64, 0.35);
        }
        .machardani-risk-pill {
            padding: 0.15rem 0.7rem;
            border-radius: 999px;
            font-size: 0.75rem;
            margin-left: 0.5rem;
        }
        .mosquito-swarm {
            position: fixed;
            inset: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 5;
            overflow: visible;
        }
        .mosquito {
            position: absolute;
            width: 120px;
            height: 60px;
            background-image: url("__MOSQUITO_DATA_URI__");
            background-size: contain;
            background-repeat: no-repeat;
            opacity: 0.4;
            filter: drop-shadow(0 0 6px rgba(255,255,255,0.3));
            animation: hoverFlight var(--speed, 20s) ease-in-out infinite;
            transform: translate3d(0, calc(var(--scroll-y, 0) * var(--scroll-factor, 0.02)), 0);
        }
        .mosquito:hover {
            opacity: 0.7;
        }
        .mosquito.m1 { top: 5%; left: 5%; --speed: 26s; --scroll-factor: 0.02; animation-delay: 0s; }
        .mosquito.m2 { top: 5%; right: 5%; --speed: 24s; --scroll-factor: 0.015; animation-delay: 3s; }
        .mosquito.m3 { bottom: 5%; left: 5%; --speed: 28s; --scroll-factor: 0.02; animation-delay: 6s; }
        .mosquito.m4 { bottom: 5%; right: 5%; --speed: 30s; --scroll-factor: 0.018; animation-delay: 9s; }
        .mosquito.m5 { top: 15%; right: 20%; --speed: 25s; --scroll-factor: 0.02; animation-delay: 12s; }
        @keyframes hoverFlight {
            0% { transform: translate3d(0, 0, 0) rotate(-4deg); }
            25% { transform: translate3d(-25px, 15px, 0) rotate(6deg); }
            50% { transform: translate3d(20px, -10px, 0) rotate(-3deg); }
            75% { transform: translate3d(30px, 20px, 0) rotate(5deg); }
            100% { transform: translate3d(0, 0, 0) rotate(-4deg); }
        }
        </style>
        """.replace("__MOSQUITO_DATA_URI__", mosquito_data_uri),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="machardani-hero-gateway" id="gateway">
            <div class="machardani-hero-card">
                <div class="machardani-badge">Audio AI</div>
                <div class="machardani-badge">Vector Surveillance</div>
                <h1 style="color:var(--neon-teal);font-size:3.8rem;letter-spacing:0.5rem;margin:0.5rem 0;">MACHARDANI</h1>
                <p style="color:#d3e6f8;font-size:1.1rem;margin-bottom:0.75rem;">Mosquito Audio Classifier</p>
                <p style="color:var(--text-muted);line-height:1.7;">
                    Decode wingbeat recordings to identify mosquito species, gauge local disease risk, and summon nearby emergency support.
                </p>
            </div>
            <div class="machardani-scroll-indicator">
                <div>Scroll to begin</div>
                <span>↓</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="mosquito-swarm">
            <div class="mosquito m1"></div>
            <div class="mosquito m2"></div>
            <div class="mosquito m3"></div>
            <div class="mosquito m4"></div>
            <div class="mosquito m5"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="machardani-root" id="main">
            <section class="machardani-hero">
                <h2 style="color:#e3f9ff;font-weight:400;">Mosquito Audio Classifier</h2>
                <p>Upload a short mosquito wingbeat recording (.wav/.mp3) to detect species,
                visualize local mosquito pressure, and surface nearby response resources.</p>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []

    model, index_to_label = load_model(checkpoint_path)
    device = next(model.parameters()).device

    st.markdown('<div class="machardani-card">', unsafe_allow_html=True)
    st.markdown('<div class="machardani-section-title">Upload & Locate</div>', unsafe_allow_html=True)
    st.caption("Drag in your recording or browse files. Clips stay local and are not retained on the server.")
    uploaded_file = st.file_uploader("Upload audio file", type=["wav", "mp3"])

    location_columns = st.columns([3, 1])
    with location_columns[0]:
        default_query = st.session_state.get("location_query_value", "")
        location_query = st.text_input(
            "Where was this audio recorded?",
            value=default_query,
            help="Begin typing a city, neighborhood, or landmark to see suggestions.",
        ).strip()
        st.session_state["location_query_value"] = location_query
    with location_columns[1]:
        use_current_location = st.button(
            "Use My Location",
            help="Approximate your current location (IP-based) and autofill the search box.",
        )

    if use_current_location:
        current_location = lookup_current_location()
        if current_location:
            st.session_state["manual_geocode"] = current_location
            st.session_state["manual_geocode_label"] = current_location["formatted_address"]
            st.session_state["location_query_value"] = current_location["formatted_address"]
            st.session_state["location_flash"] = (
                f"Using approximate current location near {current_location['formatted_address']}."
            )
            st.experimental_rerun()
        else:
            st.error("Unable to detect your current location. Please type it manually.")

    if st.session_state.get("manual_geocode_label") and location_query != st.session_state["manual_geocode_label"]:
        st.session_state.pop("manual_geocode", None)
        st.session_state.pop("manual_geocode_label", None)

    if flash := st.session_state.pop("location_flash", None):
        st.success(flash)

    suggestions: List[dict[str, object]] = []
    if len(location_query) >= 3:
        suggestions = autocomplete_locations(location_query)

    selected_location: Optional[dict[str, object]] = None
    if suggestions:
        label_to_location = {entry["label"]: entry for entry in suggestions}
        suggestion_options = ["Use typed location"] + list(label_to_location.keys())
        selected_label = st.selectbox(
            "Suggested matches",
            suggestion_options,
            key="location_suggestion",
        )
        if selected_label != "Use typed location":
            selected_location = label_to_location[selected_label]
            st.caption(f"Using suggested location: {selected_label}")
    else:
        st.session_state.pop("location_suggestion", None)

    analyze_clicked = st.button("Analyze Recording", type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
    if not analyze_clicked:
        st.info("Upload an audio sample, pick the recording location, and click **Analyze Recording**.")
        return
    if not uploaded_file:
        st.error("Please upload a mosquito audio recording to continue.")
        return
    if not location_query and not selected_location and "manual_geocode" not in st.session_state:
        st.error("Please enter or select the recording location so we can generate targeted guidance.")
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
    danger_profile = compute_danger_profile(species_key)

    severity_colors = {
        "Severe": "#ff4d4f",
        "High": "#ffab40",
        "Elevated": "#ffda47",
        "Guarded": "#2dff8b",
    }
    level = str(danger_profile.get("level", "Elevated"))
    pill_color = severity_colors.get(level, "#ffda47")

    st.markdown('<div class="machardani-card">', unsafe_allow_html=True)
    st.markdown('<div class="machardani-section-title">Detection Result</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="machardani-result-card">
            <h3 style="color:white;margin-bottom:0.2rem;">{risk.species if risk else species_key}
                <span class="machardani-risk-pill" style="background:{pill_color};color:#08121f;">
                    {level}
                </span>
            </h3>
            <p style="color:var(--text-muted);margin-bottom:0.6rem;">Dominant wingbeat signature detected from uploaded clip.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols_metrics = st.columns(2)
    cols_metrics[0].metric("Model Confidence", f"{confidence*100:.1f}%")
    cols_metrics[1].metric("Danger Index", f"{float(danger_profile.get('score', 0))*100:.0f}/100")

    with st.expander("Risk Profile & Guidance", expanded=True):
        if risk:
            st.write("**Potential diseases:** " + ", ".join(risk.diseases))
            st.write("**Symptoms:**")
            for symptom in risk.symptoms:
                st.write(f"- {symptom}")
            st.write("**Prevention checklist:**")
            for step in risk.prevention:
                st.write(f"- {step}")
        else:
            st.info("No curated risk profile available for this species.")
    st.audio(uploaded_file)
    st.markdown("</div>", unsafe_allow_html=True)

    # Determine the final geo coordinates (manual, suggestion, or typed address).
    if (
        st.session_state.get("manual_geocode")
        and st.session_state.get("manual_geocode_label", "").lower() == location_query.lower()
    ):
        geocoded = st.session_state["manual_geocode"]
    elif selected_location:
        geocoded = {
            "formatted_address": selected_location["label"],
            "lat": selected_location["lat"],
            "lng": selected_location["lng"],
        }
    else:
        geocoded = geocode_location(location_query)
    st.markdown('<div class="machardani-card">', unsafe_allow_html=True)
    st.markdown('<div class="machardani-section-title">Heat Map</div>', unsafe_allow_html=True)
    if geocoded:
        st.markdown(
            f"<p style='color:var(--text-muted)'>Pinned near: "
            f"<span style='color:var(--neon-teal)'>{geocoded['formatted_address']}</span></p>",
            unsafe_allow_html=True,
        )
        deck = build_heatmap(
            lat=geocoded["lat"],
            lng=geocoded["lng"],
            danger_score=danger_profile["score"],
            caption=geocoded["formatted_address"],
        )
        st.pydeck_chart(deck, use_container_width=True)
        st.caption(
            "Glowing areas highlight elevated mosquito activity within ~5 km of the pinned location."
        )
    else:
        st.warning("Unable to determine the provided location. The heat map could not be displayed.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="machardani-card">', unsafe_allow_html=True)
    st.markdown('<div class="machardani-section-title">Local Guidance</div>', unsafe_allow_html=True)
    if geocoded:
        hospitals = fetch_nearby_hospitals(geocoded["lat"], geocoded["lng"], limit=3)
        muni_candidates = fetch_nearby_civic_offices(geocoded["lat"], geocoded["lng"], limit=2)
        cols_guidance = st.columns(2)
        with cols_guidance[0]:
            st.markdown("#### Emergency Hospitals")
            if hospitals:
                for hospital in hospitals:
                    st.markdown(
                        f"**🏥 {hospital['name']}**  \n"
                        f"<span style='color:var(--text-muted)'>{hospital.get('address', 'Address unavailable')}</span>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No nearby hospitals found within 5 km. Contact your regional health hotline.")
        with cols_guidance[1]:
            st.markdown("#### Nagar Nigam Helpline")
            if muni_candidates:
                muni = muni_candidates[0]
                details = muni.get("address", "Address unavailable")
                st.markdown(
                    f"**🏛️ {muni['name']}**  \n"
                    f"<span style='color:var(--text-muted)'>{details}</span>  \n"
                    "Contact their control room for fumigation drives.",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Dial your local Nagar Nigam control room (commonly 1916) for mosquito mitigation.")
    else:
        st.info("Local resources unavailable without a location.")
    st.markdown("</div>", unsafe_allow_html=True)

    if geocoded:
        history_entry = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "species": risk.species if risk else species_key,
            "confidence": confidence,
            "danger_level": level,
            "danger_score": float(danger_profile.get("score", 0)),
            "location": geocoded,
            "hospitals": hospitals,
            "municipal": muni_candidates,
        }
        st.session_state["analysis_history"].insert(0, history_entry)
        st.session_state["analysis_history"] = st.session_state["analysis_history"][:5]

    history_items = st.session_state.get("analysis_history", [])
    if len(history_items) > 1:
        st.markdown('<div class="machardani-card">', unsafe_allow_html=True)
        st.markdown('<div class="machardani-section-title">Previous Analyses</div>', unsafe_allow_html=True)
        for entry in history_items[1:]:
            with st.expander(
                f"{entry['species']} • {entry['location']['formatted_address']} • {entry['timestamp']}"
            ):
                st.write(
                    f"Confidence: {entry['confidence']*100:.1f}%, "
                    f"Danger: {entry['danger_level']} ({entry['danger_score']*100:.0f}/100)"
                )
                past_deck = build_heatmap(
                    lat=entry["location"]["lat"],
                    lng=entry["location"]["lng"],
                    danger_score=entry["danger_score"],
                    caption=entry["location"]["formatted_address"],
                )
                st.pydeck_chart(past_deck, use_container_width=True)
                if entry["hospitals"]:
                    st.write("Hospitals:")
                    for hospital in entry["hospitals"]:
                        st.write(f"- {hospital['name']} — {hospital.get('address', 'Address unavailable')}")
                if entry["municipal"]:
                    muni0 = entry["municipal"][0]
                    st.write(f"Municipal contact: {muni0['name']} ({muni0.get('address', 'Address unavailable')})")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <footer class="machardani-footer">
            <div>
                <a href="#">About</a> · <a href="#">Privacy</a> ·
                <a href="#">Model Info</a> · <a href="#">Contact</a>
            </div>
            <div style="margin-top:0.5rem;">Powered by AI audio analysis · Machardani Labs</div>
        </footer>
        <script>
        document.addEventListener("scroll", () => {
            document.documentElement.style.setProperty("--scroll-y", window.scrollY || 0);
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Streamlit mosquito classifier entry")
    parser.add_argument("--checkpoint", required=True, type=Path)
    args, _ = parser.parse_known_args()
    run_app(args.checkpoint)


if __name__ == "__main__":
    main()

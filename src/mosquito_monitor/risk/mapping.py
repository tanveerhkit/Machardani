from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RiskInfo:
    species: str
    diseases: List[str]
    symptoms: List[str]
    prevention: List[str]


SPECIES_RISK_MAP: Dict[str, RiskInfo] = {
    "aedes_aegypti": RiskInfo(
        species="Aedes aegypti",
        diseases=["Dengue", "Zika", "Chikungunya", "Yellow Fever"],
        symptoms=[
            "Sudden high fever",
            "Severe joint or muscle pain",
            "Rash and eye pain",
            "Potential hemorrhagic complications",
        ],
        prevention=[
            "Remove stagnant water weekly",
            "Use DEET or picaridin repellents",
            "Install window/door screens",
        ],
    ),
    "anopheles_gambiae": RiskInfo(
        species="Anopheles gambiae",
        diseases=["Malaria"],
        symptoms=[
            "Cyclical fever and chills",
            "Vomiting and fatigue",
            "Possible cerebral malaria if untreated",
        ],
        prevention=[
            "Sleep under insecticide-treated nets",
            "Consider indoor residual spraying",
            "Seek prophylaxis guidance in endemic regions",
        ],
    ),
    "culex_quinquefasciatus": RiskInfo(
        species="Culex quinquefasciatus",
        diseases=["West Nile Virus", "Japanese Encephalitis", "Lymphatic Filariasis"],
        symptoms=[
            "Fever and body aches",
            "Neurological issues in severe cases",
            "Limb swelling for filariasis",
        ],
        prevention=[
            "Maintain screens and repair gaps",
            "Use outdoor traps or coils",
            "Coordinate with local vector control teams",
        ],
    ),
}


class RiskLookup:
    """Maps predicted species to disease risk narratives."""

    def __init__(self, risk_map: Optional[Dict[str, RiskInfo]] = None) -> None:
        self.risk_map = risk_map or SPECIES_RISK_MAP

    def describe(self, species_key: str) -> Optional[RiskInfo]:
        species_key = species_key.lower().replace(" ", "_")
        return self.risk_map.get(species_key)

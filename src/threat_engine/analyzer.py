"""
TRIDENT Threat Analyzer

Converts predictions into threat intelligence.
"""

import json

from utils.config import METADATA_DIR
from utils.logger import logger


class ThreatAnalyzer:
    """
    Analyzes predicted targets and
    returns threat information.
    """

    def __init__(self) -> None:

        with open(
            METADATA_DIR / "target_profiles.json",
            "r",
            encoding="utf-8",
        ) as file:

            self.target_profiles = json.load(file)

    # ---------------------------------------------------------

    def analyze(
        self,
        prediction: str,
        confidence: float,
    ) -> dict:

        logger.info(
            "Analyzing prediction..."
        )

        profile = self.target_profiles.get(

            prediction.lower(),

            self.target_profiles["default"],

        )

        return {

            "target": prediction,

            "confidence": confidence,

            "category": profile["category"],

            "threat_level": profile["threat_level"],

            "recommended_action": profile["recommended_action"],

        }


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    analyzer = ThreatAnalyzer()

    result = analyzer.analyze(
        "submarine",
        0.97,
    )

    print(result)
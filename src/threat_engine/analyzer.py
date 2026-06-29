"""
TRIDENT Threat Analyzer

Converts CNN predictions into
actionable threat intelligence.
"""

from threat_engine.rules import THREAT_RULES

from utils.logger import logger


class ThreatAnalyzer:
    """
    Assigns threat levels based on
    prediction results.
    """

    def analyze(
        self,
        prediction: str,
        confidence: float,
    ) -> dict:

        logger.info(
            "Analyzing threat..."
        )

        info = THREAT_RULES.get(

            prediction.lower(),

            THREAT_RULES["unknown"],

        )

        result = {

            "target": prediction,

            "confidence": confidence,

            "threat_level": info["level"],

            "recommended_action": info["action"],

        }

        logger.info(
            "Threat analysis completed."
        )

        return result


# -------------------------------------------------------------
# Testing
# -------------------------------------------------------------

if __name__ == "__main__":

    analyzer = ThreatAnalyzer()

    result = analyzer.analyze(

        "Submarine",

        0.96,

    )

    print()

    print("========== THREAT REPORT ==========")

    for key, value in result.items():

        print(f"{key} : {value}")
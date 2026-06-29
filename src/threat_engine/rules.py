"""
TRIDENT Threat Rules

Defines threat levels for
different underwater targets.
"""

THREAT_RULES = {

    "submarine": {

        "level": "HIGH",

        "action":
            "Immediate monitoring recommended.",

    },

    "torpedo": {

        "level": "CRITICAL",

        "action":
            "Immediate defensive response required.",

    },

    "ship": {

        "level": "MEDIUM",

        "action":
            "Continue tracking.",

    },

    "ambient_noise": {

        "level": "LOW",

        "action":
            "No action required.",

    },

    "whale": {

        "level": "LOW",

        "action":
            "Marine life detected.",

    },

    "dolphin": {

        "level": "LOW",

        "action":
            "Marine life detected.",

    },

    "unknown": {

        "level": "MEDIUM",

        "action":
            "Manual inspection recommended.",

    },

}
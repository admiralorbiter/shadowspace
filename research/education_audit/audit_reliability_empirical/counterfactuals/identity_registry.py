"""Identity Pair Registry for Gender Swap Interventions."""

from __future__ import annotations

from typing import Any, Dict, List

PREREGISTERED_IDENTITY_CHANNELS = [
    {
        "channel_id": "pronoun_he_she",
        "category": "pronoun",
        "to_masc": {"She": "He", "she": "he", "Her": "Him", "her": "him", "Hers": "His", "hers": "his"},
        "to_fem": {"He": "She", "he": "she", "Him": "Her", "him": "her", "His": "Her", "his": "her"},
    },
    {
        "channel_id": "name_michael_sarah",
        "category": "name",
        "sub_masc": {"Michael": "Michael"},
        "sub_fem": {"Michael": "Sarah"},
    },
    {
        "channel_id": "name_joseph_kelly",
        "category": "name",
        "sub_masc": {"Joseph": "Joseph"},
        "sub_fem": {"Joseph": "Kelly"},
    },
    {
        "channel_id": "name_david_emily",
        "category": "name",
        "sub_masc": {"David": "David"},
        "sub_fem": {"David": "Emily"},
    },
]

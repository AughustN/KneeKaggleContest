"""Shared constants for the knee package."""

from typing import List

STUDY_ID = "StudyInstanceUID"

LABELS: List[str] = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

NUM_LABELS = len(LABELS)  # 12

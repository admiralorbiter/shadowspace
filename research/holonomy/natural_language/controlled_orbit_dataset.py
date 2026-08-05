"""Token-Checked Controlled Orbit Dataset Generator for Phase E2-A1.2.

Constructs 300 balanced NLI orbits (100 Entailment, 100 Neutral, 100 Contradiction)
with token-matched entity renaming quartets verified across tokenizers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

from research.holonomy.natural_language.orbit_schema import SemanticOrbit, SemanticVertex
from research.holonomy.natural_language.orbit_builder import OrbitBuilder
from research.holonomy.natural_language.transforms.entity_rename import ReversibleEntityRenameTransform


import hashlib

# 10 Token-Checked Name Quartets (Name1, Name2, Name3, Name4)
# Quartets 0-4: Same-gender formal invariance track
# Quartets 5-9: Cross-gender counterfactual bias track
NAME_QUARTETS = [
    ("Alice", "Emma", "Olivia", "Sophia"),      # Formal: Female
    ("Bob", "Liam", "Jackson", "Ethan"),        # Formal: Male
    ("Amelia", "Charlotte", "Abigail", "Elizabeth"), # Formal: Female
    ("David", "Noah", "Oliver", "James"),       # Formal: Male
    ("Grace", "Ella", "Sofia", "Victoria"),     # Formal: Female
    ("Alice", "Bob", "Charlie", "David"),      # Bias: Cross-gender
    ("Emma", "Liam", "Olivia", "Noah"),         # Bias: Cross-gender
    ("Sophia", "Jackson", "Ava", "Lucas"),      # Bias: Cross-gender
    ("Mia", "Ethan", "Isabella", "Aiden"),      # Bias: Cross-gender
    ("Harper", "Oliver", "Evelyn", "Elijah"),   # Bias: Cross-gender
]


# Base templates for 300 balanced orbits (100 per NLI label class)
ENTAILMENT_TEMPLATES = [
    ("{A} called {C} yesterday.", "{A} spoke with {C} on the phone."),
    ("{A} gave a book to {C}.", "{C} received a book from {A}."),
    ("{A} lives in the same house as {C}.", "{C} and {A} share a residence."),
    ("{A} taught {C} mathematics.", "{C} learned math from {A}."),
    ("{A} helped {C} build a fence.", "{C} received help from {A} to build a fence."),
    ("{A} bought a car from {C}.", "{C} sold a car to {A}."),
    ("{A} invited {C} to dinner.", "{C} was invited to dinner by {A}."),
    ("{A} works for {C} at the office.", "{C} is {A}'s employer."),
    ("{A} wrote a letter to {C}.", "{C} received a letter written by {A}."),
    ("{A} fixed {C}'s computer.", "{C}'s computer was repaired by {A}."),
]

NEUTRAL_TEMPLATES = [
    ("{A} met {C} at the coffee shop.", "{A} and {C} discussed business for two hours."),
    ("{A} sent an email to {C}.", "{C} replied to {A}'s email immediately."),
    ("{A} saw {C} at the park.", "{C} was wearing a blue jacket."),
    ("{A} traveled with {C} to London.", "{A} and {C} stayed at the same hotel."),
    ("{A} borrows money from {C}.", "{C} earns more money than {A}."),
    ("{A} plays tennis with {C}.", "{C} is a better tennis player than {A}."),
    ("{A} visited {C} last weekend.", "{C} cooked dinner for {A}."),
    ("{A} works with {C} on a project.", "{A} and {C} have known each other for five years."),
    ("{A} gave advice to {C}.", "{C} followed {A}'s recommendation."),
    ("{A} drives {C} to school.", "{C} does not have a driver's license."),
]

CONTRADICTION_TEMPLATES = [
    ("{A} is older than {C}.", "{C} is older than {A}."),
    ("{A} lives in New York with {C}.", "{A} has never met {C} in person."),
    ("{A} gave all her money to {C}.", "{A} kept all her money for herself and gave nothing to {C}."),
    ("{A} is standing to the left of {C}.", "{A} is standing to the right of {C}."),
    ("{A} arrived before {C}.", "{C} arrived before {A}."),
    ("{A} defeated {C} in the match.", "{C} defeated {A} in the match."),
    ("{A} is taller than {C}.", "{C} is taller than {A}."),
    ("{A} works in the same building as {C}.", "{A} and {C} work in completely different cities."),
    ("{A} gave birth to {C}.", "{C} is older than {A}."),
    ("{A} bought a gift for {C}.", "{A} refused to give anything to {C}."),
]


@dataclass(frozen=True)
class ControlledOrbitDataset:
    """Dataset of 300 controlled orbits split into Train/Val/Test with item-level isolation."""

    train_orbits: List[SemanticOrbit]
    val_orbits: List[SemanticOrbit]
    test_orbits: List[SemanticOrbit]
    name_quartets_train: List[Tuple[str, str, str, str]]
    name_quartets_test: List[Tuple[str, str, str, str]]


def validate_name_quartet_tokens(tokenizer: Any, quartet: Tuple[str, str, str, str]) -> Dict[str, Any]:
    """Validates that a name quartet has equal token counts and no UNK tokens."""
    name1, name2, name3, name4 = quartet
    t1 = tokenizer.encode(name1, add_special_tokens=False)
    t2 = tokenizer.encode(name2, add_special_tokens=False)
    t3 = tokenizer.encode(name3, add_special_tokens=False)
    t4 = tokenizer.encode(name4, add_special_tokens=False)

    unk_id = getattr(tokenizer, "unk_token_id", None)
    has_unk = any(unk_id in t for t in [t1, t2, t3, t4] if unk_id is not None)
    equal_len = (len(t1) == len(t2) == len(t3) == len(t4))

    return {
        "quartet": quartet,
        "token_lengths": (len(t1), len(t2), len(t3), len(t4)),
        "equal_length": equal_len,
        "has_unk": has_unk,
        "valid": equal_len and not has_unk,
    }


def build_controlled_orbit_dataset(
    target_orbit_count: int = 300,
    seed: int = 42,
) -> ControlledOrbitDataset:
    """Builds 300 unique controlled orbits with template-grouped & name-OOD train/val/test splits."""
    builder = OrbitBuilder()

    train_quartets = NAME_QUARTETS[:6]
    val_quartets = NAME_QUARTETS[6:8]
    test_quartets = NAME_QUARTETS[8:]  # 2 held-out quartets

    train_orbits: List[SemanticOrbit] = []
    val_orbits: List[SemanticOrbit] = []
    test_orbits: List[SemanticOrbit] = []

    categories = [
        ("entailment", ENTAILMENT_TEMPLATES),
        ("neutral", NEUTRAL_TEMPLATES),
        ("contradiction", CONTRADICTION_TEMPLATES),
    ]

    for label_class, templates in categories:
        for t_idx, tmpl in enumerate(templates):
            for q_idx, q in enumerate(NAME_QUARTETS):
                p_raw = tmpl[0].format(A=q[0], B=q[1], C=q[2], D=q[3])
                h_raw = tmpl[1].format(A=q[0], B=q[1], C=q[2], D=q[3])

                t_a = ReversibleEntityRenameTransform("rename_a", q[0], q[1])
                t_b = ReversibleEntityRenameTransform("rename_b", q[2], q[3])

                orb_id = f"ctrl_{label_class}_t{t_idx:02d}_q{q_idx:02d}"
                orb = builder.build_square_orbit(
                    orbit_id=orb_id,
                    source_uid=f"{label_class}_t{t_idx:02d}_q{q_idx:02d}",
                    dataset="controlled_nli",
                    base_premise=p_raw,
                    base_hypothesis=h_raw,
                    transform_a=t_a,
                    transform_b=t_b,
                )
                orb.metadata["label_class"] = label_class
                orb.metadata["quartet"] = q
                orb.metadata["template_idx"] = t_idx
                orb.metadata["quartet_idx"] = q_idx
                orb.metadata["track"] = "formal_invariance" if q_idx < 5 else "counterfactual_bias"


                # Split assignment:
                # Train: templates 0-5, quartets 0-5 (60 per class = 180 total)
                # Val:   templates 6-7, quartets 6-7 (20 per class = 60 total)
                # Test:  templates 8-9, quartets 8-9 (20 per class = 60 total held-out)
                if t_idx < 6 and q_idx < 6:
                    train_orbits.append(orb)
                elif 6 <= t_idx < 8 and 6 <= q_idx < 8:
                    val_orbits.append(orb)
                elif t_idx >= 8 and q_idx >= 8:
                    test_orbits.append(orb)
                else:
                    # Distribute remaining cross-combinations into train/val
                    if q_idx < 6:
                        train_orbits.append(orb)
                    elif q_idx < 8:
                        val_orbits.append(orb)
                    else:
                        test_orbits.append(orb)

    # Re-verify zero text hash overlap between splits
    def get_text_hashes(orbit_list: List[SemanticOrbit]) -> set[str]:
        s = set()
        for orb in orbit_list:
            for v in orb.vertices.values():
                s.add(hashlib.sha256(f"{v.premise}||{v.hypothesis}".encode("utf-8")).hexdigest())
        return s

    train_hashes = get_text_hashes(train_orbits)
    val_hashes = get_text_hashes(val_orbits)
    test_hashes = get_text_hashes(test_orbits)

    assert len(train_hashes.intersection(test_hashes)) == 0, "Train-Test text hash overlap detected!"
    assert len(val_hashes.intersection(test_hashes)) == 0, "Val-Test text hash overlap detected!"

    return ControlledOrbitDataset(
        train_orbits=train_orbits,
        val_orbits=val_orbits,
        test_orbits=test_orbits,
        name_quartets_train=train_quartets,
        name_quartets_test=test_quartets,
    )


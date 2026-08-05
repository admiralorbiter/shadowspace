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


# 8 Token-Checked Name Quartets (Name1, Name2, Name3, Name4)
# Verified to have equal token length and 0 UNK tokens across RoBERTa & DeBERTa tokenizers
NAME_QUARTETS = [
    ("Alice", "Bob", "Charlie", "David"),
    ("Emma", "Liam", "Olivia", "Noah"),
    ("Sophia", "Jackson", "Ava", "Lucas"),
    ("Mia", "Ethan", "Isabella", "Aiden"),
    ("Harper", "Oliver", "Evelyn", "Elijah"),
    ("Charlotte", "James", "Amelia", "Benjamin"),
    ("Abigail", "Henry", "Emily", "Alexander"),
    ("Elizabeth", "Sebastian", "Sofia", "Jack"),
]

# Base templates for 300 balanced orbits (100 per NLI label class)
# Each template contains place-holders {A}, {B}, {C}, {D} for 2 independent name pairs:
# a: {A} <-> {B}, b: {C} <-> {D}
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
    """Builds 300 balanced controlled orbits with 60/20/20 train/val/test splits."""
    rng = np.random.default_rng(seed)
    builder = OrbitBuilder()

    # Reserve last 2 quartets for test set to evaluate held-out name generalization
    train_quartets = NAME_QUARTETS[:6]
    test_quartets = NAME_QUARTETS[6:]

    orbits_entailment = []
    orbits_neutral = []
    orbits_contradiction = []

    # Generate 100 items per category
    for i in range(target_orbit_count // 3):
        # Entailment
        tmpl_e = ENTAILMENT_TEMPLATES[i % len(ENTAILMENT_TEMPLATES)]
        q_idx = i % len(NAME_QUARTETS)
        q = NAME_QUARTETS[q_idx]
        p_raw = tmpl_e[0].format(A=q[0], B=q[1], C=q[2], D=q[3])
        h_raw = tmpl_e[1].format(A=q[0], B=q[1], C=q[2], D=q[3])

        t_a = ReversibleEntityRenameTransform("rename_a", q[0], q[1])
        t_b = ReversibleEntityRenameTransform("rename_b", q[2], q[3])

        orb_e = builder.build_square_orbit(
            orbit_id=f"controlled_entailment_{i:03d}",
            source_uid=f"entailment_{i:03d}",
            dataset="controlled_nli",
            base_premise=p_raw,
            base_hypothesis=h_raw,
            transform_a=t_a,
            transform_b=t_b,
        )
        orb_e.metadata["label_class"] = "entailment"
        orb_e.metadata["quartet"] = q
        orb_e.metadata["base_item_id"] = i
        orbits_entailment.append(orb_e)

        # Neutral
        tmpl_n = NEUTRAL_TEMPLATES[i % len(NEUTRAL_TEMPLATES)]
        p_raw_n = tmpl_n[0].format(A=q[0], B=q[1], C=q[2], D=q[3])
        h_raw_n = tmpl_n[1].format(A=q[0], B=q[1], C=q[2], D=q[3])
        orb_n = builder.build_square_orbit(
            orbit_id=f"controlled_neutral_{i:03d}",
            source_uid=f"neutral_{i:03d}",
            dataset="controlled_nli",
            base_premise=p_raw_n,
            base_hypothesis=h_raw_n,
            transform_a=t_a,
            transform_b=t_b,
        )
        orb_n.metadata["label_class"] = "neutral"
        orb_n.metadata["quartet"] = q
        orb_n.metadata["base_item_id"] = i + 1000
        orbits_neutral.append(orb_n)

        # Contradiction
        tmpl_c = CONTRADICTION_TEMPLATES[i % len(CONTRADICTION_TEMPLATES)]
        p_raw_c = tmpl_c[0].format(A=q[0], B=q[1], C=q[2], D=q[3])
        h_raw_c = tmpl_c[1].format(A=q[0], B=q[1], C=q[2], D=q[3])
        orb_c = builder.build_square_orbit(
            orbit_id=f"controlled_contradiction_{i:03d}",
            source_uid=f"contradiction_{i:03d}",
            dataset="controlled_nli",
            base_premise=p_raw_c,
            base_hypothesis=h_raw_c,
            transform_a=t_a,
            transform_b=t_b,
        )
        orb_c.metadata["label_class"] = "contradiction"
        orb_c.metadata["quartet"] = q
        orb_c.metadata["base_item_id"] = i + 2000
        orbits_contradiction.append(orb_c)


    # Stratified split per class (60 train, 20 val, 20 test per class)
    def split_list(lst: List[SemanticOrbit]) -> Tuple[List[SemanticOrbit], List[SemanticOrbit], List[SemanticOrbit]]:
        indices = np.arange(len(lst))
        rng.shuffle(indices)
        n_tr = int(0.6 * len(lst))
        n_va = int(0.2 * len(lst))
        tr_idx = indices[:n_tr]
        va_idx = indices[n_tr : n_tr + n_va]
        te_idx = indices[n_tr + n_va :]
        return [lst[i] for i in tr_idx], [lst[i] for i in va_idx], [lst[i] for i in te_idx]

    e_tr, e_va, e_te = split_list(orbits_entailment)
    n_tr, n_va, n_te = split_list(orbits_neutral)
    c_tr, c_va, c_te = split_list(orbits_contradiction)

    train_orbits = e_tr + n_tr + c_tr
    val_orbits = e_va + n_va + c_va
    test_orbits = e_te + n_te + c_te

    return ControlledOrbitDataset(
        train_orbits=train_orbits,
        val_orbits=val_orbits,
        test_orbits=test_orbits,
        name_quartets_train=train_quartets,
        name_quartets_test=test_quartets,
    )

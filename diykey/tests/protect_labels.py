# SPDX-License-Identifier: AGPL-3.0-or-later
"""Port of Protect's convertRamTagsToLabels (up/service-expanded/modules/587258_*.js, with the
lists from 594367_*.js, Protect 7.3.70): the Find Anything filter labels a thumbnail's RAM tags
produce for an object type. Test/verification helper only."""
LABEL_COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black",
                "white", "gray", "grey"]
UPPER = ["shirt", "hoodie", "coat", "suit"]
LOWER = ["skirt", "pant", "short"]
ACCESSORIES = ["glasses", "hat", "backpack", "handbag", "helmet", "mask"]
VEHICLE_TYPES = ["bike", "motorcycle", "sedan", "suv", "truck", "van", "bus"]
MALE, FEMALE = ["man", "boy"], ["woman", "girl"]


def convert_ram_tags_to_labels(object_type, tags):
    out = []
    for tag in tags:
        t = tag.lower()
        first, _, rest = t.partition(" ")
        adj, noun = (first, rest) if first in LABEL_COLORS else (None, t)
        m = []
        if adj:
            if noun in UPPER or noun == "clothing":
                m.append(f"upperClothingColor:{adj}")
            if noun in LOWER:
                m.append(f"lowerClothingColor:{adj}")
        acc = {"glass": "glasses"}.get(noun, noun)
        if acc in ACCESSORIES:
            m.append(f"accessory:{acc}")
        if noun in UPPER or noun in LOWER:
            m.append(f"clothingType:{noun}")
        if noun in MALE:
            m.append("gender:male")
        if noun in FEMALE:
            m.append("gender:female")
        if object_type == "vehicle" and noun in VEHICLE_TYPES:
            m.append(f"vehicleType:{'car' if noun == 'sedan' else noun}")
            if adj:
                m.append(f"color:{adj}")
        out.extend(x for x in m if x not in out)
    return out

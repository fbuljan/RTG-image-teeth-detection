"""Deterministic fake names for registry persons.

Maps a `person_id` (string) to a stable `<Adjective> <Noun>` two-word name
chosen from short word lists. The mapping is deterministic per person_id, so
the same person always shows the same name across server restarts.

Names are obviously fake (e.g. "Brown Computer", "Velvet Compass") to avoid
any confusion with real patient identities while remaining easy to remember
during a demo.
"""

from __future__ import annotations

import hashlib

ADJECTIVES = [
    "Amber", "Brave", "Brown", "Bright", "Calm", "Clever", "Cosmic",
    "Crimson", "Curious", "Daring", "Dusty", "Eager", "Electric", "Faint",
    "Fancy", "Fierce", "Frozen", "Gentle", "Glass", "Glowing", "Golden",
    "Grand", "Green", "Happy", "Hidden", "Humble", "Icy", "Indigo",
    "Iron", "Ivory", "Jolly", "Keen", "Kind", "Lazy", "Lemon",
    "Little", "Loud", "Lucky", "Mellow", "Mighty", "Misty", "Modern",
    "Muddy", "Neat", "Noble", "Olive", "Orange", "Plain", "Polite",
    "Proud", "Purple", "Quick", "Quiet", "Rapid", "Royal", "Rusty",
    "Sandy", "Sharp", "Shiny", "Silent", "Silly", "Silver", "Slow",
    "Smart", "Soft", "Solid", "Sour", "Spicy", "Steady", "Stormy",
    "Sunny", "Sweet", "Tall", "Tame", "Tiny", "Tough", "Trusty",
    "Ultra", "Vast", "Velvet", "Wandering", "Warm", "Wild", "Wise",
    "Witty", "Yellow", "Young", "Zesty",
]

NOUNS = [
    "Anchor", "Apple", "Arrow", "Atlas", "Badger", "Beacon", "Bell",
    "Berry", "Bird", "Boat", "Bridge", "Cabin", "Camera", "Candle",
    "Canyon", "Cat", "Cedar", "Chair", "Cherry", "Cliff", "Clock",
    "Cloud", "Clover", "Coast", "Comet", "Compass", "Computer", "Cookie",
    "Coral", "Crow", "Crystal", "Diamond", "Dolphin", "Dream", "Eagle",
    "Echo", "Falcon", "Feather", "Fern", "Flame", "Fog", "Forest",
    "Fountain", "Fox", "Galaxy", "Garden", "Gazelle", "Glacier", "Globe",
    "Harbor", "Harp", "Hawk", "Heron", "Hill", "Horizon", "Iceberg",
    "Island", "Jasmine", "Jay", "Journey", "Kite", "Lagoon", "Lake",
    "Lantern", "Leaf", "Lemon", "Lighthouse", "Lily", "Lion", "Lotus",
    "Maple", "Marble", "Meadow", "Melon", "Meteor", "Mirror", "Mist",
    "Moon", "Mountain", "Nest", "Nova", "Oak", "Ocean", "Olive",
    "Orchard", "Otter", "Owl", "Palm", "Pearl", "Pebble", "Pencil",
    "Pine", "Planet", "Pond", "Prairie", "Quartz", "Rain", "Raven",
    "Ridge", "River", "Robin", "Sage", "Sailor", "Sapphire", "Scout",
    "Sea", "Seal", "Shadow", "Shell", "Sky", "Sparrow", "Spruce",
    "Star", "Stone", "Stream", "Summit", "Sun", "Swan", "Thorn",
    "Tiger", "Trail", "Tulip", "Valley", "Vine", "Violet", "Voyage",
    "Walnut", "Wave", "Whale", "Willow", "Wind", "Wolf", "Wren",
]


def fake_name_for(person_id: str) -> str:
    """Return a stable two-word fake name for the given person_id."""
    digest = hashlib.sha256(person_id.encode("utf-8")).digest()
    adj = ADJECTIVES[digest[0] % len(ADJECTIVES)]
    noun = NOUNS[digest[1] % len(NOUNS)]
    return f"{adj} {noun}"


def disambiguate(names_by_id: dict[str, str]) -> dict[str, str]:
    """Append a numeric suffix when two distinct person_ids collide on a name.

    The first occurrence keeps the bare name; subsequent ones get " 2", " 3", …
    so the disambiguated names are still readable.
    """
    counts: dict[str, int] = {}
    out: dict[str, str] = {}
    for pid, name in names_by_id.items():
        counts[name] = counts.get(name, 0) + 1
        out[pid] = name if counts[name] == 1 else f"{name} {counts[name]}"
    return out


if __name__ == "__main__":
    # Quick sanity check — print 10 sample names
    for i in range(10):
        sample_id = f"person_{i:04d}"
        print(f"{sample_id} -> {fake_name_for(sample_id)}")

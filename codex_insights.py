import random
from typing import List

def infer_beneficiaries(text: str) -> List[str]:
    """Return a list of beneficiaries; never a single string."""
    text_l = text.lower()
    keywords = {
        "protest": "opposition groups",
        "strike": "labor unions",
        "sanctions": "rival states",
        "election": "political parties",
        "pipeline": "energy companies",
        "military": "defense contractors",
        "ai": "tech firms",
        "riot": "radical factions",
        "budget": "government insiders",
    }
    matches = [val for key, val in keywords.items() if key in text_l]
    # unique + stable order
    uniq = list(dict.fromkeys(matches))
    return uniq if uniq else ["Unclear"]

def simulate_scenarios(text: str, k: int = 2) -> List[str]:
    """Return k scenario descriptions as a list of strings (no dicts)."""
    templates = [
        "Public backlash increases, leading to policy shifts",
        "Narrative is controlled via media spin, minimal change",
        "Situation escalates into international tension",
        "Event is downplayed, fades from attention",
    ]
    # random.sample already handles k <= len(templates)
    return random.sample(templates, k)

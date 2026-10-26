from collections.abc import Sequence

from memory import manager
from memory.proposals import MemoryProposal


def save_proposals(proposals: Sequence[MemoryProposal]) -> tuple[int, int]:
    for proposal in proposals:
        manager.normalize_memory_text(proposal["file"], proposal["text"])
    saved = sum(manager.add(proposal["file"], proposal["text"]) for proposal in proposals)
    return saved, len(proposals) - saved

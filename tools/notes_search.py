from pathlib import Path

from memory.manager import user_data_dir


def search_notes(query: str) -> list[dict[str, str]]:
    terms = {term.casefold() for term in query.split() if term}
    if not terms:
        return []
    notes_dir = user_data_dir() / "notes"
    if not notes_dir.exists():
        return []
    if notes_dir.is_symlink() or not notes_dir.resolve().is_relative_to(notes_dir.parent):
        raise ValueError("Notes must stay inside the user's data folder.")
    matches: list[dict[str, str]] = []
    for path in sorted(notes_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.stat().st_size > 256_000:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if terms.intersection(word.casefold().strip(".,!?;:()[]{}") for word in line.split()):
                matches.append({"file": path.name, "line": str(line_number), "text": line})
                if len(matches) >= 10:
                    return matches
    return matches

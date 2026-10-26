from memory.manager import user_data_dir

MAX_MATCHES = 5
SNIPPET_LENGTH = 220


def search_notes(query: str) -> list[dict[str, str]]:
    terms = {term.casefold() for term in query.split() if term}
    if not terms:
        return []
    notes_dir = user_data_dir() / "notes"
    if notes_dir.is_symlink() or not notes_dir.resolve().is_relative_to(notes_dir.parent):
        raise ValueError("Notes must stay inside the user's data folder.")
    if not notes_dir.exists():
        return []
    matches: list[dict[str, str]] = []
    for path in sorted(notes_dir.iterdir()):
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.stat().st_size > 256_000:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            words = {
                word.casefold().strip(".,!?;:()[]{}")
                for word in line.split()
            }
            if terms.intersection(words):
                snippet = line.strip()
                if len(snippet) > SNIPPET_LENGTH:
                    snippet = snippet[: SNIPPET_LENGTH - 1].rstrip() + "…"
                matches.append(
                    {
                        "file": path.name,
                        "line": str(line_number),
                        "snippet": snippet,
                    }
                )
                if len(matches) >= MAX_MATCHES:
                    return matches
    return matches

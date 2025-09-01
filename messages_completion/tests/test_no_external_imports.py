import pathlib


def test_no_external_src_or_template_imports():
    root = pathlib.Path(__file__).resolve().parents[1]
    forbidden = ("from src.", "ChatTemplateManager")
    py_files = list(root.rglob("*.py"))
    # Exclude tests that may reference these strings intentionally (none do now)
    for fpath in py_files:
        text = fpath.read_text(encoding="utf-8")
        for needle in forbidden:
            assert needle not in text, f"Forbidden reference '{needle}' found in {fpath}"

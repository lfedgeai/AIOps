import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {ROOT / "tier4" / "runner.py", ROOT / "tier4" / "cli.py"}


def _imports(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
    return mods


def _sources():
    for p in ROOT.rglob("*.py"):
        if any(part in (".venv", "tests", "testbed") for part in p.relative_to(ROOT).parts):
            continue
        yield p


def test_only_runner_and_cli_import_the_executor():
    bad = [str(p.relative_to(ROOT)) for p in _sources()
           if p not in ALLOWED and p.name != "executor.py" and "tier4.executor" in _imports(p)]
    assert bad == []


def test_no_tier4_module_imports_jev_client():
    bad = [str(p.relative_to(ROOT)) for p in (ROOT / "tier4").glob("*.py")
           if any(m.startswith("jev") for m in _imports(p))]
    assert bad == []


def test_lower_tiers_never_import_tier4():
    bad = [str(p.relative_to(ROOT)) for p in _sources()
           if p.relative_to(ROOT).parts[0] != "tier4" and any(m.startswith("tier4") for m in _imports(p))]
    assert bad == []

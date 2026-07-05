import re
import sys
from pathlib import Path
import yaml

DEFINE_PATHS_FILENAME = "define_configs.yaml"

# Placeholder that IS the entire scalar value on a line (unquoted),
# for "key: ${var}" or "- ${var}" forms.
FULL_LINE_RE = re.compile(r'^(?P<prefix>\s*(?:[\w.\-]+\s*:|-)\s*)\$\{(?P<var>\w+)\}\s*$')

# Any ${var} occurrence, for embedded/quoted substitution.
PLACEHOLDER_RE = re.compile(r"\$\{(\w+)\}")

def load_variables(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def render_scalar(value) -> str:
    """Render a python value as a bare YAML scalar literal (e.g. False -> false)."""
    dumped = yaml.safe_dump(value, default_flow_style=True).strip()
    if dumped.endswith("..."):
        dumped = dumped[:-3].strip()
    return dumped


def substitute_text(text: str, variables: dict, source_file: str) -> str:
    lines = text.splitlines(keepends=True)
    out_lines = []
    missing = []

    for line in lines:
        full_match = FULL_LINE_RE.match(line)
        if full_match:
            var_name = full_match.group("var")
            if var_name not in variables:
                missing.append(var_name)
                out_lines.append(line)
                continue

            ending = ""
            if line.endswith("\r\n"):
                ending = "\r\n"
            elif line.endswith("\n"):
                ending = "\n"

            new_line = (
                f"{full_match.group('prefix')}"
                f"{render_scalar(variables[var_name])}"
                f"{ending}"
            )
            out_lines.append(new_line)
            continue

        def _sub(m):
            var_name = m.group(1)
            if var_name not in variables:
                missing.append(var_name)
                return m.group(0)
            return str(variables[var_name])

        out_lines.append(PLACEHOLDER_RE.sub(_sub, line))

    if missing:
        raise KeyError(
            f"Undefined variable(s) {sorted(set(missing))} in {source_file} "
            f"(not found in {DEFINE_PATHS_FILENAME})"
        )

    return "".join(out_lines)


def main():

    define_paths_file = Path(f"./configs/{DEFINE_PATHS_FILENAME}")
    configs_dir = Path("./configs")
    resolved_dir = Path("./configs_resolved")

    if not define_paths_file.exists():
        print(f"ERROR: {define_paths_file} not found.")
        sys.exit(1)
    if not configs_dir.exists():
        print(f"ERROR: {configs_dir} not found.")
        sys.exit(1)

    resolved_dir.mkdir(exist_ok=True)
    variables = load_variables(define_paths_file)

    config_files = sorted(
        p for p in configs_dir.rglob("*.yaml") if p.name != DEFINE_PATHS_FILENAME
    )
    if not config_files:
        print(f"No yaml files found in {configs_dir} (excluding {DEFINE_PATHS_FILENAME})")
        return

    had_error = False
    for cfg_file in config_files:
        rel_path = cfg_file.relative_to(configs_dir)
        text = cfg_file.read_text()

        try:
            resolved_text = substitute_text(text, variables, str(rel_path))
        except KeyError as e:
            print(f"ERROR in {rel_path}: {e}")
            had_error = True
            continue

        out_path = resolved_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(resolved_text)

        print(f"Resolved: {rel_path} -> {out_path}")

    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
import pathlib
import tomllib


def read_pyproject_version(pyproject_path: pathlib.Path) -> str:
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)

    # PEP 621
    if "project" in data and isinstance(data["project"], dict) and "version" in data["project"]:
        return str(data["project"]["version"])

    # Poetry
    tool = data.get("tool", {})
    poetry = tool.get("poetry", {}) if isinstance(tool, dict) else {}
    if isinstance(poetry, dict) and "version" in poetry:
        return str(poetry["version"])

    raise SystemExit("Version not found in pyproject.toml. Expected [project].version or [tool.poetry].version")


def sync_cargo_version(cargo_path: pathlib.Path, version: str) -> None:
    text = cargo_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    in_package = False
    replaced = False
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            # entering/leaving section
            in_package = stripped == "[package]"

        if in_package and stripped.startswith("version"):
            # keep indentation style
            indent = line[: len(line) - len(line.lstrip())]
            new_lines.append(f'{indent}version = "{version}"')
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise SystemExit(f'Could not find "version = ..." inside [package] in {cargo_path}')

    end_newline = "\n" if text.endswith("\n") else ""
    cargo_path.write_text("\n".join(new_lines) + end_newline, encoding="utf-8")


def main() -> None:
    root = pathlib.Path(".")
    pyproject_path = root / "pyproject.toml"
    cargo_path = root / "rust" / "Cargo.toml"

    version = read_pyproject_version(pyproject_path)
    sync_cargo_version(cargo_path, version)

    print(f"Synced rust/Cargo.toml [package].version to {version}")


if __name__ == "__main__":
    main()

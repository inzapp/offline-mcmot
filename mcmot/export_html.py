from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


ASSET_ATTRIBUTE = re.compile(r'(?P<prefix>\b(?:src|href)\s*=\s*["\'])(?P<url>[^"\']+)(?P<suffix>["\'])', re.IGNORECASE)


def _asset_data_uri(html_dir: Path, url: str, cache: dict[Path, str]) -> str | None:
    parts = urlsplit(url)
    if parts.scheme or parts.netloc or url.startswith(("#", "data:")):
        return None
    path = (html_dir / unquote(parts.path)).resolve()
    try:
        path.relative_to(html_dir.resolve().parent)
    except ValueError:
        return None
    if not path.is_file():
        return None
    if path not in cache:
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        cache[path] = f"data:{mime};base64,{encoded}"
    return cache[path] + (f"#{parts.fragment}" if parts.fragment else "")


def export_standalone_html(source: Path, target: Path) -> Path:
    """Embed local src/href assets so the resulting HTML can be sent alone."""
    source, target = source.resolve(), target.resolve()
    content = source.read_text(encoding="utf-8")
    cache: dict[Path, str] = {}

    def replace(match: re.Match) -> str:
        uri = _asset_data_uri(source.parent, match.group("url"), cache)
        return match.group("prefix") + (uri or match.group("url")) + match.group("suffix")

    exported = ASSET_ATTRIBUTE.sub(replace, content)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(exported, encoding="utf-8")
    return target


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Create a standalone HTML file with embedded local assets")
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path, nargs="?")
    args = parser.parse_args(argv)
    target = args.target or args.source.with_name(f"{args.source.stem}_export{args.source.suffix}")
    print(export_standalone_html(args.source, target))


if __name__ == "__main__":
    main()

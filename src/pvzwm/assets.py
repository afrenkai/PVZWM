from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

WIKI_API = "https://plantsvszombies.wiki.gg/api.php"
DEFAULT_PVZ1_ICON_TITLES = (
    "File:Peashooter1.png",
    "File:Sunflower1.png",
    "File:Wall-nut1.png",
    "File:Zombie1.png",
)


@dataclass(frozen=True, slots=True)
class WikiAsset:
    title: str
    filename: str
    url: str
    description_url: str
    source_sha1: str
    source_size: int
    width: int
    height: int
    license_short_name: str | None = None
    usage_terms: str | None = None
    local_sha256: str | None = None


def _metadata_value(metadata: dict[str, object], name: str) -> str | None:
    value = metadata.get(name)
    if not isinstance(value, dict):
        return None
    raw = value.get("value")
    return str(raw) if raw else None


def parse_mediawiki_assets(payload: bytes) -> list[WikiAsset]:
    value = json.loads(payload)
    pages = value.get("query", {}).get("pages", {})
    assets: list[WikiAsset] = []
    for page in pages.values():
        if "missing" in page:
            raise ValueError(f"Wiki file does not exist: {page.get('title')}")
        information = page["imageinfo"][0]
        metadata = information.get("extmetadata", {})
        title = str(page["title"])
        assets.append(
            WikiAsset(
                title=title,
                filename=title.removeprefix("File:").replace("/", "_"),
                url=str(information["url"]),
                description_url=str(information["descriptionurl"]),
                source_sha1=str(information["sha1"]),
                source_size=int(information["size"]),
                width=int(information["width"]),
                height=int(information["height"]),
                license_short_name=_metadata_value(metadata, "LicenseShortName"),
                usage_terms=_metadata_value(metadata, "UsageTerms"),
            )
        )
    return sorted(assets, key=lambda asset: asset.title)


def query_wiki_assets(
    titles: tuple[str, ...] = DEFAULT_PVZ1_ICON_TITLES,
    opener: Callable[..., object] = urllib.request.urlopen,
) -> list[WikiAsset]:
    parameters = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "imageinfo",
        "iiprop": "url|sha1|size|extmetadata",
        "format": "json",
    }
    url = WIKI_API + "?" + urllib.parse.urlencode(parameters)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "PVZWM/0.1 local research asset provenance"},
    )
    with opener(request, timeout=30) as response:
        return parse_mediawiki_assets(response.read())


def sync_wiki_assets(
    output: str | Path,
    *,
    download: bool = False,
    titles: tuple[str, ...] = DEFAULT_PVZ1_ICON_TITLES,
) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    assets = query_wiki_assets(titles)
    manifest_assets: list[dict[str, object]] = []
    for asset in assets:
        record = asdict(asset)
        if download:
            request = urllib.request.Request(
                asset.url,
                headers={"User-Agent": "PVZWM/0.1 local research asset sync"},
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                content_type = response.headers.get_content_type()
                content = response.read()
            if content_type != "image/png" or not content.startswith(b"\x89PNG\r\n\x1a\n"):
                raise RuntimeError(f"Expected PNG content for {asset.title}, got {content_type}")
            destination = output_path / asset.filename
            destination.write_bytes(content)
            with Image.open(destination) as image:
                if image.size != (asset.width, asset.height):
                    raise RuntimeError(f"Dimension mismatch for {asset.title}")
            record["local_sha256"] = hashlib.sha256(content).hexdigest()
            record["downloaded_size"] = len(content)
            # wiki.gg may losslessly recompress PNG responses at the CDN. Keep
            # both comparisons explicit instead of claiming source-byte identity.
            record["source_size_matched"] = len(content) == asset.source_size
            record["source_sha1_matched"] = (
                hashlib.sha1(content).hexdigest() == asset.source_sha1
            )
        manifest_assets.append(record)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "retrieved_at_unix": time.time(),
        "source": WIKI_API,
        "downloaded": download,
        "redistribution": "local-only; consult each description_url for rights information",
        "assets": manifest_assets,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

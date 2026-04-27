from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests


REPO_ID = "gaetanbahl/Global-Scale-Road-Dataset"
REVISION = "main"
DEFAULT_SPLITS = ("train", "val", "in-domain-test")


def _api_tree_url(path: str) -> str:
    repo = quote(REPO_ID, safe="/")
    path = quote(path, safe="/")
    return f"https://huggingface.co/api/datasets/{repo}/tree/{REVISION}/{path}"


def _resolve_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/{quote(path, safe='/')}"


def _resolve_hf_token() -> str | None:
    for env_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        token = os.environ.get(env_name)
        if token:
            return token.strip()

    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text(encoding="utf-8").strip()
        if token:
            return token
    return None


def _request_json(url: str, session: requests.Session | None = None) -> list[dict]:
    http = session or requests
    response = http.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected HuggingFace API response: {json.dumps(payload)[:500]}")
    return payload


def _split_items(split: str, session: requests.Session | None = None) -> list[dict]:
    root_items = _request_json(_api_tree_url(split), session=session)
    out: list[dict] = []
    for item in root_items:
        path = str(item.get("path", ""))
        item_type = str(item.get("type", ""))
        if item_type == "directory":
            out.extend(_request_json(_api_tree_url(path), session=session))
        else:
            out.append(item)
    return out


def _tile_key(path: str) -> str | None:
    name = Path(path).name
    for suffix in ("_sat.png", "_gt.png", "_graph_gt.pickle", "_refine_gt_graph.p", "_refine_gt_graph_samplepoints.json"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def _select_tile_files(items: Iterable[dict], max_tiles: int | None, include_graph: bool) -> list[str]:
    tile_order: list[str] = []
    by_tile: dict[str, list[str]] = {}
    allowed_suffixes = ("_sat.png", "_gt.png")
    if include_graph:
        allowed_suffixes = allowed_suffixes + (
            "_graph_gt.pickle",
            "_refine_gt_graph.p",
            "_refine_gt_graph_samplepoints.json",
        )

    for item in items:
        path = str(item.get("path", ""))
        if not path.endswith(allowed_suffixes):
            continue
        key = _tile_key(path)
        if key is None:
            continue
        if key not in by_tile:
            by_tile[key] = []
            tile_order.append(key)
        by_tile[key].append(path)

    selected = tile_order if max_tiles is None else tile_order[: max(0, int(max_tiles))]
    out: list[str] = []
    for key in selected:
        out.extend(sorted(by_tile[key]))
    return out


def _download_file(path: str, out_root: Path, retries: int = 8, session: requests.Session | None = None) -> None:
    out_path = out_root / path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    url = _resolve_url(path)
    http = session or requests
    for attempt in range(1, retries + 1):
        try:
            with http.get(url, stream=True, timeout=(30, 240)) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_path.replace(out_path)
            return
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                raise
            time.sleep(min(60.0, 3.0 * attempt))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Global-Scale Road Dataset files from HuggingFace.")
    parser.add_argument("--out-root", default="data/GlobalScaleRoad", help="Output root; split folders are created inside it.")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Splits to download.")
    parser.add_argument(
        "--max-tiles-per-split",
        type=int,
        default=None,
        help="Download only the first N tiles from each split. Omit for full selected splits.",
    )
    parser.add_argument("--include-graph", action="store_true", help="Also download graph labels in addition to sat/gt masks.")
    parser.add_argument("--retries", type=int, default=8, help="Retry count for each file before failing.")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    total_files = 0
    with requests.Session() as session:
        session.headers.update({"User-Agent": "anza-lira-global-roads-downloader/1.0"})
        token = _resolve_hf_token()
        if token:
            session.headers.update({"Authorization": f"Bearer {token}"})
            print("Using HuggingFace token authentication.", flush=True)
        for split in args.splits:
            items = _split_items(split, session=session)
            paths = _select_tile_files(items, max_tiles=args.max_tiles_per_split, include_graph=bool(args.include_graph))
            print(f"{split}: downloading {len(paths)} files", flush=True)
            for index, path in enumerate(paths, start=1):
                _download_file(path, out_root, retries=max(1, int(args.retries)), session=session)
                if index == 1 or index == len(paths) or index % 50 == 0:
                    print(f"{split}: {index}/{len(paths)} {path}", flush=True)
            total_files += len(paths)
    print(f"Done. Downloaded/verified {total_files} files under {out_root}.", flush=True)


if __name__ == "__main__":
    main()

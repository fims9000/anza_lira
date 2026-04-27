from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


BASE_URL = "https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs"

AOI_FILES = {
    "AOI_2_Vegas": [
        "SN3_roads_train_AOI_2_Vegas.tar.gz",
        "SN3_roads_train_AOI_2_Vegas_geojson_roads_speed.tar.gz",
    ],
    "AOI_3_Paris": [
        "SN3_roads_train_AOI_3_Paris.tar.gz",
        "SN3_roads_train_AOI_3_Paris_geojson_roads_speed.tar.gz",
    ],
    "AOI_4_Shanghai": [
        "SN3_roads_train_AOI_4_Shanghai.tar.gz",
        "SN3_roads_train_AOI_4_Shanghai_geojson_roads_speed.tar.gz",
    ],
    "AOI_5_Khartoum": [
        "SN3_roads_train_AOI_5_Khartoum.tar.gz",
        "SN3_roads_train_AOI_5_Khartoum_geojson_roads_speed.tar.gz",
    ],
}


def _remote_size(session: requests.Session, url: str) -> int | None:
    response = session.head(url, timeout=60)
    response.raise_for_status()
    raw = response.headers.get("Content-Length")
    return int(raw) if raw else None


def _download(session: requests.Session, url: str, out_path: Path, retries: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    expected_size = _remote_size(session, url)
    if expected_size is not None and out_path.exists() and out_path.stat().st_size == expected_size:
        print(f"verified {out_path.name} ({expected_size} bytes)", flush=True)
        return

    part_path = out_path.with_suffix(out_path.suffix + ".part")
    if out_path.exists() and not part_path.exists():
        out_path.replace(part_path)

    for attempt in range(1, retries + 1):
        resume_at = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={resume_at}-"} if resume_at > 0 else {}
        mode = "ab" if resume_at > 0 else "wb"
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 240)) as response:
                if resume_at > 0 and response.status_code != 206:
                    resume_at = 0
                    mode = "wb"
                response.raise_for_status()
                downloaded = resume_at
                with part_path.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024 * 8):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        downloaded += len(chunk)
                if expected_size is not None and part_path.stat().st_size != expected_size:
                    raise RuntimeError(
                        f"incomplete download for {out_path.name}: "
                        f"{part_path.stat().st_size} != {expected_size}"
                    )
                part_path.replace(out_path)
                print(f"downloaded {out_path.name}", flush=True)
                return
        except Exception as exc:
            if attempt >= retries:
                raise
            wait = min(90.0, 5.0 * attempt)
            print(f"retry {attempt}/{retries} for {out_path.name}: {exc}; sleep {wait:.0f}s", flush=True)
            time.sleep(wait)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SpaceNet 3 Roads tarballs from the public S3 mirror.")
    parser.add_argument("--out-root", default="data/SpaceNet3_Roads/tarballs")
    parser.add_argument("--aoi", nargs="+", default=["AOI_3_Paris"], choices=sorted(AOI_FILES))
    parser.add_argument("--retries", type=int, default=12)
    args = parser.parse_args()

    out_root = Path(args.out_root)
    with requests.Session() as session:
        session.headers.update({"User-Agent": "anza-lira-spacenet3-downloader/1.0"})
        for aoi in args.aoi:
            for filename in AOI_FILES[aoi]:
                url = f"{BASE_URL}/{filename}"
                print(f"{aoi}: {filename}", flush=True)
                _download(session, url, out_root / filename, retries=max(1, int(args.retries)))


if __name__ == "__main__":
    main()

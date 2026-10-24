"""Replay verified S9 release using its weights and privately supplied 1mH2Data.7z.

The test is on synthetic pixel images rendered from expert clicked track geometry.
It does NOT validate segmentation of real bubble-chamber BMP photographs.
"""
from __future__ import annotations
import argparse,hashlib,json,shutil,sys,tempfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from scripts.research.bubble_chamber.extract_7z_safe import extract
from scripts.research.bubble_chamber.s9_verify_checkpoints import verify
from scripts.research.bubble_chamber.s7_original_retrain import verify_sources

def validate_checksums(package:Path)->int:
    checkfile=package/"SHA256SUMS.txt"
    if not checkfile.is_file():raise FileNotFoundError(checkfile)
    n=0
    for line in checkfile.read_text().splitlines():
        digest,relative=line.split("  ",1)
        path=(package/relative).resolve()
        if not path.is_relative_to(package.resolve()):raise ValueError("Unsafe inventory path")
        if not path.is_file():raise FileNotFoundError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError("Checksum mismatch: "+relative)
        n+=1
    if n<45:raise ValueError("Incomplete S9 release")
    return n

def replay(package:Path,archive:Path)->dict:
    verify_sources(ROOT)
    package=package.resolve()
    n=validate_checksums(package)
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory(prefix="s9_replay_") as tmp:
        work=Path(tmp)
        measurements=work/"measurements"
        extract(archive,measurements)
        data=measurements/"1mH2Data"/"S1"
        if not data.is_dir():raise ValueError("Missing visualMetro 1mH2Data/S1 from source archive")
        results=work/"results"
        for seed in (20261024,20261025,20261026):
            target=results/f"s9_replication_seed{seed}"
            target.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(package/"research"/f"seed{seed}"/"metrics.json",
                            target/"metrics.json")
            for model in ("plain_unet","original_azunet"):
                dest=target/model
                dest.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(package/"weights"/f"seed{seed}"/model/"best_final.pt",
                                dest/"best_final.pt")
        rows=[verify(seed,data,results) for seed in (20261024,20261025,20261026)]
    return {"verified_checksum_files":n,"model_checkpoints_replayed":len(rows)*2,
            "all_reproduced":all(row["pass"] for row in rows),
            "synthetic_pixel_truth_only":True,
            "real_camera_segmentation_proven":False,"replay":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--package",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(replay(a.package,a.archive),indent=2))

if __name__=="__main__":main()

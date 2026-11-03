"""Experimental image-conditioned full-volume Graph-LIRA with 3D ANZA/CNN.

This adapter preserves frozen K10/15 geometry, PAIR fusion identity,
JUNCTION geometry identity, tau=0.85, 15 perturbations and 0.60 consistency.
It substitutes the CT relation-evidence score and clean relation head with
patient-OOF-trained 3D CNN/ANZA evidence.

It does NOT establish equivalence to K10/15's augmented relation training.
No clinical or held-out performance claim is made.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import run_ct28_graph_lira as graph
from ccta_3d_an_patches import PatchConfig,extract_candidate_patch
from ccta_3d_k10 import (
    K10Inference, verified_model_pack, IDENTITY, EVAL_TAU,
)
from models.ccta3d_candidate_encoder import Candidate3dEncoder


def sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda:source.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


class NeuralCCTA3dInference(K10Inference):
    def __init__(
        self,
        models:dict,
        *,
        pair_encoder:torch.nn.Module,
        junction_encoder:torch.nn.Module,
        patch_config:PatchConfig,
        variant:str,
        device:str="cpu",
        pack_sha:str="synthetic_test_only",
        min_inside_fraction:float=.75,
        pair_identity_source:str="frozen_fusion",
    ):
        super().__init__(models,pack_sha=pack_sha)
        if variant not in {"anza3d","conv3d"}:
            raise ValueError("Unknown 3D neural encoder variant")
        if not 0<=min_inside_fraction<=1:
            raise ValueError("Invalid minimum CT patch in-volume fraction")
        if pair_identity_source not in {"frozen_fusion","neural_pair"}:
            raise ValueError("PAIR identity source must be frozen_fusion or neural_pair")
        self.pair_identity_source=pair_identity_source
        self.variant=variant
        self.device=device
        self.patch_config=patch_config
        self.min_inside_fraction=min_inside_fraction
        self.encoders={
            "PAIR":pair_encoder.to(device).eval(),
            "JUNCTION":junction_encoder.to(device).eval(),
        }

    @classmethod
    def from_experiment(
        cls,
        *,
        k10_pack:Path,
        relation_dir:Path,
        pair_oof_dir:Path,
        junction_oof_dir:Path,
        variant:str,
        device:str="cpu",
        pair_identity_source:str="frozen_fusion",
    ):
        base,base_sha=verified_model_pack(k10_pack)
        relation=json.loads((relation_dir/"relation_protocol.json").read_text())
        if relation.get("status")!="experimental_clean_relation_head_no_consistency_gate":
            raise ValueError("Neural relation protocol not approved for experimental mode")
        if relation["variant"]!=variant or relation["source_K10_model_sha256"]!=base_sha:
            raise ValueError("Neural relation head mismatches baseline variant/pack")
        relation_head_file=relation_dir/"clean_relation_head.joblib"
        if sha256(relation_head_file)!=relation["model_sha256"]:
            raise ValueError("Neural relation head SHA differs from manifest")
        relation_head=joblib.load(relation_head_file)
        full_models={}
        patch_configs=[]
        for kind,directory,expected_source in (
            ("PAIR",pair_oof_dir,relation["pair_oof_sha256"]),
            ("JUNCTION",junction_oof_dir,relation["junction_oof_sha256"]),
        ):
            meta=json.loads((directory/"oof_protocol.json").read_text())
            if meta["variant"]!=variant or meta["candidate_type"]!=kind:
                raise ValueError("OOF neural candidate model type/variant mismatch")
            if meta["status"]!="train_patient_OOF_experimental_CT_evidence_not_promoted_K10":
                raise ValueError("OOF model status is not approved research evidence")
            if meta.get("source_input_control_mode","full_ct")!="full_ct":
                raise ValueError("Cannot deploy endpoint-only or scrambled-CT model as full 3D CCTA")
            if meta.get("held_out_test_accessed") is not False:
                raise ValueError("Unsealed medical test evidence in OOF pack")
            scores=directory/f"{kind}_{variant}_oof_val.csv"
            if sha256(scores)!=expected_source or sha256(scores)!=meta["score_sha256"]:
                raise ValueError("OOF source SHA changed after relation training")
            weights=directory/f"{kind}_{variant}_fulltrain.pt"
            if sha256(weights)!=meta["fulltrain_model_sha256"]:
                raise ValueError("OOF neural encoder weights changed")
            cfg=meta["train_patch_sources"][0]["patch_protocol"]
            if cfg!=meta["validation_patch_sources"][0]["patch_protocol"]:
                raise ValueError("Train and validation candidate-patch protocols differ")
            patch_configs.append(cfg)
            neural=Candidate3dEncoder(
                variant,voxel_mm=float(cfg["voxel_mm"])
            )
            state=torch.load(weights,map_location="cpu",weights_only=True)
            neural.load_state_dict(state,strict=True)
            full_models[kind]=neural
        if patch_configs[0]!=patch_configs[1]:
            raise ValueError("PAIR and JUNCTION neural patches use different preprocessing")
        pack=dict(base)
        pack["relation_heads"]=dict(base["relation_heads"])
        pack["relation_heads"]["ct_conditioned"]=relation_head
        return cls(
            pack,pair_encoder=full_models["PAIR"],
            junction_encoder=full_models["JUNCTION"],
            patch_config=PatchConfig(**patch_configs[0]),
            variant=variant,device=device,
            pack_sha=base_sha,
            pair_identity_source=pair_identity_source,
        )

    def _score(
        self,
        scene,
        endpoints:pd.DataFrame,
        candidates:pd.DataFrame,
        ct:np.ndarray,
        inverse_affine:np.ndarray,
    ):
        lookup=endpoints.set_index("endpoint_id")
        feat=pd.DataFrame([
            graph.perturbed_candidate_features(
                row,lookup,ct,inverse_affine,self.arm_names
            ) for _,row in candidates.iterrows()
        ])
        scores=self._score_sparse_candidate_types(
            feat,self.models["local_models"]
        )
        if scores[["candidate_id","candidate_type"]].duplicated().any():
            raise ValueError("Duplicate candidate identity in neural CCTA scoring")
        affine=np.linalg.inv(inverse_affine)
        by_kind={"PAIR":[],"JUNCTION":[]}
        for ix,row in enumerate(candidates.itertuples()):
            degree=2 if row.candidate_type=="PAIR" else 3
            ids=str(row.endpoint_ids).split("|")
            if len(ids)!=degree:
                raise ValueError("Neural score candidate degree mismatch")
            points=lookup.loc[ids,["x_lps","y_lps","z_lps"]].to_numpy(dtype=float)
            patch,audit=extract_candidate_patch(
                ct,affine,points,self.patch_config
            )
            if audit["sampling_inside_volume_fraction"]<self.min_inside_fraction:
                raise ValueError("CT neural patch leaves image support region")
            by_kind[str(row.candidate_type)].append((ix,patch))
        for kind,items in by_kind.items():
            if not items:
                continue
            model=self.encoders[kind]
            indices=[i for i,_ in items]
            patches=np.stack([p for _,p in items]).astype(np.float32)
            outputs=[]
            with torch.no_grad():
                for start in range(0,len(patches),8):
                    batch=torch.from_numpy(patches[start:start+8]).to(self.device)
                    outputs.extend(
                        torch.sigmoid(model(batch)).detach().cpu().numpy().tolist()
                    )
            for index,value in zip(indices,outputs):
                if not np.isfinite(value) or not 0<=value<=1:
                    raise ValueError("Nonfinite neural CT candidate score")
                scores.loc[scores.index[index],"ct_score"]=float(value)
                if kind=="PAIR" and self.pair_identity_source=="neural_pair":
                    # Explicit image-encoder identity experiment. The original
                    # K10/15 geometry+CT PAIR identity remains the default.
                    # Keep the historical column and chooser contract, but
                    # replace its numerical evidence for PAIR *only*.
                    scores.loc[scores.index[index],"geometry_plus_ct_score"]=float(value)

        plan=pd.DataFrame([{
            "scene_id":scene.scene_id,"scan_id":-1,
            "split":"unlabelled_runtime",
            "scene_family":"unlabelled_runtime",
            "relation_class":"UNLABELLED",
        }])
        predictions=graph.predict_relations(
            plan,candidates,scores,self.models,EVAL_TAU
        )
        decisions=graph.select_scene_decisions(
            plan,candidates,scores,predictions,IDENTITY
        )
        return scores,{model:result[scene.scene_id]
                       for model,result in decisions.items()}

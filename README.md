# ANZA-LIRA

Clean repo entry point for current paper materials.

## Final paper packages

- Article 2 (medical, **FIVES only**):  
  [results/a2_final_package](results/a2_final_package)
- Article 3 (cross-domain: Roads_HF + SpaceNet3 + HRF):  
  [results/a3_final_package](results/a3_final_package)
- Article 3 (public-safe, no metrics):  
  [results/a3_final_package/a3_math_impl_only.md](results/a3_final_package/a3_math_impl_only.md)

## Train

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py --config configs/fives_benchmark.yaml --variants baseline,az_thesis
```

## Public code

`https://github.com/fims9000/anza_lira`

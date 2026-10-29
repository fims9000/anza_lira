# Tver 2026 | ANZA-LIRA: structural continuation

This branch contains the **Tver journal paper** workstream. It was branched from `research/anza-lira-q1-journal` to preserve the future Q1/2.5-D study without reopening or rewriting any frozen earlier experiment. It does NOT modify `main`, `research/varvara-ccta-graph-lira`, or the bubble-chamber branches.

## Manuscript boundary

Current manuscript: `АНЗА_ЛИРА_Тверь_журнальная_статья_v6_с_докладом.docx` (2026-10-29, held as a conversation artifact). It integrates the 28-slide Zolotovskie Chteniya presentation, conference abstract, and frozen controlled experimental reports. The DOCX is **not stored in this Git branch** yet; `manuscript_v6_text_snapshot.txt` is a textual inspection copy, not an editable replacement for the DOCX.

**Claim permitted:** a decomposed ANZA/local-evidence → SBPP candidate → LIRA contextual acceptance/NONE → conditional maximin path framework, with precise controlled results, failure analysis, and limits.

**Claims NOT permitted:** externally validated natural geological fault identity; demonstrated end-to-end superiority against strong matched baselines; ANZA-specific Anosov effect; using separate stagewise metrics as a joint accuracy; treating Q1 plans as measured experiments.

## Frozen evidence (do not combine denominators)

| Study | Source unit | Result | Boundary |
| --- | --- | --- | --- |
| Historical balanced pair study | 128 positives, 128 negatives | AUROC 0.9923; 86/128 accepted positives; 1/128 false accepts | Controlled matching, NOT real geological external test |
| SBPP V3-B | 2688 positive simulated sources | CandidateRecall@12 2611/2688; weak 117/192 | Candidate-only; no P0 classifier in this study |
| TG1 Candidate Audit V2 | 1024 separate positive sources | endpoint Recall@6 818/1024; 206 misses, 148 endpoint shifts | Different frozen source/model; cannot compare directly to V3-B |
| TraceGraph P0 Endgame | 2688 positive + 1152 NONE source scenes | RR 1202/2688; FB 19/1152; AUROC 0.978948 | Source-disjoint DEVELOPMENT; failed relation gate; no E4 path |
| Controlled maximin oracle | correct pair already known | 128/128 success under that oracle setup | Path validation conditional on pair identity, not full system |

Sources are named in `claims.json`, with computation and consistency tests under `tools/tver2026` and `tests/test_tver2026.py`.

## Reproduction and scientific integrity

```bash
python tools/tver2026/evidence_audit.py --verify-repo . --out /tmp/tver_evidence.json
python tools/tver2026/background_rng_control.py --out /tmp/tver_rng.json
python -m unittest discover -s tests -p test_tver2026.py -v
```

The evidence audit verifies frozen ratios and source-text anchor tokens; it **does not replay weights/checkpoints**. The RNG control verifies that class-dependent structural-instance counts change shared RNG draw order before noise/phase sampling; it is not evidence of classifier cheating. Full paired training/evaluation with decoupled RNG streams and comparable per-candidate LIRA scores remains pending.

## What remains

1. Obtain/hash primary per-source logits, split manifests, checkpoints and reports; reproduce paper tables directly from immutable files.
2. Run one *new, explicitly separated* calibration/independent-holdout comparison B0/B1/LIRA on the same candidate pool, with source/section-grouped uncertainty. DO NOT unlock frozen STOP/confirm sets to do this.
3. Test synthetic rendering with independent RNG streams for geometry and nuisance background; report the change, including null/negative effects.
4. Final editorial checks: authors/affiliations, previously published ANZA scope, funding acknowledgement authorization, references, Word equations, and journal documents.

The separate Q1 `docs/research/anza_lira_q1_journal_spec.md` remains a prospective preregistered research plan, not retroactive Tver evidence.
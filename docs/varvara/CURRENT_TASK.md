# CURRENT TASK — CT-conditioned Graph-LIRA

Status updated: 2026-11-02. This is the current entry point; the 2026-10-06 task specification below is the historical execution plan.

Canonical branch:

`research/coronary-connectivity-repair`

## Current validation state

The CT28 pipeline has been built through structural repair/abstain and a separate path stage:

```text
PAIR local geometry + CCTA evidence
+
JUNCTION local geometry + CCTA evidence
        ->
NONE / PAIR / JUNCTION / BOTH
        ->
CT28 Graph-LIRA
        ->
tau = 0.85
consistency = 0.60
        ->
repair / abstain
```

The original execution specification is:

`docs/varvara/TASK_CT28_CT_CONDITIONED_GRAPH_LIRA_2026-10-06.md`

Read the current results first: `artifacts/varvara/ct28_graph_lira/README.md`. The specification records requirements and earlier planned steps; its unfinished wording does not override saved validation evidence.

## Selected development variant

The single current validation candidate is **K10/15**: local PAIR/JUNCTION geometry HGB with five position-only train copies, clean CT HGB, PAIR geometry+CT identity HGB with ten joint train copies, JUNCTION geometry identity, and an 84-feature relation HGB with fifteen joint train copies. Patient-level OOF train scores, `tau=0.85`, consistency `0.60`, and the 17/5/6 patient split are preserved.

On 368 validation scenes, 206 require repair. Before the gate: 155/206 structurally exact and 11/368 false. After the gate: **52 exact, 0 false**, including 2 `junction_only` and 2 `mixed`. This is exploratory validation on five patients, not a held-out test result or a frozen final protocol. The train-selected CT28 path v2 built routes for all 52 accepted scenes and improved branch-specific proximity, but this does not establish clinical or full-mask correctness. One of 34 train path scenes still fails because JUNCTION arms overlap outside the shared node.

Evidence and separate old results: `artifacts/varvara/ct28_graph_lira/README.md`, `ablations/README.md`, `consistency_diagnostics/README.md`, and `path/README.md` under that directory. The reproducible PAIR pack is `artifacts/varvara/ct28_pair/`; the frozen JUNCTION plan and feature provenance are in `artifacts/varvara/ct28_junction/`. Original JUNCTION VTK and segmentations are in the ignored `varvara/anza_lira_local_data/Varvara_CT28_junction_source_pack/`.

## Next work

User decision, 2026-11-02: stop the candidate CT path-evidence / endpoint-attachment improvement direction and retain **K10/15 as the selected working baseline**. Keep its weights, inputs, gates and saved numerical outputs unchanged. The path-evidence pilots and MLP/JS alternatives remain separate non-promoted experiments. No new experiment or test run is authorized by this decision.

The pre-test audit is recorded in `artifacts/varvara/ct28_graph_lira/README.md`. The local JUNCTION ranking and count-shortcut diagnostics did not yield a safer replacement for K10/15; its selected result remains 52/0, below the hoped-for 80 accepted exact repairs. Before opening held-out test once, review and freeze the new scene protocol despite the documented count shortcut, accept or revise the low JUNCTION/BOTH coverage, and state whether the test claim stops at structural/path metrics or requires a separately defined final-mask stage. Path v2 is recorded in `artifacts/varvara/ct28_graph_lira/path/selected_path_protocol.json`. Compare radial CT, compact CNN, and ANZA only after a stable end-to-end Graph-LIRA result.

## Test firewall

Do not open held-out patients for the current Graph-LIRA experiment until scene/relation/graph/path choices are frozen on train/validation. Earlier CT28 PAIR-only held-out results are a different completed experiment.

Held-out patients:

`954, 958, 972, 973, 980, 984`

Do not retune `tau=0.85` or consistency `0.60` on test.

# S84 preregistration: CAL-only photometric + graph-context selection of extra pixel lines

**2026-10-26; frozen before S84 CAL and any new TEST.** Experiment ID `ANZA-LIRA-BC-S84-20261026`. Earlier S80–S83 TEST results motivated this independent hypothesis but are NOT used to fit scores, select parameters, thresholds, or synthetic labels. The goal is to distinguish plausible additional 2D pixel-derived fragment instances from plausible-looking but graph-incompatible fragments. This cannot prove real H50Z particle identity.

## Locked inputs and inference boundary

- S72 frozen direction-dependent image-to-graph algorithm, S56 source TRAIN geometry logistic scorer, S56 CAL-only negative-link cutoff and global forest margin 0.04. S59 strict pair gate, S69 orientation-correct GT evaluator, S76 shape/endpoint evaluator and S80 consensus-add6 pixel proposals remain byte-identical.
- Image inference receives only uint8 192x192 grayscale pixels, frozen S56 scorer, frozen cutoff, S72 source-CAL direction choice, and two numeric thresholds calculated on SOURCE CAL pixels. It never receives simulator curves, injected identities, geometry GT, reference-film frame IDs, or physical provenance labels.
- Always preserve exact original S72 polyline objects in original order; all additional polylines remain reversible machine hypotheses. Never authorize physical automatic joins. All native particle GT remains UNKNOWN.
- S80 adds at most six consensus extra curves after symmetric duplicate suppression. A candidate's S83 **unsaturated** pixel-darkness support score is measured along its arclength independently of synthetic labels. Candidate graph context is based solely on frozen S56 pair probabilities against original S72 fragments and S59 strict admissible geometry.

## Frozen S84 arms (do NOT adjust after CAL or TEST)

1. `baseline`: original S72 only.
2. `s84_photo_q40`: S80 extras accepted only when S83 unsaturated pixel-support >= SOURCE CAL q40 (positive/negative source identities NOT used). This controls for any improvement explained by photometric gating alone.
3. `s84_context_single`: same q40 photometric gate, plus >=1 strict-S59 admissible extra-to-original edge whose frozen S56 probability exceeds the CAL global threshold by **0.04**. It must not occupy a port already selected by the baseline S72 global forest. For multiple viable original contacts, the best score must exceed the second highest by >= **0.03**; otherwise abstain.
4. `s84_context_bridge`: local support >= SOURCE CAL q25, and >=2 strict admissible extra-to-original edges using **distinct candidate endpoint ports** and **different original fragments**, both S56 probabilities >= global cutoff+**0.02**. Both original ports must be unused by the baseline global forest. If any selected contact shares an original port with an already chosen base link, abstain. The method adds the image-derived *line* only; it never forces these hypothesized links.

Exactly the same original S59/S56 global graph solver is rerun on any selected additional lines. Selection of which proposed graph edges to keep is always by the original global solver, not by synthetic correctness. Any solver failure causes no selected joins.

## Frozen CAL and fresh TEST

- VisualMetro film 070 145 measurement frames, split as whole frames 79 TRAIN, 30 CAL and 36 TEST, identical source-frame split hash `448f21ca3a709010ed972a596ed046cadff0643112905fd1d029c91f8b14faef`. Those measured lines are a different film/run from H50Z photos.
- Real images: exactly two private H50Z originals from ONE physical run; their source crop bank has 21 tiles (`005`) and 18 (`006`). Pixel crops, measured coordinates, private learned weights and individual scene outputs must not go into GitHub.
- SOURCE-photo CAL: source `005` seed **20268401**, source `006` seed **20268411**, 24 new scenes/direction, each under paired CLEAN and gaussian blur sigma=1.1. The q25 and q40 unsaturated scores are measured only on extra pixel-derived curves from these CAL scenes. No simulator GT is used in computing thresholds. CAL synthetic labels, known only post-inference, are used to preselect an arm.
- Held-out TEST: source `005` to opposite photo `006` seeds **20268403, 20268404**; source `006` to `005` seeds **20268413, 20268414**, 24 new scenes/seed = **96 scenes per condition**. Conditions: clean, Gaussian blur σ=1.1, σ=1.6. Different seeds from S80/S81/S82/S83. No threshold/model/arm tuning on TEST.
- CAL selection: on blur1.1, at least +2 synthetic correct joins and +2 S76 shape-matched true fragments relative to baseline, no increase in synthetic noncorrect joins and no loss of full artificial paths. On clean CAL, <=1 correct join lost, no increase of noncorrect joins, no loss of full paths. All solver calls must succeed. If none qualify, `baseline` is selected. Tie-break most blur correct, least blur noncorrect, fewest extras, alphabetic arm name. TEST may NEVER override CAL.

## Frozen measured outputs and uncertainty

For each arm, seed, direction and condition: total extracted lines, extra candidates considered/accepted/rejected, candidates rejected for photometric, graph weak, conflict or ambiguity, true injected fragment shape matches, endpoint-displaced fragment matches, corrected S69 synthetic-correct joins, noncorrect joins, full artificial three-piece paths, solver failures and scenes with any noncorrect link. No native H50Z physical precision/recall or true particle IDs. Report paired scene-bootstrap 1200 replicates by whole synthetic scene, not individual pixels or related H50Z crops; honest about both photo backgrounds from one run. Save public aggregate-only results, private frozen full CAL checksum, source code digests and experiment ID. Every native proposed link remains UNKNOWN and physical_auto_joins_authorized=false.

## Failure and interpretability

Any method that adds no candidates is measured as a no-op, not as a success. A candidate with graph evidence is NOT necessarily a real particle; geometric support/score can be fooled by scratches or background. Synthetic noncorrect includes unmatched/ambiguous predictions, not a proven false particle track. No 3D, momentum or species claim. Future tests on independent physical runs require authorized data and expert GT; none is invented here.

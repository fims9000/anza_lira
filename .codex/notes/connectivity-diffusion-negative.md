# Connectivity/diffusion feasibility stop (2026-08-18)

- CleanANZA formula audit passed; legacy v1 remains unchanged and uses categorical softmax memberships.
- A capacity-matched, pair-disjoint matched-gap probe found the minimum observable context at RF 9 (validation AUROC 0.8154296875).
- The frozen GT-connectivity oracle tested all 15 cells in T={1,2,4,6,8}, alpha={0.4,0.6,0.8} on v5 validation[0:512]. None passed.
- Maximum gap recovery was 0.279983 at T=8, alpha=0.8, with false bridge 0.265625 and visible Dice loss 0.166314.
- Root cause: restarted row-stochastic local averaging cannot create enough hidden-corridor foreground while preserving visible evidence; restart re-injects false bridges already present in h0.
- Status is `CONNECTIVITY_REPAIR_NEGATIVE_WITH_ROOT_CAUSE`. Do not train D0-D3, add D4, tune epsilon/threshold posthoc, or run CRACKS under this protocol.
- Any future cycle must predeclare materially different propagation mathematics and a new independent stream.


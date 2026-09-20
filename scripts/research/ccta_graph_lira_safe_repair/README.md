# Restoring archived research scripts

The scripts in this directory are stored as `.py.gz.b64` so the exact local exploratory source can be preserved through the GitHub text-content API without committing raw medical data.

Restore any script with:

```bash
base64 -d < script.py.gz.b64 | gzip -d > script.py
```

Then verify the SHA256 listed in `docs/research/ccta_graph_lira_safe_repair/CHECKPOINT.md` or the corresponding experiment manifest before treating a rerun as the same protocol.

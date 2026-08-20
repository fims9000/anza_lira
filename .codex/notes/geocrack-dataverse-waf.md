# GeoCrack Harvard Dataverse WAF

Observed on 2026-08-17 from the Linux workspace:

- DOI landing/citation page, dataset page, Native API, version API, metadata
  export, and OAI export all returned HTTP 202 with an empty body;
- the response header was `x-amzn-waf-action: challenge` from `awselb/2.0`;
- changing trailing slash, User-Agent, Accept header, and API form did not help;
- DataCite confirmed DOI metadata but exposed no media/file URLs;
- the official author GitHub has `patch_pairs.csv` and code, but no image data.

This is an external JavaScript WAF challenge, not a DOI/parser error. Preserve
`results/geocrack_study/download_attempt.log`.

On 2026-08-17 Harvard Dataverse also reported serious service-side technical
issues and a browser download was interrupted. Treat the real dataset as
temporarily unavailable. Preserve `.part`, `.crdownload`, and `.tmp` files
unchanged: do not rename, extract, or feed them to the importer. Do not start
automated Dataverse retries or attempt bypasses. A browser Resume may be tried
once only after service recovery; otherwise wait for a new completed official
archive.

After a completed official archive is stable, import it locally:

```bash
/home/lebedeffson/Code/venv/bin/python scripts/download_geocrack.py \
  --local-archive data/geocrack/incoming/geocrack_patched_data.zip
```

Do not substitute an unofficial mirror without explicit approval and checksum
provenance. The current study contract requires all 12,158 pairs. Using an
official partial Patched Data file would require an explicit protocol revision,
new expected count and provenance; never silently weaken the importer.

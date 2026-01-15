Release v1.0 — Suggested contents

Title: CUBO v1.0 — First public release

Changelog:
- First public release: 10 GB ingestion on a 16 GB laptop (test corpus from composite BEIR)
- GDPR-safe: fully offline data handling
- Added demo and one-click binaries (Windows .exe, Linux package)
- Benchmark summary and reproducible ingest scripts added to `paper/appendix/ingest`

Assets to attach to the GitHub Release:
- `CUBO.exe` (Windows single-file executable, ~180 MB)
- `CUBO_linux` (Linux onefile)
- Link to demo video (YouTube)

Suggested release notes (short):
> First public release of CUBO: local RAG for privacy-first teams. 10 GB ingestion on 16 GB RAM, reproducible benchmarks and one-click downloads.

How to create the release locally (if you prefer manual):
1. Tag the repo: `git tag -a v1.0 -m "CUBO v1.0"`
2. Push the tag: `git push origin v1.0`
3. Visit https://github.com/<owner>/<repo>/releases/new and upload the artifacts and the release notes.

CI note: we added `.github/workflows/release.yml` which builds artifacts on push of semantic tags (v*.*.*). Verify the artifacts in Actions and attach the `dist/` files to the release.
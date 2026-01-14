# CUBO Offline Logging & Search

This document describes how to use the offline logging engine in CUBO.

Features:
- Structured JSON logs
- Rotation and retention
- SQLite FTS index for offline search
- Trace id propagation across threads

Usage:
- Configure `config.json` logging block to set `format` to `json`, choose file path, and enable rotation/retention.
- Index logs incrementally with the `tools/log_indexer.py` script:

```pwsh
python tools/log_indexer.py --log-file ./logs/cubo_log.jsonl --db ./logs/index/logs.db
```

- Search logs using the CLI `tools/logcli.py`:

```pwsh
python tools/logcli.py --db ./logs/index/logs.db --query "error" --limit 20
```

Notes:
- Queries are run against an offline SQLite FTS5 database. This keeps everything local and offline.
- Trace IDs are auto-generated in `ServiceManager` for threaded tasks and are included in logs for correlation.

Privacy and Query Scrubbing:
- If you want to avoid storing plaintext user queries in logs, set `"scrub_queries": true` in `config.json` under the `logging` section.
- When enabled, queries will be replaced with a hash before being written to logs.
- To scrub queries from historical logs, use the `tools/scrub_logs.py` helper:

```pwsh
python tools/scrub_logs.py --input ./logs/cubo_log.jsonl --output ./logs/cubo_log.jsonl.scrubbed
```

This will write a new file with queries replaced by their hash.

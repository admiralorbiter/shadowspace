# Dependency Notes — Shadowspace Sprint 0

## dtour

| Field | Value |
|---|---|
| Package | dtour |
| Pinned version | 0.4.4 |
| License | MIT |
| Upstream | https://github.com/flekschas/dtour |
| Wheel hash (SHA-256) | *Record after first clean install — see instructions below* |

### Recording the wheel hash

After running `pip install -r requirements.txt`, capture the hash with:

```powershell
pip download dtour==0.4.4 --no-deps -d .pip-cache
Get-FileHash .pip-cache\dtour-0.4.4-py3-none-any.whl -Algorithm SHA256
```

Paste the output here and delete the `.pip-cache` directory.

---

## Other pinned packages

| Package | Min version | Notes |
|---|---|---|
| polars | 1.38.1 | Arrow-native dataframes; dtour's native format |
| numpy | 2.2.0 | Required by dtour; used for all matrix math |
| flask | 3.1.0 | Workbench server (Sprint 3b+) |
| pytest | 8.0.0 | Test runner |
| hypothesis | 6.100.0 | Property-based testing |
| ruff | 0.8.0 | Linter + formatter |
| mypy | 1.10.0 | Static type checker |
| jupyter | 1.1.0 | Notebook environment |

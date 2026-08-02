# Shadowspace Documentation Index

**This file is the documentation index only.** It is not the testing and validation document.

Read in this order:

1. [Project README](../README.md)
2. [Architecture and Data Contract](RESEARCH_ROADMAP.md) — system boundaries, domain model, bundle format, core interfaces
3. [Implementation Plan](ARCHITECTURE_AND_DATA_CONTRACT.md) — sprint plan, deliverables, exit gates (Sprints 0–16 complete)
4. [Testing and Validation](PROJECT_DECISIONS.md) — test layers, fixtures, invariant tests, and manual scripts (the August 1 snapshot records 176 passing tests and 87.3% coverage)
5. [Mathematical and Research Knowledge Base](README.md) — probability geometry, tours, reliability references
6. [Project Decisions](MATHEMATICAL_AND_RESEARCH_KNOWLEDGE_BASE.md) — ADR-001 through ADR-016
7. [ChaosNLI Research Package](studies/chaosnli/README.md) — study status, reports, methods, and protocols

The research snapshot is **August 1, 2026**. Recent 2026 works marked as preprints should be rechecked before publication or implementation claims.

## Recent API additions (Sprints 13–16)

| Endpoint | Method | Description |
|---|---|---|
| `/api/topology` | GET | k-NN edge classification (preserved / false / torn) |
| `/api/distortion-grid` | GET | 32×32 local distortion ratio grid with actual coordinate bounds |
| `/api/subspace-angles` | GET | Canonical principal angles and Grassmannian distance between two catalog views |
| `/api/point-stability` | GET | Vectorized per-point neighborhood persistence overlap ratios ($S_i$) across views |
| `/api/rashomon-set` | GET | Structurally diverse candidate projection bases on $\mathrm{Gr}(2, p)$ meeting quality threshold $\tau$ |
| `/api/import-record` | POST | Import & cryptographically validate InvestigationRecord JSON before view restoration |
| `/api/health` | GET | System health, sqlite-vec status, and hardening milestone completion |

## File integrity

If you suspect a file has been renamed, truncated, or mixed up, check the leading heading of each file:

| Filename | Expected heading |
|---|---|
| `../README.md` | Shadowspace |
| `RESEARCH_ROADMAP.md` | Shadowspace Architecture and Data Contract |
| `ARCHITECTURE_AND_DATA_CONTRACT.md` | Shadowspace Implementation Plan |
| `PROJECT_DECISIONS.md` | Shadowspace Testing and Validation |
| `README.md` | Shadowspace Mathematical and Research Knowledge Base |
| `MATHEMATICAL_AND_RESEARCH_KNOWLEDGE_BASE.md` | Shadowspace Project Decisions |
| `TESTING_AND_VALIDATION.md` | Shadowspace Documentation Index |


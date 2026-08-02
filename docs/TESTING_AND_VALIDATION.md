# Shadowspace Documentation Index

**This file is the documentation index only.** It is not the testing and validation document.

Read in this order:

1. [Project README](../README.md)
2. [Architecture and Data Contract](RESEARCH_ROADMAP.md) — system boundaries, domain model, bundle format, core interfaces
3. [Implementation Plan](ARCHITECTURE_AND_DATA_CONTRACT.md) — sprint plan, deliverables, exit gates (Sprints 0–13 complete)
4. [Testing and Validation](PROJECT_DECISIONS.md) — test layers, fixtures, invariant tests, manual scripts
5. [Mathematical and Research Knowledge Base](docs/README.md) — probability geometry, tours, reliability references
6. [Research Roadmap](README.md) — research questions, hypotheses, experiment sequence
7. [Project Decisions](MATHEMATICAL_AND_RESEARCH_KNOWLEDGE_BASE.md) — ADR-001 through ADR-014

The research snapshot is **August 1, 2026**. Recent 2026 works marked as preprints should be rechecked before publication or implementation claims.

## Sprint 13 API additions (current)

| Endpoint | Method | Description |
|---|---|---|
| `/api/topology` | GET | k-NN edge classification (preserved / false / torn) |
| `/api/distortion-grid` | GET | 32×32 local distortion ratio grid with actual coordinate bounds |
| `/api/subspace-angles` | GET | Canonical principal angles and Grassmannian distance between two catalog views |

## File integrity

If you suspect a file has been renamed, truncated, or mixed up, check the leading heading of each file:

| Filename | Expected heading |
|---|---|
| `RESEARCH_ROADMAP.md` | Shadowspace Architecture and Data Contract |
| `ARCHITECTURE_AND_DATA_CONTRACT.md` | Shadowspace Implementation Plan |
| `PROJECT_DECISIONS.md` | Shadowspace Testing and Validation |
| `docs/README.md` | Shadowspace Mathematical and Research Knowledge Base |
| `README.md` | Shadowspace Research Roadmap |
| `MATHEMATICAL_AND_RESEARCH_KNOWLEDGE_BASE.md` | Shadowspace Project Decisions |
| `TESTING_AND_VALIDATION.md` | Shadowspace Documentation Index |


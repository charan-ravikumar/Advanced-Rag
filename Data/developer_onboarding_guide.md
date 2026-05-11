---
title: Developer Onboarding Guide — NovaMind RAG Platform
document_id: ENG-ONBOARD-2025-04
version: "1.4"
last_updated: "2025-09-22"
maintained_by: Platform Engineering Team
slack_channel: "#eng-rag-platform"
classification: internal
---

# Developer Onboarding Guide — NovaMind RAG Platform

> **Version:** 1.4 | **Last Updated:** September 22, 2025
> **Maintained by:** Platform Engineering Team (`#eng-rag-platform`)
> **Onboarding buddy:** Contact Marcus Okonkwo for pairing

---

## Table of Contents

1. [Welcome & Overview](#1-welcome--overview)
2. [Environment Setup](#2-environment-setup)
3. [Repository Structure](#3-repository-structure)
4. [Local Development](#4-local-development)
5. [Architecture Deep-Dive](#5-architecture-deep-dive)
6. [Testing Guidelines](#6-testing-guidelines)
7. [Deployment & CI/CD](#7-deployment--cicd)
8. [On-Call & Observability](#8-on-call--observability)
9. [Team Norms & Communication](#9-team-norms--communication)
10. [Your First 30/60/90 Days](#10-your-first-306090-days)

---

## 1. Welcome & Overview

Welcome to NovaMind Platform Engineering! This guide will get you from zero to productive on the RAG Platform in approximately **3 working days**. Follow each section in order — steps build on one another.

### What Is the RAG Platform?

The **Retrieval-Augmented Generation (RAG) Platform** is NovaMind's core enterprise product. It enables customers to build AI-powered knowledge bases and Q&A systems over their own documents. The platform reached General Availability (GA) on **August 14, 2025** and currently handles **27.6 million API calls per month** across all customer namespaces.

### Key People to Know

| Name                       | Role                    | Slack                   | Timezone      |
| -------------------------- | ----------------------- | ----------------------- | ------------- |
| Marcus Okonkwo             | Tech Lead, RAG Platform | `@marcus.okonkwo`       | US-Eastern    |
| Priya Nakamura             | Principal Architect      | `@priya.nakamura`       | US-Pacific    |
| Tyler Johansson            | Infrastructure Lead      | `@tyler.johansson`      | EU-Stockholm  |
| Sadia Ramirez              | MLOps Lead              | `@sadia.ramirez`        | US-Central    |
| Dr. Ananya Krishnamurthy   | VP Engineering           | `@ananya.k`             | US-Eastern    |
| Rahul Gupta                | Security Engineering     | `@rahul.gupta`          | US-Eastern    |
| Bipin Patel                | ML Research Lead         | `@bipin.patel`          | US-Pacific    |
| Lena Osei                  | AI Governance Lead       | `@lena.osei`            | EU-London     |

### Core Services at a Glance

The RAG Platform is composed of **six microservices**. Your first week focuses on `ingestion-service` and `retrieval-service`.

| Service               | Language     | Repo                      | Owner          |
| --------------------- | ------------ | ------------------------- | -------------- |
| `ingestion-service`   | Python 3.11  | `novamind/rag-ingestion`  | M. Okonkwo     |
| `retrieval-service`   | Go 1.22      | `novamind/rag-retrieval`  | P. Nakamura    |
| `embedding-service`   | Python 3.11  | `novamind/rag-embedding`  | S. Ramirez     |
| `generation-gateway`  | Go 1.22      | `novamind/rag-generation` | M. Okonkwo     |
| `metadata-service`    | Python 3.11  | `novamind/rag-metadata`   | T. Johansson   |
| `rag-api-gateway`     | Go 1.22      | `novamind/rag-gateway`    | P. Nakamura    |

---

## 2. Environment Setup

### 2.1 Prerequisites

Install the following tools. **Use the pinned versions** — newer major versions may break compatibility.

```bash
# Verify versions after installing
python --version        # Requires 3.11.x
go version              # Requires 1.22.x
docker --version        # Requires 24.x+
kubectl version         # Requires 1.30.x
helm version            # Requires 3.14.x
terraform --version     # Requires 1.9.x
```

**Recommended IDE:** VS Code with these extensions:
- `ms-python.python` — Python
- `golang.go` — Go
- `ms-kubernetes-tools.vscode-kubernetes-tools` — Kubernetes
- `eamodio.gitlens` — GitLens
- `ms-azuretools.vscode-docker` — Docker

### 2.2 Access Checklist

Submit requests in `#it-access-requests`. Standard onboarding access:

- [ ] **GitHub** — `novamind-engineering` org (ask your manager)
- [ ] **AWS** dev account `012345678901` — submit `access-request-aws-dev.yaml`
- [ ] **Pinecone** dev environment — request invite in `#eng-rag-platform`
- [ ] **Datadog** — auto-provisioned 24h after GitHub access
- [ ] **PagerDuty** — required from Week 2; request via `#it-access-requests`
- [ ] **Confluence** — auto-provisioned with Google SSO
- [ ] **Linear** — ask Marcus to add you to the RAG Platform workspace

> ⚠️ **Security:** Never store API keys in code — even in private branches. Use AWS Secrets Manager. All access is audited. Violations are reviewed by the Security team (Rahul Gupta).

### 2.3 SSH & GPG Setup

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your.name@novamind.ai"
# Add public key to GitHub: Settings → SSH Keys
cat ~/.ssh/id_ed25519.pub

# GPG commit signing (required for all merges to main)
gpg --full-generate-key
git config --global user.signingkey YOUR_GPG_KEY_ID
git config --global commit.gpgsign true
```

---

## 3. Repository Structure

All RAG Platform repos follow a standard layout. Below is `novamind/rag-ingestion` (your primary Week 1 repo):

```
rag-ingestion/
├── src/
│   ├── ingestion/
│   │   ├── parsers/          # Format-specific document parsers
│   │   │   ├── pdf_parser.py
│   │   │   ├── docx_parser.py
│   │   │   ├── html_parser.py
│   │   │   ├── markdown_parser.py
│   │   │   ├── xlsx_parser.py
│   │   │   └── txt_parser.py
│   │   ├── chunking/
│   │   │   ├── recursive_splitter.py
│   │   │   └── semantic_splitter.py  # Experimental — do not use in prod
│   │   ├── pipeline.py       # Main orchestration entrypoint
│   │   └── models.py         # Pydantic v2 data models
│   └── api/
│       ├── routes.py         # FastAPI route definitions
│       └── middleware.py     # Auth, rate limiting, tracing
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/             # Sample files: PDF, DOCX, HTML, MD, XLSX, TXT
├── k8s/
│   ├── deployment.yaml
│   ├── configmap.yaml
│   └── hpa.yaml
├── .github/workflows/        # CI pipelines (lint → test → build → scan)
├── pyproject.toml            # Poetry dependencies
├── Dockerfile
├── Makefile
└── README.md
```

### Branch Strategy

We use **trunk-based development**:

| Branch pattern             | Purpose                                           | Lifespan   |
| -------------------------- | ------------------------------------------------- | ---------- |
| `main`                     | Protected; 2 approvals + CI green required        | Permanent  |
| `dev/<name>/<feature>`     | Feature work                                      | < 5 days   |
| `hotfix/<description>`     | Emergency prod fixes; on-call lead approval req.  | < 24 hours |

---

## 4. Local Development

### 4.1 Running the Stack

```bash
# Clone
git clone git@github.com:novamind/rag-ingestion.git
cd rag-ingestion

# Configure environment
cp .env.example .env
# Edit .env: OPENAI_API_KEY, AWS_PROFILE=dev, PINECONE_API_KEY=<dev-key>

# Install Python deps
poetry install

# Start Docker Compose stack (Kafka, Elasticsearch, mock Pinecone, Postgres)
make dev-up

# Run the service with hot-reload
make serve
# → http://localhost:8080

# Test ingestion end-to-end
make test-ingest FILE=tests/fixtures/sample_policy.pdf
```

### 4.2 Makefile Reference

| Command                    | Description                                         |
| -------------------------- | --------------------------------------------------- |
| `make dev-up`              | Start full Docker Compose stack                     |
| `make dev-down`            | Stop and remove containers                          |
| `make serve`               | Run FastAPI with hot-reload on :8080                |
| `make test`                | Full test suite (unit + integration)                |
| `make test-unit`           | Unit tests only — no Docker required, fast          |
| `make lint`                | `ruff` + `mypy` + `black` check                     |
| `make format`              | Auto-format with `black` and `isort`                |
| `make test-ingest FILE=…`  | Ingest a sample file against local stack            |
| `make logs SERVICE=kafka`  | Tail Docker Compose service logs                    |

### 4.3 Environment Variables

| Variable                  | Description                          | Example                                         |
| ------------------------- | ------------------------------------ | ----------------------------------------------- |
| `APP_ENV`                 | Runtime environment                  | `local` / `staging` / `production`              |
| `PINECONE_API_KEY`        | Pinecone auth                        | `pcsk_...`                                      |
| `PINECONE_ENV`            | Pinecone environment                 | `us-east-1-aws`                                 |
| `EMBEDDING_SERVICE_URL`   | gRPC endpoint for embedding service  | `localhost:50051`                               |
| `KAFKA_BOOTSTRAP_SERVERS` | Kafka brokers                        | `localhost:9092`                                |
| `POSTGRES_DSN`            | PostgreSQL connection string         | `postgresql://rag:secret@localhost:5432/ragdb`  |
| `CHUNK_SIZE`              | Default chunk size in tokens         | `512`                                           |
| `CHUNK_OVERLAP`           | Default chunk overlap in tokens      | `64`                                            |
| `LOG_LEVEL`               | Logging verbosity                    | `DEBUG` / `INFO` / `WARNING`                    |

---

## 5. Architecture Deep-Dive

### 5.1 Ingestion Pipeline Flow

```
Customer Upload
      │
      ▼
[API Gateway] ──HMAC auth──▶ validates namespace + permissions
      │
      ▼
[Ingestion Service]
      ├─▶ Raw file stored to S3 (AES-256)
      ├─▶ Kafka message → topic: ingestion.jobs
      └─▶ Job record → PostgreSQL (status = PENDING)
      │
      ▼  (async Kafka consumer)
[Document Parser]  ──format detection──▶
      ├─ PDF     → pdfplumber → OCR fallback if < 50 chars/page (Tesseract)
      ├─ DOCX    → python-docx → pandoc normalization
      ├─ HTML    → trafilatura main-content extraction → BeautifulSoup cleanup
      ├─ MD      → markdown-it-py → AST → text with heading metadata
      ├─ XLSX    → openpyxl → per-sheet extraction → each sheet = document
      └─ TXT     → charset auto-detection → UTF-8 normalize
      │
      ▼
[Chunk Manager]  (recursive character splitter)
      ├─ chunk_size=512 tokens (cl100k_base)
      ├─ overlap=64 tokens
      ├─ split priority: paragraphs → sentences → words → chars
      ├─ heading-aware: force boundary at each heading
      └─ table preservation: rows never split mid-table
      │
      ▼
[Embedding Service]  ──gRPC──▶ NM-Embed-768 on Triton (A10G GPU)
      └─ Batch: 512 chunks | p95 latency: 18ms | 768-dim L2-normalized
      │
      ▼
[Vector Store Adapter]
      ├─▶ Upsert to Pinecone (dense HNSW index)
      └─▶ Index to Elasticsearch (BM25 sparse index)
      │
      ▼
[Metadata Service]  ──▶ PostgreSQL
      └─ doc_id, chunk_idx, source_page, section_heading, ingestion_ts
      │
      ▼
Job status → COMPLETED  (Postgres update + Kafka event on ingestion.completed)
```

### 5.2 Retrieval Pipeline

Queries enter the **Retrieval Service** and undergo a 5-step process:

1. **Query pre-processing** — lowercase, stopword filtering, intent classification
2. **Parallel retrieval** — dense (Pinecone top-100) + sparse BM25 (Elasticsearch top-100)
3. **RRF fusion** — Reciprocal Rank Fusion with `k=60`; duplicate chunks deduplicated by `chunk_id`
4. **Cross-encoder re-ranking** — top-20 fused chunks re-scored by `ms-marco-MiniLM-L-6-v2`
5. **Context assembly** — top-k chunks (default 5) formatted into LLM prompt with source citations

### 5.3 Key Pydantic Models

```python
# src/ingestion/models.py

class IngestionJob(BaseModel):
    job_id: str                # UUID v4
    namespace_id: str
    document_id: str           # SHA-256 of raw file (content hash — idempotency key)
    filename: str
    file_format: Literal["pdf", "docx", "html", "md", "xlsx", "txt"]
    status: Literal["PENDING", "PARSING", "CHUNKING", "EMBEDDING", "INDEXING",
                    "COMPLETED", "FAILED"]
    created_at: datetime
    completed_at: Optional[datetime]
    chunk_count: Optional[int]
    error_message: Optional[str]

class Chunk(BaseModel):
    chunk_id: str              # f"{document_id}_{chunk_index}"
    document_id: str
    namespace_id: str
    text: str
    token_count: int
    chunk_index: int
    source_page: Optional[int]
    section_heading: Optional[str]
    embedding: Optional[list[float]]    # 768-dim; populated after embedding step
    metadata: dict[str, Any]            # Customer-supplied metadata pass-through
    ingestion_timestamp: datetime
```

---

## 6. Testing Guidelines

### 6.1 Test Pyramid

| Type          | Location            | Command           | Coverage Target |
| ------------- | ------------------- | ----------------- | --------------- |
| Unit          | `tests/unit/`       | `make test-unit`  | > 90%           |
| Integration   | `tests/integration/`| `make test`       | > 75%           |
| End-to-end    | `tests/e2e/`        | `make test-e2e`   | Critical paths  |
| Load (k6)     | `tests/load/`       | `make test-load`  | Pre-release only|

### 6.2 Writing Tests

Use `pytest` + `pytest-asyncio`. Follow the conventions below:

```python
# tests/unit/parsers/test_pdf_parser.py
import pytest
from pathlib import Path
from ingestion.parsers.pdf_parser import PDFParser

FIXTURES = Path("tests/fixtures")

class TestPDFParser:
    def test_extracts_text_from_normal_pdf(self):
        result = PDFParser().parse(FIXTURES / "sample_quarterly_report.pdf")
        assert len(result.text) > 100
        assert result.page_count == 3

    def test_falls_back_to_ocr_for_scanned_pdf(self):
        result = PDFParser().parse(FIXTURES / "scanned_invoice.pdf")
        assert result.ocr_used is True
        assert result.text  # non-empty

    @pytest.mark.parametrize("filename,expected_chunks", [
        ("short_memo.pdf", 2),
        ("long_report.pdf", 47),
        ("tables_heavy.pdf", 8),
    ])
    def test_chunk_count(self, filename, expected_chunks):
        result = PDFParser().parse(FIXTURES / filename)
        assert result.chunk_count == expected_chunks
```

### 6.3 Test Fixtures Inventory

Pre-built fixtures live in `tests/fixtures/`. Do **not** replace these without team discussion — they are canonical regression anchors.

| Fixture                         | Format  | Content                                    | Tests                            |
| ------------------------------- | ------- | ------------------------------------------ | -------------------------------- |
| `sample_quarterly_report.pdf`   | PDF     | Q3 AI project report (3 pages)             | Happy-path PDF ingestion         |
| `engineering_docs.docx`         | DOCX    | Technical platform docs (5 pages)          | DOCX parser, heading detection   |
| `knowledge_base.html`           | HTML    | Company FAQ page with nav chrome           | HTML extraction, nav stripping   |
| `developer_onboarding.md`       | MD      | This guide with front matter               | Markdown parser, metadata        |
| `company_data.xlsx`             | XLSX    | 4-sheet workbook (Employees, Revenue, etc.)| Multi-sheet XLSX ingestion       |
| `support_logs.txt`              | TXT     | Operational logs with mixed encoding       | TXT charset edge cases           |
| `scanned_invoice.pdf`           | PDF     | Image-only PDF (300 DPI scan)              | OCR fallback path                |
| `merged_cells.xlsx`             | XLSX    | Workbook with merged cells                 | Known limitation regression      |

---

## 7. Deployment & CI/CD

### 7.1 Pipeline Overview

```
Feature branch push
      │
      ▼
[GitHub Actions CI]
  ├─ ruff lint + mypy type check + black format check
  ├─ pytest unit tests (3 min)
  ├─ pytest integration tests with Docker Compose (12 min)
  ├─ docker build + trivy vulnerability scan
  └─ ECR push (tagged with git SHA)
      │
      ▼  (merge to main, after 2 approvals + all checks green)
[ArgoCD → Staging]  ←── GitOps: watches novamind/rag-k8s-config
  └─ Auto-deploy to staging EKS cluster (us-east-1)
      │
      ▼  (QA sign-off in Linear; manual promotion via ArgoCD UI)
[ArgoCD → Production]
  └─ Canary: 5% → 25% → 100% traffic over 30 minutes
     └─ Auto-rollback if error rate > 1% or p99 > 1500ms
```

### 7.2 Environment Urls

| Environment | API Base URL                          | Grafana                                  |
| ----------- | ------------------------------------- | ---------------------------------------- |
| Local       | `http://localhost:8080/v1/rag`        | N/A                                      |
| Staging     | `https://staging-api.novamind.ai/v1/rag` | `https://grafana.staging.novamind.ai` |
| Production  | `https://api.novamind.ai/v1/rag`     | `https://grafana.novamind.ai`            |

---

## 8. On-Call & Observability

### 8.1 Monitoring Stack

- **Metrics:** Prometheus → Grafana (dashboards in `novamind/rag-dashboards`)
- **Tracing:** OpenTelemetry → Tempo (full distributed traces for every request)
- **Logging:** Structured JSON → Loki → Grafana Explore
- **Alerting:** PagerDuty (Sev-1/2) | Slack `#eng-rag-alerts` (Sev-3/4)

### 8.2 Key Dashboards

| Dashboard                    | URL Slug                    | What to Watch                      |
| ---------------------------- | --------------------------- | ---------------------------------- |
| RAG Platform Overview        | `d/rag-overview`            | Request rate, error rate, latency  |
| Ingestion Pipeline           | `d/rag-ingestion`           | Queue depth, job throughput        |
| Retrieval Quality            | `d/rag-retrieval-quality`   | MRR@10, latency p50/p99            |
| Infrastructure Costs         | `d/rag-infra-costs`         | GPU utilization, Pinecone vectors  |

### 8.3 Severity Definitions

| Severity | Definition                                            | Response Time  | Escalation        |
| -------- | ----------------------------------------------------- | -------------- | ----------------- |
| Sev-1    | Full production outage / data loss risk               | 5 minutes      | Page on-call lead |
| Sev-2    | Partial outage; > 5% error rate in production         | 15 minutes     | Page on-call      |
| Sev-3    | Degraded performance; SLA at risk                     | 1 hour         | Slack alert       |
| Sev-4    | Minor issue; no customer impact                       | Next business day | Linear ticket  |

### 8.4 Common Runbook Links

- [Ingestion job stuck in PENDING](https://confluence.novamind.ai/runbooks/rag-ingestion-pending)
- [High retrieval latency > 1500ms p99](https://confluence.novamind.ai/runbooks/rag-high-latency)
- [Pinecone degradation — activate Weaviate failover](https://confluence.novamind.ai/runbooks/rag-weaviate-failover)
- [Kafka consumer lag > 50K messages](https://confluence.novamind.ai/runbooks/rag-kafka-lag)

---

## 9. Team Norms & Communication

### 9.1 Meetings

| Meeting                       | Cadence         | Time (ET)         | Notes                            |
| ----------------------------- | --------------- | ----------------- | -------------------------------- |
| RAG Platform Standup          | Daily Mon–Fri   | 9:30 AM           | Async-first; 15 min max          |
| Sprint Planning               | Bi-weekly Mon   | 10:00 AM          | 2-week sprints in Linear         |
| Architecture Review           | Weekly Thu      | 3:00 PM           | Open to all; design proposals    |
| Incident Retrospectives       | After each Sev-1/2 | Within 48h     | Blameless; use `retro-template`  |
| Quarterly Planning (OKRs)     | Quarterly       | TBD by schedule   | All hands + breakouts            |

### 9.2 Code Review Standards

- **All PRs need 2 approvals** before merge (1 from the service owner, 1 from any senior engineer)
- **Response SLA:** Reviewers respond within **1 business day**
- **PR size:** Keep PRs under 400 lines changed. Break larger changes into stacked PRs.
- **Description:** Every PR must include: summary, testing steps, and a link to the Linear ticket
- **Breaking changes:** Require a migration guide comment and a `migration/` note in the PR

### 9.3 Slack Etiquette

- `#eng-rag-platform` — Technical discussion, design questions, incident coordination
- `#eng-rag-alerts` — Automated only; do not post manually
- `#general-engineering` — Cross-team announcements
- Use **threads** for all replies; keep channels scannable
- Tag `@here` only for Sev-2+ incidents in `#eng-rag-platform`

---

## 10. Your First 30/60/90 Days

### Days 1–5: Orientation

- [ ] Complete environment setup (Section 2)
- [ ] Run the local stack and ingest a sample PDF end-to-end
- [ ] Read `ENG-RAG-2025-017` (Internal Engineering Docs — ask Marcus for link)
- [ ] Shadow on-call with Marcus for one week
- [ ] Attend Architecture Review and Sprint Planning

### Days 6–30: First Contributions

- [ ] Pick up a `good-first-issue` ticket from Linear (tag: `good-first-issue`)
- [ ] Submit your first PR and go through code review
- [ ] Write at least 5 unit tests for the parser of your choice
- [ ] Pair with Tyler Johansson on a Kubernetes cluster walkthrough
- [ ] Complete NovaMind Security Training (mandatory; link in Confluence)

### Days 31–60: Growing Independence

- [ ] Own at least one feature end-to-end (design → implementation → deploy)
- [ ] Join the on-call rotation (Week 5 at the earliest)
- [ ] Present a 15-minute technical deep-dive at Architecture Review
- [ ] Contribute to the `tests/fixtures/` library with at least one new fixture
- [ ] Identify and document one area of technical debt in the repo

### Days 61–90: Full Productivity

- [ ] Lead sprint planning for one sprint
- [ ] Review PRs from newer team members
- [ ] Propose and drive at least one improvement to the ingestion pipeline
- [ ] Contribute to the Q4 roadmap planning session
- [ ] Completed 360 feedback with Ananya (VP Engineering)

---

## Appendix A: Glossary

| Term        | Definition                                                                                      |
| ----------- | ----------------------------------------------------------------------------------------------- |
| **RAG**     | Retrieval-Augmented Generation — a technique combining document retrieval with LLM generation   |
| **Chunk**   | A segment of a document, typically 512 tokens, that is the unit of retrieval                   |
| **HNSW**    | Hierarchical Navigable Small World — the approximate nearest-neighbor index algorithm used by Pinecone |
| **RRF**     | Reciprocal Rank Fusion — algorithm for combining ranked lists from different retrieval methods  |
| **MRR@10**  | Mean Reciprocal Rank at 10 — primary retrieval quality metric (target: ≥ 0.80)                 |
| **Namespace** | A logical tenant boundary in the RAG Platform; each customer has one or more namespaces      |
| **Embedding** | A fixed-length floating-point vector representing the semantic meaning of a text chunk       |
| **BM25**    | Best Match 25 — the sparse lexical retrieval algorithm used for keyword-based search           |
| **NM-Embed-768** | NovaMind's in-house embedding model; 768-dim, fine-tuned from BGE-large-en-v1.5          |
| **Triton**  | NVIDIA Triton Inference Server — used to serve the embedding model on A10G GPUs               |

---

## Appendix B: Useful Commands

```bash
# Watch Kubernetes pod status in real time
kubectl get pods -n rag -w

# Stream logs from the ingestion service
kubectl logs -n rag -l app=ingestion-service -f --since=1h

# Port-forward the retrieval service locally for debugging
kubectl port-forward -n rag svc/retrieval-service 8081:8080

# Check Kafka consumer group lag
kubectl exec -n rag -it kafka-0 -- \
  kafka-consumer-groups.sh --bootstrap-server localhost:9092 \
  --describe --group ingestion-consumer

# Force-scale ingestion workers during high queue depth
kubectl scale -n rag deployment/ingestion-service --replicas=8

# Check Pinecone index stats via API
curl -X GET https://api.novamind.ai/v1/rag/namespaces/my-namespace/stats \
  -H "Authorization: Bearer $NOVAMIND_API_KEY"
```

---

*Last reviewed by Marcus Okonkwo — September 22, 2025*
*Next scheduled review: December 2025*
*Feedback? Open a PR against `novamind/eng-docs` or post in `#eng-rag-platform`.*

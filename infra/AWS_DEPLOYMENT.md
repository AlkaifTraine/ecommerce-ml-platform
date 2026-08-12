# AWS deployment design

> **Status: designed and costed, NOT deployed.** Everything below is the
> intended cloud topology with real prices attached. Nothing in this document
> has been provisioned — no AWS resource has been created for this project.
> Claims of "deployed on AWS" would be untrue, so the README does not make one.

Target account for a future deploy: `434441229445` (`alkaif-admin`), us-east-1.

---

## Local → cloud mapping

| Local | AWS | Why |
|---|---|---|
| Parquet archive on D: | **S3** (Standard-IA after 30d) | 1.9 GB compressed; the immutable source of truth |
| `ecomml-postgres` | **RDS Postgres** `db.t4g.micro` | OLTP hot window; Graviton is ~20% cheaper than t3 |
| `ecomml-redis` (Streams) | **Kinesis Data Streams** (on-demand) | Same append-log semantics; the sink interface already abstracts this |
| DuckDB warehouse | **Athena** over S3 + Iceberg | Serverless; a 2.3 GB warehouse does not justify Redshift |
| dbt-duckdb | **dbt-athena** | Same models, different adapter |
| `ecomml-airflow` | **ECS Fargate Spot** (1 task) | MWAA is ~$350/mo minimum — indefensible at this scale |
| MLflow (SQLite) | MLflow on Fargate + RDS backend | Registry needs a real DB for versions/aliases |
| FastAPI on 8500 | **ECS Fargate** behind an ALB | p99 is 0.9 ms, so one small task carries it |
| Model artifacts | S3 | Versioned alongside the MLflow registry |

## Cost estimate

Priced for a portfolio-scale deployment running continuously, us-east-1,
on-demand unless noted.

| Service | Configuration | Monthly |
|---|---|---|
| S3 | 5 GB + requests | $0.15 |
| Athena | ~20 GB scanned/mo @ $5/TB | $0.10 |
| RDS Postgres | db.t4g.micro, 20 GB gp3, single-AZ | $13.50 |
| ECS Fargate — API | 0.25 vCPU / 0.5 GB, always on | $8.90 |
| ECS Fargate Spot — Airflow | 0.5 vCPU / 1 GB, ~70% Spot discount | $5.30 |
| ECS Fargate Spot — MLflow | 0.25 vCPU / 0.5 GB | $2.70 |
| ALB | 1 ALB, minimal LCU | $16.20 |
| Kinesis | on-demand, 1 shard-equivalent | $11.00 |
| ECR | 3 images, ~4 GB | $0.40 |
| CloudWatch | logs + basic metrics | $3.00 |
| **Total** | | **≈ $61 / month** |

### Getting it under $25

The ALB and Kinesis are over half the bill and neither is load-bearing at this
scale:

- **Drop the ALB** (−$16.20): a Fargate task with a public IP and API-key auth
  is adequate for a demo. An ALB earns its cost at multiple targets or TLS
  termination at volume, neither of which applies.
- **Replace Kinesis with SQS** (−$10.50): ordering per session is what actually
  matters, and an SQS FIFO queue provides it. Kinesis is the right answer for
  replay and multiple independent consumers; there is one consumer here.
- **Schedule Airflow off outside demo hours** (−$3.50): EventBridge scaling the
  service to zero.

That lands near **$24/month**, which is the figure worth putting on a CV —
alongside the reasoning, because the reasoning is the part that demonstrates
judgement.

### What would change at real scale

These choices are right for portfolio scale and wrong at production scale, and
it is worth being explicit about where they break:

- **Athena → Redshift Serverless / Snowflake** once scans exceed ~1 TB/month;
  per-query pricing stops being cheaper than provisioned compute.
- **Fargate Airflow → MWAA** once more than one person depends on it. The
  single-container LocalExecutor here has no HA and loses queued task state
  when the task is replaced.
- **SQS → Kinesis/MSK** as soon as a second consumer needs the same stream, or
  replay from an offset is required.
- **RDS single-AZ → Multi-AZ** the moment anything real depends on it; single-AZ
  is a deliberate cost choice, not an oversight.

## Deployment order

1. S3 buckets + IAM roles (least privilege per service, no shared role)
2. ECR repositories; push `ecomml-airflow`, `ecomml-api`, `ecomml-mlflow`
3. RDS Postgres; run `docker/postgres/init.sql`
4. Glue catalog + Iceberg tables over the S3 archive; point dbt-athena at them
5. Fargate services: MLflow, then Airflow, then the API
6. EventBridge schedules for the DAGs

## Why this is not deployed

Two honest reasons:

1. **It costs money.** Standing this up is a real monthly bill on a personal
   account, and that is the account owner's decision, not something to be
   assumed.
2. **Nothing here would be proven by deploying it.** The engineering claims —
   AUC 0.8406, the leakage audit, the drift detection, the parity guarantee,
   sub-millisecond serving — are all demonstrated locally against the full
   109,950,743-event dataset. Running the same containers on Fargate would
   demonstrate that Fargate runs containers.

What a deploy *would* add is a live URL, which has real value for a portfolio.
That is a reasonable thing to want; it is just not the same as a technical
claim, and this document keeps the two separate.

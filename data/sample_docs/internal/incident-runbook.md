# Incident Response Runbook — Production (Internal)

**CONFIDENTIAL**

This runbook is for the on-call engineer responding to a production
incident. Work through the steps in order. Escalate if uncertain.

## Step 1: Acknowledge and classify

1. Acknowledge the PagerDuty page within 5 minutes.
2. Open the incident channel: `#inc-YYYY-MM-DD-short-slug` in Slack.
3. Classify severity using the SEV matrix below.

### Severity matrix

- **SEV-1**: Full outage or data loss. All hands; notify exec-on-call.
- **SEV-2**: Major feature degraded for many customers. Notify director.
- **SEV-3**: Minor degradation, workaround exists. Handle in-hours.
- **SEV-4**: Cosmetic. File a ticket, close the page.

## Step 2: Initial triage

Check these dashboards in order:

1. API gateway error rate — Grafana → "Prod / Edge"
2. Control-plane Postgres health — Grafana → "Prod / DB"
3. Data-plane Kubernetes pod status — Grafana → "Prod / DP"

Common failure patterns:

- Gateway 5xx spike + DB CPU saturated → run `DB-CPU-RECOVERY` playbook
- Pod pending in data plane → check node-pool autoscaling limits
- Sudden 401s → suspect IdP outage; check SSO status page

## Step 3: Mitigation playbooks

### Playbook: DB-CPU-RECOVERY

1. Check top queries in `pg_stat_statements`.
2. If a runaway query is identified, terminate with
   `SELECT pg_terminate_backend(pid)`.
3. If index bloat is suspected, run `REINDEX CONCURRENTLY` off-peak — not
   during the incident.

### Playbook: REGIONAL-FAILOVER

Only executed on SEV-1 with director approval.

1. Validate the backup region is healthy via the region-health dashboard.
2. Update Route53 weighted record to send 100% to the backup region.
3. Announce in `#incidents` with the failover timestamp.
4. Open a follow-up ticket to fail back after the primary region recovers.

## Step 4: Communication

- SEV-1 and SEV-2: post updates every 30 minutes in the incident channel.
- External status page: update within 15 minutes for SEV-1/2.
- Customer support: share draft messaging for inbound tickets.

## Step 5: Post-incident

Within 5 business days the incident owner must file a post-mortem using
the [post-mortem template](./post-mortem-template.md). The post-mortem is
reviewed in the weekly reliability council.

Emergency contacts:
- sre-oncall@acme.internal
- +1-555-0100 (PagerDuty override line)

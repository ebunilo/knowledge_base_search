# Acme Platform — Architecture Overview

The Acme Platform is a multi-region system for scheduling and running
data jobs. This document gives a high-level overview of the main
components and the data flow between them.

## Components

### API Gateway

The entry point for all client traffic. Terminates TLS, authenticates
requests with bearer tokens, and forwards to the appropriate service.
Rate limits are enforced at this layer.

### Control Plane

Keeps the authoritative state of workspaces, jobs, and users. Backed by a
highly available Postgres cluster. All writes go through the control
plane; reads can be served from a Redis cache.

### Data Plane

Where jobs actually run. Each region has its own data plane consisting
of a Kubernetes cluster, object storage, and a streaming backbone. Jobs
are isolated by namespace and can only talk to resources in their own
workspace.

### Observability Stack

Metrics are collected by Prometheus, logs by Loki, and traces by
Tempo. All three are queried through a single Grafana instance. Alerts
are routed to PagerDuty through Alertmanager.

## Data flow: submitting a job

1. Client calls `POST /jobs` on the API gateway.
2. Gateway verifies the token and forwards to the control plane.
3. Control plane validates the request, writes the job record, and
   publishes a `job.created` event.
4. The regional data plane consumes the event and schedules the job on
   Kubernetes.
5. Job output is written to object storage; the control plane is
   notified when the job finishes.

## Failure modes

| Failure                  | Impact                                 | Mitigation                    |
|--------------------------|----------------------------------------|-------------------------------|
| Control-plane Postgres   | All writes fail                        | Automatic failover to replica |
| Regional data plane      | Jobs in that region stop running       | Failover to backup region     |
| API gateway              | New requests rejected                  | Multi-AZ load balancer        |

## Non-goals

- Acme is not a general-purpose compute platform. Workloads must fit the
  job abstraction.
- Acme does not provide managed databases. Use an external service for
  stateful workloads.

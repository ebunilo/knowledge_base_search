# Acme Platform — Getting Started

Welcome to the Acme Platform. This guide walks you through setting up your
first workspace and running a sample job. It is written for public audiences
and can be shared externally.

## Prerequisites

Before you begin, make sure you have:

- An Acme account (free tier is fine)
- Python 3.10 or newer
- A terminal with `curl` installed

## Create your workspace

1. Sign in to https://app.acme.com and click **New workspace**.
2. Choose a region close to your users — `eu-west-1` or `us-east-1`.
3. Name the workspace and click **Create**.

Your workspace is ready in about 30 seconds. You will receive an API key by
email; store it in `~/.acme/credentials` for the CLI to pick up.

## Run your first job

```bash
acme job submit --name hello --image acme/hello:latest
```

The job runs to completion in under a minute. You can watch it from the
dashboard under **Jobs → Recent**.

## Where to next

- Read the [architecture overview](./architecture.md) for a system-level view.
- Browse the [API reference](https://docs.acme.com/api).
- Join our community on Slack for questions.

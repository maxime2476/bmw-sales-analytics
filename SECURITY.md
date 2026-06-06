# Security Policy

## Supported versions

This is an actively-maintained portfolio project; the `main` branch is the
supported version.

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

- Preferred: open a **private vulnerability report** via GitHub
  (*Security → Report a vulnerability*).
- Or email **maxime.gourguechon76@gmail.com** with details and reproduction steps.

You can expect an acknowledgement within a few days. Thank you for helping keep
the project and its users safe.

## Automated security posture

This repository runs continuous supply-chain checks:

- **`pip-audit`** — dependency vulnerability scan in CI.
- **Trivy** — container image vulnerability scan in CI.
- **Dependabot** — automated updates for pip, GitHub Actions and Docker.

The application runs **offline by default** (no secrets required) and the
container runs as an unprivileged user.

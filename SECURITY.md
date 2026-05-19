# Security policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.7.x   | Yes       |
| < 0.7   | Best effort |

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security-sensitive reports.

Email or message the maintainer via the contact on the [GitHub profile](https://github.com/vishalstark512) with:

- Description of the issue
- Steps to reproduce
- Impact assessment (if known)

We aim to acknowledge within 7 days and provide a fix or mitigation timeline when confirmed.

## Scope

This library runs user-supplied models and data locally. Typical risks: untrusted pickle artifacts, arbitrary code in user training scripts (out of scope), and dependency vulnerabilities in PyTorch / Transformers stacks—keep dependencies updated.

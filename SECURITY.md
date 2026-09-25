# Security

## Reporting a vulnerability

Please report vulnerabilities privately through
[GitHub security advisories](https://github.com/inboxpraveen/ThinkLess/security/advisories/new),
not in a public issue. Include what you found, how to reproduce it, and the
impact you expect. You will get a response within a week, and a fix or a plan
for one as soon as the issue is understood.

## Supported versions

Security fixes go into the latest release.

## What counts

ThinkLess sits between user input and actions your agent takes, so the
following are in scope:

- A way for input to change which provider answers a question, to inject an
  answer, or to bypass a question's allowed answers.
- Trace content leaking when `capture_content` is off, or the OpenTelemetry
  sink exporting content it should drop.
- Credentials appearing in traces, logs or error messages.
- Unsafe deserialization or file handling in trace, pricing or dataset
  loading.

Model quality issues, such as a decision model answering a question wrongly,
are not vulnerabilities; please open a regular issue with the example. Keep
consequential actions behind deterministic checks, as described in the
[production guide](docs/guides/production.md).

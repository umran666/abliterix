# Security Policy

## Supported versions

Security fixes are applied to the current `master` branch and, where practical, the latest published PyPI release. Older releases may not receive backported fixes.

## Reporting a vulnerability

Please do not open a public issue for a vulnerability that could put users, credentials, systems, or data at immediate risk. Report it privately to:

**wangzhangwu1216@gmail.com**

Include enough information to reproduce and assess the issue:

- affected version, commit, platform, and configuration;
- a concise description of the impact and attack prerequisites;
- reproducible steps or a minimal proof of concept;
- relevant logs or stack traces with secrets and personal data removed; and
- any suggested mitigation, if known.

You should receive an initial acknowledgment within seven days. Investigation and remediation time will depend on severity, reproducibility, and dependency or upstream involvement. Please allow a reasonable opportunity to investigate before public disclosure.

## Scope

This policy covers vulnerabilities in the Abliterix software itself, including its CLI, Web UI, configuration handling, dependency integrations, credential handling, model loading and export paths, and Hugging Face upload workflow.

The following are normally outside this software-vulnerability process:

- harmful, inaccurate, biased, or otherwise undesirable model output without an accompanying software vulnerability;
- vulnerabilities that exist solely in a third-party model, library, API, service, driver, or hardware platform; and
- reports based only on automated scanner output without a reproducible security impact.

Where appropriate, issues in third-party components or upstream models should also be reported to their respective maintainers. Safety or abuse concerns involving Abliterix can still be sent privately to the address above when public disclosure could increase risk.

## Research expectations

When investigating, act in good faith: avoid accessing data that is not yours, disrupting services, degrading availability, violating privacy, or using a finding beyond what is necessary to demonstrate the issue. Do not include live credentials, personal information, or harmful payloads that are unnecessary for reproduction.

This policy does not modify the [AGPL-3.0-or-later license](LICENSE), create a warranty, or constitute a promise of compensation or legal safe harbor.

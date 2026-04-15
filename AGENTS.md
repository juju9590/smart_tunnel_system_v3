# AGENTS.md

## Project overview
This repository is for an AI-based smart tunnel system.

Main goals:
- vehicle detection
- vehicle tracking
- traffic state analysis
- accident and anomaly detection
- dashboard and web integration

This project may be developed in a collaborative environment.
Preserve shared architecture unless the user explicitly asks to change it.

## Development principles
- Do not break shared frontend or backend structure.
- Prefer minimal, local changes over broad refactors.
- Keep existing naming, file layout, and module boundaries unless restructuring is explicitly requested.
- When editing code, explain what changed, why it changed, and where it changed.
- When generating code, include comments for learning and maintenance.
- For debugging, identify the root cause first and patch minimally.

## Frontend rules
- Do not modify shared routing structure unless explicitly requested.
- Do not break compatibility with existing `host` prop usage.
- Keep the feature-based module structure.
- New frontend feature files should remain under:
  - `src/modules/{feature}/`
- Main UI entry should remain in:
  - `src/modules/{feature}/index.jsx`
- API calls should be separated into:
  - `src/modules/{feature}/api.js`
- Shared UI components such as Sidebar, common layout, and shared routing must not be changed unless explicitly requested.

## Backend rules
- Backend routes must be added under:
  - `backend_flask/modules/{feature}/`
- Prefer extending existing module files rather than creating duplicate route structures.
- Do not change shared app bootstrap or global configuration unless explicitly requested.
- Do not hardcode new environment-specific values when an existing config pattern already exists.

## Tunnel project-specific rules
- Focus on tunnel traffic analysis, not license plate recognition.
- Focus on detection, tracking, and logic-based state judgment.
- Treat accident detection and traffic state logic as post-detection logic, not as simple object detection output.
- Preserve logging outputs whenever possible because logs are used for later tuning and evaluation.
- When changing accident or state logic, keep debug visibility high:
  - preserve frame-level logs
  - preserve IDs
  - preserve speed, state, and risk-related values when available

## Coding style
- Prefer readable Python and React code.
- Add comments for non-obvious logic.
- Keep functions focused and modular.
- Avoid large rewrites unless explicitly requested.
- When fixing bugs, identify the cause first, then patch minimally.

## When uncertain
- Ask before changing shared structure.
- Ask before deleting files.
- Ask before renaming core modules used by teammates.
- If a requested change may break collaboration, propose a safe alternative first.

## Output expectations
For code changes, provide:
1. what changed
2. why it changed
3. where it changed

Provide full updated code only when the user explicitly asks for full code.

For debugging:
- prioritize root cause analysis over broad rewrites

For experiment code:
- keep logs and saved outputs easy to inspect
# Sablier MCP Server — Instructions for Claude

## PyPI Versioning

Version is in `pyproject.toml`. CI auto-publishes to PyPI on push to `main` — but only if the version changed (duplicates are silently skipped).

**When to bump the version:**
Only when changes affect the local/stdio experience (new tools, tool changes, bug fixes, dependency changes). Cloud Run-only changes (auth, remote transport, deployment config) don't need a PyPI bump.

**How to bump:**
- **Patch** (0.2.1 → 0.2.2): bug fixes, small tool tweaks, description changes
- **Minor** (0.2.2 → 0.3.0): new tools added, tools removed, breaking parameter changes
- **Major** (0.x → 1.0.0): first stable release (not yet)

If unsure, use patch. Don't bump on every push — only when the change is user-facing for `uvx sablier-mcp` users.

## Transport Modes

- **stdio** (local): auto-selected when `SABLIER_API_KEY` env var is set. No auth server needed.
- **streamable-http** (Cloud Run): default when no API key. Uses OAuth 2.0, stateless sessions.

## Branches

- `main` = production Cloud Run + PyPI publish
- `dev` = staging Cloud Run (keep in sync with main)

---
name: research-notebook
description: >
  Instructs Claude Code agents to maintain a NOTEBOOK.md file in the project root
  as a running audit trail of code changes. Use this skill whenever the user wants
  to track what the agent is doing across a session — e.g. "keep a log of changes",
  "write down what you do", "I want to review what changed", "maintain a research
  notebook", or any request for change tracking, auditability, or session history.
  Also trigger proactively on long or complex agentic tasks where change tracking
  would clearly benefit the user, even if they don't explicitly ask.
---

# Research Notebook Skill

## Purpose

Maintain a `NOTEBOOK.md` file in the project root as a living audit trail. After
completing each logical task or feature, append an entry documenting what changed,
why, and any assumptions or trade-offs made. This lets the user review the full
history of the session at any time.

---

## Writing to the Notebook (File Locking)

Because multiple agents may run concurrently, always use a lockfile when appending
to `NOTEBOOK.md` to prevent interleaved or overwritten entries.

Use this shell pattern every time you write an entry:

```bash
(
  flock -w 10 200 || { echo "Could not acquire notebook lock, skipping entry"; exit 1; }
  cat >> NOTEBOOK.md << 'EOF'
[your entry here]
EOF
) 200>NOTEBOOK.md.lock
```

- `flock -w 10` waits up to 10 seconds for the lock before giving up
- The lockfile (`NOTEBOOK.md.lock`) is created automatically and is harmless to leave around
- If the lock times out (extremely rare), skip the entry rather than blocking the agent

---

## Setup

At the start of a session, check whether `NOTEBOOK.md` exists in the project root.

- **If it does not exist**, create it with this header:

```markdown
# Research Notebook

_Auto-maintained by Claude Code. One entry per completed task._

---
```

- **If it already exists**, do not overwrite it — append new entries below the existing ones.

---

## When to Write an Entry

Write an entry **after completing each logical task or feature** — not after every
individual file edit or command. Use your judgment: a "task" is a coherent unit of
work the user asked for (e.g. "add authentication", "fix the failing test", "refactor
the data pipeline").

Do **not** write an entry for:
- Exploratory reads (just reading files to understand the codebase)
- Failed attempts that were fully reverted with no net change
- Mid-task intermediate steps

---

## Entry Format

Append entries in this format:

```markdown
## [Short task title] — YYYY-MM-DD HH:MM

**Goal:** One sentence describing what this task was trying to achieve.

**Changes:**
- `path/to/file.py` — what was added, removed, or modified
- `path/to/other.ts` — what was added, removed, or modified
- _(list every file with a meaningful change; omit unchanged files)_

**Why:** Brief explanation of the reasoning — why this approach was chosen, what
problem it solves, or what the user asked for.

**Assumptions & trade-offs:**
- Any assumption made when the user's intent was ambiguous
- Any trade-off chosen (e.g. speed vs. correctness, simplicity vs. flexibility)
- Any known limitation of the solution
- _(Omit this section entirely if there are no notable assumptions or trade-offs)_

---
```

---

## Guidelines

- **Be concise.** Entries are for human review, not exhaustive logs. A few bullet
  points per section is usually enough.
- **Be specific about files.** Always include the file path relative to the project
  root, not just the filename.
- **Write the "Why" in plain language.** Assume the reader is the user coming back
  to this later — not you.
- **Don't duplicate intent.** If the user's request was clear, the "Goal" and "Why"
  may overlap — that's fine. Keep it brief rather than padding.
- **Timestamp format:** Use the local system time if available (`date` command),
  otherwise use UTC. Format: `YYYY-MM-DD HH:MM`.

---

## Example Entry

```markdown
## Add JWT authentication middleware — 2025-11-03 14:32

**Goal:** Protect all `/api/*` routes so only authenticated users can access them.

**Changes:**
- `src/middleware/auth.ts` — new file; implements JWT verification using `jsonwebtoken`
- `src/app.ts` — registered auth middleware before API route handlers
- `src/routes/auth.ts` — added `/login` endpoint that issues signed JWTs
- `package.json` — added `jsonwebtoken` and `@types/jsonwebtoken` dependencies

**Why:** User asked to lock down the API. JWT was chosen over sessions because the
app is stateless and already uses REST conventions. The secret is read from
`process.env.JWT_SECRET` to avoid hardcoding credentials.

**Assumptions & trade-offs:**
- Assumed token expiry of 24h is acceptable; user did not specify
- No refresh token mechanism — kept simple for now, can be added later
- Public routes (`/health`, `/login`) are explicitly excluded from the middleware

---
```

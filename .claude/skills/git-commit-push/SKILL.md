---
name: git-commit-push
description: >
  Run this skill automatically at the end of EVERY task that modifies files in a git repository —
  even if the user didn't explicitly ask for it. Triggers include: finishing a coding task,
  editing config files, creating new files, refactoring, bug fixes, writing scripts, or any
  file-system change inside a git repo. The skill stages changes, proposes .gitignore additions,
  writes a commit message from the task summary, and pushes to the claude-dev branch so the user
  can review everything via a pull request before it touches main.
---

# Git Commit & Push Skill

Run this at the end of every task that touches files in a git repository. The goal is to keep a
clean, reviewable history on a dedicated `claude-dev` branch so the user can restore any previous
state or merge via pull request.

---

## Step-by-step workflow

### 1. Confirm the repo root

```bash
git rev-parse --show-toplevel
```

All subsequent commands run from that directory. If this fails (not a git repo), stop and tell the
user — don't proceed.

---

### 2. Inspect what changed

```bash
git status --short
git diff --stat HEAD 2>/dev/null || true
```

Build a mental list of:
- New untracked files/folders
- Modified tracked files
- Deleted files

---

### 3. Propose .gitignore additions (BEFORE staging)

Look at every **untracked** file/folder from step 2. For each one, decide whether it should be
ignored rather than committed. Common candidates:

| Pattern | Reason |
|---|---|
| `__pycache__/`, `*.pyc`, `*.pyo` | Python bytecode |
| `.env`, `.env.*` | Secrets / environment variables |
| `node_modules/` | JS dependencies (should be installed, not committed) |
| `*.log` | Log files |
| `dist/`, `build/`, `.next/`, `out/` | Build artefacts |
| `.DS_Store`, `Thumbs.db` | OS metadata |
| `*.sqlite`, `*.db` | Local databases |
| `*.key`, `*.pem`, `*.p12` | Private keys / certificates |
| `.venv/`, `venv/`, `env/` | Python virtual environments |
| `coverage/`, `.coverage` | Test coverage reports |
| `*.egg-info/` | Python packaging artefacts |
| `tmp/`, `temp/` | Temporary directories |

**Propose additions** to the user before touching anything:

> "Before staging, I noticed the following files that might belong in `.gitignore`:
> - `__pycache__/` — Python bytecode
> - `.env` — may contain secrets
>
> Shall I add these? (yes / no / choose)"

Wait for the user's confirmation. Then:
- If yes (or partially yes): append the confirmed patterns to `.gitignore` and stage `.gitignore`
  itself.
- If no: proceed without modifying `.gitignore`.

If nothing looks like it should be ignored, skip this step silently.

---

### 4. Ensure the `claude-dev` branch exists and is active

```bash
# Check if we're already on claude-dev
CURRENT=$(git branch --show-current)

if [ "$CURRENT" != "claude-dev" ]; then
  # Does the branch exist locally?
  if git show-ref --verify --quiet refs/heads/claude-dev; then
    git checkout claude-dev
  else
    # Branch doesn't exist — create it from current HEAD
    git checkout -b claude-dev
    echo "Created new branch: claude-dev"
  fi
fi
```

> **Note**: if the user is on `main` or another protected branch and has uncommitted changes,
> stash first, switch, then pop:
> ```bash
> git stash
> git checkout -b claude-dev   # or checkout if exists
> git stash pop
> ```

---

### 5. Stage all changes

```bash
git add -A
```

Then confirm what will be committed:

```bash
git status --short
```

Show this to the user so they can see exactly what's going in.

---

### 6. Write and commit

Generate a commit message from the task just completed. Format:

```
<short imperative summary, ≤72 chars>

- <bullet: what was done>
- <bullet: what was done>
- <bullet: files affected or created>
```

Example:
```
Add user authentication with JWT

- Implement login/logout endpoints in auth.py
- Add JWT token generation and validation middleware
- Create users table migration (migrations/002_users.sql)
- Update .gitignore to exclude .env
```

Commit:
```bash
git commit -m "<message>"
```

---

### 7. Push to remote (with rebase fallback)

```bash
git push origin claude-dev
```

If the push is rejected because the remote has diverged:

```bash
git pull --rebase origin claude-dev
git push origin claude-dev
```

If rebase produces conflicts, stop and report them clearly:
> "There are merge conflicts in `<file>`. Please resolve them and I'll complete the push."

---

### 8. Report to the user

After a successful push, summarise:

```
✅ Changes committed and pushed to `claude-dev`

Branch:  claude-dev
Commit:  <hash> — <first line of message>
Files:   <N> changed, <X> insertions(+), <Y> deletions(-)

Next step: open a pull request from claude-dev → main when you're ready to merge.
To restore this state later: git checkout claude-dev && git reset --hard <hash>
```

---

## Edge cases

| Situation | Action |
|---|---|
| Nothing to commit (`git status` clean) | Say "Nothing to commit — working tree clean." and stop |
| Not inside a git repo | Say "This directory isn't a git repository. Run `git init` first." and stop |
| `.gitignore` doesn't exist yet | Create it before appending |
| Push fails after rebase | Report the error verbatim; don't force-push |
| User is mid-rebase or merge | Warn and stop: "Git is in the middle of a rebase/merge. Resolve that first." |

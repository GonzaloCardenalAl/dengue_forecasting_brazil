---
name: git-undo-last-commit
description: Hard-resets the claude-dev branch to the commit before the last one, discarding the most recent auto-pushed commit and all its file changes. Use this skill whenever the user says they don't like the last changes, wants to undo the last commit, wants to restore to before the last push, or asks to roll back / revert / undo the agent's last changes. Trigger even if the user phrases it casually like "undo that", "go back", "revert what you just did", or "I don't like those changes".
---

# Git Undo Last Commit

Hard-resets the `claude-dev` branch to `HEAD~1`, permanently discarding the most recent commit and all associated file changes.

## When to use

- User dislikes the agent's last set of changes
- User wants to roll back to the state before the last auto-push
- User says things like "undo", "revert", "restore", "go back", "I don't like that"

---

## Steps

### 1. Confirm the last commit with the user

Before doing anything destructive, show the user what will be deleted:

```bash
cd <repo-root> && git log --oneline -3
```

Display the top 3 commits and clearly indicate which one will be removed (the most recent). Say something like:

> "This will permanently delete the following commit and all its file changes:"
> `abc1234 — <commit message>`
> "Are you sure?"

Wait for explicit confirmation before proceeding.

### 2. Hard reset to HEAD~1

```bash
cd <repo-root> && git reset --hard HEAD~1
```

This discards the last commit and all associated file changes from the working tree.

### 3. Force-push to claude-dev

Since history has been rewritten, a force-push is required:

```bash
cd <repo-root> && git push origin claude-dev --force
```

### 4. Confirm success

Show the user the new HEAD so they can confirm the repo is in the right state:

```bash
cd <repo-root> && git log --oneline -3
```

Tell the user the rollback is complete and what the repo now looks like.

---

## Safety notes

- **Always show the commit that will be deleted and ask for confirmation before running the reset.**
- This is a destructive, irreversible operation. Do not skip the confirmation step.
- If the user is on a branch other than `claude-dev`, flag it and ask before proceeding.
- If `HEAD~1` doesn't exist (i.e. there's only one commit), tell the user there's nothing to roll back to.

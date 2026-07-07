---
name: handumi-commit
description: Create concise, conventional-commit-formatted git commits, splitting unrelated work into separate commits. Use when the user asks to commit changes in this repo, or explicitly invokes /handumi-commit.
---

# Git Commit Skill

Create well-formatted git commits following conventional commit standards. Keep
each commit message short — a few lines, not a report.

## Usage

```text
/handumi-commit
```

## Behavior

1. Inspect `git status --short` and `git diff` (staged and unstaged).
2. Split the changes into one or more coherent units of work (see below).
3. For each unit, stage only the files that belong to it and commit separately.
4. Write a conventional commit message: short subject, optional 1-3 line body.

## One commit = one unit of work

Before committing, ask: "do these changes belong together, or are they two
different things I happened to do in the same session?"

- If the changes serve one purpose (a bug fix, a feature, a refactor, its
  tests), commit them together.
- If the changes serve two+ unrelated purposes (e.g. a bug fix plus an
  unrelated doc update, or two independent features), **split them into
  separate commits**, even if that means more than one commit per request.
  Mixing unrelated work in one commit makes history harder to read, revert,
  or bisect later.
- Don't fold in incidental drive-by edits (whitespace, unrelated renames,
  typos) unless they're part of the same change or the user asks for them
  explicitly.
- Never commit files unrelated to the requested change just because they
  happen to be modified in the working tree.

## Commit message format

```text
<type>(<scope>): <short description>

<1-3 line body: what changed and why, only if the subject line isn't enough>
```

Keep it tight — most commits need nothing beyond a good subject line. Only add
a body when the "why" isn't obvious from the diff or the subject alone. Skip
verification reports, file-by-file changelogs, and footers unless the user
asks for that level of detail.

## Types

- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- style: Code style changes
- refactor: Code refactoring
- test: Adding or modifying tests
- chore: Maintenance tasks

## Example

```text
fix(parser): handle empty configuration files

Returns the default config instead of throwing when the file is empty.
```

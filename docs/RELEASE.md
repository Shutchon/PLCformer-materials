# Publish the materials repository

This package is designed to be a separate Git repository with a fresh history. It must not inherit manuscript files or the main project's history.

## Before publishing

```bash
python tools/verify_artifacts.py
git status --short
git ls-files
```

The distributed files comprise source, tests, synthetic input traces, tensor exports, results, and documentation. No manuscript PDF, LaTeX source, cover letter, or review report belongs in this repository.

## GitHub publication

The intended repository name is `PLCformer-materials` in the `Shutchon` account. This is a proposed destination until the remote has been created and checked.

With the GitHub CLI installed and authenticated to an account permitted to create repositories:

```bash
gh auth login
gh repo create Shutchon/PLCformer-materials --public --source . --remote origin --push
git push origin v1.0.0
```

Use `gh repo create` only if that repository does not already exist. If a repository already exists, inspect its content and ownership before connecting this checkout; do not overwrite or force-push an existing history.

Verify that the new public repository can be opened without logging in. Then cite its actual URL and the immutable `v1.0.0` tag or full commit hash in the manuscript. The main project repository should not be substituted for this package URL.

The repository contains a record of the original source snapshot and checksums. Timing reruns belong under `results/local/` or in a new documented release, not in place of the immutable reference results.

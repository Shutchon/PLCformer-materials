# Materials release 1.1.0

This consolidated package contains the original four workloads and the additional
production-line case. The separate materials repository has its own history;
the main project's manuscript and Git history are excluded.

## Supplementary attachment

The submission route prepared for this version is the single archive
`PLCformer-materials-v1.1.0.zip`. Attach it as reviewer-accessible supplementary
material when submitting the manuscript. It replaces the two older attachments
`PLCformer-materials-v1.0.0.zip` and `PLCformer-production-line-case.zip` for that
submission; those remain historical artifacts and need not be uploaded as well.
The preparation of this archive does not itself upload it to a journal portal.

After extracting the archive, start with the root README. Verify the full
inventory and export a readable report without installing experiment packages:

```sh
python tools/verify_artifacts.py
python tools/summarize_results.py
```

To build an archive from the tagged repository, use:

```sh
git archive --format=zip --prefix=PLCformer-materials-v1.1.0/ \
  --output=../PLCformer-materials-v1.1.0.zip v1.1.0
```

The archive includes a file inventory with SHA-256 hashes. It excludes `.git`,
virtual environments, generated native libraries, local runs, and the manuscript.
Keep the reference results and versioned tag unchanged; new measurements belong
under `results/local/` or in a new documented release.

## Optional future public repository

No public repository URL or DOI is claimed in this release. If the authors later
publish it, verify ownership and existing contents before creating or connecting
a remote. Do not replace an existing history or force-push release tags.
Push the separate materials repository and its exact release tag, then check the
release and its downloadable files without authentication. Only after that check
should the manuscript cite the actual public URL and immutable release/commit.
The supplementary attachment is usable without a public hosting account.

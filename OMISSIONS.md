# Deliberate omissions and verification limits

The archive includes only allowlisted project source and configuration, license
notices, packaging documentation, and a source provenance manifest.

Excluded: every `data/` and `reports/` file, all databases/WAL/SHM companions,
corpus text and samples, numeric research arrays and token IDs, model cache,
review/process logs, historical source blobs, original protection inventories,
release path mappings, Git objects and commit identities, worktrees, Finder
metadata, credentials, environment files, and local application settings.

The source manifest records source paths and digests only. It does not expose
private absolute paths or list private corpus filenames. No protected artifact
was sanitized, regraded or assigned a replacement digest to create this export.

Full historical audit execution requires the omitted evidence, model bytes and
release-adapter dependency closure. The bundled verifier is preserved source;
it cannot report a complete campaign pass without those inputs. Missing-data
failures must not be reported as scientific failures or successful checks.

Later public derivatives, if authorized, must have their own names, hashes,
transformation records and honest verification scope. Do not push the original
private Git history to a public repository.

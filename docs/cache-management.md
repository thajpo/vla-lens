# Research cache management

VLA Lens treats captured tensors and completed research artifacts as durable
data. Feature matrices under `.vla_cache/` are disposable copies made to speed
up repeated analysis. Deleting a feature cache should cost compute time, not
scientific evidence.

## Inspect and protect the working set

```bash
uv run vla-cache --root /path/to/dataset status
uv run vla-cache --root /path/to/dataset pin features/<cache-key>
uv run vla-cache --root /path/to/dataset unpin features/<cache-key>
```

`status` reads the small manifests already stored in `.vla_cache`; it does not
walk the raw episode or activation trees. Each manifest records the normalized
recipe, input-change and output-content fingerprints, shape, dtype, axes, size,
creation/access times, pin state, and enough selector information to rebuild the
entry.

## Prepare shared campaign inputs

A campaign can name several experiments that need the same feature matrix:

```yaml
campaign_id: object-representation-wave
precomputes:
  - name: object-identity
    selector:
      module: pi05.vlm.layers.*
      layers: [0, 4, 8, 12, 17]
      tensor_type: hidden_tokens
      token_kind: image_patch
      policy_calls: [0]
      reduce_tokens: none
  - name: query-localization
    selector:
      module: pi05.vlm.layers.*
      layers: [0, 4, 8, 12, 17]
      tensor_type: hidden_tokens
      token_kind: image_patch
      policy_calls: [0]
      reduce_tokens: none
```

Run:

```bash
uv run vla-cache --root /path/to/dataset prepare --campaign campaign.yaml
```

Identical selectors collapse to one build. A process lock ensures that separate
agents requesting the same key wait for and reuse that build instead of writing
the same Zarr store concurrently.

## Reclaim disk space

```bash
uv run vla-cache --root /path/to/dataset prune \
  --max-gib 10 \
  --min-free-gib 25
```

Prune is a dry run by default. Add `--apply` only after reviewing its list. It
removes least-recently-used, unpinned, manifest-backed entries and is hard-bound
to the dataset's `.vla_cache` directory. It never removes captures, artifacts,
or arbitrary paths.

## Parallel-write guarantees

- One process builds a given cache key at a time.
- Builders write to a temporary sibling, validate it, then swap it into place.
- Interrupted temporary directories are recovered or removed on the next build.
- Dataset-level artifact writes share one process lock, so two workers cannot
  silently overwrite each other's artifact-index row.
- Artifact JSON and Parquet index files are written through same-directory
  temporaries and atomically replaced.

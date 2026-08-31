# Vendored source dependency

`alex/` is a snapshot of the local helper library imported by the phase-field,
mesh, and legacy evaluation scripts in this archive.

The snapshot was copied from `/home/utils/alex` on 2026-06-12 so that the
research-data deposit contains the application-level source used by the
archived workflow.

Add this directory to `PYTHONPATH` before running simulations:

```bash
export PYTHONPATH="$PWD/code/vendor:$PYTHONPATH"
```

The authors must confirm and add the applicable source-code license before
public redistribution.

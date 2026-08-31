# Zenodo upload checklist

## Required author decisions

- [ ] Select a data license, for example CC BY 4.0 or CC0.
- [ ] Select a source-code license and add its full `LICENSE` file.
- [ ] Confirm that the vendored `code/vendor/alex/` code may be distributed
      under that license.
- [ ] Review third-party files such as the Springer Nature class/template and
      remove anything that may not be redistributed.
- [ ] Confirm author names, ordering, affiliations, and ORCID identifiers.
- [ ] Add the associated article DOI when available.

## Archive preparation

- [ ] Remove OS/cache files such as `.DS_Store` and `__pycache__` from the
      upload copy if they are not needed for provenance.
- [ ] Decide whether historical ZIP files and duplicate submission snapshots
      should remain. They are not needed to reproduce the numerical figures.
- [ ] Keep every `.xdmf` beside its referenced `.h5` companion.
- [ ] Keep `results/new_W_whole_boundary/`, `resources/`, all numbered source
      scripts, `00_jobs/`, `000_template/`, and `code/vendor/alex/`.
- [ ] Run `python3 archive_tools/build_manifest.py --checksums`.
- [ ] Test `bash archive_tools/reproduce_manuscript_plots.sh` from a clean
      extracted copy.
- [ ] Open representative XDMF files in ParaView.
- [ ] Confirm that the manuscript builds from
      `68c3b8d0b7dca7b64b8b7a93/main.tex`.

## Suggested Zenodo metadata

**Upload type:** Dataset

**Title:** Research data and code for "Brittle fracture in topology-optimized
structures"

**Creators:** Alexander Schlüter; Ján Pravda; Dustin Roman Jantos; Philipp
Junker; Ralf Müller

**Description:** Simulation inputs, DOLFINx phase-field fracture source code,
HPC workflow scripts, raw XDMF/HDF5 finite-element results, scalar response
histories, postprocessing code, generated figures, and manuscript source for
the study "Brittle fracture in topology-optimized structures."

**Keywords:** brittle fracture; phase-field fracture; topology optimization;
porous materials; DOLFINx; finite-element method; research software

**Related identifier:** Add the article DOI with relation "is supplement to"
after publication.

## Large-upload note

The complete working folder is approximately 26 GB. `results/new_W_whole_boundary`
is approximately 8.9 GB, `resources/` approximately 16 GB, and generated plots
approximately 0.2 GB. Check the current Zenodo per-file and total-record limits
before packaging. Splitting the deposit into logically named archives is
acceptable as long as `README.md`, `MANIFEST.csv`, and `SHA256SUMS` describe
the complete record.

Suggested split:

```text
01_code_documentation_manuscript.tar.gz
02_simulation_inputs_resources.tar
03_primary_results_new_W_whole_boundary.tar
04_generated_plots.tar.gz
```

Avoid compressing HDF5 aggressively without testing; it may consume substantial
time with limited size reduction.


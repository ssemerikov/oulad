# Repository Guidelines

## Project Structure & Module Organization

- `oulad.py` contains feature engineering, model evaluation, plotting, and model serialization. It executes at module load.
- `source/` holds the LaTeX manuscript, `references.bib`, journal class, logos, and compiled artifacts.
- `results/results/` contains archived CSV metrics, PNG plots, environment metadata, and `models/{W30,W60,W90,FULL}/` pipelines. `results.zip` is another archive.
- `dataset.url` identifies the dataset source. Raw OULAD CSVs are not included.

## Build, Test, and Development Commands

Use a Python virtual environment; consult `results/environment.json` for recorded dependency versions.

- `python -m pip install numpy pandas scipy scikit-learn matplotlib seaborn joblib` installs analysis dependencies.
- `python -m py_compile oulad.py` checks syntax without running experiments.
- `OULAD_DIR=/path/to/oulad OUT_DIR=/tmp/oulad-check QUICK=1 python oulad.py` runs all observation windows with fewer cross-validation repeats and bootstrap samples.
- `OULAD_DIR=/path/to/oulad OUT_DIR=/tmp/oulad-full python oulad.py` runs the default experiment. Add `TUNE=1 TUNE_ITER=15` to enable hyperparameter search.
- `make -B -C source` rebuilds the manuscript using pdfLaTeX and BibTeX. Install the manuscript's required LaTeX packages first.

`OULAD_DIR` must contain `courses.csv`, `studentInfo.csv`, `studentRegistration.csv`, `assessments.csv`, `studentAssessment.csv`, and `studentVle.csv`. Choose a separate `OUT_DIR` to preserve archived results.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions and variables, and `UPPER_SNAKE_CASE` constants. Preserve `W30`, `W60`, `W90`, and `FULL` artifact names. Match existing LaTeX labels (`sec:`, `ssec:`) and bibliography keys. No formatter or linter configuration is supplied.

## Testing Guidelines

There is no automated test suite or coverage target. After analysis changes, run syntax validation and a quick experiment; inspect output tables and confirm `reload_identical=True` in `saved_models.csv`. Preserve seed 42, student-grouped splits, observation cutoffs, and exam exclusion unless explicitly changing methodology. After manuscript edits, rebuild and inspect the PDF for citation, layout, and numerical consistency.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout. Use short imperative messages, such as `Fix observation-window filtering`, as a suggested convention. Keep changes focused. PRs should describe changes, list validation commands and environment details, and link relevant issues. Include figure previews or PDF excerpts for visual changes; explain regenerated results and model artifacts.

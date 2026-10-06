# Digit Classification

An EMNIST Digits study that moves from data inspection and baseline classifiers to CNN experiments, writer-separated validation, error analysis, robustness checks, and a small real-world survey evaluation.

## Project contents

- `src/` contains shared data loading, preprocessing, model, training, augmentation, robustness, writer-evaluation, and survey-evaluation code.
- `notebooks/` contains the ordered analysis and experiments, from `01_data_understanding.ipynb` through `11_test.ipynb`.
- `data/raw/emnist-digits.mat` is the EMNIST Digits input.
- `data/survey/realworld_dataset.npz` is the example survey archive used by notebook 11.
- `models/` stores trained models and experiment summaries. Notebook 11 writes generated reports to `outputs/realworld/`.

## Notebook sequence

| Notebook | Purpose |
| --- | --- |
| 01 | Inspect images, labels, duplicates, blank images, and writer overlap. |
| 02–03 | Compare baseline classifiers and train a CNN. |
| 04–06 | Compare optimizers, regularization, and augmentation. |
| 07 | Explore learning rate, dropout, and batch size. This saved experiment selects settings using test scores, so treat it as exploratory. |
| 08 | Select settings with a writer-grouped validation split, then report a test result. |
| 09–10 | Examine test errors and stress the saved model with image corruptions. |
| 11 | Evaluate labeled survey images and create per-writer reports. |

Run notebooks in order when later notebooks depend on models or result files saved by earlier ones. The notebooks use relative paths from the `notebooks/` working directory and import modules from the project root. Configure the notebook environment so the working directory is `notebooks/` and the project root is on Python's import path.

## Dependencies

Use Python 3.12 with the packages listed in `requirements.txt`. Select the project's environment as the notebook kernel.

## Evaluation notes

Notebook 08 keeps writers separate between fitting and validation, but the supplied EMNIST test split overlaps writer IDs with the original training split. Earlier exploratory notebooks also reuse the test split for comparisons. Read the saved test scores as descriptive project results, not as a completely untouched benchmark. The real-world survey scores describe only the submitted survey examples.

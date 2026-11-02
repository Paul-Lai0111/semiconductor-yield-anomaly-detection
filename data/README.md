# SECOM Source Data

The loader downloads the original archive from
`https://archive.ics.uci.edu/static/public/179/secom.zip` into ignored `data/raw/`.
Raw observations are not duplicated in this repository.

Michael McCann and Adrian Johnston (2008), **SECOM**, UCI Machine Learning
Repository. DOI: https://doi.org/10.24432/C54305. Dataset license: **CC BY 4.0**.

| Raw file | Expected SHA-256 |
|---|---|
| `secom.data` | `20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c` |
| `secom_labels.data` | `126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03` |

`secom/data.py` checks these hashes on every load, including cached files. A
checksum change stops execution instead of silently changing the experiment.
The manifest also records the expected 1,567 rows, 590 numeric sensor columns,
1,463 passes, 104 failures, and timestamp range. NaN sensor measurements are
allowed; infinite measurements, unexpected dimensions, and incorrect labels are
rejected.

Labels -1 and +1 map to pass 0 and failure 1. Timestamp strings are parsed as
day/month/year and kept as metadata, not model inputs. The study exports row IDs,
labels, timestamps, and predictions for auditability under the dataset's license.

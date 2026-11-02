"""Download and validate the original UCI files without learning preprocessing statistics."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
DATA_DOI = "https://doi.org/10.24432/C54305"
FILES = ("secom.data", "secom_labels.data")
# Pinned after downloading the published UCI archive; changes require explicit review.
CHECKSUMS = {
    "secom.data": "20f0e7ee434f7dcbae0eea9ffff009a2b57f42d6b0dc9a5bd4f00782c0a3374c",
    "secom_labels.data": "126884cf453705c9e61a903fe906f0665a3b45ce3639e621edc5c93c89627e03",
}


@dataclass
class Dataset:
    X: pd.DataFrame
    y: np.ndarray
    timestamps: pd.DatetimeIndex
    manifest: dict


def load_dataset(cache_dir: Path) -> Dataset:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not all((cache_dir / name).is_file() for name in FILES):
        print(f"Downloading original UCI files: {DATA_URL}", flush=True)
        with urlopen(DATA_URL, timeout=60) as response:
            archive_bytes = response.read()
        with ZipFile(BytesIO(archive_bytes)) as archive:
            for name in FILES:
                content = archive.read(name)
                digest = sha256(content).hexdigest()
                if name in CHECKSUMS and digest != CHECKSUMS[name]:
                    raise ValueError(f"UCI checksum mismatch for {name}; cached data not replaced.")
                (cache_dir / name).write_bytes(content)

    hashes = {name: sha256((cache_dir / name).read_bytes()).hexdigest() for name in FILES}
    for name, expected in CHECKSUMS.items():
        if hashes[name] != expected:
            raise ValueError(f"Checksum mismatch for {name}; review the raw data before running.")

    X = pd.read_csv(cache_dir / "secom.data", sep=r"\s+", header=None)
    lines = (cache_dir / "secom_labels.data").read_text().strip().splitlines()
    labels, timestamps = zip(*(line.split(maxsplit=1) for line in lines), strict=True)
    labels = np.asarray(labels, dtype=int)
    if X.shape != (1567, 590) or len(labels) != len(X):
        raise ValueError(f"Unexpected SECOM dimensions: features={X.shape}, labels={len(labels)}")
    if set(labels) != {-1, 1} or np.count_nonzero(labels == 1) != 104:
        raise ValueError("Unexpected SECOM class labels or failure count.")
    if np.isinf(X.to_numpy()).any():
        raise ValueError("Sensor values must be finite or missing.")
    X.columns = [f"sensor_{index:03d}" for index in range(X.shape[1])]
    dates = pd.DatetimeIndex(
        pd.to_datetime([value.strip('"') for value in timestamps], format="%d/%m/%Y %H:%M:%S")
    )
    y = (labels == 1).astype(np.int8)
    return Dataset(
        X,
        y,
        dates,
        {
            "url": DATA_URL,
            "doi": DATA_DOI,
            "license": "CC BY 4.0",
            "sha256": hashes,
            "rows": len(X),
            "sensor_columns": X.shape[1],
            "passes": int(np.count_nonzero(y == 0)),
            "failures": int(y.sum()),
            "missing_fraction": float(X.isna().to_numpy().mean()),
            "timestamp_min": dates.min().isoformat(),
            "timestamp_max": dates.max().isoformat(),
        },
    )

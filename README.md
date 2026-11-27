# PlanetariUM-ML

Machine learning on half a century of solar system dynamics: how far can a model predict the motion of the planets before its error exceeds 1% of the distance to the Sun?

PlanetariUM-ML is a research project studying the predictability limits of orbital motion through machine learning. It uses the hourly state vectors of 16 solar system bodies across almost 52 years of high-precision ephemeris data as a benchmark for sequence modeling and long-horizon time-series prediction.

## Dataset

The full dataset is archived and citable:

> **PlanetariUM-ML: Hourly state vectors of 16 solar system bodies from JPL Horizons, 1975-2026**
> [https://doi.org/10.5281/zenodo.24574075](https://doi.org/10.5281/zenodo.24574075)

- **Source:** JPL Horizons (Jet Propulsion Laboratory, NASA), retrieved in a single session on 27 November 2026, 01:30 CET, and frozen as downloaded.
- **Bodies:** the eight planets, the four IAU-recognised dwarf planets (Pluto, Eris, Haumea, Makemake), and four main-belt asteroids (Ceres, Vesta, Pallas, Meliboea).
- **Format:** Cartesian state vectors (x, y, z in AU; vx, vy, vz in AU/day), ICRF heliocentric frame, hourly cadence, 7,281,040 rows in total (455,065 per body).
- **Split:** time-based, fixed at dataset creation and before any model training: train 1975-2024, test 2025-2026.

The dataset is fully reproducible with the fetcher in this repository (see below), which retrieves the same data from the public JPL Horizons API. Note that JPL may update its ephemeris solutions over time, so a re-generated dataset is physically equivalent but not guaranteed to be bit-identical to the archived version. For the exact data used in the paper, use the DOI.

## Repository structure

```
planetarium-ml/
├── planetarium_fetch.py   # Fetcher + merge tool (download, status, merge)
├── data/
│   ├── raw/                # Raw Horizons chunks (not tracked by git)
│   └── planetarium_ml_dataset.csv   # Merged dataset (not tracked by git)
└── README.md
```

`data/` is excluded from version control (see `.gitignore`): regenerate it locally with the fetcher, or download the frozen version from Zenodo.

## Usage

Requires Python 3.10+ with `requests` and `pandas`.

```bash
# Download all raw chunks from JPL Horizons (resumable, cached in data/raw/)
python planetarium_fetch.py download

# Check download progress per body
python planetarium_fetch.py status

# Merge raw chunks into the final dataset
python planetarium_fetch.py merge
```

The fetcher is resumable: interrupted downloads continue from the last completed chunk per body, so the pipeline survives network failures without re-downloading anything.

## Citation

If you use this dataset or code, please cite the Zenodo record:

```bibtex
@dataset{frutos_plaza_2026_planetariumml,
  author  = {Frutos Plaza, Mois{\'e}s},
  title   = {{PlanetariUM-ML}: Hourly state vectors of 16 solar system bodies from {JPL} {Horizons}, 1975-2026},
  year    = {2026},
  doi     = {10.5281/zenodo.24574075},
  url     = {https://doi.org/10.5281/zenodo.24574075}
}
```

## License

The dataset is derived from public JPL Horizons data (NASA/JPL-Caltech) and is distributed under CC-BY 4.0. The code in this repository is released under the MIT license.

## Acknowledgements

Ephemeris data provided by the Jet Propulsion Laboratory's Horizons system, a service of the Solar System Dynamics Group of NASA/JPL-Caltech.

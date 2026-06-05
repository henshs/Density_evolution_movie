# Density_evolution_movie
This script generates 2D snapshots of Cactus-based simulations and makes movie of the density evolutions.

This repository contains a high-performance Python pipeline tool to visualize 2D Matter Density (ρ) evolution profiles from Einstein Toolkit/Cactus simulations. Leveraging the `kuibit` physics analysis framework, it extracts data, transforms grid architectures, and uses parallel workers to render fluid dynamics time-steps into a shareable `.mp4` video format.

## Features
- **Parallel Computing:** Integrates `joblib` and `multiprocessing` to drastically scale up image processing frame generation speeds.
- **Dynamic Grid Conversion:** Resamples 2D data directly onto customized spatial coordinates (`kuibit.grid_data.UniformGrid`).
- **Memory Optimized:** Employs explicit cache clearing routines (`rho.clear_cache()`) and individual figure shutdowns to minimize multi-threaded resource bloat.
- **Progress Tracking:** Beautiful CLI terminal rendering utilizing `tqdm` and `tqdm_joblib`.

## Prerequisites

Ensure you have a modern Python 3 environment active. Install the necessary physics computation frameworks and media visualization libraries.

```bash
pip install numpy matplotlib astropy opencv-python tqdm joblib tqdm-joblib kuibit
```

*Note: Access to the `kuibit` library requires its specific environmental system requirements depending on your scientific calculation cluster settings.*

## Repository Organization

The visualization engine relies on specific relative file positions. Structure your directory as follows:

```text
├── your-density-data-location/    # Cactus 2D data files
└── your-repo-name/
    ├── make_movie.py              # Main execution script
    ├── README.md                  # This file
    ├── frames_dir/                # Automatically generated; stores individual image matrices
    └── movie_dir/                 # Automatically generated; holds the output .mp4 video
```

## How to Use

Run the program from the script folder using your preferred terminal engine.

### Default Mode (Parallel Generation)
By default, the script detects available hardware processors and launches multi-core workers to speed up processing (leaving 2 system safety cores open).

```bash
python make_movie.py
```

### Advanced Arguments Configuration
Tailor frame parameters, render performance configurations, and temporal skips via standard flag parameters:


| Command Line Flag | Type | Default Value | Functional Role |
| :--- | :--- | :--- | :--- |
| `--stepsize` | `int` | `1` | Stride gap interval between processing data frames (e.g., `2` selects every second step). |
| `--fps` | `int` | `60` | Speed frame-rate of the resulting video track output. |
| `--serial` | `flag` | `False` | Forces sequential execution without launching parallel processors (useful for debug testing). |
| `--nproc` | `int` | `Cores - 2` | Allocates custom core quantities to manage execution tasks. |

#### Examples:
To generate a fast preview checking every 5th iteration framework step at a 24fps export rate:
```bash
python make_movie.py --stepsize 5 --fps 24
```

To limit memory consumption or run on lightweight server environments by focusing threads into 4 specific processors:
```bash
python make_movie.py --nproc 4
```

## Output Production
Once operations finish running, files populate inside these output targets:
1. **`frames_dir/`**: Populated with individual raw plot graphs labeled chronologically (`rho_00000.png`). 
2. **`movie_dir/rho-movie.mp4`**: Compiled structural time series output containing mapped analytical properties matching simulated boundary conditions.

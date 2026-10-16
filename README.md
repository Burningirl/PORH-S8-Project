# PORH Project

An automatic pipeline for analyzing post-occlusive reactive hyperemia (PORH) signals. The pipeline detects key physiological events in Laser-Doppler perfusion signals and then predicts PORH parameters using an LSTM model.

## Overview

1. **Event detection:** identifies key points in the perfusion signal (e.g., [occlusion release, peak, return to baseline])
2. **Parameter prediction:** an LSTM model predicts PORH parameters such as [peak flux, time to peak, ...]

## Repository Structure

```
├── src/        # source code
├── data/       # input signals (or instructions to obtain them)
├── results/    # figures and output tables
└── Quick_File_Navigation_Guide.md
```

For a detailed description of every file, see the [Quick File Navigation Guide](Quick_File_Navigation_Guide.md).

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
pip install -r requirements.txt
```


## Data

The data are not available to the public to use or see.

## License

MIT License

# PyData DC 2018

## Presentation

The slides can be downloaded at: [download link](https://github.com/thomasjpfan/pydata2018_dc_skorch/raw/master/slides.pdf).

## To clone this repo run:

```bash
git clone --depth 1 https://github.com/thomasjpfan/pydata2018_dc_skorch
```

## Setup

To run the notebook locally, please following the following setup procedure:

1. Install dependencies: `conda env create -n pydata_dc_2018 -f environment.yml `
1. Activate env: `conda activate pydata_dc_2018`

## Part 3

1. Follow [Kaggle's installation and configuration documentation](https://github.com/Kaggle/kaggle-api#installation) to install and configure the kaggle cli
1. Go to Kaggle's [2018 Data Science Bowl Competition](https://www.kaggle.com/c/data-science-bowl-2018), click on "Late Submission" and accept the terms and conditions to get access to the data.
1. Run `./dl_extract_prepare.sh` to download, extract and prepare the data.

## Run Jupyter Lab

1. Activate env: `conda activate pydata_dc_2018`
1. Launch jupyter lab: `jupyter lab`

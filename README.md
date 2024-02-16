# FreeSurfer/FastSurfer reconstruction quality prediction

This repo contains ML scripts for quality prediction and inference utility

## Deploy

Use python3.11

```bash
python3 -m venv .env
source .env/bin/activate
pip instal -r requirements.txt
```

## Usage

```bash
python3 infer.py --model <path-to-model> --input-csv <path-to-fsqc-result>
```

Note: result of multiple subject QA can be providad as an input, in this case classes will be printed in the .csv order, separated by 'space'.

## Example 1:

<b>Input:</b>

|feature    |value    |
|-----------|---------|
|wm_snr_orig|9.817538 |
|gm_snr_orig|6.132985 |
|wm_snr_norm|17.524861|
|gm_snr_norm|6.141379 |
|holes_lh   |20.000000|
|holes_rh   |10.000000|
|defects_lh |25.000000|
|defects_rh |21.000000|
|topo_lh    |1.000000 |
|topo_rh    |2.900000 |
|con_snr_lh |3.506817 |
|con_snr_rh |3.540243 |



<b>Command:</b>

```bash
python3 infer.py --model catboost_model0.80_pipeline.pkl --input sub-000103_ffs_q2.csv
```

<b>Output:</b>

```bash
0
```

## Example 2:

<b>Input:</b>

[data/mr-art_q3_incomplete/fsqc-results.csv](data/mr-art_q3_incomplete/fsqc-results.csv)

<b>Command:</b>

```bash
python3 infer.py --model catboost_model0.80_pipeline.pkl --input data/mr-art_q3_incomplete/fsqc-results.csv
```

<b>Output:</b>

```bash
3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 2 3 3 3 3 3 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
```
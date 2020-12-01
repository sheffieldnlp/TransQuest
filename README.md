# TransQuest : Transformer based Translation Quality Estimation. 

This is an updated version of TransQuest that:

- Uses the newest versions of the HuggingFace Transformers and Datasets libraries.
- Inclues models for Word-Level Quality Estimation.

## Instalation from Source

```
git clone https://github.com/sheffieldnlp/TransQuest.git
cd TransQuest
pip install --editable ./
```

## Run on the WMT 2020 Task 2 Data

### Training and Evaluation (Dev Set)

```
#!/bin/bash

transquest_dir="/experiments/falva/tools/TransQuest"
data_dir="/data/falva/wmt20qe_hter/for_qe/en-de"
output_dir="/experiments/falva/wordlevel_qe/wmt2020qe_hter_base"

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "${transquest_dir}/transquest_cli/run_wordlevel.py" \
    --task_name "qe-wordlevel" \
    --model_name_or_path "xlm-roberta-base" \
    --dataset_name "wmt20qe_hter" \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}"/ \
    --do_train --num_train_epochs 5 \
    --do_eval \
    --overwrite_output_dir

```
## Prediction on Test Set

```
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 "${transquest_dir}/transquest_cli/run_wordlevel.py" \
    --task_name "qe-wordlevel" \
    --model_name_or_path "${output_dir}" \
    --dataset_name "wmt20qe_hter" \
    --data_dir "${data_dir}" \
    --output_dir "${output_dir}" \
    --do_predict \
```
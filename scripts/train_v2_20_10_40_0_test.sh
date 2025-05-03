#!/bin/bash
echo ""
echo ./configs/v2_nn_20-na_10-nab_40-test.yaml
#accelerate launch
python3 /n/home12/cfpark00/ML/tools/run_sft_accelerate.py ./configs/v2_nn_20-na_10-nab_40-test.yaml --overwrite


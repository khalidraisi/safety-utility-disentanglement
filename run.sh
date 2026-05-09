#!/bin/bash
set -e

git clone https://github.com/khalidraisi/safety-utility-disentanglement.git
cd safety-utility-disentanglement
pip install -r requirements.txt

python3 code/extract_activations.py

python3 code/locatelayers/run_ll.py
python3 code/dom/run_dom.py
python3 code/svd/run_actsvd.py
python3 code/svd/plot_csvd.py
python3 code/ablation/run_experiment.py
python3 code/ablation/bench_advbench.py
python3 code/ablation/bench_helpsteer.py
python3 code/ablation/analyze.py

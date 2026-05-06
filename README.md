# Disentangling Safety and Utility in LLM Activation Spaces

**CS 639 — Advanced NLP | University of Wisconsin-Madison**  
Khalid Al-Raisi, Mohammed Al-Hinai, Will Frost, Vivek Saravanan, Zakereya Chehime, David Wu

---

## Overview

This project investigates whether *safety* (refusal of harmful instructions) and *utility* (ability to follow benign instructions) are geometrically separable within the internal activation spaces of large language models.

We test the **Linear Representation Hypothesis (LRH)** — that concepts occupy linear subspaces — by extracting safety and utility directions using two complementary methods:

- **Difference of Means (DoM)** — extracts a single refusal direction by contrasting mean activations of harmful vs. harmless prompts
- **Activation-aware SVD (actSVD)** — identifies higher-dimensional safety/utility subspaces via singular value decomposition

We then test whether these subspaces are orthogonal and causally independent via ablation experiments.

---

## Models

- Gemma3 1b it
- Gemma4 E2B it
- Llama3.2-3b it
- Qwen3.5-4b it

---

## Datasets

| Dataset | Purpose |
|---|---|
| [AdvBench](https://github.com/llm-attacks/llm-attacks) | Harmful prompts (safety pole) |
| [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) | Harmless instructions |
| [HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | High/low utility prompts |

---

## Installation

```bash
git clone https://github.com/khalidraisi/safety-utility-disentanglement.git
cd safety-utility-disentanglement
pip install -r requirements.txt
```

Requires Python 3.10+

---

## Project Structure

```
.
├── code/
│   ├── EDA/            # Exploratory data analysis and PCA visualization
│   ├── ablation/       # Safety and utility ablation
│   ├── dom/            # Difference-of-Means direction
│   ├── locatelayers/   # Layer selection and similarity analysis
│   ├── output/         # Experimental results and figures
│   ├── svd/            # actSVD subspace extraction
│   └── extract_activations.py    # Main script for extracting model hidden states
├── freport/            # Final report
│   ├── chapters/       # Individual report chapters
│   ├── report.tex      # Main LaTeX source
│   ├── report.pdf      # Compiled final report
│   ├── custom.bib      # Bibliography
│   ├── anthology.bib.txt
│   ├── acl_latex.tex
│   ├── acl.sty
│   └── acl_natbib.bst
├── mreport/            # Milestone report source files
├── results/            # Aggregated experiment outputs
├── mreport.pdf         # Milestone report
└── freport.pdf         # Final report
```

Find the extracted activations [here](https://drive.google.com/drive/folders/1kUK2QM1RxWK4J2mQQNQGo1SaMWzvjtNI?usp=drive_link)
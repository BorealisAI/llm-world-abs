# Do LLMs Build World Representations? Probing through the Lens of State Abstraction

## Setup environment

`pip install -r requirements.txt`

**Hardware requirement**: all experiments can be done with one A100 GPU.

## Synthesize dataset

`python llm_world_abs/src/{gripper, cook}/syndata/generate_dataset.py`

## Fine-tuning LLMs

`sh llm_world_abs/src/{gripper, cook}/scripts/run_ft.sh`

## Training and testing probing models

`sh llm_world_abs/src/{gripper, cook}/scripts/run_probe.sh`

## Making plots 

* make sure you install `latex`
* check `viz/make_plots.ipynb`

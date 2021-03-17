Code for "On Losses for Modern Language Models"  
[ACL Anthology](https://www.aclweb.org/anthology/2020.emnlp-main.403/), [arxiv](https://arxiv.org/abs/2010.01694)

This repository is primarily for reproducibility and posterity. It is not maintained.

Thank you to the [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/tree/0399d32c75b4719c89b91c18a173d05936112036) and NYU's 
[jiant](https://github.com/nyu-mll/jiant/tree/14d9e3d294b6cb4a29b70325b2b993d5926fe668) repos for their code which helped create the base of this repo. 

# Setup
Only tested on python3.6.

```
python -m pip install virtualenv
virtualenv bert_env
source bert_env/bin/activate
pip install -r requirements.txt
```


# Usage
The code enables pre-training a transformer (size specified in bert_config.json) using any combination of the following tasks (aka modes/losses):
"mlm", "nsp", "psp", "sd", "so", "rg", "fs", "tc", "sc", "sbo", "wlen", "cap", "tf", "tf_idf", or "tgs". See paper for details regarding the modes.
NOTE: PSP (previous sentence prediction) is equivalent to ASP (adjacent sentence prediction) from the paper. RG (referential game) is equivalent to QT (quick thoughts variant) from the paper.

They can be combined using any of the following methods:
- Summing all losses (default, incompatible between a small subset of tasks, see paper for more detail)
- Continuous Multi-Task Learning, based on ERNIE 2.0 (--continual-learning True)
- Alternating between losses (--alternating True)

With the following modifiers:
- Always using MLM loss (--always-mlm True, which is the default and highly recommended, see paper for more details)
- Incrementally add tasks each epoch (--incremental)
- Use data formatting for tasks, but zero out losses from auxiliary tasks (--no-aux True, not recommended, used for testing)

Set paths to read/save/load from in paths.py

To create datasets, see data_utils/make_dataset.py

For tf_idf prediction, you need to first calculate the idf score for your dataset. See idf.py for a script to do this.

## Pre-training
To run pretraining :
`bash olfmlm/scripts/pretrain_bert.sh --model-type [model type]`
Where model type is the name of the model you want to train. If model type is one of the modes, it will train using mlm and that mode (if model type is mlm, it will train using just mlm).
The --modes argument will override this default behaviour. If model type is not a specified mode, the--modes argument is required.

## Distributed Pretraining
Use pretrain_bert_distributed.sh instead.
`bash olfmlm/scripts/pretrain_bert_distributed.sh --model-type [model type]`

## Evaluation
To run evaluation:
You will need to convert the saved state dict of the required model using the convert_state_dict.py file.
Then run:
`python3 -m olfmlm.evaluate.main --exp_name [experiment name]`
Where experiment name is the same as the model type above. If using a saved checkpoint instead of the best model, use the --checkpoint argument.

## Citation
If this code was useful, please cite the paper:
```
@inproceedings{aroca-ouellette-rudzicz-2020-losses,
    title = "{O}n {L}osses for {M}odern {L}anguage {M}odels",
    author = "Aroca-Ouellette, St{\'e}phane  and
      Rudzicz, Frank",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.403",
    doi = "10.18653/v1/2020.emnlp-main.403",
    pages = "4970--4981",
}
```



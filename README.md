# Smart Multi-objective Evolutionary GANs for Tabular Data Synthesis

## Environment Setup

Before running the training scripts, ensure that you have the required Python packages installed. You can install them using the following commands:

```bash
pip install lmdb
pip install ctgan
pip install rdt
pip install pyreadstat
```

## Training the CTGAN
```bash
python ctgan_model.py --problem [chosen census data] --suffix [the folder suffix to store the trained model] --seed [seed]
```
Example:
```bash
python ctgan_model.py --problem Canada --suffix 1000 --seed 42
```

## Training the TabularSOMEGAN 
```bash
python train.py --problem [chosen census data] --post_fix [the folder suffix to store the trained model] -mu [population size in evolution] --seed [seed] --select [the training steps to do one multi-objective selection]
```
Example: 
```bash
python train.py --problem Canada --post_fix select8 -mu 4 --seed 42 --select 8
```

## Evaluate the TabularSOMEGAN
```bash
python eval.py --path [model path xxx.pth] --problem [chosen census data]
```

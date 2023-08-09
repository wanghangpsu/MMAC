# Improved Activation Clipping for Universal Backdoor Mitigation and Test-Time Detection

This is the implementation of the paper: Improved Activation Clipping for Universal Backdoor Mitigation and Test-Time Detection



This repository includes:
- Training code for the clean model and attacked model.
- MMAC backdoor mitigation code.
- MMDF backdoor defense framework.



## Requirements
Ubuntu 20.04
Python 3.7
- Install required python packages:
```bash
$ pip install numpy
$ pip install torch
$ pip install torchvision
$ pip install matplotlib
$ pip install scipy
$ pip install pillow
```


## Training
For clean model training,
run command:
```bash
$ ./run_clean.sh
```
Which gives 10 clean models saved in ./clean0 to ./clean9 folders

For attack models (BadNet attack)
run_command:
```bash
$ ./run_attack.sh
```

Which gives 10 attacked modes saved in ./model0 to ./model9 folders

## MMAC Mitigation
Run_command:
```bash
$ ./run_mmac.sh
```


## MMDF

Run:
```bash
$ ./run_mmdf.sh
```

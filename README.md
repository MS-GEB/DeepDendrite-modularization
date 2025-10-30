# DeepDendrite-modularization
Extension to [DeepDendrite](https://github.com/pkuzyc/DeepDendrite) with modularized layer APIs for building & data-driven training of deep, detailed multi-compartment neural networks

Code associated with the paper "Gan He, Tiejun Huang and Kai Du, (2025). Going deeper with morphologically detailed neural networks by error-backpropagating mirror neuron" (soon on biorxiv when ready).  Demo code for training, testing & transfer attack are provided.
<div align="center">
  <img src="https://github.com/MS-GEB/DeepDendrite-modularization/blob/main/img/overview.jpg">
</div>

## Directories
**src:** Source code of DeepDendrite with support for layer APIs and modified NEURON simulator.  
**mod:** MOD files required by layer APIs.  
**coredat:** Folders containing layer APIs and generated network models for training & validation.  
**adversarial:** Transfer adversarial attack on deep detailed network.  

## System Requirements
A CUDA capable GPU is required.

Following the dependencies of DeepDendrite:
- OS: Ubuntu >= 20.04
- CMake >= 3.10.2
- MPI >= 3.0
- Python >= 3.6
- NVIDIA HPC SDK >= 22.1
- CUDA Toolkit >= 10.1

Also the following dependencies are tested on Windows WSL2:
- OS: Ubuntu-WSL == 24.04
- GCC == 13.3.0
- CMake == 3.28.3
- Python == 3.8.20
- NVDIA HPC SDK == 25.7
- CUDA Toolkit == 12.9

## Installation Guide
The modified NEURON and DeepDendrite will be installed in ./install
### 1. Compile the modified NEURON simulator
**NOTICE:** Do not load NVIDIA HPC SDK to compile NEURON
```
cd src/nrn_modify
./configure --prefix="$PWD/../../install" --without-iv --with-paranrn --with-nrnpython=`which python` --disable-rx3d
make -j8 && make install
```
### 2. Compile DeepDendrite
**NOTICE:** Load NVIDIA HPC SDK to compile DeepDendrite
```
cd src/DeepDendrite
module load /path/to/your/nvhpc/modulefiles
./install.sh
```

## Demo for training & validation
A fully-connected detailed network with five hidden layers for image classification on MNIST
### 1. Compile MOD files
```
./install/x86_64/nrnivmodl ./mod
```
### 2. Generate dataset for DeepDendrite
```
python3 gen_bin_data.py
```
### 3. Build networks and perform training & validation
```
cd coredat/demo_train
python3 task.py
```
Training takes around 3 days on single A100 GPU

## Demo for transfer attack
### 1. Train a 20-layer ResNet as base network and generate adversarial samples
Requires [TensorFlow](https://github.com/tensorflow/tensorflow) and [FoolBox](https://github.com/bethgelab/foolbox)
```
cd adversarial
python3 attack_resnet20.py
python3 gen_adv_bin_data.py
```
### 2. Validate accuracy on adversarial samples
```
cd coredat/demo_train
python3 test_adv.py
```

## License
This project is covered under the Apache License 2.0.

## Contact
For any questions please contact Gan He via email (hegan@mail.tsinghua.edu.cn).

# GP-RNN_UAI2019 oral presentation
Implementaion of Gaussian Process Recurrent Neural Networks (GP-RNN) developed in "Neural Dynamics Discovery via Gaussian Process Recurrent Neural Networks", Qi She, Anqi Wu, UAI2019

If you find this useful in your research, please consider citing:

    @article{Qi2019Nerual,
    title={Neural Dynamics Discovery via Gaussian Process Recurrent Neural Networks},
    author={Qi She, Anqi Wu},
    Conference={The Conference on Uncertainty in Artificial Intelligence (UAI), 2019},
    year={2019}
    }

Latent dynamics discovery is challenging in extracting complex dynamics from high dimensional noisy neural data. we propose a novel latent dynamic model that is capable of capturing nonlinear, non-Markovian, long-short term time-dependent dynamics via recurrent neural networks and tackling complex nonlinear embedding via non-parametric Gaussian process. The model can be easily adapted to exploring latent representations of time series with either Gaussian (continuous) or Poisson (discrete) observations.

## 1. Model overview
### Model
Our model composes RNN (e.g.LSTM) with hidden states (**h**) encoded in the prior distribution of latent states (**z**), and then maps to the noisy-free space of observations (**x**).  
<div  align="center">    
<img src="https://raw.githubusercontent.com/sheqi/GP-RNN_UAI2019/master/figs/scheme.png" width = "350" height = "180" alt="Image of GP-RNN" align=center />
</div>

###  inference
The inference methods are non-trivial with Gaussian and Poisson observations, one of powerful inference networks is bi-LSTM as shown below, please see the details in the paper.
<div  align="center">    
<img src="https://raw.githubusercontent.com/sheqi/GP-RNN_UAI2019/master/figs/bi-LSTM.png" width = "350" height = "180" alt="Image of GP-RNN" align=center />
</div>

## 2. Dependences
- Ubuntu 16.04
- python3.6
- tensorflow-gpu 1.5.0 (cpu version is also compatible with minimal modification in the code)

## 3. Start
- recommend install in virtual environment
```
$ conda create -n yourenvname python=3.6 anaconda
```
- Install the required the packages inside the virtual environment
```
$ source activate yourenvname
$ pip install -r requirements.txt
```

## 4. Demo
- please run jupyter lab to test attached demo.

Feel free to submit an issue to the repo and contact the corresponding author [Qi She](sheqi1991@gmail.com) if you have any problem or would like to contribute.

MVA Deep RL Project
==============================

**Authors**: Raphaël Rozenberg, Arthur Pignet,  Frédéric Zheng

The aim of this project is to reimplement the algorithm from the paper [Continious control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) 

Project Organization
------------
    │   .gitignore
    │   LICENSE
    │   README.md           <- The top-level README.
    │   requirements.txt    <- List of third parts libraries used in this project.
    │
    ├───models              <- Trained models. Please don't push them
    │       .gitkeep
    │
    ├───notebooks           <- Jupyter notebooks used in Colab.
    │       DDPG.ipynb
    │       
    │
    ├───results
    │   │   .gitkeep
    │   │   log1.npy
    │   │   loss_error.png
    │   │
    │   └───figures
    │           actor_loss.png
    │           critic_loss.png
    │           returns.png
    │
    ├───src                <- Source code used in this project.
    │       agents.py                   <- DDPG Agent main class.
    │       data.py                     <- Data structure classes.
    │       environments.py             <- Wrapped environment classes.
    │       interaction_loops.py        <- Functions used to interact with the env and train the agent.
    │       logs.py                     <- Log utils.
    │       networks.py                 <- Neural network definitions.
    │       plot_utils.py               <- Plot utils.
    │       replay_buffer.py            <- Replay buffer class.
    │       utils.py                    <- General utils.
    │       __init__.py
    │
    └───_can_be_deleted    <- Trash bin (!! git ignored)


Installation 
---------------

Clone this repo and install the requierement. Note that we are using (Jax)[https://jax.readthedocs.io/en/latest/], so you need to be in a compatible environment. 

```bash
$ git clone https://github.com/arthurPignet/mva-drl-project.git
$ cd mva-drl-project/
$ pip install -v .
```
Tested configuration
---------------
This installation was tested in Google Colab (free version) and with a Azure VM running with Ubuntu 20.04.4 LTS

Result example
---------------

Unfortunatly do to a lack of computational power, we were not able to replicate the results of the paper. However we manage to have an agent learning something on the Inverted Pendulum environment.

**Losses**

![Actor loss](https://github.com/arthurPignet/mva-drl-project/blob/main/results/figures/actor_loss.png)
![Critic loss](https://github.com/arthurPignet/mva-drl-project/blob/main/results/figures/critic_loss.png)

**Returns' evolution** 

![Returns](https://github.com/arthurPignet/mva-drl-project/blob/main/results/figures/returns.png)


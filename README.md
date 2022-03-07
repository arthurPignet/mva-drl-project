MVA Deep RL Project
==============================

**Authors**: Raphaël Rozenberg, Arthur Pignet,  Frédéric Zheng

The aim of this project is to reimplement the algorithm from the paper [Continious control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf) 

Project Organization
------------

    │   .gitignore
    │   LICENSE
    │   README.md           <- The top-level README.
    │   requirements.txt
    │
    ├───models              <- Trained and serialized models.
    │       .gitkeep
    │
    ├───notebooks           <- Jupyter notebooks used in Colab.
    │       .gitkeep
    │
    ├───report              <- project LateX report. 
    │   │   report.tex
    │   │   rl_biblio.bib
    │   │
    │   └───figures
    │           .gitkeep
    │
    ├───results
    │       .gitkeep
    │
    ├───src                  <- Source code used in this project.
    │   │   agent.py
    │   │   plot_utils.py   <- Script for visualization, graph, animation...
    │   │   utils.py
    │   │   __init__.py
    │   │
    │   └───environments    <- Wrapped environment classes.
    │           inverted_pendulum.py
    │           reacher_v1.py
    │           __init__.py
    │
    └───_can_be_deleted     <- Trash bin (!! git ignored)


Result example
---------------
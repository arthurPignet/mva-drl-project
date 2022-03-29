import matplotlib.pyplot as plt
import numpy as np


def plot_log(path: str,
             n: int = 100  # number of steps for moving average/10
             ):
    with open(path) as f:
        dico = {}
        while True:
            line = f.readline()
            if line.startswith("iteration"):
                continue
            else:
                line = line.replace('|', '').replace(':', '').split()
                for i in range(len(line) // 2):
                    if line[2 * i] not in dico:
                        dico[line[2 * i]] = []
                    dico[line[2 * i]].append(float(line[2 * i + 1]))
            if not line:
                break
        dico = {k: np.array(v) for k, v in dico.items()}

    for k, v in dico.items():
        plt.plot(np.convolve(v, np.ones(n) / n, 'valid'))
        plt.title(k)
        plt.show()

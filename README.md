# Density Aware Farthest Point Sampling
This repository contains the code necessary to replicate the results presented in the TMLR paper titled ["Density Aware Farthest Point Sampling"](https://openreview.net/forum?id=vI47lgIfYc). This repository includes a Jupyter notebook that replicates the experiments on the smaller QM7 dataset using KRR in just a few minutes of runtime.

## Python Packages
- Python (>= 3.7)\
- Pytorch 1.11.0\
- Install packages in requirements.txt

## Repository Structure

```plaintext
.
├── datasets/                   # Folder containing code to access data. It is also for data storage. 
│   ├── Datasets_Class.py           # Code for downloading, reading, and preprocessing datasets.
│    
├── notebooks/                  # Folder containing Jupiter notebooks.
│   ├── experiments.ipynb           # Jupyter Notebook replicating experiments using KRR on QM7, including data preprocessing, 
│                                   data selection and regression task
│   
├── Passive_sampling/           # Folder containing code to implement considered selection approaches.
│   ├──fps_selectors.py             # Code for implementing DA-FPS and FPS.
│   ├──sampling_process.py          # Code for implementing data sampling strategies used in the paper.
│                                    See experiment.ipynb for additional details
├── utils/                      # Folder containing basic code to run and plot experiments.
│   ├──FNN.py                       # Code containing the FNN architecture, training and testing procedures.
│   ├──plots.py                     # Code plotting the result of the experiments.
    
└── README.md                   #  README file.
└── requirements.txt            # Python packages are required to run the code.
```

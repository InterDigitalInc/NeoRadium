# A Self-Refining Multi-Layer Receiver Pipeline

This repository contains the Python modules and Jupyter notebooks used to reproduce the experiments presented in:

> **[A Self-Refining Multi-Layer Receiver Pipeline](https://ieeexplore.ieee.org/abstract/document/11443343)**  
> *2025 59th Asilomar Conference on Signals, Systems, and Computers*  
> https://ieeexplore.ieee.org/abstract/document/11443343

The project implements a deep learning-based channel estimation framework for multi-layer MIMO communication systems and evaluates its impact on end-to-end receiver performance.

---

**Workflow Overview**

The experiments are organized into five stages:

1. Dataset Generation – Generate training, validation, and test datasets.
2. Model Training – Train the channel estimation neural network.
3. NMSE Evaluation – Measure channel estimation accuracy.
4. BLER Evaluation – Measure end-to-end communication reliability.
5. HARQ Evaluation – Measure end-to-end communication throughput.

The notebooks should generally be executed in this order.

---

**Dataset Generation**

`MLChEstDataGen.ipynb`: Generates the training, validation, and test datasets used throughout the experiments. The notebook simulates a multi-layer MIMO communication system and creates supervised learning samples consisting of:

- Received resource grids and DMRS information (inputs)
- Ground-truth channel responses (targets)

> ⚠️ The generated datasets require approximately **30 GB** of disk space.

---

**Model Training**

`MLChEstTrain.ipynb`: Trains the deep neural network used for multi-layer channel estimation. The notebook:

- Loads the generated datasets
- Initializes the channel estimation network
- Trains the model using supervised learning
- Selects the best model based on validation performance
- Saves the trained model to the `Models` directory

An already-trained model is included with the repository. You may either train a new model or proceed directly to the evaluation notebooks using the provided model.

---

**Model Evaluation**

Three complementary evaluation notebooks are provided:

`MLChEstEvaluateNMSE.ipynb`: Evaluates channel estimation accuracy using Normalized Mean Squared Error (NMSE). This notebook compares the predicted channel estimates against the corresponding ground-truth channels and provides a direct measure of estimation performance.


`MLChEstEvaluateBLER.ipynb`: Evaluates end-to-end communication performance using Block Error Rate (BLER). The notebook compares multiple channel estimation methods across a range of signal-to-noise ratio (SNR) values and measures the probability of unsuccessful block decoding. BLER provides a practical measure of communication reliability and demonstrates how channel estimation quality affects receiver performance.


`MLChEstEvaluateHARQ.ipynb`: Evaluates end-to-end communication throughput using Hybrid Automatic Repeat reQuest (HARQ). The notebook compares multiple channel estimation methods while accounting for retransmissions and measures the resulting throughput across different SNR values. Throughput reflects the combined effects of channel estimation accuracy, decoding performance, and HARQ retransmissions, making it a useful system-level performance metric.

---

**Core Modules**

`ChEstNet.py`: Defines:

- The dataset class used by the training and evaluation notebooks
- The deep residual neural network architecture used for multi-layer channel estimation


`ChEstUtils.py`: Provides utility functions shared across the notebooks, including:

- Channel estimation helpers
- Evaluation utilities
- Plotting and analysis functions

---

**Notes**

- All experiments were developed and tested using **Python** and **PyTorch**.
- Dataset generation is the most storage-intensive step and requires approximately **30 GB** of free disk space.
- Running the BLER and HARQ evaluations may require a significant amount of computation time, depending on the number of simulated transmissions and SNR points.

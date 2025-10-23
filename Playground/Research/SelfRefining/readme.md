# A Self-Refining Multi-Layer Receiver Pipeline

This directory contains Python scripts and Jupyter notebooks that implement the experiments described in the research paper:

> **“A Self-Refining Multi-Layer Receiver Pipeline”**  
> Presented at the *Asilomar Conference on Signals, Systems, and Computers.*

---

## 📁 Contents

### **Data Generation**

**`MLChEstDataGen.ipynb`**  
Generates the training, validation, and test datasets required for the experiments.  
> ⚠️ The generated datasets require approximately **30 GB** of disk space.

---

### **Model Training**

**`MLChEstTrain.ipynb`**  
Trains the deep neural network for multi-layer channel estimation.  
An **already-trained model** is provided in the `Models` directory.  
A comprehensive hyperparameter search was performed to obtain this model. You may use this notebook to train your own model or proceed directly to evaluation using the included model.

---

### **Model Evaluation**

**`MLChEstEvaluateNMSE.ipynb`**  
Evaluates the trained model on the test dataset using **Normalized Mean Squared Error (NMSE)** metrics.

**`MLChEstEvaluateBLER.ipynb`**  
Evaluates the trained model within an **end-to-end communication pipeline**, comparing different channel estimation methods at various **SNR levels** using **Block Error Rate (BLER)**.

**`MLChEstEvaluateHARQ.ipynb`**  
Assesses **communication throughput** in an end-to-end pipeline with **Hybrid Automatic Repeat reQuest (HARQ)**.  
Runs multiple channel estimation methods across different SNR values and reports throughput performance.

---

### **Core Modules**

**`ChEstNet.py`**  
Defines the dataset class and the **PyTorch** implementation of the **deep residual neural network** used for multi-layer channel estimation.

**`ChEstUtils.py`**  
Provides **utility functions** shared across the other scripts and notebooks.

---

### Notes

- All experiments were developed and tested using **Python** and **PyTorch**.  
- The results presented in the paper are based on the trained model provided in the `Models` directory. You can reproduce these results by ensuring that the random seeds remain unchanged
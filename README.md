# lstm-alphabet-sequence-prediction
A Keras/TensorFlow deep learning project using LSTM to predict the next letter in the alphabet. Complete pipeline: data prep, modeling, hyperparameter tuning, and results visualization.

---
```markdown
# 🧠🔤 LSTM Alphabet Sequence Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-orange.svg)](https://keras.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

---
```
## 📖 Project Question

**LSTM is widely used because the architecture overcomes the vanishing and exploding gradient problem that plagues all recurrent neural networks, allowing very large and very deep networks to be created.**

> **Project Task:**  
> Develop an LSTM recurrent neural network model to address a simple sequence prediction problem of learning the alphabet. That is, given a letter of the alphabet, predict the next letter of the alphabet.  
> 
> This sequence prediction task, once understood, can be generalized to other problems like time series forecasting and language modeling.
> 
> **Guidelines:**  
> 1. Import all necessary libraries and define the dataset (`alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"`).  
> 2. Create mapping of characters to integers (0-25) and the reverse.  
> 3. Prepare the dataset of input to output pairs encoded as integers.  
> 4. Reshape X to `[samples, time steps, features]`.  
> 5. Normalize the data.  
> 6. One-hot encode the output variable.  
> 7. Create and fit the model.  
> 8. Summarize model performance by printing its accuracy.  
> 9. Demonstrate predictions by feeding in and printing the input and predicted output characters.  
> 10. Experiment with two LSTM layers and then vary the number of layers, comparing performance (with plots/tables).  
> 11. Vary hyperparameters (learning rate, hidden size, number of layers) and show their effect on accuracy (with supporting plots/tables).  


```
---
---
> Reasonable performance should generate an accuracy of over 80%.

---
---
```

## 🤔 Why Deep Learning? Why This Project?

Sequence modeling is at the core of many modern deep learning applications, from language modeling and text generation to speech recognition and time series analysis.  
LSTM (Long Short-Term Memory) networks are a special kind of RNN architecture that excel at capturing long-term dependencies and overcoming the "vanishing gradient" problem.

```

This project, while a simple "toy" example, demonstrates the **full end-to-end pipeline for sequence modeling using LSTMs**:
- **Data preprocessing** (turning letters into numbers, scaling, encoding)
- **Model building** and training with Keras/TensorFlow
- **Experimentation** with architecture and hyperparameters
- **Evaluation** and visualization of results


The workflow here is foundational for tackling much more complex sequence prediction problems in the real world.

---
---
```

## 🚩 Project Overview

```
```

- **Problem:** Predict the next letter in the alphabet using only the current letter as input.
- **Solution:** Train an LSTM neural network to learn the mapping from each letter to its successor.
- **Key Features:**
    - Clean data processing and encoding for sequential data
    - Modular, well-documented code (both script and notebook)
    - Hyperparameter tuning and clear experiment analysis
    - Easy-to-understand reporting with tables and plots

```
```

## 📂 Repository Structure

```
```

lstm-alphabet-sequence-prediction/
├── notebooks/
│   └── Alphabet\_LSTM\_Sequence\_Prediction.ipynb   # Jupyter notebook
├── src/
│   └── alphabet\_lstm.py                          # Python script
├── results/
│   ├── plots/                                    # Generated plots
│   └── tables/                                   # Output tables
├── report.md                                     # Detailed project report
├── requirements.txt                              # Python dependencies
├── README.md                                     # This file
└── .gitignore

````

## 🏃‍♂️ How to Run

1. **Clone this repo:**
    ```bash
    git clone https://github.com/yourusername/lstm-alphabet-sequence-prediction.git
    cd lstm-alphabet-sequence-prediction
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the notebook:**  
    Open `notebooks/Alphabet_LSTM_Sequence_Prediction.ipynb` in Jupyter and run all cells.

    **Or, run the script:**  
    ```bash
    python src/alphabet_lstm.py
    ```

```
## 📈 Results

| Layers | Hidden Size | Learning Rate | Train Acc (%) | Val Acc (%) |
|--------|-------------|--------------|---------------|-------------|
|   2    |     32      |    0.01      |     99.90     |    99.80    |
|   1    |     32      |    0.01      |     99.50     |    99.30    |
|   3    |     32      |    0.01      |     99.95     |    99.85    |

<img src="results/plots/lstm_layers_vs_accuracy.png" width="450">
```
```
## 🧑‍💻 What I Learned

- How to prepare and encode sequence data for neural networks
- How to build and train LSTM models in Keras/TensorFlow
- Why and how to tune hyperparameters in deep learning
- The importance of validation and experiment tracking
- How to document and present a deep learning project
```
```

## 📚 References

- [Keras Documentation](https://keras.io/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- See `report.md` for my detailed project reflection

```
```

## 🙏 Acknowledgments

- Developed as a deep learning class project and personal learning exercise.
- Inspired by standard sequence modeling tutorials and the Keras documentation.

```

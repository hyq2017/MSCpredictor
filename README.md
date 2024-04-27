## Overview
The multi-scale collaborative (MSC) AI model is developed for accelerating protein evolution and engineering, enabling high-throughput function prediction and directed design with high accuracy, reliability and universality compared to state-of-the-art approaches. The proposed MCS AI approach enables low-cost and high-throughput directed protein functional improvement (activity, stability, substrate specificity, et. al.), and has great potential in many fields such as biocatalysis and synthetic biology. Here, we provide a demo for predicting enzyme activity from the case study, with five professional models (PMs) trained from enzymatic activity database including Schizochytrium-sourced Acyltransferase.

## Python Dependencies
pandas
numpy
gensim
joblib
sklearn

## Demo
Running (several seconds):

```
python main.py
```

Check the output of the test.xlsx file in result directory.

## Using MSC on your data
Following the instruction of MSC framework in the manuscript, re-train new GM and PM from your data with feature extracted from the provided SVD model.

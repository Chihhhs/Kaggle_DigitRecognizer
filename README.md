# Kaggle_Mnist_Recognizer

[Kaggle Recognizer Competition.](https://www.kaggle.com/competitions/digit-recognizer)

+ Install package.

```bash

pip install -r requirements.txt
```

+ Download dataset

    [Train data](https://www.kaggle.com/competitions/digit-recognizer/data?select=train.csv)

    [Test csv](https://www.kaggle.com/competitions/digit-recognizer/data?select=test.csv)

    [sample_submission.csv](https://www.kaggle.com/competitions/digit-recognizer/data?select=sample_submission.csv)

## Training

`Train.py` : Train and Val.

```python

from sklearn.model_selection import train_test_split
import numpy as np
Total_data = np.loadtxt(FILE,delimiter=",",dtype=np.float32 ,skiprows=1)
train_data, test_data = train_test_split(Total_data, test_size=0.2, random_state=42)

```

Split **train.csv** to train_data (0.8) and test_data (0.2).

Create **CusDataset** class , add **DataLoader** to set dataset and load data.

## Write ans

Evaluate model with **test.csv** and write prediction to **sample_submission.csv**.

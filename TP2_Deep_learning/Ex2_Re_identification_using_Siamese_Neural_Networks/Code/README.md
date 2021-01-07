# REID CODE

## Dependencies

- Python >= 3.5
- PyTorch >= 0.4.0
- torchvision
- scipy
- numpy
- scikit_learn



## Current Result

| Re-Ranking| backbone |  mAP | rank1 | rank3 | rank5 | rank10 |  
| :------: | :------: |  :------: | :------: | :------: | :------: |  :------: |   
| yes | resnet50 |  88.10 | 90.59  | 94.33 | 95.28 | 96.59 |
| no | resnet50 |  72.87 | 88.03 | 93.82 | 95.58| 97.21 |




## Data

The data structure would look like:
```
data/
    bounding_box_train/
    bounding_box_test/
    query/
```
#### Market1501 
unzip it from the data folder

#### to launch the code please write
```
python main.py --mode train --data_path <path/to/Market-1501-v15.09.15>
python main.py --mode train --data_path ./data/Market-1501-v15.09.15
```






# Bitcoin-Ransomware-Classifier

### Motivation to solve the Problem
We know that Bitcoin is one of the most popular cryptocurrency and might play a crucial role in WEB 3.0 and free and open currency system.  
However it is often used by scammers and hackers to demand ransom from an individual or a corporate or even governments in exchange for unlocking their system from a ransomware. It is often important to know the type of ransomware used so appropriate counter measures can be taken.
Scammers often give us a bitcoin address due to its untracable nature. Our ML Model aims to `predict the ransomware given the Bitcoin Address` and other parameters. If the given bitcoin address in our training data is not related to any known ransomware we write ***("white")*** as the label there.
### Dataset and Publications Used
- The data is obtained from the following   [`dataset`](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset).
- This [ `paper`](https://appliednetsci.springeropen.com/articles/10.1007/s41109-020-00261-7) helped me understand the above dataset as the above dataset is based on this research paper.

### Goals of the Project
This clearly is a `multi class classification problem` where we need to classify a Bitcoin Address along with other parameters to `different classes of Ransomware` if known.
The following steps need to be perfomed :  
1. Data Preprocessing. We split the data to 70:15:15 for training, testing and validation sets.
2. We run `Decision trees` with different ***(depths)*** from 4 to 20 along with GINI Gain and Entropy criteria. We check which of the depth and criteria combination gives the best accuracy on the testing set.
3. We do ensembling by training weak decision tree classifiers with depth=3 on 50% data. We take 100 such classifiers and then take the majority vote. This actually forms our `random forest`
4. We now using Adaboost boosting technique to try to improve performance of our chosen Decision Tree in Part 2 for different values of n estimators from 4 to 20.

### File Structure  
- `BitcoinHeistClassifier.ipynb` -> Jupyter Notebook with ML Model  
- `BitcoinHeistData.csv` -> Dataset
## Methodology and Results Obtained.


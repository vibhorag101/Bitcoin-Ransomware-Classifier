# Bitcoin-Ransomware-Classifier

### Motivation to solve the Problem
We know that Bitcoin is one of the most popular cryptocurrency and might play a crucial role in WEB 3.0 and free and open currency system.  
However it is often used by scammers and hackers to demand ransom from an individual or a corporate or even governments in exchange for unlocking their system from a ransomware. It is often important to know the type of ransomware used so appropriate counter measures can be taken.
Scammers often give us a bitcoin address due to its untracable nature. Our ML Model aims to `predict the ransomware given the Bitcoin Address` and other parameters. If the given bitcoin address in our training data is not related to any known ransomware we write ***"white"*** as the label there.
### Dataset and Publications Used
- The data is obtained from the following   [`dataset`](https://archive.ics.uci.edu/ml/datasets/BitcoinHeistRansomwareAddressDataset). It only considers the transactions above 0.3BTC as transactions less than this are usually not ransomware transactions.
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

## Part 1.
- We first convert the ***label*** and ***address*** column of dataset to numerical values using `LabelEncoder`.
- Also we drop the label column from the dataframe and store as xTrain and label in yTrain.
- We first do a train test validation set split using our custom function as shown below
```
df['label'] = LE.fit_transform(df['label'])
df['address'] = LE.fit_transform(df['address'])

df1 = df.drop(['label'], axis=1)
labelTemp = df['label']

trainX = df1[:int(len(df1)*0.7)]
trainY = labelTemp[:int(len(labelTemp)*0.7)]

valX = df1[int(len(df1)*0.7):int(len(df1)*0.85)]
valY = labelTemp[int(len(labelTemp)*0.7):int(len(labelTemp)*0.85)]

testX = df1[int(len(df1)*0.85):]
testY = labelTemp[int(len(labelTemp)*0.85):]

```

## Part 2
- Here we run Decision Trees for heights as (4,8,10,15,20) and **Gini Gain** and **Entropy** as the criteria for building decision tree.

#### ***The results we obtain are as follows*** :
- As we can observe the Decision Tree with **depth=15** and **criteria = Entropy** gives the best accuracy we choose it for further analysis.  
```
Accuracy for max height of tree =  4  is  0.9858378761385584  for gini
Accuracy for max height of tree =  4  is  0.9859087324716289  for gini on validation set
Accuracy for max height of tree =  8  is  0.9863567273516874  for gini
Accuracy for max height of tree =  8  is  0.9864161552439401  for gini on validation set
Accuracy for max height of tree =  10  is  0.9870492908652473  for gini
Accuracy for max height of tree =  10  is  0.9871864321550611  for gini on validation set
Accuracy for max height of tree =  15  is  0.9883612758711329  for gini
Accuracy for max height of tree =  15  is  0.9884321322042033  for gini on validation set
Accuracy for max height of tree =  20  is  0.988009279893944  for gini
Accuracy for max height of tree =  20  is  0.9878287105290225  for gini on validation set
Accuracy for max height of tree =  4  is  0.9857921623752871  for entropy
Accuracy for max height of tree =  4  is  0.985840161826722  for entropy on validation set
Accuracy for max height of tree =  8  is  0.9862287288145278  for entropy
Accuracy for max height of tree =  8  is  0.9862790139541262  for entropy on validation set
Accuracy for max height of tree =  10  is  0.9873761442726369  for entropy
Accuracy for max height of tree =  10  is  0.9874561433583616  for entropy on validation set
Accuracy for max height of tree =  15  is  0.9888961269014068  for entropy
Accuracy for max height of tree =  15  is  0.9887544142352659  for entropy on validation set
Accuracy for max height of tree =  20  is  0.9880024228294534  for entropy
Accuracy for max height of tree =  20  is  0.9879589947543457  for entropy on validation set
```


## Part 3
- We now perform ensembling by the following function by taking 50% of the data for training weak classifiers
- Now we make 100 weak classifiers as decison tree with max_depth= 3. We train them on an ensemble data. We take majority vote against them to get the final predicted value for a given input features or data.
- This is also called `random forest`.
#### ***The results we obtain are as follows*** :
 ```
Accuracy for ensemble is  0.9857921623752871
Accuracy for ensemble on validation set is  0.985840161826722
```
- We observe that the accuracy remains almost the same for decision tree and random forest.
## Part 4
- We now use ***Adaboost Classifier*** and ***Random Forest Classifier*** for improving the performance of our model with base estimator as decision tree of `depth=15` and `criteria = entropy`.
- We find very little improvements in the model accuracy.
## What's Next ❓
- We can try to implement the various other ML models like SVM etc
- We can read the attached research paper more thoroughly to understand the problem much more effectively and present even better solutions.

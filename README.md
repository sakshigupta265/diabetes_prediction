# Diabetes Prediction

## Prompt
Diabetes is a disease that occurs when your blood glucose, also called blood sugar, is too high. Blood glucose is your main source of energy and comes from the food you eat. Insulin, a hormone made by the pancreas, helps glucose from food get into your cells to be used for energy.
The disease is depended on other health factors like glucose level, blood pressure etc. The aim of this project is to predict the possibility of having diabetes (presently or in near future) by analysing the statistics of the other health factors.

## Solution
I have used a Machine Learning Model called 'KNN' (k-Nearest Neighbours) for predicting if a person has diabetes or not.
The steps involved in reaching the final results are:
 - Reading the dataset
 - Extracting the useful information 
 - Cleaning the dataset
 - Understanding the interfence of each factor
 - Dividing the dataset into train and test sets
 - Creating the algorithm for prediction
 - Making test predictions
 - Calculate accuracy of our Model
 
I have also made prediction using the model provided by [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), to compare the end results of both the models.

## Algorithm
Let's see what is KNN Algorithm

### Overview 
KNN is a supervised machine learning algorithm, which relies on labeled input data to learn a function that produces an appropriate output when given new unlabeled data. 

The KNN algorithm assumes that similar things exist in close proximity. In other words, similar things are near to each other.

### Working
So if we have a dataset of cells which have categories as: Plant Cell and Animal Cell and we have a new unlabeled cell. Our task is to find out that our 'new cell' belongs to which category.

![image](https://user-images.githubusercontent.com/54631569/111068870-80a13c80-84f0-11eb-852a-b464d46e3bb1.png)

Then decide upon the value of 'K' for now lets take it to be 5, so we will calculate the distance of the 5 most nearest cells (the most common method is the Euclidean Distance). And simply pick the category with the most votes. Here the "new cell" will belong to the Animal Cell Category

 ![image](https://user-images.githubusercontent.com/54631569/111069514-6157de80-84f3-11eb-8d95-6dfaaa3843a9.png)
 
 ### Steps for Implementing Algorithm
1. Load the data
Initialize K to your chosen number of neighbors
2. For each example in the data

    2.1 Calculate the distance between the query example and the current example from the data.
    
    2.2 Add the distance and the index of the example to an ordered collection
3. Sort the ordered collection of distances and indices from smallest to largest (in ascending order) by the distances
4. Pick the first K entries from the sorted collection
5. Get the labels of the selected K entries
6. If regression, return the mean of the K labels
7. If classification, return the mode of the K labels
 
### Snipet
![image](https://user-images.githubusercontent.com/54631569/111070453-d6c5ae00-84f7-11eb-92ca-7c91b4a5548a.png)

### Choosing the right value for K
To select the K that’s right for your data, we run the KNN algorithm several times with different values of K and choose the K that reduces the number of errors we encounter while maintaining the algorithm’s ability to accurately make predictions when it’s given data it hasn’t seen before.

### Advantages
- The algorithm is simple and easy to implement.
- There’s no need to build a model, tune several parameters, or make additional assumptions.
- The algorithm is versatile. It can be used for classification, regression, and search (as we will see in the next section).

### Disadvantages
- The algorithm gets significantly slower as the number of examples and/or predictors/independent variables increase.


## Data Analysis

### Dataframe
![image](https://user-images.githubusercontent.com/54631569/111070711-e396d180-84f8-11eb-8331-f6e927ac061d.png)


### Binary Histogram for categorization
![image](https://user-images.githubusercontent.com/54631569/111070686-c82bc680-84f8-11eb-9f0f-202243f430b7.png)

### Dependency of each factor on outcome

- with Nan values

![image](https://user-images.githubusercontent.com/54631569/111070834-894a4080-84f9-11eb-80c7-6d7459219ec0.png)

![image](https://user-images.githubusercontent.com/54631569/111070871-b0a10d80-84f9-11eb-9f31-ff86fe1d9f26.png)

![image](https://user-images.githubusercontent.com/54631569/111070889-c0205680-84f9-11eb-9396-74ebee2deac7.png)

- without Nan values

![image](https://user-images.githubusercontent.com/54631569/111070932-e9d97d80-84f9-11eb-9306-66137bf34b2d.png)

![image](https://user-images.githubusercontent.com/54631569/111070943-f2ca4f00-84f9-11eb-8a51-a4ae111ee6ca.png)

![image](https://user-images.githubusercontent.com/54631569/111070952-fcec4d80-84f9-11eb-9941-cf88e7192a7a.png)

### Pair plots
Pair Plots are a really simple (one-line-of-code simple!) way to visualize relationships between each variable. It produces a matrix of relationships between each variable in your data for an instant examination of our data. It can also be a great jumping off point for determining types of regression analysis to use.

![image](https://user-images.githubusercontent.com/54631569/111071042-5fdde480-84fa-11eb-900e-515a5174e5af.png)

### Heatmaps
A heatmap is a graphical representation of data in two-dimension, using colors to demonstrate different factors. Heatmaps are a helpful visual aid for a viewer, enabling the quick dissemination of statistical or data-driven information.

![image](https://user-images.githubusercontent.com/54631569/111071130-ba774080-84fa-11eb-907c-3be32187a1b6.png)

### Visualising Results

![image](https://user-images.githubusercontent.com/54631569/111071162-dda1f000-84fa-11eb-8ffe-656211478bd3.png)


### Confusion Matrix

![image](https://user-images.githubusercontent.com/54631569/111071203-0924da80-84fb-11eb-9e9e-0e0d53ad9e41.png)

## Directory Structure

**Dateset:** [diabetes_dataset.csv](https://github.com/sakshigupta265/diabetes_prediction/blob/main/diabetes_dataset.csv)

**Source Code:** [diabetes_prediction.ipynb](https://github.com/sakshigupta265/diabetes_prediction/blob/main/diabetes_prediction.ipynb)

**Results:** [result.csv](https://github.com/sakshigupta265/diabetes_prediction/blob/main/result.csv)

**Readme File:** [README.md](https://github.com/sakshigupta265/diabetes_prediction/blob/main/README.md)

**Contribution File** [CONTRIBUTION.md](https://github.com/sakshigupta265/diabetes_prediction/blob/main/CONTRIBUTION.md)

## Testing
To test this project on your local computer follow the given steps:
 
 ` 1. fork this repository `
 
 ` 2. clone it `
 
 `3. make sure you have all the Prerequisites mentioned below`
 
 `4. run the` [diabetes_prediction.ipynb](https://github.com/sakshigupta265/diabetes_prediction/blob/main/diabetes_prediction.ipynb) `file `

## Prerequisites
Make sure you have the latest version of python3, if not you can easily download it from [here](https://www.python.org/downloads/).

Make sure to update `pip` to latest version using `'python -m pip install –upgrade pip` .

The project uses a few python libraries, so make sure you have them too:

`numpy`:  download it using this [documentation](https://pypi.org/project/numpy/).

`pandas`: download it using this [documentation](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html).

`matplotlib`: download it using this [documentation](https://pypi.org/project/matplotlib/).

`scikit-learn`: download it using this [documentation](http://scikit-learn.org/stable/install.html).

`seaborn`: download it using this [documentation](https://seaborn.pydata.org/installing.html).

## Conclusion
The KNN algorithm which we used had an accuracy of 73.37%
The KNN algoritm by sklearn had an accuracy of 75.32%

For making the KNN algorithm more accurate we can play-around with the value of 'K'.

If you do not wish to use kNN we can always go for more accurate Machine Learning Models such as Vector Quantization, Naive Bayes, Support Vactor Machines, etc. I will surely try to solve this problem using different algorithms to show the difference.

## References
If you are curious about kNN algorithms, you can learn more from [StatQuest](https://www.youtube.com/watch?v=HVXime0nQeI)

## Want to contribute?
I would love to recieve your contributions towards this project. Refer to [CONTRIBUTION.md](https://github.com/sakshigupta265/diabetes_prediction/blob/main/CONTRIBUTION.md) for more details.

## Thanks! ✨


# Boston-Housing-Linear-Regression-

## Summary

In this project, we first did EDA on the dataset and did initial visual analysis on the variables including both univariate and bivariate analysis. We then checked for the assumptions of linear regression to make sure our data is fit for model building. We then ran our linear regression model on the data and observed the results, after which we dropped variables that were not significant and examined the changes in the R-Squared variable. We then finally obtained our regression equation which is displayed at the bottom of this page. 


## Context

Our task is to predict the housing prices of a town or a suburb based on the features of the locality provided to us. Meanwhile, we also want to identify the most important features in the dataset. To achieve this, we apply different techniques of preprocessing and then we build a linear regression model to predict the prices for us. 

## Technologies Used

* Numpy
* Pandas
* Matplotlib
* Seaborn
* Statsmodels
* ProbPlot (from statsmodels)
* ols 
* Sklearn
* LinearRegression

## The Data

This Data was obtained from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. The detailed attribute information is provided below: 

* **CRIM**: per capita crime rate by town
* **ZN**: proportion of residential land zoned for lots over 25,000 sq.ft.
* **INDUS**: proportion of non-retail business acres per town
* **CHAS**: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
* **NOX**: nitric oxides concentration (parts per 10 million)
* **RM**: average number of rooms per dwelling
* **AGE**: proportion of owner-occupied units built before 1940
* **DIS**: weighted distances to five Boston employment centers
* **RAD**: index of accessibility to radial highways
* **TAX**: full-value property-tax rate per 10,000 dollars
* **PTRATIO**: pupil-teacher ratio by town
* **LSTAT**: %lower status of the population
* **MEDV**: Median value of owner-occupied homes in 1000 dollars


## Exploratory Data Analysis

It is important for us to assess the distributions of our variables so in this section we undertook univariate and bivariate analysis to see get an overview of the distributions of different variables as well as any potential correlations between them. 

![Screen Shot 2021-11-29 at 11 41 17 AM](https://user-images.githubusercontent.com/88220704/143916722-350f2310-98c0-4c4f-8c49-115558cd79fc.png)

For example, the above graph shows the distribution of MEDV which is our dependent variable and what we are trying to predict. Since it is slightly skewed to the right, we decided it would be a good idea to try applying a log transformation to see if we can get an approximately normal distribution. 

![Screen Shot 2021-11-29 at 11 43 09 AM](https://user-images.githubusercontent.com/88220704/143916979-20bd8e3d-c4b4-49a4-a1be-dda849e076b2.png)

Our transformation gave us an approximately normal distribution without skew. As mentioned before, we also conducted bivariate analysis and we initially looked at a heatmap to see if anything stands out. 

![Screen Shot 2021-11-29 at 11 45 05 AM](https://user-images.githubusercontent.com/88220704/143917243-bb5fa390-74a3-4873-bcd6-7a87dcc94288.png)


This heatmap showed us various interesting insights, such as how us there seems to be a strong correlation between nitric oxide concentration and the proportion of non-retail business acres per town. 

To get a closer look, we visualized the relationship between the pairs of features that have significant correlations. For instance, the below graph shows how the price of the houses increases as the value of RM(rooms) increases, which is expected given that the price of houses is generally higher for those with more rooms than others. 

![Screen Shot 2021-11-29 at 11 50 16 AM](https://user-images.githubusercontent.com/88220704/143917982-a6d9fdaf-bc85-459e-b942-299c9b079d47.png)


## Checking for assumptions

Before initializing linear regression, we need to make sure that our data meets the assumptions for linear regression. They are the following: 

1. Checking for multicollinearity 
2. Mean of residuals should be 0
3. No Heteroscedasticity
4. Linearity of variables
5. Normality of error terms

### Checking for multicollinearity

Multicollinearity occurs when the independent variables are too highly correlated with each other. To check for this, we used the Variance Inflation Factor (VIF) to cehck if there is multicollinearity in the data. Features having a VIF score > 5 were dropped/treated until all the features have VIF score < 5. 

![Screen Shot 2021-11-29 at 12 05 55 PM](https://user-images.githubusercontent.com/88220704/143920032-c2408964-8646-4c13-9596-ecda9779b969.png)

After creating a function to give us the above values, we can see that both RAD and TAX have a high VIF. After dropping the TAX column from the training data, we can see below that the multicollinearity was removed.

![Screen Shot 2021-11-29 at 12 07 38 PM](https://user-images.githubusercontent.com/88220704/143920227-1cda0d7a-8279-41b3-a08d-fbb6126f0613.png)

### Mean of residuals should be 0. 

the mean of our residuals was -3.7647 which is very close to 0, thus, the assumption for residuals is satisfied. 

### No Heteroscedasticity 

If the residuals are not symmetrically distributed across the regression line, then the data is said to be heteroscedastic. To check for this, we used the **Goldfeldquandt Test**. This test showed us that the assumptoin for no heteroscedasticity is satisfied since it gave us a p value greater than 0.05 which is our alpha value. 

### Linearity of Variables

This assumption states that the predictor variables must have a linear relation with the dependent variable. We checked this assumption by plotting the residuals and fitting the values on on a plot to ensure that the residuals do not form a strong pattern. 

![Screen Shot 2021-11-29 at 12 16 09 PM](https://user-images.githubusercontent.com/88220704/143921304-439c85ad-b5a9-46a0-a1ef-163482f48799.png)

We can see no clear pattern, so therefore the linearity assumption is satisfied. 

### Normality of error terms

the residuals should be normally distributed, and as we can see in the histogram and q-q plot below, they clearly are. 

![Screen Shot 2021-11-29 at 12 17 50 PM](https://user-images.githubusercontent.com/88220704/143921512-ce46a0f5-4f14-4041-9911-1ae6871011f2.png)

![Screen Shot 2021-11-29 at 12 18 03 PM](https://user-images.githubusercontent.com/88220704/143921536-ab2daa77-f7a1-442b-a11b-37c42a2fb379.png)

## The Linear Regression Model

### Initial model 
After checking for the assumptions, we created our model and used three different performance metrics: 

**RMSE**: The Root Mean Squared Error 
**MAE**: The Mean Absolute Error
**MAPE**: The Mean Absolute Percentage Error. 

Below were the results of our model after splitting our data set for testing and training, fitting the model on the training data, and then testing its performance on the test data. 
![Screen Shot 2021-11-29 at 12 29 35 PM](https://user-images.githubusercontent.com/88220704/143922957-1c1274d3-e020-4a70-8932-aaa76b1069e9.png)



![Screen Shot 2021-11-29 at 12 24 08 PM](https://user-images.githubusercontent.com/88220704/143922283-3f50b58e-0aa9-4038-b2a9-fdb00cf2c251.png)


### Getting the model coefficients: 

![Screen Shot 2021-11-29 at 12 33 06 PM](https://user-images.githubusercontent.com/88220704/143923422-68187382-1227-405a-b2c8-04c72866a3a3.png)

After obtaining our coefficients, we could now write our regression equation which is: 

## log (Price) =	( 4.649385823266645 ) *  const + ( -0.01250045507910428 ) *  CRIM + ( 0.11977319077019619 ) *  CHAS + ( -1.0562253516683233 ) *  NOX + ( 0.05890657510927927 ) *  RM + ( -0.04406889079940582 ) *  DIS + ( 0.007848474606243973 ) *  RAD + ( -0.04850362079499883 ) *  PTRATIO + ( -0.029277040479797102 ) *  LSTAT




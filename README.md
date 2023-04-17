# House Prices Prediction With NLP Model

This project is inspired by a Kaggle challenge, [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/). The goal of this project is to use machine learning methods to predict house prices in Ames, Iowa based on a dataset of 79 explanatory variables describing many aspects of residential homes. 

First of all, we did some exploratory data analysis to examine the relationships between the target variable, SalePrice, and housing related features.

![image](https://user-images.githubusercontent.com/25331292/232405547-6b17cbb4-9186-4f20-b000-2c3fcb5928a8.png)

The box plot above shows that sale price increases as the score of overall material and finish of the house increases. This is reasonable as we usually believe prices are positively associated with the quality of goods.

![image](https://user-images.githubusercontent.com/25331292/232408187-b81d8832-64f3-47db-9f04-fae7088aa9a0.png)

We calculated the correlation coefficents between variables, and created a pairplot of SalePrice and its most correlated variables. The plot indicates the strong correlation between SalePrice between OverallQual(Overall material and finish quality), GrLivArea(Above ground living area square feet), GarageCars(Size of garage in car capacity), and TotalBsmtSF(Total square feet of basement area). In addition, the scatter plot of 'TotalBsmtSF' and 'GrLiveArea' seems interesting. Almost all the data points are contrained in a border where 'GrLiveArea' is greater or equal to 'TotalBsmtSF'. It makes sense since basement areas is not expected to be bigger than the the living area above.

Natural Language Processing is a subfield of machine learning to analyze natural language data and understand the meaning behind it. We used the Transformer model from the torch module in Python to predict the housing price, since each observation can be concatenated and treated as a sentence for the model to run through. We achieved a loss score of 0.18 after 36 iterations with a learning rate of 0.01.

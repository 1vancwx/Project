# Capstone - Cross-selling products and sales prediction

## https://nbviewer.org/github/1vancwx/Project/blob/main/Capstone/Med.ipynb
## https://nbviewer.org/github/1vancwx/Project/blob/main/Capstone/Modeling.ipynb
Please click on above links to look at the codings for this project.

### Problem Statement

A Medical Multi-National Company based in Singapore is seeking opportunities to increase the market share across South-east Asia. 
However, due to commercial confidentiality, we are unable to disclose further information. Thus, we are only able to obtain Customer Id, Country, Region, Division, Material information, Date of transactions, Invoiced Value. With that, i will have to recommend and illustrate what/how to increase the market shares based on the analysis from the data given.

### Executive Summary
For this project, i have worked on focusing on existing customers and recommending what other similar product which existing customer can actually purchased based on what other customers have purchased before. In the meanwhile, recommending quantity of the recommended product will be recommended as well to the existing customer based on their past transaction's buying power. Thereafter, we will be forecasting the sales trend together with the recommended quantity. Having said that, throughout this project we will be focusing on the quantity rather than the invoice value as monetary stratgy will be out of scope.

With that, i have splitted the project into 3 parts: -

**1) Cross-selling products**
I have used Mixed Basket analysis (MBA) together with collaborative filtering (CF) models to compute what are the similiarity between each two combination of materials. Each models will generate a list of recommended materials to purchase with, which i call it an opportunity cross-selling product. The final list of opportunity cross-selling product will be generated after both MBA and CF have interception and recommending the same combination of products.

**2) Opportunity cross-selling quantity**
After having the list of opportunity cross-selling products, i have computed the recommending quantity based on individual customer buying power by using what is the weightage of the customer's actual purchase vs average customers purchase. With that weightage, it will be multiplied by the opportunity cross-selling product's average purchase to get the recommending quantity. Lastly, recommending quantity will by multiplied by the percentage of probability calculated by MBA model to simulate whether the customer will eventually purchase the opportunity cross-selling product.

**3) Sales trend with opportunity cross-selling quantity**
With all the above, last steps will be using ARIMA and Facebook prophet to forecast future sales trend of the existing purchase together with the possible opportunity cross-selling quantity. i have smoothen the sales trend by averaging both model's prediction to achieve better accuracy. On top of that, i will come out with top 10 sales trend for material level. 
**Country/Region level view will be out of scope for this project.**

### Conclusion and Recommendations
With above models, we can see that these models has identified areas/products to focus on to cross-sell to existing customers. Thus, we can use these analysis to encourage the sales team to promote or convince existing customers to catch more sales opportunities available in the market. Furthermore, we can use the forecast to set a KPI for sales team to achieve for the year,


### Limitations
Due to limited information given, i am unable cross validate the relevancy of the recommended combination suggested by the models.
In addition, i am not able to do user-based analysis to identify customer's purchasing behaviors or relevancy of the recommended material to their business trade. Therefore, this project maybe theoretically logical but it might differ from the reality. Has no information on business sales acumen / strategy which may differ from the simulation computed by the models.

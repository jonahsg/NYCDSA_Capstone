# NYCDSA Capstone - Markerr Project
Team: Ben Burkey, Jonah Gerstel, Michael McGuigan, Lena Chretien

## Datasets Used

- ACS Census Data
- Effective Federal Funds Rate
- Gross Domestic Product (GDP)
- Business Information by ZIP Code
- Consumer Price Index (CPI)
- Zillow Rent Index (ZRI) for Multi-Family Homes
- Zillow Observed Rent Index (ZORI)

## Business Approach

Goal: Obtain insight into rent price indices by comparing pre-Covid predictions to Covid projections for multi-family homes in large cities for Markerr, a real estate analytics company focused on providing deep insight into the supply and demand dynamics in the real estate market to improve valuation accuracy and strengthen ROI.

## Wrangling and Cleaning

- Merged various datasets
- Filled NAs with extrapolated data based on ZIP code
- Ensured columns and column names were accurate and workable (ex. no white space, values were correct data type, etc
- Removed data for all ZIP codes with a population of less than 100,000
- Removed duplicates

## Feature Engineering

- Created ratios of variables reduce multi-collinearity (ex. Employed Population/Total Population)
- Removed variables with high VIF or little importance with regard to target variable (ex. Males between the age of 45 and 64 with Associate's Degrees)
- Principal Component Analysis (PCA)
  - Removed 75% of variables while retaining 95% of variability
  - Found that two variables, Total Population and Income Per Capita, explain 37% and 15% of variability, respectively
  - Found that the additional datasets for CPI (inflation) and GDP (Gross Domestic Product) were also important
  - In order to evaluate the effect of PCA, models were run both with and without PCA

## Clustering
- KMeans Clustering techniques were used to cluster various ZIP codes together
  - ZIP codes were clustered based on Rent, Income, and Population
  - Found that 3 clusters was optimal based on both metrics and silhouette charts

## Models Used
These models were tuned hyperparameters with GridSearchCV, and ran both with and without Clustering and PCA:
- Lasso
- Random Forest
- XGBoost

## Results
There were three facets to the modeling work done in this project in order to understand current urban rent prices if COVID had not happened in order to compare those prices to current COVID rent prices.

### Modeling post-2019 ZRI numbers via ZORI

This was done due to the fact that post-2019 ZRI numbers were unavailable for later years (post-COVID), while ZORI numbers were unavailable in earlier years where the initial prediction model was trained and were also not specific to multi-family units, the focus of this project. In order to create this model, the initial training was done on the 2018 dataset, the last year where both ZORI and ZRI were available. Overall, the model performed well with an RMSE of $73.57 an average over estimation of $18.67. The Top 5 Zip codes with the largest under estimations did not have a major pattern between them, however, interestingly enough, all of the Top 5 Zip codes with the largest over estimations were in the LA area.

### Extrapolating ACS data with foreign data prior to COVID in order to model ZRI numbers if COVID had not happened

This was calculated via the optimal modeline technique (Random Forest with PCA and no Clustering).

### Comparing pre-COVID projections to COVID predictions

A percent change metric was created for each ZIP code based on rent prices including COVID and without COVID, while also factoring in the model error for the ZORI/ZRI model. This percent change metric was then used to understand how rental prices in over 700 urban ZIP codes were affected by COVID and what percent change rental prices have deviated from non-COVID 2021 projections.

Results were compared and mapped for visualization purposes. Results were also broken down from extremely high-level to ZIP code by ZIP code:

- On a grand scale, urban multi-family home rental prices decreased by 5.5% due to COVID
- Rental markets in the West saw larger decreases than those in the East
- Regionally (via Federal Housing Finance Agency Regions), areas like the Pacific (California, Washington, and Oregon) and the West North Central (Minnesota, Iowa, Missouri, etc) saw much more sizeable decreases in rent prices (7.2% and 7.6% decreases, respectively). While areas like the East North Central (ex. Michigan, Wisconsin, Ohio, etc) and Middle/South Atlantic (Florida through New York)
- Certain specific findings were that Alexandria/Silver Spring both actually experienced rises in rent prices, while Sacramento and Oakland saw the second and third largest decreases in rent due to COVID (only Frisco, TX saw larger decreases in rent prices)

From all the models run, it was found that:
- PCA improved results on tree based models, but hurt those in linear models
- Clustering improved results on linear models, but hurt those in tree based models

## Next Steps

- Test different clustering methods and groupings (ex. DBScan and Spectral Clustering)
- Include more data for more accurate predictions (ex. include housing prices)
- Incorporate COVID data directly (instead of using ZORI to predict ZRI)
- Look at single family homes and compare those results to multi-family homes

## Link to Presentation
https://docs.google.com/presentation/d/1AsWpkrCV3TIQwhBr6ra0uC-lQNUAXpVMPyqoss32yy0/edit?usp=sharing

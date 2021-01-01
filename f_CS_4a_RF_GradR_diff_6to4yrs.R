#Capstone Project: Predict increase in graduation rate between 6-yr and 4-yr (used ACT comp scores)

library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform
library(AmesHousing)
library(ggplot2)
library(dplyr)

#______________________________________________________________________________________________________

#Read in project data file
df1 <- read.csv("US_Univ_UG_shortnames.csv")

#_______________________________________________________________________________________________________
#Initial data processing

#Create new column showing diff in grad rate between 6 and 4 yrs
df1$Gradrate_diff_6to4yrs <- df1$Gradrate_6yrs - df1$Gradrate_4yrs

#Check number of NAs in each column
sapply(df1, function(x) sum(is.na(x)))
nrow(df1)

#No of rows with NA for ACTComp is less than SAT scores. So choosing ACT scores as driver. 
#Remove all rows where ACTComp_25 and ACTComp_75 is NA
df2 <- df1[!is.na(df1$ACTComp_25)&!is.na(df1$ACTComp_75),]
nrow(df2)    #Remaining rows = 1199

#List names of columns
colnames(df2)

#Cull columns by name
#Removed select columns - SAT scores and some identifier info
df3 <- df2[ , -which(names(df2) %in% c("ID_number", "Name", "Yr", "ZIP", "County", "Longitude",
                                       "Latitude", "SATRead_25", "SATRead_75", "SATMath_25", 
                                       "SATMath_75", "SATWrite_25", "SATWrite_75", "Gradrate_5yrs",
                                       "Gradrate_6yrs", "Gradrate_4yrs"))]

#Label - Difference in graduation rate between 6 and 4 yrs
#Remove all rows where Gradrate_diff_6to4yrs is NA
df3b <- df3[!is.na(df3$Gradrate_diff_6to4yrs),]
nrow(df3b) #has 1195 rows

#Check number of NAs in each column
sapply(df3b, function(x) sum(is.na(x)))

#Remove select columns - with large number of NA values
#Random forest regression requirement - NA values can't be present in any column
df3c <- df3b[ , -which(names(df3b) %in% c("P_1stUG_instate", "P_1stUG_outstate", "P_1stUG_foreign", 
                                          "P_1stUG_resNA"))]

#Remove rows with NA values in any column
df4 <- df3c[complete.cases(df3c), ]
nrow(df4) #Rows without NA values = 1151

#Check if any column has NA values
sapply(df4, function(x) sum(is.na(x)))

#__________________________________________________________________________________________________________
#Split data into Training and Testing sets 

# Create training (70%) and test (30%) sets 
# Use set.seed for reproducibility
set.seed(123)
df4_split <- initial_split(df4, prop = .7)
df5_train <- training(df4_split)
df5_test  <- testing(df4_split)
nrow(df5_train)  #has 806 rows
nrow(df5_test)   #has 345 rows

#_________________________________________________________________________________________________
#Random Forest (RF) regressions:

#----------------------------------------------------------------------------------------------
#Run 1: Initial RF run with default values 

# for reproduciblity
set.seed(123)

# default Random Forest model
m1 <- randomForest(
  formula = Gradrate_diff_6to4yrs ~ .,
  data    = df5_train
)

m1
#Number of trees: 500, No. of variables tried at each split: 16
#default mtry = features/3 = 49/3 = around 16
#mtry: the number of variables to randomly sample as candidates at each split. 
#Mean of squared residuals: 41.83
#% Var explained: 47.42

plot(m1)
#error rate as we average across more trees
#shows that our error rate stabalizes with around 100 trees 
#but continues to decrease slowly until around 300 or so trees.

# number of trees with lowest MSE
#plotted error rate above is based on the OOB (out of box) sample error 
#and can be accessed directly at m1$mse
which.min(m1$mse)
#106 trees result in the lowest error

# RMSE of this optimal random forest model with 106 trees
sqrt(m1$mse[which.min(m1$mse)])
#average error in predicting Gradrate_4yrs = 6.45%
#Random forests are one of the best out-of-the-box machine learning algorithms. 
#They typically perform remarkably well with very little tuning required.

#---------------------------------------------------------------------------------------------------------
#Run 2: Tune mtry (using randomForest::tuneRF) 

#We can still seek improvement by tuning our random forest model.
#there are only a handful of tuning parameters.
#primary parameter is mtry - number of candidate variables to select from at each split
##Additional tuning hyperparameters:
#ntree: number of trees - want enough trees to stabalize the error 
#mtry: the number of variables to randomly sample as candidates at each split. 
#sampsize: the number of samples to train on - default is 63.25% of the training set
#nodesize: minimum number of samples within the terminal nodes. 
#maxnodes: maximum number of terminal nodes.

#Initial tuning with random forest
#Tune the mtry parameter - use randomForest::tuneRF
#Start with mtry = 5, increase by 1.5 until the OOB error stops improving by 1%

# names of features
features <- setdiff(names(df5_train), "Gradrate_diff_6to4yrs")

set.seed(123)

m2 <- tuneRF(
  x          = df5_train[features],
  y          = df5_train$Gradrate_diff_6to4yrs,
  ntreeTry   = 300, #no reduction in error after 300 trees
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)

m2
#Min OOB Error (when improve = 0.01) is at mtry = 4
#NOTE:default mtry = features/3 = 49/3 = around 16 

#------------------------------------------------------------------------------------------------------
# Run 3: Full grid search with ranger()

#Full grid search with ranger
#Perform a larger grid search across several hyperparameters 
#Create a grid and loop through each hyperparameter combination and evaluate the model
#This is where randomForest becomes quite inefficient since it does not scale well. 
#Instead, use ranger - it is 6 times faster than randomForest

#To perform the grid search, first we want to construct our grid of hyperparameters. 
#Search across different models with varying mtry, minimum node size, and sample size.
# hyperparameter grid search
hyper_grid <- expand.grid(
  mtry       = seq(2, 30, by = 4),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

# total number of combinations
nrow(hyper_grid)
#So, will be searching across 128 different models

#Loop through each hyperparameter combination and apply 300 trees 
#(our previous work showed that 300 was plenty to achieve a stable error rate)
for(i in 1:nrow(hyper_grid)) {
  
  # train model
  model <- ranger(
    formula         = Gradrate_diff_6to4yrs ~ ., 
    data            = df5_train, 
    num.trees       = 300,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)
#Top 10 performing models - all have RMSE values around 6.37 
#Results: models with deeper trees (node_size = 3-5 observations in terminal node),
#mtry = 14, and sample size = 0.7 perform best

#___________________________________________________________________________________________________
#Apply best random forest model on test data

#So far, best RF model - retains columnar categorical variables and 
#uses mtry = 14, terminal node_size of 3 observations, and a sample size of 70%.
#Repeat this model to get a better expectation of our error rate.
OOB_RMSE <- vector(mode = "numeric", length = 100)
for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = Gradrate_diff_6to4yrs ~ ., 
    data            = df5_train, 
    num.trees       = 300,
    mtry            = 14,
    min.node.size   = 3,
    sample.fraction = .7,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

#Historgram of range of prediction error (RMSE)
hist(OOB_RMSE, breaks = 10, xlim = c(6.3,6.5)) 
#Result: Expected error ranges between 6.35 - 6.49 
#with a most likely of 6.42

#Range of values for Gradrate_diff_6to4yrs
hist(df4$Gradrate_diff_6to4yrs, breaks = 10)  #most values range from 0 to 40 percent

#Set importance = 'impurity' - allows us to assess variable importance.
#Variable importance measured by decrease in MSE when a variable is used as a node split.
#The remaining error after a node split is known as node impurity 
#A variable that reduces this impurity is considered more important
#Accumulate the reduction in MSE for each variable across all the trees 
#Variable with the greatest accumulated impact is considered important or impactful
View(optimal_ranger)

#Try best random forest model on test data
OOB_RMSE_test <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE_test)) {
  
  optimal_ranger <- ranger(
    formula         = Gradrate_diff_6to4yrs ~ ., 
    data            = df5_test, 
    num.trees       = 300,
    mtry            = 14,
    min.node.size   = 3,
    sample.fraction = .7,
    importance      = 'impurity'
  )
  
  OOB_RMSE_test[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE_test, breaks = 12) #xlim = c(9.6,9.8)
#Result: Expected error ranges between 6.4 and 6.55% 
#with a most likely of 6.8%

#________________________________________________________________________________________________________
# Variable importance plot

#Extracting variable.importance and converting to data.frame
df6 <- as.data.frame(optimal_ranger$variable.importance)

#Creating new column containing row names
df6$names <- rownames(df6)

#Shorten column name optimal_ranger$variable.importance to 'importance'
df6$importance <- df6$'optimal_ranger$variable.importance'

#Bar plot of variable importance - top 20 in descending order
df6 %>% 
  arrange(desc(importance)) %>%
  slice(1:40) %>%
  ggplot(., aes(x=reorder(names, importance), y=importance))+
  geom_bar(stat='identity') + coord_flip()



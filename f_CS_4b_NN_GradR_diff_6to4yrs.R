#Capstone Project: Use ANN to predict the increase in graduation rate between 6 and 4 yrs 
#Used ACT comp scores only
#Only numeric columns modeled - categorical columns NOT included
#ANN - artificial neural network

library(tidyverse)
library(neuralnet)
library(GGally)
library(dplyr)   #For scaling
library(NeuralNetTools)
library(nnet)
library("data.table")

#________________________________________________________________________________________________
#Load data

#Read in project data file
df1 <- read.csv("US_Univ_UG_shortnames.csv")
colnames(df1)
#________________________________________________________________________________________________
#Initial data processing

#Remove all rows where Gradrate_6yrs and Gradrate_4yrs are NAs
#Will create a calculated column that shows the diff in graduation rate between 6 and 4 yrs
#So don't want any NA values in this calculated column
df1b <- df1[!is.na(df1$Gradrate_6yrs)&!is.na(df1$Gradrate_4yrs),]

#Create new column showing diff in grad rate between 6 and 4 yrs
df1b$Gradrate_diff_6to4yrs <- df1b$Gradrate_6yrs - df1b$Gradrate_4yrs

#No. of rows with NA for ACTComp is less than SAT scores. So choosing ACT scores as driver (feature). 
#Remove all rows where ACTComp_25 and ACTComp_75 is NA
df2 <- df1b[!is.na(df1b$ACTComp_25)&!is.na(df1b$ACTComp_75),]

#Cull columns by name
#Removed select columns - SAT scores and some identifier info
df3 <- df2[ , -which(names(df2) %in% c("ID_number", "Name", "Yr", "ZIP", "County", "Longitude",
                                       "Latitude", "SATRead_25", "SATRead_75", "SATMath_25", 
                                       "SATMath_75", "SATWrite_25", "SATWrite_75", "Gradrate_4yrs",
                                       "Gradrate_5yrs", "Gradrate_6yrs"))]

#Check number of NAs in each column
sapply(df3, function(x) sum(is.na(x)))
#Label - difference in graduation rate between 6 and 4 yrs, i.e., Gradrate_diff_6to4yrs
#Gradrate_diff_6to4yrs has no NAs

#Remove select columns - with large number of NA values
#ANN requirement - NA values can't be present in any column
df3b <- df3[ , -which(names(df3) %in% c("P_1stUG_instate", "P_1stUG_outstate", "P_1stUG_foreign", 
                                          "P_1stUG_resNA"))]

#Remove rows with NA values in any column
df4 <- df3b[complete.cases(df3b), ]

#Check if any column has NA values
sapply(df4, function(x) sum(is.na(x)))

#No of cases (rows) without NA values
nrow(df4) #Rows without NA values = 1151

#______________________________________________________________________________________________________
#Remove categorical variables and scale numeric Data

#Before we split dataset to train and test, we scale each numeric feature in [0,1] interval
#The scale01() function maps each data observation onto the [0,1] interval
#Scale numeric columns in a batch

#Start NN analysis with df5
df5 <- df4

#NN model uses only numeric features. So removing all categorical features.
df5 = subset(df4, select = -c(Relgious_y_n, State, Region, Status, HBCU, Urbanization, 
                              Type_of_univ))
colnames(df5)
#Create set of numeric columns
num_cols = c('Applicants', 'Admission', 'UG1st_Enrolled', 'P_submit_SAT', 'P_submit_ACT'
             ,'ACTComp_25', 'ACTComp_75', 'P_admit', 'Yield', 'Tuition2010_11', 'Tuition2011_12'
             ,'Tuition2012_13', 'Tuition2013_14', 'Price_instate_2013_14', 'Price_outstate_2013_14'
             ,'UG_enroll', 'Grad_enroll', 'FTUG_enroll', 'PTUG_enroll', 'P_Amer_Indian'
             ,'P_Asian', 'P_Af_American', 'P_Latino', 'P_PIslander', 'P_White', 'P_2orM_races'
             ,'P_RaceNA', 'P_NRAlien', 'P_Asian_Native_PIslander', 'P_Women' 
             ,'P_1yrUG_any_aid', 'P_1yrUG_Fed_state_grant', 'P_1yrUG_Fed_grant', 'P_1yrUG_Pell_grant'
             ,'P_1yrUG_otherFed_grant', 'P_1yrUG_state_local_grant', 'P_1yrUG_institute_grant'
             ,'P_1yrUG_student_loan', 'P_1yrUG_Fed_student_loan', 'P_1yrUG_other_loan'
             ,'SB_Endowment_per_FTE', 'Gradrate_diff_6to4yrs')

scale01 <- function(x){
  (x - min(x)) / (max(x) - min(x))
}

#Apply scale01 function by calling in the dplyr mutate_all() function.
df5[,num_cols] <- df5[,num_cols] %>% mutate_all(scale01)

#____________________________________________________________________________________________________
#Split into test and train sets

#Provide a seed for reproducible results
#Randomly extract (without replacement) 80% of the observations to build the Training data set.
set.seed(12345)
df5_Train <- sample_frac(tbl = df5, replace = FALSE, size = 0.80)

#Use dplyr??s anti_join() function to extract all the observations that are not within the 
#df5_Train data set to create our test data set i.e., df5_Test
df5_Test <- anti_join(df5, df5_Train)
nrow(df5_Train)  #Training set has 921 cases
nrow(df5_Test)   #Test set has 230 cases

#_________________________________________________________________________________________________
#ANN regressions:

#----------------------------------------------------------------------------------------------
#Run 1: 1st ANN regression - using default hyper parameters

#construct a 1-hidden layer ANN with 1 neuron, the simplest of all neural network
#The df6_NN1 is a list containing all parameters of the regression ANN 
#and the results of the neural network on the test data set 
set.seed(12321)
df6_NN1 <- neuralnet(Gradrate_diff_6to4yrs ~ Applicants + Admission + UG1st_Enrolled
                     + P_submit_SAT + P_submit_ACT + ACTComp_25 + ACTComp_75 + P_admit
                     + Yield + Tuition2010_11 + Tuition2011_12 + Tuition2012_13 + Tuition2013_14
                     + Price_instate_2013_14 + Price_outstate_2013_14 + UG_enroll + Grad_enroll
                     + FTUG_enroll + PTUG_enroll + P_Amer_Indian + P_Asian + P_Af_American
                     + P_Latino + P_PIslander + P_White + P_2orM_races + P_RaceNA + P_NRAlien
                     + P_Asian_Native_PIslander + P_Women + P_1yrUG_any_aid + P_1yrUG_Fed_state_grant
                     + P_1yrUG_Fed_grant + P_1yrUG_Pell_grant + P_1yrUG_otherFed_grant
                     + P_1yrUG_state_local_grant + P_1yrUG_institute_grant + P_1yrUG_student_loan
                     + P_1yrUG_Fed_student_loan + P_1yrUG_other_loan + SB_Endowment_per_FTE, 
                     data = df5_Train)

#To view a diagram of the df6_NN1 use the plot() function.
plot(df6_NN1, rep = 'best')

#Convert df5_Train to data table - to calculate SSE
#install.packages("data.table")           
df5_Train_2 <- setDT(df5_Train)

#Manually compute the SSE (sum of squares error)
#SSE: sum of the squared differences between each observation and its group's mean
NN1_Train_SSE <- sum((df6_NN1$net.result - df5_Train_2[, 'Gradrate_diff_6to4yrs'])^2)/2
paste("SSE: ", round(NN1_Train_SSE, 4))
#Training error = 2.508

#To calculate SSE in test dataset - need to place the label at the last column (to the right) of test data
#Label Gradrate_diff_6to4yrs is placed at the rightmost column of df5_Test 

#To calculate the test error, we first must run our test observations through the df6_NN1 ANN.
#This is accomplished with the neuralnet package compute() function
#1st input of compute() - the desired neural network object created by the neuralnet() function, 
#2nd argument of compute() - the test data set feature (independent variable(s)) values
Test_NN1_Output <- compute(df6_NN1, df5_Test[, 1:41])$net.result
NN1_Test_SSE <- sum((Test_NN1_Output - df5_Test[, 42])^2)/2
NN1_Test_SSE
#The compute() function outputs the response variable, in our case the Gradrate_diff_6to4yrs, 
#as estimated by the neural network. 
#Once we have the ANN estimated response we can compute the NN1_Test_SSE.
#Test error = 0.741 while training error = 2.508  
#So test error is smaller than training error.

#----------------------------------------------------------------------------------------------
#Run 2: 2-Hidden Layers, Layer 1: 4-neurons, Layer 2: 1-neuron, logistic activation function

#Modify regression hyper-parameters in the neuralnet() function 
#To begin, depth is added to the hidden layer of the network

set.seed(12321)
df6_NN2 <- neuralnet(Gradrate_diff_6to4yrs ~ Applicants + Admission + UG1st_Enrolled
                     + P_submit_SAT + P_submit_ACT + ACTComp_25 + ACTComp_75 + P_admit
                     + Yield + Tuition2010_11 + Tuition2011_12 + Tuition2012_13 + Tuition2013_14
                     + Price_instate_2013_14 + Price_outstate_2013_14 + UG_enroll + Grad_enroll
                     + FTUG_enroll + PTUG_enroll + P_Amer_Indian + P_Asian + P_Af_American
                     + P_Latino + P_PIslander + P_White + P_2orM_races + P_RaceNA + P_NRAlien
                     + P_Asian_Native_PIslander + P_Women + P_1yrUG_any_aid + P_1yrUG_Fed_state_grant
                     + P_1yrUG_Fed_grant + P_1yrUG_Pell_grant + P_1yrUG_otherFed_grant
                     + P_1yrUG_state_local_grant + P_1yrUG_institute_grant + P_1yrUG_student_loan
                     + P_1yrUG_Fed_student_loan + P_1yrUG_other_loan + SB_Endowment_per_FTE, 
                     data = df5_Train, 
                     hidden = c(4, 1), 
                     act.fct = "logistic")

## Calculate training Error
NN2_Train_SSE <- sum((df6_NN2$net.result - df5_Train[, 'Gradrate_diff_6to4yrs'])^2)/2
NN2_Train_SSE
#Training error = 1.786

## Calculate test Error
Test_NN2_Output <- compute(df6_NN2, df5_Test[, 1:41])$net.result
NN2_Test_SSE <- sum((Test_NN2_Output - df5_Test[, 42])^2)/2
NN2_Test_SSE
#Test error = 0.876 while training error = 1.786  
#So test error is smaller than training error.

#----------------------------------------------------------------------------------------------
#Run 3: 2-Hidden Layers, Layer 1: 4-neurons, Layer 2: 1-neuron, tanh activation

#Modify regression hyper-parameters in the neuralnet() function
#Using tanh activation function here - so need to rescale the data from [0,1] to [-1,1] scale  
#For rescaling - using the rescale package.
scale11 <- function(x) {
  (2 * ((x - min(x))/(max(x) - min(x)))) - 1
}

df5b_Train <- df5_Train %>% mutate_all(scale11)
df5b_Test <- df5_Test %>% mutate_all(scale11)

# Run NN
set.seed(12321)
df6_NN3 <- neuralnet(Gradrate_diff_6to4yrs ~ Applicants + Admission + UG1st_Enrolled
                     + P_submit_SAT + P_submit_ACT + ACTComp_25 + ACTComp_75 + P_admit
                     + Yield + Tuition2010_11 + Tuition2011_12 + Tuition2012_13 + Tuition2013_14
                     + Price_instate_2013_14 + Price_outstate_2013_14 + UG_enroll + Grad_enroll
                     + FTUG_enroll + PTUG_enroll + P_Amer_Indian + P_Asian + P_Af_American
                     + P_Latino + P_PIslander + P_White + P_2orM_races + P_RaceNA + P_NRAlien
                     + P_Asian_Native_PIslander + P_Women + P_1yrUG_any_aid + P_1yrUG_Fed_state_grant
                     + P_1yrUG_Fed_grant + P_1yrUG_Pell_grant + P_1yrUG_otherFed_grant
                     + P_1yrUG_state_local_grant + P_1yrUG_institute_grant + P_1yrUG_student_loan
                     + P_1yrUG_Fed_student_loan + P_1yrUG_other_loan + SB_Endowment_per_FTE, 
                     data = df5b_Train, 
                     hidden = c(4, 1), 
                     act.fct = "tanh",
                     stepmax=1e7)   #Needed to avoid convergence problems

## Training Error 
NN3_Train_SSE <- sum((df6_NN3$net.result - df5b_Train[, 'Gradrate_diff_6to4yrs'])^2)/2
NN3_Train_SSE
#Training error = 7.115

## Test Error 
Test_NN3_Output <- compute(df6_NN3, df5b_Test[, 1:41])$net.result
NN3_Test_SSE <- sum((Test_NN3_Output - df5b_Test[, 42])^2)/2
NN3_Test_SSE
#Test error = 16.158
#So test error is greater than training error

#----------------------------------------------------------------------------------------------
#Run 4: 1-Hidden Layer, 1-neuron, tanh activation function

set.seed(12321)
df6_NN4 <- neuralnet(Gradrate_diff_6to4yrs ~ Applicants + Admission + UG1st_Enrolled
                     + P_submit_SAT + P_submit_ACT + ACTComp_25 + ACTComp_75 + P_admit
                     + Yield + Tuition2010_11 + Tuition2011_12 + Tuition2012_13 + Tuition2013_14
                     + Price_instate_2013_14 + Price_outstate_2013_14 + UG_enroll + Grad_enroll
                     + FTUG_enroll + PTUG_enroll + P_Amer_Indian + P_Asian + P_Af_American
                     + P_Latino + P_PIslander + P_White + P_2orM_races + P_RaceNA + P_NRAlien
                     + P_Asian_Native_PIslander + P_Women + P_1yrUG_any_aid + P_1yrUG_Fed_state_grant
                     + P_1yrUG_Fed_grant + P_1yrUG_Pell_grant + P_1yrUG_otherFed_grant
                     + P_1yrUG_state_local_grant + P_1yrUG_institute_grant + P_1yrUG_student_loan
                     + P_1yrUG_Fed_student_loan + P_1yrUG_other_loan + SB_Endowment_per_FTE, 
                     data = df5b_Train, 
                     act.fct = "tanh",
                     stepmax=1e7)    #Needed to avoid convergence problems

## Calculate training Error
NN4_Train_SSE <- sum((df6_NN4$net.result - df5b_Train[, 'Gradrate_diff_6to4yrs'])^2)/2
NN4_Train_SSE
#Training error = 21.519

## Calculate test Error
Test_NN4_Output <- compute(df6_NN4, df5b_Test[, 1:41])$net.result
NN4_Test_SSE <- sum((Test_NN4_Output - df5b_Test[, 42])^2)/2
NN4_Test_SSE
#Test error = 23.681
#So test error is higher than the training error.

#____________________________________________________________________________________________________
# Plot of run results - for comparison
Regression_NN_Errors <- tibble(Network = rep(c("NN1", "NN2", "NN3", "NN4"), each = 2), 
                               DataSet = rep(c("Train", "Test"), time = 4), 
                               SSE = c(NN1_Train_SSE, NN1_Test_SSE, 
                                       NN2_Train_SSE, NN2_Test_SSE,
                                       NN3_Train_SSE, NN3_Test_SSE,
                                       NN4_Train_SSE, NN4_Test_SSE))

Regression_NN_Errors %>% 
  ggplot(aes(Network, SSE, fill = DataSet)) + 
  geom_col(position = "dodge") + 
  ggtitle("Regression ANN's SSE")

#Observation: The lowest error in test data occurs in Run 1.

#____________________________________________________________________________________________________
#Variable importance plots - on run with least error

#-------------------------------------------------------------------------------------------------
#Variable importance - using garson() function

# The garson function uses Garson's algorithm to evaluate relative variable importance.
# This function identifies the relative importance of explanatory variables for a single response 
#variable by deconstructing the model weights.
# The results indicate relative importance as the absolute magnitude from zero to one.
# Only neural networks with one hidden layer and one output node can be evaluated.

#Results from Run 1 and 4 can be evaluated by garson() - as they have 1 hidden layer
garson(df6_NN1) +  theme(axis.text.x=element_text(angle=90,hjust=1)) + coord_flip() 

#-------------------------------------------------------------------------------------------------
#Variable importance - using olden() function

# The olden function is an alternative and more flexible approach to evaluate variable importance.
# An advantage of this approach is the relative contributions of each connection weight are 
#maintained in terms of both magnitude and sign as compared to Garson's algorithm 
#which only considers the absolute magnitude.
# The Olden's algorithm is capable of evaluating neural networks with multiple hidden layers 
#and response variables. 
olden(df6_NN1) + theme(axis.text.x=element_text(angle=90,hjust=1)) + coord_flip()

#_____________________________________________________________________________________________________

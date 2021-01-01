#Capstone Project: Use ANN to predict 4-yr graduation rate (used ACT comp scores only)
#Only numeric columns modeled - categorical columns NOT included
#ANN - artificial neural network

library(tidyverse)
library(neuralnet)
library(GGally)
library(dplyr)               #For scaling
library(NeuralNetTools)
library(nnet)
library("data.table")

#________________________________________________________________________________________________
#Read input file

#Read in project data file
df1 <- read.csv("US_Univ_UG_shortnames.csv")
View(df1)

#________________________________________________________________________________________________
#Initial data cleanup

#Check number of NAs in each column
sapply(df1, function(x) sum(is.na(x)))
nrow(df1)

#No. of rows with NA for ACTComp is less than SAT scores. So choosing ACT scores as driver (feature). 
#Remove all rows where ACTComp_25 and ACTComp_75 is NA
df2 <- df1[!is.na(df1$ACTComp_25)&!is.na(df1$ACTComp_75),]
View(df2)
nrow(df2)

#List names of columns
colnames(df2)

#Cull columns by name
#Removed select columns - SAT scores and some identifier info
df3 <- df2[ , -which(names(df2) %in% c("ID_number", "Name", "Yr", "ZIP", "County", "Longitude",
                                       "Latitude", "SATRead_25", "SATRead_75", "SATMath_25", 
                                       "SATMath_75", "SATWrite_25", "SATWrite_75", "Gradrate_5yrs",
                                       "Gradrate_6yrs"))]

#Label - Graduation rate in 4 yrs
#Remove all rows where grad rate in 4 yrs is NA
df3b <- df3[!is.na(df3$Gradrate_4yrs),]
nrow(df3b) #has 1195 rows

#Check number of NAs in each column
sapply(df3b, function(x) sum(is.na(x)))

#Remove select columns - with large number of NA values
#ANN requirement - NA values can't be present in any column
df3c <- df3b[ , -which(names(df3b) %in% c("P_1stUG_instate", "P_1stUG_outstate", "P_1stUG_foreign", 
                                          "P_1stUG_resNA"))]
sapply(df3c, function(x) sum(is.na(x)))

#Remove rows with NA values in any column
df4 <- df3c[complete.cases(df3c), ]
nrow(df4) #Rows without NA values = 1151

#Check if any column has NA values
sapply(df4, function(x) sum(is.na(x)))

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

colnames(df4)
num_cols = c('Applicants', 'Admission', 'UG1st_Enrolled', 'P_submit_SAT', 'P_submit_ACT'
             ,'ACTComp_25', 'ACTComp_75', 'P_admit', 'Yield', 'Tuition2010_11', 'Tuition2011_12'
             ,'Tuition2012_13', 'Tuition2013_14', 'Price_instate_2013_14', 'Price_outstate_2013_14'
             ,'UG_enroll', 'Grad_enroll', 'FTUG_enroll', 'PTUG_enroll', 'P_Amer_Indian'
             ,'P_Asian', 'P_Af_American', 'P_Latino', 'P_PIslander', 'P_White', 'P_2orM_races'
             ,'P_RaceNA', 'P_NRAlien', 'P_Asian_Native_PIslander', 'P_Women', 'Gradrate_4yrs'
             ,'P_1yrUG_any_aid', 'P_1yrUG_Fed_state_grant', 'P_1yrUG_Fed_grant', 'P_1yrUG_Pell_grant'
             ,'P_1yrUG_otherFed_grant', 'P_1yrUG_state_local_grant', 'P_1yrUG_institute_grant'
             ,'P_1yrUG_student_loan', 'P_1yrUG_Fed_student_loan', 'P_1yrUG_other_loan'
             ,'SB_Endowment_per_FTE')

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

#Use dplyr’s anti_join() function to extract all the observations that are not within the 
#df5_Train data set to create our test data set i.e., df5_Test
df5_Test <- anti_join(df5, df5_Train)
nrow(df5_Train)      #Training set = 921 cases
nrow(df5_Test)       #Test set = 230 cases

#______________________________________________________________________________________________________
#Run 1: 1st ANN regression
#construct a 1-hidden layer ANN with 1 neuron, the simplest of all neural network
#The df6_NN1 is a list containing all parameters of the regression ANN 
#and the results of the neural network on the test data set 
set.seed(12321)
df6_NN1 <- neuralnet(Gradrate_4yrs ~ Applicants + Admission + UG1st_Enrolled
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
NN1_Train_SSE <- sum((df6_NN1$net.result - df5_Train_2[, 'Gradrate_4yrs'])^2)/2
paste("SSE: ", round(NN1_Train_SSE, 4))

#To calculate SSE in test dataset - need to place the label at the last column (to the right) of test data
df5_Test$Gradrate_4yrs_2 <- df5_Test$Gradrate_4yrs

#Delete Gradrate_4yrs column
df5_Test = subset(df5_Test, select = -c(Gradrate_4yrs))

#To calculate the test error, we first must run our test observations through the df6_NN1 ANN.
#This is accomplished with the neuralnet package compute() function
#1st input of compute() - the desired neural network object created by the neuralnet() function, 
#2nd argument of compute() - the test data set feature (independent variable(s)) values
Test_NN1_Output <- compute(df6_NN1, df5_Test[, 1:41])$net.result
NN1_Test_SSE <- sum((Test_NN1_Output - df5_Test[, 42])^2)/2
NN1_Test_SSE
#The compute() function outputs the response variable, in our case the Gradrate_4yrs, 
#as estimated by the neural network. 
#Once we have the ANN estimated response we can compute the NN1_Test_SSE.
#Comparing the test error of 1.064 to the training error of 4.3602 we see that 
#in our case our test error is smaller than our training error.

#______________________________________________________________________________________________
#Run 2: 2-Hidden Layers, Layer 1: 4-neurons, Layer 2: 1-neuron, logistic activation function
#Modify regression hyper-parameters
#So far, have constructed the most basic of regression ANNs without modifying any of the default 
#hyperparameters associated with the neuralnet() function. 
#Should try and improve the network by modifying its basic structure and hyperparameters
#To begin we add depth to the hidden layer of the network

set.seed(12321)
df6_NN2 <- neuralnet(Gradrate_4yrs ~ Applicants + Admission + UG1st_Enrolled
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
NN2_Train_SSE <- sum((df6_NN2$net.result - df5_Train[, 'Gradrate_4yrs'])^2)/2
NN2_Train_SSE

## Calculate test Error
Test_NN2_Output <- compute(df6_NN2, df5_Test[, 1:41])$net.result
NN2_Test_SSE <- sum((Test_NN2_Output - df5_Test[, 42])^2)/2
NN2_Test_SSE
#Comparing the test error of 2.517 to the training error of 2.755 we see that 
#in our case our test error is slightly smaller than our training error.

#__________________________________________________________________________________________
#Run 3: 2-Hidden Layers, Layer 1: 4-neurons, Layer 2: 1-neuron, tanh activation
#Then change the activation function from logistic to the tangent hyperbolicus (tanh) to 
#determine if these modifications can improve the test data set SSE.
##When using tanh activation function, we first must rescale the data from [0,1] to [-1,1] 
#using the rescale package.
#Rescale for tanh activation function
scale11 <- function(x) {
  (2 * ((x - min(x))/(max(x) - min(x)))) - 1
}

df5b_Train <- df5_Train %>% mutate_all(scale11)
df5b_Test <- df5_Test %>% mutate_all(scale11)

# Run NN
set.seed(12321)
df6_NN3 <- neuralnet(Gradrate_4yrs ~ Applicants + Admission + UG1st_Enrolled
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
NN3_Train_SSE <- sum((df6_NN3$net.result - df5b_Train[, 'Gradrate_4yrs'])^2)/2
NN3_Train_SSE
## Test Error 
Test_NN3_Output <- compute(df6_NN3, df5b_Test[, 1:41])$net.result
NN3_Test_SSE <- sum((Test_NN3_Output - df5b_Test[, 42])^2)/2
NN3_Test_SSE

#test error of 8.556 to the training error of 10.344
#_________________________________________________________________________________________________
#Run 4: 1-Hidden Layer, 1-neuron, tanh activation function
set.seed(12321)
df6_NN4 <- neuralnet(Gradrate_4yrs ~ Applicants + Admission + UG1st_Enrolled
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
NN4_Train_SSE <- sum((df6_NN4$net.result - df5b_Train[, 'Gradrate_4yrs'])^2)/2
NN4_Train_SSE

## Calculate test Error
Test_NN4_Output <- compute(df6_NN4, df5b_Test[, 1:41])$net.result
NN4_Test_SSE <- sum((Test_NN4_Output - df5b_Test[, 42])^2)/2
NN4_Test_SSE

#test error of 7.218 to the training error of 17.361
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

#_____________________________________________________________________________________________________
#Variable importance plots

# Plot importance of each variable using garson()
# The garson function uses Garson's algorithm to evaluate relative variable importance.
# This function identifies the relative importance of explanatory variables for a single response 
#variable by deconstructing the model weights.
# The results indicate relative importance as the absolute magnitude from zero to one.
# Only neural networks with one hidden layer and one output node can be evaluated.

#Results from Run 1 and 4 can be evaluated by garson() - as they have 1 hidden layer
garson(df6_NN1) +  theme(axis.text.x=element_text(angle=90,hjust=1)) + coord_flip()

#------------------------------------------------------------------------------------------------
#Variable importance in Run 1 has been plotted earlier using olden() functions.

## Plot importance of each variable using olden()
# The olden function is an alternative and more flexible approach to evaluate variable importance.
# An advantage of this approach is the relative contributions of each connection weight are 
#maintained in terms of both magnitude and sign as compared to Garson's algorithm 
#which only considers the absolute magnitude.
# The Olden's algorithm is capable of evaluating neural networks with multiple hidden layers 
#and response variables. 
olden(df6_NN1) + theme(axis.text.x=element_text(angle=90,hjust=1)) + coord_flip()



#______________________________________________________________________________________________________





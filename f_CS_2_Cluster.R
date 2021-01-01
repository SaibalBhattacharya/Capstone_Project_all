#3rd clustering code - used both numeric and categoric columns for cluster analysis
#Includes: ACTComp_25 and ACTComp_75
#Also includes: Gradrate_5yrs, Gradrate_6yrs, SATRead_25, SATRead_75, SATMath_25, SATMath_75 
#Excludes: SATWrite_25 and SATWrite_75 - as these have large number of NAs

#Clustering - find relationships between the n observations without training by a response variable
#Clustering helps to identify alike observations, and potentially categorize them.
#K-means clustering: simplest and the most commonly used clustering method for splitting a dataset 
#into a set of k groups.

library(tidyverse)  # data manipulation
library(cluster)    # clustering algorithms
library(factoextra) # clustering algorithms & visualization
library(gridExtra)  # plot multiple charts for different k values
library(doBy)       # required for summaryBy - calculate summary statistics
library(caret)      #for on-hot encoding to generate dummy variables

#Read in project data file
df1 <- read.csv("US_Univ_UG_shortnames.csv")
View(df1)

#___________________________________________________________________________________________
#Initial data processing

#Check number of NAs in each column
sapply(df1, function(x) sum(is.na(x)))
nrow(df1)

df2 <- df1

#List names of columns
colnames(df2)
#df2$Name

#Cull columns by name
#Remove select columns - that have identifier info
df3 <- df2[ , -which(names(df2) %in% c("ID_number","Yr", "ZIP", "County", "Longitude",
                                       "Latitude"))]

df3b <- df3
 
#Check number of NAs in each column
sapply(df3b, function(x) sum(is.na(x)))

#Remove select columns - with large number of NA values
#Clustering requirement - NA values can't be present in any column
df3c <- df3b[ , -which(names(df3b) %in% c("P_1stUG_instate", "P_1stUG_outstate", "P_1stUG_foreign", 
                                          "P_1stUG_resNA", "SATWrite_25", "SATWrite_75"))]
sapply(df3c, function(x) sum(is.na(x)))

#Remove rows with NA values in any column
df4 <- df3c[complete.cases(df3c), ]
nrow(df4) #Rows without NA values = 1066

#Check if any column has NA values
sapply(df4, function(x) sum(is.na(x)))

#_________________________________________________________________________________________
#University/college name will be used as row name (in code)
#So find if any university name is duplicated - can't have two rows with same name

#Find all universities that have the same name
df5 <- df4
dupe = df5[,c('Name')] # select columns to check duplicates
df6 <- df5[duplicated(dupe) | duplicated(dupe, fromLast=TRUE),] 
#fromLast = TRUE option used as it returns TRUE only from the duplicate value on-wards 
View(df6) #There are some duplicate names under Name
colnames(df6)

#Extract all info about universities that have the same names
df6b <- select(df6, Name, Relgious_y_n, State, Region, Status, HBCU, Urbanization, 
               Type_of_univ)
View(df6b)

#Arrange universities in alphabetical order
df6c <- df6b[order(df6b[,'Name']), ]

#Display universities/colleges with same name
df6d <- select(df6c, Name, State, Region)
head(df6d, 13)
#Observation: Some universities may have same name but are in different states
#So, these universities are NOT duplicates

#So create new column that combines Name and State - to make each entry unique
df5$Name_State <- paste(df5$Name, '_', df5$State)

#Check if all entries in Name_State column are unique - i.e., there are no duplicates
length(unique(df5$Name_State)) == nrow(df5)
#Returns TRUE if there are no duplicates

#Name rows using contents of Name_State column
df7 <- df5[,-57]
rownames(df7) <- df5[,57]
colnames(df7)

#Remove columns: Name and State 
#Remove categorical variable State - as Region carries this information 
df8 <- df7[ , -which(names(df7) %in% c("Name", "State"))]
head(df8, 1)
#_____________________________________________________________________________________
#Create dummy variables
# For categorical variables - use one-hot to create dummy variables
one_hot <- dummyVars(~ ., df8, fullRank = FALSE)
df8_hot <- predict(one_hot, df8) %>% as.data.frame()

#Change/shorten column names - make them easier to read
df8b_hot <- df8_hot
colnames(df8b_hot)
names(df8b_hot)[names(df8b_hot) == "Relgious_y_nNo"] <- "Religious_n"
names(df8b_hot)[names(df8b_hot) == "Relgious_y_nYes"] <- "Religious_y"
names(df8b_hot)[names(df8b_hot) == "RegionFar_West"] <- "Region_FarWest"
names(df8b_hot)[names(df8b_hot) == "RegionGreat_Lakes"] <- "Region_GreatLakes"
names(df8b_hot)[names(df8b_hot) == "RegionMid_East"] <- "Region_MidEast"
names(df8b_hot)[names(df8b_hot) == "RegionNew_England"] <- "Region_NewEngland"
names(df8b_hot)[names(df8b_hot) == "RegionPlains"] <- "Region_Plains"
names(df8b_hot)[names(df8b_hot) == "RegionRocky_Mountains"] <- "Region_Rockies"
names(df8b_hot)[names(df8b_hot) == "RegionSouth_East"] <- "Region_SouthEast"
names(df8b_hot)[names(df8b_hot) == "RegionSouth_West"] <- "Region_SouthWest"
names(df8b_hot)[names(df8b_hot) == "StatusPrivate not-for-profit"] <- "Private"
names(df8b_hot)[names(df8b_hot) == "StatusPublic"] <- "Public"
names(df8b_hot)[names(df8b_hot) == "UrbanizationCity: Large"] <- "Urban_City_L"
names(df8b_hot)[names(df8b_hot) == "UrbanizationCity: Midsize"] <- "Urban_City_M"
names(df8b_hot)[names(df8b_hot) == "UrbanizationCity: Small"] <- "Urban_City_S"
names(df8b_hot)[names(df8b_hot) == "UrbanizationRural: Distant"] <- "Urban_Rural_D"
names(df8b_hot)[names(df8b_hot) == "UrbanizationRural: Fringe"] <- "Urban_Rural_F"
names(df8b_hot)[names(df8b_hot) == "UrbanizationRural: Remote"] <- "Urban_Rural_R"
names(df8b_hot)[names(df8b_hot) == "UrbanizationSuburb: Large"] <- "Urban_Suburb_L"
names(df8b_hot)[names(df8b_hot) == "UrbanizationSuburb: Midsize"] <- "Urban_Suburb_M"
names(df8b_hot)[names(df8b_hot) == "UrbanizationSuburb: Small"] <- "Urban_Suburb_S"
names(df8b_hot)[names(df8b_hot) == "UrbanizationTown: Distant"] <- "Urban_Town_D"
names(df8b_hot)[names(df8b_hot) == "UrbanizationTown: Fringe"] <- "Urban_Town_F"
names(df8b_hot)[names(df8b_hot) == "UrbanizationTown: Remote"] <- "Urban_Town_R"
names(df8b_hot)[names(df8b_hot) == "Type_of_univBS_&_Associate's"] <- "Univ_BS_Assoc"
names(df8b_hot)[names(df8b_hot) == "Type_of_univBS_Arts_&_Sciences"] <- "Univ_BS_Arts_Sci"
names(df8b_hot)[names(df8b_hot) == "Type_of_univBS_Diverse_fields"] <- "Univ_BS_Diverse"
names(df8b_hot)[names(df8b_hot) == "Type_of_univDoctoral_Research"] <- "Univ_Doc_Res"
names(df8b_hot)[names(df8b_hot) == "Type_of_univHigh_research_univ"] <- "Univ_High_Res"
names(df8b_hot)[names(df8b_hot) == "Type_of_univMS_Large_program"] <- "Univ_MS_L"
names(df8b_hot)[names(df8b_hot) == "Type_of_univMS_Medium_program"] <- "Univ_MS_M"
names(df8b_hot)[names(df8b_hot) == "Type_of_univMS_Small_program"] <- "Univ_MS_S"
names(df8b_hot)[names(df8b_hot) == "Type_of_univVery_High_research_univ"] <- "Univ_vHigh_Res"

#________________________________________________________________________________________
#Data scaling 

#Any missing value in the data must be removed or estimated
#Data must be standardized (i.e., scaled) to make variables comparable.
#standardization: transforming variables so they have mean zero and standard deviation 1.
#scaling/standardizing the data using the R function "scale"
#don’t want the clustering algorithm to depend to an arbitrary variable unit
df9 <- scale(df8b_hot)

#__________________________________________________________________________________________
#K-means clustering - Initial run with 2 clusters

#1st step - indicate the number of clusters (k) that will be generated in the final solution
#Default - R software uses 10 as the default value for the maximum iterations for convergence
#compute k-means in R with the "kmeans" function
#will group the data into two clusters (centers = 2)
#"nstart" option - attempts multiple initial configurations and reports on the best one
# nstart = 25 - will generate 25 initial configurations
k2 <- kmeans(df9, centers = 2, nstart = 25)
str(k2)
#Output details:
#cluster: A vector of integers (from 1:k) indicating the cluster to which each point is allocated.
#centers: A matrix of cluster centers.
#totss: The total sum of squares.
#withinss: Vector of within-cluster sum of squares, one component per cluster.
#tot.withinss: Total within-cluster sum of squares, i.e. sum(withinss).
#betweenss: The between-cluster sum of squares, i.e. $totss-tot.withinss$.
#size: The number of points in each cluster.

#Print results - shows universities grouped in two clusters
k2

#View results by using fviz_cluster
fviz_cluster(k2, data = df9)
#Each cluster contained large number of universities - so none of the clusters contained top
#colleges/universities exclusively.
#NOTE: If there are more than two dimensions (variables) fviz_cluster will perform 
#principal component analysis (PCA) and plot the data points according to the first 
#two principal components that explain the majority of the variance.

#________________________________________________________________________________________
#Defining optimal number of clusters

#3 most popular methods to find optimal clusters: Elbow & Silhouette methods, Gap statistics

#----------------------------------------------------------------------
#Elbow method - Total intra-cluster variation (total within-cluster variation or 

#total within-cluster sum of square) is minimized
#The location of a bend (knee) in the plot is generally considered as an indicator 
#of the appropriate number of clusters.
set.seed(123)

# function to compute total within-cluster sum of square 
wss <- function(k) {
  kmeans(df9, k, nstart = 10 )$tot.withinss
}

# Compute and plot wss for k = 1 to k = 15
k.values <- 1:15

# extract wss for 2-15 clusters
wss_values <- map_dbl(k.values, wss)

plot(k.values, wss_values,
     type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares")
#Results: 6 clusters appear to be optimum

#-----------------------------------------------------------------------------------
#Average Silhouette Method - measures the quality of a clustering, i.e., 

#how well each object lies within its cluster
#A high average silhouette width indicates a good clustering
#optimal number of clusters k is the one that maximizes the average silhouette 
#over a range of possible values for k
#Use the silhouette function in the cluster package to compuate the average silhouette width.
# function to compute average silhouette for k clusters
avg_sil <- function(k) {
  km.res <- kmeans(df9, centers = k, nstart = 25)
  ss <- silhouette(km.res$cluster, dist(df9))
  mean(ss[, 3])
}

# Compute and plot wss for k = 2 to k = 15
k.values <- 2:15

# extract avg silhouette for 2-15 clusters
avg_sil_values <- map_dbl(k.values, avg_sil)

plot(k.values, avg_sil_values,
     type = "b", pch = 19, frame = FALSE, 
     xlab = "Number of clusters K",
     ylab = "Average Silhouettes")
#Results show that 2 clusters maximize the average silhouette values 
#with 4 clusters coming in as second optimal number of clusters.

#---------------------------------------------------------------------
#Gap statistic method

#The gap statistic compares the total intracluster variation for different values of k 
#with their expected values under null reference distribution of the data 
#(i.e. a distribution with no obvious clustering).
#The reference dataset is generated using Monte Carlo simulations of the sampling process.
#The gap statistic measures the deviation of the observed Wk value from its 
#expected value under the null hypothesis. 
#The estimate of the optimal clusters (Gapn(k)) will be the value that maximizes Gapn(k)
#This means that the clustering structure is far away from the uniform distribution of points.

#To compute the gap statistic method we can use the clusGap function 
#which provides the gap statistic and standard error for an output.
# Compute gap statistic
set.seed(123)
gap_stat <- clusGap(df9, FUN = kmeans, nstart = 25,
                    K.max = 10, B = 50)  #K.max determines number of iterations
#Increasing K.max to 100 - increases run time significantlty, but no convergence errors

# Print the result
print(gap_stat, method = "firstmax")

#visualize the results 
fviz_gap_stat(gap_stat)
#Results: the highest value for gap = 0.959 for cluster = 10
#while 6 clusters result in gap = 0.936

#__________________________________________________________________________________
# Final analysis - extracting Results based on Elbow method results 

# The Elbow method suggests that 6 is the number of optimal clusters
# perform the final analysis and extract the results using 6 clusters.
# Compute k-means clustering with k = 6
set.seed(123)
final_Elbow <- kmeans(df9, 6, nstart = 25)
print(final_Elbow)

#visualize the results using fviz_cluster
fviz_cluster(final_Elbow, data = df9)

#Find cluster of best university/colleges
df9b <- as.data.frame(final_Elbow$cluster)
#Results: The top colleges/universities are all in cluster 5. Cluster 5 has 80 members.
#--------------------------------------------------------------------
# Final analysis - extracting Results based on Gap analysis results 

# The Gap analysis indicates that 10 is the number of optimal clusters
# perform the final analysis and extract the results using 10 clusters.
# Compute k-means clustering with k = 10
set.seed(123)
final_Gap <- kmeans(df9, 10, nstart = 25)
print(final_Gap)

#visualize the results using fviz_cluster
fviz_cluster(final_Gap, data = df9)

#Find cluster of best university/colleges
df9c <- as.data.frame(final_Gap$cluster)
#Results: The top colleges/universities are all in cluster 3. Cluster 3 has 80 members.

#___________________________________________________________________________________________________
#Find summary statistics for each cluster - based on Elbow analysis

#Both Gap and Elbow analysis show that the cluster containing Ivy league colleges have 80 members.
#So 6 clusters (from Elbow analysis) are selected - fewer cluster will show distinctly diff clusters.
#10 clusters (from Gap) analysis - some clusters maynot be distinct because there are more clusters.

#Extract cluster number and row name
df10 <- as.data.frame(final_Elbow$cluster)

#Merge cluster number with original unscaled data
#df8 - contains unscaled data (all numerical columns only)
df11 <- merge(df10, df8, by="row.names", all=TRUE)

colnames(df11)

#Rename column names
df12 <- df11 %>% rename(Univ_Name = "Row.names", Cluster_no = "final_Elbow$cluster") #load library(dplyr) 

#To compare using box and violin plots - convert Cluster_no into factor
df12$Cluster_no <- as.factor(df12$Cluster_no)

#------------------------------------------------------------------------
#Compare ACTComp_75 across clusters 
summaryBy(ACTComp_75 ~ Cluster_no, data = df12,      #install library(doBy) 
          FUN = list(mean, max, min, median, sd))

e <- ggplot(df12, aes(x = Cluster_no, y = ACTComp_75))

#Compare using box plots embedded within violin plots
e + geom_violin(aes(fill = Cluster_no), trim = FALSE) + 
  geom_boxplot(width = 0.2)+
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#E495A5", "#ABB065",  
                               "#FFFF00FF"))+
  theme(legend.position = "none")

#Observation: Cluster 5 (Ivy plus colleges/universities) - has the highest ACT Composite 75 percentile scores 
#--------------------------------------------------------------------------
#Compare SATRead_75 across clusters
summaryBy(SATRead_75 ~ Cluster_no, data = df12,      
          FUN = list(mean, max, min, median, sd))

e <- ggplot(df12, aes(x = Cluster_no, y = SATRead_75))

#Compare using box plots embedded within violin plots
e + geom_violin(aes(fill = Cluster_no), trim = FALSE) + 
  geom_boxplot(width = 0.2)+
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#E495A5", "#ABB065",  
                               "#FFFF00FF"))+
  theme(legend.position = "none")

#Observation: Cluster 5 (Ivy plus colleges/universities) - has the highest SAT Reading 75 percentile scores
#--------------------------------------------------------------------------
#Compare SATMath_75 across clusters
summaryBy(SATMath_75 ~ Cluster_no, data = df12,      
          FUN = list(mean, max, min, median, sd))

e <- ggplot(df12, aes(x = Cluster_no, y = SATMath_75))

#Compare using box plots embedded within violin plots
e + geom_violin(aes(fill = Cluster_no), trim = FALSE) + 
  geom_boxplot(width = 0.2)+
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#E495A5", "#ABB065",  
                               "#FFFF00FF"))+
  theme(legend.position = "none")

#Observation: Cluster 5 (Ivy plus colleges/universities) - has the highest SAT Math 75 percentile scores
#--------------------------------------------------------------------------
#Compare P_admit across clusters 
summaryBy(P_admit ~ Cluster_no, data = df12,         
          FUN = list(mean, max, min, median, sd))

e <- ggplot(df12, aes(x = Cluster_no, y = P_admit))

#Compare using box plots embedded within violin plots
e + geom_violin(aes(fill = Cluster_no), trim = FALSE) + 
  geom_boxplot(width = 0.2)+
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#E495A5", "#ABB065",  
                               "#FFFF00FF"))+
  theme(legend.position = "none")

#Observation: Cluster 5 (Ivy plus colleges/universities) - has the lowest admission rates
#---------------------------------------------------------------------------
#Compare Gradrate_4yrs across clusters 
summaryBy(Gradrate_4yrs ~ Cluster_no, data = df12,    
          FUN = list(mean, max, min, median, sd))

e <- ggplot(df12, aes(x = Cluster_no, y = Gradrate_4yrs))

#Compare using box plots embedded within violin plots
e + geom_violin(aes(fill = Cluster_no), trim = FALSE) + 
  geom_boxplot(width = 0.2)+
  scale_fill_manual(values = c("#00AFBB", "#E7B800", "#FC4E07", "#E495A5", "#ABB065",  
                               "#FFFF00FF"))+
  theme(legend.position = "none")

#Observation: Cluster 5 (Ivy plus colleges/universities) - has the highest 4-yr graduation rates
#________________________________________________________________________________________

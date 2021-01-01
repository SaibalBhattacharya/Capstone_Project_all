# Capstone_Project_all

EXECUTIVE SUMMARY

OBJECTIVES:

Identify elite colleges/universities, parameters affecting 4-yr graduation rates, and drivers behind increase in graduation rates between 4 and 6 years

INTRODUCTION: 

This project analyzes a publicly available dataset (available for download at https://public.tableau.com/en-us/s/resources) that includes 2013 admissions data from 1534 US colleges and universities. The original data was compiled by the Integrated Postsecondary Education Data System (IPEDS) which is part of the National Center for Education Statistics (NCES).

TASKS PERFORMED: 

This capstone project attempts to perform 4 major tasks:

•	Task 1: Quick visualization of dataset (in Tableau Public)
•	Task 2: Identify highly ranked (selective) colleges and universities using unsupervised machine learning (ML) algorithms
•	Task 3: Identify drivers affecting 4-yr graduation rates in US colleges and universities using different ML techniques
•	Task 4: Identify drivers affecting the increase in graduation rates between 4 and 6 yrs in US universities and colleges using different ML methods

RESULTS:

TASK 1: Tableau proved to be a very versatile tool for a quick look at the dataset. It helped get a visual feel of the locations of all the university/colleges, and I was able to quickly identify critical variables that appeared to correlate with my objective function - 4-yr graduation rate. The link to my Tableau project is as follows: https://public.tableau.com/views/Capstone_Saibal/CapstoneStory?:language=en&:display_count=y&publish=yes&:origin=viz_share_link

TASK 2.

I used the unsupervised K-means cluster analysis to group the colleges/universities. I used three of the most popular methods to find optimal clusters: Elbow & Silhouette methods, and Gap statistics. The Elbow analysis resulted in an optimal cluster number of 6 while the Gap analysis resulted in an optimal cluster of 10. The optimal cluster number from Silhouette analysis turned up to be 2 - and it was found insufficient to group the top ranked colleges/universities. However, using cluster number = 6 and 10, resulted in the effective grouping of the top colleges/universities. In both cases, the cluster containing the top colleges/universities had 80 members (out of a total of 1534 US colleges/universities).

Based on the US News 2021 ranks for these universities it became apparent that the K-means algorithm did a good job in selecting all of the top 19 ranked National Universities. Beyond the top 19 ranked universities, the algorithm additionally selected some more universities between rank 20 and 66, but it didn’t select every university within this range. The K-means algorithm also did a good job in selecting all of the top 15 ranked National Liberal Arts Colleges. Beyond the top 15 ranked colleges, the algorithm additionally selected some more colleges between rank 20 and 63, but it didn’t select every college within this range.

I think that the top 20 National Universities and Liberal Arts Colleges are exceptional, and they stand out head and shoulders above others, and the K-means algorithm was effective in identifying them. Colleges and universities ranked lower than 20 share similar characteristics with lesser ranked institutes, and so are difficult to identify and group effectively.

TASK 3.

I used Random Forest (RF) and Neural Network (NN) regression analysis to identify variables (features) that affect the 4-yr graduation rate in US colleges/universities.

The variable importance plot from the RF model showed that the top 5 variables that affect 4-yr graduation rates are: P_1yrUG_Pell_grant (i.e., percent of 1st yr undergraduates with Pell grants), P_1yrUG_Fed_grant, Price_outstate_2013_14, Tuitions_2010_11, Tuition_2013_14, Tuition_2011_12, and Tuition_2012_13. Students on Pell grants and Federal grants mostly come from economically disadvantaged families, and so may have to balance school with other family responsibilities which might cause a delay in their graduation. Regarding total price of attendance and tuitions (for 2010_11, 2013_14, 2011_12, and 2012_13) affecting the 4-yr graduation rates, it is possible that for schools with high attendance costs, parents try to ensure that their kids graduate on time (within 4-yr window) to prevent expenses from ballooning further. Finally, and as expected, ACTComp_25 and ACTComp_75 scores also seem to affect the 4-yr graduation rate. Schools with high scores under these categories tend to enroll motivated kids who want to graduate in 4 years.

The Neural Network (NN) regression model was run using only the numeric features as it would crash due to convergence problems when run on both numeric and categorical features.

Variable importance plots showed that P_Asian (% undergraduate enrollment – Asian), SB_Endowment_per FTE (endowment per full time equivalent enrollment), Tuition 2012_13, and Price_outstate_2013_14 affect the 4-yr graduation rate positively, i.e., higher values of each of these parameters result in higher 4-yr graduation rates. It also showed that features such as P_Asian_Native_PIslander (i.e., % undergraduate enrollment - Asian/Native Hawaiian/Pacific Islander), PTUG_enroll (part time undergraduate enrollment), P_Af_American (% undergraduate - Black or African American), Price_instate_2013_14, and P_Latino (i.e., % undergraduate – Latino) affect the 4-yr graduation rate negatively.

When the variable importance plots from the RF model is compared with the NN model (with only numeric parameters), it was found that seven out of the top 15 driver features that affect the 4-yr graduation rate are common between these two models.

TASK 4.

I used Random Forest regression to identify variables (features) that affect the increase in graduation rates between 6 and 4 years in US colleges/universities. The variable importance plot showed that the top four variables that affect the increase in graduation rates between 4 and 6 years are all related to tuition: Tuition2010_11, Tuition2012_13, Tuition2011_12, and Tuition2013_14. It is understandable that parents paying high tuition want to have their kids graduate as early as possible. So, if they miss graduating in 4 years, then parents ensure that they graduate in the 5th or 6th year. The 5th most important feature is the PTUG_enroll (part time undergraduate enrollment). Many part time students have multiple obligations outside school, and so this may result in delaying their graduation beyond 4 years. The 6th most important feature is Price_instate_2013_14 (total price paid by instate students in 2013-14). I am unsure about how to explain the role that Price_instate_2013_14 plays in determining increase of graduation rate between 4 and 6 years other than that it also refers to money being spent on school, and delaying graduation beyond 4 years adds to the money spent in college. So higher this price, the more incentive a student has to graduate on time (in 4 years).

When run with numeric and categorical variables, the NN model would crash due to convergence issues (when I used the tanh activation function). Thus, I ran a NN regression model using only the numeric features. The top 5 features that significantly affect the increase in graduation rates between 4 and 6 years include: UG1st_Enrolled (undergraduate freshmen enrolled), FTUG_enroll (full time undergraduate enrollment), P_Asian_Native_PIslander (% undergraduate enrollment - Asian/Native Hawaiian/Pacific Islander), P_Asian (% undergraduate enrollment – Asian), and P_PIslander (% undergraduate enrollment - Native Hawaiian or Other Pacific Islander).

FTUG_enroll (full time enrollment), P_Asian_Native_PIslander (% undergraduate enrollment - Asian/Native Hawaiian/Pacific Islander), UG_enroll (undergraduate enrollment), P_NRAlien (% undergraduate enrollment - Nonresident Alien), and Price_instate_2013_14 (total price for instate students) affect the increase in graduation rate from 4 to 6 years positively, i.e., higher values of each of these parameters result in higher increases in graduation rates. It is understandable that increased number of students will try to graduate within 6 yrs in schools with higher number of full-time undergrads (FTUG_enroll), and that students in schools with high in-state attendance price tag (Price_instate_2013_14) will also want to complete their degrees soon after completion of 4 yrs. Also, many students who are Nonresident Alien (P_NRAlien) have financial and other limitations which prevent them for graduating in 4 years, but these same financial limitations motivate them to graduate within 6 years so as not to add to their already stressed finances.

Additionally, this analysis shows that for schools with high enrollment of undergraduate freshmen (UG1st_Enrolled), most students "possibly" graduate on time (in 4 years), and this results in lower increases in graduation rates between 4 and 6 years. This analysis also indicates that schools with a higher percentage of students of Asian origin (P_Asian) tend to show lesser increases in graduation rates beyond 4 years. This maybe because most Asian kids graduate within 4 years, and so there are less of them to graduate in the 5th or 6th years, or those you don’t graduate within 4 years take more than 6 years to graduate. This analysis also shows that if the school’s endowment (per full time equivalent enrollment), i.e., SB_Endowment_per_FTE is high, then perhaps the school has enough resource to help at risk students, and so most kids graduate within 4 years, and the increase in graduation rate between 4 and 6 years is not high.

A comparison of the top 15 features between the RF and this NN model shows that 8 features are shared between NN and RF models.

FINAL NOTE

RF and NN models are black box models – there are no known (phyics-based) laws that govern the relationship between the multiple input features and the label (here, the increase in graduation rates between 4 and 6 years) or the 4-yr graduation rate. Thus, like in any social science research, it is difficult to establish causality between feature(s) and the label – one can only provide educated guesses as to why certain features rank high on the variable importance plots.

ADDITIONAL DETAILS

The problems statements and my workflow to answer each question are detailed in the file Capstone_Project_details_final.pdf.

REFERENCES:

The neural network regression code and workflow was burrowed from http://uc-r.github.io/ann_regression. The neural network regression code and workflow was burrowed from https://uc-r.github.io/random_forests#prereq. The K-means cluster analysis code and workflow was burrowed from: http://uc-r.github.io/kmeans_clustering#elbow.


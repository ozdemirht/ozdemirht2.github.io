######################################################
#
# Trains the model wih the pml-training.csv
# Tests the trained model with pml-testing.csv
#
# Inputs: (assumes that the following files are available 
#         in the current working directory)
#   ./pml-training.csv
#   ./pml-testing.csv
#
# Outputs: Produces prediction results into the files
#          as requested by the assignment for submission
#         - assumes ./out directory exists
#
######################################################
library(ggplot2)
library(lattice)
library(rpart)        # decision tree
library(caret)        # caret package
library(corrplot)     # correlation matrix plot
library(randomForest) # random forest
set.seed(3517)
#setwd(".../PracticalMachineLearning/hw")
# Read data
MyData=read.csv("./pml-training.csv",header=T,na.strings=c(""," ","#DIV/0!"))
test=MyData[,c(8:11,37:49,60:68,84:86,102,113:124,140,151:160)]
dim(test)

# center and scale all the features but the response (last)
test_scaled=scale(test[1:ncol(test)-1],center=TRUE,scale=TRUE)
# compute the correlation matrix
corMat = cor(test_scaled)
# visualize the matrix, clustering features by correlation index.
corrplot(corMat, order = "hclust")

# Apply correlation filter at 0.70
highlyCor = findCorrelation(corMat, 0.70)
# Following predictors are higly correlated
highlyCor

test_filtered=test[,-highlyCor]
dim(test); dim(test_filtered)

inTrain=createDataPartition(test_filtered$classe,p=0.7,list=FALSE)
training=test_filtered[inTrain,]
testing=test_filtered[-inTrain,]
dim(training); dim(testing)

# Random Forest
modFit2=train(classe ~.,method="rf",data=training,trControl=trainControl(method="cv",number=4))
#modFit2=train(classe ~.,method="rf",data=training)
confusionMatrix(predict(modFit2,training),training$classe)
# Estimate out of sample error
confusionMatrix(predict(modFit2,testing),testing$classe)

# Problems
MyTest=read.csv("./pml-testing.csv",header=T)
pred=predict(modFit2,MyTest)
result=data.frame(MyTest$problem_id,pred)
names(result)=c("problem_id","predicted_classe")
result

pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./out/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}                                                         
pml_write_files(pred)


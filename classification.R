library(glmnet)
library(naniar)
library(dplyr)
library(tree)
library(randomForest)
library(gbm)
library(caret)

# read dataset
bank0 <- read.table("bank-additional-full.csv", 
                    sep=";", 
                    header = TRUE, 
                    stringsAsFactors = FALSE)
summary(bank0[,c("month", "day_of_week", "duration", "nr.employed")])

# take a subset that excludes month, day of week, duration and nr.employed
bank <- subset(bank0, 
               select = -c(month, day_of_week, duration, nr.employed))
summary(bank)

# replace unknown with NA
df <- bank %>% na_if("unknown") 
# drop NA
df <- na.omit(df)

# convert job into two levels: employed, unemployed
unemployed <-  c("student", "unemployed", "retired")
df <- df %>% mutate(job = replace(job, job %in% unemployed, "unemployed"))
df <- df %>% mutate(job = replace(job, job != "unemployed", "employed"))
count(df,job)

# convert marital into two levels: single, married
df <- df %>% mutate(marital = replace(marital, marital != "married", "single"))
count(df,marital)

# convert education into numeric orderd dummy variable
# conbine illiterate with basic.4y
df$education <- as.character(df$education)
df <- df %>% mutate(education = replace(education, 
                                        education=="illiterate", 
                                        as.character(0)))
education_level <- c("basic.4y", "basic.6y", "basic.9y", 
                     "high.school", "professional.course", "university.degree")
for (edu in education_level) {
  n <- which(education_level==edu)
  df <- df %>% mutate(education = replace(education, 
                                          education==edu, 
                                          as.character(n)))
}
df$education <- as.numeric(df$education)
count(df,education)

# split rows into train/test evenly
set.seed(1)
train <- sample( 1: nrow(df),nrow(df)/2) 
test <- -train

# convert variables into factors
df$job <- as.factor(df$job)
df$marital <- as.factor(df$marital)
df$default <- as.factor(df$default)
df$housing <- as.factor(df$housing)
df$loan <- as.factor(df$loan)
df$contact <- as.factor(df$contact)
df$poutcome <- as.factor(df$poutcome)
df$y <- as.factor(df$y)
summary(df)

train.df <- df[train,]
test.df <- df[test,]

# train a classification tree with gini criteria
gini.tree <- tree(y ~., data = train.df, split = "gini", 
                  control = tree.control(length(train), 
                                         mincut = 15, minsize = 30))
plot(gini.tree)
text(gini.tree, pretty = 0, cex = .5)

# train a classification tree with deviance criteria
dev.tree <- tree(y ~., data = train.df, split = "deviance", 
                 control = tree.control(length(train), mindev = 0.01))
plot(dev.tree)
text(dev.tree, pretty = 0, cex = .5)

# train a classification tree with deviance and mindev = 0.001
dev.tree2 <- tree(y ~., data = train.df, split = "deviance", 
                  control = tree.control(length(train), mindev = 0.001))
plot(dev.tree2)
text(dev.tree2, pretty = 0, cex = .5)

# train a random forest model
rf.tree <- randomForest(y ~., data = train.df, importance = TRUE)

# display variable importance
importance(rf.tree)
varImpPlot(rf.tree, main="Random Forest")

# train a boosted tree model
# convert Y into numeric 0 and 1
train.Y <- as.numeric(train.df$y) -1

# train boost tree with 3-fold cross validation
objControl <- trainControl(method='cv', number=3, 
                           returnResamp='none', 
                           classProbs = TRUE)

boost.tree <- train(train.df[,-17], train.df[,17], 
                    method='gbm', 
                    trControl=objControl,  
                    metric = "Accuracy",
                    verbose = FALSE)

# display variable importance
summary(boost.tree, las = 2, cBars = 16)

# prepare test set
Y.test <- test.df$y

# calculate prediction accuracy on test set

# simple classification tree with gini criteria
gini.pred <- predict(gini.tree, test.df, type="class")
r1 <- table(gini.pred, Y.test)
cat("Accuracy of Gini Tree:", (r1[1,1]+r1[2,2])/length(Y.test), "\n")

# simple classification tree with deviance criteria
dev.pred <- predict(dev.tree, test.df, type="class")
r2 <- table(dev.pred, Y.test)
cat("Accuracy of Deviance Tree:", (r2[1,1]+r2[2,2])/length(Y.test), "\n")

# random forest
rf.pred <- predict(rf.tree, test.df, type="class")
r3 <- table(rf.pred, Y.test)
cat("Accuracy of Random Forest:", (r3[1,1]+r3[2,2])/length(Y.test), "\n")

# boosted tree
boost.pred <- predict(boost.tree, test.df, n.trees = best.iter, type="raw")
r4 <- table(boost.pred, Y.test)
cat("Accuracy of Boosted Tree:", (r4[1,1]+r4[2,2])/length(Y.test), "\n")

# benchmark: if predcit every record as "no"
neg <- length(Y.test[Y.test=="no"])
cat("Accuracy Benchmark:", neg/length(Y.test), "\n")

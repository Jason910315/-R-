library(tensorflow)
library (plyr)
library(caret)
library(dplyr)
library(kknn)
library(e1071)
library(rpart)
library(rpart.plot)
library(magrittr)
install.packages("randomForest")
library(randomForest)

#建立空白frame儲存結果
result_frame <- data.frame(
  character(),
  double(),
  double(),
  double(),
  double()
)

#計算train評估結果

train_result <- function(cm,modelname,result_frame){
  tp <- cm[1, 1]
  tn <- cm[2, 2]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3)
  precision <- round(tp/(tp +fp),digits = 3)
  recall <- round(tp/(tp + fn), digits = 3)
  f1_score <- round(2/((1/precision)+(1/recall)),digits =3)
  
  local_frame <- data.frame(
    model = modelname,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
  local_frame <- rbind(result_frame,c(modelname,accuracy,precision,recall,f1_score))#frame合併
  
  return(local_frame)
}
#計算test評估結果

test_result <- function(cm,modelname,result_frame){
  tp <- cm[1, 1]
  tn <- cm[2, 2]
  fp <- cm[1, 2]
  fn <- cm[2, 1]
  accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3)
  precision <- round(tp/(tp +fp),digits = 3)
  recall <- round(tp/(tp + fn), digits = 3)
  f1_score <- round(2/((1/precision)+(1/recall)),digits =3)
  
  local_frame <- data.frame(
    model = modelname,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    f1_score = f1_score
  )
  local_frame <- rbind(result_frame,c(modelname,accuracy,precision,recall,f1_score))#frame合併
  
  return(local_frame)
}

#讀檔

data <- read.csv("C:/sound/data.csv")
set.seed(99)

#檢查遺漏值
data=na.exclude(data)

#資料處理(數值轉類別)
data$sound.files <- as.factor(data$sound.files)

#資料分割(80-20法則)
n <- 0.8*nrow(data)
test.index = sample(1:nrow(data),n)
train = data[test.index,]
test =  data[-test.index,]

#X為特徵 y為結果
train.x<-train[,c("meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","time.median","time.Q25","time.Q75","time.IQR","skew","kurt","sp.ent","time.ent","entropy","sfm","meandom","mindom","maxdom","dfrange","modindx","startdom","enddom","dfslope","meanpeakf")]
test.x <-test[,c("meanfreq","sd","freq.median","freq.Q25","freq.Q75","freq.IQR","time.median","time.Q25","time.Q75","time.IQR","skew","kurt","sp.ent","time.ent","entropy","sfm","meandom","mindom","maxdom","dfrange","modindx","startdom","enddom","dfslope","meanpeakf")]

train.y = train[,c("sound.files")]
test.y =test[,c("sound.files")]

#設定input 和 output
n <- names(data)
f <- as.formula(paste("sound.files ~", paste(n[!n %in% "sound.files"], collapse = " + ")))

#--------------------------------------------------------
#羅吉斯回歸(分類)
modelname <-"Logistic regression"
lm_model <- glm(formula= f ,data = train, family = "binomial")

#train混淆矩陣
train_result <- predict(lm_model, newdata = train.x, type = "response")
result_Approved <- ifelse(train_result > 0.5, 1, 0)
LR_train_cm <- table(train.y, result_Approved, dnn = c("實際", "預測"))#混淆矩陣

#test混淆矩陣
test_result <- predict(lm_model, newdata = test.x, type = "response")
result_Approved <- ifelse(test_result > 0.5, 1, 0)
LR_test_cm <- table(test.y, result_Approved, dnn = c("實際", "預測"))#混淆矩陣

#train結果
#拆解混淆矩陣
tp <- LR_train_cm[1, 1] 
tn <- LR_train_cm[2, 2]
fp <- LR_train_cm[1, 2]
fn <- LR_train_cm[2, 1]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的train結果的frame
train_result_l_frame <- rbind(result_frame,c(modelname,accuracy,precision,recall,f1_score))


#test結果

tp <- LR_test_cm[1, 1] 
tn <- LR_test_cm[2, 2]
fp <- LR_test_cm[1, 2]
fn <- LR_test_cm[2, 1]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的test結果的frame
test_result_l_frame <- rbind(result_frame,c(modelname,accuracy,precision,recall,f1_score))

lmImp <- varImp(lm_model, scale = FALSE)
lmImp

#--------------------------------------------------------

#SVM
modelname <-"SVM"
svm_model <- svm(f,data = train)

#train混淆矩陣
train_result <- predict(svm_model, newdata = train.x, type = "response")
svm_train_cm <- table(train.y, train_result, dnn = c("實際", "預測"))#混淆矩陣

#test混淆矩陣
test_result <- predict(svm_model, newdata = test.x, type = "response")
svm_test_cm <- table(test.y, test_result, dnn = c("實際", "預測"))#混淆矩陣

#train結果
#拆解混淆矩陣
tp <- svm_train_cm[1, 1] 
tn <- svm_train_cm[2, 2]
fp <- svm_train_cm[1, 2]
fn <- svm_train_cm[2, 1]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的train結果的frame
train_result_l_frame <- rbind(train_result_l_frame ,c(modelname,accuracy,precision,recall,f1_score))


#test結果

tp <- svm_test_cm[1, 1] 
tn <- svm_test_cm[2, 2]
fp <- svm_test_cm[1, 2]
fn <- svm_test_cm[2, 1]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的test結果的frame
test_result_l_frame <- rbind(test_result_l_frame,c(modelname,accuracy,precision,recall,f1_score))


#--------------------------------------------------------
#decision  Tree
modelname = "Decision Tree"
decision_model <- rpart(f,data =train,method = 'class')
rpart.plot(decision_model, extra = 109)

#train混淆矩陣
train_result <- predict(decision_model, newdata = train.x, type = "class") 
decition_train_cm <- table(train.y, train_result, dnn = c("實際", "預測"))#混淆矩陣

#test混淆矩陣
test_result <- predict(decision_model, newdata = test.x, type = "class")
decition_test_cm <- table(test.y, test_result, dnn = c("實際", "預測"))#混淆矩陣

#train結果
#拆解混淆矩陣
tp <- decition_train_cm[1, 1] 
tn <- decition_train_cm[2, 2]
fp <- decition_train_cm[1, 2]
fn <- decition_train_cm[2, 1]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的train結果的frame
train_result_l_frame <- rbind(train_result_l_frame,c(modelname,accuracy,precision,recall,f1_score))


#test結果

tp <- decition_test_cm[1, 1] 
tn <- decition_test_cm[2, 2]
fp <- decition_test_cm[1, 2]
fn <- decition_test_cm[2, 1]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的test結果的frame
test_result_l_frame <- rbind(test_result_l_frame,c(modelname,accuracy,precision,recall,f1_score))

decisionImp <- varImp(decision_model, scale = FALSE)
decisionImp
#--------------------------------------------------------
#Random Forest
modelname = "Random Forest"
rf_model <- randomForest(f,data =train,ntree=300)


#train混淆矩陣
train_result <- predict(rf_model, newdata = train.x, type = "class") 
rf_train_cm <- table(train.y, train_result, dnn = c("實際", "預測"))#混淆矩陣

#test混淆矩陣
test_result <- predict(rf_model, newdata = test.x, type = "class")
rf_test_cm <- table(test.y, test_result, dnn = c("實際", "預測"))#混淆矩陣

#train結果
#拆解混淆矩陣
tn <- rf_train_cm[2, 2] 
tp <- rf_train_cm[1, 1]
fn <- rf_train_cm[2, 1]
fp <- rf_train_cm[1, 2]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的train結果的frame
train_result_l_frame <- rbind(train_result_l_frame,c(modelname,accuracy,precision,recall,f1_score))


#test結果

tn <- rf_test_cm[2, 2] 
tp <- rf_test_cm[1, 1]
fn <- rf_test_cm[2, 1]
fp <- rf_test_cm[1, 2]
accuracy <- round((tp + tn)/(tp + tn + fp + fn),digits = 3) #計算accuracy
precision <- round(tp/(tp +fp),digits = 3) #計算precision
recall <- round(tp/(tp + fn), digits = 3) #計算recall
f1_score <- round(2/((1/precision)+(1/recall)),digits =3)#計算f1 score

#將結果合併為frame
local_frame <- data.frame(
  model = modelname,
  accuracy = accuracy,
  precision = precision,
  recall = recall,
  f1_score = f1_score
)
#將結果送進儲存的test結果的frame
test_result_l_frame <- rbind(test_result_l_frame,c(modelname,accuracy,precision,recall,f1_score))
#特徵重要性
rf_model$importance
varImpPlot(rf_model)

#重新命名frame
train_result_l_frame <- train_result_l_frame %>% rename(model = X.Logistic.regression.,
                                                        accuracy = X.1.,  #會因為數字不同 需要重新輸入
                                                        precision = X.1..1, #.會因為數字不同 需要重新輸入
                                                        recall = X.1..2, #會因為數字不同 需要重新輸入
                                                        f1_score = X.1..3) #會因為數字不同 需要重新輸入


test_result_l_frame <- test_result_l_frame %>% rename(model = X.Logistic.regression.,
                                                      accuracy = X.0.917., #會因為數字不同 需要重新輸入
                                                      precision = X.0.913., #會因為數字不同 需要重新輸入
                                                      recall = X.0.913..1, #會因為數字不同 需要重新輸入
                                                      f1_score = X.0.913..2) #會因為數字不同 需要重新輸入


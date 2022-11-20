library(caret)
library(psych)
library(corrplot)
library(dplyr)
library(ggplot2)
library(reshape2)
library(rpart)         # drzewa decyzyjne
library(rpart.plot)    # drzewa decyzyjne 2
library(MASS)
library(clusterSim)
library(class)
library(pROC)
library(kknn)
z

set.seed(297976)





#wczytanie danych
dane <- read.csv("winequalityN.csv")

raw_dane <- dane

head(dane)
str(dane)

#quality w skali 1-10, ale nastapi podzial na dobre wina >=7, oraz na slabe i srednie < 7

table(dane$quality)
table(is.na(dane))
sum(is.na(dane))

sapply(dane, function(x) sum(is.na(x)))
#wszystkie kolumny gdzie wystepuja braki maja dane numeryczne, zatem mozna zastapic te braki przez srednia

#zastapienie przez srednia
NA2mean <- function(x) replace(x, is.na(x), mean(x, na.rm = TRUE))
dane[,-c(1,13)] <- replace(dane[,-c(1,13)], TRUE, lapply(dane[,-c(1,13)], NA2mean))


sum(is.na(dane))

str(dane)

dane$type <- as.factor(dane$type)

dane$quality <- ifelse(dane$quality > 6, "decent", "average_bad")

str(dane)
dane$quality <- as.factor(dane$quality)


table(dane$quality)
str(dane)
######################################################


#wstepna analiza danych

#statystyki opisowe

#opis danych
#dlawszystkich
descr_stat <- subset(describe(dane[,c(-1,-13)]), select = -c(n, trimmed, min, max, range, se, mad, vars))

descr_stat


dane_decent <- dane[which(dane$quality == "decent"),]
dane_avg_bad <- dane[which(dane$quality == "average_bad"),]

#grupa decent

descr_stat_decent <- subset(describe(dane_decent[,c(-1,-13)]), select = -c(n, trimmed, min, max, range, se, mad, vars))

descr_stat_decent

#grupa avg and bad
descr_stat_avg_bad <- subset(describe(dane_avg_bad[,c(-1,-13)]), select = -c(n, trimmed, min, max, range, se, mad, vars))

descr_stat_avg_bad

#wykresy pomiedzy wszystkimi zmienny ilosciowymi
my_cols <- c("#00AFBB", "#FC4E07")  
pairs(dane[,c(-1,-13)], pch = 19,  cex = 0.5,
      col = my_cols[dane$quality],
      lower.panel=NULL)
#boxploty
dane_m <- melt(dane[,-1], id.vars = "quality")

p <- ggplot(data = dane_m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=quality),outlier.colour = "red", outlier.shape = 1, outlier.size = 1)
p + facet_wrap( ~ variable, scales="free")

#histogramy
nazwy <- colnames(dane)

par(mfrow=c(2,2))
for(i in 2:12){
  hist(dane[,i], main= nazwy[i], xlab = nazwy[i])
}


#związki miedzy zmiennymi objasniajacymi
#zwiazki miedzy zmienna objasniajaca i objasniana 


par(mfrow=c(1,1))

corrplot(cor(dane[,c(-1,-13)]), method = "number")

#analiza danych odstajacych
#na bazie 

#https://blog.usejournal.com/the-ultimate-r-guide-to-process-missing-or-outliers-in-dataset-65e2e59625c1
# outlier detection and normalizing
outlier_norm <- function(x){
  qntile <- quantile(x, probs=c(.25, .75))
  caps <- quantile(x, probs=c(.05, .95))
  H <- 1.5 * IQR(x, na.rm = T)
  x[x < (qntile[1] - H)] <- caps[1]
  x[x > (qntile[2] + H)] <- caps[2]
  return(x)
}

for(i in 2:12){
  dane[,i]=outlier_norm(dane[,i])
} 

#boxploty po usunieciu outlierow
dane_m <- melt(dane[,-1], id.vars = "quality")

p <- ggplot(data = dane_m, aes(x=variable, y=value)) + 
  geom_boxplot(aes(fill=quality),outlier.colour = "red", outlier.shape = 1, outlier.size = 1)
p + facet_wrap( ~ variable, scales="free")

#sum(is.na(dane))
#dane <- na.omit(dane)
#ew. imputacja brakow danych
#przeprowadzono za pomocą sredniej


str(dane)


#podzial na zbior treningowy i testowy



dane2 = dane


podzial <- createDataPartition(dane2$quality, 
                               p = 0.70,
                               list = FALSE)

dane_ucz  <- dane2[podzial,]
dane_test <- dane2[-podzial,]


tabela1 <- table(dane_ucz$quality)
tabela2 <-table(dane_test$quality)

decent_percent <- tabela1[2]/sum(tabela1)
decent_percent

avg_bad_percent <- tabela2[2]/sum(tabela2)
avg_bad_percent

#porownanie podzialu za pomoca boxplotu

dane$partition <- rownames(dane)

for(i in 1:nrow(dane)){
  dane$partition[i] <- ifelse(is.element(i,podzial), "uczacy", "test")
}

dane_m2 <- melt(dane[,-1], id.vars = c("quality","partition"))


ggplot(aes(y = value, x = partition, fill = quality), data = dane_m2) + geom_boxplot()  + facet_wrap( ~ variable, scales="free")





#dobrze podzielone jest, raczej git


#dla wybranej techniki ML przeprowadzic tuningowanie hiperparametrow


fitcontrol0 <- trainControl(method = "none", 
                            classProbs = TRUE)

# uzywam cv, gdzie k = number

fitControl1 <- trainControl(method = "cv", 
                            number = 5)

# uzywam cv z powtorzeniami 
# (dziele zestaw wg cv "repeats" razy)

fitControl2 <- trainControl(method = "repeatedcv",
                            number = 5, repeats = 3)


# Za budowe modelu odpowiada funkcja train(). 
# Jej konstrukcja jest nastepujaca:

# 1. Buduje model bez tuningowania hiperparametrow

knn_model <- train(quality ~ .,               # Y ~ X
                   data = dane_ucz,                   # zestaw danych
                   method = "knn",                  # metoda
                   preProc = "scale",  # wstepne przeksztalcenia danych
                   trControl = fitcontrol0,         # sposob walidacji (brak)
                   tuneGrid = data.frame(k = 7))    # wybieram parametry modelu



# Buduje prognoze i wyznaczam wyniki
knn_pr <- predict(knn_model, dane_test)
confusionMatrix(knn_pr,dane_test$quality)

#inne acc bo tutaj na testpwych sprawdzam xd

# 2. Dopuszczam tuningowanie

knn_model1 <- train(quality ~ .,               # Y ~ X
                    data = dane_ucz,                   # zestaw danych
                    method = "knn",                  # metoda
                    preProc = "scale",  # wstepne przeksztalcenia danych
                    trControl = fitControl2)   

knn_model1



knn_model2 <- train(quality ~ .,               
                    data = dane_ucz,                   
                    method = "knn",                  
                    preProc = "scale",  
                    trControl = trainControl(method = "cv", number = 10,
                                             classProbs = TRUE,                  # wyznaczaj prawdop.
                                             summaryFunction = twoClassSummary), # model dwuklasowy
                    metric = "ROC")

knn_model2
#najlepszy dla k=9


sasiedzi <- expand.grid(k = 3:17) 
# funkcja pozwala na budowe wybranych parametrow

knn_model4 <- train(quality ~ .,               
                    data = dane_ucz,                   
                    method = "knn",                  
                    preProc = "scale",  
                    trControl = trainControl(method = "cv", number = 10),
                    tuneGrid = sasiedzi)   # tutaj mam argument na "swoje" parametry

knn_model4
#tutaj k = 11


#czesc 2 

#wybrac i krotko opisac min 4 techniki uczenia statystycznego

#knn?
#kknn?



#naiwna bayesa?
#lda
#qda
#logit
#probit

#zbudowac modele oparte na w/w technikach oraz zbudowac prognoze na zbiorze testowym

#knn
?data.Normalization
str(dane_ucz)
dane_ucz_st <- dane_ucz
dane_ucz_st[,c(-1,-13)] <- data.Normalization(dane_ucz[,c(-1,-13)], type = "n1")
head(dane_ucz_st)
str(dane_ucz_st)

dane_test_st <- dane_test
dane_test_st[,c(-1,-13)] <- data.Normalization(dane_test[,c(-1,-13)], type = "n1")

#  W KNN KATEGORYCZNE TRZEBA ZAMIENIC NA NUMERYCZNE, BO NIE ZADZIALA INACZEJ :/
dane_ucz_st$type <- as.integer(dane_ucz_st$type)
str(dane_ucz_st)

dane_test_st$type <- as.integer(dane_test_st$type)
str(dane_test_st)


testowy <- dane_test_st
uczacy <- dane_ucz_st



ACC_KNN <- c()

for (i in 1:15){
  model_knn <- knn(uczacy[,-13], testowy[,-13], cl = uczacy$quality, k = i)
  summary(model_knn)
  
  
  table_knn<-table(predykcja = model_knn, prawdziwe = testowy$quality)
  table_knn
  
  #dokladnosc
  accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
  
  
  ACC_KNN[i]  <- accuracy(table_knn)
  ACC_KNN[i]
}

plot(ACC_KNN, type = "b")
#najlepszy dla k=3



ACC_KNN[3]

model_knn <- knn(uczacy[,-13], testowy[,-13], cl = uczacy$quality, k = 3)
ROC_KNN  <- roc(testowy$quality,as.ordered(model_knn),plot = TRUE)
AUC_KNN  <- auc(ROC_KNN)[1]
AUC_KNN

#kknn


?kknn
uczacy$type <- as.factor(uczacy$type)
uczacy$quality <- as.factor(uczacy$quality)

testowy$type <- as.factor(testowy$type)
testowy$quality <- as.factor(testowy$quality)

#w kknn juz wbudowane jest wybieranie najlepszego k, w tym wypadku wybralo k=7
model_kknn <- train.kknn(formula = quality~., data = uczacy, kmax = 10)
model_kknn
kknn_pred <- predict(model_kknn,testowy[,-13])

table_kknn<-table(predykcja = kknn_pred, prawdziwe = testowy$quality)
table_kknn

ACC_KKNN  <- accuracy(table_kknn)
ACC_KKNN


ROC_KKNN  <- roc(testowy$quality,as.ordered(kknn_pred),plot = TRUE)
AUC_KKNN  <- auc(ROC_KKNN)[1]
AUC_KKNN


#lda
lda<-lda(uczacy$quality~.,uczacy)
lda


prog.lda <- predict(lda,testowy)
prog.lda$class
head(prog.lda$posterior)
prog.lda.t<-predict(lda,testowy)
mac.tes<-table(prog.lda.t$class,testowy$quality)
ACC.tes<-accuracy(mac.tes)
ACC.tes


#qda
qda<-qda(uczacy$quality~., uczacy)

prog.qda.t<-predict(qda,testowy)
mac.qda.t<-table(prog.qda.t$class,testowy$quality)
ACC.qda.tes<-accuracy(mac.qda.t)
ACC.qda.tes


#logit
regresja_log <- glm(quality~., family =binomial, data=uczacy)

summary(regresja_log)

pred_log <- predict(regresja_log, testowy, type = "response")
pred_log

glm.pred <- ifelse(pred_log > 0.5, "decent", "average_bad")



mac_log <- table(glm.pred, testowy$quality)
mac_log


acc_log <- accuracy(mac_log)
acc_log

ROC_log  <- roc(testowy$quality,as.ordered(glm.pred),plot = TRUE)
AUC_log  <- auc(ROC_log)[1]
AUC_log

# #ROC dla log
# par(mfrow=c(1,1))
# #pred2 <- prediction(pred_log, testowy$quality)
# perf2 <- performance(pred_log,"tpr","fpr")
# plot(perf2,col="red")
# abline(0,1)

#probit

regresja_probit <- glm(quality~., family =binomial(link = "probit"), data=uczacy)

summary(regresja_probit)

pred_probit <- predict(regresja_probit, testowy, type = "response")
pred_probit

glm.pred_probit <- ifelse(pred_probit > 0.5, "decent", "average_bad")


mac_probit <- table(glm.pred_probit, testowy$quality)
mac_probit

acc_probit <- accuracy(mac_probit)
acc_probit

ROC_probit  <- roc(testowy$quality,as.ordered(glm.pred_probit),plot = TRUE)
AUC_probit  <- auc(ROC_probit)[1]
AUC_probit

#drzewo decyzyjne
drzewo <- rpart(quality~., data=uczacy, control=rpart.control(cp=0.01))

printcp(drzewo)   
plotcp(drzewo)

index<-which.min(drzewo$cptable[,"xerror"]) 
CP<-drzewo$cptable[index,"CP"]
drzewo.final<-prune(drzewo,cp=CP)

rpart.plot(drzewo.final,uniform=T,cex=0.5, type=4,clip.right.labs=T, branch = .6)

# prognoza drzewa decyzyjnego

drzewo.predict<-predict(drzewo.final,
                        newdata=testowy,
                        type="class")
drzewo.pr.u <- predict(drzewo.final, 
                       newdata = testowy,  
                       type = "prob")
tab_drzewo_prognoza<-table(testowy$quality,drzewo.predict)
ROC_drzewo.predict<-roc(testowy$quality,drzewo.pr.u[,1])
tab.drz.u  <- table(testowy$quality,drzewo.predict)
ACC.drz.u  <- accuracy(tab.drz.u)
ACC.drz.u 
ROC.drz.u  <- roc(testowy$quality, drzewo.pr.u[,1])
AUC.drz.u  <- auc(ROC.drz.u)[1]
plot.roc(ROC_drzewo.predict)

#wyniki






#wybrac najlepszy model i zinterpretowac uzyskane wyniki


#####################################
#Wykresy ROC 
par(pty = "s") 
#KNN
#ROC_KNN
roc.knn  <- roc(testowy$quality,as.ordered(model_knn),plot = TRUE,legacy.axes = TRUE)
#lda ROC 
roc.lda<-roc(testowy$quality, prog.lda.t$posterior[,1], plot = TRUE, legacy.axes = TRUE) 
#qda ROC 
roc.qda<-roc(testowy$quality, prog.qda.t$posterior[,1], plot = TRUE, legacy.axes = TRUE) 
#drzewo decyzyjne 
#ROC.drz.u 
roc.drzewo<-roc(testowy$quality, drzewo.pr.u[,1], plot = TRUE, legacy.axes = TRUE) 

#logit
ROC_log  <- roc(testowy$quality,as.ordered(glm.pred),plot = TRUE, legacy.axes = TRUE)

#probit
ROC_probit  <- roc(testowy$quality,as.ordered(glm.pred_probit),plot = TRUE, legacy.axes = TRUE)

# Krzywe ROC na jednym wykresie 
plot(roc.lda, col = "green") 
plot(roc.qda, add = TRUE, col="red") 
plot(roc.drzewo, add = TRUE, col="blue") 
plot(roc.knn, add = TRUE, col="black") 
plot(ROC_log, add = TRUE, col="orange")
plot(ROC_probit, add = TRUE, col="yellow")

AUC.lda <- auc(roc.lda)[1] 
AUC.lda 
AUC.qda <- auc(roc.qda)[1] 
AUC.qda 
AUC.drzewo <- auc(roc.drzewo)[1] 
AUC.drzewo 
AUC.knn <- auc(roc.knn)[1] 
AUC.knn
AUC.logit <- auc(ROC_log)[1] 
AUC.logit 
AUC.probit <- auc(ROC_probit)[1] 
AUC.probit



#macierz i wykresy z najlepszym ACC dla poszczegolnych modeli

wyniki <- as.data.frame(matrix(NA, 7,2))

modele_nazwy <- c("KNN", "KKNN","drzewo decyzyjne", "LDA", "QDA", "Regresja logitowa", "Regresja probitowa")
wyniki[,1] <- modele_nazwy


colnames(wyniki) <- c("Model", "Accuracy")

wyniki[1,2] <- ACC_KNN[3]
wyniki[2,2] <- ACC_KKNN
wyniki[3,2] <- ACC.drz.u
wyniki[4,2] <- ACC.tes
wyniki[5,2] <- ACC.qda.tes
wyniki[6,2] <- acc_log
wyniki[7,2] <- acc_probit
wyniki

ggplot(wyniki, aes(x=Model, y=Accuracy, fill=Model)) + 
  geom_bar(stat="identity")+
  scale_fill_brewer(palette="Dark2")


#minimalnie najlepszy jest KKNN
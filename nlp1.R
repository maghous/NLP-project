library(tm)
library(RWeka)
library(magrittr)
library(Matrix)
library(glmnet)
library(ROCR)
library(ggplot2)

df <-read.csv("Combined_News_DJIA.csv",stringsAsFactors = FALSE)
attach(df)
names(df)


df$Date <- as.Date(df$Date)

df$all <- paste(df$Top1, df$Top2, df$Top3, df$Top4, df$Top5, df$Top6,
                  df$Top7, df$Top8, df$Top9, df$Top10, df$Top11, df$Top12, 
                  df$Top13, df$Top14, df$Top15, df$Top16, df$Top17, df$Top18,
                  df$Top19, df$Top20, df$Top21, df$Top22, df$Top23, df$Top24,
                  df$Top25, sep=' <s> ')

df$all <- gsub('b"|b\'|\\\\|\\"', "", df$all)
df$all <- gsub("([<>])|[[:punct:]]", "\\1", df$all)

new_df <-df[,c("Date","Label","all")]

control <- list(
  removeNumbers = TRUE,
  tolower = TRUE,
  # exclude stopwords and headline tokens
  stopwords = c(stopwords(kind = 'SMART'), '<s>')
)

dtm <- Corpus(VectorSource(df$all)) %>% 
  DocumentTermMatrix(control=control)


split_index <- df$Date <= '2014-12-31'
ytrain <- as.factor(df$Label[split_index])
xtrain <- Matrix(as.matrix(dtm)[split_index, ], sparse=TRUE)

ytest <- as.factor(df$Label[!split_index])
xtest <- Matrix(as.matrix(dtm)[!split_index, ], sparse=TRUE)


#train model

glmnet.fit <- cv.glmnet(x=xtrain, y=ytrain, family='binomial', alpha=0)

preds <- predict(glmnet.fit, newx=xtest, type='response', s='lambda.min')

results <- data.frame(pred=preds, actual=ytest)

ggplot(results, aes(x=preds, color=actual)) + geom_density() + theme_light()


prediction <- prediction(preds, ytest)
perf <- performance(prediction, measure = "tpr", x.measure = "fpr")

auc <- performance(prediction, measure = "auc")
auc <- auc@y.values[[1]]

roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values))

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  geom_abline(slope=1, intercept=0, linetype='dashed') +
  ggtitle("ROC Curve") +
  ylab('True Positive Rate') +
  xlab('False Positive Rate')

options(mc.cores=1)

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))

control <- list(
  tokenize=BigramTokenizer,
  bounds = list(global = c(20, 500)))
  
dtm <- Corpus(VectorSource(df$all)) %>%
    tm_map(removeNumbers) %>%
    tm_map(stripWhitespace) %>%
    tm_map(content_transformer(tolower)) %>%
    DocumentTermMatrix(control=control)

split_index <- df$Date <= '2014-12-31'


ytrain <- as.factor(df$Label[split_index])
xtrain <- Matrix(as.matrix(dtm)[split_index, ], sparse=TRUE)

ytest <- as.factor(df$Label[!split_index])
xtest <- Matrix(as.matrix(dtm)[!split_index, ], sparse=TRUE)

glmnet.fit <- cv.glmnet(x=xtrain, y=ytrain, family='binomial', alpha=0)
preds1 <- predict(glmnet.fit, newx=xtest, type='response', s="lambda.min")
results <- data.frame(pred=preds1, actual=ytest)

ggplot(results, aes(x=preds1, color=actual)) + geom_density()


prediction <- prediction(preds1, ytest)
perf <- performance(prediction, measure = "tpr", x.measure = "fpr")

auc <- performance(prediction, measure = "auc")
auc <- auc@y.values[[1]]


roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values))

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  geom_abline(slope=1, intercept=0, linetype='dashed') +
  ggtitle("ROC Curve") +
  ylab('True Positive Rate') +
  xlab('False Positive Rate')

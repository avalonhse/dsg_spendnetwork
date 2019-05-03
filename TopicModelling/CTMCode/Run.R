
# change the path to the folder has data and result
setwd("~/Documents/TCD/Competitions/Turing")
setwd("~/TCD/Competitions/AlanTuring")

source("./CTMCode/modelCode.R")

crossSets <- c(1,2,3)
tol = 7e-4
k=200
i=0
suffix = paste("-V",i,"-K",k,sep = "")

seqK <- c(50,100,200)
  
trainFile = paste("./data/plan_a/training_sets/",i,".csv",sep = "")
trainTopic(trainFile,k,tol,suffix)

distance <- predictTopic(trainFile,suffix)
write.table(distance, sep = ",",
  file = paste("./result/prediction/validation/distance",suffix,".csv",sep = ""))

testFile = paste("./data/plan_a/testing_sets/",i,".csv",sep = "")
distance <- predictTopic(testFile,suffix)
write.table(distance, sep = ",",
  file = paste("./result/prediction/test/distance",suffix,".csv",sep = ""))

for (k in seqK) {
  suffix = paste("-V",i,"-K",k,sep = "")
  distance = read.table(sep = ",", 
  file = paste("./result/prediction/test/distance",suffix,".csv",sep=""))
  
  if (k==seqK[1]){
    data <- data.frame(cbind(1:ncol(distance)/ncol(distance),getPoints(distance,i)))
    colnames(data) <- c("X",paste("CTM",suffix))
  } else {
    data$temp <- getPoints(distance,i)
    colnames(data)[ncol(data)] <- paste("CTM",suffix)
  }

  library(ggplot2)
  library(reshape2)
  dd = melt(data, id=c("X"))
}
ggplot(dd) + geom_line(aes(x=X, y=value, colour=variable)) +
    scale_colour_manual(values=c("red","darkgreen","blue","darkorange","purple")) +
    xlab("Top Recommendation") + ylab("Ground truth") +
    coord_fixed() 


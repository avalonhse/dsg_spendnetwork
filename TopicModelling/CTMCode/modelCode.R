
source("./CTMCode/utils.R")
sourceCpp("./CTMCode/KLDivergence.cpp")#

library(stm)

trainTopic <- function(file,k,tol,suffix){
  dat =  read.csv(file, header=TRUE, sep=",", 
                  stringsAsFactors = FALSE,encoding = "UTF-8")
  
  dt <- dat  %>%  mutate_at("contract_text",filterString) 
  dt$newText <- dt$contract_text
  
  dt <- dt  %>%   mutate_at("contract_text",replaceString) %>% 
    mutate_at("newText",~gsub(pattern=";",replacement= " ",.))
  
  dt$FinalText <- paste(dt$contract_text,dt$newText)
  #############################################################
  # Compnay
  dt$temp <- paste(dt$company_name,dt$company_info)
  dt <- dt  %>%  mutate_at("temp",filterString) 
  dt$newText <- dt$temp
  
  dt <- dt  %>%  mutate_at("temp",replaceString) %>% 
    mutate_at("newText",~gsub(pattern=";",replacement= " ",.))
  
  dt$FinalText <- paste(dt$FinalText,dt$temp,dt$newText)
  
  text <- dt$FinalText
  
  processed <- textProcessor(text,lowercase = FALSE, removestopwords = FALSE, 
                             stem = FALSE, wordLengths = c(3, 25),
                             #removenumbers = FALSE, 
                             removepunctuation = FALSE,verbose = TRUE)
  out <- prepDocuments(processed$documents, processed$vocab,lower.thresh = 3)
  
  docs <- out$documents
  vocab <- out$vocab
  model <- stm(documents = out$documents, vocab = out$vocab,reportevery = 20,
               K = k, max.em.its = 80,emtol = tol, init.type = "Random")

  save(vocab,file=paste("./result/model/vocab",suffix,".RData",sep = ""))
  save(model, file=paste("./result/model/model",suffix,".RData",sep = ""))
}

predictTopic <- function(file,suffix){
  dat =  fread(file, header=TRUE, sep=",",stringsAsFactors = FALSE,encoding = "UTF-8")
  result <- processTestData(dat)
  
  suppliersPred <- topicPredictions(result[[2]]$companyFinal,suffix)
  row.names(suppliersPred) <- result[[2]]$company_name

  tenderPred <- topicPredictions(result[[1]]$contractFinal,suffix)
  row.names(tenderPred) <- result[[1]]$contract_id

  KL <- rcpp_KL_divergence(suppliersPred,tenderPred)
  #reversedKL <- t(rcpp_KL_divergence(tenderPred,suppliersPred))
  #distance <- (KL + reversedKL)/2
  distance <- KL
  
  row.names(distance) <- result[[2]]$company_name
  colnames(distance) <- result[[1]]$contract_id
  #write.table(distance, sep = ",",
  #            file = paste("./result/prediction/distance",suffix,".csv",sep = ""))
  distance
}
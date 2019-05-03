
library(data.table)
source("./CTMCode/DBTextFilter.R")
library(Rcpp)
sourceCpp('./CTMCode/string_split.cpp')

library(dplyr)
library(stringr)
library(tm)

processTestData <- function(dat){
  dt <- dat  %>%  mutate_at("contract_text",filterString) 
  dt$newText <- dt$contract_text
  dt <- dt  %>%   mutate_at("contract_text",replaceString) %>% 
    mutate_at("newText",~gsub(pattern=";",replacement= " ",.))
  
  dt$contractFinal <- paste(dt$contract_text,dt$newText)
  #############################################################
  dt$temp <- paste(dt$company_name,dt$company_text)
  dt <- dt  %>%  mutate_at("temp",filterString) 
  dt$newText <- dt$temp
  
  dt <- dt  %>%  mutate_at("temp",replaceString) %>% 
    mutate_at("newText",~gsub(pattern=";",replacement= " ",.))
  
  dt$companyFinal <- paste(dt$temp,dt$newText)
  
  text <- dt[,c(2,3,8,10)]
  contracts <- unique(text[,c(1,3)])
  companies <- unique(text[,c(2,4)])
  result <- list(contracts,companies)
}

convertNewCorpus <- function(new, old.vocab, verbose=TRUE) {
  #Starting calculations : for report purpose only
  new_vocab_size <- length(new$vocab)
  total_tokens <- sum(unlist(lapply(new$documents, function(x) sum(x[2,]))))
  wcts <- rep(0, length(old.vocab)) #create storage for word counts
  ############################################################################
  
  #drop will store documents that lose all of their words
  drop <- rep(0, length(new$documents))
  
  # map new vocab with old vocab
  id <- match(new$vocab, old.vocab, nomatch=0)
  
  for(i in 1:length(new$documents)) { # run a for over all docs
    doc <- new$documents[[i]]
    doc[1,] <- id[doc[1,]] #replace with index of old vocab
    
    doc <- doc[,which(doc[1,] != 0), drop=FALSE] #drop words not in old vocab
    # replace the old with new formatted
    new$documents[[i]] <- doc
    
    ########################################################
    # update the word counts => report only
    if(ncol(doc)==0) { drop[i] <- 1L # drop document i , do nothing
    } else { # if doc has words, add words of this doc to word counts
      wcts[doc[1,]] <- wcts[doc[1,]] + doc[2,]
    }
    
  }

  new$vocab <- old.vocab
  #################################################
  # REPORT RESULT
  #have to figure out words removed before copying new vocab
  words.removed <- new$vocab[id==0]
  prop.overlap <- c(sum(wcts!=0)/length(old.vocab), sum(wcts!=0)/new_vocab_size)

  cat(sprintf("There are %i documents with No Words which should be removed \n", length(which(drop==1))))
  cat(sprintf("Your new corpus now has %i documents, %i non-zero terms of %i total terms in the original set. \n%i terms from the new data did not match.\nThis means the new data contained %.1f%% of the old terms\nand the old data contained %.1f%% of the unique terms in the new data. \nYou have retained %i tokens of the %i tokens you started with (%.1f%%).", 
              length(new$documents), #documents
              sum(wcts>0), #nonzero terms
              length(new$vocab), #total terms
              length(words.removed),#unmatched terms
              round(prop.overlap[1]*100,3),
              round(prop.overlap[2]*100,3),
              sum(wcts),#total tokens
              total_tokens,#starting tokens
              round(100*sum(wcts)/total_tokens,3) #percent
  )) 
  #################################################
  
  new$documents <- lapply(new$documents, function(x) matrix(as.integer(x), nrow=2))
  return(list(documents=new$documents, drop = drop))
}

topicPredictions <- function(text,suffix){
  processed <- textProcessor(text,lowercase = FALSE, removestopwords = FALSE, 
                             stem = FALSE, wordLengths = c(0, Inf),
                             removepunctuation = FALSE,
                             verbose = TRUE)
  load(paste("./result/model/vocab",suffix,".RData",sep = ""))
  load(paste("./result/model/model",suffix,".RData",sep = ""))
  
  testCorp <- convertNewCorpus(new=processed, old.vocab=vocab)
  documents <- testCorp$documents[testCorp$drop==0]
  temp = fitNewDocuments(model = model, documents = documents)
  predictions = temp$theta
}

getPoints <- function(distance,i){
  ranksTable <- apply(distance,1,function(x){rank(x)})
  contracts =  fread(header=TRUE, sep=",",stringsAsFactors = FALSE,encoding = "UTF-8",
                     file = paste("./data/plan_a/testing_sets/",i,".csv",sep = ""))
  
  ranks <- unlist(lapply(1:length(contracts$company_name), function(x){
    ranksTable[x,eval(contracts$company_name[x])]
  }))
  points <- rep(0,nrow(contracts))
  for (i in 1:nrow(contracts)){
    points[i] = sum(ranks<=i)/ncol(distance)
    #points[i] = sum(ranks<=i)
  }
  points
}  

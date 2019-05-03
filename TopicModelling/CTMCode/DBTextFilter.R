
stopwordsHR <- paste0(c('will','you','to','and','our','your','<.+?>',
                        '//<br','chrome-extension-mutihighlight',
                        'pleas','includ','candid','title',
                        'apply','nan','need','look','exp'),collapse = "|")

library(tm)
stopwordsEnglish <-  paste0(stopwords("english"),collapse = "|")
stopwords <- paste0("\\b(",paste0(stopwordsHR,stopwordsEnglish,collapse = "|"), ")\\b") 

library(data.table)
library(tidyr)
library(dplyr)
library(stringr)

filterString <- function(x) {
  result <- gsub('[^ -~]|\\\\n|\\*|--|\\.', ' ', x=x) %>%
    gsub(pattern="\\\"|\\\\",replacement= ' ') %>% tolower %>% stemDocument %>%
    gsub(pattern=stopwords,replacement= ';') %>% str_trim %>% #gsub("\\s+", " ") %>%
    gsub(pattern='<.+?>',replacement= ';')  %>%
    gsub(pattern=' - |\\(| / | -|, |\\)|\\"|:|- |,|\\. |>|\\|', replacement= ';') %>%
    gsub(pattern='\\*|\\=|\\?|\\!', replacement= '') %>%
    gsub(pattern=' ; ', replacement= ';') %>%
    gsub(pattern='(;)\\1+',replacement= '\\1') %>%
    gsub(pattern="\\--|\\__|\\..|i;|i ",replacement= "") %>%
    gsub(pattern = '^\\;|\\;$', replacement = '') %>%
    gsub(pattern='; | ;', replacement= ';')
  return(result)
}

replaceString <- function(x){
  result <- gsub(pattern=" ",replacement= "_",x=x) %>% gsub(pattern=";",replacement= " ")  %>% gsub(pattern='^_',replacement= " ")
  return(result)
}

createNote <- function(data, colNames, colFilters){
  ###########################################################
  data <- data.table(unique(data %>% select("object_id" ,colNames)))
  
  data[, description := do.call(paste, c(.SD, sep = " . ")), .SDcols = colNames]
  
  #### apply filterString function
  df <- as_tibble(data)
  group <- rep(1:cl, length.out = nrow(df))
  df <- bind_cols(tibble(group), df)
  # Push data to cluster
  by_group <- df %>% partition(group, cluster = cluster)  %>%
    cluster_assign_value("stopwords", stopwords) %>% 
    cluster_copy(filterString)  %>% 
    cluster_library("dplyr") %>% cluster_library("tm") %>% cluster_library("stringr")
  # Filter the string
  df <- by_group  %>% 
    mutate_at(colFilters,filterString) %>% collect() %>% data.frame
  ###########################################################
  data <-  df[,c("object_id",colFilters)]
  colnames(data) <- paste(colnames(data),"1",sep = "_")
  newcolnames <- colnames(data)[-1] # these columns will replace ";" with "_"
  ###########################################################
  data <- cbind(df[,-1],data[,-1])
  if (length(newcolnames)==1) colnames(data)[ncol(data)] = newcolnames
  
  #### apply replaceString function
  group <- rep(1:cl, length.out = nrow(data))
  data <- bind_cols(tibble(group), data)
  by_group <- data %>% partition(group, cluster = cluster)  %>%
    cluster_copy(replaceString)  %>%
    cluster_library("dplyr") 
  
  df <- by_group  %>% 
    mutate_at(newcolnames,replaceString) %>% # these columns will replace ";" with "_"
    mutate_at(colFilters,~gsub(pattern=";",replacement= " ",.)) %>%
    collect() %>% data.frame
  ##########################################################
  
  df$note <- apply(df[,c(colNames,newcolnames)],1,paste,collapse = " " ) 
  df <- df[,c("object_id","description","note")]
  
  data <- data.table(df)[,.(
    note = paste0(note, collapse = " "),
    description  = paste0(description , collapse = " . ")
  ), by = "object_id" ]
}

# fileType: 1 & 2 for csv, 3 for rds
# 1: use fread
# 2: when fread fail, use read.csv
# 3: read rds
joinData <- function(df,fileName,idName,colNames,colFilters, fileType=1){
  # use data.table fread
  library(data.table)
  if (type==1){
    data <- fread(fileName, header=TRUE, sep="," , stringsAsFactors = FALSE,fill=TRUE)
  } else{ # use read.csv
    if (type==2){
      data <- read.csv(file=fileName, header=TRUE, sep="," ,
                       stringsAsFactors = FALSE,fill=TRUE)
      # use type = 3, read rds
    } else { data <- readRDS(file=fileName) }
  }
  
  names(data)[names(data) == idName] <- "object_id"
  data <- createNote(data,colNames,colFilters)
  
  df = merge(df, data, by="object_id" ,all.x = TRUE)
  df$note <- paste(df$note.x,df$note.y)
  df$description <- paste(df$description.x,df$description.y," . ")
  
  df <- df[,c(1,6,7)]
  ##################################################
  library(Rcpp)
  sourceCpp('./ctmglandore1/backend/pipeline/string_split.cpp')
  
  group <- rep(1:cl, length.out = nrow(df))
  df <- bind_cols(tibble(group), df)
  by_group <- df %>% partition(group, cluster = cluster) %>%
    cluster_copy(stringVec_split) 
  df <- by_group  %>% 
    mutate_at("note",stringVec_split(.,' ')) %>% 
    collect() %>% data.frame
  
  df
}
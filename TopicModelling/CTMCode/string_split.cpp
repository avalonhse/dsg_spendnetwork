#include <Rcpp.h>
using namespace Rcpp;

#include <sstream>
#include <string>
#include <vector>

// [[Rcpp::plugins(cpp11)]]

#include <iostream>
using namespace std;

typedef vector<string> stringList;
typedef vector<int> numList;

//split function to split line with desired deliminator
stringList split(const string &s, char delim){
  stringList result;
  stringstream ss(s);
  string item;
  while (getline(ss, item, delim)) result.push_back(item);
  return result;
}

// custom format for after tokenized
string formatString(string& x){
  string UserInput = x;
  if (UserInput.length()<2||UserInput.length()>40)
    //if (!isdigit(UserInput[0])) 
    return "";
  std::string chars = "_/@-";
  if (chars.find(UserInput[0]) != std::string::npos) 
    UserInput = UserInput.substr(1,UserInput.length());
  if (chars.find(UserInput[0]) != std::string::npos) 
    UserInput = UserInput.substr(1,UserInput.length());
  
  if (chars.find(UserInput[UserInput.length()]) != std::string::npos) 
    UserInput = UserInput.substr(1,UserInput.length());
  
  return UserInput;
}


//1. tokenized a list of Strings by a separator
//2. format each token by custom format function
//3. join the string back
//[[Rcpp::export]]
stringList stringVec_split(stringList x, string sep)
{
  int size = x.size();
  stringList result(size);
  char delim = sep[0];
  
  // for each string of the list
  for (int i = 0; i < size; i++)
  {
    stringList splittedString = split(x[i],delim); // split the string into words
    int splittedSize = splittedString.size();

    string str = formatString(splittedString[0]);
    for (int j = 1 ; j < splittedSize ; j++)
    {
      str = str + delim + formatString(splittedString[j]);
    }
    result[i] = str;
  }
  return result;
}

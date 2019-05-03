// https://stackoverflow.com/questions/30791410/how-to-call-user-defined-function-in-rcppparallel

// [[Rcpp::depends(RcppArmadillo)]]
#include "RcppArmadillo.h"

#include <cmath>
#include <algorithm>

#include <Rcpp.h>
using namespace Rcpp;

// generic function for kl_divergence
template <typename InputIterator1, typename InputIterator2>
inline double kl_divergence(InputIterator1 begin1, InputIterator1 end1, 
                            InputIterator2 begin2) {
  // value to return
  double rval = 0;
  
  // set iterators to beginning of ranges
  InputIterator1 it1 = begin1;
  InputIterator2 it2 = begin2;
  
  // for each input item
  while (it1 != end1) {
    
    // take the value and increment the iterator
    double d1 = *it1++;
    double d2 = *it2++;
    
    // accumulate if appropirate
    if (d1 > 0 && d2 > 0)
      rval += std::log(d1 / d2) * d1;
  }
  return rval;  
}

// [[Rcpp::depends(RcppParallel)]]
#include <RcppParallel.h>
using namespace RcppParallel;

struct KLDivergence : public Worker
{
  const arma::mat MAT1;  // input matrix to read from
  const arma::mat MAT2;  // input matrix to read from
  RMatrix<double> mat_ouput;       // output matrix to write to
  
  // Convert global input/output into RMatrix/RVector type
  KLDivergence(const arma::mat& mat_input1,
               const arma::mat& mat_input2,
               NumericMatrix& matrix_output)
    : MAT1(mat_input1),MAT2(mat_input2), 
      mat_ouput(matrix_output) {}
  
  void operator()(std::size_t begin_mat2, std::size_t end_mat2)
  {
    for (std::size_t j = begin_mat2; j < end_mat2; j++){
      for (std::size_t i = 0; i < MAT1.n_rows; i++){
        // rows we will operate on
        arma::rowvec row1 = MAT1.row(i);          // get the row of arma matrix
        arma::rowvec row2 = MAT2.row(j);
        
        // calculate divergences & write to output matrix
        mat_ouput(i,j) = kl_divergence(row1.begin(), row1.end(), 
                  row2.begin());
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix rcpp_KL_divergence(const arma::mat& mat_input1,
                                 const arma::mat& mat_input2,
                                 const double N_cores=1){

  // allocate the matrix we will return
  NumericMatrix mat_output(mat_input1.n_rows, mat_input2.n_rows);
  
  // create the worker
  KLDivergence entropy(mat_input1,mat_input2, mat_output);

  // call it with parallelFor
  //parallelFor(0, mat_input2.n_rows, entropy, Ncores);  
  parallelFor(0, mat_input2.n_rows, entropy, mat_input2.n_rows/N_cores);  
  
  return mat_output;
}

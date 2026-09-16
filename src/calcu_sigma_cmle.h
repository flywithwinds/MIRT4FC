#ifndef MIRT4FC_CALCU_SIGMA_CMLE_H
#define MIRT4FC_CALCU_SIGMA_CMLE_H

#include <RcppArmadillo.h>

double obj_func_cpp(const arma::mat& sigma, const arma::mat& sigma_hat);
arma::mat calcu_sigma_cmle_cpp(const arma::mat& theta, double tol = 1e-5);

#endif
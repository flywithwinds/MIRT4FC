#include "calcu_sigma_cmle.h"

// [[Rcpp::depends(RcppArmadillo)]]

double obj_func_cpp(const arma::mat& sigma, const arma::mat& sigma_hat) {
  arma::mat sigma_inv = arma::inv(sigma);
  return arma::accu(sigma_inv % sigma_hat) + std::log(arma::det(sigma));
}

// [[Rcpp::export]]
arma::mat calcu_sigma_cmle_cpp(const arma::mat& theta, double tol) {
  const arma::uword n = theta.n_rows;
  arma::mat sigma_hat = theta.t() * theta / n;
  arma::mat sigma0 = arma::cor(theta);
  arma::mat sigma1 = sigma0;
  arma::mat tmp;
  double eps = 1.0;

  while (eps > tol) {
    double step = 1.0;
    tmp = arma::inv(sigma0);
    sigma1 = sigma0 - step * (-tmp * sigma_hat * tmp + tmp);
    sigma1.diag().ones();

    while (obj_func_cpp(sigma0, sigma_hat) < obj_func_cpp(sigma1, sigma_hat) ||
           arma::min(arma::eig_sym(sigma1)) < 0) {
      step *= 0.5;
      sigma1 = sigma0 - step * (-tmp * sigma_hat * tmp + tmp);
      sigma1.diag().ones();
    }

    eps = obj_func_cpp(sigma0, sigma_hat) - obj_func_cpp(sigma1, sigma_hat);
    sigma0 = sigma1;
  }

  return sigma0;
}
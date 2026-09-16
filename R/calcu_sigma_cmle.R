obj_func <- function(sigma, sigma_hat) {
  sigma_inv <- solve(sigma)
  sum(sigma_inv * sigma_hat) + as.numeric(determinant(sigma, logarithm = FALSE)$modulus)
}

calcu_sigma_cmle <- function(theta, tol = 1e-5) {
  theta <- as.matrix(theta)
  n <- nrow(theta)
  sigma_hat <- crossprod(theta) / n
  sigma0 <- stats::cor(theta)
  sigma1 <- sigma0
  eps <- 1

  while (eps > tol) {
    step <- 1
    sigma_inv <- solve(sigma0)
    gradient <- -sigma_inv %*% sigma_hat %*% sigma_inv + sigma_inv
    sigma1 <- sigma0 - step * gradient
    diag(sigma1) <- 1

    while (obj_func(sigma0, sigma_hat) < obj_func(sigma1, sigma_hat) ||
           min(eigen(sigma1, symmetric = TRUE, only.values = TRUE)$values) < 0) {
      step <- step * 0.5
      sigma1 <- sigma0 - step * gradient
      diag(sigma1) <- 1
    }

    eps <- obj_func(sigma0, sigma_hat) - obj_func(sigma1, sigma_hat)
    sigma0 <- sigma1
  }

  sigma0
}
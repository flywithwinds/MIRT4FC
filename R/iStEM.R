#'iStEM
#'
#' @param Y          A # of subjects * # of blocks matrix; item responses.
#' @param BID        A # of statements * 3 matrix; item information,
#'     columns are "Block", "Item" and "Dimensions".
#' @param positive   A logical vector; indicating whether each statement is
#'     positive directional or not.
#' @param blocksize  A number; block size of FC(2/3/4).
#' @param res        A string; response format('pick'/'rank'/'mole'),
#'     pick-2(blocksize=2)/rank-2/mole-2 are equivalent, rank-3/mole-3 are equivalent.
#' @param M          A number; # of batch.
#' @param B          A number; # of iterations in each batch.
#' @param model      A string; FC model('2PL').
#' @param SE         A string; Different methods of standard error estimation('MCMC'/
#'      'FDM'/'XPD'/'CDM'/'RES'/'Louis'/'Sandwich'/'complete').
#' @param sigma      A # of dimensions * # of dimensions matrix; initial sigma
#'     parameters.
#' @param theta      A # of subjects * # of dimensions matrix; initial theta parameters.
#' @param fix.sigma  Logical; TRUE if sigma is estimated.
#' @param burnin.maxitr   A number; max burn-in allowed.
#' @param maxitr     A number; max iterations allowed.
#' @param eps1       A number; stability criteria.
#' @param eps2       A number; convergence criterion.
#' @param frac1      A number; cutoffs for calculating Geweke z.
#' @param frac2      A number; cutoffs for calculating Geweke z.
#' @param cores      A number; number of cores.
#' @param h          A number; The perturbation constant of the differential method.
#' @param fixed.blocks A vector of block indices whose item parameters (a, d) are
#'     kept unchanged during estimation; only the parameters of the other blocks
#'     are updated. Applies to all blocksize (2/3/4) and response formats.
#' @param fixed.a A matrix (blocksize x length(fixed.blocks)) of initial a values
#'     used as the fixed values for the fixed blocks. If NULL, the internal initial
#'     values are used.
#' @param fixed.d A matrix (blocksize x length(fixed.blocks)) of initial d values
#'     used as the fixed values for the fixed blocks (only rows 1:2 are used; the
#'     third row is recomputed as -(d1+d2)). If NULL, the internal initial values
#'     are used.
#'
#' @return A list of data; including estimated a, d, and sigm parameters,
#'    total batch number, final chain size, burn-in size, time.
#'
#' @export iStEM
#' @examples
#' \donttest{
#' library(doParallel)
#' library(foreach)
#' D <- 6
#' nitem.per.dim <- 10
#' nblock <- D * nitem.per.dim / 3
#' set.seed(123456)
#' item.par <- data.frame(a=seq_len(D*nitem.per.dim))
#' item.par <- within(item.par,{
#'   a <- runif(D*nitem.per.dim,0.7,3)
#'   b <- rnorm(D*nitem.per.dim)
#'   d <- a*b
#' })
#' BID <- data.frame(Block=rep(1:nblock,each=3),
#'                   Item=rep(1:3,nblock),
#'                   Dim=c(combn(D,3)[,sample(choose(D,3),nblock,replace = TRUE)]))
#' item.par$d <- c(t(aggregate(item.par$d,by=list(BID$Block),function(x)x-mean(x))[,-1]))
#' N <- 1000
#' v <- matrix(0.5,D,D)
#' diag(v) <- 1
#' theta <- mvnfast::rmvn(N,seq(-1,1,length.out = D),sigma = v)
#' Y <- data.sim(item.par,theta,BID,blocksize=3,res='rank',model='2PL')
#' x <- iStEM(Y,BID,maxitr = 100,blocksize=3,res='rank',model='2PL',fix.sigma = TRUE,cores=1)
#'
#' # Fix blocks 1 and 2 and set their (a, d) to specified initial values
#' fixed.blocks <- c(1, 2)
#' fa <- matrix(c(1.5,1.2,0.8, 2.0,1.0,1.3), nrow=3, ncol=2)
#' fd <- matrix(c(-0.5,0.3,0.2, -0.1,-0.4,0.5), nrow=3, ncol=2)
#' x2 <- iStEM(Y, BID, maxitr = 30, blocksize=3, res='rank', model='2PL',
#'             fix.sigma = TRUE, cores=1,
#'             fixed.blocks=fixed.blocks, fixed.a=fa, fixed.d=fd)
#' }
#' @importFrom stats cor optim rnorm pnorm sd
#' @importFrom utils install.packages installed.packages
#' @importFrom doParallel registerDoParallel
#' @importFrom foreach foreach registerDoSEQ %dopar%
#' @importFrom coda mcmc geweke.diag
#' @importFrom Matrix forceSymmetric
#' @importFrom parallel detectCores makeCluster stopCluster
iStEM <- function(Y,BID,positive=rep(TRUE,nrow(BID)),blocksize=3,res='rank',M=10,B=20,model='2PL',
                  SE='Louis',sigma=NULL,theta=NULL,fix.sigma = FALSE,burnin.maxitr=40,
                 maxitr=500,eps1=1.5,eps2=0.4,frac1=.2,frac2=.5,cores=NULL,h=NULL,fixed.blocks=NULL,fixed.a=NULL,fixed.d=NULL){
  if(model=='2PL'){
    x <- iStEM_2PL(Y=Y,BID=BID,positive=positive,M=M,B=B,SE=SE,blocksize=blocksize,res=res,
                sigma=sigma,theta=theta,fix.sigma = fix.sigma,burnin.maxitr=burnin.maxitr,
                maxitr=maxitr,eps1=eps1,eps2=eps2,frac1=frac1,frac2=frac2,cores=cores,model=model,h=h,
                fixed.blocks=fixed.blocks,fixed.a=fixed.a,fixed.d=fixed.d)
  }
  return(x)
}


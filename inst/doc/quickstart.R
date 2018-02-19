params <-
structure(list(EVAL = FALSE), .Names = "EVAL")

## ---- SETTINGS-knitr, include=FALSE--------------------------------------
stopifnot(require(knitr))
knitr::opts_chunk$set(
    fig.width = 7, fig.height = 5
)

## ---- message=F, warning=F-----------------------------------------------
library(rstanarm)
library(projpred)
library(ggplot2)
library(bayesplot)
#options(mc.cores = parallel::detectCores())

## ------------------------------------------------------------------------
data('df_gaussian', package = 'projpred')

## ---- results='hide', message=F, warning=F, cache=T----------------------
n <- nrow(df_gaussian$x) # 100
D <- ncol(df_gaussian$x) # 20
p0 <- 5 # prior guess for the number of relevant variables
tau0 <- p0/(D-p0) * 1/sqrt(n) # scale for tau (notice that stan_glm will automatically scale this by sigma)
prior_coeff <- hs(global_scale = tau0, slab_scale = 1) # regularized horseshoe prior
fit <- stan_glm(y ~ x, family=gaussian(), data=df_gaussian, prior=prior_coeff,
                # to make this vignette build fast, we use only 2 chains and
                # 500 draws. In practice, more conservative values, eg. 4 chains
                # and 2000 draws might be required for reliable inference.
                seed=1, chains=2, iter=500) 


## ------------------------------------------------------------------------
fit <- varsel(fit, method='forward')
fit$varsel$vind

## ------------------------------------------------------------------------
# plot predictive performance on training data 
varsel_plot(fit, stats=c('mlpd', 'mse'))

## ---- results='hide', warning=F, cache=T---------------------------------
fit_cv <- cv_varsel(fit, method='forward')

## ------------------------------------------------------------------------
# model size suggested by the program
fit_cv$varsel$ssize

## ------------------------------------------------------------------------
# plot the validation results
varsel_plot(fit_cv, stats = c('mlpd', 'mse'), deltas=T)

## ------------------------------------------------------------------------
# Visualise the three most relevant variables in the full model
mcmc_areas(as.matrix(fit), 
           pars = c('(Intercept)', names(fit$varsel$vind[1:3]), 'sigma')) + coord_cartesian(xlim = c(-2, 2))

## ------------------------------------------------------------------------
# Visualise the projected three most relevant variables
proj <- project(fit, nv = 3, ns = 500)
mcmc_areas(as.matrix(proj)) + coord_cartesian(xlim = c(-2, 2))

## ------------------------------------------------------------------------
pred <- proj_linpred(fit, xnew=df_gaussian$x, ynew=df_gaussian$y, nv = 6, integrated = TRUE)

## ------------------------------------------------------------------------
ggplot() +
  geom_point(aes(x=pred$pred,y=df_gaussian$y)) +
  geom_abline(slope = 1, color='red') +
  labs(x = 'prediction', y = 'y')

## ------------------------------------------------------------------------
data('df_binom', package = 'projpred')

## ---- results='hide', message=F, warning=F, cache=T----------------------
# fit the full model
n <- nrow(df_binom$x)
D <- ncol(df_binom$x)
p0 <- 5 # prior guess for the number of relevant variables
sigma <- 2 # approximate plug-in value for observation information (Piironen and Vehtari, 2017b)
tau0 <- p0/(D-p0) * sigma/sqrt(n)
prior_coeff <- hs(global_scale = tau0, slab_scale = 1)
fit <- stan_glm(y ~ x, family=binomial(), data=df_binom, prior=prior_coeff,
                seed=1, chains=2, iter=500)

## ------------------------------------------------------------------------
fit <- varsel(fit, method='forward')
fit$varsel$vind

## ------------------------------------------------------------------------
varsel_plot(fit, stats=c('mlpd', 'pctcorr'), deltas=F)

## ---- results='hide', warning=F, cache=T---------------------------------
fit_cv <- cv_varsel(fit, method='forward')

## ------------------------------------------------------------------------
# model size suggested by the program
fit_cv$varsel$ssize

## ------------------------------------------------------------------------
# plot the validation results
varsel_plot(fit_cv, stats=c('mlpd', 'pctcorr'), deltas=T)

## ------------------------------------------------------------------------
# evaluate the predictive distribution in a 2d grid
ng <- 20
x1g <- seq(-4,4,len=ng)
x2g <- seq(-4,4,len=ng)
xnew <- cbind( rep(x1g,each=ng), rep(x2g,ng) )
vind <- fit$varsel$vind[1:2]
pr <- proj_linpred(fit, xnew, vind=vind, transform=T, integrated=T)

# visualize the contours showing the predicted class probabilities
pp <- ggplot()
pp <- pp + geom_contour(aes(x=xnew[,1],y=xnew[,2], z=pr, colour=..level..))
pp <- pp + scale_colour_gradient(low = "red", high = "green")
pp <- pp + geom_point(aes(x=df_binom$x[,vind[1]],y=df_binom$x[,vind[2]]), color=df_binom$y+2)
pp <- pp + xlab(sprintf('Feature %d',vind[1])) + ylab(sprintf('Feature %d',vind[2]))
pp


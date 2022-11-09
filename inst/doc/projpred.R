params <-
list(EVAL = TRUE)

## ---- SETTINGS-knitr, include=FALSE-------------------------------------------
stopifnot(require(knitr))
knitr::opts_chunk$set(
  dev = "png",
  dpi = 150,
  fig.asp = 0.618,
  fig.width = 5,
  out.width = "60%",
  fig.align = "center",
  comment = NA,
  eval = if (isTRUE(exists("params"))) params$EVAL else FALSE,
  message = FALSE,
  warning = FALSE
)

## -----------------------------------------------------------------------------
data("df_gaussian", package = "projpred")
dat_gauss <- data.frame(y = df_gaussian$y, df_gaussian$x)

## -----------------------------------------------------------------------------
library(rstanarm)

## -----------------------------------------------------------------------------
# Number of regression coefficients:
( D <- sum(grepl("^X", names(dat_gauss))) )

## -----------------------------------------------------------------------------
# Prior guess for the number of relevant (i.e., non-zero) regression
# coefficients:
p0 <- 5
# Number of observations:
N <- nrow(dat_gauss)
# Hyperprior scale for tau, the global shrinkage parameter (note that for the
# Gaussian family, 'rstanarm' will automatically scale this by the residual
# standard deviation):
tau0 <- p0 / (D - p0) * 1 / sqrt(N)

## -----------------------------------------------------------------------------
# Set this manually if desired:
ncores <- parallel::detectCores(logical = FALSE)
### Only for technical reasons in this vignette (you can omit this when running
### the code yourself):
ncores <- min(ncores, 2L)
###
options(mc.cores = ncores)
refm_fit <- stan_glm(
  y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14 +
    X15 + X16 + X17 + X18 + X19 + X20,
  family = gaussian(),
  data = dat_gauss,
  prior = hs(global_scale = tau0),
  ### Only for the sake of speed (not recommended in general):
  chains = 2, iter = 500,
  ###
  seed = 2052109, QR = TRUE, refresh = 0
)

## -----------------------------------------------------------------------------
library(projpred)

## ---- results='hide'----------------------------------------------------------
cvvs <- cv_varsel(
  refm_fit,
  ### Only for the sake of speed (not recommended in general):
  validate_search = FALSE,
  nclusters_pred = 20,
  ###
  nterms_max = 9,
  seed = 411183
)

## ---- fig.asp=1.5 * 0.618-----------------------------------------------------
plot(cvvs, stats = c("elpd", "rmse"), deltas = TRUE, seed = 54548)

## -----------------------------------------------------------------------------
modsize_decided <- 6

## -----------------------------------------------------------------------------
suggest_size(cvvs)

## -----------------------------------------------------------------------------
cvvs
### Alternative modifying the number of printed decimal places:
# print(cvvs, digits = 2)
### 

## -----------------------------------------------------------------------------
( soltrms <- solution_terms(cvvs) )

## -----------------------------------------------------------------------------
( soltrms_final <- head(soltrms, modsize_decided) )

## -----------------------------------------------------------------------------
prj <- project(refm_fit, solution_terms = soltrms_final)

## -----------------------------------------------------------------------------
prj_mat <- as.matrix(prj)

## -----------------------------------------------------------------------------
library(posterior)
prj_drws <- as_draws_matrix(prj_mat)
# In the following call, as.data.frame() is used only because pkgdown
# versions > 1.6.1 don't print the tibble correctly.
as.data.frame(summarize_draws(
  prj_drws,
  "median", "mad", function(x) quantile(x, probs = c(0.025, 0.975))
))

## -----------------------------------------------------------------------------
library(bayesplot)
bayesplot_theme_set(ggplot2::theme_bw())
mcmc_intervals(prj_mat) +
  ggplot2::coord_cartesian(xlim = c(-1.5, 1.6))

## -----------------------------------------------------------------------------
refm_mat <- as.matrix(refm_fit)
mcmc_intervals(refm_mat, pars = colnames(prj_mat)) +
  ggplot2::coord_cartesian(xlim = c(-1.5, 1.6))

## -----------------------------------------------------------------------------
( dat_gauss_new <- setNames(
  as.data.frame(replicate(length(soltrms_final), c(-1, 0, 1))),
  soltrms_final
) )

## -----------------------------------------------------------------------------
prj_linpred <- proj_linpred(prj, newdata = dat_gauss_new, integrated = TRUE)
cbind(dat_gauss_new, linpred = as.vector(prj_linpred$pred))

## -----------------------------------------------------------------------------
prj_predict <- proj_predict(prj, .seed = 762805)
# Using the 'bayesplot' package:
ppc_dens_overlay(y = dat_gauss$y, yrep = prj_predict, alpha = 0.9, bw = "SJ")

## ---- eval=FALSE--------------------------------------------------------------
#  data("VerbAgg", package = "lme4")
#  refm_fit <- stan_glmer(
#    r2 ~ btype + situ + mode + (btype + situ + mode | id),
#    family = binomial(),
#    data = VerbAgg,
#    seed = 82616169, QR = TRUE, refresh = 0
#  )

## ---- eval=FALSE--------------------------------------------------------------
#  data("lasrosas.corn", package = "agridat")
#  # Convert `year` to a `factor` (this could also be solved by using
#  # `factor(year)` in the formula, but we avoid that here to put more emphasis on
#  # the demonstration of the smooth term):
#  lasrosas.corn$year <- as.factor(lasrosas.corn$year)
#  refm_fit <- stan_gamm4(
#    yield ~ year + topo + t2(nitro, bv),
#    family = gaussian(),
#    data = lasrosas.corn,
#    seed = 4919670, QR = TRUE, refresh = 0
#  )

## ---- eval=FALSE--------------------------------------------------------------
#  data("gumpertz.pepper", package = "agridat")
#  refm_fit <- stan_gamm4(
#    disease ~ field + leaf + s(water),
#    random = ~ (1 | row) + (1 | quadrat),
#    family = binomial(),
#    data = gumpertz.pepper,
#    seed = 14209013, QR = TRUE, refresh = 0
#  )


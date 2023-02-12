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
# Number of observations in the training dataset (= number of observations in
# the test dataset):
N <- 71
sim_poiss <- function(nobs = 2 * N, ncon = 10, ngrpPL = 4, nnoise = 39) {
  # Regression coefficients for continuous predictors:
  coefs_con <- rnorm(ncon)
  # Continuous predictors:
  dat_sim <- matrix(rnorm(nobs * ncon), ncol = ncon)
  # Start linear predictor:
  linpred <- 2.1 + dat_sim %*% coefs_con
  
  # Population-level (PL) categorical predictor:
  dat_sim <- data.frame(
    x = dat_sim,
    grpPL = gl(n = ngrpPL, k = nobs %/% ngrpPL, length = nobs,
               labels = paste0("grpPL", seq_len(ngrpPL)))
  )
  # Regression coefficients for the PL categorical predictor:
  coefs_catPL <- rnorm(ngrpPL)
  # Continue linear predictor:
  linpred <- linpred + coefs_catPL[dat_sim$grpPL]
  
  # Noise predictors:
  dat_sim <- data.frame(
    dat_sim,
    xn = matrix(rnorm(nobs * nnoise), ncol = nnoise)
  )
  
  # Poisson response, using the log link (i.e., exp() as inverse link):
  dat_sim$y <- rpois(nobs, lambda = exp(linpred))
  # Shuffle order of observations:
  dat_sim <- dat_sim[sample.int(nobs), , drop = FALSE]
  # Drop the shuffled original row names:
  rownames(dat_sim) <- NULL
  return(dat_sim)
}
set.seed(300417)
dat_poiss <- sim_poiss()
dat_poiss_train <- dat_poiss[1:N, , drop = FALSE]
dat_poiss_test <- dat_poiss[(N + 1):nrow(dat_poiss), , drop = FALSE]

## -----------------------------------------------------------------------------
library(rstanarm)
# Number of regression coefficients:
( D <- sum(grepl("^x|^grpPL", names(dat_poiss_train))) )
# Prior guess for the number of relevant (i.e., non-zero) regression
# coefficients:
p0 <- 10
# Prior guess for the overall magnitude of the response values, see Table 1 of
# Piironen and Vehtari (2017, DOI: 10.1214/17-EJS1337SI):
mu_prior <- 100
# Hyperprior scale for tau, the global shrinkage parameter:
tau0 <- p0 / (D - p0) / sqrt(mu_prior) / sqrt(N)
# Set this manually if desired:
ncores <- parallel::detectCores(logical = FALSE)
### Only for technical reasons in this vignette (you can omit this when running
### the code yourself):
ncores <- min(ncores, 2L)
###
options(mc.cores = ncores)
refm_fml <- as.formula(paste("y", "~", paste(
  grep("^x|^grpPL", names(dat_poiss_train), value = TRUE),
  collapse = " + "
)))
refm_fit_poiss <- stan_glm(
  formula = refm_fml,
  family = poisson(),
  data = dat_poiss_train,
  prior = hs(global_scale = tau0, slab_df = 100, slab_scale = 1),
  ### Only for the sake of speed (not recommended in general):
  chains = 2, iter = 500,
  ###
  seed = 7286013, QR = TRUE, refresh = 0
)

## -----------------------------------------------------------------------------
library(projpred)

## ---- results='hide', message=TRUE--------------------------------------------
d_test_lat_poiss <- list(
  data = dat_poiss_test,
  offset = rep(0, nrow(dat_poiss_test)),
  weights = rep(1, nrow(dat_poiss_test)),
  ### Here, we are not interested in latent-scale post-processing, so we can set
  ### element `y` to a vector of `NA`s:
  y = rep(NA, nrow(dat_poiss_test)),
  ###
  y_oscale = dat_poiss_test$y
)
time_lat <- system.time(vs_lat <- varsel(
  refm_fit_poiss,
  d_test = d_test_lat_poiss,
  ### Only for the sake of speed (not recommended in general):
  nclusters_pred = 20,
  ###
  nterms_max = 14,
  seed = 95930,
  latent = TRUE
))

## -----------------------------------------------------------------------------
print(time_lat)

## ---- fig.width = 6, out.width = "75%"----------------------------------------
( gg_lat <- plot(vs_lat, stats = "mlpd", deltas = TRUE) )

## ---- fig.width = 6, out.width = "75%"----------------------------------------
gg_lat + ggplot2::coord_cartesian(ylim = c(-10, 0.05))

## -----------------------------------------------------------------------------
modsize_decided_lat <- 11

## -----------------------------------------------------------------------------
suggest_size(vs_lat, stat = "mlpd")

## -----------------------------------------------------------------------------
smmry_lat <- summary(
  vs_lat,
  stats = "mlpd",
  type = c("mean", "se", "lower", "upper", "diff", "diff.se")
)
print(smmry_lat, digits = 3)

## -----------------------------------------------------------------------------
soltrms_lat <- solution_terms(vs_lat)
( soltrms_lat_final <- head(soltrms_lat, modsize_decided_lat) )

## ---- results='hide'----------------------------------------------------------
d_test_trad_poiss <- d_test_lat_poiss
d_test_trad_poiss$y <- d_test_trad_poiss$y_oscale
d_test_trad_poiss$y_oscale <- NULL
time_trad <- system.time(vs_trad <- varsel(
  refm_fit_poiss,
  d_test = d_test_trad_poiss,
  ### Only for the sake of speed (not recommended in general):
  nclusters_pred = 20,
  ###
  nterms_max = 14,
  seed = 95930
))

## -----------------------------------------------------------------------------
print(time_trad)
( gg_trad <- plot(vs_trad, stats = "mlpd", deltas = TRUE) )
smmry_trad <- summary(
  vs_trad,
  stats = "mlpd",
  type = c("mean", "se", "lower", "upper", "diff", "diff.se")
)
print(smmry_trad, digits = 3)

## -----------------------------------------------------------------------------
refm_fit_nebin <- stan_glm(
  formula = refm_fml,
  family = neg_binomial_2(),
  data = dat_poiss_train,
  prior = hs(global_scale = tau0, slab_df = 100, slab_scale = 1),
  ### Only for the sake of speed (not recommended in general):
  chains = 2, iter = 500,
  ###
  seed = 7304, QR = TRUE, refresh = 0
)

## ---- results='hide', message=TRUE--------------------------------------------
refm_prec <- as.matrix(refm_fit_nebin)[, "reciprocal_dispersion", drop = FALSE]
latent_ll_oscale_nebin <- function(ilpreds, y_oscale,
                                   wobs = rep(1, length(y_oscale)), cl_ref,
                                   wdraws_ref = rep(1, length(cl_ref))) {
  y_oscale_mat <- matrix(y_oscale, nrow = nrow(ilpreds), ncol = ncol(ilpreds),
                         byrow = TRUE)
  wobs_mat <- matrix(wobs, nrow = nrow(ilpreds), ncol = ncol(ilpreds),
                     byrow = TRUE)
  refm_prec_agg <- cl_agg(refm_prec, cl = cl_ref, wdraws = wdraws_ref)
  ll_unw <- dnbinom(y_oscale_mat, size = refm_prec_agg, mu = ilpreds, log = TRUE)
  return(wobs_mat * ll_unw)
}
latent_ppd_oscale_nebin <- function(ilpreds_resamp, wobs, cl_ref,
                                    wdraws_ref = rep(1, length(cl_ref)),
                                    idxs_prjdraws) {
  refm_prec_agg <- cl_agg(refm_prec, cl = cl_ref, wdraws = wdraws_ref)
  refm_prec_agg_resamp <- refm_prec_agg[idxs_prjdraws, , drop = FALSE]
  ppd <- rnbinom(prod(dim(ilpreds_resamp)), size = refm_prec_agg_resamp,
                 mu = ilpreds_resamp)
  ppd <- matrix(ppd, nrow = nrow(ilpreds_resamp), ncol = ncol(ilpreds_resamp))
  return(ppd)
}
vs_lat_nebin <- varsel(
  refm_fit_nebin,
  d_test = d_test_lat_poiss,
  ### Only for the sake of speed (not recommended in general):
  nclusters_pred = 20,
  ###
  nterms_max = 14,
  seed = 95930,
  latent = TRUE,
  latent_ll_oscale = latent_ll_oscale_nebin,
  latent_ppd_oscale = latent_ppd_oscale_nebin
)

## ---- fig.width = 6, out.width = "75%"----------------------------------------
( gg_lat_nebin <- plot(vs_lat_nebin, stats = "mlpd", deltas = TRUE) )

## ---- fig.width = 6, out.width = "75%"----------------------------------------
gg_lat_nebin + ggplot2::coord_cartesian(ylim = c(-2.5, 0.25))

## -----------------------------------------------------------------------------
modsize_decided_lat_nebin <- 11

## -----------------------------------------------------------------------------
suggest_size(vs_lat_nebin, stat = "mlpd")

## -----------------------------------------------------------------------------
smmry_lat_nebin <- summary(
  vs_lat_nebin,
  stats = "mlpd",
  type = c("mean", "se", "lower", "upper", "diff", "diff.se")
)
print(smmry_lat_nebin, digits = 3)

## -----------------------------------------------------------------------------
soltrms_lat_nebin <- solution_terms(vs_lat_nebin)
( soltrms_lat_nebin_final <- head(soltrms_lat_nebin,
                                  modsize_decided_lat_nebin) )


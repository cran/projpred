params <-
list(EVAL = FALSE)

## ----SETTINGS-knitr, include=FALSE--------------------------------------------
stopifnot(require(knitr))
knitr::opts_chunk$set(
  dev = "png",
  dpi = 150,
  fig.asp = 0.618,
  fig.width = 7,
  out.width = "90%",
  fig.align = "center",
  comment = NA,
  eval = if (isTRUE(exists("params"))) params$EVAL else FALSE
)

## ----data---------------------------------------------------------------------
# data("df_gaussian", package = "projpred")
# dat_gauss <- data.frame(y = df_gaussian$y, df_gaussian$x)

## ----rstanarm_attach, message=FALSE-------------------------------------------
# library(rstanarm)

## ----rh-----------------------------------------------------------------------
# # Number of regression coefficients:
# ( D <- sum(grepl("^X", names(dat_gauss))) )
# # Prior guess for the number of relevant (i.e., non-zero) regression
# # coefficients:
# p0 <- 5
# # Number of observations:
# N <- nrow(dat_gauss)
# # Hyperprior scale for tau, the global shrinkage parameter (note that for the
# # Gaussian family, 'rstanarm' will automatically scale this by the residual
# # standard deviation):
# tau0 <- p0 / (D - p0) * 1 / sqrt(N)

## ----ref_fit------------------------------------------------------------------
# # Set this manually if desired:
# ncores <- parallel::detectCores(logical = FALSE)
# ### Only for technical reasons in this vignette (you can omit this when running
# ### the code yourself):
# ncores <- min(ncores, 2L)
# ###
# options(mc.cores = ncores)
# set.seed(50780)
# refm_fit <- stan_glm(
#   y ~ X1 + X2 + X3 + X4 + X5 + X6 + X7 + X8 + X9 + X10 + X11 + X12 + X13 + X14 +
#     X15 + X16 + X17 + X18 + X19 + X20,
#   family = gaussian(),
#   data = dat_gauss,
#   prior = hs(global_scale = tau0),
#   ### Only for the sake of speed (not recommended in general):
#   chains = 2, iter = 1000,
#   ###
#   refresh = 0
# )

## ----projpred_attach, message=FALSE-------------------------------------------
# library(projpred)

## ----refmodel_create----------------------------------------------------------
# refm_obj <- get_refmodel(refm_fit)

## ----cv_varsel_fast-----------------------------------------------------------
# # Preliminary cv_varsel() run:
# cvvs_fast <- cv_varsel(
#   refm_obj,
#   validate_search = FALSE,
#   ### Only for the sake of speed (not recommended in general):
#   method = "L1",
#   refit_prj = FALSE,
#   ###
#   nterms_max = 20,
#   ### In interactive use, we recommend not to deactivate the verbose mode:
#   verbose = 0
#   ###
# )

## ----plot_vsel_fast-----------------------------------------------------------
# options(projpred.plot_vsel_size_position = "primary_x_bottom")
# options(projpred.plot_vsel_text_angle = 0)
# plot(cvvs_fast, stats = "mlpd", ranking_nterms_max = NA)

## ----cv_varsel_fast_refit-----------------------------------------------------
# # Preliminary cv_varsel() run with `refit_prj = TRUE`:
# cvvs_fast_refit <- cv_varsel(
#   cvvs_fast,
#   ### Only for the sake of speed (not recommended in general):
#   nclusters_pred = 20,
#   ###
#   ### In interactive use, we recommend not to deactivate the verbose mode:
#   verbose = 0
#   ###
# )

## ----plot_vsel_fast_refit-----------------------------------------------------
# plot(cvvs_fast_refit, stats = "mlpd", ranking_nterms_max = NA)

## ----cv_varsel, message=FALSE-------------------------------------------------
# # Refit the reference model K times:
# cv_fits <- run_cvfun(
#   refm_obj,
#   ### Only for the sake of speed (not recommended in general):
#   K = 2
#   ###
# )
# # For running projpred's CV in parallel (see cv_varsel()'s argument `parallel`):
# doParallel::registerDoParallel(ncores)
# # Final cv_varsel() run:
# cvvs <- cv_varsel(
#   refm_obj,
#   cv_method = "kfold",
#   cvfits = cv_fits,
#   ### Only for the sake of speed (not recommended in general):
#   method = "L1",
#   nclusters_pred = 20,
#   ###
#   nterms_max = 9,
#   parallel = TRUE,
#   ### In interactive use, we recommend not to deactivate the verbose mode:
#   verbose = 0
#   ###
# )
# # Tear down the CV parallelization setup:
# doParallel::stopImplicitCluster()
# foreach::registerDoSEQ()

## ----plot_vsel----------------------------------------------------------------
# options(projpred.plot_vsel_show_cv_proportions = TRUE)
# plot(cvvs, stats = "mlpd", deltas = TRUE)

## ----size_man-----------------------------------------------------------------
# size_decided <- 7

## ----size_sgg-----------------------------------------------------------------
# suggest_size(cvvs, stat = "mlpd")

## ----smmry_vsel---------------------------------------------------------------
# smmry <- summary(cvvs,
#                  stats = "mlpd",
#                  type = c("mean", "lower", "upper"),
#                  deltas = TRUE)
# print(smmry, digits = 1)

## ----perf_smmry---------------------------------------------------------------
# perf <- performances(smmry)
# str(perf)

## ----ranking------------------------------------------------------------------
# rk <- ranking(cvvs)

## ----cv_proportions-----------------------------------------------------------
# ( pr_rk <- cv_proportions(rk) )

## ----ranking_fulldata---------------------------------------------------------
# rk[["fulldata"]]

## ----plot_cv_proportions------------------------------------------------------
# plot(pr_rk)

## ----predictors_final---------------------------------------------------------
# ( predictors_final <- head(rk[["fulldata"]], size_decided) )

## ----plot_cv_proportions_cumul------------------------------------------------
# plot(cv_proportions(rk, cumulate = TRUE))

## ----project------------------------------------------------------------------
# prj <- project(
#   refm_obj,
#   predictor_terms = predictors_final,
#   ### In interactive use, we recommend not to deactivate the verbose mode:
#   verbose = 0
#   ###
# )

## ----as_matrix_prj------------------------------------------------------------
# prj_mat <- as.matrix(prj)

## ----posterior_attach, message=FALSE------------------------------------------
# library(posterior)

## ----smmry_prj----------------------------------------------------------------
# prj_drws <- as_draws_matrix(prj_mat)
# prj_smmry <- summarize_draws(
#   prj_drws,
#   "median", "mad", function(x) quantile(x, probs = c(0.025, 0.975))
# )
# # Coerce to a `data.frame` because some pkgdown versions don't print the
# # tibble correctly:
# prj_smmry <- as.data.frame(prj_smmry)
# print(prj_smmry, digits = 1)

## ----bayesplot_attach, message=FALSE------------------------------------------
# library(bayesplot)

## ----bayesplot_prj------------------------------------------------------------
# bayesplot_theme_set(ggplot2::theme_bw())
# mcmc_intervals(prj_mat) +
#   ggplot2::coord_cartesian(xlim = c(-1.5, 1.6))

## ----bayesplot_ref------------------------------------------------------------
# refm_mat <- as.matrix(refm_fit)
# mcmc_intervals(refm_mat, pars = colnames(prj_mat)) +
#   ggplot2::coord_cartesian(xlim = c(-1.5, 1.6))

## ----data_new-----------------------------------------------------------------
# ( dat_gauss_new <- setNames(
#   as.data.frame(replicate(length(predictors_final), c(-1, 0, 1))),
#   predictors_final
# ) )

## ----proj_linpred-------------------------------------------------------------
# prj_linpred <- proj_linpred(prj, newdata = dat_gauss_new, integrated = TRUE)
# cbind(dat_gauss_new, linpred = as.vector(prj_linpred[["pred"]]))

## ----proj_predict-------------------------------------------------------------
# prj_predict <- proj_predict(prj)
# # Using the 'bayesplot' package:
# ppc_dens_overlay(y = dat_gauss$y, yrep = prj_predict)

## ----ref_fit_mlvl, eval=FALSE-------------------------------------------------
# data("VerbAgg", package = "lme4")
# refm_fit <- stan_glmer(
#   r2 ~ btype + situ + mode + (btype + situ + mode | id),
#   family = binomial(),
#   data = VerbAgg,
#   QR = TRUE, refresh = 0
# )

## ----ref_fit_addv, eval=FALSE-------------------------------------------------
# data("lasrosas.corn", package = "agridat")
# # Convert `year` to a `factor` (this could also be solved by using
# # `factor(year)` in the formula, but we avoid that here to put more emphasis on
# # the demonstration of the smooth term):
# lasrosas.corn$year <- as.factor(lasrosas.corn$year)
# refm_fit <- stan_gamm4(
#   yield ~ year + topo + t2(nitro, bv),
#   family = gaussian(),
#   data = lasrosas.corn,
#   QR = TRUE, refresh = 0
# )

## ----ref_fit_addv_mlvl, eval=FALSE--------------------------------------------
# data("gumpertz.pepper", package = "agridat")
# refm_fit <- stan_gamm4(
#   disease ~ field + leaf + s(water),
#   random = ~ (1 | row) + (1 | quadrat),
#   family = binomial(),
#   data = gumpertz.pepper,
#   QR = TRUE, refresh = 0
# )


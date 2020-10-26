params <-
list(EVAL = TRUE)

## ---- SETTINGS-knitr, include=FALSE-------------------------------------------
stopifnot(require(knitr))
knitr::opts_chunk$set(
    fig.width = 7, fig.height = 5,
    comment = NA,
    message = FALSE,
    warning = FALSE,
    eval = if (isTRUE(exists("params"))) params$EVAL else FALSE,
    dev = "png",
    dpi = 150,
    fig.align = "center"
)

## ---- results='hide', message=FALSE, warning=FALSE----------------------------
#  library(projpred)
#  library(rstanarm)
#  library(tidyr)
#  library(dplyr)
#  library(ggplot2)
#  library(bayesplot)
#  theme_set(theme_classic())
#  #options(mc.cores = 4)

## -----------------------------------------------------------------------------
#  data("Orthodont", package = "nlme")

## ---- cache=TRUE, results="hide", message=FALSE, warning=FALSE----------------
#  fit <- stan_glmer(distance ~ age * Sex + (age | Subject),
#                    chains = 2,  data = Orthodont, seed = 1)

## ---- cache=TRUE, results="hide", message=FALSE, warning=FALSE----------------
#  ref <- get_refmodel(fit)
#  vs <- varsel(ref)

## ---- message=FALSE, warning=FALSE--------------------------------------------
#  solution_terms(vs) # selection order of the variables

## ---- fig.width=5, fig.height=4-----------------------------------------------
#  # plot predictive performance on training data
#  plot(vs, stats = c('elpd', 'rmse'))

## ---- cache=TRUE, fig.width=6, fig.height=2, message=FALSE, warning=FALSE-----
#  # Visualise the projected three most relevant variables
#  proj <- project(vs, nterms = 2, ns = 500)
#  mcmc_areas(as.matrix(proj), pars = solution_terms(vs)[1:2])

## ---- cache=TRUE, message=FALSE, warning=FALSE--------------------------------
#  pred <- proj_linpred(vs, newdata = Orthodont, nterms = 5, integrated = TRUE)

## ---- fig.width=5, fig.height=3-----------------------------------------------
#  ggplot() +
#    geom_point(aes(x = pred$pred, y = Orthodont$distance)) +
#    geom_abline(slope = 1, color = "red") +
#    labs(x = "prediction", y = "y")

## ---- fig.height=3, fig.width=5, message=FALSE, warning=FALSE-----------------
#  subset <- Orthodont %>% as_tibble() %>% dplyr::sample_n(1)
#  y_subset <- subset %>% dplyr::select(distance) %>% as.data.frame()
#  y1_rep <- proj_predict(vs, newdata = subset, nterms = 5, seed = 7560)
#  qplot(as.vector(y1_rep), bins = 25) +
#    geom_vline(xintercept = as.numeric(y_subset), color = "red") +
#    xlab("y1_rep")

## -----------------------------------------------------------------------------
#  data_pois <- read.table("data_pois.csv", header = TRUE)

## ---- cache=TRUE, message=FALSE, warning=FALSE, results="hide"----------------
#  fit <- stan_glmer(
#    phen_pois ~ cofactor + (1 | phylo) + (1 | obs), data = data_pois,
#    family = poisson("log"), chains = 2, iter = 2000,
#    control = list(adapt_delta = 0.95)
#  )

## ---- cache=TRUE, results="hide", message=FALSE, warning=FALSE----------------
#  vs <- varsel(fit)

## ---- message=FALSE, warning=FALSE--------------------------------------------
#  solution_terms(vs) # selection order of the variables

## ---- fig.width=5, fig.height=4-----------------------------------------------
#  # plot predictive performance on training data
#  plot(vs, stats = c('elpd', 'rmse'))

## ---- cache=TRUE, fig.width=6, fig.height=2, message=FALSE, warning=FALSE-----
#  # Visualise the projected two most relevant variables
#  proj <- project(vs, nterms = 2, ndraws = 10)
#  mcmc_areas(as.matrix(proj), pars = solution_terms(vs)[1:2])

## ---- cache=TRUE, message=FALSE, warning=FALSE--------------------------------
#  pred <- proj_linpred(proj, newdata = data_pois, integrated = TRUE)

## ---- fig.width=5, fig.height=3-----------------------------------------------
#  xaxis <- seq(-3, 3, length.out = 1000)
#  y_mu <- rowMeans(vs$refmodel$mu)
#  ggplot() +
#    geom_point(aes(x = pred$pred, y = y_mu)) +
#    geom_line(aes(x=xaxis, y = exp(xaxis)), color = "red") +
#    labs(x = "prediction", y = "y")

## -----------------------------------------------------------------------------
#  data("VerbAgg", package = "lme4")

## -----------------------------------------------------------------------------
#  ## subsample 50 participants
#  VerbAgg_subsample <- VerbAgg %>%
#    tidyr::as_tibble() %>%
#    dplyr::filter(id %in% sample(id, 50)) %>%
#    dplyr::mutate(r2num = as.integer(r2) - 1) # binomial family needs numeric target

## ---- cache=TRUE, results="hide", message=FALSE, warning=FALSE----------------
#  ## simple bernoulli model
#  formula_va <- r2num ~ btype + situ + mode + (btype + situ + mode | id)
#  fit_va <- stan_glmer(
#    formula = formula_va,
#    data = VerbAgg_subsample,
#    family = binomial("logit"),
#    seed = 1234,
#    chains = 2
#  )

## ---- cache=TRUE, results="hide", message=FALSE, warning=FALSE----------------
#  vs_va <- varsel(fit_va)

## -----------------------------------------------------------------------------
#  solution_terms(vs_va)

## ---- fig.height=4, fig.width=5-----------------------------------------------
#  plot(vs_va, stats = c("elpd", "acc"))

## ---- cache=TRUE, results="hide", message=FALSE, warning=FALSE----------------
#  cv_vs_va <- cv_varsel(fit_va, validate_search = FALSE)

## ---- fig.height=4, fig.width=5-----------------------------------------------
#  plot(cv_vs_va, stats = c("elpd", "acc"))

## ---- cache=TRUE, message=FALSE, warning=FALSE--------------------------------
#  pred <- proj_linpred(cv_vs_va, newdata = VerbAgg_subsample,
#                       nterms = 6, integrated = TRUE, ndraws = 10)

## ---- fig.width=5, fig.height=3-----------------------------------------------
#  xaxis <- seq(-6, 6, length.out = 1000)
#  yaxis <- cv_vs_va$family$linkinv(xaxis)
#  
#  y_mu <- rowMeans(cv_vs_va$refmodel$mu)
#  ggplot() +
#    geom_point(aes(x = pred$pred, y = y_mu)) +
#    geom_line(aes(x = xaxis, y = yaxis), color = "red") +
#    labs(x = "prediction", y = "y")


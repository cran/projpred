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
#  library(brms)
#  library(projpred)
#  library(dplyr)
#  library(ggplot2)
#  library(bayesplot)
#  theme_set(theme_classic())
#  #options(mc.cores = parallel::detectCores())

## -----------------------------------------------------------------------------
#  data('df_gaussian', package = 'projpred')

## ---- results='hide', message=FALSE, warning=FALSE----------------------------
#  split_structure <- break_up_matrix_term(y ~ x, data = df_gaussian)
#  df_gaussian <- split_structure$data
#  formula <- split_structure$formula
#  d <- df_gaussian
#  n <- nrow(df_gaussian) # 100
#  D <- ncol(df_gaussian[, -1]) # 20
#  p0 <- 5 # prior guess for the number of relevant variables
#  tau0 <- p0/(D-p0) * 1/sqrt(n) # scale for tau (notice that stan_glm will automatically scale this by sigma)
#  fit <- brm(formula, family=gaussian(), data=df_gaussian,
#             prior=prior(horseshoe(scale_global = tau0, scale_slab = 1), class=b),
#             ## To make this vignette build fast, we use only 2 chains and
#             ## 500 iterations. In practice, at least 4 chains should be
#             ## used and 2000 iterations might be required for reliable
#             ## inference.
#             seed=1, chains=2, iter=500)

## ---- results='hide', warning=FALSE, messages=FALSE---------------------------
#  refmodel <- get_refmodel(fit)

## ---- results='hide', messages=FALSE, warnings=FALSE--------------------------
#  vs <- cv_varsel(refmodel, method = "forward")

## ---- messages=FALSE, warnings=FALSE------------------------------------------
#  solution_terms(vs) # selection order of the variables

## ---- fig.width=5, fig.height=4-----------------------------------------------
#  plot(vs, stats = c('elpd', 'rmse'))

## ---- fig.width=5, fig.height=4-----------------------------------------------
#  # plot the validation results, this time relative to the full model
#  plot(vs, stats = c('elpd', 'rmse'), deltas = TRUE)

## ---- fig.width=6, fig.height=2-----------------------------------------------
#   # Visualise the three most relevant variables in the full model -->
#   mcmc_areas(as.matrix(refmodel$fit),
#              pars = c("b_Intercept", paste0("b_", solution_terms(vs)[1:3]),
#                       "sigma")) +
#     coord_cartesian(xlim = c(-2, 2))

## ---- fig.width=6, fig.height=2-----------------------------------------------
#  # Visualise the projected three most relevant variables
#  proj <- project(vs, nterms = 3, ns = 500)
#  mcmc_areas(as.matrix(proj)) +
#    coord_cartesian(xlim = c(-2, 2))

## -----------------------------------------------------------------------------
#  pred <- proj_linpred(vs, newdata = df_gaussian, nterms = 6, integrated = TRUE)

## ---- fig.width=5, fig.height=3-----------------------------------------------
#  ggplot() +
#    geom_point(aes(x = pred$pred,y = df_gaussian$y)) +
#    geom_abline(slope = 1, color = "red") +
#    labs(x = "prediction", y = "y")

## ---- fig.height=3, fig.width=5-----------------------------------------------
#  subset <- df_gaussian %>% dplyr::sample_n(1)
#  y_subset <- subset %>% dplyr::select(y)
#  y1_rep <- proj_predict(vs, newdata = subset, nterms = 6, seed = 7560)
#  qplot(as.vector(y1_rep), bins = 25) +
#    geom_vline(xintercept = as.numeric(y_subset), color = "red") +
#    xlab("y1_rep")


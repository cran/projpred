% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/search.R
\name{force_search_terms}
\alias{force_search_terms}
\title{Force search terms}
\usage{
force_search_terms(forced_terms, optional_terms)
}
\arguments{
\item{forced_terms}{A character vector of predictor terms that should be
selected first.}

\item{optional_terms}{A character vector of predictor terms that should be
subject to the variable selection after the inclusion of the "forced"
terms.}
}
\value{
A character vector that may be used as input for argument
\code{search_terms} of \code{\link[=varsel]{varsel()}} or \code{\link[=cv_varsel]{cv_varsel()}}.
}
\description{
A helper function to construct the input for argument \code{search_terms} of
\code{\link[=varsel]{varsel()}} or \code{\link[=cv_varsel]{cv_varsel()}} if certain predictor terms should be forced to be
selected first whereas other predictor terms are optional (i.e., they are
subject to the variable selection, but only after the inclusion of the
"forced" terms).
}
\examples{
\dontshow{if (requireNamespace("rstanarm", quietly = TRUE)) (if (getRversion() >= "3.4") withAutoprint else force)(\{ # examplesIf}
# Data:
dat_gauss <- data.frame(y = df_gaussian$y, df_gaussian$x)

# The `stanreg` fit which will be used as the reference model (with small
# values for `chains` and `iter`, but only for technical reasons in this
# example; this is not recommended in general):
fit <- rstanarm::stan_glm(
  y ~ X1 + X2 + X3 + X4 + X5, family = gaussian(), data = dat_gauss,
  QR = TRUE, chains = 2, iter = 500, refresh = 0, seed = 9876
)

# We will force X1 and X2 to be selected first:
search_terms_forced <- force_search_terms(
  forced_terms = paste0("X", 1:2),
  optional_terms = paste0("X", 3:5)
)

# Run varsel() (here without cross-validation and with small values for
# `nterms_max`, `nclusters`, and `nclusters_pred`, but only for the sake of
# speed in this example; this is not recommended in general):
vs <- varsel(fit, nclusters = 5, nclusters_pred = 10,
             search_terms = search_terms_forced, seed = 5555)
# Now see, for example, `?print.vsel`, `?plot.vsel`, `?suggest_size.vsel`,
# and `?ranking` for possible post-processing functions.
\dontshow{\}) # examplesIf}
}
\seealso{
\code{\link[=varsel]{varsel()}}, \code{\link[=cv_varsel]{cv_varsel()}}
}

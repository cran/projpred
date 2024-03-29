% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/helper_formula.R
\name{break_up_matrix_term}
\alias{break_up_matrix_term}
\title{Break up matrix terms}
\usage{
break_up_matrix_term(formula, data)
}
\arguments{
\item{formula}{A \code{\link{formula}} for a valid model.}

\item{data}{The original \code{data.frame} with a matrix as predictor.}
}
\value{
A \code{list} containing the expanded \code{\link{formula}} and the expanded
\code{data.frame}.
}
\description{
Sometimes there can be terms in a formula that refer to a matrix instead of a
single predictor. This function breaks up the matrix term into individual
predictors to handle separately, as that is probably the intention of the
user.
}

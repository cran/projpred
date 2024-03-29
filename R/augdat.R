#__________________________________________________________________________
# Helper functions for the augmented-data projection
#__________________________________________________________________________

#' Augmented-data projection: Internals
#'
#' The augmented-data projection makes extensive use of *augmented-rows
#' matrices* and *augmented-length vectors*. In the following, \eqn{N},
#' \eqn{C_{\mathrm{cat}}}{C_cat}, \eqn{C_{\mathrm{lat}}}{C_lat},
#' \eqn{S_{\mathrm{ref}}}{S_ref}, and \eqn{S_{\mathrm{prj}}}{S_prj} from help
#' topic [refmodel-init-get] are used. Furthermore, let \eqn{C} denote either
#' \eqn{C_{\mathrm{cat}}}{C_cat} or \eqn{C_{\mathrm{lat}}}{C_lat}, whichever is
#' appropriate in the context where it is used (e.g., for `ref_predfun`'s
#' output, \eqn{C = C_{\mathrm{lat}}}{C = C_lat}). Similarly, let \eqn{S} denote
#' either \eqn{S_{\mathrm{ref}}}{S_ref} or \eqn{S_{\mathrm{prj}}}{S_prj},
#' whichever is appropriate in the context where it is used. Then an
#' augmented-rows matrix is a matrix with \eqn{N \cdot C}{N * C} rows in \eqn{C}
#' blocks of \eqn{N} rows, i.e., with the \eqn{N} observations nested in the
#' \eqn{C} (possibly latent) response categories. For ordered response
#' categories, the \eqn{C} (possibly latent) response categories (i.e., the row
#' blocks) have to be sorted increasingly. The columns of an augmented-rows
#' matrix have to correspond to the \eqn{S} parameter draws, just like for the
#' traditional projection. An augmented-rows matrix is of class `augmat`
#' (inheriting from classes `matrix` and `array`) and needs to have the value of
#' \eqn{C} stored in an attribute called `ndiscrete`. An augmented-length vector
#' (class `augvec`) is the vector resulting from subsetting an augmented-rows
#' matrix to extract a single column and thereby dropping dimensions.
#'
#' @name augdat-internals
#' @keywords internal
NULL

# Convert a 3-dimensional array to an augmented-rows matrix.
#
# @param arr If `margin_draws` is `3`, a 3-dimensional array with dimensions N x
#   C x S. If `margin_draws` is `1`, a 3-dimensional array with dimensions S x N
#   x C. See above for a definition of these dimensions.
# @param margin_draws The index of `arr`'s margin which corresponds to the
#   parameter draws (i.e., the margin of length S). Restricted to values `1` and
#   `3`.
#
# @return An augmented-rows matrix (see above for a definition).
arr2augmat <- function(arr, margin_draws = 3) {
  stopifnot(is.array(arr), length(dim(arr)) == 3)
  stopifnot(margin_draws %in% c(1, 3))
  if (margin_draws == 1) {
    margin_obs <- 2
  } else if (margin_draws == 3) {
    margin_obs <- 1
  }
  augmat <- apply(arr, margin_draws, as.vector, simplify = FALSE)
  augmat <- do.call(cbind, augmat)
  attr(augmat, "ndiscrete") <- dim(arr)[-c(margin_draws, margin_obs)]
  class(augmat) <- "augmat"
  return(augmat)
}

# Convert an augmented-rows matrix (see above for a definition) to a
# 3-dimensional array.
#
# @param augmat An augmented-rows matrix.
# @param ndiscrete The number of (possibly latent) response categories (C).
#   Usually should not have to be specified manually (i.e., the default should
#   always work).
# @param margin_draws The index of the returned array's margin which shall
#   correspond to the parameter draws (i.e., the margin which shall be of length
#   S). Restricted to values `1` and `3`.
#
# @return If `margin_draws` is `3`, a 3-dimensional array with dimensions
#   N x C x S. If `margin_draws` is `1`, a 3-dimensional array with dimensions
#   S x N x C.
augmat2arr <- function(augmat,
                       ndiscrete = attr(augmat, "ndiscrete"),
                       margin_draws = 3) {
  stopifnot(inherits(augmat, "augmat"))
  stopifnot(!is.null(dim(augmat)))
  stopifnot(!is.null(ndiscrete))
  stopifnot(margin_draws %in% c(1, 3))
  nobs <- nrow(augmat) / ndiscrete
  stopifnot(is_wholenumber(nobs))
  nobs <- as.integer(round(nobs))
  arr <- array(augmat, dim = c(nobs, ndiscrete, ncol(augmat)))
  if (margin_draws == 1) {
    arr <- aperm(arr, perm = c(3, 1, 2))
  }
  return(arr)
}

# A t() method for class `augmat`, dropping the class and the `ndiscrete`
# attribute. This is necessary for clustering with kmeans(), for example.
#' @noRd
#' @export
t.augmat <- function(x) {
  class(x) <- NULL
  attr(x, "ndiscrete") <- NULL
  return(t(x))
}

# A t() method for class `augvec`, dropping the class and the `ndiscrete`
# attribute. This should not be necessary, but it's probably safer to have such
# a method (to avoid that the attributes are carried around after a t() call).
#' @noRd
#' @export
t.augvec <- function(x) {
  class(x) <- NULL
  attr(x, "ndiscrete") <- NULL
  return(t(x))
}

# A method for subsetting an object of class `augmat` (mainly following
# `[.factor`). This method keeps the `ndiscrete` attribute. It also keeps the
# class, except if the result is a vector (in which case the class is changed
# from `augmat` to `augvec`).
#
# Note: Subsetting the rows of an augmented-rows matrix is only legal in terms
# of the observations (individuals), not in terms of the (possibly latent)
# response categories.
#' @noRd
#' @export
`[.augmat` <- function(x, ..., drop = TRUE) {
  x_out <- NextMethod("[")
  attr(x_out, "ndiscrete") <- attr(x, "ndiscrete")
  cls_out <- oldClass(x)
  if (is.null(dim(x_out))) {
    cls_out <- sub("augmat", "augvec", cls_out, fixed = TRUE)
  }
  class(x_out) <- cls_out
  return(x_out)
}

# A method for subsetting an object of class `augvec` (mainly following
# `[.factor`). This method keeps the `ndiscrete` attribute and the class.
#
# Note: Subsetting an augmented-length vector is only legal in terms of the
# observations (individuals), not in terms of the (possibly latent) response
# categories.
#' @noRd
#' @export
`[.augvec` <- function(x, ..., drop = TRUE) {
  x_out <- NextMethod("[")
  attr(x_out, "ndiscrete") <- attr(x, "ndiscrete")
  class(x_out) <- oldClass(x)
  return(x_out)
}

# Convert an augmented-length vector to an augmented-rows matrix.
#
# @param augvec An augmented-length vector (see above for a definition).
#
# @return An augmented-rows matrix (see above for a definition) with a single
#   column.
augvec2augmat <- function(augvec) {
  stopifnot(inherits(augvec, "augvec"))
  return(structure(
    as.matrix(augvec),
    ndiscrete = attr(augvec, "ndiscrete"),
    class = sub("augvec", "augmat", oldClass(augvec), fixed = TRUE)
  ))
}

# Convert an augmented-rows matrix (with a single column) to an augmented-length
# vector.
#
# @param augmat An augmented-rows matrix (see above for a definition) with a
#   single column.
#
# @return An augmented-length vector (see above for a definition).
augmat2augvec <- function(augmat) {
  stopifnot(inherits(augmat, "augmat"))
  stopifnot(identical(ncol(augmat), 1L))
  return(augmat[, 1])
}

# Helper function for calculating log-likelihood values if the response
# distribution has finite support (as is the case in augmented-data projection,
# for example).
#
# @param mu_arr Array of probabilities for the C = C_cat response categories.
#   The structure depends on `margin_draws`: If `margin_draws` is `3`, a
#   3-dimensional array with dimensions N x C x S. If `margin_draws` is `1`, a
#   3-dimensional array with dimensions S x N x C. See above for a definition of
#   these dimensions.
# @param margin_draws The index of `mu_arr`'s margin which corresponds to the
#   parameter draws (i.e., the margin of length S). Restricted to values `1` and
#   `3`.
# @param y The response `factor` containing the observed response categories.
# @param wobs A numeric vector (recycled if of length 1) containing the
#   observation weights. Can also be of length 0 to use a vector of ones.
#
# @return If `margin_draws` is `3`, a matrix with dimensions N x S. If
#   `margin_draws` is `1`, a matrix with dimensions S x N.
ll_cats <- function(mu_arr, margin_draws = 3, y, wobs = 1) {
  stopifnot(is.array(mu_arr), length(dim(mu_arr)) == 3)
  stopifnot(margin_draws %in% c(1, 3))
  if (margin_draws == 1) {
    margin_obs <- 2
    margin_cats <- 3
    bind_fun <- cbind
  } else if (margin_draws == 3) {
    margin_obs <- 1
    margin_cats <- 2
    bind_fun <- rbind
  }
  stopifnot(is.factor(y),
            length(y) == dim(mu_arr)[margin_obs],
            nlevels(y) == dim(mu_arr)[margin_cats])
  if (length(wobs) == 0) {
    wobs <- rep(1, length(y))
  } else if (length(wobs) == 1) {
    wobs <- rep(wobs, length(y))
  } else if (length(wobs) != length(y)) {
    stop("Argument `wobs` needs to be of length 0, 1, or `length(y)`.")
  }
  return(do.call(bind_fun, lapply(seq_along(y), function(i_obs) {
    if (margin_draws == 1) {
      prbs_i <- mu_arr[, i_obs, y[i_obs]]
    } else if (margin_draws == 3) {
      prbs_i <- mu_arr[i_obs, y[i_obs], ]
    }

    # Assign some nonzero value to have a finite log() value:
    prbs_i[prbs_i == 0] <- .Machine$double.eps

    return(wobs[i_obs] * log(prbs_i))
  })))
}

# Helper function for drawing from the posterior(-projection) predictive
# distribution if the response distribution has finite support (as is the case
# in augmented-data projection, for example).
#
# @param mu_arr Array of probabilities for the C = C_cat response categories.
#   The structure depends on `margin_draws`: If `margin_draws` is `3`, a
#   3-dimensional array with dimensions N x C x S. If `margin_draws` is `1`, a
#   3-dimensional array with dimensions S x N x C. See above for a definition of
#   these dimensions.
# @param margin_draws The index of `mu_arr`'s margin which corresponds to the
#   parameter draws (i.e., the margin of length S). Restricted to values `1` and
#   `3`.
# @param wobs A numeric vector (recycled if of length 1) containing the
#   observation weights. Can also be of length 0 to use a vector of ones.
# @param return_vec A single logical value indicating whether to return a vector
#   (of length N). Only possible if S = 1.
#
# @return If `return_vec = FALSE`, then: If `margin_draws` is `3`, a matrix with
#   dimensions N x S. If `margin_draws` is `1`, a matrix with dimensions S x N.
#
#   If `return_vec = TRUE`, then: A vector of length N (requires S = 1).
ppd_cats <- function(mu_arr, margin_draws = 3, wobs = 1, return_vec = FALSE) {
  stopifnot(is.array(mu_arr), length(dim(mu_arr)) == 3)
  stopifnot(margin_draws %in% c(1, 3))
  if (margin_draws == 1) {
    margin_obs <- 2
    margin_cats <- 3
    bind_fun <- cbind
  } else if (margin_draws == 3) {
    margin_obs <- 1
    margin_cats <- 2
    bind_fun <- rbind
  }
  n_draws <- dim(mu_arr)[margin_draws]
  n_obs <- dim(mu_arr)[margin_obs]
  n_cat <- dim(mu_arr)[margin_cats]
  wobs <- parse_wobs_ppd(wobs, n_obs = n_obs)
  if (return_vec) {
    stopifnot(n_draws == 1)
    bind_fun <- c
  }
  return(do.call(bind_fun, lapply(seq_len(n_obs), function(i_obs) {
    do.call(c, lapply(seq_len(n_draws), function(i_draw) {
      if (margin_draws == 1) {
        prbs_i <- mu_arr[i_draw, i_obs, ]
      } else if (margin_draws == 3) {
        prbs_i <- mu_arr[i_obs, , i_draw]
      }
      return(sample.int(n_cat, size = 1L, prob = prbs_i))
    }))
  })))
}

# Find the maximum-probability category for each observation (with "observation"
# meaning one of the N original observations, not one of the \eqn{N \cdot C}{N *
# C} augmented observations).
#
# @param augvec An augmented-length vector (see above for a definition)
#   containing the probabilities for the response categories.
# @param lvls The response levels (as a character vector).
#
# @return A `factor` consisting of the maximum-probability categories. The
#   levels of this `factor` are those from `lvls`.
catmaxprb <- function(augvec, lvls) {
  arr <- augmat2arr(augvec2augmat(augvec))
  idxmaxprb <- do.call(c, lapply(seq_len(dim(arr)[1]), function(i_obs) {
    idx_out <- which.max(arr[i_obs, , 1])
    if (length(idx_out) == 0) {
      idx_out <- NA_integer_
    }
    return(idx_out)
  }))
  return(factor(lvls[idxmaxprb], levels = lvls))
}

fams_neg_linpred <- function() {
  return(c("cumulative", "cumulative_rstanarm", "sratio"))
}

# Link and inverse-link functions with array as input and output ----------

#' Link function for augmented-data projection with binomial family
#'
#' This is the function which has to be supplied to [extend_family()]'s argument
#' `augdat_link` in case of the augmented-data projection for the [binomial()]
#' family.
#'
#' @param prb_arr An array as described in section "Augmented-data projection"
#'   of [extend_family()]'s documentation.
#' @param link The same as argument `link` of [binomial()].
#'
#' @return An array as described in section "Augmented-data projection" of
#'   [extend_family()]'s documentation.
#'
#' @export
augdat_link_binom <- function(prb_arr, link = "logit") {
  return(linkfun_raw(prb_arr[, , -1, drop = FALSE], link_nm = link))
}

#' Inverse-link function for augmented-data projection with binomial family
#'
#' This is the function which has to be supplied to [extend_family()]'s argument
#' `augdat_ilink` in case of the augmented-data projection for the [binomial()]
#' family.
#'
#' @param eta_arr An array as described in section "Augmented-data projection"
#'   of [extend_family()]'s documentation.
#' @param link The same as argument `link` of [binomial()].
#'
#' @return An array as described in section "Augmented-data projection" of
#'   [extend_family()]'s documentation.
#'
#' @export
augdat_ilink_binom <- function(eta_arr, link = "logit") {
  prb_arr1 <- ilinkfun_raw(eta_arr, link_nm = link)
  prb_arr0 <- 1 - prb_arr1
  stopifnot(identical(dim(prb_arr0), dim(prb_arr1)))
  stopifnot(identical(dim(prb_arr1)[3], 1L))
  return(array(c(prb_arr0, prb_arr1), dim = c(dim(prb_arr1)[-3], 2L)))
}

## From brms --------------------------------------------------------------
## The functions from this (sub-)section are copied over from brms (with Paul
## Bürkner's consent) to avoid loading brms just for these special link and
## inverse-link functions. (After copying over, they have been slightly modified
## here to avoid dependencies on other brms-internal functions.)

augdat_link_cumul <- function(prb_arr, link) {
  ncat <- utils::tail(dim(prb_arr), 1)
  cumprb_arr <- apply(prb_arr[, , -ncat, drop = FALSE], c(1, 2), cumsum)
  cumprb_arr <- array(cumprb_arr,
                      dim = c(ncat - 1, utils::head(dim(prb_arr), -1)))
  cumprb_arr <- aperm(cumprb_arr, perm = c(c(1, 2) + 1, 1))
  return(linkfun_raw(cumprb_arr, link_nm = link))
}

augdat_ilink_cumul <- function(eta_arr, link) {
  cumprb_arr <- ilinkfun_raw(eta_arr, link_nm = link)
  dim_noncat <- utils::head(dim(cumprb_arr), -1)
  ones_arr <- array(1, dim = c(dim_noncat, 1))
  zeros_arr <- array(0, dim = c(dim_noncat, 1))
  return(abind::abind(cumprb_arr, ones_arr) -
           abind::abind(zeros_arr, cumprb_arr))
}

#include "glm_irls.h"
#include "../ols/ols_qr_decomp.h"
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/*

 Compiling from the root directory

 gcc -Iols -c glm/glm_irls.c -o glm/glm_irls.o

 Might also condsider Makefile

*/

// Example: a simple IRLS implementation for a GLM with a given link function
// For example, a logistic regression or a Poisson regression may be implemented
// similarly. This pseudo-code outlines the process.

// Function prototypes for link function and its derivative
double link_inverse(double eta);
double dlink_inverse_deta(double eta);

GLMResult glm_irls(double *X, double *y, int rows, int cols) {
  // Initialize beta coefficients (for intercept and other params)
  int cols_with_intercept = cols + 1;
  double *beta = (double *)calloc(cols_with_intercept, sizeof(double));

  // Set convergence criteria and maximum iterations
  int max_iter = 100;
  double tol = 1e-6;
  GLMResult result; // struct defined in your glm_irls.h

  // Allocate memory for weighted transformed variables
  double *eta = (double *)malloc(rows * sizeof(double));
  double *mu = (double *)malloc(rows * sizeof(double));
  double *z = (double *)malloc(rows * sizeof(double));
  double *sqrt_w = (double *)malloc(rows * sizeof(double));

  // Main IRLS loop
  for (int iter = 0; iter < max_iter; iter++) {
    // 1. Calculate eta = X * beta (include intercept in X if needed)
    for (int i = 0; i < rows; i++) {
      // Start with intercept
      eta[i] = beta[0];
      for (int j = 0; j < cols; j++) {
        eta[i] += X[i * cols + j] * beta[j + 1];
      }
    }

    // 2. Calculate mean response mu = g_inverse(eta) and the derivative values
    for (int i = 0; i < rows; i++) {
      mu[i] = link_inverse(eta[i]);
      // Typically the weight is computed as (g'(mu))^2/Var(y)
      // For many GLMs Var(y) depends on mu, e.g., for binomial or Poisson.
      // Here, we show a generic placeholder:
      double dmu_deta = dlink_inverse_deta(eta[i]);
      // For illustration, assume variance function V(mu) is provided or assumed
      // to be 1
      double variance = 1.0; // Replace with appropriate variance function V(mu)
      double w = (dmu_deta * dmu_deta) / variance;
      sqrt_w[i] = sqrt(w);
      // 3. Calculate the working response, z
      // The adjustment term uses (y - mu) scaled by the derivative
      z[i] = eta[i] + (y[i] - mu[i]) / dmu_deta;
    }

    // 4. Pre-weight the design matrix and working response
    // Create copies of the weighted X matrix and weighted working response.
    double *X_weighted =
        (double *)malloc(rows * cols_with_intercept * sizeof(double));
    double *z_weighted = (double *)malloc(rows * sizeof(double));
    for (int i = 0; i < rows; i++) {
      // For intercept column:
      X_weighted[i * cols_with_intercept] = sqrt_w[i];
      // For other columns:
      for (int j = 0; j < cols; j++) {
        X_weighted[i * cols_with_intercept + (j + 1)] =
            X[i * cols + j] * sqrt_w[i];
      }
      z_weighted[i] = z[i] * sqrt_w[i];
    }

    // 5. Solve the weighted least squares using your OLS QR function
    OLSResult ols_result = ols_qr(X_weighted, z_weighted, rows, cols);

    // Check for failure in OLS solving (e.g., singular matrix)
    if (ols_result.betas == NULL) {
      // Handle error (free resources, return an error result, etc.)
      fprintf(stderr, "OLS failed in iteration %d\n", iter);
      break;
    }

    // 6. Check convergence (compare new beta with current beta)
    double max_change = 0.0;
    for (int i = 0; i < cols_with_intercept; i++) {
      double change = fabs(ols_result.betas[i] - beta[i]);
      if (change > max_change) {
        max_change = change;
      }
    }

    // Update beta with the new estimates
    for (int i = 0; i < cols_with_intercept; i++) {
      beta[i] = ols_result.betas[i];
    }

    // Free the temporary OLS result if needed
    free_ols_result(&ols_result);

    // Free pre-weighted arrays for this iteration
    free(X_weighted);
    free(z_weighted);

    if (max_change < tol) {
      // Convergence achieved
      break;
    }
  }

  // Fill out the result structure as needed (e.g., final beta values, fitted
  // values, etc.)
  result.coefficients = beta;
  // Additional result fields (e.g., standard errors, residuals) can be computed
  // from a final OLS fit.

  // Free temporary working arrays
  free(eta);
  free(mu);
  free(z);
  free(sqrt_w);

  return result;
}

// Example functions for the link inverse and its derivative
// These must be defined appropriately for your GLM
double link_inverse(double eta) {
  // Example: for logistic regression, use the logistic function
  return 1.0 / (1.0 + exp(-eta));
}

double dlink_inverse_deta(double eta) {
  double mu = link_inverse(eta);
  // For logistic regression, derivative is mu*(1-mu)
  return mu * (1.0 - mu);
}

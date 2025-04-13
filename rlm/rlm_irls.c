#include "rlm_irls.h"
#include "../ols/ols_qr_decomp.h" // Using your OLS routines
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Function prototype for computing a robust scale estimate (e.g., MAD)
double compute_mad(double *r, int n);

// Computes Huber weight for a residual value
double huber_weight(double r, double k, double sigma) {
  double z = r / sigma;
  if (fabs(z) <= k) {
    return 1.0;
  } else {
    return k / fabs(z);
  }
}

RLMResult rlm_huber(double *X, double *y, int rows, int cols) {
  int cols_with_intercept = cols + 1;
  double *beta = (double *)calloc(cols_with_intercept, sizeof(double));
  double tol = 1e-6;
  int max_iter = 100;
  double k = 1.345; // Huber constant

  // Variables for IRLS
  double *residuals = (double *)malloc(rows * sizeof(double));
  double *weights = (double *)malloc(rows * sizeof(double));
  double *X_weighted =
      (double *)malloc(rows * cols_with_intercept * sizeof(double));
  double *y_weighted = (double *)malloc(rows * sizeof(double));

  // Optionally, start with an initial OLS fit:
  OLSResult initial_fit = ols_qr(X, y, rows, cols);
  if (initial_fit.betas == NULL) {
    fprintf(stderr, "Initial OLS fit failed\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < cols_with_intercept; i++) {
    beta[i] = initial_fit.betas[i];
  }
  free_ols_result(&initial_fit);

  int iter;
  for (iter = 0; iter < max_iter; iter++) {
    // 1. Compute residuals r = y - X * beta
    for (int i = 0; i < rows; i++) {
      double pred = beta[0];
      for (int j = 0; j < cols; j++) {
        pred += X[i * cols + j] * beta[j + 1];
      }
      residuals[i] = y[i] - pred;
    }

    // 2. Compute a robust scale estimate from residuals (using MAD, for
    // example)
    double mad = compute_mad(residuals, rows);
    double sigma = mad / 0.6745; // scale factor for normal distribution

    // 3. Compute Huber weights for each observation
    for (int i = 0; i < rows; i++) {
      weights[i] = huber_weight(residuals[i], k, sigma);
    }

    // 4. Pre-weight the design matrix and response vector
    for (int i = 0; i < rows; i++) {
      // For the intercept column
      X_weighted[i * cols_with_intercept] = sqrt(weights[i]);
      // For the remaining predictors
      for (int j = 0; j < cols; j++) {
        X_weighted[i * cols_with_intercept + (j + 1)] =
            X[i * cols + j] * sqrt(weights[i]);
      }
      y_weighted[i] = y[i] * sqrt(weights[i]);
    }

    // 5. Solve the weighted least squares problem using OLS
    OLSResult ols_result = ols_qr(X_weighted, y_weighted, rows, cols);
    if (ols_result.betas == NULL) {
      fprintf(stderr, "Weighted OLS failed at iteration %d\n", iter);
      break;
    }

    // 6. Check convergence of beta estimates
    double max_change = 0.0;
    for (int i = 0; i < cols_with_intercept; i++) {
      double change = fabs(ols_result.betas[i] - beta[i]);
      if (change > max_change) {
        max_change = change;
      }
    }
    for (int i = 0; i < cols_with_intercept; i++) {
      beta[i] = ols_result.betas[i];
    }
    free_ols_result(&ols_result);

    if (max_change < tol) {
      break;
    }
  }

  // Build and return the result structure
  RLMResult result;
  result.coefficients = beta;
  // Optionally, you can compute standard errors, t-stats, or other diagnostics
  // using a final OLS fit on the weighted data. For brevity, only the
  // coefficients and convergence iterations are set here.
  result.iterations = iter;

  // Clean up temporary arrays
  free(residuals);
  free(weights);
  free(X_weighted);
  free(y_weighted);

  return result;
}

// Example function to compute Median Absolute Deviation (MAD)
double compute_mad(double *r, int n) {
  // For a robust scale measure, you would:
  // 1. Compute the median of r.
  // 2. Compute the absolute deviations from the median.
  // 3. Compute the median of these deviations.
  // For simplicity, here’s a placeholder function.
  // You might want to implement a proper median computation.
  double median = r[n / 2]; // Note: This assumes r is sorted
  double *abs_dev = (double *)malloc(n * sizeof(double));
  for (int i = 0; i < n; i++) {
    abs_dev[i] = fabs(r[i] - median);
  }
  double mad = abs_dev[n / 2]; // Again, assumes abs_dev is sorted
  free(abs_dev);
  return mad;
}

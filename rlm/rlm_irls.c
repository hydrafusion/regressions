#include "rlm_irls.h"
#include "../ols/ols_qr_decomp.h" // Using OLS routines
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Computes Huber weight for a residual value
static double huber_weight(double r, double k, double sigma) {
  double z = r / sigma;
  if (fabs(z) <= k) {
    return 1.0;
  } else {
    return k / fabs(z);
  }
}

// Computes Median Absolute Deviation (MAD)
static double compute_mad(double *r, int n) {
  // Create a copy of the array to sort
  double *r_copy = (double *)malloc(n * sizeof(double));
  if (r_copy == NULL) {
    fprintf(stderr, "Memory allocation failed in compute_mad\n");
    return 0.0;
  }

  for (int i = 0; i < n; i++) {
    r_copy[i] = r[i];
  }

  // Simple bubble sort for median (for small datasets)
  // For larger datasets, consider using qsort from stdlib
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (r_copy[j] > r_copy[j + 1]) {
        double temp = r_copy[j];
        r_copy[j] = r_copy[j + 1];
        r_copy[j + 1] = temp;
      }
    }
  }

  // Compute median
  double median;
  if (n % 2 == 0) {
    median = (r_copy[n / 2 - 1] + r_copy[n / 2]) / 2.0;
  } else {
    median = r_copy[n / 2];
  }

  // Compute absolute deviations
  double *abs_dev = (double *)malloc(n * sizeof(double));
  if (abs_dev == NULL) {
    fprintf(stderr, "Memory allocation failed in compute_mad\n");
    free(r_copy);
    return 0.0;
  }

  for (int i = 0; i < n; i++) {
    abs_dev[i] = fabs(r[i] - median);
  }

  // Sort absolute deviations
  for (int i = 0; i < n - 1; i++) {
    for (int j = 0; j < n - i - 1; j++) {
      if (abs_dev[j] > abs_dev[j + 1]) {
        double temp = abs_dev[j];
        abs_dev[j] = abs_dev[j + 1];
        abs_dev[j + 1] = temp;
      }
    }
  }

  // Compute median of absolute deviations
  double mad;
  if (n % 2 == 0) {
    mad = (abs_dev[n / 2 - 1] + abs_dev[n / 2]) / 2.0;
  } else {
    mad = abs_dev[n / 2];
  }

  free(abs_dev);
  free(r_copy);

  return mad;
}

// Calculate R² for robust regression
static double compute_rlm_r2(double *y, double *y_pred, int n) {
  double ss_total = 0.0, ss_residual = 0.0, mean_y = 0.0;

  // Compute mean of y
  for (int i = 0; i < n; i++) {
    mean_y += y[i];
  }
  mean_y /= n;

  // Compute total sum of squares and residual sum of squares
  for (int i = 0; i < n; i++) {
    ss_total += (y[i] - mean_y) * (y[i] - mean_y);
    ss_residual += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
  }

  return 1.0 - (ss_residual / ss_total);
}

RLMResult rlm_huber(double *X, double *y, int rows, int cols) {
  RLMResult result;
  int cols_with_intercept = cols + 1;

  // Initialize all result fields to NULL/0
  result.coefficients = NULL;
  result.standard_errors = NULL;
  result.t_stats = NULL;
  result.residuals = NULL;
  result.r2 = 0.0;
  result.adjusted_r2 = 0.0;
  result.iterations = 0;
  result.converged = 0.0;

  // Check if we have enough observations
  if (rows <= cols) {
    fprintf(stderr, "Error: Need more observations than parameters\n");
    return result;
  }

  // Allocate memory for result fields
  result.coefficients = (double *)calloc(cols_with_intercept, sizeof(double));
  result.standard_errors =
      (double *)calloc(cols_with_intercept, sizeof(double));
  result.t_stats = (double *)calloc(cols_with_intercept, sizeof(double));
  result.residuals = (double *)malloc(rows * sizeof(double));

  if (!result.coefficients || !result.standard_errors || !result.t_stats ||
      !result.residuals) {
    fprintf(stderr, "Memory allocation failed in rlm_huber\n");
    free_rlm_result(&result);
    return result;
  }

  // IRLS algorithm parameters
  double tol = 1e-6;
  int max_iter = 100;
  double k = 1.345; // Huber constant

  // Variables for IRLS
  double *weights = (double *)malloc(rows * sizeof(double));
  double *X_weighted =
      (double *)malloc(rows * cols_with_intercept * sizeof(double));
  double *y_weighted = (double *)malloc(rows * sizeof(double));
  double *y_pred = (double *)malloc(rows * sizeof(double));

  if (!weights || !X_weighted || !y_weighted || !y_pred) {
    fprintf(stderr, "Memory allocation failed in rlm_huber\n");
    free(weights);
    free(X_weighted);
    free(y_weighted);
    free(y_pred);
    free_rlm_result(&result);
    return result;
  }

  // Start with an initial OLS fit
  OLSResult initial_fit = ols_qr(X, y, rows, cols);
  if (initial_fit.betas == NULL) {
    fprintf(stderr, "Initial OLS fit failed\n");
    free(weights);
    free(X_weighted);
    free(y_weighted);
    free(y_pred);
    free_rlm_result(&result);
    return result;
  }

  // Copy initial betas
  for (int i = 0; i < cols_with_intercept; i++) {
    result.coefficients[i] = initial_fit.betas[i];
  }
  free_ols_result(&initial_fit);

  int iter;
  double converged_value = 0.0;

  for (iter = 0; iter < max_iter; iter++) {
    // 1. Compute predictions and residuals
    for (int i = 0; i < rows; i++) {
      double pred = result.coefficients[0]; // Intercept
      for (int j = 0; j < cols; j++) {
        pred += X[i * cols + j] * result.coefficients[j + 1];
      }
      y_pred[i] = pred;
      result.residuals[i] = y[i] - pred;
    }

    // 2. Compute robust scale estimate (MAD)
    double mad = compute_mad(result.residuals, rows);
    double sigma = mad / 0.6745; // Scale factor for normal distribution

    // Avoid division by zero
    if (sigma < 1e-10) {
      sigma = 1e-10;
    }

    // 3. Compute Huber weights
    for (int i = 0; i < rows; i++) {
      weights[i] = huber_weight(result.residuals[i], k, sigma);
    }

    // 4. Pre-weight design matrix and response vector
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

    // 5. Solve weighted least squares using OLS
    OLSResult ols_result = ols_qr(X_weighted, y_weighted, rows, cols);
    if (ols_result.betas == NULL) {
      fprintf(stderr, "Weighted OLS failed at iteration %d\n", iter);
      break;
    }

    // 6. Check convergence
    double max_change = 0.0;
    for (int i = 0; i < cols_with_intercept; i++) {
      double change = fabs(ols_result.betas[i] - result.coefficients[i]);
      if (change > max_change) {
        max_change = change;
      }
    }

    // Update coefficients
    for (int i = 0; i < cols_with_intercept; i++) {
      result.coefficients[i] = ols_result.betas[i];
    }

    // Store standard errors from last OLS fit
    for (int i = 0; i < cols_with_intercept; i++) {
      result.standard_errors[i] = ols_result.standard_errors[i];
      result.t_stats[i] = ols_result.t_stats[i];
    }

    free_ols_result(&ols_result);

    converged_value = max_change;
    if (max_change < tol) {
      result.converged = 1.0;
      break;
    }
  }

  // Update final fields
  result.iterations = iter + 1;
  if (!result.converged) {
    result.converged = converged_value; // Store final change if not converged
  }

  // Compute final predictions for R² calculation
  for (int i = 0; i < rows; i++) {
    double pred = result.coefficients[0]; // Intercept
    for (int j = 0; j < cols; j++) {
      pred += X[i * cols + j] * result.coefficients[j + 1];
    }
    y_pred[i] = pred;
  }

  // Calculate R2 and adjusted R2
  result.r2 = compute_rlm_r2(y, y_pred, rows);
  result.adjusted_r2 =
      1.0 - (1.0 - result.r2) *
                ((double)(rows - 1) / (double)(rows - cols_with_intercept));

  // Clean up
  free(weights);
  free(X_weighted);
  free(y_weighted);
  free(y_pred);

  return result;
}

void free_rlm_result(RLMResult *result) {
  if (result) {
    free(result->coefficients);
    free(result->standard_errors);
    free(result->t_stats);
    free(result->residuals);

    result->coefficients = NULL;
    result->standard_errors = NULL;
    result->t_stats = NULL;
    result->residuals = NULL;
    result->r2 = 0.0;
    result->adjusted_r2 = 0.0;
    result->iterations = 0;
    result->converged = 0.0;
  }
}

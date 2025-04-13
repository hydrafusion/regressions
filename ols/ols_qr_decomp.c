#include "ols_qr_decomp.h"
#include <lapacke.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Calculate residual standard error
static double calc_residual_std_error(double *residuals, int n, int p) {
  double sum_sq = 0.0;
  for (int i = 0; i < n; i++) {
    sum_sq += residuals[i] * residuals[i];
  }
  return sqrt(sum_sq / (n - p));
}

static double compute_r2(double *y, double *y_pred, int n) {
  double ss_total = 0.0, ss_residual = 0.0, mean_y = 0.0;
  for (int i = 0; i < n; i++) {
    mean_y += y[i];
  }
  mean_y /= n;

  for (int i = 0; i < n; i++) {
    ss_total += (y[i] - mean_y) * (y[i] - mean_y);
    ss_residual += (y[i] - y_pred[i]) * (y[i] - y_pred[i]);
  }
  return 1.0 - (ss_residual / ss_total);
}

OLSResult ols_qr(double *X, double *y, int rows, int cols) {
  if (rows <= cols) {
    fprintf(stderr, "Error: Need more observations than parameters\n");
    OLSResult empty = {NULL, NULL, NULL, NULL, 0.0, 0.0};
    return empty;
  }

  OLSResult result;
  int cols_with_intercept = cols + 1;
  result.betas = (double *)malloc(cols_with_intercept * sizeof(double));
  result.standard_errors =
      (double *)malloc(cols_with_intercept * sizeof(double));
  result.t_stats = (double *)malloc(cols_with_intercept * sizeof(double));
  result.residuals = (double *)malloc(rows * sizeof(double));

  // Create augmented X matrix with intercept column
  double *X_augmented =
      (double *)malloc(rows * cols_with_intercept * sizeof(double));
  for (int i = 0; i < rows; i++) {
    X_augmented[i * cols_with_intercept] = 1.0; // Intercept column
    for (int j = 0; j < cols; j++) {
      X_augmented[i * cols_with_intercept + (j + 1)] = X[i * cols + j];
    }
  }

  // Allocate memory for working copies
  double *X_copy =
      (double *)malloc(rows * cols_with_intercept * sizeof(double));
  double *y_copy = (double *)malloc(rows * sizeof(double));

  // Copy input arrays
  for (int i = 0; i < rows * cols_with_intercept; i++) {
    X_copy[i] = X_augmented[i];
  }
  for (int i = 0; i < rows; i++) {
    y_copy[i] = y[i];
  }

  // Allocate tau for QR factorization
  double *tau = (double *)malloc(cols_with_intercept * sizeof(double));

  // Query optimal work size
  double work_query;
  int lwork = -1;
  int info =
      LAPACKE_dgeqrf_work(LAPACK_ROW_MAJOR, rows, cols_with_intercept, X_copy,
                          cols_with_intercept, tau, &work_query, lwork);

  if (info != 0) {
    fprintf(stderr, "Work query failed\n");
    free_ols_result(&result);
    OLSResult empty = {NULL, NULL, NULL, NULL, 0.0, 0.0};
    return empty;
  }

  // Allocate optimal work array
  lwork = (int)work_query;
  double *work = (double *)malloc(lwork * sizeof(double));

  // Perform QR factorization
  info = LAPACKE_dgeqrf_work(LAPACK_ROW_MAJOR, rows, cols_with_intercept,
                             X_copy, cols_with_intercept, tau, work, lwork);
  if (info != 0) {
    fprintf(stderr, "QR factorization failed\n");
    free_ols_result(&result);
    OLSResult empty = {NULL, NULL, NULL, NULL, 0.0, 0.0};
    return empty;
  }

  // Apply Q^T to y
  info =
      LAPACKE_dormqr(LAPACK_ROW_MAJOR, 'L', 'T', rows, 1, cols_with_intercept,
                     X_copy, cols_with_intercept, tau, y_copy, 1);
  if (info != 0) {
    fprintf(stderr, "Applying Q^T failed\n");
    free_ols_result(&result);
    OLSResult empty = {NULL, NULL, NULL, NULL, 0.0, 0.0};
    return empty;
  }

  // Solve the triangular system
  info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'U', 'N', 'N', cols_with_intercept, 1,
                        X_copy, cols_with_intercept, y_copy, 1);
  if (info != 0) {
    fprintf(stderr, "Solving Rb = Q^Ty failed\n");
    free_ols_result(&result);
    OLSResult empty = {NULL, NULL, NULL, NULL, 0.0, 0.0};
    return empty;
  }

  // Copy solution to result (including intercept)
  for (int i = 0; i < cols_with_intercept; i++) {
    result.betas[i] = y_copy[i];
  }

  // Compute predicted values and residuals
  double *y_pred = (double *)malloc(rows * sizeof(double));
  for (int i = 0; i < rows; i++) {
    y_pred[i] = result.betas[0]; // Intercept
    for (int j = 0; j < cols; j++) {
      y_pred[i] += X[i * cols + j] * result.betas[j + 1];
    }
    result.residuals[i] = y[i] - y_pred[i];
  }

  // Calculate residual standard error
  double rse =
      calc_residual_std_error(result.residuals, rows, cols_with_intercept);

  // Calculate standard errors using the R matrix
  for (int i = 0; i < cols_with_intercept; i++) {
    double sum_sq = 0.0;
    for (int j = 0; j <= i; j++) {
      double r_ij = X_copy[j * cols_with_intercept + i];
      sum_sq += r_ij * r_ij;
    }
    result.standard_errors[i] = rse / sqrt(sum_sq);
    result.t_stats[i] = result.betas[i] / result.standard_errors[i];
  }

  result.r2 = compute_r2(y, y_pred, rows);
  result.adjusted_r2 =
      1.0 - (1.0 - result.r2) *
                ((double)(rows - 1) / (double)(rows - cols_with_intercept));

  // Free allocated memory
  free(work);
  free(tau);
  free(y_pred);
  free(X_copy);
  free(y_copy);
  free(X_augmented);

  return result;
}

void free_ols_result(OLSResult *result) {
  if (result) {
    free(result->betas);
    free(result->standard_errors);
    free(result->t_stats);
    free(result->residuals);

    result->betas = NULL;
    result->standard_errors = NULL;
    result->t_stats = NULL;
    result->residuals = NULL;
    result->r2 = 0.0;
    result->adjusted_r2 = 0.0;
  }
}

#include "ols/ols_qr_decomp.h"
#include "rlm/rlm_irls.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int COLUMNS = 10;
const int ROWS = 30;

// Function to generate random doubles between -1.0 and 1.0
double random_double() { return 2.0 * ((double)rand() / RAND_MAX) - 1.0; }

// Function to add outliers to data for testing robust regression
void add_outliers(double *y, int rows, int num_outliers) {
  printf("\nAdding %d outliers to data...\n", num_outliers);
  // Add some strong outliers to test robust regression
  for (int i = 0; i < num_outliers; i++) {
    int idx = rand() % rows;
    // Add a large positive or negative offset
    double outlier_value = (rand() % 2 == 0) ? 5.0 : -5.0;
    y[idx] += outlier_value;
    printf("Added outlier at index %d: y[%d] = %.4f\n", idx, idx, y[idx]);
  }
}

// Function to print regression results
void print_regression_coefficients(const char *method_name, double *betas,
                                   double *std_errors, double *t_stats,
                                   int cols) {
  printf("\n%s Results:\n", method_name);
  printf("Beta coefficients:\n");
  printf("%-10s %-15s %-15s %-15s\n", "Parameter", "Coefficient", "Std Error",
         "t-stat");
  printf("%-10s %-15.6f %-15.6f %-15.4f\n", "Intercept", betas[0],
         std_errors[0], t_stats[0]);

  for (int i = 1; i < cols + 1; i++) {
    printf("%-10s %-15.6f %-15.6f %-15.4f\n", "X", betas[i], std_errors[i],
           t_stats[i]);
  }
}

int main(void) {
  // Seed the random number generator
  srand(time(NULL));

  // Create X array (flattened matrix) - ROWS * COLUMNS elements
  double flat_X[ROWS * COLUMNS];

  // Fill X with random values
  for (int i = 0; i < ROWS * COLUMNS; i++) {
    flat_X[i] = random_double();
  }

  // Create y array - ROWS elements
  double y[ROWS];
  double y_with_outliers[ROWS];

  // Fill y with random values
  for (int i = 0; i < ROWS; i++) {
    // Create y as a linear combination of X variables plus noise
    y[i] = 1.0; // Intercept
    for (int j = 0; j < COLUMNS; j++) {
      y[i] += 0.5 * flat_X[i * COLUMNS + j];
    }
    y[i] += 0.2 * random_double(); // Add some noise

    // Keep a copy for adding outliers later
    y_with_outliers[i] = y[i];
  }

  // Print some sample values to verify
  printf("First few X values:\n");
  for (int i = 0; i < 5; i++) {
    printf("X[%d][0] = %.4f, X[%d][1] = %.4f\n", i, flat_X[i * COLUMNS], i,
           flat_X[i * COLUMNS + 1]);
  }

  printf("\nFirst few y values (without outliers):\n");
  for (int i = 0; i < 5; i++) {
    printf("y[%d] = %.4f\n", i, y[i]);
  }

  printf("\n=== PART 1: OLS REGRESSION (NO OUTLIERS) ===\n");

  // Run OLS regression
  OLSResult ols_result = ols_qr(flat_X, y, ROWS, COLUMNS);

  // Print results
  print_regression_coefficients("OLS", ols_result.betas,
                                ols_result.standard_errors, ols_result.t_stats,
                                COLUMNS);

  printf("\nModel Fit:\n");
  printf("R^2: %.6f\n", ols_result.r2);
  printf("Adjusted R^2: %.6f\n", ols_result.adjusted_r2);

  // Print a few residuals
  printf("\nFirst few residuals:\n");
  for (int i = 0; i < 5; i++) {
    printf("Residual[%d] = %.6f\n", i, ols_result.residuals[i]);
  }

  // Add some outliers to test the robust regression
  add_outliers(y_with_outliers, ROWS, 3);

  printf("\n=== PART 2: COMPARING OLS VS RLM WITH OUTLIERS ===\n");

  // Run OLS on data with outliers
  OLSResult ols_outlier_result = ols_qr(flat_X, y_with_outliers, ROWS, COLUMNS);

  // Run RLM with Huber weights on the same outlier data
  RLMResult rlm_result = rlm_huber(flat_X, y_with_outliers, ROWS, COLUMNS);

  // Print OLS results with outliers
  print_regression_coefficients("OLS with outliers", ols_outlier_result.betas,
                                ols_outlier_result.standard_errors,
                                ols_outlier_result.t_stats, COLUMNS);

  printf("\nOLS Model Fit with outliers:\n");
  printf("R^2: %.6f\n", ols_outlier_result.r2);
  printf("Adjusted R^2: %.6f\n", ols_outlier_result.adjusted_r2);

  // Print RLM results
  print_regression_coefficients(
      "RLM with Huber weights", rlm_result.coefficients,
      rlm_result.standard_errors, rlm_result.t_stats, COLUMNS);

  printf("\nRLM Model Fit:\n");
  printf("R^2: %.6f\n", rlm_result.r2);
  printf("Adjusted R^2: %.6f\n", rlm_result.adjusted_r2);
  printf("IRLS Iterations: %d\n", rlm_result.iterations);
  printf("Convergence measure: %.10f\n", rlm_result.converged);

  // Compare coefficients between original OLS and RLM
  printf("\n=== PART 3: COMPARISON OF COEFFICIENTS ===\n");
  printf("%-10s %-15s %-15s %-15s\n", "Parameter", "OLS (clean)",
         "OLS (outliers)", "RLM (outliers)");
  printf("%-10s %-15.6f %-15.6f %-15.6f\n", "Intercept", ols_result.betas[0],
         ols_outlier_result.betas[0], rlm_result.coefficients[0]);

  for (int i = 1; i < COLUMNS + 1; i++) {
    printf("%-10s %-15.6f %-15.6f %-15.6f\n", "X", ols_result.betas[i],
           ols_outlier_result.betas[i], rlm_result.coefficients[i]);
  }

  // Free allocated memory
  free_ols_result(&ols_result);
  free_ols_result(&ols_outlier_result);
  free_rlm_result(&rlm_result);

  return 0;
}

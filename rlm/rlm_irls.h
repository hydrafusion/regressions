#ifndef RLM_IRLS_H
#define RLM_IRLS_H

typedef struct {
  double *coefficients;     // Regression coefficients (including intercept)
  double *standard_errors;  // Standard errors of coefficients
  double *t_stats;          // t-statistics for coefficients
  double *residuals;        // Residuals from the model
  double r2;                // R-squared value
  double adjusted_r2;       // Adjusted R-squared
  int iterations;           // Number of IRLS iterations performed
  double converged;         // Convergence indicator (1.0 if converged, otherwise final change magnitude)
} RLMResult;

// Perform robust regression using Huber weights via IRLS
RLMResult rlm_huber(double *X, double *y, int rows, int cols);

// Free memory allocated for RLMResult
void free_rlm_result(RLMResult *result);

#endif

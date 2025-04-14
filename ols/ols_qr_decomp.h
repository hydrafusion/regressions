// multivariate_intercept.h
#ifndef OLS_QR_DECOMP_H
#define OLS_QR_DECOMP_H

typedef struct {
  double *betas;
  double *standard_errors;
  double *t_stats;
  double *residuals;
  double r2;
  double adjusted_r2;
} OLSResult;
    
// Expose the OLS function
OLSResult ols_qr(double *X, double *y, int rows, int cols);

// Optionally, create a function to free allocated memory
void free_ols_result(OLSResult *result);

#endif

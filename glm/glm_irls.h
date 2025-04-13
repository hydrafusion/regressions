// GLM_IRLS_H

#ifndef GLM_IRLS_H
#define GLM_IRLS_H

typedef struct {
  double *coefficients;  // Same as betas
  double *standard_errors;  
  double *t_stats;
  double *residuals;
  double r2;
  double adjusted_r2;
  int iterations;        // New: Number of IRLS iterations performed
  double converged;      // New: Convergence indicator (0 or 1, or use tolerance value)
  // You can add more fields as needed (e.g., link function info)
} GLMResult;

GLMResult glm_irls(double *X, double *y, int rows, int cols);


#endif


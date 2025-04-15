# Regression Implmentation with LAPACK QR Decomposition

## Overview

This project implements robust and traditional regression models in C by
leveraging LAPACK routines. Its core objective is to deliver high-performance
regression analyses while remaining minimalist—eschewing heavy abstractions for
rapid computation.

The library is designed to be easily bindable to other languages, such as Rust,
for use in data analysis libraries like [Polars](https://www.pola.rs/), or used
in OLAP databases such as Clickhouse.

## Features

- **Ordinary Least Squares (OLS):** Utilize closed-form solutions with highly
  optimized matrix operations.
- **Generalized Linear Models (GLM):** Implement flexible regression models
  (e.g., logistic or Poisson regression) with iterative solvers.
- **Robust Linear Models (RLM):** Incorporate robust methods (such as Huber
  weighting) to mitigate the impact of outliers.
- **Interoperability:** Simple C interface allows seamless bindings to languages
  like Rust, aiding integrations in libraries like Polars.
- **Performance-Centric:** Direct use of LAPACK routines in C ensures minimal
  overhead and maximum computational performance.

## Motivation

The goal of this project is to harness the raw performance of LAPACK routines in
C without the heavy bloat introduced by layers of abstraction. This approach
ensures that computational tasks are efficiently integrated into modern data
analysis ecosystems. By keeping the design lean, it promotes speed and resource
efficiency, making it ideal for high-volume and real-time data processing tasks.

## Compilation

To compile the regressions, you must link lapack, also the differnt header files
that you wish to use.

```
gcc -o test_regression test.c ols/ols_qr_decomp.c -lopenblas -llapacke -lm
```

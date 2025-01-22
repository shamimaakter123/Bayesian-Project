data {
    int<lower=0> N;  // Number of observations
    int<lower=0> K;  // Number of predictors
    matrix[N, K] X;  // Predictor matrix
    vector[N] y;     // Response vector
}

parameters {
    vector[K] beta;  // Coefficients for predictors
    real alpha;      // Intercept
    real<lower=0> sigma;  // Standard deviation of the errors
}

model {
    y ~ normal(X * beta + alpha, sigma);  // Likelihood
}

generated quantities {
    vector[N] y_pred;  // Posterior predictions
    for (n in 1:N) {
        y_pred[n] = normal_rng(X[n] * beta + alpha, sigma);  // Generate predicted values
    }
}

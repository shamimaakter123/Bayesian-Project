data {
    int<lower=0> N; // Number of observations
    int<lower=0> K; // Number of predictors
    matrix[N, K] X; // Predictor variables
    vector[N] y; // Outcome variable
    int column_index_of_age; // Index for age in the X matrix
    int column_index_of_bmi; // Index for BMI in the X matrix
}

transformed data {
    vector[N] age_squared;
    vector[N] bmi_squared;
    for (i in 1:N) {
        age_squared[i] = square(X[i, column_index_of_age]);
        bmi_squared[i] = square(X[i, column_index_of_bmi]);
    }
}

parameters {
    vector[K] beta; // Coefficients for predictors
    real alpha; // Intercept
    real beta_age_squared; // Coefficient for age squared
    real beta_bmi_squared; // Coefficient for BMI squared
    real<lower=0> sigma; // Standard deviation
}

model {
    // Likelihood incorporating both linear and non-linear terms
    y ~ normal(X * beta + alpha + beta_age_squared * age_squared + beta_bmi_squared * bmi_squared, sigma);
}

generated quantities {
    vector[N] y_pred;
    for (n in 1:N) {
        y_pred[n] = normal_rng(alpha + X[n] * beta + beta_age_squared * age_squared[n] + beta_bmi_squared * bmi_squared[n], sigma);
    }
}


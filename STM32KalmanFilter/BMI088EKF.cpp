#include "BMI088EKF.h"

// For timing measurements (adapt to your STM32 setup)
#ifdef STM32F4XX
#include "stm32f4xx_hal.h"
#define GET_MICROSECONDS() (HAL_GetTick() * 1000 + (1000 - SysTick->VAL / (SystemCoreClock/1000000)))
#else
// Fallback for other platforms
#define GET_MICROSECONDS() 0
#endif

BMI088EKF::BMI088EKF(float dt, float max_accel)
    : dt_(dt), max_accel_(max_accel), consecutive_failures_(0),
      max_uncertainty_threshold_(50.0f), outlier_threshold_(3.0f) {

    // Initialize performance stats
    memset(&stats_, 0, sizeof(stats_));

    // Initialize last valid acceleration
    memset(last_valid_accel_, 0, sizeof(last_valid_accel_));

    // Initialize with conservative estimates for high-G applications
    reset();
}

void BMI088EKF::reset(const float* initial_accel) {
    // Reset state
    if (initial_accel) {
        memcpy((void*)x_, initial_accel, sizeof(x_));
        memcpy(last_valid_accel_, initial_accel, sizeof(last_valid_accel_));
    } else {
        memset((void*)x_, 0, sizeof(x_));
        // Assume 1G downward initially (rocket on pad)
        x_[2] = -9.81f;
        memcpy(last_valid_accel_, (void*)x_, sizeof(x_));
    }

    // Conservative initial covariance for rocket applications
    matrixIdentity3x3(P_);
    for (int i = 0; i < N_STATES; i++) {
        P_[i][i] = 10.0f;  // Higher initial uncertainty
    }

    // Default noise parameters (will be adapted based on flight phase)
    adaptToFlightPhase(0);  // Start in pad phase

    consecutive_failures_ = 0;
    stats_.reset_count++;
}

uint32_t BMI088EKF::predict() {
    uint32_t start_time = GET_MICROSECONDS();

    // State transition: x_k+1 = x_k (constant acceleration model)
    // No change needed since accelerations persist

    // Covariance prediction: P = F*P*F^T + Q
    // Since F = I (identity), this simplifies to: P = P + Q
    matrixAdd3x3(P_, Q_, P_);

    uint32_t end_time = GET_MICROSECONDS();
    uint32_t execution_time = end_time - start_time;

    stats_.predict_time_us = execution_time;
    return execution_time;
}

bool BMI088EKF::update(const float z[3]) {
    uint32_t start_time = GET_MICROSECONDS();

    // Outlier detection - critical for rocket flight
    if (!isValidMeasurement(z)) {
        consecutive_failures_++;
        stats_.rejected_measurements++;
        return false;
    }

    // Innovation (measurement residual)
    float y[N_STATES];
    y[0] = z[0] - x_[0];
    y[1] = z[1] - x_[1];
    y[2] = z[2] - x_[2];

    // Innovation covariance: S = H*P*H^T + R
    // Since H = I, this simplifies to: S = P + R
    matrixAdd3x3(P_, R_, S_);

    // Kalman gain: K = P * H^T * S^(-1) = P * S^(-1)
    if (!matrixInverse3x3(S_, temp1_)) {
        // Matrix inversion failed - numerical issues
        consecutive_failures_++;
        return false;
    }

    matrixMultiply3x3(P_, temp1_, K_);

    // Update state: x = x + K*y
    float K_y[N_STATES];
    matrixVectorMultiply3x3(K_, y, K_y);

    // Thread-safe state update
    volatile float* x_ptr = x_;
    x_ptr[0] += K_y[0];
    x_ptr[1] += K_y[1];
    x_ptr[2] += K_y[2];

    // Update covariance (Joseph form for numerical stability)
    // P = (I - K*H) * P * (I - K*H)^T + K*R*K^T
    // Since H = I: P = (I - K) * P * (I - K)^T + K*R*K^T

    // Compute (I - K)
    matrixIdentity3x3(temp1_);
    matrixSubtract3x3(temp1_, K_, temp2_);  // temp2 = I - K

    // Compute (I - K) * P * (I - K)^T
    matrixMultiply3x3(temp2_, P_, temp1_);

    // Transpose of (I - K) is the same since it's computed from identity
    matrixMultiply3x3(temp1_, temp2_, P_);

    // Add K*R*K^T term for Joseph form stability
    matrixMultiply3x3(K_, R_, temp1_);
    matrixMultiply3x3(temp1_, temp2_, temp1_);  // Reuse temp2 as K^T
    matrixAdd3x3(P_, temp1_, P_);

    // Store last valid measurement
    memcpy(last_valid_accel_, z, sizeof(last_valid_accel_));
    consecutive_failures_ = 0;

    uint32_t end_time = GET_MICROSECONDS();
    stats_.update_time_us = end_time - start_time;
    stats_.total_cycles++;

    return true;
}

void BMI088EKF::getFilteredAcceleration(float result[3]) const {
    // Thread-safe read of volatile state
    result[0] = x_[0];
    result[1] = x_[1];
    result[2] = x_[2];
}

void BMI088EKF::getAccelerationUncertainty(float result[3]) const {
    result[0] = sqrtf(P_[0][0]);
    result[1] = sqrtf(P_[1][1]);
    result[2] = sqrtf(P_[2][2]);
}

void BMI088EKF::adaptToFlightPhase(int flight_phase) {
    // Clear noise matrices
    memset(Q_, 0, sizeof(Q_));
    memset(R_, 0, sizeof(R_));

    switch (flight_phase) {
        case 0: // Pad phase
            Q_[0][0] = Q_[1][1] = 0.01f;  // Very low horizontal noise
            Q_[2][2] = 0.05f;             // Slightly higher vertical noise
            R_[0][0] = R_[1][1] = R_[2][2] = 0.1f;  // Low sensor noise
            outlier_threshold_ = 2.0f;    // Strict outlier detection
            break;

        case 1: // Boost phase (high acceleration, vibration)
            Q_[0][0] = Q_[1][1] = Q_[2][2] = 5.0f;  // High process noise
            R_[0][0] = R_[1][1] = R_[2][2] = 2.0f;  // Higher sensor noise due to vibration
            outlier_threshold_ = 5.0f;    // More tolerant of outliers
            break;

        case 2: // Coast phase (low acceleration, stable)
            Q_[0][0] = Q_[1][1] = Q_[2][2] = 0.1f;  // Low process noise
            R_[0][0] = R_[1][1] = R_[2][2] = 0.2f;  // Normal sensor noise
            outlier_threshold_ = 3.0f;    // Standard outlier detection
            break;

        case 3: // Descent phase (parachute deployment, variable acceleration)
            Q_[0][0] = Q_[1][1] = Q_[2][2] = 2.0f;  // Moderate process noise
            R_[0][0] = R_[1][1] = R_[2][2] = 1.0f;  // Moderate sensor noise
            outlier_threshold_ = 4.0f;    // Tolerant outlier detection
            break;
    }
}

bool BMI088EKF::isValidMeasurement(const float z[3]) const {
    // Check for NaN or infinite values
    for (int i = 0; i < N_STATES; i++) {
        if (!std::isfinite(z[i])) {
            return false;
        }
    }

    // Check magnitude against maximum expected acceleration
    float magnitude = sqrtf(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
    if (magnitude > max_accel_) {
        return false;
    }

    // Check for sudden jumps from last valid measurement
    float max_change = 0.0f;
    for (int i = 0; i < N_STATES; i++) {
        float change = fabsf(z[i] - last_valid_accel_[i]);
        if (change > max_change) {
            max_change = change;
        }
    }

    // Adaptive threshold based on current uncertainty
    float uncertainty[N_STATES];
    getAccelerationUncertainty(uncertainty);
    float max_uncertainty = fmaxf(fmaxf(uncertainty[0], uncertainty[1]), uncertainty[2]);

    float adaptive_threshold = outlier_threshold_ * (1.0f + max_uncertainty);

    return max_change < adaptive_threshold * 100.0f;  // 100 m/s² threshold scale
}

bool BMI088EKF::isHealthy() const {
    // Check for excessive consecutive failures
    if (consecutive_failures_ > 10) {
        return false;
    }

    // Check uncertainty levels
    float uncertainty[N_STATES];
    getAccelerationUncertainty(uncertainty);

    for (int i = 0; i < N_STATES; i++) {
        if (uncertainty[i] > max_uncertainty_threshold_) {
            return false;
        }
    }

    // Check for numerical issues in covariance matrix
    for (int i = 0; i < N_STATES; i++) {
        if (!std::isfinite(P_[i][i]) || P_[i][i] < 0.0f) {
            return false;
        }
    }

    return true;
}

bool BMI088EKF::checkNumericalStability() {
    // Check state vector
    for (int i = 0; i < N_STATES; i++) {
        if (!std::isfinite(x_[i])) {
            return false;
        }
    }

    // Check covariance matrix
    for (int i = 0; i < N_STATES; i++) {
        for (int j = 0; j < N_STATES; j++) {
            if (!std::isfinite(P_[i][j])) {
                return false;
            }
        }
        // Diagonal elements must be positive
        if (P_[i][i] <= 0.0f) {
            return false;
        }
    }

    return true;
}

void BMI088EKF::getPerformanceStats(PerfStats& stats) const {
    stats = stats_;
}

inline bool BMI088EKF::matrixInverse3x3(const float A[N_STATES][N_STATES],
                                             float result[N_STATES][N_STATES]) {
    // Optimized 3x3 matrix inversion with stability checking
    float det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);

    // Check for near-singular matrix
    const float MIN_DET = 1e-12f;
    if (fabsf(det) < MIN_DET) {
        return false;  // Matrix is singular or nearly singular
    }

    float inv_det = 1.0f / det;

    result[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) * inv_det;
    result[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) * inv_det;
    result[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) * inv_det;
    result[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) * inv_det;
    result[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) * inv_det;
    result[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) * inv_det;
    result[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) * inv_det;
    result[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) * inv_det;
    result[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) * inv_det;

    return true;
}

inline void BMI088EKF::matrixVectorMultiply3x3(const float mat[N_STATES][N_STATES],
                                                    const float vec[N_STATES],
                                                    float result[N_STATES]) {
    result[0] = mat[0][0]*vec[0] + mat[0][1]*vec[1] + mat[0][2]*vec[2];
    result[1] = mat[1][0]*vec[0] + mat[1][1]*vec[1] + mat[1][2]*vec[2];
    result[2] = mat[2][0]*vec[0] + mat[2][1]*vec[1] + mat[2][2]*vec[2];
}

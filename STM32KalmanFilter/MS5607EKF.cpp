#include "MS5607EKF.h"

// For timing measurements (adapt to your STM32 setup)
#ifdef STM32F4XX
#include "stm32f4xx_hal.h"
#define GET_MICROSECONDS() (HAL_GetTick() * 1000 + (1000 - SysTick->VAL / (SystemCoreClock/1000000)))
#else
// Fallback for other platforms
#define GET_MICROSECONDS() 0
#endif

MS5607EKF::MS5607EKF(float dt, float max_altitude)
    : dt_(dt), max_altitude_(max_altitude),
      sea_level_pressure_(101325.0f), reference_temperature_(15.0f),
      altitude_(0.0f), P_(100.0f),
      consecutive_failures_(0), max_uncertainty_threshold_(100.0f),
      outlier_threshold_(50.0f), last_valid_altitude_(0.0f),
      altitude_jump_threshold_(100.0f) {

    // Initialize performance stats
    memset(&stats_, 0, sizeof(stats_));
    stats_.min_altitude = 999999.0f;  // Initialize to high value
    stats_.max_altitude = -999999.0f; // Initialize to low value

    // Initialize with conservative estimates
    reset();
}

void MS5607EKF::reset(const float* initial_altitude) {
    // Reset state
    if (initial_altitude) {
        altitude_ = *initial_altitude;
        last_valid_altitude_ = *initial_altitude;
    } else {
        altitude_ = 0.0f;  // Will be set by first measurement
        last_valid_altitude_ = 0.0f;
    }

    // Conservative initial covariance (uncertainty)
    P_ = 100.0f;  // 100m initial altitude uncertainty

    // Default noise parameters (will be adapted based on flight phase)
    adaptToFlightPhase(0);  // Start in ground phase

    consecutive_failures_ = 0;
    stats_.reset_count++;
}

uint32_t MS5607EKF::predict() {
    uint32_t start_time = GET_MICROSECONDS();

    // State prediction: altitude remains constant (simple model)
    // altitude(k+1) = altitude(k)  [no change needed]

    // Covariance prediction: P = P + Q
    P_ += Q_;

    uint32_t end_time = GET_MICROSECONDS();
    uint32_t execution_time = end_time - start_time;

    stats_.predict_time_us = execution_time;
    return execution_time;
}

bool MS5607EKF::update(float altitude_measurement) {
    uint32_t start_time = GET_MICROSECONDS();

    // Outlier detection
    if (!isValidMeasurement(altitude_measurement)) {
        consecutive_failures_++;
        stats_.rejected_measurements++;
        return false;
    }

    // Innovation (measurement residual)
    float y = altitude_measurement - altitude_;

    // Innovation covariance: S = P + R
    float S = P_ + R_;

    // Check for numerical issues
    if (S <= 0.0f) {
        consecutive_failures_++;
        return false;
    }

    // Kalman gain: K = P / S
    float K = P_ / S;

    // Update state: altitude = altitude + K * y
    altitude_ += K * y;

    // Update covariance: P = (1 - K) * P
    P_ = (1.0f - K) * P_;

    // Ensure P doesn't become negative due to numerical issues
    if (P_ < 0.01f) {
        P_ = 0.01f;  // Minimum uncertainty of 10cm
    }

    // Store last valid measurement
    last_valid_altitude_ = altitude_measurement;
    consecutive_failures_ = 0;

    // Update performance statistics
    stats_.average_altitude = (stats_.average_altitude * 0.99f) + (altitude_measurement * 0.01f);
    if (altitude_measurement < stats_.min_altitude) stats_.min_altitude = altitude_measurement;
    if (altitude_measurement > stats_.max_altitude) stats_.max_altitude = altitude_measurement;

    uint32_t end_time = GET_MICROSECONDS();
    stats_.update_time_us = end_time - start_time;
    stats_.total_cycles++;

    return true;
}

float MS5607EKF::getFilteredAltitude() const {
    return altitude_;
}

float MS5607EKF::getAltitudeUncertainty() const {
    return sqrtf(P_);
}

void MS5607EKF::adaptToFlightPhase(int flight_phase) {
    switch (flight_phase) {
        case 0: // Ground phase
            Q_ = 0.1f;         // Low process noise (altitude changes slowly)
            R_ = 2.0f;         // Altitude measurement noise
            outlier_threshold_ = 10.0f;
            altitude_jump_threshold_ = 20.0f;
            break;

        case 1: // Ascent phase (rapid altitude changes)
            Q_ = 5.0f;         // Higher process noise for rapid changes
            R_ = 5.0f;         // Higher measurement noise during acceleration
            outlier_threshold_ = 50.0f;
            altitude_jump_threshold_ = 100.0f;
            break;

        case 2: // Coast phase (smooth trajectory)
            Q_ = 1.0f;         // Moderate process noise
            R_ = 3.0f;         // Normal measurement noise
            outlier_threshold_ = 30.0f;
            altitude_jump_threshold_ = 50.0f;
            break;

        case 3: // Descent phase (parachute deployment, variable dynamics)
            Q_ = 3.0f;         // Moderate-high process noise
            R_ = 4.0f;         // Higher measurement noise during descent
            outlier_threshold_ = 40.0f;
            altitude_jump_threshold_ = 75.0f;
            break;
    }
}

void MS5607EKF::setBarometricReference(float sea_level_pressure, float temperature) {
    sea_level_pressure_ = sea_level_pressure;
    reference_temperature_ = temperature;
}

float MS5607EKF::pressureToAltitude(float pressure) const {
    // Standard barometric formula
    // h = (T0/L) * (1 - (p/p0)^(R*L/g*M))
    // Simplified version for typical applications

    const float R = 287.053f;      // Specific gas constant for dry air (J/kg/K)
    const float g = 9.80665f;      // Standard gravity (m/s²)
    const float L = 0.0065f;       // Temperature lapse rate (K/m)
    const float T0 = reference_temperature_ + 273.15f;  // Temperature in Kelvin

    if (pressure <= 0.0f || sea_level_pressure_ <= 0.0f) {
        return 0.0f;  // Invalid pressure reading
    }

    float pressure_ratio = pressure / sea_level_pressure_;

    // Prevent numerical issues with extreme ratios
    if (pressure_ratio > 1.0f) pressure_ratio = 1.0f;
    if (pressure_ratio < 0.01f) pressure_ratio = 0.01f;

    float altitude = (T0 / L) * (1.0f - powf(pressure_ratio, (R * L) / g));

    return altitude;
}

bool MS5607EKF::isValidMeasurement(float altitude) const {
    // Check for NaN or infinite values
    if (!std::isfinite(altitude)) {
        return false;
    }

    // Check altitude range
    if (altitude < -1000.0f || altitude > max_altitude_) {
        return false;
    }

    // Check for sudden altitude jumps (only if we have a previous measurement)
    if (stats_.total_cycles > 0) {
        float altitude_change = fabsf(altitude - last_valid_altitude_);
        if (altitude_change > altitude_jump_threshold_) {
            return false;
        }
    }

    return true;
}

bool MS5607EKF::isHealthy() const {
    // Check for excessive consecutive failures
    if (consecutive_failures_ > 10) {
        return false;
    }

    // Check uncertainty level
    if (sqrtf(P_) > max_uncertainty_threshold_) {
        return false;
    }

    // Check for numerical issues
    if (!std::isfinite(altitude_) || !std::isfinite(P_) || P_ <= 0.0f) {
        return false;
    }

    // Check for reasonable altitude
    if (altitude_ < -2000.0f || altitude_ > max_altitude_) {
        return false;
    }

    return true;
}

void MS5607EKF::getPerformanceStats(PerfStats& stats) const {
    stats = stats_;
}

uint32_t MS5607EKF::getMicroseconds() const {
    return GET_MICROSECONDS();
}

#include <Wire.h>

#define TCA_ADDR 0x70      // TCA9548A default address
#define MPU_ADDR 0x68      // MPU6050 default I2C address
#define ACCEL_XOUT_H 0x3B  // Register for reading accelerometer X high byte 
#define PWR_MGMT_1 0x6B    // Regsiter to control IMU power management 

// Map TCA channels to fingers
const uint8_t fingerChannel[5] = {6, 5, 4, 3, 2}; 
const char* fingerName[5] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};
bool prevTouchState[5] = {false, false, false, false, false};
bool tapInProgress[5] = {false};

// Touch sensor pins (TTP223)
const uint8_t touchPin[5] = {11, 10, 9, 8, 7};  

// Tap detection parameters
bool imuTapDetected[5] = {false};
unsigned long imuTapTime[5] = {0};

float prevSliderValue[5] = {0, 0, 0, 0, 0};

int sliderPrintCounter[5] = {0};

const float fingerThreshold[5] = {
    1.15, // Thumb
    1.4,  // Index
    1.5,  // Middle
    1.5,  // Ring
    0.96  // Pinky
};

const unsigned long DEBOUNCE_US = 50; // simple debounce
unsigned long lastTapTime[5] = {0, 0, 0, 0, 0}; // Store last tap time for each finger

bool tappedActive[5];
int activeSliderFinger = -1; 

// Slide detection tuning paramters 
const float SLIDE_ACCEL_LOW = 0.85;              // lower bound of acceleration magnitude near 1g
const float SLIDE_ACCEL_HIGH = 1.25;             // upper bound of acceleration magnitude (no large spikes)
const float SLIDE_GYRO_THRESHOLD = 10;           // deg/sec minimal sliding rotation
const unsigned long SLIDE_MIN_DURATION = 20000;  // 20ms to confirm slide
const unsigned long SLIDE_TIMEOUT = 150000;      // 150ms without motion → stop
const float SLIDE_SMOOTHING = 0.1;  // 0 = no smoothing, 1 = instant jump
float smoothedMotion[5] = {0,0,0,0,0};


// --- LOOP STATE ---
uint8_t currentIMU = 0;          // Which IMU to read this loop
unsigned long lastIMURead = 0;   // Timing for IMU cycle
const unsigned long IMU_INTERVAL_US = 6000;  // Read one IMU every 8 ms

// ---- SLIDER VARIABLES ----
bool sliding[5] = {false, false, false, false, false};
unsigned long lastSlideMotion[5] = {0,0,0,0,0};
unsigned long slideStartTime[5] = {0,0,0,0,0};

float sliderValue[5] = {0,0,0,0,0};     // output (0–100 or whatever range you want)
float slideVelocity = 0;
unsigned long lastIMUTime[5] = {0};

const float SLIDE_GYRO_DEADZONE = 4.0;     // ignore tiny noise
const float SLIDE_GAIN = 0.05;             // how fast slider changes
const unsigned long SLIDE_IDLE_CUTOFF = 70000; // 70ms without movement = stop sliding

// Select channel on TCA9548A multiplexer
void tcaSelect(uint8_t channel) {
  Wire.beginTransmission(TCA_ADDR);   // Begin I2C transmission to multiplexer
  Wire.write(1 << channel);           // Send byte to select the channel
  Wire.endTransmission();             // End I2C transmission
}
// Wake up the MPU6050 IMU
void wakeIMU() {
  Wire.beginTransmission(MPU_ADDR);  // Begin I2C transmission to IMU
  Wire.write(PWR_MGMT_1);            // Access power management register
  Wire.write(0);                      // Write 0 to wake up IMU from sleep
  Wire.endTransmission();             // End I2C transmission
}

// Read accelerometer magnitude only
float readAccelMagnitude() {
  Wire.beginTransmission(MPU_ADDR);  // Begin I2C transmission
  Wire.write(ACCEL_XOUT_H);          // Point to accel registers
  Wire.endTransmission(false);       // End transmission but keep I2C active

  Wire.requestFrom(MPU_ADDR, 6);     // Request 6 bytes (X, Y, Z)
  if (Wire.available() < 6) return 0;  // Error check

  int16_t ax = (Wire.read() << 8) | Wire.read();  // Read accel X
  int16_t ay = (Wire.read() << 8) | Wire.read();  // Read accel Y
  int16_t az = (Wire.read() << 8) | Wire.read();  // Read accel Z

  // Convert raw values to g
  float gx = ax / 16384.0;
  float gy = ay / 16384.0;
  float gz = az / 16384.0;

  return sqrt(gx*gx + gy*gy + gz*gz);  // Return magnitude
}

// Read full IMU data (accel + gyro)
void readIMU(float &ax, float &ay, float &az, float &gx, float &gy, float &gz) {
  Wire.beginTransmission(MPU_ADDR);  // Start I2C transmission
  Wire.write(0x3B);                  // ACCEL_XOUT_H register
  Wire.endTransmission(false);       // End transmission but keep bus active
  Wire.requestFrom(MPU_ADDR, 14);    // Request 14 bytes (accel, temp, gyro)

  int16_t rawAx = (Wire.read() << 8) | Wire.read();  // Read accel X
  int16_t rawAy = (Wire.read() << 8) | Wire.read();  // Read accel Y
  int16_t rawAz = (Wire.read() << 8) | Wire.read();  // Read accel Z

  Wire.read(); Wire.read();  // Skip temperature

  int16_t rawGx = (Wire.read() << 8) | Wire.read();  // Read gyro X
  int16_t rawGy = (Wire.read() << 8) | Wire.read();  // Read gyro Y
  int16_t rawGz = (Wire.read() << 8) | Wire.read();  // Read gyro Z

  // Convert raw accelerometer to g
  ax = rawAx / 16384.0;
  ay = rawAy / 16384.0;
  az = rawAz / 16384.0;

  // Convert raw gyro to deg/sec
  gx = rawGx / 131.0;
  gy = rawGy / 131.0;
  gz = rawGz / 131.0;
}


// Setup function runs once
void setup() {
  Serial.begin(115200);  // Initialize serial output
  Wire.begin();           // Initialize I2C
  Wire.setWireTimeout(3000, true);
  delay(1000);            // Small delay for stabilization

  // Configure touch sensor pins
  for (int i = 0; i < 5; i++) {
    pinMode(touchPin[i], INPUT);  // TTP223 touch sensors are digital inputs
  }

  // Wake up all IMUs
  Serial.println("Waking IMUs...");
  for (uint8_t i = 0; i < 5; i++) {
    tcaSelect(fingerChannel[i]);  // Select finger channel
    wakeIMU();                    // Wake IMU
  }
  Serial.println("5-finger tap detection started!");
}

void loop() {
    unsigned long now = micros();

    // -------------------------------
    // FAST LOOP (runs every iteration)
    // Reads touch sensors instantly
    // -------------------------------
    for (uint8_t i = 0; i < 5; i++) {
        bool touched = (digitalRead(touchPin[i]) == HIGH);
        bool wasTouched = prevTouchState[i];

        // LIFT DETECTION (only if previously tapped)
        if (tappedActive[i] && !touched) {
            Serial.print(fingerName[i]);
            Serial.println(" finger lifted");

            tappedActive[i] = false;    // Reset state
            // Serial.println(tappedActive[i]);
        }

        // ---- Detect tap (requires IMU tap window set by slow loop) ----
        bool debounceOK = (now - lastTapTime[i] > DEBOUNCE_US);

        // TAP DETECTION
        if (!tappedActive[i] && touched && imuTapDetected[i] && debounceOK) {
            Serial.print(fingerName[i]);
            Serial.println(" finger tapped");
            tappedActive[i] = true;     // Now the finger is in a "tapped" state
            // Serial.println(tappedActive[i]);
            lastTapTime[i] = now;
        }


        prevTouchState[i] = touched;
    }

    // -------------------------------------------------
    // SLOW LOOP (runs every 3 ms)
    // Reads ONE IMU each cycle
    // -------------------------------------------------
    if (now - lastIMURead >= IMU_INTERVAL_US) {
        lastIMURead = now;

        // Select next IMU
        uint8_t i = currentIMU;

        tcaSelect(fingerChannel[i]);
        
        float ax, ay, az, gx, gy, gz;
        readIMU(ax, ay, az, gx, gy, gz);

        // ---- TAP DETECTION ----
        float mag = sqrt(ax*ax + ay*ay + az*az);

        // if (i == 4 && mag > 1.15) {
        //   Serial.print(" mag=");
        //   Serial.println(mag);
        // }

        if (mag > fingerThreshold[i]) {
            imuTapDetected[i] = true;
            imuTapTime[i] = now;
        }

        if (now - imuTapTime[i] > 90000) {  // 90 ms tap window
            imuTapDetected[i] = false;
        }

        // Optionally use gyro for slide
        if (tappedActive[i]) {  // Only detect slider if finger was tapped
            detectSlider(i, prevTouchState[i], gx, now);
        }

        // Next IMU for next cycle
        currentIMU = (currentIMU + 1) % 5;
    }
}


bool detectSlider(int finger, bool touched, float gx, unsigned long now) {

    // --- Check if another finger is currently sliding ---
    if (activeSliderFinger != -1 && activeSliderFinger != finger) {
        // Another finger is sliding → ignore this finger
        sliding[finger] = false;
        return false;
    }

    // 1. If touch is OFF → reset sliding state
    if (!touched) {
        sliding[finger] = false;
        slideStartTime[finger] = 0;

        // If this finger was active, release it
        if (activeSliderFinger == finger) {
            activeSliderFinger = -1;
        }

        return false;
    }

    // 2. Motion timing
    unsigned long dtMicros = now - lastIMUTime[finger];
    lastIMUTime[finger] = now;
    float dt = dtMicros / 1000000.0;    // seconds

    // 3. Use GYRO X as sliding axis
    float motion = gx;

    // special case for thumb
    if (finger == 0) motion = -gx; 

    // 4. Ignore tiny jitter
    if (abs(motion) < SLIDE_GYRO_DEADZONE) {
        if (sliding[finger] &&
            now - lastSlideMotion[finger] > SLIDE_IDLE_CUTOFF) {

            sliding[finger] = false;
            if (activeSliderFinger == finger) activeSliderFinger = -1;
        }
        return sliding[finger];
    }

    // 5. Motion detected
    lastSlideMotion[finger] = now;

    // SLIDE START
    if (!sliding[finger]) {
        sliding[finger] = true;
        activeSliderFinger = finger;

        // Reset slider value to 0 at the start
           sliderValue[finger] = 0.0;
           prevSliderValue[finger] = 0.0;
        // Serial.print(fingerName[finger]);
        // Serial.println(" SLIDE START");
    }

    // 6. Integrate slider movement
    smoothedMotion[finger] = smoothedMotion[finger] + SLIDE_SMOOTHING * (motion - smoothedMotion[finger]);
    sliderValue[finger] += smoothedMotion[finger] * SLIDE_GAIN * dt * 1000.0;

    // Clamp slider between 0 and 100
    sliderValue[finger] = constrain(sliderValue[finger], 0, 100);

    // Only print if slider value actually changed
    if (sliderValue[finger] != prevSliderValue[finger]) {
        sliderPrintCounter[finger]++; 
        if (sliderPrintCounter[finger] >= 3) {
            Serial.print(fingerName[finger]);
            Serial.print(" SLIDER: ");
            Serial.println(sliderValue[finger], 2);  // 2 decimal places
            sliderPrintCounter[finger] = 0; 
        }
        prevSliderValue[finger] = sliderValue[finger];
    }

    return true;
}

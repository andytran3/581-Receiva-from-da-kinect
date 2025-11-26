#include <Wire.h>

#define TCA_ADDR 0x70      // TCA9548A default address
#define MPU_ADDR 0x68      // MPU6050 default I2C address
#define ACCEL_XOUT_H 0x3B  // Register for reading accelerometer X high byte 
#define PWR_MGMT_1 0x6B    // Regsiter to control IMU power management 

// Map TCA channels to fingers
const uint8_t fingerChannel[5] = {6, 5, 4, 3, 2}; 
const char* fingerName[5] = {"Thumb", "Index", "Middle", "Ring", "Pinky"};

// Touch sensor pins (TTP223)
const uint8_t touchPin[5] = {6, 5, 4, 3, 2};  

// Tap detection parameters
const float TAP_THRESHOLD = 2.0;      // Acceleration magnitude threshold for detecting taps (in g)
const unsigned long DEBOUNCE_US = 50; // simple debounce
unsigned long lastTapTime[5] = {0, 0, 0, 0, 0}; // Store last tap time for each finger

// Slide detection tuning paramters 
const float SLIDE_ACCEL_LOW = 0.85;              // lower bound of acceleration magnitude near 1g
const float SLIDE_ACCEL_HIGH = 1.25;             // upper bound of acceleration magnitude (no large spikes)
const float SLIDE_GYRO_THRESHOLD = 10;           // deg/sec minimal sliding rotation
const unsigned long SLIDE_MIN_DURATION = 20000;  // 20ms to confirm slide
const unsigned long SLIDE_TIMEOUT = 150000;      // 150ms without motion → stop

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

// Loop function runs continuously
void loop() {
  unsigned long now = micros();  // Current time in microseconds

  // Loop over each finger
  for (uint8_t i = 0; i < 5; i++) {

    // 1. Read touch sensor
    bool touched = (digitalRead(touchPin[i]) == HIGH);

    // 2. Read accelerometer magnitude for tap detection
    tcaSelect(fingerChannel[i]);  
    delayMicroseconds(200);       // Small settle delay

    float ax, ay, az, gx, gy, gz;
    readIMU(ax, ay, az, gx, gy, gz);

    // 3. Detect tap
    float mag = readAccelMagnitude();
    bool imuTap = (mag > TAP_THRESHOLD);  
    bool debounceOK = (now - lastTapTime[i] > DEBOUNCE_US);

    if (touched && imuTap && debounceOK) {
      Serial.print(fingerName[i]);
      Serial.print(" finger tapped at ");
      Serial.print(now);
      Serial.println(" us");
      lastTapTime[i] = now;  // Update last tap time
    }

    // ---- SLIDER DETECTION ----
    // detectSlider(i, touched, gx, now);
  }
}

bool detectSlider(int finger, bool touched, float gx, unsigned long now) {

    // 1. If touch is OFF → reset sliding state
    if (!touched) {
        sliding[finger] = false;
        slideStartTime[finger] = 0;
        return false;
    }

    // 2. Motion timing
    unsigned long dtMicros = now - lastIMUTime[finger];
    lastIMUTime[finger] = now;
    float dt = dtMicros / 1000000.0;    // seconds

    // 3. Use GYRO X as sliding axis
    float motion = gx;

    // 4. Ignore tiny jitter
    if (abs(motion) < SLIDE_GYRO_DEADZONE) {
        if (sliding[finger] &&
            now - lastSlideMotion[finger] > SLIDE_IDLE_CUTOFF) {

            sliding[finger] = false;
            Serial.print(fingerName[finger]);
            Serial.println(" SLIDE END");
        }
        return sliding[finger];
    }
    
    // 5. Motion detected
    lastSlideMotion[finger] = now;

    // SLIDE START
    if (!sliding[finger]) {
        sliding[finger] = true;
        Serial.print(fingerName[finger]);
        Serial.println(" SLIDE START");
    }

    // 6. Integrate slider movement
    sliderValue[finger] += motion * SLIDE_GAIN * dt * 1000.0; // scaled

    // Clamp slider between 0 and 100
    sliderValue[finger] = constrain(sliderValue[finger], 0, 100);

    Serial.print(fingerName[finger]);
    Serial.print(" SLIDER: ");
    Serial.println(sliderValue[finger]);

    return true;
}




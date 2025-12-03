#include <Wire.h>
#include <Adafruit_DRV2605.h>

Adafruit_DRV2605 drv;

void setup() {
  Serial.begin(115200);
  Wire.begin();

  if (!drv.begin()) {
    Serial.println("DRV2605L not found. Check wiring!");
    while (1);
  }

  drv.selectLibrary(1);

  Serial.println("Haptic driver ready.");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "BUZZ") {
      buzz();
    }
  }
}

void buzz() {
  drv.setWaveform(0, 47);
  drv.setWaveform(84, 0); 

  drv.go();
}

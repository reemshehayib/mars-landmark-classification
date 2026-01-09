#include <zephyr/drivers/gpio.h> // Zephyr specific

// This uses the board's actual built-in LED definition
#define LED0_NODE DT_ALIAS(led0)

void setup() {
  Serial.begin(115200); 
  // Use LED_BUILTIN - standard Arduino-to-Zephyr mapping
  pinMode(LED_BUILTIN, OUTPUT); 
}

void loop() {
  if (Serial.available() > 0) {
    int incomingByte = Serial.read();
    
    // We check for both ASCII '1' and raw byte 1 just in case
    if (incomingByte == '1' || incomingByte == 1) {
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (incomingByte == '0' || incomingByte == 0) {
      digitalWrite(LED_BUILTIN, LOW);
    }
  }
}
void setup() {
  Serial.begin(115200);  // USB/Console
  Serial1.begin(115200); // Internal Bridge
  
  pinMode(13, OUTPUT);          // Standard Arduino LED pin
  pinMode(LED_BUILTIN, OUTPUT); // Zephyr Alias LED pin
}

void loop() {
  // If ANY data comes in from ANYWHERE
  if (Serial.available() > 0 || Serial1.available() > 0) {
    char val = (Serial.available() > 0) ? Serial.read() : Serial1.read();
    
    if (val == '1') {
      digitalWrite(13, HIGH);
      digitalWrite(LED_BUILTIN, HIGH);
    } else if (val == '0') {
      digitalWrite(13, LOW);
      digitalWrite(LED_BUILTIN, LOW);
    }
  }
}
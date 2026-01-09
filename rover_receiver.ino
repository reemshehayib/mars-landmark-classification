/* Robust Mars Rover Receiver for Zephyr/UNO Q */

void setup() {
  // Serial is the Network Console (what we see in terminal)
  Serial.begin(115200); 
  // Serial1 is the actual internal HS1 hardware bridge 
  Serial1.begin(115200); 
  
  // LED_BUILTIN is usually Pin 13, but Zephyr maps it to the board's USER LED
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Flash 3 times on startup to prove the LED works
  for(int i=0; i<3; i++) {
    digitalWrite(LED_BUILTIN, HIGH); delay(100);
    digitalWrite(LED_BUILTIN, LOW); delay(100);
  }
}

void loop() {
  // Listen on the internal hardware bridge (Serial1)
  if (Serial1.available() > 0) {
    char val = Serial1.read();
    
    if (val == '1') {
      digitalWrite(LED_BUILTIN, HIGH);
      Serial.println("Bridge Received: LED ON"); // Debug back to terminal
    } 
    else if (val == '0') {
      digitalWrite(LED_BUILTIN, LOW);
      Serial.println("Bridge Received: LED OFF");
    }
  }
}
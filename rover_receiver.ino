void setup() {
  Serial1.begin(115200); 
  
  // Using the exact names from your compiler error
  pinMode(LED4_R, OUTPUT);
  pinMode(LED3_G, OUTPUT);
  
  // Startup Test: Turn both ON (LOW) then OFF (HIGH)
  digitalWrite(LED4_R, LOW);
  digitalWrite(LED3_G, LOW);
  delay(500);
  digitalWrite(LED4_R, HIGH);
  digitalWrite(LED3_G, HIGH);
}

void loop() {
  if (Serial1.available() > 0) {
    char val = Serial1.read();
    
    if (val == '1') {
      // CRATER: Red ON, Green OFF
      digitalWrite(LED4_R, LOW); 
      digitalWrite(LED3_G, HIGH); 
    } 
    else if (val == '0') {
      // CLEAR: Green ON, Red OFF
      digitalWrite(LED4_R, HIGH); 
      digitalWrite(LED3_G, LOW); 
    }
  }
}
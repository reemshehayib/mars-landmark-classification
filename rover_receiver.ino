void setup() {
  Serial1.begin(115200); // Internal Bridge
  
  // Initialize the RGB pins near the logo
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  
  // Start with both OFF (True for common anode, use HIGH to turn off)
  digitalWrite(LEDR, HIGH); 
  digitalWrite(LEDG, HIGH);
}

void loop() {
  if (Serial1.available() > 0) {
    char val = Serial1.read();
    
    if (val == '1') {
      // CRATER: Red ON, Green OFF
      digitalWrite(LEDR, LOW);  // LOW is ON for many RGBs
      digitalWrite(LEDG, HIGH); 
    } 
    else if (val == '0') {
      // CLEAR: Green ON, Red OFF
      digitalWrite(LEDR, HIGH); 
      digitalWrite(LEDG, LOW); 
    }
  }
}
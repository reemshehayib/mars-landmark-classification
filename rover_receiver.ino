/* * MARS ROVER HARDWARE RECEIVER 
 * Target: Arduino UNO Q (Zephyr RTOS)
 */

void setup() {
  // We use Serial1 because ttyHS1 on the Qualcomm chip 
  // is physically wired to the STM32's second UART.
  Serial1.begin(115200); 
  
  // Use the built-in LED (usually the blue/red user LED)
  pinMode(LED_BUILTIN, OUTPUT);
  
  // Visual Startup Check: Flash 3 times
  for(int i = 0; i < 3; i++) {
    digitalWrite(LED_BUILTIN, HIGH);
    delay(100);
    digitalWrite(LED_BUILTIN, LOW);
    delay(100);
  }
}

void loop() {
  // Listen for commands from the Qualcomm chip
  if (Serial1.available() > 0) {
    char command = Serial1.read();
    
    if (command == '1') {
      digitalWrite(LED_BUILTIN, HIGH); // Crater found!
    } 
    else if (command == '0') {
      digitalWrite(LED_BUILTIN, LOW);  // Clear terrain
    }
    // Newlines (\n) and carriage returns (\r) are ignored
  }
}
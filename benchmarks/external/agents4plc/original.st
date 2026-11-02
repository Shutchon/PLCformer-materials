FUNCTION_BLOCK FB_ProductionLineControl
{ S7_Optimized_Access := 'TRUE' }
VAR_INPUT 
Mode : BOOL; // Manual/Auto mode selection
ForwardButton : BOOL; // Forward rotation button
ReverseButton : BOOL; // Reverse rotation button
SensorA : BOOL; // Sensor A input
SensorB : BOOL; // Sensor B input
SensorC : BOOL; // Sensor C input
ButtonA : BOOL; // Button A input
ButtonB : BOOL; // Button B input
ButtonC : BOOL; // Button C input
END_VAR

VAR_OUTPUT 
MotorForward : BOOL; // Motor forward output
MotorReverse : BOOL; // Motor reverse output
CompletionLight : BOOL; // Process completion indicator
END_VAR

VAR 
State : INT; // State machine current state
END_VAR

// State machine initialization
IF State = 0 THEN
MotorForward := FALSE;
MotorReverse := FALSE;
CompletionLight := FALSE;
State := 1; // Initial state
END_IF;

// Manual mode operation
IF Mode = FALSE THEN
IF ForwardButton AND NOT ReverseButton THEN
 MotorForward := TRUE;
 MotorReverse := FALSE;
ELSIF ReverseButton AND NOT ForwardButton THEN
 MotorReverse := TRUE;
 MotorForward := FALSE;
ELSE
 MotorForward := FALSE;
 MotorReverse := FALSE;
END_IF;
ELSE // Automatic mode operation
// Reset MotorReverse at the beginning of automatic mode
CASE State OF
 1: // Wait for product at position A
IF SensorA THEN
State := 2;
END_IF;
 2: // Wait for button A press
IF ButtonA THEN
MotorForward := TRUE;
MotorReverse := FALSE;
State := 3;
END_IF;
 3: // Wait for product at position B
IF SensorB THEN
MotorForward := FALSE;
State := 4;
END_IF;
 4: // Wait for button B press
IF ButtonB THEN
MotorForward := TRUE;
MotorReverse := FALSE;
State := 5;
END_IF;
 5: // Wait for product at position C
IF SensorC THEN
MotorForward := FALSE;
State := 6;
END_IF;
 6: // Wait for button C press
IF ButtonC THEN
CompletionLight := TRUE;
State := 7;
END_IF;
 7: // Wait for product removal
IF NOT SensorC THEN
CompletionLight := FALSE;
State := 1;
END_IF;
END_CASE;
END_IF;

END_FUNCTION_BLOCK

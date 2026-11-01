; ============================================================================
; Standard IEC 61131-3 IL Benchmark Program 1: Self-Holding Motor Starter
; Equivalent Structured Text (ST):
;   Y0 := (X0 OR Y0) AND NOT X1;
;
; I/O Mapping:
;   X0 : Start Pushbutton (Normally Open, Momentary)
;   X1 : Stop Pushbutton / Overload Relay (Normally Closed logic via ANI)
;   Y0 : Motor Contactor Coil & Seal-in Contact
; ============================================================================

LD  X0      ; Load Start Pushbutton X0 into ACC
OR  Y0      ; Parallel seal-in (self-holding) contact Y0
ANI X1      ; Series normally-closed Stop pushbutton X1
OUT Y0      ; Energize Motor Contactor Y0
END         ; End of Scan Cycle


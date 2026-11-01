; ============================================================================
; Standard IEC 61131-3 IL Benchmark Program 3: Automated Conveyor Sorting & Batching
;
; I/O & Counter Mapping:
;   X0 : Photoelectric Part Sensor (Rising edge triggers CTU C0)
;   X1 : Batch Reset Pushbutton (Resets C0 and Batch Full Lamp Y1)
;   X2 : Conveyor Run Switch
;   C0 : Part Counter (Preset K = 3 parts per box)
;   Y0 : Conveyor Belt Motor (Runs when X2=1 and box not full C0.DN=0)
;   Y1 : Box Full Indicator & Diverter Solenoid (Energized when C0.DN=1)
; ============================================================================

; Rung 1: Count parts passing sensor X0 up to K3
LD  X0
CTU C0, K3

; Rung 2: Reset Counter C0 when X1 is pressed
LD  X1
RST C0

; Rung 3: Box Full Indicator / Diverter Y1 follows C0.DN
LD  C0.DN
OUT Y1

; Rung 4: Conveyor Motor Y0 runs when Run Switch X2=1 and Box not full (ANI C0.DN)
LD  X2
ANI C0.DN
OUT Y0

END


; ============================================================================
; Standard IEC 61131-3 IL Benchmark Program 2: Star-Delta Motor Starter
;
; I/O & Timer Mapping:
;   X0 : Start Pushbutton (NO)
;   X1 : Stop Pushbutton / Overload Relay (NC logic via ANI)
;   M0 : Master Run Relay (Self-Holding)
;   T0 : Star-to-Delta Transition On-Delay Timer (Preset K = 5 ticks)
;   Y0 : Main Contactor (KM_Main)
;   Y1 : Star Contactor (KM_Star)  -- ON when M0=1, T0.DN=0, and Y2=0 (Interlock)
;   Y2 : Delta Contactor (KM_Delta) -- ON when M0=1, T0.DN=1, and Y1=0 (Interlock)
; ============================================================================

; Rung 1: Master Run Relay M0 (Self-Holding Start/Stop) + Main Contactor Y0
LD  X0
OR  M0
ANI X1
OUT M0
OUT Y0

; Rung 2: Star-to-Delta Timer T0 (Preset = 5 scan cycles)
LD  M0
TON T0, K5

; Rung 3: Star Contactor Y1 (Energized before T0 expires, interlocked with Y2)
LD  M0
ANI T0.DN
ANI Y2
OUT Y1

; Rung 4: Delta Contactor Y2 (Energized after T0 expires, interlocked with Y1)
LD  M0
AND T0.DN
ANI Y1
OUT Y2

END


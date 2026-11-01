; ============================================================================
; Standard IEC 61131-3 IL Benchmark Program 4: 4-Way Traffic Light Controller
;
; Sequential 4-Phase Cascade with Mutual Green/Yellow Conflict Interlocks:
;   X0 : Master Controller Enable Switch (1 = Run, 0 = All Off / Reset)
;   M0 : Cycle Timer Bus (Resets automatically when T3.DN completes a full cycle)
;   T0 : Phase 1 Timer (K4 ticks)  -> NS Green (Y0),  EW Red (Y5)
;   T1 : Phase 2 Timer (K6 ticks)  -> NS Yellow (Y1), EW Red (Y5)
;   T2 : Phase 3 Timer (K10 ticks) -> NS Red (Y2),    EW Green (Y3)
;   T3 : Phase 4 Timer (K12 ticks) -> NS Red (Y2),    EW Yellow (Y4)
;
; Output Mapping:
;   Y0 : North-South Green   (Interlocked with EW Green Y3 & EW Yellow Y4)
;   Y1 : North-South Yellow
;   Y2 : North-South Red
;   Y3 : East-West Green     (Interlocked with NS Green Y0 & NS Yellow Y1)
;   Y4 : East-West Yellow
;   Y5 : East-West Red
; ============================================================================

; Rung 1: Master Cycle Bus M0 (Auto-resets timers when T3.DN reaches 1)
LD  X0
ANI T3.DN
OUT M0

; Rung 2: Cascaded Phase Timers T0 (K4), T1 (K6), T2 (K10), T3 (K12)
LD  M0
TON T0, K4
LD  M0
TON T1, K6
LD  M0
TON T2, K10
LD  M0
TON T3, K12

; Rung 3: North-South Green (Y0) -- Active in Phase 1 (X0=1, T0.DN=0) with conflict interlock
LD  X0
ANI T0.DN
ANI Y3
ANI Y4
OUT Y0

; Rung 4: North-South Yellow (Y1) -- Active in Phase 2 (X0=1, T0.DN=1, T1.DN=0)
LD  X0
AND T0.DN
ANI T1.DN
OUT Y1

; Rung 5: North-South Red (Y2) -- Active in Phases 3 & 4 (X0=1, T1.DN=1)
LD  X0
AND T1.DN
OUT Y2

; Rung 6: East-West Green (Y3) -- Active in Phase 3 (X0=1, T1.DN=1, T2.DN=0) with conflict interlock
LD  X0
AND T1.DN
ANI T2.DN
ANI Y0
ANI Y1
OUT Y3

; Rung 7: East-West Yellow (Y4) -- Active in Phase 4 (X0=1, T2.DN=1, T3.DN=0)
LD  X0
AND T2.DN
ANI T3.DN
OUT Y4

; Rung 8: East-West Red (Y5) -- Active in Phases 1 & 2 (X0=1, T1.DN=0)
LD  X0
ANI T1.DN
OUT Y5

END


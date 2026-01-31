# Phase F v5: Multi-Lane Roads - Speed Zone Fix

## Experiment ID
- **Run ID**: `phase-F-v5`
- **Config**: `python/configs/planning/vehicle_ppo_phase-F-v5.yaml`
- **Scene**: `PhaseF_MultiLane`
- **Date**: 2026-01-31

## Motivation

Phase F v4 validated strict curriculum staggering (P-002) but discovered a Unity code bug:
`GenerateSpeedZones()` placed Residential (30 km/h) as the first zone when transitioning
from 1 zone (60 km/h) to 2 zones, causing -2790 reward crash at step 3.78M.

## Code Fix

### WaypointManager.cs - GenerateSpeedZones()

**Before (v4 and earlier):**
```csharp
SpeedZoneType[] zoneTypes = {
    SpeedZoneType.Residential,    // 30 km/h  <-- FIRST ZONE
    SpeedZoneType.UrbanNarrow,    // 50 km/h
    SpeedZoneType.UrbanGeneral,   // 60 km/h
    SpeedZoneType.Expressway      // 80 km/h
};
float[] zoneLimits = { 8.33f, 13.89f, 16.67f, 22.22f };
```
- count=2: Residential(30) + UrbanNarrow(50)
- Agent at 60 km/h gets -3.0/step in 30 km/h zone

**After (v5):**
```csharp
SpeedZoneType[] zoneTypes = {
    SpeedZoneType.UrbanGeneral,   // 60 km/h  <-- MATCHES DEFAULT
    SpeedZoneType.UrbanNarrow,    // 50 km/h
    SpeedZoneType.Residential,    // 30 km/h
    SpeedZoneType.Expressway      // 80 km/h
};
float[] zoneLimits = { 16.67f, 13.89f, 8.33f, 22.22f };
```
- count=2: UrbanGeneral(60) + UrbanNarrow(50)
- First zone matches single-zone default (60 km/h), gentle 10 km/h drop in second zone

### Zone Layouts by Count

| Count | Zone 0 | Zone 1 | Zone 2 | Zone 3 |
|-------|--------|--------|--------|--------|
| 1 | UrbanGeneral (60) | - | - | - |
| 2 | UrbanGeneral (60) | UrbanNarrow (50) | - | - |
| 3 | UrbanGeneral (60) | UrbanNarrow (50) | Residential (30) | - |
| 4 | UrbanGeneral (60) | UrbanNarrow (50) | Residential (30) | Expressway (80) |

## Training Config

Identical to v4 (curriculum thresholds unchanged):
- **learning_rate_schedule**: constant at 1.5e-4
- **max_steps**: 10,000,000
- **15 unique thresholds**: 150-900 range, min 50-point gaps
- **Initialize from**: Phase E checkpoint

## Expected Behavior Change

At step ~3.78M (speed_zone 1→2 transition):
- **v4**: First zone drops to 30 km/h → -2790 crash
- **v5**: First zone stays at 60 km/h, second zone at 50 km/h → mild adjustment expected

## Success Criteria

1. speed_zone transition causes < -200 reward drop (not -2790)
2. All 15 curriculum transitions complete within 10M steps
3. Final reward: +400 or higher
4. NPC phase reached (threshold 700+)

## Policy References

- **P-002**: Staggered Curriculum (strict, from v4)
- **P-013**: Speed Zone Curriculum Ordering (NEW - first zone must match default)

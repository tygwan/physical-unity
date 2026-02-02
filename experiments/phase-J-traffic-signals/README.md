# Phase J: Traffic Signals + Stop Lines

## Overview

Phase J introduces **traffic signal compliance** — the agent must learn to stop at red lights, proceed on green, and handle yellow caution signals at intersections.

**Start**: Phase I v2 checkpoint (reward 770, 260D observation)
**Target**: 268D observation (+8D traffic signal info), reward 650+
**Scene**: PhaseH_NPCIntersection (reuse existing intersection scene)

## Background

### Agent Capabilities After Phase I v2
- 260D observation space
- Curved roads with full curvature + direction variation
- 3 NPCs with speed ratio 0.85, variation 0.15
- Multi-lane (2 lanes), center line
- Speed zones (2 zones)
- Goal distance 230m
- **Intersection skills from Phase G/H still encoded in weights** (but not active in Phase I)

### Why Traffic Signals Now?
1. **Natural progression**: Agent already handles intersections (Phase G/H) and curves+NPCs (Phase I)
2. **Roadmap alignment**: Original plan has Phase J = 신호등 + 정지선
3. **Stop lines already exist**: PhaseSceneCreator creates StopLine objects at intersection approaches
4. **Real-world relevance**: Traffic signal compliance is fundamental to autonomous driving

## Architecture

### New Component: TrafficLightController.cs

```
Assets/Scripts/Environment/TrafficLightController.cs
```

Manages traffic light state cycling per Training Area:
- **States**: Red → Green → Yellow → Red
- **Timing**: Configurable via environment parameters
  - Green duration: 8-15s (curriculum-controlled)
  - Yellow duration: 3s (fixed, Korean standard)
  - Red duration: 8-15s (matches green)
- **Per-area**: Each Training Area has its own traffic light with independent timing
- **Randomized start**: Each episode starts at a random point in the cycle

### Observation Space (+8D → 268D)

| Index | Observation | Range | Description |
|-------|------------|-------|-------------|
| 0 | Red | 0/1 | One-hot: light is red |
| 1 | Yellow | 0/1 | One-hot: light is yellow |
| 2 | Green | 0/1 | One-hot: light is green |
| 3 | None | 0/1 | One-hot: no traffic light |
| 4 | Distance to stop line | [0, 1] | Normalized by 200m |
| 5 | Time remaining | [0, 1] | Normalized by max cycle time |
| 6 | Stopping required | 0/1 | 1 if red/yellow AND within braking distance |
| 7 | Stopped at line | 0/1 | 1 if speed < 0.5 AND within 5m of stop line |

### Reward Design

| Event | Reward | Condition |
|-------|--------|-----------|
| Red light violation | -5.0 + EndEpisode | Crossing stop line Z while red |
| Proper red stop | +0.3/step | Stopped within 5m before stop line while red |
| Yellow caution | -2.0 | Entering intersection zone on yellow (from > 15m) |
| Yellow committed | 0.0 | Already in intersection when yellow starts |
| Green proceed | normal | Standard progress reward applies |
| Unnecessary stop | -0.1/step | Stopped at green light for > 2s |

### Modified Files

| File | Changes |
|------|---------|
| `E2EDrivingAgent.cs` | +8D observation, `enableTrafficSignalObservation` flag, red light reward |
| `DrivingSceneManager.cs` | Read `traffic_signal_enabled` param, configure TrafficLightController |
| `PhaseSceneCreator.cs` | Add traffic light visual (tall pole + colored sphere) per Training Area |
| `TrafficLightController.cs` | NEW — state machine, timing, episode randomization |

## Curriculum Design

### Warm Start Strategy

Phase I v2 → Phase J requires **re-enabling intersections** that were locked in Phase I.
The agent's weights still contain intersection skills from Phase G/H training.

**Two-phase approach:**
1. **Steps 0-2M**: Re-enable intersections WITHOUT traffic signals (agent re-learns intersections)
2. **Steps 2M+**: Enable traffic signals (agent learns to stop at red)

### Environment Parameters

#### Quick-unlock (warm start ~770, re-enable existing skills):

| Parameter | Thresholds | Values | min_lesson_length |
|-----------|-----------|--------|-------------------|
| num_lanes | 70 | 1 → 2 | - |
| center_line_enabled | 90 | 0 → 1 | - |
| goal_distance | 110, 130 | 150 → 200 → 230 | - |

#### NPC params (previously mastered, unlock quickly):

| Parameter | Thresholds | Values | min_lesson_length |
|-----------|-----------|--------|-------------------|
| num_active_npcs | 550, 620, 680 | 0 → 1 → 2 → 3 | - |
| npc_speed_ratio | 600, 660 | 0.5 → 0.7 → 0.85 | - |
| npc_speed_variation | 685, 690, 693 | 0 → 0.05 → 0.10 → 0.15 | - |

#### Intersection re-enable (previously mastered in Phase G/H):

| Parameter | Thresholds | Values | min_lesson_length |
|-----------|-----------|--------|-------------------|
| intersection_type | 695, 710, 725 | 0 → 1 → 2 → 3 | 2000 |
| turn_direction | 700, 715 | 0 → 1 → 2 | 2000 |

#### Phase J NEW — Traffic signals:

| Parameter | Thresholds | Values | min_lesson_length |
|-----------|-----------|--------|-------------------|
| traffic_signal_enabled | 740 | 0 → 1 | 3000 |
| signal_green_ratio | 750, 760 | 0.7 → 0.5 → 0.4 | 2000 |

**P-018 compliance**: Intersection thresholds spaced by 15+ points (695/710/725).
Traffic signal thresholds start at 740 (15 points above last intersection threshold 725).

#### Locked params (curves inactive during intersection training):

| Parameter | Value | Reason |
|-----------|-------|--------|
| road_curvature | 0 | Mutually exclusive with intersections |
| curve_direction_variation | 0 | No curves |
| speed_zone_count | 1 | Simplify for signal learning |

### Threshold Rationale

```
700 ─────── intersection_type 0→1 (T-junction)
│           turn_direction 0→1 (left)
715 ─────── intersection_type 1→2 (Cross) + turn_direction 1→2 (right)
725 ─────── intersection_type 2→3 (Y-junction)
│
│  15-point gap (P-018)
│
740 ─────── traffic_signal_enabled 0→1 (lights turn on)
750 ─────── signal_green_ratio 0.7→0.5 (less green time)
760 ─────── signal_green_ratio 0.5→0.4 (even less green time)
```

## Training Config

```yaml
# vehicle_ppo_phase-J.yaml
behaviors:
  E2EDrivingAgent:
    trainer_type: ppo
    init_path: results/phase-I-v2/E2EDrivingAgent/E2EDrivingAgent-5000080.pt

    hyperparameters:
      batch_size: 4096
      buffer_size: 40960
      learning_rate: 1.5e-4
      beta: 5.0e-3
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 5
      learning_rate_schedule: constant
      beta_schedule: constant
      epsilon_schedule: constant

    network_settings:
      normalize: false
      hidden_units: 512
      num_layers: 3

    reward_signals:
      extrinsic:
        gamma: 0.99
        strength: 1.0

    max_steps: 7000000   # Longer: new concept + intersection re-learning
    time_horizon: 256
    summary_freq: 10000
    keep_checkpoints: 5
    checkpoint_interval: 500000
    threaded: false

env_settings:
  env_path: Builds/PhaseJ/PhaseJ.exe   # NEW BUILD REQUIRED
  num_envs: 3
  base_port: 5004
  timeout_wait: 600
  no_graphics: true

torch_settings:
  device: cuda
```

## Implementation Order

### Step 1: TrafficLightController.cs (NEW)
- Create state machine: Red → Green → Yellow → Red
- Configurable timing via public properties
- Random start phase on episode begin
- API: `GetCurrentState()`, `GetTimeRemaining()`, `GetStopLineZ()`

### Step 2: E2EDrivingAgent.cs (MODIFY)
- Add `enableTrafficSignalObservation` toggle (like existing lane/intersection toggles)
- Add `CollectTrafficSignalObservations()` method (+8D)
- Add `TrafficLightController` reference
- Add red light violation check in `CalculateRewards()`
- Update observation space comment (268D)

### Step 3: DrivingSceneManager.cs (MODIFY)
- Read `traffic_signal_enabled` environment parameter
- Configure TrafficLightController on episode reset
- Read `signal_green_ratio` for timing control

### Step 4: PhaseSceneCreator.cs (MODIFY)
- Add `CreatePhaseJ()` method (fork of Phase H with traffic lights)
- Add traffic light visual: tall pole (Cylinder) + signal head (3 spheres)
- Position at stop line (Z=93, approach side)

### Step 5: Build + Config
- Create PhaseJ scene via editor menu
- Build PhaseJ executable
- Create `vehicle_ppo_phase-J.yaml`

### Step 6: Training
- Run training: 7M steps
- Monitor intersection re-learning (expect ~2M for re-stabilization)
- Monitor traffic signal learning (expect ~3-4M for compliance)

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Agent forgets curve skills | Low | Curves locked anyway; weights preserved for later |
| Intersection re-learning slow | Medium | Thresholds start at 695, well below warm start 770 |
| Red light reward too harsh | Medium | -5 + EndEpisode may cause avoidance; tune to -3 first |
| Signal timing too fast | Low | Start with long green (70%), gradually reduce |
| New build required | Low | PhaseSceneCreator handles scene creation |
| Observation space change (260→268) | Medium | Must rebuild; old checkpoints compatible with new dims via init_path |

## Expected Timeline

- Intersection re-enable: Complete by ~2M steps
- Traffic signal basic compliance: ~3-4M steps
- Green ratio reduction: ~5-6M steps
- Final stabilization: ~7M steps
- Expected final reward: 650-700 (lower than Phase I due to stopping penalty eating progress reward)

## Key Decisions

1. **Curves locked**: Can't combine curves + intersections (WaypointManager mutual exclusivity)
2. **Warm start from Phase I v2**: Despite curves being locked, the latest weights are strongest overall
3. **New build required**: 268D observation + TrafficLightController need new Unity build
4. **Simple signal model**: Single traffic light per intersection (no cross-traffic signal)
5. **Episode-end on red violation**: Strong learning signal, may need tuning

## Success Criteria

- [ ] Agent stops at red lights > 90% of the time
- [ ] Agent proceeds on green without unnecessary delay
- [ ] Intersection navigation maintained (T/Cross/Y)
- [ ] NPC interaction preserved (3 NPCs, varied speeds)
- [ ] Final reward > 600 (lower acceptable due to stop time)

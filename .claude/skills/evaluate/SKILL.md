---
name: evaluate
description: 모델 평가 스킬. 벤치마크 실행 및 메트릭 계산을 지원합니다. "/evaluate" 명령으로 호출.
---

# Model Evaluation Skill

자율주행 Planning 모델의 평가 및 벤치마킹을 위한 스킬입니다.

## Commands

### /evaluate [model] [scenarios]
모델 평가 실행

```bash
/evaluate ppo_v1/best.pt val
/evaluate ppo_v1/best.pt test
```

### /evaluate benchmark [model]
nuPlan 벤치마크 실행

### /evaluate compare [model1] [model2]
모델 비교

### /evaluate report [model]
평가 보고서 생성

## Success Criteria

| Metric | Target | Description |
|--------|--------|-------------|
| Collision Rate | < 5% | Safety-critical |
| Route Completion | > 85% | Progress metric |
| Max Jerk | < 2 m/s³ | Comfort metric |
| Max Lat Acc | < 3 m/s² | Comfort metric |

## Workflow

### 1. Quick Evaluation
```bash
python python/src/evaluation/evaluate.py \
  --model experiments/checkpoints/ppo_v1/best.pt \
  --scenarios datasets/splits/val.txt \
  --output experiments/results/ppo_v1_val/
```

### 2. Full Benchmark
```bash
python python/src/evaluation/nuplan_benchmark.py \
  --model experiments/checkpoints/ppo_v1/best.pt \
  --scenarios nuplan_val \
  --output experiments/results/ppo_v1_nuplan/
```

### 3. Scenario-specific
```bash
# Urban
python python/src/evaluation/evaluate.py \
  --model experiments/checkpoints/ppo_v1/best.pt \
  --scenarios datasets/processed/scenarios/urban/

# Highway
python python/src/evaluation/evaluate.py \
  --scenarios datasets/processed/scenarios/highway/
```

### 4. Compare Models
```bash
python python/src/evaluation/compare_models.py \
  --models ppo_v1/best.pt sac_v1/best.pt bc_v1/best.pt \
  --scenarios datasets/splits/test.txt
```

## Evaluation Report

```markdown
## Evaluation Report: ppo_v1

### Summary
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Collision Rate | 4.2% | < 5% | ✅ PASS |
| Route Completion | 87.3% | > 85% | ✅ PASS |
| Max Jerk | 1.8 m/s³ | < 2 | ✅ PASS |

### Overall: ✅ PASS (3/3 criteria met)

### Scenario Breakdown
| Scenario | Collision | Completion |
|----------|-----------|------------|
| Urban | 5.1% | 85.2% |
| Highway | 2.8% | 92.1% |
| Intersection | 6.2% | 81.4% |

### Model Comparison
| Model | Collision | Completion |
|-------|-----------|------------|
| PPO | 4.2% | 87.3% |
| SAC | 3.8% | 88.1% |
| BC | 8.5% | 75.1% |
```

## Integration

- **benchmark-evaluator agent**: Detailed evaluation
- **evaluation module**: python/src/evaluation/

---
name: dataset
description: 데이터셋 관리 스킬. 다운로드, 전처리, 분할을 지원합니다. "/dataset" 명령으로 호출.
---

# Dataset Management Skill

자율주행 데이터셋의 다운로드, 전처리, 관리를 위한 스킬입니다.

## Commands

### /dataset list
사용 가능한 데이터셋 목록

### /dataset download [name]
데이터셋 다운로드

```bash
/dataset download nuplan-mini
/dataset download waymo-motion
```

### /dataset status
데이터셋 상태 확인

### /dataset process [name]
데이터셋 전처리

## Supported Datasets

| Dataset | Size | Purpose | Status |
|---------|------|---------|--------|
| nuPlan (mini) | ~50GB | IL, Planning | Primary |
| nuPlan (full) | ~1TB | IL, Planning | Optional |
| Waymo Motion | ~100GB | Trajectory | Primary |
| highD | ~5GB | Highway | Secondary |
| INTERACTION | ~2GB | Intersection | Secondary |

## Workflow

### 1. Check Status
```bash
# Disk space
df -h datasets/

# Current data
ls -la datasets/raw/
ls -la datasets/processed/
```

### 2. Download
```bash
# nuPlan (requires registration at nuscenes.org)
# Manual download to datasets/raw/nuplan/

# After download
ls -la datasets/raw/nuplan/
```

### 3. Process
```bash
# Extract archives
tar -xzf datasets/raw/nuplan/*.tar.gz -C datasets/raw/nuplan/

# Run preprocessing
python python/src/data/process_nuplan.py \
  --input datasets/raw/nuplan/ \
  --output datasets/processed/nuplan/
```

### 4. Create Splits
```bash
python python/src/data/create_splits.py \
  --data datasets/processed/ \
  --output datasets/splits/ \
  --train-ratio 0.8
```

## Data Structure

```
datasets/
├── raw/                    # Original downloads
│   ├── nuplan/
│   ├── waymo/
│   └── highd/
├── processed/              # Preprocessed data
│   ├── scenarios/
│   │   ├── urban/
│   │   ├── highway/
│   │   └── intersection/
│   └── features/
└── splits/                 # Train/val/test
    ├── train.txt
    ├── val.txt
    └── test.txt
```

## Statistics Report

```markdown
## Dataset Status

| Dataset | Downloaded | Processed | Scenarios |
|---------|------------|-----------|-----------|
| nuPlan  | ✅ 52GB    | ✅        | 15,234    |
| Waymo   | ⏳ 45%     | ❌        | -         |

### Storage
- Total: 155GB / 4TB (4%)
- Raw: 120GB
- Processed: 35GB

### Splits
- Train: 12,000
- Val: 1,500
- Test: 1,500
```

## Integration

- **dataset-curator agent**: Detailed data management
- **data module**: python/src/data/

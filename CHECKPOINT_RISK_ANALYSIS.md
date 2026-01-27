# Checkpoint & Training Data Risk Analysis

**Date:** 2026-01-27
**Focus:** Risk of renaming experiments folders with training data

---

## Current Situation

### Training Data Location
```
experiments/phase-0-foundation/results/
├── E2EDrivingAgent/
│   ├── checkpoint.pt                      # Latest checkpoint (8M steps)
│   ├── E2EDrivingAgent-6499978.pt         # 6.5M checkpoint
│   ├── E2EDrivingAgent-6999908.pt         # 7M checkpoint
│   ├── E2EDrivingAgent-7499782.pt         # 7.5M checkpoint
│   └── E2EDrivingAgent-8000047.pt         # 8M checkpoint (FINAL)
├── E2EDrivingAgent.onnx                   # Exported ONNX
├── configuration.yaml                     # Training config
└── run_logs/                              # TensorBoard logs (108M total)
```

**Critical Finding:** results/ folder does NOT have v10g data!
- results/v10g/ was deleted during reorganization
- All v10g training data is now ONLY in experiments/phase-0-foundation/results/

---

## ML-Agents Checkpoint Loading Mechanism

### How --initialize-from Works

```bash
mlagents-learn config.yaml --run-id=new_run --initialize-from=old_run
```

**Search Path:**
1. Looks in `results/old_run/BehaviorName/checkpoint.pt`
2. If not found, looks in `results/old_run/checkpoint.pt`
3. Does NOT look in `experiments/` folder!

**Example:**
```bash
--initialize-from=v10g_lane_keeping
# Searches: results/v10g_lane_keeping/E2EDrivingAgent/checkpoint.pt
# NOT: experiments/phase-0-foundation/results/...
```

---

## Current Problem (Already Exists!)

### Phase A Documentation Says:
```bash
mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml \
  --run-id=phase-A-overtaking \
  --initialize-from=v10g_lane_keeping    # ❌ THIS PATH DOESN'T EXIST!
```

**Reality Check:**
```bash
$ ls results/v10g*
ls: cannot access 'results/v10g*': No such file or directory

$ ls results/
archives/  ml_inference_test/  phaseB_inf2/  phase-G/
```

**Conclusion:** The initialize path is ALREADY BROKEN before any renaming!

---

## Risk Assessment for Folder Rename

### ✅ LOW RISK: experiments/ folder rename

**Why LOW RISK:**
1. **ML-Agents doesn't use experiments/ path**
   - --initialize-from searches results/ folder only
   - experiments/ is for documentation/organization only

2. **Training data is self-contained**
   - checkpoint.pt doesn't care about parent folder name
   - ONNX models don't reference folder paths
   - TensorBoard logs are relative paths

3. **Git mv preserves data**
   - No data loss during rename
   - History preserved
   - All files moved atomically

**What WILL break:**
- ❌ Documentation links to experiments/phase-0-foundation/
- ❌ README paths showing wrong folder names
- ❌ TRAINING-LOG.md commands with old names

**What WON'T break:**
- ✅ Checkpoint files themselves
- ✅ ONNX models
- ✅ TensorBoard logs
- ✅ Training can still run (if proper initialize path used)

---

## Real Risks (NOT from folder rename)

### 🔴 RISK 1: Incorrect --initialize-from Path

**Current Issue:**
```bash
# Documentation says:
--initialize-from=v10g_lane_keeping

# But should be:
--initialize-from=v10g  # If results/v10g/ exists
# OR copy checkpoint to expected location
```

**Solution Options:**

**Option A: Copy checkpoint to results/**
```bash
mkdir -p results/v10g_lane_keeping
cp -r experiments/phase-0-foundation/results/E2EDrivingAgent \
      results/v10g_lane_keeping/
```

**Option B: Use absolute path (if ML-Agents supports)**
```bash
--initialize-from=experiments/phase-0-foundation/results/E2EDrivingAgent
# (Need to verify if ML-Agents supports this)
```

**Option C: Update documentation to not use initialize-from**
```bash
# Train Phase A from scratch (no initialization)
mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml \
  --run-id=phase-A-overtaking
# (Slower but safer)
```

### 🔴 RISK 2: Config YAML File Names

**Current:**
```
python/configs/planning/vehicle_ppo_v10g.yaml
python/configs/planning/vehicle_ppo_phase-A.yaml
```

**After Rename:**
```
python/configs/planning/vehicle_ppo_phase-0.yaml
python/configs/planning/vehicle_ppo_phase-A.yaml
```

**Risk:** Documentation/commands reference old config names

**Impact:** Medium
- Commands in README.md will be wrong
- Training will fail if wrong config name used
- Easy to fix (update docs + rename files)

---

## Rename Impact Summary

| Item | Risk Level | Breaks Training? | Breaks Docs? |
|------|-----------|------------------|--------------|
| experiments/ folder name | 🟢 LOW | ❌ No | ✅ Yes |
| Checkpoint data | 🟢 LOW | ❌ No | ❌ No |
| ONNX models | 🟢 LOW | ❌ No | ❌ No |
| TensorBoard logs | 🟢 LOW | ❌ No | ❌ No |
| Config YAML names | 🟡 MEDIUM | ⚠️ Indirect | ✅ Yes |
| --initialize-from paths | 🔴 HIGH | ✅ YES | ✅ Yes |

---

## Safe Rename Strategy

### Step 1: Fix initialize-from FIRST (Before rename)

**Create proper checkpoint locations:**
```bash
# Option: Copy v10g checkpoint to results/ with proper name
mkdir -p results/phase-0-foundation
cp -r experiments/phase-0-foundation/results/E2EDrivingAgent \
      results/phase-0-foundation/

# Verify:
ls results/phase-0-foundation/E2EDrivingAgent/checkpoint.pt
```

### Step 2: Rename experiments/ folders

```bash
git mv experiments/phase-0-foundation experiments/phase-0-foundation
git mv experiments/phase-A-overtaking experiments/phase-A-overtaking
# ... etc
```

**Why this is safe:**
- Training data moved with folder (git mv is atomic)
- results/ already has copy of checkpoint (from Step 1)
- No training disruption

### Step 3: Update documentation references

```bash
# Update all 302 references
find . -type f \( -name "*.md" -o -name "*.yaml" -o -name "*.txt" \) \
  -exec sed -i 's/phase-0-foundation/phase-0-foundation/g' {} +
```

### Step 4: Rename config files

```bash
git mv python/configs/planning/vehicle_ppo_v10g.yaml \
      python/configs/planning/vehicle_ppo_phase-0.yaml
```

### Step 5: Update Phase A initialize path

```bash
# In docs and configs, change:
--initialize-from=v10g_lane_keeping
# To:
--initialize-from=phase-0-foundation
```

---

## Verification Checklist

After rename, verify:

- [ ] `experiments/phase-0-foundation/results/E2EDrivingAgent/checkpoint.pt` exists
- [ ] `results/phase-0-foundation/E2EDrivingAgent/checkpoint.pt` exists (copy)
- [ ] Config files renamed and internal paths updated
- [ ] Documentation shows correct paths
- [ ] Test command works:
  ```bash
  mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml \
    --run-id=test_phase_a \
    --initialize-from=phase-0-foundation
  ```

---

## Worst Case Scenarios

### Scenario 1: Checkpoint not found
**Symptom:** ML-Agents error: "Could not find checkpoint"
**Cause:** --initialize-from path wrong
**Fix:** Copy checkpoint to correct results/ location
**Data Loss:** None (checkpoint still in experiments/)

### Scenario 2: Training starts from scratch
**Symptom:** Reward starts at 0 instead of continuing from v10g
**Cause:** initialize-from failed silently
**Fix:** Stop training, fix path, restart
**Data Loss:** Wasted training time only

### Scenario 3: Wrong config file used
**Symptom:** Training uses wrong hyperparameters
**Cause:** Old config file name used
**Fix:** Stop training, use correct config name
**Data Loss:** Wasted training time only

---

## Conclusion

### ✅ SAFE TO RENAME experiments/ folders

**Reasons:**
1. Training data is self-contained (checkpoint.pt doesn't reference parent folder)
2. ML-Agents uses results/ folder, not experiments/
3. Git mv preserves all data atomically
4. Easy rollback if issues arise

### ⚠️ BUT MUST FIX FIRST:

1. **Create checkpoint copies in results/** (for initialize-from)
2. **Update all initialize-from paths** in docs/configs
3. **Rename config files** consistently
4. **Verify one training command** before committing

### Recommendation:

**Proceed with rename, BUT:**
- Do Step 1 (fix initialize-from) FIRST
- Test one training command with new paths
- Then do full rename + documentation update
- Keep backup branch for rollback

**Estimated Risk After Fixes:** 🟢 LOW

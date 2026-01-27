# Experiment Folder Renaming - Complete Analysis

**Date:** 2026-01-27
**Status:** Analysis Complete - Ready for Review

---

## Executive Summary

**Total References Found:** 302
- v10g: 169 references
- phase-A: 48 references
- phase-B: 24 references
- phase-C: 16 references
- phase-E: 17 references
- phase-F: 16 references
- phase-G: 12 references

**Risk Level:** HIGH (config files + extensive documentation)

---

## 1. Current → Target Mapping

| Current Name | Target Name | Ref Count | Status |
|--------------|-------------|-----------|--------|
| `phase-0-foundation` | `phase-0-foundation` | 169 | 🔴 Critical |
| `phase-A-overtaking` | `phase-A-overtaking` | 48 | 🔴 Critical |
| `phase-B_decision_learning` | `phase-B-decision` | 24 | 🟡 Medium |
| `phase-C_multi_npc` | `phase-C-multi-npc` | 16 | 🟡 Medium |
| `phase-E_curved_roads` | `phase-E-curved-roads` | 17 | 🟡 Medium |
| `phase-F_multi_lane` | `phase-F-multi-lane` | 16 | 🟡 Medium |
| `phase-G_intersection` | `phase-G-intersection` | 12 | 🟡 Medium |

---

## 2. Reference Categories

### 🔴 CRITICAL (Break if wrong)

**Python Config Files (.yaml)**
- `python/configs/planning/vehicle_ppo_v*.yaml`
- Contains: `run_id`, `initialize_from` paths
- Impact: Training will fail

**Example:**
```yaml
# python/configs/planning/vehicle_ppo_phase-A.yaml
run_id: phase-A  # MUST CHANGE
initialize_from: v10g_lane_keeping  # VERIFY PATH
```

**Files to Check:**
- [ ] `python/configs/planning/vehicle_ppo_v10g.yaml`
- [ ] `python/configs/planning/vehicle_ppo_phase-A.yaml`
- [ ] All other phase configs

### 🟡 MEDIUM (Documentation links)

**Main Documentation**
- `docs/TRAINING-LOG.md` (training history)
- `docs/PROGRESS.md` (current status)
- `docs/LEARNING-ROADMAP.md` (roadmap)
- `experiments/README.md` (experiment index)

**Cross-references:**
```markdown
[v10g Analysis](../experiments/phase-0-foundation/ANALYSIS.md)
# Must become:
[v10g Analysis](../experiments/phase-0-foundation/ANALYSIS.md)
```

### 🟢 LOW (Self-references, Archives)

**Experiment Self-Documentation**
- `experiments/phase-0-foundation/README.md` (self-reference)
- `docs/archives/*.md` (historical, can keep old names)

---

## 3. Detailed Reference Locations

### phase-0-foundation (169 refs)

**Top Referenced Files:**
1. `experiments/phase-0-foundation/README.md` (50+ self-references)
2. `docs/LEARNING-ROADMAP.md` (multiple sections)
3. `docs/TRAINING-LOG.md` (results section)
4. `docs/PROGRESS.md` (status updates)
5. `python/configs/planning/vehicle_ppo_v10g.yaml`

**Critical Patterns:**
```
experiments/phase-0-foundation/          # Folder path
phase-0-foundation                       # Run ID
v10g_lane_keeping                     # Checkpoint name
vehicle_ppo_v10g.yaml                 # Config filename
```

### phase-A (48 refs)

**Critical Files:**
- `python/configs/planning/vehicle_ppo_phase-A.yaml` (config)
- `experiments/phase-A-overtaking/` (all docs)
- `docs/TRAINING-LOG.md` (training commands)

**Patterns:**
```
phase-A                            # Run ID
phase-A-overtaking           # Folder name
vehicle_ppo_phase-A.yaml           # Config file
--run-id=phase-A                   # CLI argument
```

---

## 4. Risk Assessment

### HIGH RISK Areas

1. **Config YAML Files**
   - Risk: Training fails if paths wrong
   - Files: `python/configs/planning/*.yaml`
   - Action: Manual verification required

2. **Initialize_from Paths**
   - Risk: Checkpoint loading fails
   - Pattern: `initialize_from: v10g_lane_keeping`
   - Action: Verify checkpoint paths exist

3. **Run IDs in Commands**
   - Risk: Documentation shows wrong commands
   - Pattern: `--run-id=phase-A`
   - Action: Update all training commands

### MEDIUM RISK Areas

1. **Cross-document Links**
   - Risk: Broken markdown links
   - Pattern: `[text](../experiments/phase-0-foundation/...)`
   - Action: Update relative paths

2. **README References**
   - Risk: Confusion, outdated info
   - Files: All README.md files
   - Action: Systematic replacement

### LOW RISK Areas

1. **Archive Files**
   - Risk: Historical only, no functional impact
   - Files: `docs/archives/*.md`
   - Action: Optional update

2. **Self-references**
   - Risk: Automatically fixed by folder rename
   - Pattern: References within own folder
   - Action: Update after folder rename

---

## 5. Execution Strategy

### Phase 1: Preparation (5 min)
1. ✅ Git status clean (already done)
2. Create backup branch: `git checkout -b backup-before-rename`
3. Return to master: `git checkout master`
4. Create RENAME_PLAN.md with exact commands

### Phase 2: Folder Rename (2 min)
```bash
# Rename folders (preserves git history)
git mv experiments/phase-0-foundation experiments/phase-0-foundation
git mv experiments/phase-A-overtaking experiments/phase-A-overtaking
git mv experiments/phase-B_decision_learning experiments/phase-B-decision
git mv experiments/phase-C_multi_npc experiments/phase-C-multi-npc
git mv experiments/phase-E_curved_roads experiments/phase-E-curved-roads
git mv experiments/phase-F_multi_lane experiments/phase-F-multi-lane
git mv experiments/phase-G_intersection experiments/phase-G-intersection
```

### Phase 3: Reference Update (10-15 min)
**Use sed for systematic replacement:**

```bash
# 1. phase-0-foundation → phase-0-foundation
find . -type f \( -name "*.md" -o -name "*.yaml" -o -name "*.txt" \) \
  -exec sed -i 's/phase-0-foundation/phase-0-foundation/g' {} +

# 2. phase-A → phase-A
find . -type f \( -name "*.md" -o -name "*.yaml" -o -name "*.txt" \) \
  -exec sed -i 's/phase-A/phase-A/g' {} +

# 3-7. Other phases...
```

**Manual Review Required:**
- [ ] `python/configs/planning/*.yaml` (verify run_id)
- [ ] `docs/TRAINING-LOG.md` (verify commands)
- [ ] `experiments/README.md` (verify structure)

### Phase 4: Config File Rename (3 min)
```bash
# Rename config files to match new naming
git mv python/configs/planning/vehicle_ppo_v10g.yaml \
      python/configs/planning/vehicle_ppo_phase-0.yaml

git mv python/configs/planning/vehicle_ppo_phase-A.yaml \
      python/configs/planning/vehicle_ppo_phase-A.yaml

# Update internal references in renamed files
```

### Phase 5: Validation (5 min)
```bash
# Check for remaining old names
grep -r "phase-0-foundation" --include="*.md" --include="*.yaml" .
grep -r "v12_phase[A-G]" --include="*.md" --include="*.yaml" .

# Should return 0 results (except archives)
```

### Phase 6: Commit (2 min)
```bash
git add -A
git commit -m "refactor: Rename experiments to phase-centric naming

- phase-0-foundation → phase-0-foundation
- phase-A → phase-A-overtaking
- Updated all 302 references across documentation
- Renamed config files to match new structure
- Verified all cross-references and paths"

git push origin master
```

---

## 6. Conflict Analysis

### Potential Issues

1. **Config filename vs folder name mismatch**
   - Old: `vehicle_ppo_v10g.yaml` + `experiments/phase-0-foundation/`
   - New: `vehicle_ppo_phase-0.yaml` + `experiments/phase-0-foundation/`
   - Solution: Update both consistently

2. **Checkpoint names in results/**
   - Old: `results/v10g/` (may still exist)
   - New: References become `phase-0`?
   - Solution: Keep results/ unchanged, only update experiments/

3. **Archive files reference old names**
   - Impact: Low (historical only)
   - Solution: Add note at top of archives: "Historical names preserved"

---

## 7. Validation Checklist

After renaming, verify:

- [ ] All 7 folders renamed successfully
- [ ] `experiments/README.md` updated with new structure
- [ ] `docs/TRAINING-LOG.md` commands use new names
- [ ] `docs/PROGRESS.md` references updated
- [ ] `docs/LEARNING-ROADMAP.md` roadmap updated
- [ ] Config files renamed and internal refs updated
- [ ] No grep results for old names (except archives)
- [ ] Git commit includes all changes
- [ ] Training command still works: `mlagents-learn python/configs/planning/vehicle_ppo_phase-A.yaml`

---

## 8. Rollback Strategy

If issues arise:

```bash
# Option 1: Git revert
git revert HEAD

# Option 2: Reset to backup branch
git reset --hard backup-before-rename

# Option 3: Cherry-pick individual fixes
git log --oneline  # Find commit before rename
git reset --hard <commit-hash>
```

---

## 9. Time Estimate

| Phase | Time | Risk |
|-------|------|------|
| Preparation | 5 min | Low |
| Folder Rename | 2 min | Low |
| Reference Update | 10-15 min | Medium |
| Config Rename | 3 min | Medium |
| Validation | 5 min | Low |
| Commit/Push | 2 min | Low |
| **TOTAL** | **27-32 min** | **Medium** |

---

## 10. Recommendation

### Proceed with Caution

**Strengths:**
- Clear mapping defined
- Systematic replacement strategy
- Validation checks in place
- Rollback strategy ready

**Risks:**
- 302 references to update
- Config files must be perfect
- Cross-references can break

**Recommendation:**
1. Review this analysis thoroughly
2. Create backup branch first
3. Use provided sed commands
4. Manually verify config files
5. Run validation checks
6. Test one training command before committing

**Alternative:**
- Do rename in stages (v10g first, test, then phases)
- Or: Keep current names, only rename new phases going forward

---

**Decision Required:** Proceed with rename? Y/N

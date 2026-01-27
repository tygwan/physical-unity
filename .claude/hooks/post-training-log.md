# Post-Training Documentation Hook

## 트리거 조건
학습 완료 감지 시 (다음 중 하나):
- 사용자가 "학습 완료" 언급
- mlagents-learn 프로세스 종료 확인
- max_steps 도달 로그 확인

## 자동 실행 작업

### 1. 학습 데이터 수집
```bash
# 최신 학습 로그 확인
tail -50 results/<run_id>/run_logs/training_status.json
tail -100 <background_task_output>
```

### 2. PROGRESS.md 업데이트
- 현재 Phase 상태
- 최종 reward 수치
- 커리큘럼 완료 상태
- 체크포인트 정보

### 3. TRAINING-LOG.md 업데이트
- Training Summary 테이블
- Training Progress 테이블
- Checkpoints Saved 목록
- Key Achievements

### 4. LEARNING-ROADMAP.md 업데이트
- Executive Summary 상태 변경
- 성공/실패 이력에 기록 추가
- 다음 Phase 전략 제안

## 업데이트 템플릿

### PROGRESS.md 섹션
```markdown
### Current Training Status
| Metric | Value |
|--------|-------|
| Phase | Phase X |
| Steps | X,XXX,XXX / X,XXX,XXX |
| Current Reward | +XXX |
| Status | ✅ Completed / 🔄 In Progress |
| Last Updated | YYYY-MM-DD HH:MM |
```

### TRAINING-LOG.md 섹션
```markdown
### Phase X Training Log - COMPLETED YYYY-MM-DD

#### Training Summary
| Metric | Value |
|--------|-------|
| Total Steps | X,XXX,XXX |
| Final Reward | +XXX |
| Peak Reward | +XXX (at X.XM steps) |
| Training Time | ~XX minutes |

#### Key Achievements
1. Achievement 1
2. Achievement 2

#### Lessons Learned
1. Lesson 1
2. Lesson 2
```

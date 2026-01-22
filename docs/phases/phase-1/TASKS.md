# Phase 1: Tasks

## Task List

### 1. WSL Python 환경 설정
- [x] Python 3.10 설치 확인
- [x] 가상환경 생성 (`.venv`)
- [x] PyTorch 설치
- [x] mlagents 설치
- [x] TensorBoard 설치
- [x] `mlagents-learn --help` 테스트

### 2. Unity 환경 설정
- [ ] Unity Hub에서 2023.2+ 버전 확인
- [ ] 새 3D 프로젝트 생성 (`unity/` 폴더)
- [ ] ML-Agents Package 설치 (`com.unity.ml-agents`)
- [ ] ML-Agents 예제 Import

### 3. 연동 테스트
- [ ] WSL에서 `mlagents-learn` 실행
- [ ] Unity Editor에서 Play
- [ ] 소켓 연결 확인 (localhost:5004)
- [ ] 3D Ball 예제 학습 시작

### 4. 기초 학습
- [ ] Unity Rigidbody 기초 이해
- [ ] ML-Agents Agent 클래스 구조 파악
- [ ] 학습 설정 파일 (yaml) 구조 이해

### 5. 문서화
- [ ] 설치 과정 README 업데이트
- [ ] 트러블슈팅 기록

---

## Progress

| Task | Status | Note |
|------|--------|------|
| WSL Python 환경 | ✅ 완료 | Python 3.10, PyTorch 2.8, mlagents |
| Unity 환경 | ⏳ 대기 | Windows에서 진행 필요 |
| 연동 테스트 | ⏳ 대기 | - |
| 기초 학습 | ⏳ 대기 | - |
| 문서화 | 🔄 진행중 | 기본 문서 완료 |

---

## Blockers

*현재 블로커 없음*

---

## Notes

- Unity 프로젝트는 Windows 경로에 생성 권장
- WSL에서 Windows Unity 프로젝트 접근: `/mnt/c/...`
- 첫 학습은 CPU 모드로 진행 (GPU 설정은 Phase 2에서)

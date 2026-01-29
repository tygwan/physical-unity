# Phase 1: Checklist

## Pre-requisites

- [x] Windows 11 with WSL2 설치됨
- [x] Unity Hub 설치됨
- [x] Git 설치됨
- [x] 프로젝트 저장소 초기화됨

## Environment Setup

### WSL Python
- [ ] Python 3.10.x 설치
- [ ] 가상환경 생성: `python3.10 -m venv .venv`
- [ ] 가상환경 활성화: `source .venv/bin/activate`
- [ ] PyTorch 설치: `pip install torch`
- [ ] ML-Agents 설치: `pip install mlagents`
- [ ] 설치 확인: `mlagents-learn --help`

### Unity
- [ ] Unity 2023.2+ 버전 설치
- [ ] 새 프로젝트 생성 (3D URP 권장)
- [ ] Package Manager → `com.unity.ml-agents` 추가
- [ ] 예제 Assets Import

## Integration Test

- [ ] WSL 터미널: `mlagents-learn config.yaml --run-id=test`
- [ ] Unity Editor: Play 버튼 클릭
- [ ] 연결 성공 메시지 확인
- [ ] 학습 시작 확인 (스텝 증가)
- [ ] TensorBoard 실행: `tensorboard --logdir=logs`

## Documentation

- [ ] README.md 설치 과정 검증
- [ ] 발생한 이슈 기록
- [ ] 해결 방법 문서화

## Completion Criteria

- [ ] `mlagents-learn` 정상 동작
- [ ] Unity ML-Agents 패키지 정상 동작
- [ ] 3D Ball 예제 학습 가능
- [ ] TensorBoard 모니터링 가능

---

## Sign-off

- [ ] 모든 체크리스트 완료
- [ ] Phase 1 완료 확인
- [ ] Phase 2 시작 준비

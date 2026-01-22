# Physical AI with Unity

Unity ML-Agents를 활용한 Physical AI 학습 프로젝트

## 프로젝트 구조

```
physical-unity/
├── docs/                    # 문서 (학습 자료, 설계 문서)
├── python/                  # Python ML 코드 (WSL)
│   ├── src/                 # 학습 스크립트
│   ├── configs/             # 학습 설정 파일 (yaml)
│   └── notebooks/           # Jupyter 실험 노트북
├── unity/                   # Unity 프로젝트 (Windows)
├── scripts/                 # 유틸리티 스크립트
├── models/                  # 학습된 모델 (.onnx)
└── logs/                    # TensorBoard 로그
```

## 환경 구성

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                        Windows                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Unity Editor (2023.2+)                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│  │  │ ML-Agents   │  │  물리엔진   │  │   센서      │  │    │
│  │  │ Package 3.0 │  │ (Rigidbody) │  │ (Ray, Cam)  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │    │
│  └────────────────────────┬────────────────────────────┘    │
│                           │ Socket (localhost:5004)          │
│  ┌────────────────────────┴────────────────────────────┐    │
│  │                      WSL2                            │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│  │  │  Python     │  │  PyTorch    │  │  mlagents   │  │    │
│  │  │  3.10.x     │  │   2.0+      │  │  (trainer)  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 필수 요구사항

| 구성요소 | 버전 | 설치 위치 |
|---------|------|----------|
| Unity | 2023.2 이상 | Windows |
| ML-Agents Unity Package | 3.0.0 | Unity Package Manager |
| Python | 3.10.x | WSL |
| PyTorch | 2.0+ | WSL (pip) |
| mlagents | 1.0+ | WSL (pip) |

### WSL 환경 설정

```bash
# 1. Python 가상환경 생성
cd /home/coffin/dev/physical-unity
python3.10 -m venv .venv
source .venv/bin/activate

# 2. 의존성 설치
pip install torch torchvision torchaudio
pip install mlagents

# 3. 설치 확인
mlagents-learn --help
```

### Unity 설정

1. Unity Hub에서 새 프로젝트 생성 (3D URP 권장)
2. Window > Package Manager > Add package by name
3. `com.unity.ml-agents` 추가
4. 프로젝트를 `unity/` 폴더에 저장

## 학습 실행

```bash
# WSL에서 학습 시작
mlagents-learn python/configs/trainer_config.yaml --run-id=my_run

# Unity에서 Play 버튼 클릭하여 환경 연결
```

## 참고 자료

- [Unity ML-Agents 공식 문서](https://unity-technologies.github.io/ml-agents/)
- [ML-Agents GitHub](https://github.com/Unity-Technologies/ml-agents)
- [Installation Guide](https://github.com/Unity-Technologies/ml-agents/blob/develop/docs/Installation.md)

## License

MIT

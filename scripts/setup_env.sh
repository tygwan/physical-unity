#!/bin/bash
# Physical AI with Unity - WSL 환경 설정 스크립트

set -e

PYTHON_VERSION="3.10"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "========================================"
echo "Physical AI with Unity - 환경 설정"
echo "========================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# 1. Python 버전 확인
echo ""
echo "1. Python 버전 확인..."
if command -v python${PYTHON_VERSION} &> /dev/null; then
    PYTHON_CMD="python${PYTHON_VERSION}"
    print_status "Python ${PYTHON_VERSION} 발견: $(which $PYTHON_CMD)"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    INSTALLED_VERSION=$(python3 --version | cut -d' ' -f2)
    print_warning "Python3 사용 (버전: $INSTALLED_VERSION)"
    print_warning "권장: Python 3.10.x"
else
    print_error "Python을 찾을 수 없습니다"
    exit 1
fi

# 2. 가상환경 생성
echo ""
echo "2. 가상환경 생성..."
VENV_PATH="${PROJECT_ROOT}/.venv"

if [ -d "$VENV_PATH" ]; then
    print_warning "기존 가상환경 발견: $VENV_PATH"
    read -p "재생성하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_PATH"
        $PYTHON_CMD -m venv "$VENV_PATH"
        print_status "가상환경 재생성 완료"
    fi
else
    $PYTHON_CMD -m venv "$VENV_PATH"
    print_status "가상환경 생성 완료: $VENV_PATH"
fi

# 가상환경 활성화
source "${VENV_PATH}/bin/activate"
print_status "가상환경 활성화됨"

# 3. pip 업그레이드
echo ""
echo "3. pip 업그레이드..."
pip install --upgrade pip -q
print_status "pip 업그레이드 완료"

# 4. 의존성 설치
echo ""
echo "4. 의존성 설치..."

# PyTorch 설치 (CPU 버전 - GPU 사용시 수정 필요)
echo "   PyTorch 설치 중..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
print_status "PyTorch 설치 완료"

# ML-Agents 설치
echo "   ML-Agents 설치 중..."
pip install mlagents -q
print_status "ML-Agents 설치 완료"

# 추가 유틸리티
echo "   추가 패키지 설치 중..."
pip install tensorboard jupyter matplotlib -q
print_status "추가 패키지 설치 완료"

# 5. 설치 확인
echo ""
echo "5. 설치 확인..."
echo ""
echo "설치된 패키지:"
echo "  - PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  - ML-Agents: $(pip show mlagents | grep Version | cut -d' ' -f2)"
echo ""

# mlagents-learn 테스트
if mlagents-learn --help > /dev/null 2>&1; then
    print_status "mlagents-learn 정상 동작"
else
    print_error "mlagents-learn 실행 실패"
fi

# 6. 완료 메시지
echo ""
echo "========================================"
echo -e "${GREEN}환경 설정 완료!${NC}"
echo "========================================"
echo ""
echo "다음 단계:"
echo "  1. 가상환경 활성화: source .venv/bin/activate"
echo "  2. Unity에서 ML-Agents 패키지 설치"
echo "  3. 예제 실행: mlagents-learn python/configs/trainer_config.yaml --run-id=test"
echo ""

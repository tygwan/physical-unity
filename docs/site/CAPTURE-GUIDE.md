---
layout: default
title: Capture Guide
---

# 이미지/비디오 캡처 가이드

GitHub Pages용 학습 과정 시각 자료 확보 방법

---

## 1. Unity 스크린샷 캡처

### 방법 A: Unity Recorder (권장)

```
1. Window > General > Recorder > Recorder Window
2. Add Recorder > Image Sequence
3. 설정:
   - Output Format: PNG
   - Resolution: 1920x1080 (또는 Game View)
   - Output Path: Assets/Screenshots/
4. Recording Mode: Single Frame (또는 Time Interval)
5. START RECORDING 클릭
```

### 방법 B: Game View 스크린샷

```csharp
// 코드로 스크린샷 캡처
ScreenCapture.CaptureScreenshot("Assets/Screenshots/phase-g-screenshot.png");
```

### 방법 C: Windows 기본 도구

```
1. Unity Game View 활성화
2. Win + Shift + S (Snipping Tool)
3. 영역 선택 후 저장
```

---

## 2. Unity 비디오 녹화

### 방법 A: Unity Recorder (권장)

```
1. Window > General > Recorder > Recorder Window
2. Add Recorder > Movie
3. 설정:
   - Output Format: MP4 (H.264)
   - Resolution: 1920x1080
   - Frame Rate: 30 FPS
   - Output Path: Assets/Videos/
4. Recording Mode: Manual (또는 Time Interval)
5. START RECORDING → 녹화 → STOP RECORDING
```

### 방법 B: OBS Studio

```
1. OBS Studio 설치
2. Sources > Window Capture > Unity Editor
3. 녹화 시작/중지
4. MP4로 저장
```

### 방법 C: Windows Game Bar

```
1. Unity 창 활성화
2. Win + G (Game Bar 열기)
3. 녹화 버튼 클릭 또는 Win + Alt + R
```

---

## 3. 파일 저장 규칙

### 스크린샷 네이밍

```
Assets/Screenshots/
├── phase-{phase}-{stage}-{description}.png
│
├── phase-a-overtake-beside.png       # Phase A: 추월 중
├── phase-e-curved-sharp.png          # Phase E: 급커브
├── phase-f-multilane-2lane.png       # Phase F: 2차선
├── phase-g-t-junction-approach.png   # Phase G: T자 진입
└── phase-g-intersection-turn.png     # Phase G: 교차로 회전
```

### 비디오 네이밍

```
Assets/Videos/
├── phase-{phase}-demo.mp4            # 전체 데모
├── phase-{phase}-{feature}.mp4       # 특정 기능
│
├── phase-a-overtake-demo.mp4
├── phase-f-lane-change.mp4
└── phase-g-intersection-turn.mp4
```

---

## 4. GitHub Pages 업로드

### 이미지 복사

```bash
# Assets에서 docs/site로 복사
cp Assets/Screenshots/phase-*.png docs/site/gallery/screenshots/
```

### 비디오 처리

**옵션 1: GIF 변환 (작은 파일)**
```bash
# ffmpeg로 MP4 → GIF
ffmpeg -i phase-a-demo.mp4 -vf "fps=10,scale=640:-1" phase-a-demo.gif
```

**옵션 2: YouTube 업로드 후 임베드**
```markdown
{% include youtube.html id="VIDEO_ID" %}
```

**옵션 3: GitHub Releases에 첨부**
- 큰 비디오 파일은 Release Assets로 업로드
- 링크로 참조

---

## 5. 캡처 타이밍 가이드

### 학습 중 캡처할 장면

| 시점 | 캡처 내용 | 파일명 예시 |
|------|----------|------------|
| 학습 시작 | 초기 환경 | phase-g-start.png |
| 커리큘럼 전환 | 새 환경 첫 장면 | phase-g-t-junction-first.png |
| 성공적 행동 | 추월/회전 성공 | phase-g-turn-success.png |
| 실패 케이스 | 충돌/이탈 | phase-g-collision.png |
| 학습 완료 | 최종 성능 | phase-g-final.png |

### TensorBoard 그래프 캡처

```
1. tensorboard --logdir=results
2. 브라우저에서 localhost:6006
3. Reward 그래프 선택
4. 우클릭 > Save Image As...
5. docs/site/gallery/charts/phase-g-reward.png
```

---

## 6. 자동 캡처 스크립트 (선택)

### Unity 자동 스크린샷

```csharp
// ScreenshotCapture.cs
using UnityEngine;

public class ScreenshotCapture : MonoBehaviour
{
    public string screenshotPath = "Assets/Screenshots/";
    public KeyCode captureKey = KeyCode.F12;

    void Update()
    {
        if (Input.GetKeyDown(captureKey))
        {
            string filename = $"{screenshotPath}capture_{System.DateTime.Now:yyyyMMdd_HHmmss}.png";
            ScreenCapture.CaptureScreenshot(filename);
            Debug.Log($"Screenshot saved: {filename}");
        }
    }
}
```

### 주기적 자동 캡처

```csharp
// 학습 중 10만 스텝마다 자동 캡처
public class TrainingCapture : MonoBehaviour
{
    private int lastCaptureStep = 0;
    private int captureInterval = 100000;

    void OnEpisodeBegin()
    {
        int currentStep = Academy.Instance.StepCount;
        if (currentStep - lastCaptureStep >= captureInterval)
        {
            CaptureScreenshot($"training_step_{currentStep}");
            lastCaptureStep = currentStep;
        }
    }
}
```

---

## 7. 체크리스트

### 각 Phase 완료 시 캡처 목록

- [ ] 시작 장면 스크린샷
- [ ] 주요 커리큘럼 단계별 스크린샷
- [ ] 성공 케이스 스크린샷/GIF
- [ ] TensorBoard reward 그래프
- [ ] 30초~1분 데모 비디오
- [ ] 실패 케이스 (있다면)

---

## Quick Reference

```bash
# Unity Recorder 단축키 설정
Edit > Shortcuts > Recorder > Start Recording: Ctrl+Shift+R

# 스크린샷 저장 위치
Assets/Screenshots/

# 비디오 저장 위치
Assets/Videos/

# GitHub Pages 이미지 위치
docs/site/gallery/screenshots/

# GitHub Pages 비디오 위치
docs/site/gallery/videos/
```

---

[← Back to Home](./)

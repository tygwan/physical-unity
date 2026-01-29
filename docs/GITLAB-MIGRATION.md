# GitLab Migration Guide

GitHub에서 GitLab으로 프로젝트를 마이그레이션하는 가이드입니다.

---

## 마이그레이션 옵션

### Option 1: GitLab으로 완전 이전 (권장)
GitHub를 버리고 GitLab만 사용

### Option 2: Dual Remote (양쪽 유지)
GitHub와 GitLab을 동시에 유지

### Option 3: GitLab 미러링
GitHub를 primary로, GitLab을 mirror로 사용

---

## 1. GitLab 프로젝트 생성

### 1-1. GitLab.com 계정 확인
- https://gitlab.com 로그인
- 계정이 없으면 회원가입

### 1-2. 새 프로젝트 생성

**옵션 A: 웹 UI에서 생성**
1. https://gitlab.com/projects/new 접속
2. **Create blank project** 선택
3. 프로젝트 정보 입력:
   - **Project name**: `physical-unity`
   - **Project URL**: `gitlab.com/[username]/physical-unity`
   - **Visibility**: Public (또는 Private)
   - **Initialize repository with a README**: 체크 해제 (기존 코드 push할 예정)
4. **Create project** 클릭

**옵션 B: GitLab CLI로 생성**
```bash
# glab CLI 설치 필요 (https://gitlab.com/gitlab-org/cli)
glab project create physical-unity --public
```

---

## 2. Remote 설정

### Option 1: GitLab으로 완전 이전

```bash
cd C:\Users\user\Desktop\dev\physical-unity

# GitHub remote 제거
git remote remove origin

# GitLab remote 추가
git remote add origin https://gitlab.com/[username]/physical-unity.git

# 확인
git remote -v
```

### Option 2: Dual Remote (GitHub + GitLab)

```bash
cd C:\Users\user\Desktop\dev\physical-unity

# GitHub를 'github'로 이름 변경
git remote rename origin github

# GitLab을 'origin'으로 추가 (primary)
git remote add origin https://gitlab.com/[username]/physical-unity.git

# 또는 GitLab을 'gitlab'으로 추가 (secondary)
git remote add gitlab https://gitlab.com/[username]/physical-unity.git

# 확인
git remote -v
# github  https://github.com/tygwan/physical-unity.git (fetch)
# github  https://github.com/tygwan/physical-unity.git (push)
# origin  https://gitlab.com/[username]/physical-unity.git (fetch)
# origin  https://gitlab.com/[username]/physical-unity.git (push)
```

### Option 3: Push to Both

```bash
# origin remote를 양쪽으로 push하도록 설정
git remote set-url --add --push origin https://github.com/tygwan/physical-unity.git
git remote set-url --add --push origin https://gitlab.com/[username]/physical-unity.git

# 확인
git remote show origin
# Push  URL: https://github.com/tygwan/physical-unity.git
# Push  URL: https://gitlab.com/[username]/physical-unity.git
```

---

## 3. Code Push to GitLab

### 3-1. 전체 히스토리 Push

```bash
cd C:\Users\user\Desktop\dev\physical-unity

# master 브랜치 push
git push origin master

# 모든 브랜치 push
git push origin --all

# 모든 태그 push
git push origin --tags
```

### 3-2. gh-pages 브랜치 Push (GitHub Pages용)

GitLab Pages는 다른 방식으로 동작하므로, 나중에 별도 설정 필요.

```bash
# gh-pages 브랜치는 일단 push만
git push origin gh-pages
```

---

## 4. GitLab Pages 설정

GitLab Pages는 **CI/CD 파이프라인**으로 배포합니다.

### 4-1. `.gitlab-ci.yml` 생성

**Option A: Jekyll 사이트 배포**

```yaml
# .gitlab-ci.yml
image: ruby:2.7

variables:
  JEKYLL_ENV: production

before_script:
  - cd physical-unity-site
  - bundle install

pages:
  stage: deploy
  script:
    - bundle exec jekyll build -d public
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == "master"
```

**Option B: 정적 파일 배포 (HTML/CSS/JS)**

```yaml
# .gitlab-ci.yml
pages:
  stage: deploy
  script:
    - mkdir -p public
    - cp -r physical-unity-site/* public/
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == "master"
```

**Option C: VitePress/Docusaurus (나중에 마이그레이션 후)**

```yaml
# .gitlab-ci.yml (VitePress 예시)
image: node:18

pages:
  stage: deploy
  cache:
    paths:
      - node_modules/
  script:
    - npm install
    - npm run docs:build
    - mv docs/.vitepress/dist public
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == "master"
```

### 4-2. GitLab Pages URL

배포 후 다음 URL로 접속 가능:
```
https://[username].gitlab.io/physical-unity/
```

---

## 5. GitLab vs GitHub 차이점

| Feature | GitHub | GitLab |
|---------|--------|--------|
| **Pages URL** | `[user].github.io/[repo]` | `[user].gitlab.io/[repo]` |
| **Pages 배포** | 자동 (gh-pages) | CI/CD 파이프라인 필요 |
| **CI/CD** | GitHub Actions (`.github/workflows/*.yml`) | GitLab CI (`.gitlab-ci.yml`) |
| **Issue Tracking** | GitHub Issues | GitLab Issues (더 강력) |
| **Wiki** | GitHub Wiki | GitLab Wiki (더 통합적) |
| **Container Registry** | GitHub Packages | GitLab Container Registry (무료) |
| **프라이빗 저장소** | 유료 (일부 무료) | 무료 무제한 |

---

## 6. CI/CD 마이그레이션

GitHub Actions → GitLab CI 변환

### GitHub Actions 예시
```yaml
# .github/workflows/deploy.yml
name: Deploy to Pages
on:
  push:
    branches: [master]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build
        run: npm run build
```

### GitLab CI 변환
```yaml
# .gitlab-ci.yml
stages:
  - build
  - deploy

build:
  stage: build
  image: node:18
  script:
    - npm run build
  artifacts:
    paths:
      - dist/

pages:
  stage: deploy
  dependencies:
    - build
  script:
    - cp -r dist public
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == "master"
```

---

## 7. 프로젝트 Import (대안)

GitLab의 Import 기능 사용:

1. https://gitlab.com/projects/new#import_project 접속
2. **Import project** 탭 선택
3. **GitHub** 클릭
4. GitHub 계정 연결
5. `tygwan/physical-unity` 선택
6. **Import** 클릭

**장점**:
- 전체 히스토리, 이슈, PR 모두 마이그레이션
- Wiki, Releases도 가져옴

**단점**:
- GitHub 계정 연결 필요
- CI/CD는 수동 설정 필요

---

## 8. Dual Remote 워크플로우

양쪽을 모두 유지하는 경우:

### Push to Both
```bash
# master 브랜치만
git push github master
git push gitlab master

# 또는 모두 (origin이 both로 설정된 경우)
git push origin master
```

### Fetch from Primary
```bash
# GitLab을 primary로
git fetch origin
git pull origin master

# GitHub는 백업으로
git fetch github
```

### Branch Strategy
```bash
# 새 기능 개발
git checkout -b feature/new-feature
git push origin feature/new-feature  # GitLab MR용
git push github feature/new-feature  # GitHub PR용
```

---

## 9. README 업데이트

GitLab 마이그레이션 후 README 배지 업데이트:

```markdown
<!-- GitHub 배지 -->
![GitHub Stars](https://img.shields.io/github/stars/tygwan/physical-unity?style=social)

<!-- GitLab 배지 -->
[![pipeline status](https://gitlab.com/[username]/physical-unity/badges/master/pipeline.svg)](https://gitlab.com/[username]/physical-unity/-/commits/master)
[![coverage report](https://gitlab.com/[username]/physical-unity/badges/master/coverage.svg)](https://gitlab.com/[username]/physical-unity/-/commits/master)
```

---

## 10. 체크리스트

마이그레이션 완료 확인:

- [ ] GitLab 프로젝트 생성
- [ ] Remote 설정 (origin → gitlab)
- [ ] master 브랜치 push
- [ ] 모든 브랜치 push
- [ ] 태그 push
- [ ] `.gitlab-ci.yml` 작성
- [ ] GitLab Pages 배포 확인
- [ ] CI/CD 파이프라인 동작 확인
- [ ] README 업데이트 (GitLab URL)
- [ ] 기존 이슈/PR 마이그레이션 (선택)

---

## Quick Start Commands

### 완전 이전 (GitHub → GitLab)

```bash
cd C:\Users\user\Desktop\dev\physical-unity

# 1. Remote 변경
git remote remove origin
git remote add origin https://gitlab.com/[username]/physical-unity.git

# 2. Push
git push -u origin master
git push origin --all
git push origin --tags

# 3. CI/CD 설정 파일 생성
cat > .gitlab-ci.yml << 'EOF'
pages:
  stage: deploy
  script:
    - mkdir -p public
    - cp -r physical-unity-site/* public/
  artifacts:
    paths:
      - public
  rules:
    - if: $CI_COMMIT_BRANCH == "master"
EOF

# 4. Commit & Push
git add .gitlab-ci.yml
git commit -m "ci: Add GitLab CI/CD configuration"
git push origin master
```

### Dual Remote (GitHub + GitLab)

```bash
cd C:\Users\user\Desktop\dev\physical-unity

# 1. Remote 추가
git remote rename origin github
git remote add origin https://gitlab.com/[username]/physical-unity.git

# 2. Push to GitLab
git push -u origin master

# 3. 이후 push는 양쪽으로
git push github master
git push origin master
```

---

## Troubleshooting

### Issue 1: Authentication Failed

**증상**: `remote: HTTP Basic: Access denied`
**해결**:
```bash
# Personal Access Token 생성 필요
# GitLab → Settings → Access Tokens → Create token
# Scopes: api, read_repository, write_repository

# Token으로 인증
git remote set-url origin https://oauth2:[TOKEN]@gitlab.com/[username]/physical-unity.git
```

### Issue 2: Large Files (>100MB)

**증상**: `remote: fatal: pack exceeds maximum allowed size`
**해결**:
```bash
# Git LFS 사용
git lfs install
git lfs track "*.onnx"
git lfs track "*.mp4"
git add .gitattributes
git commit -m "chore: Add Git LFS tracking"
```

### Issue 3: CI/CD Pipeline Fails

**증상**: GitLab CI 빌드 실패
**해결**:
1. https://gitlab.com/[username]/physical-unity/-/pipelines 접속
2. 실패한 job 클릭
3. 로그 확인
4. `.gitlab-ci.yml` 수정

---

## Next Steps

GitLab 마이그레이션 완료 후:

1. **프론트엔드 업그레이드**: VitePress/Docusaurus로 변경
2. **CI/CD 최적화**: 테스트, 린트, 배포 자동화
3. **Container Registry**: Docker 이미지 저장
4. **GitLab Runner**: 자체 호스팅 빌드 환경

---

**Generated**: 2026-01-29
**Purpose**: GitHub → GitLab 마이그레이션 가이드

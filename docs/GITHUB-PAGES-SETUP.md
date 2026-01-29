# GitHub Pages Setup Guide

## Current Issue: 404 Not Found

**URL**: https://tygwan.github.io/physical-unity/
**Status**: 404 Error (Page not found)

---

## Diagnosis Checklist

### ✅ Confirmed Working
- [x] `gh-pages` branch exists
- [x] `_config.yml` properly configured:
  - `baseurl: "/physical-unity"`
  - `theme: minima`
- [x] `index.md` exists with frontmatter
- [x] Recent commits pushed (2026-01-29)
- [x] Jekyll structure is correct

### ❓ Needs Verification
- [ ] GitHub Pages enabled in Repository Settings
- [ ] Source branch set to `gh-pages`
- [ ] Jekyll build succeeded (no errors)
- [ ] Repository is public (not private)

---

## Fix Steps

### Step 1: Enable GitHub Pages (Web UI Required)

**GitHub Pages는 웹 UI에서만 설정 가능합니다.**

1. 웹 브라우저에서 https://github.com/tygwan/physical-unity/settings/pages 접속
2. **Source** 섹션에서:
   - Branch: `gh-pages` 선택
   - Folder: `/ (root)` 선택
3. **Save** 클릭
4. 5-10분 대기 (첫 배포는 시간 소요)

### Step 2: Verify Repository Visibility

GitHub Pages는 기본적으로 public repository만 지원합니다.

1. https://github.com/tygwan/physical-unity/settings 접속
2. **Danger Zone** → **Change repository visibility**
3. Repository가 **Public**인지 확인

### Step 3: Trigger Rebuild (Optional)

빌드를 다시 트리거하려면:

```bash
cd C:/Users/user/Desktop/dev/physical-unity-site
echo "" >> index.md  # 빈 줄 추가
git add index.md
git commit -m "docs: Trigger GitHub Pages rebuild"
git push origin gh-pages
```

### Step 4: Check Build Status

#### Option A: GitHub UI (권장)

1. https://github.com/tygwan/physical-unity/actions 접속
2. `pages build and deployment` 워크플로우 확인
3. 실패 시 에러 로그 확인

#### Option B: GitHub CLI (설치 필요)

```bash
gh run list --workflow="pages-build-deployment" --limit 5
gh run view <run-id> --log
```

---

## Common Issues

### Issue 1: Jekyll Build Failure

**증상**: GitHub Actions에서 빌드 실패
**해결**:
```bash
# 로컬에서 Jekyll 검증 (Ruby 필요)
cd C:/Users/user/Desktop/dev/physical-unity-site
bundle install
bundle exec jekyll serve

# 브라우저에서 http://localhost:4000/physical-unity/ 접속
```

### Issue 2: Theme Not Found

**증상**: `theme: minima` 인식 안됨
**해결**: `_config.yml` 수정
```yaml
# 기존
theme: minima

# 변경
remote_theme: jekyll/minima
```

### Issue 3: Baseurl Mismatch

**증상**: 페이지는 뜨지만 CSS/이미지가 깨짐
**해결**: `_config.yml` 확인
```yaml
baseurl: "/physical-unity"  # ✅ Correct
# baseurl: ""                # ❌ Wrong for project page
```

### Issue 4: Custom Domain CNAME

**증상**: Custom domain이 설정되어 있으면 기본 URL이 안됨
**해결**: `CNAME` 파일 제거
```bash
cd C:/Users/user/Desktop/dev/physical-unity-site
rm CNAME
git add CNAME
git commit -m "docs: Remove CNAME for default GitHub Pages URL"
git push origin gh-pages
```

---

## Automated Deployment (Future)

### Option 1: GitHub Actions Workflow

`.github/workflows/deploy-pages.yml`:
```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [master]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Pages
        uses: actions/configure-pages@v4

      - name: Build Jekyll site
        uses: actions/jekyll-build-pages@v1
        with:
          source: ./
          destination: ./_site

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

### Option 2: Manual Script

`scripts/deploy-site.sh`:
```bash
#!/bin/bash
set -e

# Build site
cd physical-unity-site
bundle exec jekyll build

# Deploy
git add -A
git commit -m "docs: Update site $(date +%Y-%m-%d)"
git push origin gh-pages

echo "✅ Deployed to https://tygwan.github.io/physical-unity/"
```

---

## Verification

배포 완료 후 다음을 확인:

1. **Main Page**: https://tygwan.github.io/physical-unity/
2. **Phase Pages**: https://tygwan.github.io/physical-unity/phases/phase-a/
3. **Gallery**: https://tygwan.github.io/physical-unity/gallery/
4. **CSS**: 스타일이 제대로 적용되는지
5. **Images**: 이미지가 로드되는지

---

## Current Site Structure

```
physical-unity-site/ (gh-pages branch)
├── _config.yml           # Jekyll configuration
├── index.md              # Home page
├── lessons-learned.md    # Lessons learned
├── CAPTURE-GUIDE.md      # Guide for capturing
├── phases/               # Phase documentation
│   ├── foundation/
│   ├── phase-a/
│   └── phase-b-v2/
└── gallery/              # Training videos/images
    └── phase-*/
```

---

## Next Steps

1. **웹 브라우저에서 GitHub 설정 확인** (필수)
   - https://github.com/tygwan/physical-unity/settings/pages

2. **Pages 상태 확인**
   - https://github.com/tygwan/physical-unity/deployments

3. **Actions 로그 확인**
   - https://github.com/tygwan/physical-unity/actions

4. **문제 지속 시**:
   - Repository를 public으로 변경
   - Jekyll 로컬 빌드 테스트
   - GitHub Support 문의

---

**Generated**: 2026-01-29
**Issue**: 404 Not Found on https://tygwan.github.io/physical-unity/
**Solution**: Verify GitHub Pages settings in web UI

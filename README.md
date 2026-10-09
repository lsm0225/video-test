# video-test

스토리보드를 받아 코드로 영상(애니매틱)을 만드는 테스트 프로젝트.
첫 작업: 성문고등학교 홍보영상 스토리보드 v5 → 16:9 1080p, 2분 37초, 내레이션 + 씬별 배경음악 + 자막.

## 구조
- `src/na_gen.py` 스토리보드 NA 문구로 가이드 내레이션 생성
- `src/build_timeline.py` 내레이션 길이와 씬별 음악 박자에 맞춰 컷 길이 확정 → `build/timeline.json`
- `src/video.html` 컷별 카메라 모션, 모션그래픽, 자막 (컷 연출은 파일 안 `SPEC`, 자막은 `SUBS`)
- `src/render.py` 위 페이지를 프레임 단위로 캡처해 mp4로 인코딩
- `src/mix.py` 음악 편집, 효과음 합성, 내레이션 믹스 → `build/audio.wav`
- `assets/audio/` 음원과 내레이션, 출처는 `assets/audio/LICENSES.md`
- `storyboard/timeline.json` 확정된 컷 타임라인

## 실행
```bash
pip install playwright librosa soundfile scipy sherpa-onnx
python3 src/build_timeline.py
python3 src/render.py            # --no-tag 를 붙이면 좌상단 컷 번호 표시 제거
python3 src/mix.py
ffmpeg -i build/video.mp4 -i build/audio.wav -c:v libx264 -crf 22 -c:a aac -shortest output/out.mp4
```

## 저장소에 넣지 않은 것
- `assets/ref/` 컷별 참고 이미지(스토리보드에서 추출해 4배 업스케일)와 스토리보드 원본 PDF: 클라이언트 자료라 공개 저장소에서 제외
- `output/`, `build/` 렌더 결과물
- 내레이션 음성 모델(`models/`)

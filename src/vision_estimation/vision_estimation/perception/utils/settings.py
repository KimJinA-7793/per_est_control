# settings.py
# ============================================================
# 최소 설정 버전 (경로/구조 설명용)
#
# ❗ 중요 원칙
# - 알고리즘 로직에는 관여하지 않는다
# - 좌표계/하드코딩/파라미터에는 영향 없다
# - "어디에 뭐가 있고, 누가 뭘 하는지"를 정리하는 용도
# ============================================================


# =========================
# 📂 프로젝트 파일 역할 정리
# =========================
#
# main.py
#   - 프로그램 실행 엔트리
#   - realsense_loop.run() 호출
#   - 각 모듈의 콜백만 연결 (배선 역할)
#
# realsense_loop.py
#   - RealSense 카메라 루프
#   - color/depth 프레임 수신
#   - 키 입력(space/r/esc) 처리
#   - 화면 표시 + 클릭 콜백 연결
#
# click_points.py
#   - 마우스 클릭 좌표 관리
#   - 스냅샷 시점의 클릭 포인트를 world 좌표로 변환
#   - Save_Cam() 제공 (color, depth, points_3d)
#
# detector.py
#   - 물체 자동 검출 로직
#   - 초록 박스: depth + DBSCAN + 회전 사각형
#   - 파랑 박스: RGB 윤곽 + depth hole
#   - run() -> (vis, items) 반환
#
# depth_utils.py
#   - depth 관련 유틸 모음
#   - FakeDepthFrameFromNpy (depth.npy 래퍼)
#   - 박스 중심 계산
#   - 파랑 박스용 안전 depth 탐색
#
# coordinate.py
#   - pixel + depth -> world 좌표 변환
#   - camcalib.npz 로드
#   - 5x5 depth median + 보정값 적용
#
# pipeline.py
#   - 스페이스바 눌렀을 때 실행되는 핵심 파이프라인
#   - 저장 → 검출 → world 계산 → flat list 생성
#   - flat_world_list / flat_clicked_xy print 유지
#   - 리턴 튜플 형태 유지
#
# settings.py (이 파일)
#   - 경로만 중앙 관리
#   - 구조/역할 설명용 메모 역할
#
# outputs/
#   - 실행 결과 저장 폴더
#   - color.jpg / depth.npy 저장


# =========================
# 📁 출력 경로 설정
# =========================

# 출력 폴더
OUTPUT_DIR = "outputs"

# 저장 파일 경로
COLOR_PATH = "outputs/color.jpg"
DEPTH_PATH = "outputs/depth.npy"

# 캘리브레이션 파일 (프로젝트 루트에 위치)
CALIB_PATH = "camcalib.npz"
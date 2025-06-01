import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 예시 데이터 ─ 실제 사용 시엔 gt, pred 텐서를 여기에 그대로 넣으세요
# ------------------------------------------------------------------
L = 120                     # time-series length
np.random.seed(0)
gt   = (np.random.rand(L) < 0.2).astype(int)   # ground truth (20 % ones)
pred = (np.random.rand(L) < 0.2).astype(int)   # prediction (20 % ones)

# ------------------------------------------------------------------
# 1) 두 줄짜리 타임라인 + 색으로 정답/오답 표시
# ------------------------------------------------------------------
def plot_hits(gt, pred):
    gt   = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    t    = np.arange(len(gt))

    # 마커 위치를 0–1 두 줄로 나눔
    y_gt, y_pr = np.ones_like(t), np.zeros_like(t)

    # 분류 결과
    tp = (gt == 1) & (pred == 1)
    fn = (gt == 1) & (pred == 0)
    fp = (gt == 0) & (pred == 1)

    plt.figure(figsize=(12, 2.4))
    # ground truth 라인 (검은색)
    plt.scatter(t[gt == 1], y_gt[gt == 1],
                marker='|', s=300, c='black', label='Ground-Truth = 1')
    # prediction 라인 (파란색)
    plt.scatter(t[pred == 1], y_pr[pred == 1],
                marker='|', s=300, c='blue',  label='Prediction = 1')

    # 옳고 그름을 겹쳐서 컬러로 강조
    plt.scatter(t[tp], y_pr[tp], marker='|', s=300, c='green',  label='True Positive')
    plt.scatter(t[fn], y_pr[fn], marker='|', s=300, c='red',    label='False Negative')
    plt.scatter(t[fp], y_pr[fp], marker='|', s=300, c='orange', label='False Positive')

    plt.yticks([0, 1], ['Prediction', 'Ground Truth'])
    plt.xlabel('Time Step')
    plt.xlim(-1, len(gt))
    plt.legend(bbox_to_anchor=(1.02, 1.15), loc='upper left', ncol=3)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
# 2) 3행 히트맵(GT·Pred·Match) — 아주 긴 시계열을 한눈에 보기 편함
# ------------------------------------------------------------------
def plot_heatmap(gt, pred):
    gt   = np.asarray(gt).astype(int)
    pred = np.asarray(pred).astype(int)
    match = (gt == pred).astype(int)     # 1=일치, 0=불일치

    mat = np.vstack([gt, pred, match])   # (3, L)

    plt.figure(figsize=(12, 1.6))
    plt.imshow(mat, aspect='auto', cmap='Greys')
    plt.yticks([0, 1, 2], ['GT', 'Pred', 'Match'])
    plt.xticks([])                       # 시간축 눈금 생략(많으면 너무 빽빽함)
    plt.colorbar(ticks=[0, 1], label='0 / 1')
    plt.title('Binary Sequence Heat-map')
    plt.tight_layout()
    plt.show()

# -------------------- 사용 --------------------
plot_hits(gt, pred)     # 타임라인 방식
plot_heatmap(gt, pred)  # 히트맵 방식
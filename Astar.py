A* 알고리즘

import heapq
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Patch
from matplotlib.widgets import Button

WALL = 1
PATH = 0
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))  # 상, 하, 좌, 우
BLUE, YELLOW, GREEN, RED = (0.2, 0.5, 1), (1, 1, 0), (0, 1, 0), (1, 0, 0)


# 1. 미로 생성
def generate_maze(rows, cols, extra_openings=0.05, seed=None):
    """랜덤 미로를 만들어 (grid, start, goal)을 반환한다."""
    if seed is not None:
        random.seed(seed)

    rows = rows if rows % 2 == 1 else rows + 1
    cols = cols if cols % 2 == 1 else cols + 1

    grid = [[WALL] * cols for _ in range(rows)]

    start = (1, 1)
    grid[start[0]][start[1]] = PATH
    stack = [start]
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and grid[nr][nc] == WALL:
                neighbors.append((nr, nc, dr, dc))
        if neighbors:
            nr, nc, dr, dc = random.choice(neighbors)
            grid[r + dr // 2][c + dc // 2] = PATH
            grid[nr][nc] = PATH
            stack.append((nr, nc))
        else:
            stack.pop()

    candidates = []
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            if grid[r][c] != WALL:
                continue
            horizontal = grid[r][c - 1] == PATH and grid[r][c + 1] == PATH
            vertical = grid[r - 1][c] == PATH and grid[r + 1][c] == PATH
            if horizontal or vertical:
                candidates.append((r, c))
    random.shuffle(candidates)
    for r, c in candidates[: int(len(candidates) * extra_openings)]:
        grid[r][c] = PATH

    goal = (rows - 2, cols - 2)
    return grid, start, goal


# 2. A* (우선순위 큐)
def manhattan(cell, goal):
    """현재 칸에서 도착점까지 남은 거리의 예측값 (행 차이 + 열 차이)."""
    return abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])


def astar(grid, start, goal, heuristic=manhattan):
    """A*로 start에서 goal까지의 최단 경로를 찾는다. (path, explored)를 반환한다.

    우선순위 f = g + h
        g : 시작점에서 그 칸까지 실제로 이동한 거리
        h : 그 칸에서 도착점까지 남은 거리의 예측값
    """
    rows, cols = len(grid), len(grid[0])

    g_score = {start: 0}                   # 시작점에서 각 칸까지 알려진 최소 이동 거리
    parent = {start: None}
    h0 = heuristic(start, goal)
    heap = [(h0, h0, start)]               # (f, h, 칸): f가 같으면 h가 작은(도착점에 가까운) 칸 먼저
    done = [[False] * cols for _ in range(rows)]
    explored = []

    while heap:
        _, _, (r, c) = heapq.heappop(heap)  # f가 가장 작은 칸부터 꺼냄
        if done[r][c]:                     # 이미 확정된 칸이면 건너뜀
            continue
        done[r][c] = True
        explored.append((r, c))

        if (r, c) == goal:
            return reconstruct_path(parent, goal), explored

        for dr, dc in DIRECTIONS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and grid[nr][nc] == PATH and not done[nr][nc]):
                ng = g_score[(r, c)] + 1   # 한 칸 이동 비용 1
                if ng < g_score.get((nr, nc), float("inf")):
                    g_score[(nr, nc)] = ng  # 더 짧은 길을 찾으면 갱신
                    parent[(nr, nc)] = (r, c)
                    h = heuristic((nr, nc), goal)
                    heapq.heappush(heap, (ng + h, h, (nr, nc)))

    return None, explored


def reconstruct_path(parent, goal):
    """도착점에서 부모를 따라 거슬러 올라가 경로를 만든다."""
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


# 3. 탐색 과정 애니메이션
def main(rows, cols):
    plt.rcParams["font.family"] = ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6, 6.6))
    fig.subplots_adjust(top=0.93, bottom=0.17)

    def setup(event=None):
        # 진행 중인 애니메이션 중지 (이미 끝난 애니메이션은 event_source가 None)
        if hasattr(fig, "anim") and fig.anim.event_source is not None:
            fig.anim.event_source.stop()
        grid, s, e = generate_maze(rows, cols)

        # 탐색 시간 측정: 알고리즘 실행 직전과 직후의 시각 차이 (미로 생성, 화면 출력 시간은 제외)
        t0 = time.perf_counter()
        path, explored = astar(grid, s, e)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        path = path or []

        img = np.repeat(np.array(grid, float)[..., None], 3, axis=2)  # 벽=흰색, 길=검정
        img[s], img[e] = GREEN, RED

        # 프레임 = 한 번에 그릴 칸 묶음 (미로가 클수록 한 번에 많이 그림)
        step = max(1, len(explored) // 100)
        pstep = max(1, len(path) // 60)
        frames = [[(c, BLUE) for c in explored[i:i + step]] for i in range(0, len(explored), step)]
        frames += [[(c, YELLOW) for c in path[i:i + pstep]] for i in range(0, len(path), pstep)]

        # 프레임별 누적 탐색 칸 수 (프레임이 중복 호출돼도 숫자가 어긋나지 않게 미리 계산)
        cumulative, total = [], 0
        for f in frames:
            total += sum(color == BLUE for _, color in f)
            cumulative.append(total)

        ax.clear()
        im = ax.imshow(img)
        ax.set(xticks=[], yticks=[], title="A* Search")
        ax.legend(handles=[Patch(color=BLUE, label="탐색한 칸"), Patch(color=YELLOW, label="최종 경로")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2)

        def update(k):
            for cell, color in frames[k]:
                if cell not in (s, e):
                    img[cell] = color
            im.set_data(img)
            title = f"A* Search  |  탐색 {cumulative[k]}칸"
            if k == len(frames) - 1:
                title += (f"  |  경로 {len(path) - 1}칸  |  {elapsed_ms:.2f}ms"
                          if path else "  |  경로 없음")
            ax.set_title(title)

        fig.anim = FuncAnimation(fig, update, frames=len(frames), interval=30, repeat=False)
        fig.canvas.draw_idle()

    fig.btn = Button(fig.add_axes([0.35, 0.03, 0.3, 0.07]), "다시 생성하기")
    fig.btn.on_clicked(setup)
    setup()
    plt.show()


if __name__ == "__main__":
    r, c = map(int, input("미로의 가로와 세로의 크기를 입력하세요: ").split())
    main(r, c)

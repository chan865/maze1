"""
- 0 = 길, 1 = 벽
- 알고리즘: 랜덤 DFS(재귀 백트래킹)로 미로를 만든 뒤,
  벽 일부를 추가로 뚫어서 길이 여러 갈래가 되게 함
  (완전 미로는 길이 하나뿐이라 A*와 다익스트라 차이가 안 보이기 떄문)
"""
import random

WALL, PATH = 1, 0

def generate_maze(rows, cols, extra_openings=0.05, seed=None):
    """랜덤 미로를 만들어 (grid, start, goal)을 반환한다.
    
    row, cols : 격자 크기 (짝수를 넣으면 홀수로 보정됨)
    extra_openings : 완성된 미로에서 추가로 뚫을 벽의 비율 (0이면 길이 하나뿐인 미로)
    seed: 같은 미로를 다시 보고 싶은 때만 지정 (None이면 매번 다름)
    """
    if seed is not None:
        random.seed(seed)

    #홀수 크기로 보정 (벽-길-벽 구조를 만들기 위해)
    rows = rows if rows %2 == 1 else rows + 1
    cols = cols if cols %2 == 1 else cols + 1

    grid = [[WALL] * cols for _ in range(rows)]

    #반복형 DFS(재귀 한도 문제 방지)
    start = (1,1)
    grid[start[0]][start[1]] = PATH
    stack = [start]
    while stack:
        r, c = stack[-1]
        #두 칸 떨어진 이동 후보 중 아직 벽인 곳
        neighbors = []
        for dr, dc in ((-2,0), (2,0), (0,-2), (0,2)):
            nr, nc = r +dr, c+ dc
            if 0 < nr < rows - 1 and 0 <nc < cols - 1 and grid[nr][nc] == WALL:
                neighbors.append((nr,nc,dr,dc))
        if neighbors:
            nr, nc, dr, dc = random.choice(neighbors)
            grid[r + dr//2][c+ dc//2] = PATH #사이 벽 없애기
            grid[nr][nc] = PATH
            stack.append((nr,nc))
        else:
            stack.pop()

    #벽 일부를 추가로 뚫어 우회로 만들기
    candidates = []
    for r in range(1, rows -1):
        for c in range(1, cols -1):
            if grid[r][c] != WALL:
                continue
                #좌우가 길이거나 상하가 길인 벽만 
            horizontal = grid[r][c-1] == PATH and grid[r][c+1] == PATH
            vertical = grid[r-1][c] == PATH and grid[r+1][c] == PATH
            if horizontal or vertical:
                candidates.append((r,c))
    random.shuffle(candidates)
    for r, c in candidates[: int(len(candidates) * extra_openings)]:
        grid[r][c] = PATH

    goal = (rows - 2, cols - 2)
    return grid, start, goal

def show_maze(rows, cols, extra_openings=0.05, save_path=None):
    """matplotlib으로 미로 그리기
    
    - '다시 생성하기' : 새 미로 생성
    - '숫자 보기' : 칸마다 길은 0, 벽은 1로 표시
    save_path를 주면 버튼 없이 이미지로 저장
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.widgets import Button

    plt.rcParams["font.family"] = ["Malgun Gothic"]
    plt.rcParams["axes.unicode_minus"] = False

    grid, start, goal = generate_maze(rows, cols, extra_openings)

    fig, ax = plt.subplots(figsize=(6, 6.6))
    fig.subplots_adjust(top=0.93, bottom=0.17)
    im = ax.imshow(grid, cmap=ListedColormap(["white", "black"]), interpolation="nearest")
    # zorder를 높여 시작/도착 표시가 숫자 위에 보이게 함
    ax.scatter(start[1], start[0], c="lime", s=60, marker="s", label="start", zorder=5)
    ax.scatter(goal[1], goal[0], c="red", s=60, marker="s", label="goal", zorder=5)
    ax.set_xticks([])
    ax.set_yticks([]) 
    ax.set_title("Random Maze")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=120)
        plt.close(fig)
        return

    #현재 상태 (버튼 콜백에서 공유)
    state = {"grid": grid, "show_numbers": False, "texts": []}

    def refresh_numbers():
        for t in state["texts"]:
            t.remove()
        state["texts"] = []

        #숫자 모드에서는 흰색/검정색 미로 그림을 숨김
        im.set_visible(not state["show_numbers"])

        if state["show_numbers"]:
            g = state["grid"]
            n_rows, n_cols = len(g), len(g[0])
            fontsize = max(3, 250 / max(n_rows, n_cols)) #미로가 커질수록 글자 작게
            for r in range(n_rows):
                for c in range(n_cols):
                    t = ax.text(c, r, str(g[r][c]), ha="center", va="center", fontsize=fontsize, color="black", zorder=3)
                    state["texts"].append(t)
        fig.canvas.draw_idle()

    # 버튼 두 개를 나란히 배치
    regen_ax = fig.add_axes([0.15, 0.03, 0.3, 0.07])
    num_ax = fig.add_axes([0.55, 0.03, 0.3, 0.07])
    regen_button = Button(regen_ax, "다시 생성하기")
    num_button = Button(num_ax, "숫자 보기")

    def regenerate(event):
        new_grid, _, _ = generate_maze(rows, cols, extra_openings)
        state["grid"] = new_grid
        im.set_data(new_grid)
        refresh_numbers() #숫자 모드가 켜져 있으면 새 미로에도 숫자 표시

    def toggle_numbers(event):
        state["show_numbers"] = not state["show_numbers"]
        num_button.label.set_text("그림 보기" if state["show_numbers"] else "숫자 보기")
        refresh_numbers()

    regen_button.on_clicked(regenerate)
    num_button.on_clicked(toggle_numbers)
    fig._buttons = (regen_button, num_button)
    plt.show()

if __name__ == "__main__":
    rows, cols = map(int, input("미로의 가로와 세로의 크기를 입력하세요: ").split())
    show_maze(rows, cols, extra_openings=0.05)
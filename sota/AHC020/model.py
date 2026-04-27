# --OPTION--
import math


class Model:
    """Greedy seed solver for AHC020 Broadcasting.

    Turns ON every edge so all stations are reachable from station 1, then
    sets each station's output strength P_i to the smallest integer that
    covers every resident assigned to it (each resident is assigned to its
    nearest station). Valid but very far from Pareto-optimal — meant as a
    starting point for evolution.
    """

    def solve(self, input_str: str) -> str:
        lines = input_str.strip().split("\n")
        idx = 0

        N, M, K = map(int, lines[idx].split())
        idx += 1

        coords = []
        for _ in range(N):
            x, y = map(int, lines[idx].split())
            coords.append((x, y))
            idx += 1

        edges = []
        for _ in range(M):
            u, v, w = map(int, lines[idx].split())
            edges.append((u, v, w))
            idx += 1

        residents = []
        for _ in range(K):
            a, b = map(int, lines[idx].split())
            residents.append((a, b))
            idx += 1

        P = [0] * N
        for (a, b) in residents:
            best_i = 0
            best_d2 = None
            for i in range(N):
                x, y = coords[i]
                d2 = (a - x) * (a - x) + (b - y) * (b - y)
                if best_d2 is None or d2 < best_d2:
                    best_d2 = d2
                    best_i = i
            need = math.ceil(math.sqrt(best_d2))
            if need > P[best_i]:
                P[best_i] = need

        P = [min(p, 5000) for p in P]
        B = [1] * M

        return " ".join(map(str, P)) + "\n" + " ".join(map(str, B))

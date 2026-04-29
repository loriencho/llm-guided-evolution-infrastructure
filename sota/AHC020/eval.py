import argparse
import importlib
import os
import random
import sys
import time
from pathlib import Path as p
from os.path import join as pj

import numpy as np


T_MAX = 60.0
DEFAULT_REPS = 3


def instance_score(n_covered, K, S):
    """Official AHC020 per-instance score (maximization).

    If n < K:  round(10^6 * (n+1) / K)
    If n == K: round(10^6 * (1 + 10^8 / (S + 10^7)))
    """
    if n_covered < K:
        return round(1_000_000 * (n_covered + 1) / K)
    return round(1_000_000 * (1 + 1e8 / (S + 1e7)))


def create_save_dir(save_root):
    if not p(save_root).exists():
        p(save_root).mkdir(exist_ok=True, parents=True)

    n = []
    for exp_dir in p(save_root).iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            i = -1
            while exp_name[i].isdigit():
                i -= 1
            i += 1
            if i != 0:
                n.append(int(exp_name[i:]))

    if len(n) == 0:
        save_dir = pj(save_root, "exp1")
    else:
        save_dir = f"{save_root}/exp{sorted(n)[-1] + 1}"

    return save_dir


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model")
    parser.add_argument('--save_dir', type=str, default="trained")
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--variant_dir', type=str, default='models')
    parser.add_argument('--test_cases_dir', type=str, default='test_cases')
    parser.add_argument('--reps', type=int, default=DEFAULT_REPS)
    parser.add_argument('--t_max', type=float, default=T_MAX)
    return parser.parse_args()


def parse_input(text):
    lines = text.strip().split("\n")
    idx = 0
    N, M, K = map(int, lines[idx].split()); idx += 1
    coords = []
    for _ in range(N):
        x, y = map(int, lines[idx].split()); idx += 1
        coords.append((x, y))
    edges = []
    for _ in range(M):
        u, v, w = map(int, lines[idx].split()); idx += 1
        edges.append((u, v, w))
    residents = []
    for _ in range(K):
        a, b = map(int, lines[idx].split()); idx += 1
        residents.append((a, b))
    return N, M, K, coords, edges, residents


def parse_output(text, N, M):
    if text is None:
        return None
    try:
        lines = [ln for ln in text.strip().split("\n") if ln.strip() != ""]
        if len(lines) < 2:
            return None
        P = list(map(int, lines[0].split()))
        B = list(map(int, lines[1].split()))
        if len(P) != N or len(B) != M:
            return None
        if any(pi < 0 or pi > 5000 for pi in P):
            return None
        if any(bj not in (0, 1) for bj in B):
            return None
        return P, B
    except (ValueError, IndexError):
        return None


def score_instance(P, B, N, M, K, coords, edges, residents):
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cable_cost = 0
    for j, (u, v, w) in enumerate(edges):
        if B[j] == 1:
            cable_cost += w
            union(u - 1, v - 1)

    root1 = find(0)
    component = [i for i in range(N) if find(i) == root1]

    power_cost = sum(pi * pi for pi in P)

    n_covered = 0
    for (a, b) in residents:
        for i in component:
            x, y = coords[i]
            pi = P[i]
            if pi == 0:
                continue
            dx = a - x
            dy = b - y
            if dx * dx + dy * dy <= pi * pi:
                n_covered += 1
                break

    S = cable_cost + power_cost
    return n_covered, S


def evaluate_one(model_module, input_text, reps, t_max):
    N, M, K, coords, edges, residents = parse_input(input_text)

    best_time = float('inf')
    last_output = None

    for _ in range(reps):
        model = model_module.Model()
        t0 = time.perf_counter()
        try:
            out = model.solve(input_text)
        except Exception as exc:
            print(f"  solve() raised: {exc}", file=sys.stderr)
            out = None
        elapsed = time.perf_counter() - t0

        if elapsed > t_max:
            return instance_score(0, K, 0), t_max
        if elapsed < best_time:
            best_time = elapsed
        last_output = out

    parsed = parse_output(last_output, N, M)
    if parsed is None:
        return instance_score(0, K, 0), best_time

    P, B = parsed
    n_covered, S = score_instance(P, B, N, M, K, coords, edges, residents)
    return instance_score(n_covered, K, S), best_time


if __name__ == '__main__':
    script_directory = p(__file__).parent.resolve()
    os.chdir(script_directory)

    args = get_args()
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    sys.path.append(args.variant_dir)
    model_module = importlib.import_module(args.model)

    try:
        gene_id = args.model.split('model_')[1]
    except Exception:
        gene_id = 'seed'

    save_dir = f'{args.save_dir}/{gene_id}'
    create_save_dir(save_dir)

    test_files = sorted(p(args.test_cases_dir).glob('*.txt'))
    if not test_files:
        print(f"WARNING: no test cases found in {args.test_cases_dir}", file=sys.stderr)

    score_total = 0
    T_total = 0.0

    for tf in test_files:
        with open(tf, 'r') as f:
            input_text = f.read()
        score_inst, t_inst = evaluate_one(model_module, input_text, args.reps, args.t_max)
        print(f"  {tf.name}: score={score_inst} T={t_inst:.4f}s")
        score_total += score_inst
        T_total += t_inst

    results_text = f"{score_total},{T_total}"

    filename = os.path.abspath(f'results/{gene_id}_results.txt')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as fh:
        fh.write(results_text)

    print(f"results have been written to {filename}")
    print('=' * 120); print('job done'); print('=' * 120)

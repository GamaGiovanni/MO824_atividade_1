#!/usr/bin/env python3
# gen_max_sc_qbf_instances.py
# Gera 15 instâncias para MAX-SC-QBF combinando n∈{25,50,100,200,400} e 3 padrões dos S_i.
# Saída no formato exigido: n; |S1|..|Sn|; linhas com elementos de cada S_i (1-based);
# em seguida A (triangular superior): a11..a1n; a22..a2n; ...; ann.

import argparse
import os
import random
from typing import List, Set, Tuple

def _avg_size(n: int) -> int:
    """Tamanho médio alvo de |S_i| ~ sqrt(n)."""
    return max(2, int(round(n ** 0.5)))

def pattern_uniform(n: int, rng: random.Random) -> List[Set[int]]:
    """Cada elemento entra em S_i independentemente com p≈sqrt(n)/n."""
    avg = _avg_size(n)
    p = avg / n
    S = []
    for _ in range(n):
        s = {k for k in range(1, n+1) if rng.random() < p}
        S.append(s)
    return S

def pattern_interval(n: int, rng: random.Random) -> List[Set[int]]:
    """S_i é intervalo circular centrado em i, comprimento ~ sqrt(n) (±50%)."""
    avg = _avg_size(n)
    S = []
    for i in range(1, n+1):
        L = max(1, int(round(avg * (0.5 + rng.random()))))  # [0.5*avg, 1.5*avg]
        elems = [((i - 1 + t) % n) + 1 for t in range(L)]
        S.append(set(elems))
    return S

def pattern_pareto(n: int, rng: random.Random) -> List[Set[int]]:
    """Tamanhos |S_i| com cauda pesada (Pareto α=2), truncado em [1, 3*avg]."""
    avg = _avg_size(n)
    S = []
    for _ in range(n):
        val = rng.paretovariate(2.0)  # >=1
        size = max(1, min(3 * avg, int(round(avg * min(val, 3.0)))))
        if size >= n:
            s = set(range(1, n+1))
        else:
            s = set(rng.sample(range(1, n+1), size))
        S.append(s)
    return S

def enforce_coverage(S: List[Set[int]], n: int, rng: random.Random) -> None:
    """Garante que cada k∈{1..n} esteja em algum S_i; se faltar, injeta no conjunto menor."""
    covered = [0] * (n + 1)
    for s in S:
        for k in s:
            covered[k] += 1
    missing = [k for k in range(1, n+1) if covered[k] == 0]
    for k in missing:
        idx = min(range(len(S)), key=lambda i: len(S[i]))
        S[idx].add(k)

def nonempty_fix(S: List[Set[int]], n: int, rng: random.Random) -> None:
    """Evita S_i vazio."""
    for i in range(len(S)):
        if not S[i]:
            S[i].add(rng.randrange(1, n+1))

def gen_A_triangular(n: int, rng: random.Random, density: float = 0.2,
                     diag_range: Tuple[int, int] = (-10, 10),
                     off_range: Tuple[int, int] = (-5, 5)) -> List[List[int]]:
    """
    Gera A triangular superior como lista de linhas rows[i] = [a_ii, a_i,i+1, ..., a_in].
    - 'density' é a prob. de a_ij!=0 (i<j).
    - valores inteiros com sinais mistos (evita trivialidade).
    """
    rows: List[List[int]] = []
    for i in range(n):
        row = []
        aii = rng.randint(diag_range[0], diag_range[1])
        if aii == 0:
            aii = 1 if rng.random() < 0.5 else -1
        row.append(aii)
        for _j in range(i+1, n):
            if rng.random() < density:
                aij = rng.randint(off_range[0], off_range[1])
                if aij == 0:
                    aij = 1 if rng.random() < 0.5 else -1
            else:
                aij = 0
            row.append(aij)
        rows.append(row)
    return rows

def write_instance(path: str, n: int, S: List[Set[int]], A_rows: List[List[int]]) -> None:
    """Escreve o arquivo no formato especificado."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{n}\n")
        f.write(" ".join(str(len(s)) for s in S) + "\n")
        for s in S:
            f.write((" ".join(str(k) for k in sorted(s))) + "\n")
        for i in range(n):
            f.write(" ".join(str(v) for v in A_rows[i]) + "\n")

def build_sets(n: int, pattern: str, rng: random.Random) -> List[Set[int]]:
    if pattern == "uniform":
        S = pattern_uniform(n, rng)
    elif pattern == "interval":
        S = pattern_interval(n, rng)
    elif pattern == "pareto":
        S = pattern_pareto(n, rng)
    else:
        raise ValueError(f"Padrão desconhecido: {pattern}")
    enforce_coverage(S, n, rng)
    nonempty_fix(S, n, rng)
    return S

def file_name(n: int, pattern: str, seed: int) -> str:
    return f"maxscqbf_n{n}_{pattern}_seed{seed}.txt"

def main():
    parser = argparse.ArgumentParser(description="Gerador de instâncias MAX-SC-QBF (triangular superior)")
    parser.add_argument("-n", "--n", type=int, help="n de variáveis (e conjuntos)")
    parser.add_argument("-p", "--pattern", choices=["uniform", "interval", "pareto"],
                        help="padrão de geração dos conjuntos S_i")
    parser.add_argument("--rho", type=float, default=0.20, help="densidade dos termos fora da diagonal de A (0..1)")
    parser.add_argument("--seed", type=int, default=None, help="semente RNG (int)")
    parser.add_argument("-o", "--output", type=str, default=None, help="arquivo de saída (.txt)")
    parser.add_argument("--all", action="store_true", help="gerar as 15 instâncias padrão no diretório out/")
    parser.add_argument("--outdir", type=str, default="out", help="diretório para --all (default: out)")
    args = parser.parse_args()

    if args.all:
        os.makedirs(args.outdir, exist_ok=True)
        Ns = [25, 50, 100, 200, 400]
        patterns = ["uniform", "interval", "pareto"]
        base_seed = 20250819 if args.seed is None else args.seed
        generated = []
        for n in Ns:
            for pi, pat in enumerate(patterns):
                seed = base_seed + 1000 * pi + n
                rng = random.Random(seed)
                S = build_sets(n, pat, rng)
                A_rows = gen_A_triangular(n, rng, density=args.rho)
                path = os.path.join(args.outdir, file_name(n, pat, seed))
                write_instance(path, n, S, A_rows)
                generated.append(path)
        print("Geradas as seguintes instâncias:")
        for p in generated:
            print(" -", p)
        return

    if args.n is None or args.pattern is None:
        parser.error("Para gerar uma instância única, informe -n e -p, ou use --all.")

    n = args.n
    seed = args.seed if args.seed is not None else (20250819 + hash(args.pattern) % 1000 + n)
    rng = random.Random(seed)
    S = build_sets(n, args.pattern, rng)
    A_rows = gen_A_triangular(n, rng, density=args.rho)
    out_path = args.output or file_name(n, args.pattern, seed)
    write_instance(out_path, n, S, A_rows)
    print(f"Instância gerada em: {out_path}")

if __name__ == "__main__":
    main()

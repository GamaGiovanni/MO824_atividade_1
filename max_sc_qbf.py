# Uso:
#   python max_sc_qbf.py caminho/instancia.txt --timelimit 600 --mipgap 0.0

import argparse
from gurobipy import Model, GRB, quicksum
import sys

def parse_instance(path):
    """
    Formato:
      <n>
      <s1> <s2> ... <sn>
      <lista de elementos cobertos por S1>
      ...
      <lista de elementos cobertos por Sn>
      <a11> <a12> ... <a1n>
      <a22> ... <a2n>
      ...
      <ann>
    A matriz A é triangular superior.
    """
    def token_stream(lines):
        for line in lines:
            line = line.strip()
            if not line:
                continue
            for tok in line.split():
                yield tok

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    toks = token_stream(lines)

    try:
        n = int(next(toks))
    except StopIteration:
        raise ValueError("Arquivo vazio ou n ausente.")

    # tamanhos si
    s = []
    for i in range(n):
        try:
            s.append(int(next(toks)))
        except StopIteration:
            raise ValueError("Linha de tamanhos s_i incompleta.")

    # subsets
    subsets = [set() for i in range(n)]
    for i in range(n):
        need = s[i]
        for j in range(need):
            try:
                k = int(next(toks))
            except StopIteration:
                raise ValueError(f"Lista de cobertura de S{i+1} incompleta.")
            if not (1 <= k <= n):
                raise ValueError(f"Elemento {k} fora do universo 1..{n}.")
            subsets[i].add(k-1)  # internamente 0-based

    # Matrix A
    needed = n * (n + 1) // 2
    coeffs = []
    for i in range(needed):
        try:
            coeffs.append(float(next(toks)))
        except StopIteration:
            raise ValueError("Matriz A (triangular superior) incompleta.")

    # Dicionário {(i,j): aij} com i<=j
    A = {}
    idx = 0
    for i in range(n):
        for j in range(i, n):
            aij = coeffs[idx]
            idx += 1
            if aij != 0.0:
                A[(i, j)] = aij

    return n, subsets, A

def build_and_solve(n, subsets, A, timelimit=600, mipgap=None, verbose=True):
    """
    Constrói e resolve o MILP.
    Retorna um dicionário com solução e estatísticas.
    """
    m = Model("MAX_SC_QBF")
    if not verbose:
        m.Params.OutputFlag = 0
    if timelimit is not None:
        m.Params.TimeLimit = float(timelimit)
    if mipgap is not None:
        m.Params.MIPGap = float(mipgap)

    # x_i binárias
    x = m.addVars(n, vtype=GRB.BINARY, name="x")

    # y_ij (para i<=j com a_ij != 0). Contínuas [0,1] bastam por McCormick
    y = {}
    for (i, j), aij in A.items():
        y[(i, j)] = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name=f"y[{i},{j}]")

    m.update()

    # Cobertura: para cada elemento k, soma dos x_i que o cobrem >= 1
    for k in range(n):
        coverers = [i for i in range(n) if k in subsets[i]]
        m.addConstr(quicksum(x[i] for i in coverers) >= 1, name=f"cover[{k}]")

    # Linearização:
    for (i, j) in y:
        if i == j:
            # y_ii = x_i
            m.addConstr(y[(i, i)] == x[i], name=f"diag[{i}]")
        else:
            m.addConstr(y[(i, j)] <= x[i], name=f"mc1[{i},{j}]")
            m.addConstr(y[(i, j)] <= x[j], name=f"mc2[{i},{j}]")
            m.addConstr(y[(i, j)] >= x[i] + x[j] - 1, name=f"mc3[{i},{j}]")

    # Objetivo: maximizar sum_{i<=j} a_ij * y_ij
    m.setObjective(quicksum(aij * y[(i, j)] for (i, j), aij in A.items()), GRB.MAXIMIZE)

    m.optimize()

    result = {
        "status": m.Status,
        "runtime": m.Runtime,
        "obj_val": None,
        "best_bound": None,
        "mip_gap": None,
        "x": None,
        "selected_sets": None,
        "n_selected": None
    }

    # Preencher resultados (mesmo se não ótimo)
    try:
        result["best_bound"] = m.ObjBound
    except Exception:
        pass
    try:
        result["mip_gap"] = m.MIPGap
    except Exception:
        pass

    if m.SolCount > 0:
        result["obj_val"] = m.ObjVal
        xv = [int(round(x[i].X)) for i in range(n)]
        result["x"] = xv
        result["selected_sets"] = [i for i in range(n) if xv[i] == 1]
        result["n_selected"] = sum(xv)

    return result

def main():
    ap = argparse.ArgumentParser(description="Solver MILP para MAX-SC-QBF com Gurobi")
    ap.add_argument("instance", help="Caminho do arquivo de instância")
    ap.add_argument("--timelimit", type=float, default=600.0, help="Limite de tempo (s)")
    ap.add_argument("--mipgap", type=float, default=None, help="MIP gap alvo (ex.: 0.01 = 1%)")
    ap.add_argument("--quiet", action="store_true", help="Silenciar log do Gurobi")
    args = ap.parse_args()

    try:
        n, subsets, A = parse_instance(args.instance)
    except Exception as e:
        print(f"Erro ao ler a instância: {e}", file=sys.stderr)
        sys.exit(2)

    res = build_and_solve(n, subsets, A, timelimit=args.timelimit, mipgap=args.mipgap, verbose=not args.quiet)

    print("=== Resultado ===")
    print(f"Status: {res['status']}")
    print(f"Tempo (s): {res['runtime']:.2f}")
    print(f"Melhor limite (bound): {res['best_bound']}")
    print(f"Gap: {res['mip_gap']}")
    if res["obj_val"] is not None:
        print(f"Valor da solução: {res['obj_val']}")
        print(f"Selecionados ({res['n_selected']}): {res['selected_sets']}")
    else:
        print("Nenhuma solução incumbente encontrada.")

if __name__ == "__main__":
    main()

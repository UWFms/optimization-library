import cvxpy as cp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pulp import LpMaximize, LpProblem, LpVariable, lpSum, value, LpMinimize, PULP_CBC_CMD
from scipy.optimize import linprog, minimize


def output_task(obj, constraints, rhs, is_maximization, console_print=False):
    optimization_type = "Maximize" if is_maximization else "Minimize"

    objective_terms = []
    for i, coeff in enumerate(obj):
        term = f"{coeff}*x{i + 1}"
        objective_terms.append(term)
    objective_str = " + ".join(objective_terms)

    if console_print:
        print(f"{optimization_type} z = {objective_str}")
        print("subject to the constraints:")

    for i, constraint in enumerate(constraints):
        constraint_terms = []
        for j, coeff in enumerate(constraint):
            term = f"{coeff}*x{j + 1}"
            constraint_terms.append(term)
        constraint_str = " + ".join(constraint_terms)

        if console_print:
            print(f"{constraint_str} <= {rhs[i]}")


def simplex(obj, constraints, rhs, is_maximization, console_print=False):
    n = len(obj)
    m = len(constraints)

    iteration = 0

    if not is_maximization:
        obj = [-x for x in obj]

    table = [[0 for _ in range(n + 1 + m)] for _ in range(m + 1)]

    for i in range(n):
        table[0][i] = -obj[i]

    for i in range(1, m + 1):
        for j in range(n):
            if j < len(constraints[i - 1]):
                table[i][j] = constraints[i - 1][j]

    table[0][-1] = 0
    for i in range(1, m + 1):
        table[i][-1] = rhs[i - 1]

    for i in range(1, m + 1):
        for j in range(n, n + m):
            if j - n == i - 1:
                table[i][j] = 1

    basis = [n + i for i in range(m)]

    if console_print:
        print("Initial Simplex Tableau:")
        print(table)

    z_value = 0
    z_value_list = []

    while any(round(table[0][i], 8) < 0 for i in range(n + m)):
        key_col = min(range(n + m), key=lambda i: table[0][i])
        if key_col == -1:
            print("No valid pivot column found. The method is not applicable!")

        if all(table[i][key_col] <= 0 for i in range(1, m + 1)):
            return -1, -1, [], [], -1, []
        key_row = min(
            (i for i in range(1, m + 1) if table[i][key_col] > 0),
            key=lambda i: table[i][-1] / table[i][key_col]
        )

        if key_row == -1:
            print("Unbounded solution. The method is not applicable!")

        pivot = table[key_row][key_col]
        for j in range(n + m + 1):
            table[key_row][j] /= pivot

        for i in range(m + 1):
            if i != key_row:
                ratio = table[i][key_col]
                for j in range(n + m + 1):
                    table[i][j] -= ratio * table[key_row][j]

        basis[key_row - 1] = key_col
        z_value = table[0][-1]
        z_value_list.append(z_value)

        if console_print:
            print("Updated Simplex Tableau:")
            print(table, key_row, key_col)

        iteration += 1

    answers = [0] * n
    for i in range(m):
        if basis[i] < n:
            answers[basis[i]] = table[i + 1][-1]

    return z_value, answers, basis, table, iteration, z_value_list


def relaxation_method(obj, constraints, rhs, is_maximization, alpha=0.1,
                      epsilon=1e-6, max_iter=1000, console_print=False):
    A = np.array(constraints, dtype=float)
    b = np.array(rhs, dtype=float)
    c = np.array(obj, dtype=float)

    x = np.ones(len(c)) / len(c)  # разумное начальное приближение
    z_value_list = []
    x_value_list = []

    for iteration in range(max_iter):
        grad = -c if is_maximization else c
        x_prev = x.copy()

        # шаг градиента
        x_new = x + alpha * grad

        # проекция на допустимое множество Ax ≤ b, x ≥ 0
        x = project_onto_feasible_set(x_new, A, b)
        x = np.maximum(x, 0)

        z = np.dot(c, x)
        z_value_list.append(-z) if is_maximization else z_value_list.append(z)
        x_value_list.append(x)

        if console_print and iteration < 20:
            print(f"Iter {iteration}: z = {z:.6f}, x = {x}")

        if np.linalg.norm(x - x_prev) < epsilon:
            if console_print:
                print(f"Converged at iteration {iteration}")
            break

    return np.dot(c, x), x.tolist(), iteration, z_value_list, x_value_list


def solve_lp_pulp(c, A, b, is_maximization=True):
    num_vars = len(c)
    num_constraints = len(b)

    prob = LpProblem("Main_Problem", LpMaximize if is_maximization else LpMinimize)
    x_vars = [LpVariable(f"x_{j}", lowBound=0) for j in range(num_vars)]

    for i in range(num_constraints):
        prob += lpSum(A[i][j] * x_vars[j] for j in range(num_vars)) <= b[i], f"constraint_{i}"

    prob += lpSum(c[j] * x_vars[j] for j in range(num_vars))

    solver = PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    x_values = [value(x_vars[j]) if value(x_vars[j]) is not None else 0 for j in range(num_vars)]
    z_value = value(prob.objective) if value(prob.objective) is not None else 0

    return z_value, x_values


def column_generation(c, A, b, is_maximization=True, console_print=False):
    num_vars = len(c)
    num_constraints = len(b)

    I = [0]

    iteration = 0
    z_value_list = []
    x_value_list = []

    while True:
        # 1. Решаем ограниченную задачу через pulp
        sub_c = [c[j] for j in I]
        sub_A = [[A[i][j] for j in I] for i in range(num_constraints)]

        z_value, x_sub = solve_lp_pulp(sub_c, sub_A, b, is_maximization)
        print(f"{z_value=} {x_sub=}")
        z_value_list.append(z_value)
        x_value_list.append(x_sub)

        # 2. Формируем dual-задачу и решаем её
        prob = LpProblem("Dual", LpMaximize if is_maximization else LpMinimize)
        duals = [LpVariable(f"pi_{i}") for i in range(num_constraints)]

        for idx, j in enumerate(I):
            col = [A[i][j] for i in range(num_constraints)]
            prob += lpSum(duals[i] * col[i] for i in range(num_constraints)) == c[j], f"dual_constr_{idx}"

        obj_func = lpSum(duals[i] * b[i] for i in range(num_constraints))
        prob.setObjective(obj_func)

        solver = PULP_CBC_CMD(msg=False)
        prob.solve(solver)

        pi = [duals[i].varValue if duals[i].varValue is not None else 0 for i in range(num_constraints)]

        # 3. Reduced cost
        entering_candidates = []
        for j in range(num_vars):
            if j in I:
                continue
            a_j = [A[i][j] for i in range(num_constraints)]
            rc = c[j] - sum(pi[i] * a_j[i] for i in range(num_constraints))
            if console_print:
                print(f"x_{j}: reduced cost = {rc} pi: {pi}")

            if (is_maximization and rc > 1e-5) or (not is_maximization and rc < -1e-5):
                entering_candidates.append((j, rc))

        if not entering_candidates:
            # Конструируем итоговое решение
            x_full = [0] * num_vars
            for idx, j in enumerate(I):
                x_full[j] = x_sub[idx]
            return z_value, x_full, iteration, z_value_list, x_value_list

        # Добавим переменную с наибольшим улучшением
        j_min, rc_min = min(entering_candidates, key=lambda x: x[1])
        if console_print:
            print(f"Добавляем переменную x_{j_min} с reduced cost {rc_min:.4f}")
        I.append(j_min)

        iteration += 1


def admm_lp(c, A, b, epsi=1e-4, rho=1.0, max_iter=1000, console_print=False):
    n = len(c)
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    A = np.array(A)
    b = np.array(b)
    c = np.array(c)

    z_value_list = []
    x_value_list = []

    for iteration in range(max_iter):
        # Сохраняем предыдущее значение z перед обновлением
        z_old = z.copy()  # <-- Добавлено определение z_old

        # x-update: min c^T x + (rho/2) ||x - z + u||^2, x >= 0
        x = np.maximum(z - u - (c / rho), 0)  # аналитическое решение

        # z-update: min (rho/2) ||x - z + u||^2, A z <= b
        z_var = cp.Variable(n)
        obj = cp.Minimize(cp.sum_squares(z_var - (x + u)))
        constraints = [A @ z_var <= b]
        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)
        z = z_var.value

        # u-update
        u += x - z

        # Проверка сходимости
        r = np.linalg.norm(x - z)
        s = rho * np.linalg.norm(z - z_old)

        if console_print and iteration < 10:
            print(f"Iter {iteration}: x = {x}, z = {z}, u = {u}, r = {r:.3f}, s = {s:.3f}")

        z_value_list.append(-(c @ x))
        x_value_list.append(x)

        if r < epsi and s < epsi:
            break

    obj_val = c @ x
    return obj_val, x.tolist(), iteration + 1, z_value_list, x_value_list


def project_onto_feasible_set(y, A, b):
    n = len(y)

    def objective(x):
        return 0.5 * np.sum((x - y) ** 2)

    constraints = []
    if A is not None and b is not None:
        constraints.append({'type': 'ineq', 'fun': lambda x: b - A @ x})

    bounds = [(0, None)] * n

    result = minimize(objective, y, bounds=bounds, constraints=constraints)

    if not result.success:
        print("⚠️ Проекция не удалась, возвращаю нули.")
        return np.zeros_like(y)

    return result.x


def mirror_descent(c, A, b, epsi=1e-5, is_maximization=True, alpha=0.1, max_iter=1000, console_print=False):
    result_iteration = max_iter
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float) if A is not None else None
    b = np.array(b, dtype=float) if b is not None else None

    n = len(c)
    z_value_list = []
    x_value_list = []

    x = np.ones(n) / n  # Старт: единичный вектор

    for iteration in range(max_iter):
        grad = c.copy()
        x_prev = x.copy()

        # Шаг градиентного спуска
        y = x - alpha * grad

        # Проекция на допустимое множество: x >= 0, Ax <= b
        x = project_onto_feasible_set(y, A, b)
        x = np.maximum(x, 0)

        diff = np.linalg.norm(x - x_prev)
        if console_print and iteration < 10:
            print(f"Iter {iteration}, diff = {diff:.4e}, x = {x}")

        z_value_list.append(np.dot(-c, x)) if is_maximization else z_value_list.append(np.dot(c, x))
        x_value_list.append(x)

        if diff < epsi:
            result_iteration = iteration
            break

    obj_val = np.dot(c, x)
    return obj_val, x.tolist(), result_iteration, z_value_list, x_value_list


def solve_lp(method, c, A, b, epsi, is_maximization, console_print=True):
    print()
    output_task(c, A, b, is_maximization)
    print()
    if method == 'simplex':
        z, x, _, _, iteration, z_value_list = simplex(c, A, b, is_maximization, console_print=console_print)
        x_value_list = []
        name = 'Симплекс метод'

    elif method == 'highs':
        result = linprog(c if not is_maximization else [-x for x in c],
                         A_ub=A, b_ub=b, method="highs"
                         )
        if result.success:
            z, x, iteration = result.fun, result.x, None
        else:
            z, x, iteration = None, None, None
        z = -z if is_maximization else z
        z_value_list = []
        x_value_list = []
        name = 'Scipy highs'

    elif method == 'relaxation':
        z, x, iteration, z_value_list, x_value_list = relaxation_method(c if not is_maximization else [-x for x in c],
                                                                        A, b, is_maximization, alpha=0.02, epsilon=epsi, console_print=console_print)
        z = -z if is_maximization else z
        name = 'Метод релаксации'

    elif method == 'column-generation':
        z, x, iteration, z_value_list, x_value_list = column_generation(c, A, b, is_maximization, console_print=console_print)
        name = 'Метод генерации столбцов'

    elif method == 'ADMM':
        z, x, iteration, z_value_list, x_value_list = admm_lp(c if not is_maximization else [-x for x in c], A, b, epsi, console_print=console_print)
        z = -z if is_maximization else z
        name = 'Метод ADMM'

    elif method == 'mirror-descent':
        z, x, iteration, z_value_list, x_value_list = mirror_descent(c if not is_maximization else [-x for x in c], A, b, epsi,
                                                                     is_maximization, console_print=console_print)
        z = -z if is_maximization else z
        name = 'Mirror Descent'

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    if console_print:
        print(f"\n=== {name} ===")
        if z is not None:
            print(f"Максимальное значение функции: {z}") if is_maximization \
                else print(f"Минимальное значение функции: {z}")
        print(f"Значения x: {x}")

    result = linprog(c if not is_maximization else [-x for x in c],
                     A_ub=A, b_ub=b, method="highs"
                     )
    true_z = -result.fun if is_maximization else result.fun

    return {
        'method': method,
        'name': name,
        'iteration': iteration,
        'z': np.round(z, 6),
        'x': np.round(x, 6),
        'z_value_list': np.round(z_value_list, 6),
        'x_value_list': x_value_list,
        'true_z': np.round(true_z, 6)
    }


def post_processing_linear_approximation_logs(results, visual=True, file_print=True, filename="linear_logs.xlsx"):
    logs = []

    for res in results:
        logs.append({
            'Метод': res['name'],
            'Итераций': res['iteration'],
            'Максимальное значение функции': res['z'],
            'Значения x': res['x'],
            'Значения целевой функции': res['z_value_list'],
            'Значения параметров функции': res['x_value_list'],
            'Истинное значение': res['true_z']
        })

    if visual:
        plt.figure(figsize=(10, 6))

        # Определим общее число итераций (по максимальному методу)
        max_len = max(len(entry['Значения целевой функции']) for entry in logs)
        x_common = np.linspace(0, max_len - 1, max_len)  # общая сетка итераций

        for entry in logs:
            method_name = entry['Метод']
            z_values = entry['Значения целевой функции']
            orig_len = len(z_values)

            if orig_len == 0:
                print(f"⚠ Метод '{method_name}' не содержит значений целевой функции — пропущен.")
                continue  # пропустить метод с пустым списком
            if orig_len == 1:
                # если только одна точка — просто растянем её по всему интервалу
                y_interp = [z_values[0]] * max_len
            else:
                # создаем оригинальные X (например, [0, 1, 2, ..., orig_len - 1])
                x_orig = np.linspace(0, max_len - 1, orig_len)
                # интерполируем на общую сетку
                y_interp = np.interp(x_common, x_orig, z_values)

            plt.plot(x_common, y_interp, label=method_name)

        plt.title('Изменение значения целевой функции по итерациям')
        plt.xlabel('Итерации (нормированные до общего масштаба)')
        plt.ylabel('Значение целевой функции')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        for entry in logs:
            method_name = entry['Метод']
            z_values = entry['Значения целевой функции']
            true_value = entry['Истинное значение']
            orig_len = len(z_values)

            if orig_len == 0:
                continue

            if orig_len == 1:
                y_interp = [z_values[0]] * max_len
            else:
                x_orig = np.linspace(0, max_len - 1, orig_len)
                y_interp = np.interp(x_common, x_orig, z_values)

            # Вычисляем ошибку MSE на каждой итерации (нарастающий список)
            mse = [(np.mean((np.array(y_interp[:i + 1]) - true_value) ** 2)) for i in range(max_len)]

            plt.plot(x_common, mse, label=method_name)

        plt.title('Ошибка (MSE) относительно истинного значения по итерациям')
        plt.xlabel('Итерации (нормированные до общего масштаба)')
        plt.ylabel('MSE')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 6))
        for entry in logs:
            method_name = entry['Метод']
            x_values = entry.get('Значения параметров функции', [])

            # Фильтрация по методам и размерности
            if not x_values or len(x_values[0]) != 2:
                continue

            x1_coords = [vec[0] for vec in x_values]
            x2_coords = [vec[1] for vec in x_values]

            plt.plot(x1_coords, x2_coords, label=method_name)

            # Отметим только начальную точку
            plt.scatter(x1_coords[0], x2_coords[0], s=100, marker='s', label=f'{method_name} start')

    plt.title('Траектория переменных x₁ и x₂ по итерациям')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True)
    plt.legend()
    plt.axis('equal')  # сохраняем пропорции
    plt.tight_layout()
    plt.show()

    if file_print:
        df = pd.DataFrame(logs)
        df.to_excel(filename, index=False)
        print(f"Логи сохранены в файл: {filename}")


if __name__ == "__main__":
    # c = [1, 1]
    # A = [[0.81, 0.4], [0.4, 1.1], [-1, 0], [0, -1]]
    # b = [1, 1, 0, 0]

    task_list = [
        (
            [1, 4],
            [[2, 3],
             [4, 1]],
            [3, 2],
            True
        ),
        (
            [1, 4, 2],
            [[2, 0.1, 3],
             [4, 1, -1]],
            [3, 2],
            True
        ),
        (
            [1, 4, 5],
            [[1, 2, 3],
             [3, 1, 2],
             [2, 3, 1]],
            [2, 2, 4],
            True  # 2.2
        ),
        (
            [1, 4, 5],
            [[1, 2, 3],
             [3, 1, 2],
             [2, 3, 1]],
            [2, 2, 4],
            True  # 2.2
        ),
        (
            [1, 2, 3, 4, 5, 7],
            [[1, 1, 0, 0, 0, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 1, 1],
             [1, 0, 1, 0, 1, 0],
             [0, 1, 0, 1, 0, 1]],
            [1, 1, 1, 1, 1],
            True  # 2.4
        ),
        (
            [3, 5, 1, 2, 4, 6],
            [[0, 1, 1, 0, 1, 0],
             [0, 1, 0, 1, 1, 0],
             [1, 0, 0, 0, 0, 1],
             [0, 1, 0, 0, 1, 0],
             [0, 0, 1, 1, 0, 0]],
            [2, 4, 8, 1, 3],
            True
        ),
        (
            [8.8, 5.55],
            [[3, 5],
             [-3, 5]],
            [14, 12],
            True
        ),
        (
            [1, 2, 3, 4],
            [[6, 7, 8, 9]],
            [100],
            True
        ),
        (
            [1, 1],
            [[0.81, 0.4],
             [0.4, 1.1]],
            [1, 1],
            True
        ),
        (
            [23, 12],
            [[111, 21],
             [-10, 6]],
            [123, 14],
            True
        ),
        (
            [2, 3, 5, 7],
            [[2, 0, 0, 25],
             [3, 1, 4, 0],
             [5, 6, 0, 0],
             [0, 0, 2, 7]],
            [20, 22, 10, 11],
            True
        ),
        (
            [1, 2, 4, 8],
            [[4, 0, 1, 0],
             [5, 4, 2, 0],
             [0, 0, 8, 11],
             [16, 3, 14, 0]],
            [110, 100, 90, 60],
            True
        ),
        (
            [4, 5, 2],
            [[2, -1, 2],
             [3, 5, 4],
             [1, 1, 2]],
            [9, 8, 2],
            True  # 4.3
        )]

    total_results = []

    for task in task_list:
        c, A, b, is_maximization = task

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)

        results = []

        methods = ['simplex', 'relaxation', 'column-generation', 'ADMM', 'mirror-descent']
        for method in methods:
            res = solve_lp(method, c, A, b, 10 ** -6, is_maximization, True)
            results.append(res)
            total_results.append(res)
            print("-" * 100)
        post_processing_linear_approximation_logs(results, True, False)

    post_processing_linear_approximation_logs(total_results, False, True)

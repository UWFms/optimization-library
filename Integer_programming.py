import math
from copy import deepcopy

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, value, pulp

from linear_programming import simplex


def is_integer(x, tol=1e-5):
    return np.all((np.abs(x - np.round(x)) < tol) | (np.isnan(x)))


def branch_and_bound(weights, values, capacity, console_print, max_iter=10000, epsilon=1e-5):
    from copy import deepcopy
    from collections import deque

    n = len(weights)
    best_value = float('-inf')
    best_solution = None
    stack = deque()
    stack.append({})
    iteration = 0
    z_value_list = []

    base_constraints_A = [weights]
    base_constraints_b = [capacity]

    # Ограничения 0 <= x_i <= 1
    for i in range(n):
        unit = [1 if j == i else 0 for j in range(n)]
        base_constraints_A.append(unit)  # x_i <= 1
        base_constraints_b.append(1)
        base_constraints_A.append([-u for u in unit])  # -x_i <= 0  => x_i >= 0
        base_constraints_b.append(0)

    while stack and iteration < max_iter:
        fixed = stack.popleft()

        A = deepcopy(base_constraints_A)
        b = deepcopy(base_constraints_b)

        # Добавление ограничений на фиксированные переменные: x_i = val
        for i, val in fixed.items():
            eq = [1 if j == i else 0 for j in range(n)]
            A.append(eq)
            b.append(val)
            A.append([-x for x in eq])
            b.append(-val)

        c = values.copy()

        if console_print:
            print(f"Пытаемся решить LP с fixed={fixed}")

        try:
            z_value, answers, *_ = simplex(c, A, b, True, console_print=False)
        except Exception as e:
            print(f"[!] Исключение при фиксированных {fixed}: {e}")
            iteration += 1
            continue

        if console_print:
            print(f"\n--- Branch and Bound Iteration {iteration + 1} ---")
            print(f"Fixed variables (applied constraints):")
            for i, val in fixed.items():
                print(f"  x[{i}] = {val} (answer: {answers[i]:.4f}) {'✔' if abs(answers[i] - val) < epsilon else '✘'}")
            print(f"z = {z_value:.4f}")
            print(f"x = {answers}")

        # Если текущее решение целочисленное
        if all(abs(xi - round(xi)) < epsilon for xi in answers[:n]):
            if z_value > best_value:
                best_value = z_value
                best_solution = answers
                z_value_list.append(z_value)
                if console_print:
                    print(f"Новое лучшее целочисленное решение: z = {best_value}, x = {best_solution}")
            continue

        if z_value <= best_value + epsilon:
            iteration += 1
            continue

        # Поиск переменной с дробным значением
        fractional_index = next(
            (i for i in range(n) if i not in fixed and abs(answers[i] - round(answers[i])) > epsilon),
            -1
        )

        if fractional_index != -1:
            k = fractional_index
            stack.append({**fixed, k: 0})
            stack.append({**fixed, k: 1})

        iteration += 1

    return best_value, best_solution, iteration, z_value_list


def add_gomory_cut(A, b, c, tableau_row, basis_row_index):
    from math import floor

    # Получаем дробные части коэффициентов
    fractional_row = [val - floor(val) for val in tableau_row[:-1]]
    rhs = tableau_row[-1]
    fractional_rhs = rhs - floor(rhs)

    # Новый cut: сумма дробных коэффициентов ≤ дробная часть свободного члена
    new_cut = [round(f % 1, 4) for f in fractional_row]

    # Добавляем slack переменную
    for row in A:
        row.append(0.0)  # добавляем 0 в каждую строку — slack-переменная ещё нигде не участвует
    new_cut.append(1.0)  # это slack-переменная в новой строке

    A.append(new_cut)
    b.append(round(fractional_rhs, 4))
    c.append(0.0)  # цель не зависит от этой переменной

    return A, b, c


def is_integral(x, eps=1e-5):
    return all(abs(val - round(val)) <= eps for val in x)

def find_fractional_row(x, basis, eps=1e-5):
    for row_idx, var_idx in enumerate(basis):
        if var_idx < len(x):
            val = x[var_idx]
            if abs(val - round(val)) > eps:
                return row_idx
    return -1


def gomory_cut_binary(weights, values, capacity, console_print, max_iter=100, epsilon=1e-5):
    from scipy.optimize import linprog
    import numpy as np

    n = len(weights)
    A = [weights]
    b = [capacity]
    c = [-v for v in values]

    # Ограничения x_i <= 1
    for i in range(n):
        a_row = [0] * n
        a_row[i] = 1
        A.append(a_row)
        b.append(1)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    iteration = 0
    z_value_list = []

    while iteration < max_iter:
        bounds = [(0, None)] * n
        res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

        if not res.success or res.x is None:
            if console_print:
                print(f'❌ LP не решена на итерации {iteration}')
            return float('-inf'), None, iteration, z_value_list

        x_lp = res.x
        z_lp = -res.fun
        z_value_list.append(z_lp)

        if console_print:
            print(f"\n--- Gomory Iteration {iteration + 1} ---")
            print(f"z = {z_lp:.4f}")
            print("x =", [f"{xi:.4f}" for xi in x_lp])

        # Проверяем, есть ли дробные переменные
        frac_vars = [(i, xi) for i, xi in enumerate(x_lp) if abs(xi - round(xi)) > epsilon]
        if not frac_vars:
            # Все целочисленные
            x_int = [round(xi) for xi in x_lp]
            return z_lp, x_int, iteration + 1, z_value_list

        # Берём первую дробную переменную
        frac_idx, frac_val = frac_vars[0]

        # Добавляем ограничение x_i <= floor(x_i)
        new_row = np.zeros(n)
        new_row[frac_idx] = 1.0
        A = np.vstack([A, new_row])
        b = np.append(b, np.floor(frac_val))

        if console_print:
            print(f"Добавлено отсечение: x[{frac_idx}] <= {np.floor(frac_val):.4f}")

        iteration += 1

    if console_print:
        print("⚠️ Превышено число итераций, решение не целочисленное")

    return float('-inf'), None, iteration, z_value_list


def cutting_planes_method(weights, values, capacity, console_print=True, max_iter=20, epsilon=1e-5):
    from pulp import LpProblem, LpVariable, LpMaximize, lpDot, PULP_CBC_CMD, LpStatus
    import numpy as np

    n = len(weights)
    A = [weights]
    b = [capacity]
    c = values.copy()
    bounds = [(0, 1)] * n
    integer_indices = list(range(n))

    prob = LpProblem("CuttingPlanes", LpMaximize)
    x_vars = [LpVariable(f"x{i}", lowBound=bounds[i][0], upBound=bounds[i][1]) for i in range(n)]
    prob += lpDot(c, x_vars)

    for i, row in enumerate(A):
        prob += lpDot(row, x_vars) <= b[i]

    solver = PULP_CBC_CMD(msg=False)

    best_z = None
    best_x = None
    z_value_list = []
    iteration = 0

    while iteration < max_iter:
        prob.solve(solver)
        status = LpStatus[prob.status]
        if status != "Optimal":
            if console_print:
                print(f"Iteration {iteration}: problem status = {status}, stopping.")
            break

        x_val = np.array([v.varValue for v in x_vars])
        z_val = prob.objective.value()

        z_value_list.append(z_val)

        # Ищем переменную с максимальной дробной частью среди целочисленных
        fractional_index = -1
        max_frac = 0
        for i in integer_indices:
            frac = abs(x_val[i] - round(x_val[i]))
            if frac > epsilon and frac > max_frac:
                fractional_index = i
                max_frac = frac

        if fractional_index == -1:
            if z_val == best_z:
                return best_z, best_x, iteration, z_value_list
            else:
                best_z = z_val
                best_x = x_val.copy()

        if console_print:
            print(f"Iteration {iteration}: z = {z_val:.6f}, x = {x_val}, max fractional index = {fractional_index}, max fraction = {max_frac:.6f}")

        if fractional_index is None:
            if console_print:
                print("All integer variables within epsilon tolerance. Stopping.")
            break

        floor_val = np.floor(x_val[fractional_index])
        prob += x_vars[fractional_index] <= floor_val

        iteration += 1

    return best_z, best_x, iteration, z_value_list


def lagrangian_relaxation(weights, values, capacity, console_print, max_iter=100, epsilon=1e-6, alpha=2.0):
    n = len(values)
    u = 0.0
    best_value = float('-inf')
    best_x = [0] * n
    z_value_list = []

    for iteration in range(max_iter):
        # Вычисляем лагранжевы стоимости
        lagr_c = [v - u * w for v, w in zip(values, weights)]

        # Ограничения для 0 ≤ x ≤ 1
        A, b = [], []
        for i in range(n):
            A.append([1 if j == i else 0 for j in range(n)]);
            b.append(1)
            A.append([-1 if j == i else 0 for j in range(n)]);
            b.append(0)

        try:
            z, x_lp, *_ = simplex(lagr_c, A, b, is_maximization=True, console_print=False)
        except Exception:
            break

        # Округление решения: сортировка по значению v/w
        scores = [(i, values[i] / weights[i] if weights[i] > 0 else float('inf')) for i in range(n)]
        scores.sort(key=lambda x: -x[1])

        x_bin = [0] * n
        total_weight = 0

        for i, _ in scores:
            if total_weight + weights[i] <= capacity:
                x_bin[i] = 1
                total_weight += weights[i]

        # Проверка качества бинарного решения
        total_value = sum(values[i] * x_bin[i] for i in range(n))
        total_weight = sum(weights[i] * x_bin[i] for i in range(n))
        subgrad = sum(weights[i] * x_lp[i] for i in range(n)) - capacity


        if console_print:
            print(f"\n--- Lagrangian Relaxation Iteration {iteration + 1} ---")
            print(f"lagrangian_obj = {lagr_c}")
            print(f"z = {best_value}")
            print(f"x = {best_x}")
            print(f"subgrad = {subgrad}")
            print(f"u = {u}")

        z_value_list.append(best_value)

        if total_value == best_value:
            return best_value, best_x, iteration + 1, z_value_list

        # Улучшение
        if total_weight <= capacity and total_value > best_value + epsilon:
            best_value = total_value
            best_x = x_bin.copy()

        # Проверка сходимости
        if abs(subgrad) < epsilon:
            break

        # Шаг субградиентного метода
        step_size = alpha / (iteration + 1)
        u = max(0.0, u + step_size * subgrad)

    return best_value, best_x, iteration + 1, z_value_list


def make_row(size, values):
    row = [0] * size
    for i, v in values.items():
        row[i] = v
    return row


def sherali_adams_level_1(weights, values, capacity, console_print, max_iter=100, epsilon=1e-5):
    import itertools

    n = len(weights)
    A = [weights]
    b = [capacity]

    # Ограничения 0 ≤ x_i ≤ 1
    for i in range(n):
        unit = [1 if j == i else 0 for j in range(n)]
        A.append(unit)
        b.append(1)
        A.append([-u for u in unit])
        b.append(0)

    y_idx = list(itertools.combinations(range(n), 2))
    z_idx = list(itertools.combinations(range(n), 3))
    m, p = len(y_idx), len(z_idx)

    total_vars = n + m + p
    c = values + [0] * (m + p)
    A_ext = [row + [0] * (m + p) for row in A]
    b_ext = b.copy()

    # SA1
    for k, (i, j) in enumerate(y_idx):
        y_col = n + k
        A_ext.append(make_row(total_vars, {i: -1, y_col: 1}))
        b_ext.append(0)
        A_ext.append(make_row(total_vars, {j: -1, y_col: 1}))
        b_ext.append(0)
        A_ext.append(make_row(total_vars, {i: 1, j: 1, y_col: -1}))
        b_ext.append(1)

    # SA2 + SA3
    for k, (i, j, k_var) in enumerate(z_idx):
        z_col = n + m + k
        # Найти индекс y_ij
        try:
            y_pos = y_idx.index((i, j))
        except ValueError:
            y_pos = y_idx.index((j, i))
        y_col = n + y_pos

        A_ext.append(make_row(total_vars, {y_col: -1, z_col: 1}))
        b_ext.append(0)
        A_ext.append(make_row(total_vars, {k_var: -1, z_col: 1}))
        b_ext.append(0)
        A_ext.append(make_row(total_vars, {y_col: 1, k_var: 1, z_col: -1}))
        b_ext.append(1)

        for var in (i, j, k_var):
            A_ext.append(make_row(total_vars, {var: -1, z_col: 1}))
            b_ext.append(0)
        A_ext.append(make_row(total_vars, {i: 1, j: 1, k_var: 1, z_col: -1}))
        b_ext.append(2)

    try:
        z, x_full, *_ = simplex(c, A_ext, b_ext, is_maximization=True, console_print=False)
    except Exception:
        return None, None, 0, []

    x = x_full[:n]

    if console_print:
        print(f"Максимальное значение функции: {z}")
        print(f"Значения x: {x}")

    if all(abs(xi - round(xi)) < epsilon for xi in x):
        return z, [int(round(xi)) for xi in x], 1, [z]

    return branch_and_bound(weights, values, capacity, False, max_iter,
                            epsilon=epsilon)


def solve_ip(method, weights, values, capacity, epsilon, console_print=True):
    if method == 'branch_and_bound':
        z, x, iteration, z_value_list = branch_and_bound(weights, values, capacity, console_print, 100, epsilon=epsilon)
        name = 'Метод ветвей и границ'

    elif method == 'gomory':
        z, x, iteration, z_value_list = gomory_cut_binary(weights, values, capacity, console_print, 100, epsilon=epsilon)
        name = 'Метод Гомори'

    elif method == 'cutting_planes':
        z, x, iteration, z_value_list = cutting_planes_method(weights, values, capacity, console_print, 100,
                                                              epsilon=epsilon)
        name = 'Метод отсеканий'

    elif method == 'lagrangian_relaxation':
        z, x, iteration, z_value_list = lagrangian_relaxation(weights, values, capacity, console_print, 100,
                                                              epsilon=epsilon)
        name = 'Метод Лагранжевой релаксаций'

    elif method == 'Sherali-Adams-1':
        z, x, iteration, z_value_list = sherali_adams_level_1(weights, values, capacity, console_print, 100,
                                                              epsilon=epsilon)
        name = 'Метод Шерали-Адамса'

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    if console_print:
        print(f"\n=== {name} ===")
        print(f"Максимальное значение функции: {z}")
        print(f"Значения x: {x}")
        print(f"итераций: {iteration}")

    model = LpProblem("0-1 Knapsack Problem", LpMaximize)
    x_opt = [LpVariable(f"x_{i}", cat="Binary") for i in range(len(weights))]
    model += lpSum([values[i] * x_opt[i] for i in range(len(weights))])
    model += lpSum([weights[i] * x_opt[i] for i in range(len(weights))]) <= capacity
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    true_z = value(model.objective)

    return {
        'method': method,
        'name': name,
        'iteration': iteration,
        'z': z,
        'x': x,
        'z_value_list': np.round(z_value_list, 6),
        'true_z': np.round(true_z, 6),
        'weights': weights,
        'values': values,
        'capacity': capacity
    }


def post_processing_integer_approximation_logs(results, visual=True, file_print=True, filename="integer_logs.xlsx"):
    logs = []

    for res in results:
        logs.append({
            'Метод': res['name'],
            'Итераций': res['iteration'],
            'Максимальное значение функции': res['z'],
            'Значения x': res['x'],
            'Значения целевой функции': res['z_value_list'],
            'Истинное значение': res['true_z'],
            'Веса': res['weights'],
            'Цены': res['values'],
            'Вместимость': res['capacity']
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

    if file_print:
        df = pd.DataFrame(logs)
        df.to_excel(filename, index=False)
        print(f"Логи сохранены в файл: {filename}")


if __name__ == "__main__":
    # weights = [2, 3, 4, 5]
    # values = [3, 4, 5, 6]
    # capacity = 10

    task_list = [
        (
            [3.4, 4.1, 5.2, 6.1, 1.3, 2.8, 7.2],
            [1.2, 2.4, 4.1, 5.4, 0.1, 3.5, 4.4],
            15
        ),
        (
            [100, 600, 1200, 2400, 500, 2000],
            [8, 12, 13, 64, 22, 41],
            1600
        ),
        (
            [600.1, 310.5, 1800, 3850, 18.6, 198.7, 882, 4200, 402.5, 327],
            [20, 5, 100, 200, 2, 4, 60, 150, 80, 40],
            1000
        ),
        (
            [100, 220, 90, 400, 300, 400, 205, 120, 160, 580],
            [8, 24, 13, 80, 70, 80, 45, 15, 28, 90],
            600
        ),
        (
            [400, 140, 100, 1300, 650, 320, 480, 80, 60, 2550],
            [130, 32, 20, 120, 40, 30, 20, 6, 3, 180],
            600
        ),
        (
            [3100, 1100, 950, 450, 300, 220, 200, 520],
            [220, 50, 30, 50, 12, 5, 8, 18],
            1000
        ),
        (
            [560, 1125, 300, 620, 2100, 431, 68, 328, 47, 122, 322, 196],
            [40, 91, 10, 30, 160, 20, 3, 12, 3, 18, 9, 25],
            800
        ),
        (
            [560, 1125, 300, 620, 2100],
            [316, 72, 71, 49, 108],
            1600
        )
    ]

    total_results = []

    for task in task_list:
        weights, values, capacity = task
        print(f"{weights=} {values=} {capacity=}")

        methods = ['branch_and_bound', 'gomory', 'cutting_planes', 'lagrangian_relaxation', 'Sherali-Adams-1']
        results = []

        for method in methods:
            res = solve_ip(method, weights, values, capacity, console_print=True, epsilon=1e-4)
            results.append(res)
            total_results.append(res)
            print("-" * 100)
        post_processing_integer_approximation_logs(results, False, False)

    post_processing_integer_approximation_logs(total_results, False, True)

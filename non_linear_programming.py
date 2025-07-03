import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# Целевая функция
def objective(model, x, t, y):
    return np.mean((y - model(x, t)) ** 2)


# Градиент (численно)
def gradient(model, x, t, y, h=1e-5):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x1 = np.array(x)
        x2 = np.array(x)
        x1[i] -= h
        x2[i] += h
        grad[i] = (objective(model, x2, t, y) - objective(model, x1, t, y)) / (2 * h)
    return grad


def norm(vec):
    return np.sqrt(np.sum(vec ** 2))


# Градиентный спуск
def gradient_descent(x0, t, y, epsilon, model, alpha=0.001, max_iter=5000, console_print=True):
    x = np.array(x0)
    errors_list = []
    x_value_list = []
    for i in range(max_iter):
        grad = gradient(model, x, t, y)
        if norm(grad) < epsilon:
            break
        if i < 3 and console_print:
            print(f"x_{i} = {x} grad = {grad} x_{i + 1} = x - alpha * grad = {x - alpha * grad}")
        x = x - alpha * grad

        errors_list.append(objective(model, x, t, y))
        x_value_list.append(x)
    return x, i, errors_list, x_value_list


# Метод Ньютона
def newton_method(x0, t, y, bounds, epsilon, model, max_iter=5000, regularization=1e-4, console_print=True):
    def gradient(x, t, y):
        eps = 1e-8
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective(model, x_plus, t, y) - objective(model, x_minus, t, y)) / (2 * eps)
        return grad

    def hessian(x, t, y):
        eps = 1e-6
        n = len(x)
        hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x_pp = x.copy()
                x_pp[i] += eps
                x_pp[j] += eps
                x_pm = x.copy()
                x_pm[i] += eps
                x_pm[j] -= eps
                x_mp = x.copy()
                x_mp[i] -= eps
                x_mp[j] += eps
                x_mm = x.copy()
                x_mm[i] -= eps
                x_mm[j] -= eps
                hess[i, j] = (objective(model, x_pp, t, y) - objective(model, x_pm, t, y) -
                              objective(model, x_mp, t, y) + objective(model, x_mm, t, y)) / (4 * eps * eps)
        return hess

    x = np.array(x0)
    bounds = np.array(bounds)
    best_x = x.copy()
    best_obj = float('inf')
    errors_list = []
    x_value_list = []

    for i in range(max_iter):
        grad = gradient(x, t, y)
        hess = hessian(x, t, y)

        # Регуляризация гессиана
        try:
            direction = -np.linalg.solve(hess + regularization * np.eye(len(x)), grad)
        except np.linalg.LinAlgError:
            direction = -grad

        # Поиск шага с условием Армихо
        alpha = 1.0
        c = 0.5
        current_obj = objective(model, x, t, y)

        for _ in range(5):
            x_new = np.clip(x + alpha * direction, bounds[:, 0], bounds[:, 1])
            new_obj = objective(model, x_new, t, y)
            if new_obj < current_obj + c * alpha * np.dot(grad, direction):
                break
            alpha *= 0.5

        x = np.clip(x + alpha * direction, bounds[:, 0], bounds[:, 1])

        if current_obj < best_obj:
            best_x = x.copy()
            best_obj = current_obj

        if console_print and i < 10:
            print(f"Iter {i}: x={x}, grad_norm={np.linalg.norm(grad)}, alpha={alpha}, obj={current_obj}")

        errors_list.append(objective(model, x, t, y))
        x_value_list.append(x)

        if np.linalg.norm(grad) < epsilon:
            break

    return np.round(x, 6), i, errors_list, x_value_list


# Метод наискорейшего спуска
def steepest_descent(x0, t, y, bounds, epsilon, model, max_iter=5000, console_print=True):
    def gradient(x, t, y, model):
        eps = 1e-5
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective(model, x_plus, t, y) - objective(model, x_minus, t, y)) / (2 * eps)
        return grad

    def backtracking_line_search(x, direction, grad, t, y, model, alpha=1.0, rho=0.5, c=1e-4):
        fx = objective(model, x, t, y)
        while True:
            x_new = x + alpha * direction
            x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])
            fx_new = objective(model, x_new, t, y)
            if fx_new <= fx + c * alpha * np.dot(grad, direction):
                break
            alpha *= rho
            if alpha < 1e-8:  # минимальный шаг
                break
        return alpha

    x = np.array(x0)
    bounds = np.array(bounds)
    errors_list = []
    x_value_list = []

    for i in range(max_iter):
        grad = gradient(x, t, y, model)
        direction = -grad

        if np.linalg.norm(grad) < epsilon:
            break

        alpha = backtracking_line_search(x, direction, grad, t, y, model)
        x_new = x + alpha * direction
        x_new = np.clip(x_new, bounds[:, 0], bounds[:, 1])

        x = x_new
        errors_list.append(objective(model, x, t, y))
        x_value_list.append(x)

        if console_print and i < 10:
            print(f"Iter {i}: x={x}, grad_norm={np.linalg.norm(grad)}, alpha={alpha}")

    return x, i + 1, errors_list, x_value_list


def adam_optimizer(x0, t, y, epsilon, model, alpha=0.001, beta1=0.9, beta2=0.99, max_iter=5000, tol=1e-8,
                   console_print=True):
    x = np.array(x0, dtype=float)
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    errors_list = []
    x_value_list = []

    for t_iter in range(1, max_iter + 1):
        grad = gradient(model, x, t, y)

        if norm(grad) < epsilon:
            break

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t_iter)
        v_hat = v / (1 - beta2 ** t_iter)

        if t_iter < 4 and console_print:
            print(f"x_{t_iter} = {x} grad = {grad} m_hat = {m_hat} v_hat = {v_hat} "
                  f"x_{t_iter + 1} = x - alpha * m_hat / (sqrt(v_hat) + epsi) = "
                  f"{x - alpha * m_hat / (np.sqrt(v_hat) + epsilon)}")
        x = x - alpha * m_hat / (np.sqrt(v_hat) + tol)

        errors_list.append(objective(model, x, t, y))
        x_value_list.append(x)

    return np.round(x, 6), t_iter, errors_list, x_value_list


def nelder_mead(objective, x0, t, y, epsilon, model,
                alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5,
                max_iter=5000, console_print=True):
    n = len(x0)
    # Инициализируем симплекс (n+1 точка)
    simplex = [x0]
    for i in range(n):
        x = np.array(x0, copy=True)
        x[i] += 0.05 if x[i] == 0 else 0.05 * x[i]
        simplex.append(x)
    simplex = np.array(simplex)
    errors_list = []
    x_value_list = []

    for iteration in range(max_iter):
        # Сортировка по значению функции
        simplex = sorted(simplex, key=lambda x: objective(model, x, t, y))

        errors_list.append(objective(model, simplex[0], t, y))
        x_value_list.append(simplex[0])

        f_values = [objective(model, x, t, y) for x in simplex]

        best = simplex[0]
        worst = simplex[-1]
        second_worst = simplex[-2]

        # Проверка сходимости по разнице значений функции
        if np.max(np.abs(np.array(f_values) - f_values[0])) < epsilon:
            break

        # Центр масс всех точек, кроме худшей
        x_bar = np.mean(simplex[:-1], axis=0)

        # Отражение
        x_r = x_bar + alpha * (x_bar - worst)
        f_r = objective(model, x_r, t, y)

        if iteration < 4 and console_print:
            print(f"x_{iteration} = {simplex[0]} \n best = {best} worst = {worst} second_worst = {second_worst} \n "
                  f"x_bar = {x_bar} x_r = {x_r} \n x_{iteration + 1} = best + sigma * (simplex[0] - best) = "
                  f"{list(best + sigma * (simplex[0] - best))[0]}")

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1] = x_r
            if iteration < 4:
                print(f"{f_r=}, {x_r=}")
            continue

        # Расширение
        if f_r < f_values[0]:
            x_e = x_bar + gamma * (x_r - x_bar)
            f_e = objective(model, x_e, t, y)
            simplex[-1] = x_e if f_e < f_r else x_r
            if iteration < 4:
                print(f"{f_r=}, f_best = {f_values[0]}, {x_e=}, {f_e=},worst = {simplex[-1]}")
            continue

        # Сжатие
        x_c = x_bar + rho * (worst - x_bar)
        f_c = objective(model, x_c, t, y)

        if iteration < 4:
            print(
                f"x_{iteration} = {simplex[0]} \n best = {best} worst = {worst} second_worst = {second_worst} \n x_bar = {x_bar} x_r = {x_r} x_c = {x_c} \n x_{iteration + 1} = best + sigma * (simplex[0] - best) = {list(best + sigma * (simplex[0] - best))[0]}")

        if f_c < f_values[-1]:
            simplex[-1] = x_c
            continue

        # Сжатие симплекса к лучшей точке
        for i in range(1, len(simplex)):
            simplex[i] = best + sigma * (simplex[i] - best)

    return np.round(simplex[0], 6), iteration, errors_list, x_value_list


def solve_nlp(method, x0, t, y, model, epsilon, bounds, console_print=True):
    if method == 'gd':
        x_opt, iteration, errors_list, x_value_list = gradient_descent(x0, t, y, epsilon, model, console_print=console_print)
        name = "Градиентный спуск"

    elif method == 'newton':
        x_opt, iteration, errors_list, x_value_list = newton_method(x0, t, y, bounds, epsilon, model, console_print=console_print)
        name = "Метод Ньютона"

    elif method == 'sd':
        x_opt, iteration, errors_list, x_value_list = steepest_descent(x0, t, y, bounds, epsilon, model,
                                                                       console_print=console_print)
        name = "Метод наискорейшего спуска"

    elif method == 'adam':
        x_opt, iteration, errors_list, x_value_list = adam_optimizer(x0, t, y, epsilon, model, console_print=console_print)
        name = "Метод Adam"

    elif method == 'nelder-mead':
        x_opt, iteration, errors_list, x_value_list = nelder_mead(objective, x0, t, y, epsilon, model, console_print=console_print)
        name = "Метод Nelder–Mead"

    elif method == 'lbfgs':
        result = minimize(objective, x0, args=(t, y), bounds=bounds, method='L-BFGS-B', tol=epsilon)
        x_opt = np.round(result.x, 6)
        iteration = None
        errors_list = []
        x_value_list = []
        name = "Эталонный метод (L-BFGS-B)"

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    y_pred = np.round(model(x_opt, t), 6)
    errors = np.round((y_pred - y) ** 2, 6)

    if console_print:
        print(f"\n=== {name} ===")
        print(f"Оптимальные параметры: {np.round(x_opt, 3)}")
        print(f"Итераций: {iteration}")
        print(f"Предсказания: {np.round(y_pred, 3)}")
        print(f"Истинные значения: {np.round(y, 3)}")
        print(f"Ошибки по итерациям: {np.round(errors_list, 3)}")
        print(f"Ошибки предсказаний: {np.round(errors, 3)}")
        print(f"Средняя ошибка предсказаний: {np.round(np.mean(errors), 3)}")

    return {
        'method': method,
        'name': name,
        'x_opt': np.round(x_opt, 3),
        'iteration': iteration,
        'y_pred': np.round(y_pred, 3),
        'y_data': np.round(y, 3),
        'errors_list': np.round(errors_list, 3),
        'x_value_list': x_value_list,
        'errors_pred': np.round(errors, 3),
        'mean_error_pred': np.round(np.mean(errors), 3)
    }


def post_processing_non_linear_approximation_logs(results, visual=True, file_print=True,
                                                  filename="non_linear_logs.xlsx"):
    logs = []

    for res in results:
        logs.append({
            'Метод': res['name'],
            'Итераций': res['iteration'],
            'Параметры модели': res['x_opt'],
            'Предсказание': res['y_pred'],
            'Истинное значение': res['y_data'],
            'Ошибки по итерациям': res['errors_list'],
            'Значения параметров функции': res['x_value_list'],
            'Ошибки предсказаний': res['errors_pred'],
            'Средняя ошибка предсказаний': res['mean_error_pred'],
        })

    if visual:
        plt.figure(figsize=(10, 6))

        # Определим общее число итераций (по максимальному методу)
        max_len = max(len(entry['Ошибки по итерациям']) for entry in logs)
        x_common = np.linspace(0, max_len - 1, max_len)  # общая сетка итераций

        # Список интерполированных y для всех методов
        all_y_interp = []

        for entry in logs:
            method_name = entry['Метод']
            z_values = entry['Ошибки по итерациям']

            orig_len = len(z_values)

            if orig_len == 1:
                # если только одна точка — просто растянем её по всему интервалу
                y_interp = [z_values[0]] * max_len
            else:
                # создаем оригинальные X (например, [0, 1, 2, ..., orig_len - 1])
                x_orig = np.linspace(0, max_len - 1, orig_len)
                # интерполируем на общую сетку
                y_interp = np.interp(x_common, x_orig, z_values)

            # фильтруем слишком большие значения
            y_filtered = np.array(y_interp)
            y_filtered[y_filtered > 100000000] = np.nan

            plt.plot(x_common, y_filtered, label=method_name)

        plt.title('Изменение ошибки по итерациям')
        plt.xlabel('Итерации (нормированные до общего масштаба)')
        plt.ylabel('Значение ошибки')
        plt.grid(True)
        plt.yscale('log')
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
    t_value = np.linspace(0, 2 * np.pi, 10)
    task_list = [
        (
            np.array([0.25, 0.25]),
            np.exp(-0.5 * t_value) + np.sin(-1.2 * t_value),
            t_value,
            lambda x, t: np.exp(-x[0] * t) + np.sin(x[1] * t),
            [(-3.5, 2.5), (-4.2, 2.8)]
        ),
        (
            np.array([0.25, 0.25, 0.25, 0.25]),
            2 * np.exp(-0.5 * t_value) + np.sin(-1.2 * t_value),
            t_value,
            lambda x, t: x[0] * np.exp(-x[1] * t) + x[2] * np.sin(x[3] * t),
            [(-1, 5), (-3.5, 2.5), (-2, 4), (-4.2, 2.8)]
        ),
        (
            np.array([0.25, 0.25]),
            2 * np.abs(-1 * t_value),
            t_value,
            lambda x, t: x[0] * np.abs(x[1] * t),
            [(-1, 5), (-4, 2)]
            # Абсолютная функция
        ),
        (
            np.array([0.25, 0.25, 0.25]),
            3 * np.sin(5 * t_value) + 4 * t_value ** 2,
            t_value,
            lambda x, t: x[0] * np.sin(x[1] * t) + x[2] * t ** 2,
            [(0, 6), (2, 8), (1, 7)]
            # Синусоидальная функция
        ),
        (
            np.array([0.25, 0.25, 0.25]),
            t_value ** 2 - 10 * np.cos(2 * np.pi * t_value) + 10,
            t_value,
            lambda x, t: t ** 2 - x[0] * np.cos(x[1] * np.pi * t) + x[2],
            [(7, 13), (-1, 5), (7, 13)]
            # Функция Растригина
        ),
        (
            np.array([0.25]),
            (1 - t_value) ** 2,
            t_value,
            lambda x, t: (x[0] - t) ** 2,
            [(-2, 4)]
            # Функция Розенброка
        ),
        (
            np.array([0.25, 0.25, 0.25]),
            t_value ** 4 - 16 * t_value ** 2 + 5 * t_value,
            t_value,
            lambda x, t: x[0] * t ** 4 - x[1] * t ** 2 + x[2] * t,
            [(-2, 4), (13, 19), (2, 8)]
            # Функция Бивила
        ),
        (
            np.array([0.25, 0.25, 0.25, 0.25]),
            0.5 + (np.sin(t_value ** 2 - 0.5) ** 2) / (1 + 0.001 * t_value ** 2) ** 2,
            t_value,
            lambda x, t: x[0] + (np.sin(t ** 2 - x[1]) ** 2) / (x[2] + x[3] * t ** 2) ** 2,
            [(-2.5, 3.5), (-2.5, 3.5), (-2, 4), (-0.004, 0.006)]
            # Функция Шаффера no2
        ),
        (
            np.array([0.25]),
            t_value ** 2 * np.cos(2 * np.pi * t_value),
            t_value,
            lambda x, t: t ** 2 * np.cos(x[0] * np.pi * t),
            [(-2, 5)]
            # Функция Буте no2
        ),
        (
            np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]),
            -20 * np.exp(-0.2 * (np.sqrt(0.5 * t_value ** 2))) - np.exp(0.5 * np.cos(2 * np.pi * t_value)) + np.e + 20,
            t_value,
            lambda x, t: -x[0] * np.exp(-x[1] * (np.sqrt(x[2] * t ** 2))) - np.exp(
                x[3] * np.cos(x[4] * np.pi * t)) + np.e + x[5],
            [(-23, -17), (-0.5, 0.1), (-0.5, 1), (-0.5, 1), (-1, 5), (17, 23)]
            # Функция Экли
        )]

    total_results = []

    for task in task_list:
        x0_manual, y_data_manual, t_data_manual, model_manual, bounds_manual = task

        methods = ['gd', 'newton', 'sd', 'adam', 'nelder-mead']
        results = []

        for method in methods:
            res = solve_nlp(method, x0_manual, t_data_manual, y_data_manual, model_manual, bounds=bounds_manual,
                            epsilon=1e-6, console_print=True)
            results.append(res)
            total_results.append(res)
            print("-" * 100)
        post_processing_non_linear_approximation_logs(results, True, False)

    post_processing_non_linear_approximation_logs(total_results, False, True)

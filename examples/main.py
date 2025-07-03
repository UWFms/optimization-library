import re
from typing import List, Union

import numpy as np

from optimization_library.optimization_library.Integer_programming import post_processing_integer_approximation_logs, solve_ip
from optimization_library.optimization_library.linear_programming import post_processing_linear_approximation_logs, solve_lp
from optimization_library.optimization_library.non_linear_programming import solve_nlp, post_processing_non_linear_approximation_logs


def get_float_list(prompt, length):
    while True:
        try:
            values = input(prompt).strip().split()
            if len(values) != length:
                print(f"Введите ровно {length} чисел.")
                continue
            return [float(v) for v in values]
        except ValueError:
            print("Ошибка: введите числа, разделённые пробелами.")


def get_constraint(n):
    while True:
        try:
            parts = input(
                f"Введите ограничение в формате: {n} коэф. знак свободный член (например: 1 2 3 >= 5): ").strip().split()
            if len(parts) != n + 2:
                print(f"Ошибка: должно быть {n} коэф., знак, свободный член.")
                continue
            coeffs = list(map(float, parts[:n]))
            sign = parts[n]
            rhs = float(parts[n + 1])
            if sign not in (">=", "<="):
                print("Поддерживаются только знаки '>=' и '<='.")
                continue
            if sign == ">=":
                # Переводим >= в <= путём умножения на -1
                coeffs = [-x for x in coeffs]
                rhs = -rhs
            return coeffs, rhs
        except ValueError:
            print("Ошибка: убедитесь, что все значения числовые и правильно введены.")


def parse_and_build_model(expr: str) -> tuple[
    List[Union[int, float]],
    int,
    callable
]:
    """Преобразует математическое выражение в параметризованную модель"""

    """Преобразует математическое выражение в параметризованную модель"""

    # 1. Заменяем все x на t (кроме случаев в np.x)
    expr = re.sub(r'(?<!\w)x(?!\w)', 't', expr)

    # 2. Заменяем функции на их numpy-аналоги
    func_replacements = {
        'exp': 'np.exp',
        'sin': 'np.sin',
        'cos': 'np.cos',
        'tg': 'np.tan',
        'ctg': '1/np.tan',
        'abs': 'np.abs',
        'sqrt': 'np.sqrt',
        'pow': 'np.power'
    }

    for func, repl in func_replacements.items():
        expr = expr.replace(func, repl)

    # 3. Находим все числовые константы (исключая те, что уже в x[...])
    numbers = []

    def record_number(match):
        num = match.group()
        numbers.append(float(num) if '.' in num else int(num))
        return f"__NUM_{len(numbers) - 1}__"

    temp_expr = re.sub(r'(?<!__NUM_)(?<!\w)(?:\d+\.\d+|\.\d+|\d+\.?)(?!\w)', record_number, expr)

    # 4. Восстанавливаем числа с заменой на x[i]
    for i in range(len(numbers)):
        temp_expr = temp_expr.replace(f"__NUM_{i}__", f"x[{i}]")

    # 5. Проверяем синтаксис
    try:
        test_env = {'np': np, 'x': np.array([1.0] * len(numbers)), 't': 1.0}
        eval(temp_expr, test_env)
    except Exception as e:
        raise ValueError(f"Ошибка в выражении: {str(e)}\nПреобразованное: {temp_expr}")

    # 6. Создаем функцию
    def model(x: Union[List[Union[int, float]], np.ndarray], t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        try:
            if isinstance(x, list):
                x = np.array(x, dtype=float)
            return eval(temp_expr, {'np': np, 'x': x, 't': t})
        except Exception as e:
            raise ValueError(f"Ошибка вычисления: {str(e)}\nПри x={x}, t={t}\nВыражение: {temp_expr}")

    return numbers, len(numbers), model


def main():
    print("Выберите тип задачи:")
    print("1. Линейное программирование")
    print("2. Нелинейное программирование")
    print("3. Целочисленное программирование")
    task_type = input("Введите номер задачи: ").strip()

    if task_type not in ["1", "2", "3"]:
        print("Реализованы только эти типы задач.")
        return

    if task_type == "1":
        print("Макет задачи: Задача о рационе\n")
        n = int(input("Введите количество переменных: "))

        c = get_float_list(f"Введите {n} коэффициентов целевой функции через пробел: ", n)

        A = []
        b = []
        print("Теперь введите ограничения.")
        while True:
            coeffs, rhs = get_constraint(n)
            A.append(coeffs)
            b.append(rhs)
            cont = input("Добавить ещё одно ограничение? (y/n): ").strip().lower()
            if cont != 'y':
                break

        cont = input("Необходимо найти максимум целевой функции? (y/n): ").strip().lower()
        is_maximization = True if cont == 'y' else False

        epsi = int(input("Введите степень погрешности: "))

        available_methods = ['simplex', 'relaxation', 'column-generation', 'ADMM', 'mirror-descent']
        print("Доступные методы:")
        for i, method in enumerate(available_methods, start=1):
            print(f"{i}. {method}")
        method_indices = input("Введите номера выбранных методов через пробел: ").strip().split()
        methods = [available_methods[int(i) - 1] for i in method_indices if
                   i.isdigit() and 1 <= int(i) <= len(available_methods)]

        cont = input("Необходима визуализация? (y/n): ").strip().lower()
        visual = True if cont == 'y' else False

        cont = input("Необходим вывод в файл? (y/n): ").strip().lower()
        file_print = True if cont == 'y' else False

        cont = input("Необходим вывод в консоль? (y/n): ").strip().lower()
        console_print = True if cont == 'y' else False

        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        c = np.array(c, dtype=float)

        results = []
        for method in methods:
            print(f"\nЗапуск метода: {method}")
            res = solve_lp(method, c, A, b, 10 ** -epsi, is_maximization, console_print)
            results.append(res)
            print("-" * 100)

        if len(results) > 0:
            post_processing_linear_approximation_logs(results, visual, file_print)

    elif task_type == "2":
        print("Макет задачи: Задача об оптимизации параметров модели\n")
        print("Введите выражение функции y = f(x), где x — это переменная времени.")
        print("Поддерживаются функции:  exp, sin, cos, tg, ctg, abs, sqrt, pow")
        expr_input = input("y(x) = ").strip()

        try:
            param_map, num_params, model = parse_and_build_model(expr_input)
        except ValueError as e:
            print("Проблема в функции:", e)
            return

        t_data = np.linspace(0, 2 * np.pi, 10)
        try:
            x_true = param_map
            y_data = model(x_true, t_data)
        except Exception as e:
            print("Ошибка при вычислении значений функции:", e)

        delta = 3.0
        bounds = [(x - delta, x + delta) for x in x_true]
        x0 = np.ones(num_params) / num_params

        epsi = int(input("Введите степень погрешности: "))

        available_methods = ['gd', 'newton', 'sd', 'adam', 'nelder-mead']
        print("Доступные методы:")
        for i, method in enumerate(available_methods, start=1):
            print(f"{i}. {method}")
        method_indices = input("Введите номера выбранных методов через пробел: ").strip().split()
        methods = [available_methods[int(i) - 1] for i in method_indices if
                   i.isdigit() and 1 <= int(i) <= len(available_methods)]

        cont = input("Необходима визуализация? (y/n): ").strip().lower()
        visual = True if cont == 'y' else False

        cont = input("Необходим вывод в файл? (y/n): ").strip().lower()
        file_print = True if cont == 'y' else False

        cont = input("Необходим вывод в консоль? (y/n): ").strip().lower()
        console_print = True if cont == 'y' else False

        results = []
        for method in methods:
            print(f"\nЗапуск метода: {method}")
            res = solve_nlp(method, x0, t_data, y_data, model=model, bounds=bounds, epsilon=10 ** -epsi,
                            console_print=console_print)
            results.append(res)
            print("-" * 100)

        if len(results) > 0:
            post_processing_non_linear_approximation_logs(results, visual, file_print)

    elif task_type == "3":
        print("Макет задачи: Задача о рюкзаке (бинарная модификация)\n")
        n = int(input("Введите количество предметов: "))

        weights = get_float_list(f"Введите веса {n} предметов через пробел: ", n)

        values = get_float_list(f"Введите стоимости {n} предметов через пробел: ", n)

        capacity = int(input("Введите вместимость рюкзака: "))

        epsi = int(input("Введите степень погрешности: "))

        available_methods = ['branch_and_bound', 'gomory', 'cutting_planes', 'lagrangian_relaxation', 'Sherali-Adams-1']
        print("Доступные методы:")
        for i, method in enumerate(available_methods, start=1):
            print(f"{i}. {method}")
        method_indices = input("Введите номера выбранных методов через пробел: ").strip().split()
        methods = [available_methods[int(i) - 1] for i in method_indices if
                   i.isdigit() and 1 <= int(i) <= len(available_methods)]

        cont = input("Необходима визуализация? (y/n): ").strip().lower()
        visual = True if cont == 'y' else False

        cont = input("Необходим вывод в файл? (y/n): ").strip().lower()
        file_print = True if cont == 'y' else False

        cont = input("Необходим вывод в консоль? (y/n): ").strip().lower()
        console_print = True if cont == 'y' else False

        results = []
        for method in methods:
            print(f"\nЗапуск метода: {method}")
            res = solve_ip(method, weights, values, capacity, epsilon=10 ** -epsi, console_print=console_print)
            results.append(res)
            print("-" * 100)

        if len(results) > 0:
            post_processing_integer_approximation_logs(results, visual, file_print)


if __name__ == "__main__":
    main()

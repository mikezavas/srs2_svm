import numpy as np
from scipy import optimize
import math


# Задание функции и её производной
def f(x):
    """Исходная функция f(x) = e^x - 3x"""
    return np.exp(x) - 3 * x


def f_prime(x):
    """Производная f'(x) = e^x - 3"""
    return np.exp(x) - 3


def phi(x):
    """Итерационная функция φ(x) = e^x / 3"""
    return np.exp(x) / 3


def phi_prime(x):
    """Производная φ'(x) = e^x / 3"""
    return np.exp(x) / 3


# Табулирование функции на интервале [0, 1]
print("1. Табулирование функции на интервале [0, 1]")

a_interval = 0
b_interval = 1
step = 0.1

x_values = np.arange(a_interval, b_interval + step, step)
f_values = f(x_values)

print(f"{'x':<10} {'f(x)':<15}")
print("-" * 25)
for x, fx in zip(x_values, f_values):
    print(f"{x:<10.2f} {fx:<15.6f}")

# Определение участка с корнем
print("\n")
print("2. Определение участка с корнем")
for i in range(len(x_values) - 1):
    if f_values[i] * f_values[i + 1] < 0:
        print(f"Корень находится на интервале [{x_values[i]:.1f}, {x_values[i + 1]:.1f}]")
        print(f"f({x_values[i]:.1f}) = {f_values[i]:.6f}")
        print(f"f({x_values[i + 1]:.1f}) = {f_values[i + 1]:.6f}")
        a_root = x_values[i]
        b_root = x_values[i + 1]
        break


# Метод бисекции
def bisection_method(f, a, b, epsilon, max_iter=100):
    print(f"ε = {epsilon}")

    if f(a) * f(b) >= 0:
        print("Ошибка: f(a) и f(b) должны иметь разные знаки")
        return None

    print(f"{'k':<5} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<15} {'Ошибка':<12}")
    print("-" * 70)

    for k in range(max_iter):
        c = (a + b) / 2
        fc = f(c)
        error = abs(b - a) / 2

        print(f"{k:<5} {a:<12.6f} {b:<12.6f} {c:<12.6f} {fc:<15.6f} {error:<12.6f}")

        if error < epsilon:
            print(f"\nНайденный корень: x* = {c:.6f}")
            print(f"Число итераций: {k + 1}")
            print(f"Невязка |f(x*)| = {abs(f(c)):.6f}")
            return c, k + 1, abs(f(c))

        if f(a) * fc < 0:
            b = c
        else:
            a = c

    print("Достигнуто максимальное число итераций")
    return None


# Метод хорд
def chord_method(f, a, b, epsilon, max_iter=100):
    print(f"ε = {epsilon}")

    if f(a) * f(b) >= 0:
        print("Ошибка: f(a) и f(b) должны иметь разные знаки")
        return None

    print(f"{'k':<5} {'a':<12} {'b':<12} {'x_k':<12} {'f(x_k)':<15} {'Ошибка':<12}")
    print("-" * 68)

    x_prev = None

    for k in range(max_iter):
        fa = f(a)
        fb = f(b)
        x_k = a - fa * (b - a) / (fb - fa)
        fx_k = f(x_k)

        error = abs(x_k - x_prev) if x_prev is not None else abs(b - a)

        print(f"{k:<5} {a:<12.6f} {b:<12.6f} {x_k:<12.6f} {fx_k:<15.6f} {error:<12.6f}")

        if x_prev is not None and error < epsilon:
            print(f"\nНайденный корень: x* = {x_k:.6f}")
            print(f"Число итераций: {k + 1}")
            print(f"Невязка |f(x*)| = {abs(fx_k):.6f}")
            return x_k, k + 1, abs(fx_k)

        if fa * fx_k < 0:
            b = x_k
        else:
            a = x_k

        x_prev = x_k

    print("Достигнуто максимальное число итераций")
    return None


# Метод Ньютона
def newton_method(f, f_prime, x0, epsilon, max_iter=100):
    print(f"x₀ = {x0}, ε = {epsilon}")

    print(f"{'k':<5} {'x_k':<12} {'f(x_k)':<15} {'f\'(x_k)':<15} {'x_k+1':<12} {'Ошибка':<12}")
    print("-" * 71)

    x_k = x0

    for k in range(max_iter):
        fx_k = f(x_k)
        fpx_k = f_prime(x_k)

        if abs(fpx_k) < 1e-10:
            print("Ошибка: производная близка к нулю")
            return None

        x_next = x_k - fx_k / fpx_k
        error = abs(x_next - x_k)

        print(f"{k:<5} {x_k:<12.6f} {fx_k:<15.6f} {fpx_k:<15.6f} {x_next:<12.6f} {error:<12.6f}")

        if error < epsilon:
            print(f"\nНайденный корень: x* = {x_next:.6f}")
            print(f"Число итераций: {k + 1}")
            print(f"Невязка |f(x*)| = {abs(f(x_next)):.6f}")
            return x_next, k + 1, abs(f(x_next))

        x_k = x_next

    print("Достигнуто максимальное число итераций")
    return None


# Метод простой итерации
def simple_iteration_method(phi, f, x0, epsilon, max_iter=100):
    print(f"x₀ = {x0}, ε = {epsilon}")

    # Проверка условия сходимости
    print(f"Проверка условия сходимости |φ'(x)| < 1:")
    print(f"  |φ'({x0})| = {abs(phi_prime(x0)):.6f}")
    if abs(phi_prime(x0)) >= 1:
        print("Условие сходимости не выполняется!")
    else:
        print("Условие сходимости выполняется")

    print(f"\n{'k':<5} {'x_k':<12} {'φ(x_k)':<12} {'x_k+1':<12} {'Ошибка':<12} {'|f(x_k+1)|':<12}")
    print("-" * 67)

    x_k = x0

    for k in range(max_iter):
        phi_x_k = phi(x_k)
        x_next = phi_x_k
        error = abs(x_next - x_k)
        fx_next = abs(f(x_next))

        print(f"{k:<5} {x_k:<12.6f} {phi_x_k:<12.6f} {x_next:<12.6f} {error:<12.6f} {fx_next:<12.6f}")

        if error < epsilon:
            print(f"\nНайденный корень: x* = {x_next:.6f}")
            print(f"Число итераций: {k + 1}")
            print(f"Невязка |f(x*)| = {fx_next:.6f}")
            return x_next, k + 1, fx_next

        x_k = x_next

    print("Достигнуто максимальное число итераций")
    return None


# Библиотечные функции SciPy

def scipy_methods(f, a, b, x0):
    print("\n")
    print("     7. Библиотечные методы SciPy")

    # Метод бисекции
    root_bisect = optimize.bisect(f, a, b, xtol=1e-5)
    print(f"SciPy bisect: x* = {root_bisect:.6f}, f(x*) = {f(root_bisect):.6f}")

    # Метод Ньютона
    root_newton = optimize.newton(f, x0, fprime=f_prime, tol=1e-5)
    print(f"SciPy newton: x* = {root_newton:.6f}, f(x*) = {f(root_newton):.6f}")

    # Общий метод
    root_scalar = optimize.root_scalar(f, bracket=[a, b], method='brentq', xtol=1e-5)
    print(f"SciPy root_scalar: x* = {root_scalar.root:.6f}, f(x*) = {f(root_scalar.root):.6f}")

    return root_bisect, root_newton, root_scalar.root


# Основная программа

if __name__ == "__main__":
    print("\n")
    print("Вариант 3: f(x) = e^x - 3x = 0, интервал [0, 1]")

    # Точности
    epsilon_1 = 0.001
    epsilon_2 = 0.00001

    # Начальные приближения
    x0_1 = 0.5
    x0_2 = 0.8

    # Метод бисекции
    print("\n")
    print(f"    3. Метод бисекции")
    result_bisect_1 = bisection_method(f, 0, 1, epsilon_1)
    print("\n")
    result_bisect_2 = bisection_method(f, 0, 1, epsilon_2)

    # Метод хорд
    print("\n")
    print(f"    4. Метод хорд")
    result_chord_1 = chord_method(f, 0, 1, epsilon_1)
    print("\n")
    result_chord_2 = chord_method(f, 0, 1, epsilon_2)

    # Метод Ньютона (два начальных приближения)
    print("\n")
    print(f"    5. Метод Ньютона")
    result_newton_1_1 = newton_method(f, f_prime, x0_1, epsilon_1)
    print("\n")
    result_newton_1_2 = newton_method(f, f_prime, x0_2, epsilon_1)
    print("\n")
    result_newton_2_1 = newton_method(f, f_prime, x0_1, epsilon_2)
    print("\n")
    result_newton_2_2 = newton_method(f, f_prime, x0_2, epsilon_2)

    # Метод простой итерации (два начальных приближения)
    print("\n")
    print(f"    6. Метод простой итерации")
    result_iter_1_1 = simple_iteration_method(phi, f, x0_1, epsilon_1)
    print("\n")
    result_iter_1_2 = simple_iteration_method(phi, f, x0_2, epsilon_1)
    print("\n")
    result_iter_2_1 = simple_iteration_method(phi, f, x0_1, epsilon_2)
    print("\n")
    result_iter_2_2 = simple_iteration_method(phi, f, x0_2, epsilon_2)

    # Библиотечные методы
    scipy_results = scipy_methods(f, 0, 1, x0_1)

    # Итоговая таблица сравнения
    print("\n")
    print("     8. Итоговая таблица сравнения")

    print(f"{'Метод':<20} {'ε':<10} {'x₀':<10} {'x*':<12} {'Итерации':<10} {'|f(x*)|':<12} {'SciPy':<12} {'Excel':<12}")
    print("-" * 98)

    # Данные из Excel (из предыдущих расчетов)
    excel_results = {
        'bisect_1': (0.6181641, 10, 0.0010261),
        'bisect_2': (0.6190567, 17, 0.0000052),
        'chord_1': (0.6191854, 7, 0.0001418),
        'chord_2': (0.6190644, 10, 0.0000036),
        'newton_1_1': (0.6190613, 3, 0.0000000039),
        'newton_1_2': (0.619013, 4, 0.000000000003),
        'newton_2_1': (0.6190613, 4, 0.0000000000000002),
        'newton_2_2': (0.6190613, 4, 0.000000000003),
        'iter_1_1': (0.6176860, 9, 0.0015735),
        'iter_1_2': (0.6202664, 11, 0.0013759),
        'iter_2_1': (0.6190499, 19, 0.0000130),
        'iter_2_2': (0.6190713, 21, 0.0000114)
    }


    def format_result(result, excel_key):
        if result:
            x_star, iterations, f_val = result
            excel_x, excel_iter, excel_f = excel_results[excel_key]
            return f"{x_star:<12.6f} {iterations:<10} {f_val:<12.6f} {scipy_results[0]:<12.6f} {excel_x:<12.6f}"
        return "N/A"


    print(f"{'Бисекция':<20} {'10^-3':<10} {'-':<10} {format_result(result_bisect_1, 'bisect_1')}")
    print(f"{'Бисекция':<20} {'10^-5':<10} {'-':<10} {format_result(result_bisect_2, 'bisect_2')}")
    print(f"{'Хорды':<20} {'10^-3':<10} {'-':<10} {format_result(result_chord_1, 'chord_1')}")
    print(f"{'Хорды':<20} {'10^-5':<10} {'-':<10} {format_result(result_chord_2, 'chord_2')}")
    print(f"{'Ньютон':<20} {'10^-3':<10} {'0.5':<10} {format_result(result_newton_1_1, 'newton_1_1')}")
    print(f"{'Ньютон':<20} {'10^-3':<10} {'0.8':<10} {format_result(result_newton_1_2, 'newton_1_2')}")
    print(f"{'Ньютон':<20} {'10^-5':<10} {'0.5':<10} {format_result(result_newton_2_1, 'newton_2_1')}")
    print(f"{'Ньютон':<20} {'10^-5':<10} {'0.8':<10} {format_result(result_newton_2_2, 'newton_2_2')}")
    print(f"{'Простая итерация':<20} {'10^-3':<10} {'0.5':<10} {format_result(result_iter_1_1, 'iter_1_1')}")
    print(f"{'Простая итерация':<20} {'10^-3':<10} {'0.8':<10} {format_result(result_iter_1_2, 'iter_1_2')}")
    print(f"{'Простая итерация':<20} {'10^-5':<10} {'0.5':<10} {format_result(result_iter_2_1, 'iter_2_1')}")
    print(f"{'Простая итерация':<20} {'10^-5':<10} {'0.8':<10} {format_result(result_iter_2_2, 'iter_2_2')}")

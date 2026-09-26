import struct
import numpy as np
import matplotlib.pyplot as plt

def mle_mean(X):
    n = X.shape[0]
    total = np.zeros(X.shape[1])
    for i in range(n):
        total = total + X[i]
    return total / n

def mle_cov(X, mu):
    n = X.shape[0]; d = X.shape[1]
    total = np.zeros((d, d))
    for i in range(n):
        diff = (X[i] - mu).reshape(d, 1)
        total = total + diff @ diff.T
    return total / n

def format_projection_formula(w, feature_names):
    terms = [f'{w[i]:.5f}*{feature_names[i]}' for i in range(len(w))]
    return 'Y(x) = ' + ' + '.join(terms)

def build_D(class_a, class_b, B):
    d_local = class_a.shape[1]
    D = np.zeros((d_local, d_local))
    for x in class_a:
        for y in class_b:
            diff = (x - y).reshape(d_local, 1)
            dist = np.linalg.norm(x - y)
            D = D + (diff @ diff.T) / (dist ** B)
    return D
def project(X, w1, w2):
    y1 = X @ w1
    y2 = X @ w2
    return np.column_stack((y1, y2))
def top_eigenvector(D):
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    idx = np.argmax(eigenvalues)
    v = eigenvectors[:, idx]
    return v / np.linalg.norm(v)

if __name__ == '__main__':
    with open('IRIS.DAT', 'rb') as f:
        raw_bytes = f.read()
    n_floats = len(raw_bytes) // 4
    flat_values = struct.unpack('<' + 'f' * n_floats, raw_bytes)
    data = np.array(flat_values, dtype=np.float32).reshape(150, 4)
    se, ve, vi = data[0:50], data[50:100], data[100:150]

    se_x2_reversed = se[:, 1][::-1]
    se_x3_reversed = se[:, 2][::-1]
    se_x4 = se[:, 3]

    ve_x2_reversed = ve[:, 1][::-1]
    ve_x3_reversed = ve[:, 2][::-1]
    ve_x4 = ve[:, 3]

    Vi = np.column_stack((vi[:, 1][::-1], vi[:, 2][::-1], vi[:, 3]))

    Se = np.column_stack((se_x2_reversed, se_x3_reversed, se_x4))
    Ve = np.column_stack((ve_x2_reversed, ve_x3_reversed, ve_x4))

    n_se, n_ve = Se.shape[0], Ve.shape[0]

    # =====================================================================
    # ЛР2, ЗАДАНИЕ 1, МЕТОД 1: Линейный дискриминант Фишера (ЛДФ)
    # =====================================================================
    print('--- Метод 1: Линейный дискриминант Фишера (ЛДФ) ---')

    m1 = mle_mean(Se)   # среднее класса Se
    m2 = mle_mean(Ve)   # среднее класса Ve
    print('m1 (среднее Se) =', m1)
    print('m2 (среднее Ve) =', m2)

    sigma_se = mle_cov(Se, m1)
    sigma_ve = mle_cov(Ve, m2)

    # Sw = sum_{x in Se}(x-m1)(x-m1)^T + sum_{x in Ve}(x-m2)(x-m2)^T  (БЕЗ усреднения)
    Sw = n_se * sigma_se + n_ve * sigma_ve
    print('Sw =')
    print(Sw)

    Sw_inv = np.linalg.inv(Sw)
    print('Sw^-1 =')
    print(Sw_inv)

    w_fisher = Sw_inv @ (m1 - m2)
    print('w (вектор ЛДФ, до нормировки) =', w_fisher)

    # Нормируем до единичной длины - чтобы координата проекции
    # соответствовала реальной длине проекции точки на прямую в исходном пространстве
    w_fisher_unit = w_fisher / np.linalg.norm(w_fisher)
    feature_names = ['x2', 'x3', 'x4']
    print('w (нормированный, ||w||=1) =', w_fisher_unit)
    print(format_projection_formula(w_fisher_unit, feature_names))

    proj_se_fisher = Se @ w_fisher_unit
    proj_ve_fisher = Ve @ w_fisher_unit

    se_min_f, se_max_f = proj_se_fisher.min(), proj_se_fisher.max()
    ve_min_f, ve_max_f = proj_ve_fisher.min(), proj_ve_fisher.max()
    print(f'Se: мин={se_min_f:.6f}, макс={se_max_f:.6f}')
    print(f'Ve: мин={ve_min_f:.6f}, макс={ve_max_f:.6f}')

    if se_max_f < ve_min_f:
        rho_fisher = ve_min_f - se_max_f
    elif ve_max_f < se_min_f:
        rho_fisher = se_min_f - ve_max_f
    else:
        rho_fisher = 0.0
        print('ВНИМАНИЕ: проекции пересекаются')
    print(f'Минимальное расстояние между проекциями классов (ЛДФ): {rho_fisher:.5f}')

    # =====================================================================
    # ЛР2, ЗАДАНИЕ 1, МЕТОД 2: МППРП
    # =====================================================================
    print()
    print('--- Метод 2: МППРП ---')

    B = 2

    d = Se.shape[1]
    D = np.zeros((d, d))
    for x in Se:
        for y in Ve:
            diff = (x - y).reshape(d, 1)
            dist = np.linalg.norm(x - y)
            D = D + (diff @ diff.T) / (dist ** B)

    print('D =')
    print(D)

    eigenvalues, eigenvectors = np.linalg.eigh(D)
    print('Собственные значения D:', eigenvalues)

    max_idx = np.argmax(eigenvalues)
    lambda_max = eigenvalues[max_idx]
    v_raw = eigenvectors[:, max_idx]
    v_unit = v_raw / np.linalg.norm(v_raw)

    print('lambda_max =', lambda_max)
    print('d (нормированный собственный вектор) =', v_unit)
    print(format_projection_formula(v_unit, feature_names))

    proj_se_mmpr = Se @ v_unit
    proj_ve_mmpr = Ve @ v_unit

    se_min_m, se_max_m = proj_se_mmpr.min(), proj_se_mmpr.max()
    ve_min_m, ve_max_m = proj_ve_mmpr.min(), proj_ve_mmpr.max()
    print(f'Se: мин={se_min_m:.6f}, макс={se_max_m:.6f}')
    print(f'Ve: мин={ve_min_m:.6f}, макс={ve_max_m:.6f}')

    if se_max_m < ve_min_m:
        rho_mmpr = ve_min_m - se_max_m
    elif ve_max_m < se_min_m:
        rho_mmpr = se_min_m - ve_max_m
    else:
        rho_mmpr = 0.0
        print('ВНИМАНИЕ: проекции пересекаются')
    print(f'Минимальное расстояние между проекциями классов (МППРП): {rho_mmpr:.5f}')

    # =====================================================================
    # ИТОГОВОЕ СРАВНЕНИЕ
    # =====================================================================
    print()
    print(f'Расстояние (ЛДФ)   = {rho_fisher:.5f}')
    print(f'Расстояние (МППРП) = {rho_mmpr:.5f}')
    if rho_fisher > rho_mmpr:
        print('Вывод: Расстояние, полученное методом ЛДФ, лучше, чем расстояние, '
              'полученное методом МППРП. Метод ЛДФ лучше разделяет эти 2 класса.')
    else:
        print('Вывод: Расстояние, полученное методом МППРП, лучше, чем расстояние, '
              'полученное методом ЛДФ. Метод МППРП лучше разделяет эти 2 класса.')

    classes = [Se, Ve, Vi]
    class_names = ['Se', 'Ve', 'Vi']
    N_list = [X.shape[0] for X in classes]
    N_total = sum(N_list)
    d = Se.shape[1]

    print('=' * 70)
    print('ЛР2, ЗАДАНИЕ 2: Множественный дискриминантный анализ (МДА)')
    print('=' * 70)
    m_list = [mle_mean(X) for X in classes]
    for name, m, N in zip(class_names, m_list, N_list):
        print(f'm_{name} = {m}   (N_{name}={N})')

    m_overall = np.zeros(d)
    for N, m in zip(N_list, m_list):
        m_overall = m_overall + N * m
    m_overall = m_overall / N_total
    print('m (общее среднее) =', m_overall)
    # =====================================================================
    # ЧАСТЬ 2. Sw (внутриклассовый разброс) и Sb (межклассовый разброс)
    # =====================================================================
    Sw = np.zeros((d, d))
    for X, m in zip(classes, m_list):
        for x in X:
            diff = (x - m).reshape(d, 1)
            Sw = Sw + diff @ diff.T
    print()
    print('Sw =')
    print(Sw)

    Sb = np.zeros((d, d))
    for N, m in zip(N_list, m_list):
        diff = (m - m_overall).reshape(d, 1)
        Sb = Sb + N * (diff @ diff.T)
    print('Sb =')
    print(Sb)

    Sw_inv = np.linalg.inv(Sw)
    print('Sw^-1 =')
    print(Sw_inv)

    SwInvSb = Sw_inv @ Sb
    print('Sw^-1 * Sb =')
    print(SwInvSb)
    # =====================================================================
    # ЧАСТЬ 3. Собственные значения/векторы Sw^-1*Sb (матрица несимметрична!)
    # =====================================================================
    eigenvalues, eigenvectors = np.linalg.eig(SwInvSb)
    # Приводим к вещественным числам (мнимая часть должна быть ~0 из-за
    # свойств обобщённой задачи на собственные значения с Sw>0)
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    print()
    print('Собственные значения Sw^-1*Sb:', eigenvalues)

    order = np.argsort(eigenvalues)[::-1]  # сортировка по убыванию
    eigenvalues_sorted = eigenvalues[order]
    eigenvectors_sorted = eigenvectors[:, order]

    print('Собственные значения (по убыванию):', eigenvalues_sorted)
    for i in range(d):
        print(f'  собственный вектор {i + 1} (lambda={eigenvalues_sorted[i]:.5f}):',
              eigenvectors_sorted[:, i])

    # Берём 2 главных собственных вектора (нормируем до единичной длины)
    w1 = eigenvectors_sorted[:, 0] / np.linalg.norm(eigenvectors_sorted[:, 0])
    w2 = eigenvectors_sorted[:, 1] / np.linalg.norm(eigenvectors_sorted[:, 1])
    print()
    print('w1 =', w1)
    print('w2 =', w2)
    print(f'y1(x) = {w1[0]:.5f}*x2 + {w1[1]:.5f}*x3 + {w1[2]:.5f}*x4')
    print(f'y2(x) = {w2[0]:.5f}*x2 + {w2[1]:.5f}*x3 + {w2[2]:.5f}*x4')


    # =====================================================================
    # ЧАСТЬ 4. Проекция всех классов на плоскость (y1, y2) - метод МДА
    # =====================================================================

    Se_proj_mda = project(Se, w1, w2)
    Ve_proj_mda = project(Ve, w1, w2)
    Vi_proj_mda = project(Vi, w1, w2)

    centers_mda = [mle_mean(Se_proj_mda), mle_mean(Ve_proj_mda), mle_mean(Vi_proj_mda)]


    def nearest_centroid_errors(projections, centers):
        """Считает ошибки классификации по правилу ближайшего центра (Евклидово расстояние)."""
        total_points = 0
        total_errors = 0
        for true_idx, proj in enumerate(projections):
            for point in proj:
                dists = [np.linalg.norm(point - c) for c in centers]
                predicted_idx = int(np.argmin(dists))
                total_points += 1
                if predicted_idx != true_idx:
                    total_errors += 1
        return total_errors, total_points


    errors_mda, total_mda = nearest_centroid_errors(
        [Se_proj_mda, Ve_proj_mda, Vi_proj_mda], centers_mda)
    error_rate_mda = errors_mda / total_mda
    print()
    print(f'МДА: ошибок = {errors_mda} из {total_mda}, вероятность ошибки = {error_rate_mda:.4f}')

    # =====================================================================
    # ЧАСТЬ 5. МППРП для 3 классов - последовательные пары (Se,Ve) и (Ve,Vi)
    # =====================================================================
    print()
    print('--- МППРП для 3 классов (пары Se-Ve и Ve-Vi) ---')

    B = 2

    D1 = build_D(Se, Ve, B)
    v1 = top_eigenvector(D1)
    print('D1 (Se-Ve) =')
    print(D1)
    eigenvalues1, eigenvectors1 = np.linalg.eig(D1)

    print('Собственные значения D1:', np.real(eigenvalues1))
    print('v1 =', v1)


    D2 = build_D(Ve, Vi, B)
    v2 = top_eigenvector(D2)
    print('D2 (Ve-Vi) =')
    print(D2)
    eigenvalues2, eigenvectors2 = np.linalg.eig(D2)

    print('Собственные значения D1:', np.real(eigenvalues2))

    print('v2 =', v2)

    print(f'y1(x) = {v1[0]:.5f}*x2 + {v1[1]:.5f}*x3 + {v1[2]:.5f}*x4')
    print(f'y2(x) = {v2[0]:.5f}*x2 + {v2[1]:.5f}*x3 + {v2[2]:.5f}*x4')

    Se_proj_mmpr = project(Se, v1, v2)
    Ve_proj_mmpr = project(Ve, v1, v2)
    Vi_proj_mmpr = project(Vi, v1, v2)

    centers_mmpr = [mle_mean(Se_proj_mmpr), mle_mean(Ve_proj_mmpr), mle_mean(Vi_proj_mmpr)]

    errors_mmpr, total_mmpr = nearest_centroid_errors(
        [Se_proj_mmpr, Ve_proj_mmpr, Vi_proj_mmpr], centers_mmpr)
    error_rate_mmpr = errors_mmpr / total_mmpr
    print(f'МППРП: ошибок = {errors_mmpr} из {total_mmpr}, вероятность ошибки = {error_rate_mmpr:.4f}')

    # =====================================================================
    # ЧАСТЬ 6. Итоговое сравнение
    # =====================================================================
    print()
    print(f'Вероятность ошибки (МДА)   = {error_rate_mda:.4f}')
    print(f'Вероятность ошибки (МППРП) = {error_rate_mmpr:.4f}')
    if error_rate_mda < error_rate_mmpr:
        print('Вывод: МДА показал лучший результат - меньше ошибок классификации.')
    elif error_rate_mmpr < error_rate_mda:
        print('Вывод: МППРП показал лучший результат - меньше ошибок классификации.')
    else:
        print('Вывод: методы дали ОДИНАКОВУЮ вероятность ошибки - '
              'на этих данных ни один не показал явного преимущества.')


    def draw_projection(ax, proj_list, centers, title):
        colors = ['#1f77b4', '#2ca02c', '#ffdd00']
        names = ['Se', 'Ve', 'Vi']
        for proj, color, name in zip(proj_list, colors, names):
            ax.scatter(proj[:, 0], proj[:, 1], color=color, s=25,
                       edgecolor='black', linewidth=0.3, label=name)
        for c, color in zip(centers, colors):
            ax.scatter(*c, color=color, marker='X', s=200, edgecolor='black', linewidth=1.2)

        # перпендикулярные биссектрисы между соседними центрами (Se-Ve, Ve-Vi)
        for (ca, cb) in [(centers[0], centers[1]), (centers[1], centers[2])]:
            mid = (ca + cb) / 2
            direction = cb - ca
            perp = np.array([-direction[1], direction[0]])
            perp = perp / np.linalg.norm(perp)
            p1 = mid - perp * 5
            p2 = mid + perp * 5
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='red', linewidth=1.5)

        ax.set_title(title)
        ax.set_xlabel('y1')
        ax.set_ylabel('y2')
        ax.legend()
        ax.grid(True, alpha=0.3)


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    draw_projection(axes[0], [Se_proj_mda, Ve_proj_mda, Vi_proj_mda], centers_mda,
                    f'МДА (ошибка={error_rate_mda:.3f})')
    draw_projection(axes[1], [Se_proj_mmpr, Ve_proj_mmpr, Vi_proj_mmpr], centers_mmpr,
                    f'МППРП (ошибка={error_rate_mmpr:.3f})')
    plt.tight_layout()
    plt.savefig('lab2_task2.png', dpi=150)
    print()
    print('График сохранён')

    #построение собственных решающих правил на глаз в разных плоскостях
    #(x2, x3)
    Se2 = np.column_stack((se_x2_reversed, se_x3_reversed))  # только (x2, x3) для этой картинки
    Ve2 = np.column_stack((ve_x2_reversed, ve_x3_reversed))
    Vi2 = np.column_stack((vi[:, 1][::-1], vi[:, 2][::-1]))


    def count_errors(class_a, class_b, a, b, c):
        """Прямая: a*x2 + b*x3 + c = 0.
        Сторону класса определяем автоматически по знаку в его центре -
        не предполагаем заранее, кто "положительный", кто "отрицательный"."""
        mean_a = class_a.mean(axis=0)
        mean_b = class_b.mean(axis=0)
        sign_a = np.sign(a * mean_a[0] + b * mean_a[1] + c)
        sign_b = np.sign(b * mean_b[1] + a * mean_b[0] + c)

        errors = 0
        for p in class_a:
            if np.sign(a * p[0] + b * p[1] + c) != sign_a:
                errors += 1
        for p in class_b:
            if np.sign(a * p[0] + b * p[1] + c) != sign_b:
                errors += 1
        return errors


    def draw_pair(ax, class_a, class_b, name_a, name_b, a, b, c, color_a, color_b, cor_name1, cor_name2):
        ax.scatter(class_a[:, 0], class_a[:, 1], color=color_a, s=35, edgecolor='black',
                   linewidth=0.4, label=name_a)
        ax.scatter(class_b[:, 0], class_b[:, 1], color=color_b, s=35, edgecolor='black',
                   linewidth=0.4, label=name_b)

        x2_range = np.linspace(min(class_a[:, 0].min(), class_b[:, 0].min()) - 0.3,
                               max(class_a[:, 0].max(), class_b[:, 0].max()) + 0.3, 100)
        if abs(b) > 1e-9:
            x3_line = (-a * x2_range - c) / b
            ax.plot(x2_range, x3_line, color='red', linewidth=2,
                    label=f'{a:.3f}·{cor_name1} + {b:.3f}·{cor_name2} + {c:.3f} = 0')
        errors = count_errors(class_a, class_b, a, b, c)
        ax.set_title(f'{name_a} vs {name_b}  (ошибок: {errors})')
        ax.set_xlabel(cor_name1)
        ax.set_ylabel(cor_name2)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return errors


    # --- Se vs Ve: разделены по x3 с большим запасом (Se x3<=1.9, Ve x3>=3.0) ---
    # Прямая: x3 = 2.4 (горизонтальная, примерно посередине зазора 1.9-3.0)
    a1, b1, c1 = 0, 1, -2.4  # y - 2.4 = 0  =>  x3 = 2.4

    # --- Ve vs Vi: пересекаются - линия подобрана перебором угла/сдвига,
    #     минимизирующим число ошибок (это эквивалент "на глаз, но точнее") ---
    a2, b2, c2 = 0.2250, 0.9744, -5.3500

    # --- Se vs Vi: разделены с большим запасом (Se x3<=1.9, Vi x3>=4.5) ---
    a3, b3, c3 = 0, 1, -3.2  # x3 = 3.2

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    e1 = draw_pair(axes[0], Se2, Ve2, 'Se', 'Ve', a1, b1, c1, '#d62728', '#1f77b4', 'x2', 'x3')
    e2 = draw_pair(axes[1], Ve2, Vi2, 'Ve', 'Vi', a2, b2, c2, '#1f77b4', '#ffcc00', 'x2', 'x3')
    e3 = draw_pair(axes[2], Se2, Vi2, 'Se', 'Vi', a3, b3, c3, '#d62728', '#ffcc00', 'x2', 'x3')

    plt.tight_layout()
    plt.savefig('lab2_task2_pairs.png', dpi=150)


    #(x3, x4)
    Se2 = np.column_stack((se_x3_reversed, se_x4)) # только (x2, x3) для этой картинки
    Ve2 = np.column_stack((ve_x3_reversed, ve_x4))
    Vi2 = np.column_stack((vi[:, 2][::-1], vi[:, 3]))

    a1, b1, c1 = 0.5, 1, -2.15
    a2, b2, c2 = 0.52, 1, -4.28
    a3, b3, c3 = 0.5, 1, -2.55

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    e1 = draw_pair(axes[0], Se2, Ve2, 'Se', 'Ve', a1, b1, c1, '#d62728', '#1f77b4', 'x3', 'x4')
    e2 = draw_pair(axes[1], Ve2, Vi2, 'Ve', 'Vi', a2, b2, c2, '#1f77b4', '#ffcc00', 'x3', 'x4')
    e3 = draw_pair(axes[2], Se2, Vi2, 'Se', 'Vi', a3, b3, c3, '#d62728', '#ffcc00', 'x3', 'x4')

    plt.tight_layout()
    plt.savefig('lab2_task2_pairsX3X4.png', dpi=150)


    #(x2, x4)

    Se2 = np.column_stack((se_x2_reversed, se_x4))  # только (x2, x3) для этой картинки
    Ve2 = np.column_stack((ve_x2_reversed, ve_x4))
    Vi2 = np.column_stack((vi[:, 1][::-1], vi[:, 3]))

    a1, b1, c1 = 0.14, -1, 0.31
    a2, b2, c2 = 0.5, 1, -3
    a3, b3, c3 = 0.2, -1, 0.45

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    e1 = draw_pair(axes[0], Se2, Ve2, 'Se', 'Ve', a1, b1, c1, '#d62728', '#1f77b4', 'x2', 'x4')
    e2 = draw_pair(axes[1], Ve2, Vi2, 'Ve', 'Vi', a2, b2, c2, '#1f77b4', '#ffcc00', 'x2', 'x4')
    e3 = draw_pair(axes[2], Se2, Vi2, 'Se', 'Vi', a3, b3, c3, '#d62728', '#ffcc00', 'x2', 'x4')

    plt.tight_layout()
    plt.savefig('lab2_task2_pairsX2X4.png', dpi=150)








# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from experiment import Experiment


def get_exp_from_dir(dir_path):
    if os.path.isdir(dir_path):
        metadata_path = os.path.join(dir_path, 'metadata.json')
        
        if os.path.exists(metadata_path):
            return Experiment(metadata_path)
        
        raise FileNotFoundError(f"Файл '{metadata_path}' не найден")
    raise FileNotFoundError(f"Директория '{dir_path}' не найдена")


def cleanup_checkpoints_all(experiments, number_of_best_epoch):
    for exp in experiments:
        print(exp)
        try:
            exp.cleanup_checkpoints(number_of_best_epoch)
        except ValueError as e:
            print(e)
        print('\n')


# +
experiments_basepath = '/app/pgrachev/moco/experiments/'

experiments = []
for dir_name in os.listdir(experiments_basepath):
    try:
        dir_path = os.path.join(experiments_basepath, dir_name)
        exp = get_exp_from_dir(dir_path)
    except FileNotFoundError as e:
        print(e)
    else:
        experiments.append(exp)

experiments = sorted(experiments, key=lambda e: e.exp_number)

# +
# cleanup_checkpoints_all(experiments, 3)

# +
# exp_num_to_plots = [12, 18, 20, 13, 17, 19]
exp_num_to_plots = [17, 19, 20]
# exp_num_to_plots = [11]

exp_to_plots = []
for exp in experiments:
    if exp.exp_number in exp_num_to_plots:
        exp_to_plots.append(exp)

# +
plt.figure(figsize=(10, 6))
for exp in exp_to_plots:
    epochs = exp.epochs_data
    loss_values = exp.loss_values

    # Создаем DataFrame для удобства вычисления скользящего среднего
    data = pd.DataFrame({'Epoch': epochs, 'Loss': loss_values})

    # Вычисляем скользящее среднее с окном 5 (можно изменить по необходимости)
    data['Rolling_Loss'] = data['Loss'].rolling(window=3).mean()

    # Исключаем первую строку, так как она может содержать NaN значения из-за скользящего среднего
    plt.plot(data['Epoch'], data['Rolling_Loss'], marker='o', linestyle='-', markersize=2, linewidth=1, label=f'{exp}')

plt.xlim(0, 50)
plt.title('Loss values per Epochs (Rolling Average)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=1)
plt.tight_layout()

plt.show()

# +
last_exp = experiments[-1]
plt.figure(figsize=(12, 8))


# Подграфик 1: Зависимость лосса от эпох
plt.subplot(2, 1, 1)  # (строки, столбцы, индекс подграфика)
plt.plot(last_exp.epochs_data, last_exp.loss_values, color='green', marker='o', linestyle='-', markersize=3, linewidth=1)
plt.xlim(0, 50)
plt.title(f'Loss per Epoch {last_exp}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, which='both', linestyle='--', linewidth=2)

# Подграфик 2: Зависимость эпох от времени
plt.subplot(2, 1, 2)
plt.plot(last_exp.hours_timestamp, last_exp.epochs_data, marker='x', linestyle='-', color='red', markersize=3, linewidth=1)
plt.xlim(0, 10)
plt.ylim(0, 60)
plt.title(f'Epochs over Time {last_exp}')
plt.xlabel('Time (hours)')
plt.ylabel('Epoch')
plt.grid(True, which='both', linestyle='--', linewidth=2)

# Настраиваем макет, чтобы подграфики не перекрывались
plt.tight_layout()

# Показываем графики
plt.show()

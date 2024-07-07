# -*- coding: utf-8 -*-
import json
import os


class Experiment:
    
    def __init__(self, metadata_path: str):
        # Загрузка метаданных из JSON файла
        with open(metadata_path, 'r') as file:
            metadata = json.load(file)
        
        # Динамическое создание атрибутов на основе ключей JSON
        for key, value in metadata.items():
            setattr(self, key, value)
        
        # Предполагаем, что номер эксперимента содержится в пути
        self.exp_number = int(self.experiment_path.split('exp')[-1])
        
        self.epochs_data = []
        self.loss_values = []
        self.hours_timestamp = []
        self.shared_data = []
        self._load_experiment_csv(self.experiment_path + '/loss_per_epochs.csv')
    
    def _load_experiment_csv(self, filepath: str):
        with open(filepath, 'r') as file:
            headers = next(file)
            for line in file:
                line = line.strip().split(',')

                epoch = int(line[0])
                loss = round(float(line[1]), 4)
                self.epochs_data.append(epoch)
                self.loss_values.append(loss)
                
                hour = None
                if len(line) == 3:
                    hour = round(int(line[2])/3600, 2)
                    self.hours_timestamp.append(hour)

                self.shared_data.append((epoch, loss, hour))
    
    def get_min_losses(self, n):
        """
        Возвращает список размером n, по возрастанию для самых минимальных значений лосса,
        а также номер эпохи, на котором этот лосс достигнут.
        """
        sorted_pairs = sorted(self.shared_data, key=lambda x: x[1])
        min_pairs = sorted_pairs[:n]
        return [(epoch, loss) for epoch, loss, _ in min_pairs]
    
    def cleanup_checkpoints(self, number_of_best_epoch):
        """
        Удаляет ненужные чекпоинты в folder_path и оставляет только number_of_best_epoch,
        на которых наименьшее значение лосса
        """
        folder_path = self.experiment_path + '/checkpoints'
        # Получаем список эпох для сохранения
        epochs_to_keep = [epoch for epoch, _ in self.get_min_losses(number_of_best_epoch)]
        print(f'Оставлены лучшие {number_of_best_epoch} эпох, c номерами:', epochs_to_keep)
        # Получаем список всех файлов в папке
        all_files = sorted(os.listdir(folder_path))
        
        all_files_epoch_nums = [int(file_name.split('_ep')[1].split('.')[0]) for file_name in all_files]
        for epoch_num in epochs_to_keep:
            if epoch_num not in all_files_epoch_nums:
                raise ValueError(f'Файла с номером эпохи {epoch_num} нет в директории')
                
        # Перебираем все файлы и удаляем ненужные
        for file_name in all_files:
            # Извлекаем номер эпохи из имени файла
            epoch_number = int(file_name.split('_ep')[1].split('.')[0])
            # Удаляем файл, если его эпоха не в списке для сохранения
            if epoch_number not in epochs_to_keep:
                file_path = os.path.join(folder_path, file_name)
                os.remove(file_path)
                print(f"Файл {file_name} удален.")
        print("Операция по очистке завершена.")
                
    def __str__(self):
        return f'Exp{self.exp_number:<2} {self.model_name}(batch_size={self.dataset_params["batch_size"]:<2}, pretrained={str(self.pretrained):<5})'

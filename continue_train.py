# -*- coding: utf-8 -*-
import os 
import csv
import json
import time
import torchvision
import torch
from torch.utils.tensorboard import SummaryWriter
from moco.loader import SpectrogramDataset
from moco.builder import MoCoV3, SiameseNet, make_base_encoder
from utils.multithread_utils import ThreadedDataLoader
from utils.save_utils import save_loss, save_checkpoint, save_metadata
from train import train_single_epoch


def train_model(
    model: torch.nn.Module,
    optimizer,
    dataloader,
    start_epoch,
    epochs,
    writer,
    loss_csv_filepath,
    checkpoints_dir,
    add_time,
    device
):
    
    model.train()
    start_time = time.time()
    for epoch in range(start_epoch, epochs + 1):

        print(f'Epoch {epoch} / {epochs}')
        train_loss = train_single_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            writer=writer,
            current_epoch=epoch,
            device=device
        )
        
        time_cutoff = add_time + time.time() - start_time
        save_loss(loss_csv_filepath, train_loss, epoch, time_cutoff)
        writer.add_scalar('train_loss', train_loss, epoch)
        if epoch >= 10:
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_dir=checkpoints_dir)


if __name__ == '__main__':
    EXP_PATH = '/app/pgrachev/moco/experiments/exp019'
    MODEL_FILENAME = "model_ep0185.pt"
    
    with open(EXP_PATH + '/metadata.json', 'r', encoding='utf-8') as file:
        metadata = json.load(file)
    
    # Узнаем на какой эпохе остановилось обучение и сколько времени это заняло
    # Нужна только последняя строчка в файле
    with open(EXP_PATH + '/loss_per_epochs.csv', 'r', encoding='utf-8') as file:
        for line in csv.DictReader(file):
            start_epoch = int(line['Epoch'])
            add_time = int(line['Time_cutoff'])
    
    dataloader = ThreadedDataLoader(
        dataset=SpectrogramDataset(**metadata['dataset_params']),
        **metadata['dataloader_params']
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    # Инициализируем модель обученными весами
    # model = MoCoV3(base_encoder=torchvision.models.resnet50, m=metadata['moco_momentum'])
    model = SiameseNet(base_encoder=torchvision.models.resnet50)
    
    checkpoint_path = os.path.join(metadata["experiment_path"], "checkpoints", MODEL_FILENAME)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=metadata['learning_rate'], momentum=metadata['sgd_momentum'])
    
    # Запишем измененную metadata
    metadata['epochs'] += 100
    save_metadata(metadata)
    
    # Прозводим старт потоков, выполняем функцию обучения, завершаем потоки
    # --------------------------------------------------------------------------------------------------------------
    dataloader.start_threads()
    
    train_model(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        start_epoch=start_epoch + 1,
        epochs=metadata['epochs'],
        writer=SummaryWriter(log_dir=os.path.join(metadata['experiment_path'], 'tb_logs')),
        loss_csv_filepath=os.path.join(metadata['experiment_path'], 'loss_per_epochs.csv'),
        checkpoints_dir=os.path.join(metadata['experiment_path'], 'checkpoints'),
        add_time=add_time,
        device=device
    )
    
    dataloader.stop_threads()

# -*- coding: utf-8 -*-
import librosa
import librosa.display
import os
import numpy as np
import matplotlib.pyplot as plt


def get_audio_sample(audio, sr: float, start: float, duration: float):
    return audio[start * sr:(start + duration) * sr]


def get_spectrogram_chunk(
        audio,
        sr: float,
        start: float,
        duration: float,
        volume: float = 1,
        n_fft: int = 2048,
        hop_length: int = 512
):    
    sample = get_audio_sample(audio, sr, start=start, duration=duration) * volume
    f_img = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
    chunk = np.abs(np.real(f_img))
    return chunk


def get_spectrogram_chunks(
    audio,
    sr: float,
    n_chunks: int,
    start: float,
    duration: float,
    overlap: float,
    volume: float = 1,
    n_fft: int = 2048,
    hop_length: int = 512
) -> np.ndarray:
    
    chunks = []
    for i in range(n_chunks):
        chunk = get_spectrogram_chunk(
            audio=audio,
            sr=sr,
            start=start + i*(duration - overlap),
            duration=duration,
            volume=volume,
            n_fft=n_fft,
            hop_length=hop_length
        )
        chunks.append(chunk)

    return np.array(chunks)


def plot_spectrograms2(f_imgs, sr, hop_length):
    plt.figure(figsize=(10, 4))
    for i, f_img in enumerate(f_imgs):
        plt.subplot(2, 1, i+1)
        S_dB = librosa.amplitude_to_db(np.abs(f_img), ref=np.max)
        plt.imshow(
            S_dB,
            aspect='auto',
            origin='lower',
            extent=[0, len(f_img[0]) * hop_length/sr, 0, sr/2],
            cmap='coolwarm'
        )
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.colorbar(format='%+2.0f dB')


def trim_audio(audio, sr: float, start_time: float, end_time: float):
    """
    Обрезает audio в пределах [start_time, end_time] секунд  
    """
    return audio[start_time * sr: end_time * sr]


def extract_chunk_fourier_img(
    f_img,
    sr: float,
    hop_length: int,
    start_time: float,
    duration: float = None,
    end_time: float = None
):
    """
    Вырезает часть фурье образа (либо его спектрограммы с теми же размерностями), в зависимости от параметров времени
    - start_time (время начала куска в секундах)
    - duration (длительность куска)
    - end_time (время окончания куска)
    исходного WAV сигнала
    Если переданы оба параметра duration и end_time, предпочтение отдается end_time.
    """
    
    if end_time is None and duration is None:
        raise ValueError("Должен быть указан хотя бы один из параметров: end_time или duration.")
        
    num_hops = sr // hop_length  # Количество кадров в сигнале 

    start_index = int(start_time * num_hops)
    
    if duration is not None:
        end_index = int((start_time + duration) * num_hops)
        
    if end_time is not None:
        end_index = int(end_time * num_hops)

    return f_img[:, start_index:end_index]


def get_fourier_chunks(
    f_img,
    sr: float,
    n_chunks: int,
    hop_length: int,
    start_time: float,
    duration: float,
    overlap: float
) -> list:
    """
    Вырезает n_chunks кусков из фурье образа, в зависимости от параметров времени
    - start_time (время с которого начнется нарезка chunks)
    - duration (длительность chunks)
    - end_time (время окончания chunks)
    - overlap (время перекрытия по длительности в секундах между chunks)
    исходного WAV сигнала.
    """
    chunks = []
    for i in range(n_chunks):
        chunk = extract_chunk_fourier_img(
            f_img,
            sr,
            hop_length,
            start_time=start_time + i*(duration - overlap),
            duration=duration
        )
        chunks.append(chunk)

    return chunks


def plot_spectrograms(mel_specs, sr, hop_length):
    plt.figure(figsize=(10, 8))
    for i, mel_spec in enumerate(mel_specs):
        plt.subplot(len(mel_specs), 1, i+1)
        librosa.display.specshow(mel_spec, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='inferno')
        plt.colorbar(format='%+2.0f dB')
        plt.ylabel('Mel Frequency [Hz]')
        plt.xlabel('Time [sec]')
    plt.tight_layout()
    plt.show()


# +
def change_volume(audio, factor):
    return audio * factor


def change_pitch(audio, sr, n_steps):
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)


def change_tempo(audio, factor):
    return librosa.effects.time_stretch(audio, factor)


# +
def get_mel_spectrogram_chunk(
        audio,
        sr: float,
        start: float,
        duration: float,
        volume: float = 1,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128
):
    sample = get_audio_sample(audio, sr, start=start, duration=duration) * volume
    mel_spec = librosa.feature.melspectrogram(y=sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


def get_mel_spectrogram_chunks(
    audio,
    sr: float,
    n_chunks: int,
    start: float,
    duration: float,
    overlap: float,
    volume: float = 1,
    n_fft: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128
) -> np.ndarray:

    chunks = []
    for i in range(n_chunks):
        chunk = get_mel_spectrogram_chunk(
            audio=audio,
            sr=sr,
            start=start + i*(duration - overlap),
            duration=duration,
            volume=volume,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        chunks.append(chunk)

    return np.array(chunks)


# +
if __name__ == '__main__':
    folder_path = '/app/pgrachev/data/tracks_wav'
    filenames = os.listdir(folder_path)

    audio, sr = librosa.load(os.path.join(folder_path, filenames[4]), sr=44100, mono=True)
    audio = trim_audio(audio, sr, start_time=10, end_time=int(len(audio)/sr) - 10)

    n_fft = 2048  # Размер окна в семплах
    hop_length = 512  # Длина скачка меджу окнами в семплах
    n_chunks = 2

    # Параметры
    start_time = 20  # начало в секундах
    duration = 20    # длительность в секундах
    overlap = 15

#     # Исходные мел-спектрограммы
#     mel_chunks_original = get_mel_spectrogram_chunks(audio, sr, n_chunks=n_chunks, start=0, duration=duration, overlap=overlap)

#     # Изменение громкости
#     audio_volume = change_volume(audio, 1.5)
#     mel_chunks_volume = get_mel_spectrogram_chunks(audio_volume, sr, n_chunks=n_chunks, start=0, duration=duration, overlap=overlap)

#     # Изменение тональности
#     audio_pitch = change_pitch(audio, sr, 2)
#     mel_chunks_pitch = get_mel_spectrogram_chunks(audio_pitch, sr, n_chunks=n_chunks, start=0, duration=duration, overlap=overlap)

#     # Изменение темпа
#     audio_tempo = change_tempo(audio, 1.5)
#     mel_chunks_tempo = get_mel_spectrogram_chunks(audio_tempo, sr, n_chunks=n_chunks, start=0, duration=duration, overlap=overlap)

#     # Отображение мел-спектрограмм
#     plot_spectrograms([mel_chunks_original[0], mel_chunks_volume[0]], sr, hop_length)
#     plot_spectrograms([mel_chunks_original[0], mel_chunks_pitch[0]], sr, hop_length)
#     plot_spectrograms([mel_chunks_original[0], mel_chunks_tempo[0]], sr, hop_length)
    
    import time
    def measure_time(func, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return end_time - start_time, result

    # Измерение времени для простого семплирования с перекрытием
    time_basic, sp_chunks_basic = measure_time(
        get_spectrogram_chunks,
        audio,
        sr,
        n_chunks=2,
        start=0,
        duration=duration,
        overlap=overlap
    )

    # Выводим время выполнения простого семплирования с перекрытием
    print(f"Время выполнения простого семплирования с перекрытием: {time_basic:.4f} секунд")

    # Аугментации
    volume_factor = 1.5
    tempo_factor = 1.2
    n_steps = 4

    # Измерение времени для изменения громкости
    time_volume, _ = measure_time(
        change_volume,
        audio,
        volume_factor
    )

    # Измерение времени для изменения темпа
    time_tempo, _ = measure_time(
        change_tempo,
        audio,
        tempo_factor
    )

    # Измерение времени для изменения питча
    time_pitch, _ = measure_time(
        change_pitch,
        audio,
        sr,
        n_steps
    )

    # Выводим время выполнения и относительные затраты времени
    print(f"Изменение громкости: {time_volume:.4f} секунд, Относительные затраты: {time_volume / time_basic:.2f}x")
    print(f"Изменение темпа: {time_tempo:.4f} секунд, Относительные затраты: {time_tempo / time_basic:.2f}x")
    print(f"Изменение питча: {time_pitch:.4f} секунд, Относительные затраты: {time_pitch / time_basic:.2f}x")
    
    
    # Отображение спектрограмм
    sp_chunks_volume = get_spectrogram_chunks(
        change_volume(audio, volume_factor),
        sr,
        n_chunks=2,
        start=0,
        duration=duration,
        overlap=overlap
    )
    sp_chunks_tempo = get_spectrogram_chunks(
        change_tempo(audio, tempo_factor),
        sr,
        n_chunks=2,
        start=0,
        duration=duration,
        overlap=overlap
    )
    sp_chunks_pitch = get_spectrogram_chunks(
        change_pitch(audio, sr, n_steps),
        sr,
        n_chunks=2,
        start=0,
        duration=duration,
        overlap=overlap
    )

    plot_spectrograms(sp_chunks_basic, sr, hop_length)
    plot_spectrograms(sp_chunks_volume, sr, hop_length)
    plot_spectrograms(sp_chunks_tempo, sr, hop_length)
    plot_spectrograms(sp_chunks_pitch, sr, hop_length)
# -

if __name__ == '__main__':
    folder_path = '/app/pgrachev/data/tracks_wav'
    filenames = os.listdir(folder_path)
    
    audio, sr = librosa.load(os.path.join(folder_path, '6okxuiiHx2w.wav'), sr=44100, mono=True)
    audio = trim_audio(audio, sr, start_time=10, end_time=int(len(audio)/sr) - 10)
    
    
    n_fft = 2048  # Размер окна в семплах
    hop_length = 512  # Длина скачка меджу окнами в семплах
    n_chunks = 2
    
    
    #f_img = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)  # Фурье образ
    
    # Параметры
    start_time = 10  # начало в секундах
    duration = 20    # длительность в секундах
    end_time = 50
    overlap = 15
    
    sp_chunks = get_spectrogram_chunks(audio, sr, n_chunks=2, start=0, duration=duration, overlap=overlap)

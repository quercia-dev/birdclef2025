import os
import pandas as pd
import torch
import torchaudio
import time
from concurrent.futures import ProcessPoolExecutor

wav_sec = 5
sample_rate = 32000
min_segment = sample_rate * wav_sec

root_path = 'data/'
input_path = root_path + '/train_audio/'
output_path = '.' + f'/train_raw{wav_sec}/'
metadata_path = root_path + 'train.csv'
backend='soundfile' # requires: pip install soundfile

os.makedirs(output_path, exist_ok=True)
ta_metadata = pd.read_csv(metadata_path)


def crop_and_save(index):
    filename = ta_metadata.iloc[index].filename
    filepath = input_path + filename

    try:

        sig, _ = torchaudio.load(filepath, backend=backend)
        if sig.shape[1]<=min_segment:
            sig = torch.cat([sig, torch.zeros(1, min_segment-sig.shape[1])], dim=1)

        dir_path = output_path + filename.split('/')[0] + '/'
        os.makedirs(dir_path, exist_ok=True)

        tmp_savename = output_path + filename
        torchaudio.save(uri=tmp_savename, src=sig[:,:min_segment], sample_rate=sample_rate, backend=backend)
        return 1
    except Exception as e:
        print(f'Error processing {filename}: {e}')
        return 0

if __name__ == "__main__":
    start_time = time.time()
    total = len(ta_metadata)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(crop_and_save, range(total)))

    end_time = time.time()

    print(f'Processed {total} audio files.')
    print(f'Cropped and saved {sum(results)} snippets.')
    print(f'Total time: {end_time - start_time:.2f} seconds.')

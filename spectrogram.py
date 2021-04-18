import os

from scipy.io import wavfile
from scipy.signal import spectrogram
from matplotlib import pyplot as plt
from json import JSONDecoder

DIR = "raw_data"

for f in os.listdir(DIR):
    print(f)
    (basename, extension) = f.split(".")

    # If it's a wav file, process it
    if extension == "wav":
        sample_rate, samples = wavfile.read(os.path.join(DIR, f))

        # Get corresponding ground truth JSON file
        labels_file = open(os.path.join(DIR, basename + ".json"))
        labels = JSONDecoder().decode(labels_file.read())

        # Compute spectrogram
        freqs, times, spec = spectrogram(samples, sample_rate, nperseg=20000, noverlap=5000)

        # Plot spectrogram using color map
        plt.pcolormesh(times, freqs, spec, cmap="inferno", vmin=0, vmax=1000)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.ylim([0, 5000])

        # Annotate spectogram plot with key labels
        ax = plt.gca()
        for timestamp in labels:
            label = labels[timestamp]
            timestamp = int(timestamp)
            ax.text(timestamp / 1000, 2300, label, color="white")
        plt.show()

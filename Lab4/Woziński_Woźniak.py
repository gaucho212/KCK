import sys
import numpy as np
from scipy.io import wavfile


def measure_f0(signal,sr):
    return 0



def pick_gender(file_path: str, border_hz: float = 165.0):
    sr, signal = wavfile.read(file_path)
    f0 = measure_f0(signal, sr)
    gender = "K" if f0 >= border_hz else "M"
    return gender, f0


def main():
    if len(sys.argv) < 2:
        return

    file_path = sys.argv[1]
    try:
        gender, _ = pick_gender(file_path)
        print(gender)
    except Exception:
        return


if __name__ == "__main__":
    main()
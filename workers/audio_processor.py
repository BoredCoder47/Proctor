import numpy as np
import time


class AudioProcessor:
    """
    Ultra-light backend noise detector.

    Works with raw float32 PCM from browser.
    Stable, fast, and secure.
    """

    def __init__(self):
        self.noise_threshold = 0.03   # tune if mic quiet
        self.cooldown = 2.0
        self.last_noise_time = 0

    def process_pcm(self, audio: np.ndarray):
        if audio is None or len(audio) == 0:
            return {"noise": False}

        # RMS energy
        energy = float(np.sqrt(np.mean(audio ** 2)))
        now = time.time()

        noise = False
        if energy > self.noise_threshold and (now - self.last_noise_time) > self.cooldown:
            noise = True
            self.last_noise_time = now

        return {
            "noise": noise,
            "energy": round(energy, 4)
        }
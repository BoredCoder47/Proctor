import numpy as np
import time


class AudioProcessor:
    """
    Ultra-light realtime noise detector using RMS energy.
    Designed for backend PCM streaming from browser.
    """

    def __init__(self):
        self.noise_threshold = 0.03
        self.cooldown = 2.0
        self.last_noise_time = 0

    def process_pcm(self, audio: np.ndarray):
        if audio is None or len(audio) == 0:
            return {"noise": False}

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
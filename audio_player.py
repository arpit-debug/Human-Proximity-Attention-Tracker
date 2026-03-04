"""
Audio playback helper using pygame.

Behavior:
- Start playing when `num_tracked > 0` and not already playing.
- When playback remaining time <= `restart_threshold` (seconds),
  if people still present -> restart (loop behavior).
- If no people present -> allow track to finish naturally.
"""

import os
import time
import pygame


class AudioPlayer:
    def __init__(self, path, restart_threshold=2.0):
        self.path = path
        self.restart_threshold = restart_threshold
        self.enabled = bool(path) and os.path.exists(path)

        self._initialized = False
        self._length = None
        self._start_time = None

        if self.enabled:
            try:
                pygame.mixer.init()
                pygame.mixer.music.load(self.path)
                self._length = pygame.mixer.Sound(self.path).get_length()
                self._initialized = True
                print(f"✓ Audio loaded: {self.path}")
            except Exception as e:
                print(f"Audio init failed: {e}")
                self.enabled = False

    # ---------------------------------------------------------
    # Static helper to locate campaign audio
    # ---------------------------------------------------------
    @staticmethod
    def find_campaign_audio(campaign_name, base_dir):
        """
        Find first .mp3 or .wav file inside BaseDir/CampaignName
        """
        if not campaign_name:
            return None

        base_dir = os.path.abspath(base_dir)
        campaign_dir = os.path.join(base_dir, campaign_name)

        if not os.path.isdir(campaign_dir):
            return None

        for fname in sorted(os.listdir(campaign_dir)):
            if fname.lower().endswith((".mp3", ".wav")):
                return os.path.join(campaign_dir, fname)

        return None

    # ---------------------------------------------------------
    # Core controls
    # ---------------------------------------------------------
    def play(self):
        if not self.enabled or not self._initialized:
            return

        try:
            pygame.mixer.music.play()
            self._start_time = time.time()
        except Exception:
            pass

    def stop(self):
        if not self.enabled:
            return

        try:
            pygame.mixer.music.stop()
        except Exception:
            pass

    def is_playing(self):
        if not self.enabled:
            return False

        return pygame.mixer.music.get_busy()

    def time_remaining(self):
        """
        Estimate remaining playback time (seconds).
        """
        if not self.enabled or self._length is None or self._start_time is None:
            return None

        elapsed = time.time() - self._start_time
        return max(0.0, self._length - elapsed)

    # ---------------------------------------------------------
    # Manage loop behavior
    # ---------------------------------------------------------
    def manage(self, num_tracked):
        if not self.enabled:
            return

        playing = self.is_playing()
        rem = self.time_remaining()

        # If people present and not playing -> start
        if num_tracked > 0 and not playing:
            self.play()
            return

        # If playing and near end -> restart only if people present
        if playing and rem is not None:
            if rem <= self.restart_threshold:
                if num_tracked > 0:
                    self.play()

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def cleanup(self):
        if self.enabled:
            pygame.mixer.quit()
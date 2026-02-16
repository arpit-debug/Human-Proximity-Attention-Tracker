"""Audio playback helper using python-vlc.

Behavior:
- Start playing when `num_tracked > 0` and not already playing.
- When playback remaining time <= `restart_threshold` (seconds), check
  `num_tracked`: if > 0, restart the track (loop); if 0, allow it to finish.
"""

import time
import os


class AudioPlayer:
    def __init__(self, path, restart_threshold=2.0):
        self.path = path
        self.restart_threshold = restart_threshold
        self.enabled = bool(path) and os.path.exists(path)
        self._vlc = None
        self._instance = None
        self._player = None
        if self.enabled:
            try:
                import vlc
                self._vlc = vlc
                #self._instance = vlc.Instance()
                self._instance = vlc.Instance("--aout=directsound")
                self._player = self._instance.media_player_new()
            except Exception:
                # If vlc is not available, disable audio
                self.enabled = False

    @staticmethod
    def find_campaign_audio(campaign_name, base_dir):
        """Find the first .mp3 file in BaseDir/CampaignName and return its full path.

        Returns None if not found.
        """
        if not campaign_name:
            return None
        base_dir = os.path.abspath(base_dir)
        campaign_dir = os.path.join(base_dir, campaign_name)
        if not os.path.isdir(campaign_dir):
            return None
        for fname in sorted(os.listdir(campaign_dir)):
            if fname.lower().endswith('.mp3'):
                return os.path.join(campaign_dir, fname)
        return None

    def _ensure_media(self):
        if not self.enabled:
            return
        media = self._instance.media_new(self.path)
        self._player.set_media(media)

    def play(self):
        if not self.enabled:
            return
        try:
            self._ensure_media()
            self._player.play()
        except Exception:
            pass

    def stop(self):
        if not self.enabled:
            return
        try:
            self._player.stop()
        except Exception:
            pass

    def is_playing(self):
        if not self.enabled:
            return False
        try:
            return self._player.is_playing() == 1
        except Exception:
            return False

    def time_remaining(self):
        """Return remaining time in seconds or None if unknown."""
        if not self.enabled:
            return None
        try:
            length = self._player.get_length()  # ms
            pos = self._player.get_time()
            if length <= 0 or pos < 0:
                return None
            return max(0.0, (length - pos) / 1000.0)
        except Exception:
            return None

    def manage(self, num_tracked):
        """
        Manage playback according to `num_tracked`.

        Called each loop iteration with current tracked count.
        """
        if not self.enabled:
            return

        playing = self.is_playing()
        rem = self.time_remaining()

        # If people present and not playing -> start
        if num_tracked > 0 and not playing:
            self.play()
            return

        # If playing and close to end, do nothing here to avoid interrupting the
        # current playback. When playback finishes, `is_playing()` becomes False
        # and the top clause (num_tracked > 0 and not playing) will restart it.

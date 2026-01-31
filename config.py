"""Runtime configuration for the Human Proximity project.

Settings:
- `DEBUG` (bool): when True show GUI with annotations; when False run headless and print stats.
- `STALE_FACE_TIMEOUT` (float): seconds after which an unseen face is removed from tracking.
- `Campaign_name` (str): name of the campaign folder under `Campain_Audio/`.
	The program will look for the first .mp3 inside `Campain_Audio/<Campaign_name>/` and play it
	while faces are present. Leave empty to disable campaign audio.
"""

# When True the GUI window will be shown and annotated frames displayed.
# When False the script will not open any windows and will only pprint stats.
DEBUG = True

# Face tracking: timeout (in seconds) for removing stale faces
# If a face is not detected for more than this duration, it's removed from tracking
STALE_FACE_TIMEOUT = 3.0

# Campaign audio configuration
# Set `Campaign_name` to the folder name under `Campain_Audio/` that contains your .mp3 files.
# Example: if Campaign_name = "Coca-Cola Zero", place MP3s under
#   Campain_Audio/Coca-Cola Zero/*.mp3
# The script will pick the first .mp3 (alphabetically) found in that folder.
# Leave empty ("") to disable campaign audio.
Campaign_name = "Coca-Cola Zero"
 
# If running on a Raspberry Pi with the official camera module set this to True
# When True the detector will try to use Picamera2 for better FPS/latency.
IS_RASPBERRY_PI = False

from ctypes import cast, POINTER
from typing import Annotated
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from semantic_kernel.functions import kernel_function

class VolPlugin:

    @kernel_function(
        name="SetVolume",
        description="Sets the system volume to a specified level.",
    )
    def setVolume(self, desired_level: Annotated[float, "Volume level from 0.0 (mute) to 1.0 (max)"]):
    # Get default audio device
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        volume.SetMasterVolumeLevelScalar(desired_level, None)

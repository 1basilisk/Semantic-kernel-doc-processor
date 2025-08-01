from semantic_kernel.functions import kernel_function
import os
import ctypes

class wallpaperPlugin:
    @kernel_function(
        name="set_wallpaper",
        description="Sets the wallpaper to a specified image URL",
    )


    def set_wallpaper(self, image_path):
        # Constants for Windows API
        SPI_SETDESKWALLPAPER = 20
        SPIF_UPDATEINIFILE = 0x01
        SPIF_SENDWININICHANGE = 0x02

        # Convert to absolute path
        abs_path = os.path.abspath(image_path)

        # Call Windows API
        result = ctypes.windll.user32.SystemParametersInfoW(
            SPI_SETDESKWALLPAPER, 0, abs_path,
            SPIF_UPDATEINIFILE | SPIF_SENDWININICHANGE
        )
        return result
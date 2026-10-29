"""

This script downloads real world sun position data and imports them into the
pipelines time of day system.

(c) 2016 tobpsr

"""

from __future__ import print_function, division

# You can get this data from http://sunpath.azurewebsites.net/
CONFIG = {
    "year": 2016,
    "month": 7,
    "day": 21,
    "latitude": 50.778228759765625,
    "longitude": 6.088640213012695,
    "height": 167,
    "precision": 6
}

if __name__ == "__main__":

    import sys
    import math

    sys.path.insert(0, "../../")
    from rplibs.six.moves import urllib  # pylint: disable=import-error

    CONFIG["end_day"] = CONFIG["day"]

    API = "http://www.nrel.gov/midc/apps/spa.pl?syear={year}&smonth={month}&sday={day}&eyear={year}&emonth={month}&eday={end_day}&step={precision}&stepunit=1&latitude={latitude}&longitude={longitude}&timezone=%2B1&elev={height}&press=1013&temp=20&dut1=0.0&deltat=64.797&azmrot=180&slope=0&refract=0.5667&field=2&field=35&field=36&field=38&zip=0"  # noqa

    url = API.format(**CONFIG)

    query_str = (
        "Are you sure you want to import the data? This overrides your "
        "current sun azimuth, altitude and intensity settings! (y/n): "
    )
    if sys.version_info.major >= 3:
        query_result = input(query_str)
    else:
        query_result = raw_input(query_str)

    if query_result.strip().lower() not in ["y", "yes", "j"]:
        print("Aborted.")
        sys.exit(0)

    print("Querying api ..")
    handle = urllib.request.urlopen(url)
    data = handle.read().decode("utf-8")
    handle.close()

    # Extract data
    print("Extracting data ..")
    lines = data.replace("\r", "").split("\n")[1:]

    # Load render pipeline api
    print("Loading plugin api ..")
    sys.path.insert(0, "../../")
    from rpcore.pluginbase.manager import PluginManager
    from rpcore.mount_manager import MountManager

    mount_mgr = MountManager(None)
    mount_mgr.mount()

    plugin_mgr = PluginManager(None)
    plugin_mgr.load()

    convert_to_linear = plugin_mgr.day_settings["scattering"]["sun_intensity"].get_linear_value

    hour, minutes = 0, 0

    data_points_azimuth = []
    data_points_altitude = []
    data_points_intensity = []

    for line in lines:
        if not line:
            break
        date, time, azim_angle, declination, ascension, elevation = line.split(",")

        float_azim = (float(azim_angle) + 180) / 360
        float_elev = (float(elevation) / 60) * 0.5 + 0.5

        float_time = (minutes + hour * 60) / (24 * 60)

        data_points_azimuth.append((float_time, float_azim))
        data_points_altitude.append((float_time, float_elev))

        # Approximated intensity
        approx_intensity = 2.0 * (1 - math.cos(math.pi * max(0, 8.0 + float(elevation)) / 60.0))
        approx_intensity = max(0, min(150.0, approx_intensity * 0.25))
        data_points_intensity.append((float_time, approx_intensity))

        minutes += CONFIG["precision"]
        if minutes >= 60:
            hour += 1
            minutes = minutes % 60

    print("Saving ..")
    settings = plugin_mgr.day_settings["scattering"]
    settings["sun_azimuth"].curves[0].control_points = data_points_azimuth
    settings["sun_altitude"].curves[0].control_points = data_points_altitude
    settings["sun_intensity"].curves[0].control_points = data_points_intensity

    plugin_mgr.save_daytime_overrides("/$$rpconfig/daytime.yaml")

    print("Done!")

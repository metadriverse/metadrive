"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

# Disable the unused variable warning - this occurs since we read a lot more
# properties from the IES file than we actually need
# pylint: disable=unused-variable

from __future__ import print_function

import re

from panda3d.core import PTAFloat, Filename, SamplerState, VirtualFileSystem
from panda3d.core import get_model_path
from direct.stdpy.file import open, join, isfile

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

from metadrive.render_pipeline.rpcore.native import IESDataset
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.rpobject import RPObject


class InvalidIESProfileException(Exception):
    """ Exception which is thrown when an error occurs during loading an IES
    Profile """


class IESProfileLoader(RPObject):
    """ Loader class to load .IES files and create an IESDataset from it.
    It generates a LUT for each loaded ies profile which is used by the lighting
    pipeline later on. """

    # Supported IES Profiles
    PROFILES = [
        "IESNA:LM-63-1986", "IESNA:LM-63-1991", "IESNA91", "IESNA:LM-63-1995", "IESNA:LM-63-2002",
        "ERCO Leuchten GmbH  BY: ERCO/LUM650/8701", "ERCO Leuchten GmbH"
    ]

    # Regexp for extracting keywords
    KEYWORD_REGEX = re.compile(r"\[([A-Za-z0-8_-]+)\](.*)")

    def __init__(self, pipeline):
        RPObject.__init__(self)
        self._pipeline = pipeline
        self._entries = []
        self._max_entries = 32
        self._create_storage()

    def _create_storage(self):
        """ Internal method to create the storage for the profile dataset textures """
        self._storage_tex = Image.create_3d("IESDatasets", 512, 512, self._max_entries, "R16")
        self._storage_tex.set_minfilter(SamplerState.FT_linear)
        self._storage_tex.set_magfilter(SamplerState.FT_linear)
        self._storage_tex.set_wrap_u(SamplerState.WM_clamp)
        self._storage_tex.set_wrap_v(SamplerState.WM_repeat)
        self._storage_tex.set_wrap_w(SamplerState.WM_clamp)

        self._pipeline.stage_mgr.inputs["IESDatasetTex"] = self._storage_tex
        self._pipeline.stage_mgr.defines["MAX_IES_PROFILES"] = self._max_entries

    def load(self, filename):
        """ Loads a profile from a given filename and returns the internal
        used index which can be assigned to a light."""

        # Make sure the user can load profiles directly from the ies profile folder
        data_path = join("/$$rp/data/ies_profiles/", filename)
        if isfile(data_path):
            filename = data_path

        # Make filename unique
        fname = Filename.from_os_specific(filename)
        if not VirtualFileSystem.get_global_ptr().resolve_filename(fname, get_model_path().get_value(), "ies"):
            self.error("Could not resolve", filename)
            return -1
        fname = fname.get_fullpath()

        # Check for cache entries
        if fname in self._entries:
            return self._entries.index(fname)

        # Check for out of bounds
        if len(self._entries) >= self._max_entries:
            # TODO: Could remove unused profiles here or regenerate texture
            self.warn("Cannot load IES Profile, too many loaded! (Maximum: 32)")

        # Try loading the dataset, and see what happes
        try:
            dataset = self._load_and_parse_file(fname)
        except InvalidIESProfileException as msg:
            self.warn("Failed to load profile from", filename, ":", msg)
            return -1

        if not dataset:
            return -1

        # Dataset was loaded successfully, now copy it
        dataset.generate_dataset_texture_into(self._storage_tex, len(self._entries))
        self._entries.append(fname)

        return len(self._entries) - 1

    def _load_and_parse_file(self, pth):
        """ Loads a .IES file from a given filename, returns an IESDataset
        which is used by the load function later on. """
        self.debug("Loading ies profile from", pth)

        try:
            with open(pth, "r") as handle:
                lines = handle.readlines()
        except IOError as msg:
            self.error("Failed to open", pth, ":", msg)
            return None

        lines = [i.strip() for i in lines]

        # Parse version header
        self._check_version_header(lines.pop(0))

        # Parse arbitrary amount of keywords
        keywords = self._extract_keywords(lines)  # noqa

        # Next line should be TILT=NONE according to the spec
        if lines.pop(0) != "TILT=NONE":
            raise InvalidIESProfileException("Expected TILT=NONE line, but none found!")

        # From now on, lines do not matter anymore, instead everything is
        # space seperated
        new_parts = (' '.join(lines)).replace(",", " ").split()

        def read_int():
            return int(new_parts.pop(0))

        def read_float():
            return float(new_parts.pop(0))

        # Amount of Lamps
        if read_int() != 1:
            raise InvalidIESProfileException("Only 1 Lamp supported!")

        # Extract various properties
        lumen_per_lamp = read_float()  # noqa
        candela_multiplier = read_float()  # noqa
        num_vertical_angles = read_int()
        num_horizontal_angles = read_int()

        if num_vertical_angles < 1 or num_horizontal_angles < 1:
            raise InvalidIESProfileException("Invalid of vertical/horizontal angles!")

        photometric_type = read_int()  # noqa
        unit_type = read_int()

        # Check for a correct unit type, should be 1 for meters and 2 for feet
        if unit_type not in [1, 2]:
            raise InvalidIESProfileException("Invalid unit type")

        width = read_float()  # noqa
        length = read_float()  # noqa
        height = read_float()  # noqa
        ballast_factor = read_float()  # noqa
        future_use = read_float()  # Unused field for future usage  # noqa
        input_watts = read_float()  # noqa

        # Read vertical angles
        vertical_angles = [read_float() for i in range(num_vertical_angles)]
        horizontal_angles = [read_float() for i in range(num_horizontal_angles)]

        candela_values = []
        candela_scale = 0.0

        for i in range(num_horizontal_angles):
            vertical_data = [read_float() for i in range(num_vertical_angles)]
            candela_scale = max(candela_scale, max(vertical_data))
            candela_values += vertical_data

        # Rescale values, divide by maximum
        candela_values = [i / candela_scale for i in candela_values]

        if len(new_parts) != 0:
            self.warn("Unhandled data at file-end left:", new_parts)

            # Dont abort here, some formats like those from ERCO Leuchten GmbH
            # have an END keyword, just ignore everything after the data was
            # read in.

        dataset = IESDataset()
        dataset.set_vertical_angles(self._list_to_pta(vertical_angles))
        dataset.set_horizontal_angles(self._list_to_pta(horizontal_angles))
        dataset.set_candela_values(self._list_to_pta(candela_values))

        # Testing code to write out the LUT
        # from panda3d.core import Texture
        # tex = Texture("temp")
        # tex.setup_3d_texture(512, 512, 1, Image.T_float, Image.F_r16)
        # dataset.generate_dataset_texture_into(tex, 0)
        # tex.write("generated.png")

        return dataset

    def _list_to_pta(self, list_values):
        """ Converts a list to a PTAFloat """
        pta = PTAFloat.empty_array(len(list_values))
        for i, val in enumerate(list_values):
            pta[i] = val
        return pta

    def _check_version_header(self, first_line):
        """ Checks if the IES version header is correct and the specified IES
        version is supported """
        if first_line not in self.PROFILES:
            raise InvalidIESProfileException("Unsupported Profile: " + first_line)

    def _extract_keywords(self, lines):
        """ Extracts the keywords from a list of lines, and removes all lines
        containing keywords """
        keywords = {}
        while lines:
            line = lines.pop(0)
            if not line.startswith("["):

                # Special format used by some IES files, indicates end of properties
                # By just checking for the tilt keyword instead of validating each line,
                # we can read even malformed lines, like those from ERCO Leuchten GmbH
                if line != "TILT=NONE":
                    continue

                # No keyword, we popped the line already tho, so append it again
                lines.insert(0, line)
                return keywords
            else:

                # Try matching the keywords
                match = self.KEYWORD_REGEX.match(line)
                if match:
                    key, val = match.group(1, 2)
                    keywords[key.strip()] = val.strip()
                else:
                    raise InvalidIESProfileException("Invalid keyword line: " + line)

        return keywords

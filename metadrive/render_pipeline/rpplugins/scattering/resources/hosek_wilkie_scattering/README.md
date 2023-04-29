
# Hosek & Wilkie Skylight Model

This uses the precomputed scattering model proposed by:

<a href="http://cgg.mff.cuni.cz/projects/SkylightModelling/">http://cgg.mff.cuni.cz/projects/SkylightModelling/</a>

It generates a LUT evaluating the model offline, which then
will be used in the scattering shader later on.

### Usage

In case you want to generate the lookup texture yourself, you need to first
compile the code (see below), and then run the program.
The program will ask you for a turbidity and ground albedo, using
`turbidity=3.0` and `groundAlbedo=0.1` is a good starting point.

### Compilation

Just run `python update_module_builder.py`, after that you should see a `build.py`,
run it with `python build.py` to generate the module. When the compilation succeeded,
run `python generate_table.py` to generate the LUT. After that you can use the
scattering method in the pipeline.


### License

```

           "An Analytic Model for Full Spectral Sky-Dome Radiance"
       "Adding a Solar Radiance Function to the Hosek Skylight Model"

                                   both by 

                       Lukas Hosek and Alexander Wilkie
                Charles University in Prague, Czech Republic

                        Version: 1.4a, February 22nd, 2013


This source is published under the following 3-clause BSD license.

Copyright (c) 2012, Lukas Hosek and Alexander Wilkie
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * None of the names of the contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

```


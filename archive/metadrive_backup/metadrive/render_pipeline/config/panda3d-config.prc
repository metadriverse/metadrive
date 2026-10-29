

# This is the config file used to configure basic settings for Panda3D.
# The pipeline loads it at startup to ensure the environment is setup properly.

# --------------  Development options  --------------

pstats-gpu-timing #t
gl-debug #t
gl-debug-object-labels #t

# -------------- Production options ---------------

# pstats-gpu-timing #f
# gl-debug #f
# gl-debug-object-labels #f

# ----------------- Misc Settings -----------------

# Disable V-Sync
sync-video #f

# Limit the pstats-rate. This causes huge lag on windows 10.
pstats-max-rate 200

# No stack trace on assertion, set this to true to make panda crash on assertions
# (which will allow to debug it)
# assert-abort #t
# show-dll-error-dialog #f

# File system should be case sensitive
# NOTICE: Set this to #f if you are using tempfile, since it returns
# wrong cased directory paths
vfs-case-sensitive #t

# Enable state cache, this seems to actually help the performance by a lot
state-cache #t
transform-cache #t

# Hide frame rate meter (we have our own)
show-frame-rate-meter #f

# Set text settings
text-minfilter linear
text-magfilter linear
text-page-size 512 512
text-rwap-mode WM_border_clor

# Better text performance since rdb's patch
text-flatten 0
text-dynamic-merge 1

# For smoother animations
# even-animation #t

# Threading, really buggy!
#threading-model App/Cull/Draw

# Disable stencil, not supported/required
support-stencil #f
framebuffer-stencil #f

# Don't use srgb correction, we do that in the final shader
framebuffer-srgb #f

# Don't use multisamples
framebuffer-multisample #f
multisamples 0

# Don't rescale textures which are no power-of-2
textures-power-2 none

# Set default texture filters
texture-anisotropic-degree 16
texture-magfilter linear
texture-minfilter linear
texture-quality-level fastest

# Enable seamless cubemap filtering, important for environment filtering
gl-cube-map-seamless #t

# Disable caching of textures
model-cache-textures #f

# Disable the annoying SRGB warning from pnmimage
notify-level-pnmimage error

# Disable the buffer viewer, we have our own
show-buffers #f

# Use the default coordinate system, this makes our matrix transformations
# faster because we don't have have to transform them to a different coordinate
# system before
gl-coordinate-system default

# This makes animations smoother, especially if they were exported at 30 FPS
# and are played at 60 FPS
interpolate-frames #t

# Disable workarround in panda which causes our shadow atlas to take twice
# the amount of vram it should, due to an intel driver bug.
gl-force-fbo-color #f

# ----------- OpenGL / Performance Settings ------------

# Require OpenGL 4.3 at least, necessary for Intel drivers on Linux
gl-version 4 3

# Animations on the gpu. The default shader has to get adjusted to support this
# feature before this option can be turned on.
# hardware-animated-vertices #t

# Try this options for performance
# uniquify-matrix #t
# uniquify-transforms #t
# uniquify-states #t
# uniquify-attribs #f

# Enable garbarge collection
garbage-collect-states #t
# garbage-collect-states-rate 0.2

# Compress textures on the drivers?
# driver-compress-textures #t

# Faster animations? (Have to test)
# matrix-palette #t
# display-list-animation #t

# Better GL performance by not using gl-finish and so on
gl-finish #f
gl-force-no-error #t
gl-check-errors #f
gl-force-no-flush #t
gl-force-no-scissor #t

# Eventually disable memory barriers, have to check if this is faster
gl-enable-memory-barriers #f

# Disable threading
lock-to-one-cpu #t
support-threads #f

# Let the driver generate the mipmaps
driver-generate-mipmaps #t

# Use immutable texture storage, it is *supposed* to be faster, but might not be
# XXX: Seems to produce an GL_INVALID_VALUE when disabled
gl-immutable-texture-storage #f

# Default window settings
# depth-bits 0
color-bits 0
framebuffer-depth #f

# Small performance gain by specifying fixed vertex attribute locations.
# Might cause issues with some (incorrectly converted/loaded) meshes though
gl-fixed-vertex-attrib-locations #f

# Disable the fragment shader performance warning
gl-validate-shaders #f
gl-skip-shader-recompilation-warnings #t

alpha-scale-via-texture #f
pstats-name Render Pipeline Stats
rescale-normals #f
screenshot-extension png

# Required for correct velocity
always-store-prev-transform #t
allow-incomplete-render #t


no-singular-invert #f

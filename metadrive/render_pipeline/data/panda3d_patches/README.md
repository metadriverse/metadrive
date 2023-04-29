
## Patches for Panda3D

This folder contains patches for the Panda3D Engine which have not yet made
it to the Panda3D-Repository, but are used by some render pipeline features.

**IMPORTANT**: After you applied a patch, delete it, so the render pipeline
knows the patch is available.

## List of patches

#### prev-model-view-matrix.diff

This patch adds the p3d_PrevModelViewMatrix to add support for per object
velocities. When this patch is enabled, moving objects recieve motion blur (per
object motion blur). Also don't forget to apply the second part of the patch, too.

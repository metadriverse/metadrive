/**
 *
 * RenderPipeline
 *
 * Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 */


#ifndef RP_SHADOW_SOURCE_H
#define RP_SHADOW_SOURCE_H

#include "pandabase.h"
#include "luse.h"
#include "transformState.h"
#include "look_at.h"
#include "compose_matrix.h"
#include "perspectiveLens.h"
#include "boundingVolume.h"
#include "boundingSphere.h"
#include "boundingHexahedron.h"
#include "geometricBoundingVolume.h"

#include "gpu_command.h"

/**
 * @brief This class represents a single shadow source.
 * @details The ShadowSource can be seen as a Camera. It is used by the Lights
 *   to render their shadows. Each ShadowSource has a position in the atlas,
 *   and a view-projection matrix. The shadow manager regenerates the shadow maps
 *   using the data from the shadow sources.
 */
class ShadowSource {

public:
    ShadowSource();

    inline void write_to_command(GPUCommand &cmd) const;

    inline void set_needs_update(bool flag);
    inline void set_slot(int slot);
    inline void set_region(const LVecBase4i& region, const LVecBase4f& region_uv);
    inline void set_resolution(size_t resolution);
    inline void set_perspective_lens(float fov, float near_plane,
                                     float far_plane, LVecBase3f pos, LVecBase3f direction);
    inline void set_matrix_lens(const LMatrix4f& mvp);

    inline bool has_region() const;
    inline bool has_slot() const;

    inline void clear_region();

    inline int get_slot() const;
    inline bool get_needs_update() const;
    inline size_t get_resolution() const;
    inline const LMatrix4f& get_mvp() const;
    inline const LVecBase4i& get_region() const;
    inline const LVecBase4f& get_uv_region() const;

    inline const BoundingSphere& get_bounds() const;

private:
    int _slot;
    bool _needs_update;
    size_t _resolution;
    LMatrix4f _mvp;
    LVecBase4i _region;
    LVecBase4f _region_uv;

    BoundingSphere _bounds;
};

#include "shadow_source.I"

#endif // RP_SHADOW_SOURCE_H

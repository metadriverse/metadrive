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

#pragma once

/*

Normal packing as described in:
A Survey of Efficient Representations for Independent Unit Vectors
Source: http://jcgt.org/published/0003/02/01/paper.pdf

*/

// For each component of v, returns -1 if the component is < 0, else 1
vec2 sign_not_zero(vec2 v) {
    #if 1
        // Branch-Less version
        return fma(step(vec2(0.0), v), vec2(2.0), vec2(-1.0));
    #else
        // Version with branches (for GLSL < 4.00)
        return vec2(
            v.x >= 0 ? 1.0 : -1.0,
            v.y >= 0 ? 1.0 : -1.0
        );
    #endif
}

// Packs a 3-component normal to 2 channels using octahedron normals
vec2 pack_normal_octahedron(vec3 v) {
    #if 0
        // Version as proposed by the paper
        // Project the sphere onto the octahedron, and then onto the xy plane
        vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
        // Reflect the folds of the lower hemisphere over the diagonals
        return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * sign_not_zero(p)) : p;
    #else
        // Faster version using newer GLSL capatibilities
        v.xy /= dot(abs(v), vec3(1));

        #if 0
            // Version with branches
            if (v.z <= 0) v.xy = (1.0 - abs(v.yx)) * sign_not_zero(v.xy);
            return v.xy;
        #else
            // Branch-Less version
            return mix(v.xy, (1.0 - abs(v.yx)) * sign_not_zero(v.xy), step(v.z, 0.0));
        #endif
    #endif
}


// Unpacking from octahedron normals, input is the output from pack_normal_octahedron
vec3 unpack_normal_octahedron(vec2 packed_nrm) {
    #if 1
        // Version using newer GLSL capatibilities
        vec3 v = vec3(packed_nrm.xy, 1.0 - abs(packed_nrm.x) - abs(packed_nrm.y));
        #if 1
            // Version with branches, seems to take less cycles than the
            // branch-less version
            if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * sign_not_zero(v.xy);
        #else
            // Branch-Less version
            v.xy = mix(v.xy, (1.0 - abs(v.yx)) * sign_not_zero(v.xy), step(v.z, 0));
        #endif

        return normalize(v);
    #else
        // Version as proposed in the paper.
        vec3 v = vec3(packed_nrm, 1.0 - dot(vec2(1), abs(packed_nrm)));
        if (v.z < 0)
            v.xy = (vec2(1) - abs(v.yx)) * sign_not_zero(v.xy);
        return normalize(v);
    #endif
}

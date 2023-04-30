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

#ifndef RP_PSSM_HELPER_H
#define RP_PSSM_HELPER_H

// Only include the pssm helper if actually required
#ifdef RP_REQ_PSSM_HELPER

#include "pandabase.h"
#include "luse.h"

/**
 * @brief Class to generate projection matrices.
 */
class PSSMHelper {

    PUBLISHED:
        static LMatrix4f find_projection_mat(
            const LVector4f &near_ul,
            const LVector4f &near_ur,
            const LVector4f &near_ll,
            const LVector4f &near_lr,

            const LVector4f &far_ul,
            const LVector4f &far_ur,
            const LVector4f &far_ll,
            const LVector4f &far_lr);

};

#endif // RP_REQ_PSSM_HELPER

#endif // RP_PSSM_HELPER_H

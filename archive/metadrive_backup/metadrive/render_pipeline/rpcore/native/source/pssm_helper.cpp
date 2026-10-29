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

// Only include the pssm helper if actually required
#ifdef RP_REQ_PSSM_HELPER

#include "pssm_helper.h"

#include <Eigen/Dense>

/**
 * @brief Internal method to generate a set of equations
 * @details This generates a set of equations into a given equation system.
 *   Based on a given start point and end point, it inserts an equation to the
 *   system which will, when solved, solve the following equation:
 *   vec * <SOLVED-TRANSFORM> = expected
 *
 * @param eq_system The equation system
 * @param eq_results The target result vector of the equation system
 * @param vec The actual vector to be transformed
 * @param expected The expected output of the transformation
 * @param offset The index of the equation
 */
void generate_equations(Eigen::MatrixXf &eq_system, Eigen::VectorXf &eq_results, const LVector4f &vec, LVector4f expected, size_t offset) {
    size_t write_offset = offset * 4;
    for (size_t row = 0; row < 4; ++row) {
        float expected_coeff = expected.get_cell(row);
        size_t col_offset = row * 4;
        for (size_t col = 0; col < 4; ++col) {
            eq_system(write_offset, col_offset + col) = vec.get_cell(col);
        }
        eq_results(write_offset++) = expected_coeff;
    }
}
/**
 * @brief Finds a projection mat arround the given set of points.
 * @details This methods finds a projection matrix which projects the given set
 *   of frustum points to a unit cube, which can be used as a camera matrix.
 *   The eight points should determine the frustum uniquely.
 *
 *  @param near_ul The Upper-Left point of the frustum on the near plane
 *  @param near_ur The Upper-Right point of the frustum on the near plane
 *  @param near_ll The Lower-Left point of the frustum on the near plane
 *  @param near_lr The Lower-Right point of the frustum on the near plane
 *  @param far_ul The Upper-Left point of the frustum on the far plane
 *  @param far_ur The Upper-Right point of the frustum on the far plane
 *  @param far_ll The Lower-Left point of the frustum on the far plane
 *  @param far_lr The Lower-Right point of the frustum on the far plane
 *
 */
LMatrix4f PSSMHelper::find_projection_mat(
            const LVector4f &near_ul,
            const LVector4f &near_ur,
            const LVector4f &near_ll,
            const LVector4f &near_lr,

            const LVector4f &far_ul,
            const LVector4f &far_ur,
            const LVector4f &far_ll,
            const LVector4f &far_lr) {

    // We have 8*4 = 32 equations, which require 16 coefficients each
    Eigen::MatrixXf equation_system(32, 16);
    Eigen::VectorXf equation_results(32);
    equation_system.fill(0);

    // Generate the equations
    size_t offset = 0;
    generate_equations(equation_system, equation_results, near_ul, LVector4f(-1,  1, 0, 1), offset++);
    generate_equations(equation_system, equation_results, near_ur, LVector4f( 1,  1, 0, 1), offset++);
    generate_equations(equation_system, equation_results, near_ll, LVector4f(-1, -1, 0, 1), offset++);
    generate_equations(equation_system, equation_results, near_lr, LVector4f( 1, -1, 0, 1), offset++);

    generate_equations(equation_system, equation_results, far_ul,  LVector4f(-1,  1, 1, 1), offset++);
    generate_equations(equation_system, equation_results, far_ur,  LVector4f( 1,  1, 1, 1), offset++);
    generate_equations(equation_system, equation_results, far_ll,  LVector4f(-1, -1, 1, 1), offset++);
    generate_equations(equation_system, equation_results, far_lr,  LVector4f( 1, -1, 1, 1), offset++);

    // Solve the equation system
    Eigen::VectorXf solved_system = equation_system.colPivHouseholderQr().solve(equation_results);

    // Construct result matrix and return it. We also need to transpose the matrix.
    LMatrix4f result(
            solved_system(0), solved_system(4), solved_system(8),  solved_system(12),
            solved_system(1), solved_system(5), solved_system(9),  solved_system(13),
            solved_system(2), solved_system(6), solved_system(10), solved_system(14),
            solved_system(3), solved_system(7), solved_system(11), solved_system(15)
        );
    return result;
}


#endif // RP_REQ_PSSM_HELPER

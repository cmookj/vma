//
//  test.cpp
//  Google test for linear algebra
//
//  Created by Changmook Chun on 2022-12-22.
//

#include "vma.hpp"
#include <cstdlib>
#include <gtest/gtest.h>
#include <iostream>

using namespace std;
using namespace gpw::vma;

bool
mem_limit (const int l) {
    if (getenv ("MEM_LIMIT") && atoi (getenv ("MEM_LIMIT")) >= l) {
        return true;
    }
    return false;
}

TEST (Vector, CreationAccess) {
    vec<5> v1;
    for (std::size_t i = 1; i <= v1.dim(); ++i)
        EXPECT_EQ (v1 (i), 0.);

    for (auto e = v1.begin(); e < v1.end(); ++e)
        EXPECT_EQ (*e, 0.);

    for (const auto& elem : v1)
        EXPECT_EQ (elem, 0.);

    vec<5> v2{1.};
    std::cout << to_string (v2) << std::endl;
    EXPECT_FLOAT_EQ (v2 (1), 1.);
    for (std::size_t i = 2; i <= v2.dim(); ++i)
        EXPECT_FLOAT_EQ (v2 (i), 0.);

    double a[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec<5> v3{a};
    for (std::size_t i = 1; i <= v3.dim(); ++i)
        EXPECT_EQ (v3 (i), a[i - 1]);

    vec<5> v4{1, 3, 5, 7, 9};
    for (std::size_t i = 1; i <= v4.dim(); ++i)
        EXPECT_EQ (v4 (i), 2. * i - 1.);

    vec<5> v5{1.0e1, 3.0e2, 5.0e3, 7.0e4, 9.0e5};
    for (std::size_t i = 1; i <= v5.dim(); ++i)
        EXPECT_EQ (v5 (i), (2. * i - 1.) * std::pow (10., i));

    vec<5> v6{-1, 3, -5, 7, -9};
    for (std::size_t i = 1; i <= v6.dim(); ++i)
        EXPECT_EQ (v6 (i), (2. * i - 1.) * std::pow (-1., i));

    std::cout << to_string (v6, output_fmt::sht) << std::endl;
    std::cout << to_string (v6) << std::endl;
    std::cout << to_string (v6, output_fmt::ext) << std::endl;
    std::cout << to_string (v6, output_fmt::sci) << std::endl;
    std::cout << to_string (v6, output_fmt::scx) << std::endl;

    auto v7 = rand<100>();

    auto v8 = randn<100>();
}

TEST (Vector, Collinear) {
    vec<9> v1{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    vec<9> v2{2. * v1};

    EXPECT_EQ (collinear (v1, v2), true);

    vec<9> v3{1., 0., 3., 0., 5., 0., 7., 0., 9.};
    vec<9> v4{0.3 * v3};

    EXPECT_EQ (collinear (v3, v4), true);

    vec<4, complex_t> cv1{
        {1.,  2. },
        {3.,  4. },
        {-2., -1.},
        {-5., 3. }
    };
    vec<4, complex_t> cv2{
        {2.,   4. },
        {6.,   8. },
        {-4.,  -2.},
        {-10., 6. }
    };
    EXPECT_EQ (collinear (cv1, cv2), true);

    vec<4, complex_t> cv3{
        {1.,  2. },
        {3.,  0. },
        {0.,  -1.},
        {-5., 3. }
    };
    vec<4, complex_t> cv4{
        {2.,   4. },
        {6.,   0. },
        {0.,   -2.},
        {-10., 6. }
    };
    EXPECT_EQ (collinear (cv3, cv4), true);
}

TEST (Vector, Comparison) {
    vec<5> v1{1, 2, 3, 4, 5};
    vec<5> v2{1. + EPS, 2, 3, 4, 5};

    EXPECT_EQ (v1, v1);
    EXPECT_TRUE (v1 != v2);

    EXPECT_TRUE (similar (v1, v2));

    vec<3> v3{1, 2, 3};
    vec<3> v4{4, 5, 6};

    EXPECT_TRUE (v3 < v4);
}

TEST (Vector, IndexAssignmentComparison) {
    vec<5> v1{1, 2, 3, 4, 5};

    v1 (2) = -v1 (2);
    v1 (4) = -v1 (4);
    EXPECT_EQ (v1 (1), 1.);
    EXPECT_EQ (v1 (2), -2.);
    EXPECT_EQ (v1 (3), 3.);
    EXPECT_EQ (v1 (4), -4.);
    EXPECT_EQ (v1 (5), 5.);

    auto v2{v1};
    EXPECT_EQ (v2, v2);
    EXPECT_EQ (v2, v1);

    auto v3{v1};
    EXPECT_EQ (v3, v1);
}

TEST (Vector, AdditionSubtraction) {
    vec<5> v1{1, 3, 5, 7, 9};
    vec<5> v2 (1.);

    auto v3{v1 + v2};
    for (std::size_t i = 1; i < 6; ++i)
        EXPECT_EQ (v3 (i), v1 (i) + 1.);

    auto v4{v1 - v2};
    for (std::size_t i = 1; i < 6; ++i)
        EXPECT_EQ (v4 (i), v1 (i) - 1.);

    auto v5 = -v2;
    for (std::size_t i = 1; i < 6; ++i)
        EXPECT_EQ (v5 (i) + v2 (i), 0.);

    v1 += 3;
    for (std::size_t i = 1; i < 6; ++i)
        EXPECT_EQ (v1 (i), 2. * i - 1. + 3.);

    v1 -= 3;
    for (std::size_t i = 1; i < 6; ++i)
        EXPECT_EQ (v1 (i), 2. * i - 1.);
}

TEST (Vector, MultiplicationDivision) {
    const double v[3] = {1., 3., 5.};
    vec<3>       v1{v};

    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ (v1[i], v[i]);

    auto v2{v1 * 2.};
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ (v2[i], v1[i] * 2.);

    v2 *= 2.0;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ (v2[i], v[i] * 4.);

    auto v3 = v1 / 2.0;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ (v3[i], v1[i] / 2.);

    v3 /= 2.0;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ (v3[i], v1[i] / 4.);

    auto v4 = .5 * v1 * 3. / 2.;
    for (std::size_t i = 0; i < 3; ++i)
        EXPECT_EQ (v4[i], .5 * v1[i] * 3. / 2.);
}

TEST (Vector, VectorMatrixMultiplication) {
    mat<3, 3> m1{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };
    vec<3> v1{1, 3, 5};

    auto v2 = m1 * v1;  // 22, 49, 76
    EXPECT_EQ (v2[0], 22.);
    EXPECT_EQ (v2[1], 49.);
    EXPECT_EQ (v2[2], 76.);

    auto v3 = v1 * m1;  // 48, 57, 66
    EXPECT_EQ (v3[0], 48.);
    EXPECT_EQ (v3[1], 57.);
    EXPECT_EQ (v3[2], 66.);
}

TEST (Vector, NormInnerProductNormalize) {
    vec<3> v0{};

    vec<3> v1{1, 3, 5};
    EXPECT_EQ (norm (v1, 1), 1. + 3. + 5.);
    EXPECT_EQ (norm (v1, 2), std::sqrt (1. + 9. + 25.));
    EXPECT_EQ (norm_inf (v1), 5.);

    vec<3> v2{-1., 3, -5.};
    EXPECT_EQ (norm (v2, 1), 1. + 3. + 5.);
    EXPECT_EQ (norm (v2, 2), std::sqrt (1. + 9. + 25.));
    EXPECT_EQ (norm_inf (v2), -5.);
    EXPECT_FLOAT_EQ (abs (v2), norm (v2));

    vec<3> v3{2, 4, 6};
    EXPECT_EQ (inner (v1, v3), 1. * 2. + 3. * 4. + 5. * 6.);
    EXPECT_EQ (dist (v0, v3), norm (v3));

    vec<6> v4{1, 5, 2, 4, 7, 6};
    EXPECT_EQ (norm (normalize (v4)), 1.);

    vec<4> v5{1, 1, 1, 1};
    vec<4> v6{2, 2, 2, 2};
    EXPECT_EQ (dist (v5, v6), std::sqrt (4 * 1.));
}

TEST (Matrix, CreationIndexing) {
    mat<4, 4> m0{};
    for (std::size_t i = 1; i <= m0.count_rows(); ++i)
        for (std::size_t j = 1; j <= m0.count_cols(); ++j)
            EXPECT_EQ (m0 (i, j), 0.);

    for (auto e = m0.begin(); e < m0.end(); ++e)
        EXPECT_EQ (*e, 0.);

    for (const auto& elem : m0)
        EXPECT_EQ (elem, 0.);

    mat<3, 4> m1{
        {1,  2,  3,  4 },
        {2., 3., 4., 5.},
        {3,  4,  5., 6 }
    };
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            EXPECT_EQ (m1 (i, j), (i - 1.) + j);

    std::cout << to_string (m1, output_fmt::sht) << std::endl;
    std::cout << to_string (m1) << std::endl;
    std::cout << to_string (m1, output_fmt::ext) << std::endl;
    std::cout << to_string (m1, output_fmt::sci) << std::endl;
    std::cout << to_string (m1, output_fmt::scx) << std::endl;

    mat<3, 4> m2{
        {1.,  -2, 3,  -4},
        {-2., 3,  -4, 5.},
        {3.,  -4, 5,  -6}
    };
    for (std::size_t i = 1; i <= m2.count_rows(); ++i)
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            EXPECT_EQ (m2 (i, j), ((i - 1.) + j) * std::pow (-1., i + j));

    std::vector<double> el0{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::vector<double> el00{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    mat<3, 3>           m21{std::move (el0)};
    EXPECT_EQ (m21.count_rows(), 3);
    EXPECT_EQ (m21.count_cols(), 3);
    EXPECT_EQ (m21.elem(), el00);

    auto m3 = identity<10>();
    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            EXPECT_FLOAT_EQ (m3 (i, j), (i == j ? 1. : 0.));

    auto m4 = rand<100, 100>();

    auto m5 = randn<100, 100>();

    std::vector<double> el1{1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    auto                m6 = diag<10> (el1);
    for (std::size_t i = 1; i <= m6.count_rows(); ++i)
        for (std::size_t j = 1; j <= m6.count_cols(); ++j)
            EXPECT_EQ (m6 (i, j), (i == j ? el1[i - 1] : 0.));

    double el2[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    auto   m7    = diag<10> (el2);
    for (std::size_t i = 1; i <= m7.count_rows(); ++i)
        for (std::size_t j = 1; j <= m7.count_cols(); ++j)
            EXPECT_EQ (m7 (i, j), (i == j ? el1[i - 1] : 0.));
}

TEST (Matrix, Comparison) {
    std::vector<double> e1{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    std::vector<double> e2{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    mat<3, 4>           m1{std::move (e1)};
    mat<3, 4>           m2{std::move (e2)};

    EXPECT_EQ (m1, m2);
}

TEST (Matrix, VectorExtraction) {
    mat<8, 10> m1{
        {1, 2, 3,  4,  5,  6,  7,  8,  9,  10},
        {2, 3, 4,  5,  6,  7,  8,  9,  10, 11},
        {3, 4, 5,  6,  7,  8,  9,  10, 11, 12},
        {4, 5, 6,  7,  8,  9,  10, 11, 12, 13},
        {5, 6, 7,  8,  9,  10, 11, 12, 13, 14},
        {6, 7, 8,  9,  10, 11, 12, 13, 14, 15},
        {7, 8, 9,  10, 11, 12, 13, 14, 15, 16},
        {8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    };

    // Extraction of columns
    vec<8> col0{};
    EXPECT_EQ (m1.col (0), col0);

    vec<8> col11{};
    EXPECT_EQ (m1.col (11), col11);

    vec<8> col1{1, 2, 3, 4, 5, 6, 7, 8};
    EXPECT_EQ (m1.col (1), col1);

    vec<8> col3{3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ (m1.col (3), col3);

    vec<8> col9{9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT_EQ (m1.col (9), col9);

    // Extraction of rows
    vec<10> row0{};
    EXPECT_EQ (m1.row (0), row0);

    vec<10> row9{};
    EXPECT_EQ (m1.row (9), row9);

    vec<10> row1{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ (m1.row (1), row1);

    vec<10> row4{4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    EXPECT_EQ (m1.row (4), row4);

    vec<10> row7{7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    EXPECT_EQ (m1.row (7), row7);
}

TEST (Matrix, Transpose) {
    mat<8, 10> m1{};

    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1 (i, j) = 0.5 * i + 0.3 * j;

    auto m2 = transpose (m1);

    EXPECT_EQ (m1.count_rows(), m2.count_cols());
    EXPECT_EQ (m1.count_cols(), m2.count_rows());

    for (std::size_t i = 1; i <= m2.count_rows(); ++i)
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            EXPECT_NEAR (m2 (i, j), 0.5 * j + 0.3 * i, 0.000000001);

    mat<3, 3> m3{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    auto m4 = transpose (m3);

    for (std::size_t i = 1; i <= m4.count_rows(); ++i)
        for (std::size_t j = 1; j <= m4.count_cols(); ++j)
            EXPECT_NEAR (m4 (i, j), m3 (j, i), 0.000000001);

    std::cout << "BEFORE TRANSPOSE\n";
    for (std::size_t i = 1; i <= m3.count_rows(); ++i) {
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            std::cout << m3 (i, j) << " ";
        std::cout << "\n";
    }

    std::cout << "AFTER TRANSPOSE\n";
    for (std::size_t i = 1; i <= m4.count_rows(); ++i) {
        for (std::size_t j = 1; j <= m4.count_cols(); ++j)
            std::cout << m4 (i, j) << " ";
        std::cout << "\n";
    }
}

template <size_t sz_r, size_t sz_c>
void
test_transpose (mat<sz_r, sz_c>& m) {
    for (std::size_t i = 1; i <= m.count_rows(); ++i)
        for (std::size_t j = 1; j <= m.count_cols(); ++j)
            m (i, j) = 0.5 * i + 0.3 * j;

    auto tr = transpose (m);

    EXPECT_EQ (m.count_rows(), tr.count_cols());
    EXPECT_EQ (m.count_cols(), tr.count_rows());

    for (std::size_t i = 1; i <= tr.count_rows(); ++i)
        for (std::size_t j = 1; j <= tr.count_cols(); ++j)
            EXPECT_NEAR (tr (i, j), 0.5 * j + 0.3 * i, 0.000000001);
}

TEST (Matrix, HeavyTranspose_4GiB) {
    if (!mem_limit (4)) {
        GTEST_SKIP() << "Environment variable MEM_LIMIT not set of less than 4."
                        "  Skipping HeavyTranspose_4GiB test.";
    }

    mat<16384, 16384> m1{};
    test_transpose (m1);
}

TEST (Matrix, HeavyTranspose_8GiB) {
    if (!mem_limit (8)) {
        GTEST_SKIP() << "Environment variable MEM_LIMIT not set of less than 8."
                        "  Skipping HeavyTranspose_8GiB test.";
    }

    mat<32768, 16384> m1{};
    test_transpose (m1);
}

TEST (Matrix, HeavyTranspose_16GiB) {
    if (!mem_limit (16)) {
        GTEST_SKIP() << "Environment variable MEM_LIMIT not set of less than 16."
                        "  Skipping HeavyTranspose_16GiB test.";
    }

    mat<32768, 32768> m1{};
    test_transpose (m1);
}

TEST (Matrix, Multiplication) {
    // Small matrices
    mat<8, 10> m1{
        {1, 2, 3,  4,  5,  6,  7,  8,  9,  10},
        {2, 3, 4,  5,  6,  7,  8,  9,  10, 11},
        {3, 4, 5,  6,  7,  8,  9,  10, 11, 12},
        {4, 5, 6,  7,  8,  9,  10, 11, 12, 13},
        {5, 6, 7,  8,  9,  10, 11, 12, 13, 14},
        {6, 7, 8,  9,  10, 11, 12, 13, 14, 15},
        {7, 8, 9,  10, 11, 12, 13, 14, 15, 16},
        {8, 9, 10, 11, 12, 13, 14, 15, 16, 17}
    };
    mat<10, 6> m2{
        {1,  2,  3,  4,  5,  6 },
        {2,  3,  4,  5,  6,  7 },
        {3,  4,  5,  6,  7,  8 },
        {4,  5,  6,  7,  8,  9 },
        {5,  6,  7,  8,  9,  10},
        {6,  7,  8,  9,  10, 11},
        {7,  8,  9,  10, 11, 12},
        {8,  9,  10, 11, 12, 13},
        {9,  10, 11, 12, 13, 14},
        {10, 11, 12, 13, 14, 15}
    };

    auto m12 = m1 * m2;

    mat<8, 6> m12_desired = {
        {385, 440, 495,  550,  605,  660 },
        {440, 505, 570,  635,  700,  765 },
        {495, 570, 645,  720,  795,  870 },
        {550, 635, 720,  805,  890,  975 },
        {605, 700, 795,  890,  985,  1080},
        {660, 765, 870,  975,  1080, 1185},
        {715, 830, 945,  1060, 1175, 1290},
        {770, 895, 1020, 1145, 1270, 1395}
    };
    EXPECT_EQ (m12, m12_desired);
}

template <size_t sz>
void
heavy_multiplication (mat<sz, sz>& mm0, mat<sz, sz>& mm1, mat<sz, sz>& mm2) {

    for (std::size_t i = 1; i <= mm0.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm0.count_cols(); ++j)
            mm0 (i, j) = i;

    for (std::size_t i = 1; i <= mm1.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm1.count_cols(); ++j)
            mm1 (i, j) = i;

    for (std::size_t i = 1; i <= mm2.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm2.count_cols(); ++j)
            mm2 (i, j) = j;

    auto prev_count_rows = mm0.count_rows();
    mm0 *= mm2;
    EXPECT_EQ (mm0.count_rows(), prev_count_rows);
    EXPECT_EQ (mm0.count_cols(), mm2.count_cols());

    const std::size_t p{mm0.count_cols()};
    for (std::size_t i = 1; i <= mm0.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm0.count_cols(); ++j)
            EXPECT_EQ (mm0 (i, j), p * i * j);

    auto mm12 = mm1 * mm2;
    for (std::size_t i = 1; i <= mm12.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm12.count_cols(); ++j)
            EXPECT_EQ (mm12 (i, j), mm1.count_cols() * i * j);
}

TEST (Matrix, HeavyMultiplication_512MiB) {
    mat<4096, 4096> mm0;
    mat<4096, 4096> mm1;
    mat<4096, 4096> mm2;

    heavy_multiplication (mm0, mm1, mm2);
}

TEST (Matrix, HeavyMultiplication_2GiB) {
    if (!mem_limit (2)) {
        GTEST_SKIP() << "Environment variable MEM_LIMIT not set of less than 2."
                        "  Skipping HeavyMultiplication_2GiB test.";
    }

    mat<8192, 8192> mm0;
    mat<8192, 8192> mm1;
    mat<8192, 8192> mm2;

    heavy_multiplication (mm0, mm1, mm2);
}

TEST (Matrix, HeavyMultiplication_8GiB) {
    if (!mem_limit (8)) {
        GTEST_SKIP() << "Environment variable MEM_LIMIT not set of less than 8."
                        "  Skipping HeavyMultiplication_8GiB test.";
    }

    mat<16384, 16384> mm0;
    mat<16384, 16384> mm1;
    mat<16384, 16384> mm2;

    heavy_multiplication (mm0, mm1, mm2);
}

TEST (Matrix, HeavyMultiplication_32GiB) {
    if (!mem_limit (32)) {
        GTEST_SKIP() << "Environment variable MEM_LIMIT not set of less than 32."
                        "  Skipping HeavyMultiplication_32GiB test.";
    }

    mat<32768, 32768> mm0;
    mat<32768, 32768> mm1;
    mat<32768, 32768> mm2;

    heavy_multiplication (mm0, mm1, mm2);
}

TEST (Matrix, AdditionSubtraction) {
    mat<4, 3> m1{};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1 (i, j) = double (i) + 2. * double (j);

    mat<4, 3> m2{};
    for (std::size_t i = 1; i <= m2.count_rows(); ++i)
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            m2 (i, j) = 3. * double (i) - double (j);

    auto m3 = m1 + m2;
    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            EXPECT_EQ (m3 (i, j), double (i) + 2. * double (j) + 3. * double (i) - double (j));

    auto m4 = m1 - m2;
    for (std::size_t i = 1; i <= m4.count_rows(); ++i)
        for (std::size_t j = 1; j <= m4.count_cols(); ++j)
            EXPECT_EQ (m4 (i, j), double (i) + 2. * double (j) - 3. * double (i) + double (j));
}

TEST (Matrix, ScalarOperation) {
    mat<6, 8> m1{};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1 (i, j) = i + j - 1.;

    m1 += 4.;
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1 (i, j) = i + j - 1. + 4.;

    m1 -= 4.;
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1 (i, j) = i + j - 1.;

    auto m12 = m1 * 2.;
    for (std::size_t i = 1; i <= m12.count_rows(); ++i)
        for (std::size_t j = 1; j <= m12.count_cols(); ++j)
            EXPECT_EQ (m12 (i, j), m1 (i, j) * 2.);

    auto m12n = 2. * m1;
    EXPECT_EQ (m12, m12n);

    auto m3 = m1 / 2.;
    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            EXPECT_EQ (m3 (i, j), m1 (i, j) / 2.);

    auto m4 = m1 + 4.;
    for (std::size_t i = 1; i <= m4.count_rows(); ++i)
        for (std::size_t j = 1; j <= m4.count_cols(); ++j)
            EXPECT_EQ (m4 (i, j), m1 (i, j) + 4.);

    auto m5 = 3. + m1;
    for (std::size_t i = 1; i <= m5.count_rows(); ++i)
        for (std::size_t j = 1; j <= m5.count_cols(); ++j)
            EXPECT_EQ (m5 (i, j), m1 (i, j) + 3.);

    auto m6 = m1 - 4.;
    for (std::size_t i = 1; i <= m6.count_rows(); ++i)
        for (std::size_t j = 1; j <= m6.count_cols(); ++j)
            EXPECT_EQ (m6 (i, j), m1 (i, j) - 4.);

    auto m7 = 3. - m1;
    for (std::size_t i = 1; i <= m7.count_rows(); ++i)
        for (std::size_t j = 1; j <= m7.count_cols(); ++j)
            EXPECT_EQ (m7 (i, j), 3. - m1 (i, j));
}

TEST (Matrix, Negation) {
    mat<10, 10> m1{};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1 (i, j) = i + j - 1.;

    auto m12 = -m1;
    for (std::size_t i = 1; i <= m12.count_rows(); ++i)
        for (std::size_t j = 1; j <= m12.count_cols(); ++j)
            EXPECT_EQ (m12 (i, j), -(i + j - 1.));
}

TEST (Matrix, Manipulation) {
    mat<4, 4> m0{
        {1, 2, 3, 4},
        {2, 3, 4, 5},
        {3, 4, 5, 6},
        {4, 5, 6, 7}
    };

    auto m1{m0};

    vec<4> v1{};
    m1.set_col (2, v1);

    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            if (j == 2) EXPECT_EQ (m1 (i, j), 0.);
            else EXPECT_EQ (m1 (i, j), m0 (i, j));

    m1 = m0;
    m1.set_row (3, v1);
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            if (i == 3) EXPECT_EQ (m1 (i, j), 0.);
            else EXPECT_EQ (m1 (i, j), m0 (i, j));
}

TEST (Matrix, MatrixVectorMultiplication) {
    auto m1 = identity<3>();
    auto v1 = vec<3>{1.};

    auto v11 = m1 * v1;
    EXPECT_EQ (v11, v1);

    auto v12 = v1 * m1;
    EXPECT_EQ (v12, v1);
}

TEST (Matrix, Trace) {
    auto m = identity<10>();
    EXPECT_EQ (tr (m), 10.);

    mat<2, 2> m1{
        {3, 7 },
        {1, -4}
    };
    EXPECT_EQ (tr (m1), -1.);

    mat<3, 3> m2{
        {1, 2, 3},
        {3, 2, 1},
        {2, 1, 3}
    };
    EXPECT_EQ (tr (m2), 6.);

    mat<4, 4> m3{
        {1.1,  2.2,  3.4,  -4.2},
        {4.1,  -3.4, 2.3,  1.2 },
        {-2.1, 1.4,  3.2,  4.1 },
        {3.3,  2.2,  -1.4, 4.1 }
    };
    EXPECT_EQ (tr (m3), 5.);
}

TEST (Matrix, Determinant) {
    auto m = identity<10>();
    EXPECT_EQ (det (m), 1.);

    mat<2, 2> m1{
        {3, 7 },
        {1, -4}
    };
    EXPECT_EQ (det (m1), -19.);

    mat<3, 3> m2{
        {1, 2, 3},
        {3, 2, 1},
        {2, 1, 3}
    };
    EXPECT_EQ (det (m2), -12.);

    mat<4, 4> m3{
        {1.1,  2.2,  3.4,  -4.2},
        {4.1,  -3.4, 2.3,  1.2 },
        {-2.1, 1.4,  3.2,  4.1 },
        {3.3,  2.2,  -1.4, 4.1 }
    };
    EXPECT_FLOAT_EQ (det (m3), -1028.5596);
}

TEST (Matrix, Inversion) {
    mat<3, 3> m1{
        {1, 2, 3},
        {4, 1, 6},
        {7, 8, 1}
    };
    auto m2 = inv (m1);
    auto m3 = m1 * m2;

    EXPECT_EQ (m3.count_rows(), m1.count_rows());
    EXPECT_EQ (m3.count_cols(), m1.count_cols());

    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            EXPECT_EQ (std::fabs (m3 (i, j) - (i == j ? 1. : 0.)) < TOL * 1e2, true);

    mat<3, 3> m4{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
    bool m4_invertible = true;
    try {
        auto m4_inv = inv (m4);
    }
    catch (...) {
        m4_invertible = false;
    }

    EXPECT_FALSE (m4_invertible);
}

TEST (Matrix, ConditionNumber) {
    mat<3, 3> m0{{1, 2, 3}, {2, 3, 4}, {3, 4, 5}};
    double cond_m0 = cond(m0);
    EXPECT_NEAR(cond_m0, 25931300179480048.0000, 10.);

    mat<2, 4> m1{{1, 2, 3, 4}, {1, 3, 5, 7}};
    double cond_m1 = cond(m1);
    EXPECT_NEAR(cond_m1, 25.451885122913790127086031, 1e-10);

    mat<2, 2> m2 {{4.1, 2.8}, {9.7, 6.6}};
    double cond_m2 = cond(m2);
    EXPECT_NEAR(cond_m2, 1622.9993838565622, 1e-12);

    auto m2_inv = inv(m2);
    EXPECT_NEAR(m2_inv(1, 1), -66.00000000000022737367544323205948, 1e-10);
    EXPECT_NEAR(m2_inv(1, 2), 28.00000000000009947598300641402602, 1e-10);
    EXPECT_NEAR(m2_inv(2, 1), 97.00000000000034106051316484808922, 1e-10);
    EXPECT_NEAR(m2_inv(2, 2), -41.00000000000014210854715202003717, 1e-10);

    mat<2, 2> m3 {{4.1, 2.8}, {9.671, 6.608}};
    auto m3_inv = inv(m3);
    EXPECT_NEAR(m3_inv(1, 1), 472.00000000002108890839735977351665, 1e-10);
    EXPECT_NEAR(m3_inv(1, 2), -200.00000000000892441676114685833454, 1e-10);
    EXPECT_NEAR(m3_inv(2, 1), -690.78571428574502988340100273489952, 1e-10);
    EXPECT_NEAR(m3_inv(2, 2), 292.85714285715590676772990263998508, 1e-10);
}


TEST (Matrix, SingularValueDecomposition) {
    mat<6, 4> M{
        {7.52,  -1.10, -7.95, 1.08 },
        {-0.76, 0.62,  9.34,  -7.10},
        {5.13,  6.62,  -5.66, 0.87 },
        {-4.75, 8.52,  5.75,  5.30 },
        {1.33,  4.91,  -5.49, -3.52},
        {-2.40, -6.77, 2.34,  3.95 }
    };

    // One possible SVD of the matrix above:
    //   M = U.Σ.V^†
    //   where
    //   M = ( 7.52   -1.1    -7.95    1.08
    //        -0.76    0.62    9.34   -7.1
    //         5.13    6.62   -5.66    0.87
    //        -4.75    8.52    5.75    5.3
    //         1.33    4.91   -5.49   -3.52
    //        -2.4    -6.77    2.34    3.95 )
    //   U = (-0.572674    0.177563    0.0056271    0.529022    0.58299
    //   -0.144023
    //         0.459422   -0.107528   -0.724027     0.417373    0.167946
    //         0.225273
    //        -0.450447   -0.413957    0.00417222   0.36286    -0.532307
    //        0.459023
    //         0.334096   -0.692623    0.494818     0.185129    0.358495
    //         -0.0318806
    //        -0.317397   -0.308371   -0.280347    -0.60983     0.437689
    //        0.402626
    //         0.213804    0.459053    0.390253     0.0900183   0.168744
    //         0.744771 )
    //   Σ = (18.366   0      0       0
    //         0      13.63   0       0
    //         0       0     10.8533  0
    //         0       0      0       4.49157
    //         0       0      0       0
    //         0       0      0       0 )
    //   V = (-0.516645    0.0786131  -0.280639   0.805071
    //        -0.121232   -0.992329   -0.0212036  0.0117076
    //         0.847064   -0.0945254  -0.141271   0.503578
    //        -0.0293912  -0.0129938   0.949123   0.313262 )

    auto SVD = svd (M);
    EXPECT_EQ (similar (SVD.U * diag<6, 4, 4> (SVD.S) * transpose (SVD.V), M, 0.000001), true);

    vec<4> singular_values{
        18.365978454889984, 13.629979679210999, 10.85333572722705, 4.491569094526893
    };
    EXPECT_EQ (norm (SVD.S - singular_values) < TOL * 1e2, true);
    EXPECT_EQ (similar (SVD.U * transpose (SVD.U), identity<6>(), TOL * 1e1), true);
    EXPECT_EQ (similar (SVD.V * transpose (SVD.V), identity<4>(), TOL * 1e1), true);
}

TEST (Matrix, Norm) {
    mat<2, 2> m1 = {
        {1., 2.},
        {3., 4.}
    };
    EXPECT_EQ (std::fabs (norm_frobenius (m1) - 5.47723) < 0.0001, true);

    mat<2, 3> m2 = {
        {1., 4., 6. },
        {7., 9., 10.}
    };
    EXPECT_EQ (std::fabs (norm_frobenius (m2) - 16.8226) < 0.0001, true);
}

TEST (Matrix, Eigensystem) {
    // Case study: symmetric matrix
    mat<4, 4> m1{
        {1, 2, 3, 4},
        {2, 2, 3, 4},
        {3, 3, 3, 4},
        {4, 4, 4, 4}
    };
    auto es1 = eigen (m1, eigen::vec);

    vec<4> eval1{-2.0531157635369963, -0.5146427793906165, -0.2943264517738027, 12.862084994701407};

    mat<4, 4> evec1{
        {-0.700349, -0.514374, 0.276678,  -0.410342},
        {-0.359234, 0.485103,  -0.663359, -0.442245},
        {0.156851,  0.541978,  0.650425,  -0.508532},
        {0.59654,   -0.454262, -0.245667, -0.614356}
    };

    // Eigenvalues
    EXPECT_TRUE (similar (gpw::vma::real (es1.eigvals), eval1, 0.00001));

    // Eigenvectors
    for (std::size_t j = 1; j < 5; ++j)
        EXPECT_EQ (
            collinear (gpw::vma::real (es1.eigvecs_rt.col (j)), evec1.col (j), 0.0001), true
        );

    // Case study: asymmetric matrix
    mat<4, 4> m2{
        {0, 2,  0,  1 },
        {2, 2,  3,  2 },
        {4, -3, 0,  1 },
        {6, 1,  -6, -5}
    };
    auto es2 = eigen (m2, eigen::vec);

    vec<4, complex_t> eval2{
        {4.177484212271297,   0.               },
        {-4.820108319356918,  0.               },
        {-1.1786879464571869, 3.19870513679807 },
        {-1.1786879464571869, -3.19870513679807}
    };

    mat<4, 4, complex_t> r_evec2{
        {{0.47184, 0.},    {0.132877, 0.},  {-0.0806923, 0.0788731}, {-0.0806923, -0.0788731}},
        {{0.783851, 0.},   {0.159705, 0.},  {0.279715, -0.175539},   {0.279715, 0.175539}    },
        {{-0.0145526, 0.}, {0.188274, 0.},  {0.422325, 0.431654},    {0.422325, -0.431654}   },
        {{0.403401, 0.},   {-0.959891, 0.}, {-0.71661, 0.0},         {-0.71661, 0.}          }
    };

    mat<4, 4, complex_t> l_evec2{
        {{0.739463, 0.}, {0.827813, 0.},  {-0.741732, 0.0},        {-0.741732, 0.0}        },
        {{0.622002, 0.}, {-0.309312, 0.}, {0.458239, -0.0363744},  {0.458239, 0.0363744}   },
        {{0.11783, 0.},  {-0.277047, 0.}, {0.0224866, 0.479373},   {0.0224866, -0.479373}  },
        {{0.228962, 0.}, {-0.377222, 0.}, {-0.0220256, 0.0879727}, {-0.0220256, -0.0879727}}
    };

    for (std::size_t j = 1; j < 5; ++j) {
        // Eigenvalue
        EXPECT_TRUE (std::abs (es2.eigvals (j) - eval2 (j)) < 0.00001);

        // Eigenvectors
        EXPECT_TRUE (collinear (es2.eigvecs_rt.col (j), r_evec2.col (j), 0.0001));
        EXPECT_TRUE (collinear (es2.eigvecs_lft.col (j), l_evec2.col (j), 0.0001));
    }
}

TEST (Vector, Complex) {
    vec<4, complex_t> cv1{};
    for (std::size_t i = 0; i < cv1.dim(); ++i)
        EXPECT_EQ ((cv1[i].real() == 0.) && (cv1[i].imag() == 0.), true);

    vec<4, complex_t> cv2{
        {1, 2},
        {2, 3},
        {3, 4},
        {4, 5}
    };
    for (std::size_t i = 0; i < cv2.dim(); ++i)
        EXPECT_EQ ((cv2[i].real() == i + 1.) && (cv2[i].imag() == i + 2.), true);

    std::cout << to_string (cv2, output_fmt::sht) << std::endl;
    std::cout << to_string (cv2) << std::endl;
    std::cout << to_string (cv2, output_fmt::ext) << std::endl;
    std::cout << to_string (cv2, output_fmt::sci) << std::endl;
    std::cout << to_string (cv2, output_fmt::scx) << std::endl;

    double            re[] = {1., 2., 3., 4.};
    vec<4, complex_t> cv3{re};
    for (std::size_t i = 0; i < cv3.dim(); ++i)
        EXPECT_EQ (cv3[i].real() == re[i] && cv3[i].imag() == 0., true);

    vec<4, complex_t> cv4 = conj (cv2);
    for (std::size_t i = 0; i < cv4.dim(); ++i)
        EXPECT_EQ (cv2[i].real() == cv4[i].real() && cv2[i].imag() == -cv4[i].imag(), true);

    vec<4>            rv1{re};
    vec<4, complex_t> cv5 = conj (rv1);
    for (std::size_t i = 0; i < cv5.dim(); ++i)
        EXPECT_EQ (cv5[i].real() == rv1[i] && cv5[i].imag() == 0., true);

    double im[] = {5., 6., 7., 8.};
    auto   cv6  = cvec<4> (re, im);
    for (std::size_t i = 0; i < cv6.dim(); ++i)
        EXPECT_EQ (cv6[i].real() == re[i] && cv6[i].imag() == im[i], true);

    vec<4> iv1{im};
    auto   cv7 = cvec<4> (rv1, iv1);
    EXPECT_EQ (cv7, cv6);

    EXPECT_EQ (gpw::vma::real (cv7), rv1);
    EXPECT_EQ (imag (cv7), iv1);
}

TEST (Matrix, Complex) {
    mat<2, 2, complex_t> m1{
        {{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}}
    };

    auto c11 = complex_t{1, 2};
    auto c12 = complex_t{3, 4};
    auto c21 = complex_t{5, 6};
    auto c22 = complex_t{7, 8};

    EXPECT_EQ (m1 (1, 1), c11);
    EXPECT_EQ (m1 (1, 2), c12);
    EXPECT_EQ (m1 (2, 1), c21);
    EXPECT_EQ (m1 (2, 2), c22);

    auto              col1 = m1.col (1);
    vec<2, complex_t> col1_ans{
        {1, 2},
        {5, 6}
    };
    EXPECT_EQ (col1, col1_ans);

    auto              col2 = m1.col (2);
    vec<2, complex_t> col2_ans{
        {3, 4},
        {7, 8}
    };
    EXPECT_EQ (col2, col2_ans);

    auto              row1 = m1.row (1);
    vec<2, complex_t> row1_ans{
        {1, 2},
        {3, 4}
    };
    EXPECT_EQ (row1, row1_ans);

    auto              row2 = m1.row (2);
    vec<2, complex_t> row2_ans{
        {5, 6},
        {7, 8}
    };
    EXPECT_EQ (row2, row2_ans);

    std::cout << to_string (m1) << std::endl;

    vec<2, complex_t> v1{
        {1, 2},
        {3, 4}
    };

    auto              mv = m1 * v1;
    vec<2, complex_t> mv_ans{
        {-10, 28},
        {-18, 68}
    };
    EXPECT_EQ (mv, mv_ans);

    std::cout << to_string (mv) << std::endl;

    mat<2, 2, complex_t> m2 = conj (m1);

    EXPECT_EQ (m2 (1, 1).real() == c11.real() && m2 (1, 1).imag() == -c11.imag(), true);
    EXPECT_EQ (m2 (1, 2).real() == c12.real() && m2 (1, 2).imag() == -c12.imag(), true);
    EXPECT_EQ (m2 (2, 1).real() == c21.real() && m2 (2, 1).imag() == -c21.imag(), true);
    EXPECT_EQ (m2 (2, 2).real() == c22.real() && m2 (2, 2).imag() == -c22.imag(), true);

    mat<3, 3> rm1{
        {1, 2, 3},
        {2, 3, 4},
        {3, 4, 5}
    };

    auto cm1 = conj (rm1);
    for (std::size_t i = 1; i <= cm1.count_rows(); ++i)
        for (std::size_t j = 1; j <= cm1.count_cols(); ++j)
            EXPECT_EQ (cm1 (i, j).real() == rm1 (i, j) && cm1 (i, j).imag() == 0., true);

    double re[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double im[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    auto   cm2  = cmat<2, 5> (re, im);
    for (std::size_t i = 1; i <= cm2.count_rows(); ++i)
        for (std::size_t j = 1; j <= cm2.count_cols(); ++j) {
            auto c = complex_t{re[(i - 1) * 5 + (j - 1)], im[(i - 1) * 5 + (j - 1)]};
            EXPECT_EQ (cm2 (i, j), c);
        }
}

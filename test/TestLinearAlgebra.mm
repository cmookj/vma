//
//  TestLinearAlgebra.mm
//  TestLinearAlgebra
//
//  Created by Changmook Chun on 2022-11-25.
//

#import <XCTest/XCTest.h>
#include "linear_algebra.hpp"

using namespace std;
using namespace tls::blat;

@interface TestLinearAlgebra : XCTestCase
{
    mat<256, 256> mm0;
    
    mat<1024, 256> mm1;
    mat<256, 256> mm2;
}
@end

@implementation TestLinearAlgebra

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
    for (std::size_t i = 1; i <= mm0.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm0.count_cols(); ++j)
            mm0(i, j) = i;
    
    for (std::size_t i = 1; i <= mm1.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm1.count_cols(); ++j)
            mm1(i, j) = i;
    
    for (std::size_t i = 1; i <= mm2.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm2.count_cols(); ++j)
            mm2(i, j) = j;
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testVectorCreation {
    vec<5> v1;
    for (std::size_t i = 1; i <= v1.dim(); ++i)
        XCTAssert(v1(i) == 0.);
    
    vec<5> v2{1.};
    for (std::size_t i = 1; i <= v2.dim(); ++i)
        XCTAssert(v2(i) == 1.);
    
    double a[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    vec<5> v3{a};
    for (std::size_t i = 1; i <= v3.dim(); ++i)
        XCTAssert(v3(i) == a[i - 1]);
    
    vec<5> v4{1, 3, 5, 7, 9};
    for (std::size_t i = 1; i <= v4.dim(); ++i)
        XCTAssert(v4(i) == 2.*i - 1.);
    
    vec<5> v5{1.0e1, 3.0e2, 5.0e3, 7.0e4, 9.0e5};
    for (std::size_t i = 1; i <= v5.dim(); ++i)
        XCTAssert(v5(i) == (2.*i - 1.)*std::pow(10., i));
    
    vec<5> v6{-1, 3, -5, 7, -9};
    for (std::size_t i = 1; i <= v6.dim(); ++i)
        XCTAssert(v6(i) == (2.*i - 1.) * std::pow(-1., i));
    
    std::cout << to_string(v6, output_fmt::sht) << std::endl;
    std::cout << to_string(v6) << std::endl;
    std::cout << to_string(v6, output_fmt::ext) << std::endl;
    std::cout << to_string(v6, output_fmt::sci) << std::endl;
    std::cout << to_string(v6, output_fmt::scx) << std::endl;
    
    auto v7 = rand<100>();
    
    auto v8 = randn<100>();
}

- (void)testCollinearVectors {
    vec<9> v1 {1., 2., 3., 4., 5., 6., 7., 8., 9.};
    vec<9> v2 {2.*v1};
    
    XCTAssert(collinear(v1, v2));
    
    vec<9> v3 {1., 0., 3., 0., 5., 0., 7., 0., 9.};
    vec<9> v4 {0.3 * v3};
    
    XCTAssert(collinear(v3, v4));
    
    vec<4, complex_t> cv1 {{1., 2.}, {3., 4.}, {-2., -1.}, {-5., 3.}};
    vec<4, complex_t> cv2 {{2., 4.}, {6., 8.}, {-4., -2.}, {-10., 6.}};
    XCTAssert(collinear(cv1, cv2));
    
    vec<4, complex_t> cv3 {{1., 2.}, {3., 0.}, {0., -1.}, {-5., 3.}};
    vec<4, complex_t> cv4 {{2., 4.}, {6., 0.}, {0., -2.}, {-10., 6.}};
    XCTAssert(collinear(cv3, cv4));
}

- (void)testVectorComparison {
    vec<5> v1 {1, 2, 3, 4, 5};
    vec<5> v2 {1. + EPS, 2, 3, 4, 5};
    
    XCTAssert (v1 == v1);
    XCTAssert (v1 != v2);
    
    XCTAssert (close(v1, v2));
}

- (void)testIndexAssignmentComparison {
    vec<5> v1{1, 2, 3, 4, 5};
    
    v1(2)= -v1(2);
    v1(4)= -v1(4);
    XCTAssert(v1(1) == 1.);
    XCTAssert(v1(2) == -2.);
    XCTAssert(v1(3) == 3.);
    XCTAssert(v1(4) == -4.);
    XCTAssert(v1(5) == 5.);
    
    auto v2 {v1};
    XCTAssert(v2 == v2);
    XCTAssert(v2 == v1);
    
    auto v3 {v1};
    XCTAssert(v3 == v1);
}

- (void)testAdditionSubtraction {
    vec<5> v1{1, 3, 5, 7, 9};
    vec<5> v2(1.);
    
    auto v3 {v1 + v2};
    for (std::size_t i = 1; i < 6; ++i)
        XCTAssert(v3(i) == v1(i) + 1.);

    auto v4 {v1 - v2};
    for (std::size_t i = 1; i < 6; ++i)
        XCTAssert(v4(i) == v1(i) - 1.);

    auto v5 = -v2;
    for (std::size_t i = 1; i < 6; ++i)
        XCTAssert(v5(i) + v2(i) == 0.);

    v1 += 3;
    for (std::size_t i = 1; i < 6; ++i)
        XCTAssert(v1(i) == 2.*i - 1. + 3.);
    
    v1 -= 3;
    for (std::size_t i = 1; i < 6; ++i)
        XCTAssert(v1(i) == 2.*i - 1.);
}

- (void)testMultiplicationDivision {
    const double v[3] = {1., 3., 5.};
    vec<3> v1{v};
    
    for (std::size_t i = 0; i < 3; ++i)
        XCTAssert(v1[i] == v[i]);

    auto v2 {v1*2.};
    for (std::size_t i = 0; i < 3; ++i)
        XCTAssert(v2[i] == v1[i] * 2.);

    v2 *= 2.0;
    for (std::size_t i = 0; i < 3; ++i)
        XCTAssert(v2[i] == v[i] * 4.);
    
    auto v3 = v1/2.0;
    for (std::size_t i = 0; i < 3; ++i)
        XCTAssert(v3[i] == v1[i] / 2.);
    
    v3 /= 2.0;
    for (std::size_t i = 0; i < 3; ++i)
        XCTAssert(v3[i] == v1[i] / 4.);
    
    auto v4 = .5 * v1 * 3./2.;
    for (std::size_t i = 0; i < 3; ++i)
        XCTAssert(v4[i] == .5 * v1[i] * 3./2.);
}

- (void)testVecMatMultiplication {
    mat<3, 3> m1{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    vec<3> v1{1, 3, 5};
 
    auto v2 = m1*v1; // 22, 49, 76
    XCTAssert(v2[0] == 22.);
    XCTAssert(v2[1] == 49.);
    XCTAssert(v2[2] == 76.);
    
    auto v3 = v1*m1; // 48, 57, 66
    XCTAssert(v3[0] == 48.);
    XCTAssert(v3[1] == 57.);
    XCTAssert(v3[2] == 66.);
}

- (void)testNormInnerProductNormalize {
    vec<3> v0 {};
    
    vec<3> v1 {1, 3, 5};
    XCTAssert(norm(v1, 1) == 1. + 3. + 5.);
    XCTAssert(norm(v1, 2) == std::sqrt(1. + 9. + 25.));
    XCTAssert(norm_inf(v1) == 5.);
    
    vec<3> v2 {-1., 3, -5.};
    XCTAssert(norm(v2, 1) == 1. + 3. + 5.);
    XCTAssert(norm(v2, 2) == std::sqrt(1. + 9. + 25.));
    XCTAssert(norm_inf(v2) == -5.);

    vec<3> v3 {2, 4, 6};
    XCTAssert(inner(v1, v3) == 1.*2. + 3.*4. + 5.*6.);
    XCTAssert(dist(v0, v3) == norm(v3));
    
    vec<6> v4 {1, 5, 2, 4, 7, 6};
    XCTAssert(norm(normalize(v4)) == 1.);
    
    vec<4> v5 {1};
    vec<4> v6 {2};
    XCTAssert(dist(v5, v6) == std::sqrt(4 * 1.));
}

- (void)testMatrixCreationIndexing {
    mat<4, 4> m0 {};
    for (std::size_t i = 1; i <= m0.count_rows(); ++i )
        for (std::size_t j = 1; j <= m0.count_cols(); ++j)
            XCTAssert(m0(i, j) == 0. );
    
    mat<3, 4> m1 {{1, 2, 3, 4}, {2., 3., 4., 5.}, {3, 4, 5., 6}};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i )
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            XCTAssert(m1(i, j) == (i - 1.) + j);
    
    std::cout << to_string(m1, output_fmt::sht) << std::endl;
    std::cout << to_string(m1) << std::endl;
    std::cout << to_string(m1, output_fmt::ext) << std::endl;
    std::cout << to_string(m1, output_fmt::sci) << std::endl;
    std::cout << to_string(m1, output_fmt::scx) << std::endl;
    
    mat<3, 4> m2 {{1., -2, 3, -4}, {-2., 3, -4, 5.}, {3., -4, 5, -6}};
    for (std::size_t i = 1; i <= m2.count_rows(); ++i )
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            XCTAssert(m2(i, j) == ((i - 1.) + j) * std::pow(-1., i + j));
    
    std::array<double, 9> el0 {1., 2., 3., 4., 5., 6., 7., 8., 9.};
    std::array<double, 9> el00 {1., 2., 3., 4., 5., 6., 7., 8., 9.};
    mat<3,3> m21{std::move(el0)};
    XCTAssert(m21.count_rows() == 3);
    XCTAssert(m21.count_cols() == 3);
    XCTAssert(m21.elem() == el00);
    
    auto m3 = identity<10>();
    for (std::size_t i = 1; i <= m3.count_rows(); ++i )
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            XCTAssert(m3(i, j) == (i == j ? 1. : 0.) );
    
    auto m4 = rand<100, 100>();
    
    auto m5 = randn<100, 100>();
    
    std::array<double, 10> el1 {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    auto m6 = diag<10>(el1);
    for (std::size_t i = 1; i <= m6.count_rows(); ++i)
        for (std::size_t j = 1; j <= m6.count_cols(); ++j)
            XCTAssert(m6(i, j) == (i == j ? el1[i - 1] : 0.));
    
    double el2[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    auto m7 = diag<10>(el2);
    for (std::size_t i = 1; i <= m7.count_rows(); ++i)
        for (std::size_t j = 1; j <= m7.count_cols(); ++j)
            XCTAssert(m7(i, j) == (i == j ? el1[i - 1] : 0.));
}

- (void)testMatrixComparison {
    std::array<double, 12> e1 {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    std::array<double, 12> e2 {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.};
    mat<3, 4> m1 {std::move(e1)};
    mat<3, 4> m2 {std::move(e2)};
    
    XCTAssert(m1 == m2);
}

- (void)testMatrixVecExtraction {
    mat<8, 10> m1 {
        {1,  2,  3,  4,  5,  6,  7,  8,  9, 10},
        {2,  3,  4,  5,  6,  7,  8,  9, 10, 11},
        {3,  4,  5,  6,  7,  8,  9, 10, 11, 12},
        {4,  5,  6,  7,  8,  9, 10, 11, 12, 13},
        {5,  6,  7,  8,  9, 10, 11, 12, 13, 14},
        {6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
        {7,  8,  9, 10, 11, 12, 13, 14, 15, 16},
        {8,  9, 10, 11, 12, 13, 14, 15, 16, 17}
    };
    
    // Extraction of columns
    vec<8> col0 {};
    XCTAssert(m1.col(0) == col0);
    
    vec<8> col11 {};
    XCTAssert(m1.col(11) == col11);
    
    vec<8> col1 {1, 2, 3, 4, 5, 6, 7, 8};
    XCTAssert(m1.col(1) == col1);
    
    vec<8> col3 {3, 4, 5, 6, 7, 8, 9, 10};
    XCTAssert(m1.col(3) == col3);
    
    vec<8> col9 {9, 10, 11, 12, 13, 14, 15, 16};
    XCTAssert(m1.col(9) == col9);
    
    // Extraction of rows
    vec<10> row0 {};
    XCTAssert(m1.row(0) == row0);
    
    vec<10> row9 {};
    XCTAssert(m1.row(9) == row9);
    
    vec<10> row1 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    XCTAssert(m1.row(1) == row1);
    
    vec<10> row4 {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    XCTAssert(m1.row(4) == row4);
    
    vec<10> row7 {7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    XCTAssert(m1.row(7) == row7);
}

- (void)testMatrixTranspose {
    mat<8, 10> m1 {
        {1,  2,  3,  4,  5,  6,  7,  8,  9, 10},
        {2,  3,  4,  5,  6,  7,  8,  9, 10, 11},
        {3,  4,  5,  6,  7,  8,  9, 10, 11, 12},
        {4,  5,  6,  7,  8,  9, 10, 11, 12, 13},
        {5,  6,  7,  8,  9, 10, 11, 12, 13, 14},
        {6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
        {7,  8,  9, 10, 11, 12, 13, 14, 15, 16},
        {8,  9, 10, 11, 12, 13, 14, 15, 16, 17}
    };
    auto m2 = transpose(m1);
    
    XCTAssert(m1.count_rows() == m2.count_cols());
    XCTAssert(m1.count_cols() == m2.count_rows());
    
    for (std::size_t i = 1; i <= m1.count_rows(); ++i )
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            XCTAssert(m1(i, j) == (i - 1.) + j);
    
    for (std::size_t i = 1; i <= m2.count_rows(); ++i )
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            XCTAssert(m2(i, j) == (i - 1.) + j);
}

- (void)testMatrixMultiplication {
    // Small matrices
    mat<8, 10> m1 {
        {1,  2,  3,  4,  5,  6,  7,  8,  9, 10},
        {2,  3,  4,  5,  6,  7,  8,  9, 10, 11},
        {3,  4,  5,  6,  7,  8,  9, 10, 11, 12},
        {4,  5,  6,  7,  8,  9, 10, 11, 12, 13},
        {5,  6,  7,  8,  9, 10, 11, 12, 13, 14},
        {6,  7,  8,  9, 10, 11, 12, 13, 14, 15},
        {7,  8,  9, 10, 11, 12, 13, 14, 15, 16},
        {8,  9, 10, 11, 12, 13, 14, 15, 16, 17}
    };
    mat<10, 6> m2 {
        { 1,  2,  3,  4,  5,  6},
        { 2,  3,  4,  5,  6,  7},
        { 3,  4,  5,  6,  7,  8},
        { 4,  5,  6,  7,  8,  9},
        { 5,  6,  7,  8,  9, 10},
        { 6,  7,  8,  9, 10, 11},
        { 7,  8,  9, 10, 11, 12},
        { 8,  9, 10, 11, 12, 13},
        { 9, 10, 11, 12, 13, 14},
        {10, 11, 12, 13, 14, 15}
    };
    
    auto m12 = m1 * m2;
    
    mat<8, 6> m12_desired = {
        {385, 440,  495,  550,  605,  660},
        {440, 505,  570,  635,  700,  765},
        {495, 570,  645,  720,  795,  870},
        {550, 635,  720,  805,  890,  975},
        {605, 700,  795,  890,  985, 1080},
        {660, 765,  870,  975, 1080, 1185},
        {715, 830,  945, 1060, 1175, 1290},
        {770, 895, 1020, 1145, 1270, 1395}
    };
    XCTAssert(m12 == m12_desired);
    
    auto prev_count_rows = mm0.count_rows();
    mm0 *= mm2;
    XCTAssert(mm0.count_rows() == prev_count_rows);
    XCTAssert(mm0.count_cols() == mm2.count_cols());

    const std::size_t p {mm0.count_cols()};
    for (std::size_t i = 1; i <= mm0.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm0.count_cols(); ++j)
            XCTAssert(mm0(i, j) == p * i * j);

    auto mm12 = mm1 * mm2;
    for (std::size_t i = 1; i <= mm12.count_rows(); ++i)
        for (std::size_t j = 1; j <= mm12.count_cols(); ++j)
            XCTAssert(mm12(i, j) == mm1.count_cols() * i * j);
}

- (void)testMatrixAdditionSubtraction {
    mat<4, 3> m1 {};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1(i, j) = double(i) + 2.*double(j);

    mat<4, 3> m2 {};
    for (std::size_t i = 1; i <= m2.count_rows(); ++i)
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            m2(i, j) = 3.*double(i) - double(j);

    auto m3 = m1 + m2;
    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            XCTAssert(m3(i, j) == double(i) + 2.*double(j) + 3.*double(i) - double(j));
    
    auto m4 = m1 - m2;
    for (std::size_t i = 1; i <= m4.count_rows(); ++i)
        for (std::size_t j = 1; j <= m4.count_cols(); ++j)
            XCTAssert(m4(i, j) == double(i) + 2.*double(j) - 3.*double(i) + double(j));
}

- (void)testMatrixScalarOperation {
    mat<6, 8> m1 {};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1(i, j) = i + j - 1.;
    
    m1 += 4.;
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1(i, j) = i + j - 1. + 4.;
    
    m1 -= 4.;
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1(i, j) = i + j - 1.;

    auto m12 = m1 * 2.;
    for (std::size_t i = 1; i <= m12.count_rows(); ++i)
        for (std::size_t j = 1; j <= m12.count_cols(); ++j)
            XCTAssert(m12(i, j) == m1(i, j) * 2.);
    
    auto m12n = 2. * m1;
    XCTAssert(m12 == m12n);
    
    auto m3 = m1 / 2.;
    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            XCTAssert(m3(i, j) == m1(i, j) / 2.);

    auto m4 = m1 + 4.;
    for (std::size_t i = 1; i <= m4.count_rows(); ++i)
        for (std::size_t j = 1; j <= m4.count_cols(); ++j)
            XCTAssert(m4(i, j) == m1(i, j) + 4.);
    
    auto m5 = 3. + m1;
    for (std::size_t i = 1; i <= m5.count_rows(); ++i)
        for (std::size_t j = 1; j <= m5.count_cols(); ++j)
            XCTAssert(m5(i, j) == m1(i, j) + 3.);
    
    auto m6 = m1 - 4.;
    for (std::size_t i = 1; i <= m6.count_rows(); ++i)
        for (std::size_t j = 1; j <= m6.count_cols(); ++j)
            XCTAssert(m6(i, j) == m1(i, j) - 4.);
    
    auto m7 = 3. - m1;
    for (std::size_t i = 1; i <= m7.count_rows(); ++i)
        for (std::size_t j = 1; j <= m7.count_cols(); ++j)
            XCTAssert(m7(i, j) == 3. - m1(i, j));
}

- (void)testMatrixNegation {
    mat<10, 10> m1 {};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            m1(i, j) = i + j - 1.;
    
    auto m12 = -m1;
    for (std::size_t i = 1; i <= m12.count_rows(); ++i)
        for (std::size_t j = 1; j <= m12.count_cols(); ++j)
            XCTAssert(m12(i, j) == -(i + j - 1.));
}

- (void)testMatrixManipulation {
    mat<4, 4> m0 {
        {1, 2, 3, 4},
        {2, 3, 4, 5},
        {3, 4, 5, 6},
        {4, 5, 6, 7}
    };
    
    auto m1 {m0};
    
    vec<4> v1 {};
    m1.set_col(2, v1);
    
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            if (j == 2)
                XCTAssert(m1(i, j) == 0.);
            else
                XCTAssert(m1(i, j) == m0(i, j));
    
    m1 = m0;
    m1.set_row(3, v1);
    for (std::size_t i = 1; i <= m1.count_rows(); ++i)
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            if (i == 3)
                XCTAssert(m1(i, j) == 0.);
            else
                XCTAssert(m1(i, j) == m0(i, j));

}

- (void)testMatrixVectorMultiplication {
    auto m1 = identity<3>();
    auto v1 = vec<3>{1.};
    
    auto v11 = m1 * v1;
    XCTAssert(v11 == v1);
    
    auto v12 = v1 * m1;
    XCTAssert(v12 == v1);
}

- (void)testMatrixTrace {
    auto m = identity<10>();
    XCTAssert(tr(m) == 10.);
    
    mat<2,2> m1 {{3, 7}, {1, -4}};
    XCTAssert(tr(m1) == -1.);
    
    mat<3,3> m2 {{1, 2, 3}, {3, 2, 1}, {2, 1, 3}};
    XCTAssert(tr(m2) == 6.);
    
    mat<4,4> m3 {
        {1.1, 2.2, 3.4, -4.2},
        {4.1, -3.4, 2.3, 1.2},
        {-2.1, 1.4, 3.2, 4.1},
        {3.3, 2.2, -1.4, 4.1}
    };
    XCTAssert(tr(m3) == 5.);
}

- (void)testMatrixDeterminant {
    auto m = identity<10>();
    XCTAssert(det(m) == 1.);
    
    mat<2,2> m1 {{3, 7}, {1, -4}};
    XCTAssert(det(m1) == -19.);
    
    mat<3,3> m2 {{1, 2, 3}, {3, 2, 1}, {2, 1, 3}};
    XCTAssert(det(m2) == -12.);
    
    mat<4,4> m3 {
        {1.1, 2.2, 3.4, -4.2},
        {4.1, -3.4, 2.3, 1.2},
        {-2.1, 1.4, 3.2, 4.1},
        {3.3, 2.2, -1.4, 4.1}
    };
    XCTAssert(det(m3) == -1028.5596);
}

- (void)testMatrixInversion {
    mat<3, 3> m1 {{1, 2, 3}, {4, 1, 6}, {7, 8, 1}};
    auto m2 = inv(m1);
    auto m3 = m1 * m2;
    
    XCTAssert(m3.count_rows() == m1.count_rows());
    XCTAssert(m3.count_cols() == m1.count_cols());
    
    for (std::size_t i = 1; i <= m3.count_rows(); ++i)
        for (std::size_t j = 1; j <= m3.count_cols(); ++j)
            XCTAssert(std::fabs(m3(i, j) - (i == j ? 1. : 0.)) < TOLERANCE*1e2);
}

- (void)testMatrixSingularValueDecomposition {
    mat<6, 4> M {
        { 7.52, -1.10, -7.95,  1.08},
        {-0.76,  0.62,  9.34, -7.10},
        { 5.13,  6.62, -5.66,  0.87},
        {-4.75,  8.52,  5.75,  5.30},
        { 1.33,  4.91, -5.49, -3.52},
        {-2.40, -6.77,  2.34,  3.95}
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
    //   U = (-0.572674    0.177563    0.0056271    0.529022    0.58299    -0.144023
    //         0.459422   -0.107528   -0.724027     0.417373    0.167946    0.225273
    //        -0.450447   -0.413957    0.00417222   0.36286    -0.532307    0.459023
    //         0.334096   -0.692623    0.494818     0.185129    0.358495   -0.0318806
    //        -0.317397   -0.308371   -0.280347    -0.60983     0.437689    0.402626
    //         0.213804    0.459053    0.390253     0.0900183   0.168744    0.744771 )
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
    
    auto SVD = svd(M);
    XCTAssert(approx(SVD.U * diag<6, 4, 4>(SVD.S) * transpose(SVD.V), M, 0.000001));
    
    vec<4> singular_values {
        18.365978454889984,
        13.629979679210999,
        10.85333572722705,
        4.491569094526893
    };
    XCTAssert(norm(SVD.S - singular_values) < TOLERANCE * 1e2);
    XCTAssert(approx(SVD.U * transpose(SVD.U), identity<6>(), TOLERANCE * 1e1));
    XCTAssert(approx(SVD.V * transpose(SVD.V), identity<4>(), TOLERANCE * 1e1));
}

- (void)testMatrixNorm {
    mat<2, 2> m1 = {{1., 2.}, {3., 4.}};
    XCTAssert(std::fabs(norm_frobenius(m1) - 5.47723) < 0.0001);
    
    mat<2, 3> m2 = {{1., 4., 6.}, {7., 9., 10.}};
    XCTAssert(std::fabs(norm_frobenius(m2) - 16.8226) < 0.0001);
}

- (void)testEigensystem {
    // Case study: symmetric matrix
    mat<4, 4> m1 {
        {1, 2, 3, 4},
        {2, 2, 3, 4},
        {3, 3, 3, 4},
        {4, 4, 4, 4}
    };
    auto es1 = eigen(m1, eigen::vec);
    
    vec<4> eval1 {
        -2.0531157635369963,
        -0.5146427793906165,
        -0.2943264517738027,
        12.862084994701407
    };
    
    mat<4, 4> evec1 {
        {-0.700349, -0.514374,  0.276678, -0.410342},
        {-0.359234,  0.485103, -0.663359, -0.442245},
        { 0.156851,  0.541978,  0.650425, -0.508532},
        { 0.59654,  -0.454262, -0.245667, -0.614356}
    };
    
    // Eigenvalues
    XCTAssert(close(real(es1.eigvals), eval1, 0.00001));
    
    // Eigenvectors
    for (std::size_t j = 1; j < 5; ++j)
        XCTAssert(close_collinear(real(es1.eigvecs_rt.col(j)), evec1.col(j), 0.0001));
        
    // Case study: asymmetric matrix
    mat<4, 4> m2 {
        {0, 2, 0, 1},
        {2, 2, 3, 2},
        {4, -3, 0, 1},
        {6, 1, -6, -5}
    };
    auto es2 = eigen(m2, eigen::vec);
    
    vec<4, complex_t> eval2 {
        {  4.177484212271297, 0.},
        { -4.820108319356918, 0.},
        { -1.1786879464571869, 3.19870513679807},
        { -1.1786879464571869, -3.19870513679807}
    };

    mat<4, 4, complex_t> r_evec2 {
        {{ 0.47184,   0.}, { 0.132877, 0.}, {-0.0806923, 0.0788731},{-0.0806923, -0.0788731}},
        {{ 0.783851,  0.}, { 0.159705, 0.}, { 0.279715, -0.175539}, {  0.279715,  0.175539}},
        {{-0.0145526, 0.}, { 0.188274, 0.}, { 0.422325,  0.431654}, {  0.422325, -0.431654}},
        {{ 0.403401,  0.}, {-0.959891, 0.}, {-0.71661,   0.0},      { -0.71661,   0.}}
    };
    
    mat<4, 4, complex_t> l_evec2 {
        {{0.739463, 0.}, { 0.827813, 0.}, {-0.741732,  0.0},       {-0.741732,   0.0}},
        {{0.622002, 0.}, {-0.309312, 0.}, { 0.458239, -0.0363744}, { 0.458239,   0.0363744}},
        {{0.11783,  0.}, {-0.277047, 0.}, { 0.0224866, 0.479373},  { 0.0224866, -0.479373}},
        {{0.228962, 0.}, {-0.377222, 0.}, {-0.0220256, 0.0879727}, {-0.0220256, -0.0879727}}
    };

    for (std::size_t j = 1; j < 5; ++j) {
        auto eval = es2.eigvals[j];
        
        // Eigenvalue
        XCTAssert(std::abs(es2.eigvals(j) - eval2(j)) < 0.00001);
        
        // Eigenvectors
        XCTAssert(close_collinear(es2.eigvecs_rt.col(j), r_evec2.col(j), 0.0001));
        XCTAssert(close_collinear(es2.eigvecs_lft.col(j), l_evec2.col(j), 0.0001));
    }
}

- (void)testComplexVector {
    vec<4, complex_t> cv1 {};
    for (std::size_t i = 0; i < cv1.dim(); ++i)
        XCTAssert((cv1[i].real() == 0.) && (cv1[i].imag() == 0.));
    
    vec<4, complex_t> cv2 {{1, 2}, {2, 3}, {3, 4}, {4, 5}};
    for (std::size_t i = 0; i < cv2.dim(); ++i)
        XCTAssert((cv2[i].real() == i + 1.) && (cv2[i].imag() == i + 2.));

    std::cout << to_string(cv2, output_fmt::sht) << std::endl;
    std::cout << to_string(cv2) << std::endl;
    std::cout << to_string(cv2, output_fmt::ext) << std::endl;
    std::cout << to_string(cv2, output_fmt::sci) << std::endl;
    std::cout << to_string(cv2, output_fmt::scx) << std::endl;

    double re[] = {1., 2., 3., 4.};
    vec<4, complex_t> cv3 {re};
    for (std::size_t i = 0; i < cv3.dim(); ++i)
        XCTAssert(cv3[i].real() == re[i] && cv3[i].imag() == 0.);
    
    vec<4, complex_t> cv4 = conj(cv2);
    for (std::size_t i = 0; i < cv4.dim(); ++i)
        XCTAssert(cv2[i].real() == cv4[i].real() && cv2[i].imag() == -cv4[i].imag());
    
    vec<4> rv1 {re};
    vec<4, complex_t> cv5 = conj(rv1);
    for (std::size_t i = 0; i < cv5.dim(); ++i)
        XCTAssert(cv5[i].real() == rv1[i] && cv5[i].imag() == 0.);
    
    double im[] = {5., 6., 7., 8.};
    auto cv6 = cvec<4>(re, im);
    for (std::size_t i = 0; i < cv6.dim(); ++i)
        XCTAssert(cv6[i].real() == re[i] && cv6[i].imag() == im[i]);
    
    vec<4> iv1 {im};
    auto cv7 = cvec<4>(rv1, iv1);
    XCTAssert(cv7 == cv6);
    
    XCTAssert(real(cv7) == rv1);
    XCTAssert(imag(cv7) == iv1);
}

- (void)testMatrixComplex {
    mat<2, 2, complex_t> m1 {
        {{1, 2}, {3, 4}},
        {{5, 6}, {7, 8}}
    };
    
    auto c11 = complex_t{1, 2};
    auto c12 = complex_t{3, 4};
    auto c21 = complex_t{5, 6};
    auto c22 = complex_t{7, 8};
    
    XCTAssert(m1(1, 1) == c11);
    XCTAssert(m1(1, 2) == c12);
    XCTAssert(m1(2, 1) == c21);
    XCTAssert(m1(2, 2) == c22);
    
    auto col1 = m1.col(1);
    vec<2, complex_t> col1_ans {{1, 2}, {5, 6}};
    XCTAssert(col1 == col1_ans);
    
    auto col2 = m1.col(2);
    vec<2, complex_t> col2_ans {{3, 4}, {7, 8}};
    XCTAssert(col2 == col2_ans);
    
    auto row1 = m1.row(1);
    vec<2, complex_t> row1_ans {{1, 2}, {3, 4}};
    XCTAssert(row1 == row1_ans);
    
    auto row2 = m1.row(2);
    vec<2, complex_t> row2_ans {{5, 6}, {7, 8}};
    XCTAssert(row2 == row2_ans);
    
    std::cout << to_string(m1) << std::endl;
    
    vec<2, complex_t> v1 {
        {1, 2}, {3, 4}
    };
    
    auto mv = m1 * v1;
    vec<2, complex_t> mv_ans {{-10, 28}, {-18, 68}};
    XCTAssert(mv == mv_ans);
    
    std::cout << to_string(mv) << std::endl;
    
    mat<2, 2, complex_t> m2 = conj(m1);
        
    XCTAssert(m2(1, 1).real() == c11.real() && m2(1, 1).imag() == -c11.imag());
    XCTAssert(m2(1, 2).real() == c12.real() && m2(1, 2).imag() == -c12.imag());
    XCTAssert(m2(2, 1).real() == c21.real() && m2(2, 1).imag() == -c21.imag());
    XCTAssert(m2(2, 2).real() == c22.real() && m2(2, 2).imag() == -c22.imag());
    
    mat<3, 3> rm1 {
        {1, 2, 3}, {2, 3, 4}, {3, 4, 5}
    };
    
    auto cm1 = conj(rm1);
    for (std::size_t i = 1; i <= cm1.count_rows(); ++i)
        for (std::size_t j = 1; j <= cm1.count_cols(); ++j)
            XCTAssert(cm1(i, j).real() == rm1(i, j) && cm1(i, j).imag() == 0.);
    
    double re[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double im[] = {10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    auto cm2 = cmat<2, 5>(re, im);
    for (std::size_t i = 1; i <= cm2.count_rows(); ++i)
        for (std::size_t j = 1; j <= cm2.count_cols(); ++j) {
            auto c = complex_t{re[(i - 1) * 5 + (j - 1)], im[(i - 1) * 5 + (j - 1)]};
            XCTAssert(cm2(i, j) == c);
        }
}

- (void)testPerformanceMatrixMultiplication {
    // This is an example of a performance test case.
    [self measureBlock:^{
        auto m12 = mm1 * mm2;
    }];
}

@end

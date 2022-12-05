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
        XCTAssert(v3(i) == a[i-1]);
    
    vec<5> v4("1, 3, 5, 7, 9");
    for (std::size_t i = 1; i <= v4.dim(); ++i)
        XCTAssert(v4(i) == 2.*i - 1.);
    
    vec<5> v5("1.0e1 3.0e2 5.0e3 7.0e4 9.0e5");
    for (std::size_t i = 1; i <= v5.dim(); ++i)
        XCTAssert(v5(i) == (2.*i - 1.)*std::pow(10., i));
    
    vec<5> v6("-1 3 -5 7 -9");
    for (std::size_t i = 1; i <= v6.dim(); ++i)
        XCTAssert(v6(i) == (2.*i - 1.) * std::pow(-1., i));
    
    std::cout << v6.str(output_fmt::sht) << std::endl;
    std::cout << v6.str() << std::endl;
    std::cout << v6.str(output_fmt::ext) << std::endl;
    std::cout << v6.str(output_fmt::sci) << std::endl;
    std::cout << v6.str(output_fmt::scx) << std::endl;
    
    auto v7 = rand<100>();
    
    auto v8 = randn<100>();
}

- (void)testIndexAssignmentCompariton {
    vec<5> v1("1, 2, 3, 4, 5");
    
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
    vec<5> v1("1, 3, 5, 7, 9");
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
    mat<3, 3> m1("1 2 3; 4 5 6; 7 8 9");
    vec<3> v1("1 3 5");
 
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
    
    vec<3> v1 {"1 3 5"};
    XCTAssert(v1.norm(1) == 1. + 3. + 5.);
    XCTAssert(v1.norm(2) == std::sqrt(1. + 9. + 25.));
    XCTAssert(v1.norm_inf() == 5.);
    
    vec<3> v2 {"-1. 3 -5."};
    XCTAssert(v2.norm(1) == 1. + 3. + 5.);
    XCTAssert(v2.norm(2) == std::sqrt(1. + 9. + 25.));
    XCTAssert(v2.norm_inf() == -5.);

    vec<3> v3 {"2 4 6"};
    XCTAssert(inner(v1, v3) == 1.*2. + 3.*4. + 5.*6.);
    XCTAssert(dist(v0, v3) == norm(v3));
    
    vec<6> v4 {"1 5 2 4 7 6"};
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
    
    mat<3, 4> m1 {"1 2 3 4; 2. 3. 4. 5.; 3 4 5 6;"};
    for (std::size_t i = 1; i <= m1.count_rows(); ++i )
        for (std::size_t j = 1; j <= m1.count_cols(); ++j)
            XCTAssert(m1(i, j) == (i - 1.) + j);
    
    std::cout << m1.str(output_fmt::sht) << std::endl;
    std::cout << m1.str() << std::endl;
    std::cout << m1.str(output_fmt::ext) << std::endl;
    std::cout << m1.str(output_fmt::sci) << std::endl;
    std::cout << m1.str(output_fmt::scx) << std::endl;
    
    mat<3, 4> m2 {" 1. -2. 3. -4.; -2 3 -4 5; 3 -4 5 -6;"};
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

- (void)testMatrixTranspose {
    mat<8, 10> m1 {
        "1  2  3  4  5  6  7  8  9 10;"
        "2  3  4  5  6  7  8  9 10 11;"
        "3  4  5  6  7  8  9 10 11 12;"
        "4  5  6  7  8  9 10 11 12 13;"
        "5  6  7  8  9 10 11 12 13 14;"
        "6  7  8  9 10 11 12 13 14 15;"
        "7  8  9 10 11 12 13 14 15 16;"
        "8  9 10 11 12 13 14 15 16 17;"
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
        "1  2  3  4  5  6  7  8  9 10;"
        "2  3  4  5  6  7  8  9 10 11;"
        "3  4  5  6  7  8  9 10 11 12;"
        "4  5  6  7  8  9 10 11 12 13;"
        "5  6  7  8  9 10 11 12 13 14;"
        "6  7  8  9 10 11 12 13 14 15;"
        "7  8  9 10 11 12 13 14 15 16;"
        "8  9 10 11 12 13 14 15 16 17;"
    };
    mat<10, 6> m2 {
        " 1  2  3  4  5  6;"
        " 2  3  4  5  6  7;"
        " 3  4  5  6  7  8;"
        " 4  5  6  7  8  9;"
        " 5  6  7  8  9 10;"
        " 6  7  8  9 10 11;"
        " 7  8  9 10 11 12;"
        " 8  9 10 11 12 13;"
        " 9 10 11 12 13 14;"
        "10 11 12 13 14 15;"
    };
    
    auto m12 = m1 * m2;
    
    mat<8, 6> m12_desired = {
        "385  440   495   550   605   660;"
        "440  505   570   635   700   765;"
        "495  570   645   720   795   870;"
        "550  635   720   805   890   975;"
        "605  700   795   890   985  1080;"
        "660  765   870   975  1080  1185;"
        "715  830   945  1060  1175  1290;"
        "770  895  1020  1145  1270  1395;"
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

- (void)testMatrixTrace {
    auto m = identity<10>();
    XCTAssert(tr(m) == 10.);
    
    mat<2,2> m1 {"3 7; 1 -4"};
    XCTAssert(tr(m1) == -1.);
    
    mat<3,3> m2 {"1 2 3; 3 2 1; 2 1 3"};
    XCTAssert(tr(m2) == 6.);
    
    mat<4,4> m3 {"1.1, 2.2, 3.4, -4.2; 4.1, -3.4, 2.3, 1.2; -2.1, 1.4, 3.2, 4.1; 3.3, 2.2, -1.4, 4.1"};
    XCTAssert(tr(m3) == 5.);
}

- (void)testMatrixDeterminant {
    auto m = identity<10>();
    XCTAssert(det(m) == 1.);
    
    mat<2,2> m1 {"3 7; 1 -4"};
    XCTAssert(det(m1) == -19.);
    
    mat<3,3> m2 {"1 2 3; 3 2 1; 2 1 3"};
    XCTAssert(det(m2) == -12.);
    
    mat<4,4> m3 {"1.1, 2.2, 3.4, -4.2; 4.1, -3.4, 2.3, 1.2; -2.1, 1.4, 3.2, 4.1; 3.3, 2.2, -1.4, 4.1"};
    XCTAssert(det(m3) == -1028.5596);
}

- (void)testMatrixInversion {
    mat<3, 3> m1 {"1 2 3; 4 1 6; 7 8 1"};
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
        " 7.52 -1.10 -7.95  1.08;"
        "-0.76  0.62  9.34 -7.10;"
        " 5.13  6.62 -5.66  0.87;"
        "-4.75  8.52  5.75  5.30;"
        " 1.33  4.91 -5.49 -3.52;"
        "-2.40 -6.77  2.34  3.95;"
    };
    XCTAssert(M.count_rows() == 6);
    XCTAssert(M.count_cols() == 4);
    XCTAssert(M(1, 1) == 7.52);
    XCTAssert(M(1, 2) == -1.10);
    
    mat<6, 6> U {};
    mat<4, 4> Vt {};
    auto S = svd(M, U, Vt);
    
    vec<4> singular_values {
        "18.365978454889984 "
        "13.629979679210999 "
        "10.85333572722705 "
        "4.491569094526893 "
    };
        
    XCTAssert(norm(S - singular_values) < TOLERANCE * 1e2);
}

- (void)testPerformanceMatrixMultiplication {
    // This is an example of a performance test case.
    [self measureBlock:^{
        auto m12 = mm1 * mm2;
    }];
}

@end

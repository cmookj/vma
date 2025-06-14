//
//  vma.cpp
//  Basic Linear Algebra Toolkit
//
//  Created by Changmook Chun on 2022-12-15.
//

#include "vma.hpp"

using namespace gpw::vma;

double
gpw::vma::det (const mat<2, 2>& M) {
    return M (1, 1) * M (2, 2) - M (1, 2) * M (2, 1);
}

mat<2, 2>
gpw::vma::inv (const mat<2, 2>& M) {
    double dt = det (M);

    if (std::abs(dt) < TOL) {
        std::stringstream strm;
        strm << "Matrix not invertible, Determinant = " << dt;
        throw (std::runtime_error{strm.str()});
    }

    return mat<2, 2>{
               {M (2,  2), -M (1, 2)},
               {-M (2, 1), M (1,  1)}
    } /
           dt;
}

double
gpw::vma::det (const mat<3, 3>& M) {
    double a = M (1, 1);
    double b = M (1, 2);
    double c = M (1, 3);

    double d = M (2, 1);
    double e = M (2, 2);
    double f = M (2, 3);

    double g = M (3, 1);
    double h = M (3, 2);
    double i = M (3, 3);

    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
}

mat<3, 3>
gpw::vma::inv (const mat<3, 3>& M) {

    double dt = det (M);
    if (std::abs(dt) < TOL) {
        std::stringstream strm;
        strm << "Matrix not invertible, Determinant = " << dt;
        throw (std::runtime_error{strm.str()});
    }

    double a = M (1, 1);
    double b = M (1, 2);
    double c = M (1, 3);

    double d = M (2, 1);
    double e = M (2, 2);
    double f = M (2, 3);

    double g = M (3, 1);
    double h = M (3, 2);
    double i = M (3, 3);

    mat<3, 3> C = {
        {e * i - f * h,    -(b * i - c * h), b * f - c * e   },
        {-(d * i - f * g), a * i - c * g,    -(a * f - c * d)},
        {d * h - e * g,    -(a * h - b * g), a * e - b * d   }
    };

    return C / dt;
}

int
gpw::vma::set_format (std::stringstream& strm, output_fmt fmt) {
    int width;
    int precision;

    std::ios_base::fmtflags options;

    switch (fmt) {
    case output_fmt::sht:
        options   = std::ios_base::fixed;
        precision = 2;
        width     = 8;
        break;

    case output_fmt::nml:
        options   = std::ios_base::fixed;
        precision = 4;
        width     = 10;
        break;

    case output_fmt::ext:
        options   = std::ios_base::fixed;
        precision = 8;
        width     = 14;
        break;

    case output_fmt::sci:
        options   = std::ios_base::scientific;
        precision = 4;
        width     = 10;
        break;

    case output_fmt::scx:
        options   = std::ios_base::scientific;
        precision = 8;
        width     = 18;
    }

    strm.setf (options, std::ios_base::floatfield);
    strm.precision (precision);

    return width;
}

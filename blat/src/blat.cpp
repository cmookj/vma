//
//  blat.cpp
//  Basic Linear Algebra Toolkit
//
//  Created by Changmook Chun on 2022-12-15.
//

#include "blat.hpp"

using namespace tls::blat;

int tls::blat::set_format(std::stringstream& strm, output_fmt fmt) {
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

    strm.setf(options, std::ios_base::floatfield);
    strm.precision(precision);

    return width;
}

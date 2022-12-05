//===------------------------------------------------------------*- C++ -*-===//
//
//  linear_algebra.hpp
//  LinearAlgebra
//
//  Created by Changmook Chun on 2022/11/25.
//  Copyright © 2022 Teaeles.com. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <random>
#include <stdlib.h>

// -----------------------------------------------------------------------------
#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <iostream>

#if defined(__APPLE__)
#include <vecLib/vecLib.h>
using integer_t = __CLPK_integer;
using real_t    = __CLPK_doublereal;
#endif

#if defined(_WIN32) || defined(_WIN64)
#include "clapack.h"
#include "f2c.h"

#include <cstdint>
#include <ctime>
using integer_t = integer;
using real_t    = doublereal;
#endif

#if defined(__linux) || defined(__linux__)
#include "clapack.h"
#include "f2c.h"
using integer_t = integer;
using real_t    = doublereal;
#endif

namespace tls::blat {

// Constants
const double TOLERANCE = 2.2204e-16;
const double EPS       = std::numeric_limits<double>::epsilon();

const unsigned INF =
#if defined(__linux__) || defined(__linux)
    100000000;
#else
#if defined(_WIN32) || defined(_WIN64)
#ifdef max
#undef max
#endif
#endif
    std::numeric_limits<unsigned>::max();
#endif

// =============================================================================
//                                                                  Enumerations
// =============================================================================

// Format specification to generate string representation
enum class output_fmt { sht, nml, ext, sci, scx };

// Eigenvalue problem specification to control the problem setup and solution
enum class eigen_spec {
    EIGENVALUE_ONLY,
    EIGENVALUE_AND_EIGENVEC,
    EIGENVALUE_AND_LEFT_EIGENVEC,
    EIGENVALUE_AND_RIGHT_EIGENVEC
};

// =============================================================================
//                                                     C L A S S  :  V E C T O R
// =============================================================================

/**
 @brief Vector
 */
template <std::size_t DIM> class vec {
protected:
    std::array<double, DIM> _elem;

public:
    // Default constructor and destructor
    vec() { _elem.fill(0.); }
    virtual ~vec() = default;

    // Copy constructor & assignment
    vec(const vec&)            = default;
    vec& operator=(const vec&) = default;

    // Move constructor & assignment
    vec(vec&&) noexcept            = default;
    vec& operator=(vec&&) noexcept = default;

    // Other constructors
    vec(std::array<double, DIM>&& elm)
        : _elem {elm} { }

    vec(const double v) {
        std::for_each(_elem.begin(), _elem.end(), [&v](double& e) { e = v; });
    }

    vec(const double* vp) {
        std::for_each(_elem.begin(), _elem.end(), [&vp](double& e) { e = *(vp++); });
    }

    vec(const char* fmt) {
        _elem.fill(0.);
        _format(fmt);
    }

    // Access methods
    std::array<double, DIM>&       elem() { return _elem; }
    const std::array<double, DIM>& elem() const { return _elem; }

    // Subscript operators
    double& operator[](const std::size_t n) { return const_cast<double&>(static_cast<const vec&>(*this)[n]); }
    double& operator()(const std::size_t n) { return const_cast<double&>(static_cast<const vec&>(*this)(n)); }
    const double& operator[](const std::size_t n) const { return _elem[n]; }
    const double& operator()(const std::size_t n) const { return _elem[n - 1]; }


    // Dimension
    std::size_t dim() const { return _elem.size(); }

    // Equality
    bool operator==(const vec& rhs) const { return _elem == rhs._elem; }
    bool operator!=(const vec& rhs) const { return !(_elem == rhs._elem); }

    // Binary arithmetic operators
    vec& operator+=(const vec& rhs) {
        std::transform(
            _elem.cbegin(), _elem.cend(), rhs._elem.cbegin(), _elem.begin(), std::plus<> {});
        return *this;
    }

    vec& operator-=(const vec& rhs) {
        std::transform(
            _elem.cbegin(), _elem.cend(), rhs._elem.cbegin(), _elem.begin(), std::minus<> {});
        return *this;
    }

    vec& operator*=(const double& s) {
        std::transform(
            _elem.cbegin(), _elem.cend(), _elem.begin(), [s](const auto& v) { return v * s; });
        return *this;
    }

    vec& operator/=(const double& s) {
        if (s == 0.)
            throw std::runtime_error("[ERROR] Division by zero");
        std::transform(
            _elem.cbegin(), _elem.cend(), _elem.begin(), [s](const auto& v) { return v / s; });
        return *this;
    }

    // -------------------------------------------------------------------------
    //  Algebraic operations for vectors
    // -------------------------------------------------------------------------
    /**
     @brief Calculates p-norm of a vector

     @details The vector norm |x|_p for p = 1, 2, ... is defined as
         |x|_p = (sum |x_i|^p)^(1/p).
     */
    double norm(const unsigned p = 2) const {
        auto powered_fold = [p](const double a, const double b) {
            return a + std::pow(std::abs(b), double(p));
        };
        return std::pow(std::accumulate(_elem.cbegin(), _elem.cend(), 0., powered_fold),
                        1. / double(p));
    }

    /**
     @brief Calculates infinite vector norm

     @details The special case |x|_inf is defined as
         |x|_inf = max |x_i|
     */
    double norm_inf() const {
        auto index = std::max_element(_elem.begin(), _elem.end(), [](double a, double b) {
            return std::abs(a) < std::abs(b);
        });
        return *index;
    }

    vec unit(const unsigned p = 2) const {
        vec    u {*this};
        double n {this->norm(p)};
        if (n > EPS)
            u /= n;
        return u;
    }

    double inner(const vec& b) const {
        return std::transform_reduce(_elem.cbegin(), _elem.cend(), b._elem.cbegin(), 0.);
    }

    /**
     @brief Represents the vector as a text string
     */
    std::string str(output_fmt fmt = output_fmt::nml) const {
        std::stringstream strm {};

        int width;
        int precision;

        std::ios_base::fmtflags options;
        
        switch(fmt){
            case output_fmt::sht:
                options = std::ios_base::fixed;
                precision = 2;
                width = 8;
                break;
                
            case output_fmt::nml:
                options = std::ios_base::fixed;
                precision = 4;
                width = 10;
                break;
                
            case output_fmt::ext:
                options = std::ios_base::fixed;
                precision = 8;
                width = 14;
                break;
                
            case output_fmt::sci:
                options = std::ios_base::scientific;
                precision = 4;
                width = 10;
                break;
                
            case output_fmt::scx:
                options = std::ios_base::scientific;
                precision = 8;
                width = 18;
        }
        
        strm.setf(options, std::ios_base::floatfield);
        strm.precision(precision);
        
        auto              print = [&strm, &width](const double& v) { strm.width(width); strm << v << ", "; };
        
        strm << "[ ";
        std::for_each(_elem.cbegin(), _elem.cend(), print);
        strm << "]";

        return strm.str();
    }

protected:
    void _format(const char* fmt) {
        bool        numeric_rep_preceded = false;
        bool        eol                  = false;
        char        ch;
        std::size_t count = 0;
        unsigned    idx   = 0;

        while (!eol) {
            ch = fmt[idx++];
            if (ch == '\t' || ch == ' ' || ch == ',' || ch == '\0') {
                if (numeric_rep_preceded == true) {
                    count++;
                    numeric_rep_preceded = false;
                }
                if (ch == '\0') {
                    eol = true;
                }
            }
            else
                numeric_rep_preceded = true;
        }

        char buf[256];
        eol          = false;
        idx          = 0;
        unsigned ddx = 0;
        unsigned i   = 0;
        while (!eol) {
            ch = fmt[idx++];
            if (ch == '\t' || ch == ' ' || ch == ',' || ch == '\0') {
                if (ddx > 0) {
                    buf[ddx] = '\0';
                    _elem[i] = atof(buf);
                    i++;
                    ddx = 0;
                }
                if (ch == '\0')
                    eol = true;
            }
            else
                buf[ddx++] = ch;
        }
    }
};

/**
 @brief Outputs a vector to stream
 */
template <std::size_t DIM>
std::ostream& operator<<(std::ostream& strm, const vec<DIM>& v) {
    v.print(output_fmt::sht, strm);
    return strm;
}

// -----------------------------------------------------------------------------
//                                                       Special Vector Creation
// -----------------------------------------------------------------------------
/**
 @brief Creates a vector with random numbers in uniform distribution
 */
template <std::size_t DIM> vec<DIM> rand() {
    std::array<double, DIM> elm;

    std::random_device               rdu;
    std::mt19937                     genu(rdu());
    std::uniform_real_distribution<> ud(0, 1);
    for (std::size_t i = 0; i < DIM; ++i) {
        elm[i] = ud(genu);
    }

    return vec<DIM> {std::move(elm)};
}

/**
 @brief Creates a vector with random numbers in normal distribution
 */
template <std::size_t DIM> vec<DIM> randn() {
    std::array<double, DIM> elm;

    std::random_device         rdn;
    std::mt19937               genn(rdn());
    std::normal_distribution<> nd(0, 1);
    for (std::size_t i = 0; i < DIM; ++i) {
        elm[i] = nd(genn);
    }

    return vec<DIM> {std::move(elm)};
}

// -----------------------------------------------------------------------------
//                                                             Vector Operations
// -----------------------------------------------------------------------------
/**
 @brief Calculates inner product of two vectors
 */
template <std::size_t DIM> double inner(const vec<DIM>& a, const vec<DIM>& b) {
    return a.inner(b);
}

/**
 @brief Normalizes a vector
 */
template <std::size_t DIM> vec<DIM>& normalize(vec<DIM>& v, const unsigned p = 2) {
    double n = v.norm(p);
    if (n > EPS)
        v /= n;
    return v;
}

/**
 @brief Calculates the norm of a vector
 */
template <std::size_t DIM> double norm(const vec<DIM>& v, const unsigned p = 2) {
    return v.norm(p);
}

/**
 @brief Adds two vectors
 */
template <std::size_t DIM> vec<DIM> operator+(const vec<DIM>& a, const vec<DIM>& b) {
    vec result {a};
    result += b;
    return result;
}

/**
 @brief Subtracts a vector from another
 */
template <std::size_t DIM> vec<DIM> operator-(const vec<DIM>& a, const vec<DIM>& b) {
    vec result {a};
    result -= b;
    return result;
}

/**
 @brief Negate a vector
 */
template <std::size_t DIM> vec<DIM> operator-(const vec<DIM>& a) {
    vec result {a};
    result *= -1.;
    return result;
}

/**
 @brief Multiplies a scalar to a vector
 */
template <std::size_t DIM> vec<DIM> operator*(const vec<DIM>& a, const double s) {
    vec result {a};
    result *= s;
    return result;
}

/**
 @brief Multiplies a scalar to a vector
 */
template <std::size_t DIM> vec<DIM> operator*(const double s, const vec<DIM>& a) {
    vec result {a};
    result *= s;
    return result;
}

/**
 @brief Divides a vector by a scalar
 */
template <std::size_t DIM> vec<DIM> operator/(const vec<DIM>& a, const double s) {
    vec result {a};
    result /= s;
    return result;
}

// =============================================================================
//                                                     C L A S S  :  M A T R I X
// =============================================================================
/**
 @brief Matrix

 @details This class stores the elements of a matrix in column major order,
 for easier and efficient integration with LAPACK.
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS> class mat {
protected:
    std::size_t                             _count_rows = DIM_ROWS;
    std::size_t                             _count_cols = DIM_COLS;
    std::array<double, DIM_ROWS * DIM_COLS> _elem;

public:
    // Default constructor and destructor
    mat() { _elem.fill(0.); }
    virtual ~mat() = default;

    // Copy constructor & assignment
    mat(const mat&)            = default;
    mat& operator=(const mat&) = default;

    // Move constructor & assignment
    mat(mat&&) noexcept            = default;
    mat& operator=(mat&&) noexcept = default;

    // Other constructors
    mat(const std::size_t m, const std::size_t n, std::array<double, DIM_ROWS * DIM_COLS>&& elm)
        : _count_rows {m}
        , _count_cols {n}
        , _elem {elm} { }

    mat(const double v) {
        std::for_each(_elem.begin(), _elem.end(), [&v](double& e) { e = v; });
    }

    mat(const double* vp) {
        std::for_each(_elem.begin(), _elem.end(), [&vp](double& e) { e = *(vp++); });
    }

    mat(const char* fmt) {
        _elem.fill(0.);
        _format(fmt);
    }

    // Access methods
    std::array<double, DIM_ROWS * DIM_COLS>&       elem() { return _elem; }
    const std::array<double, DIM_ROWS * DIM_COLS>& elem() const { return _elem; }

    /**
     @brief Index operator

     @details Note that the elements of the matrix are in column major order.
     */
    const double& operator()(const std::size_t i, const std::size_t j) const {
        return _elem[(i - 1) + (j - 1) * _count_rows];
    }
    
    double& operator()(const std::size_t i, const std::size_t j) {
        return const_cast<double&>(static_cast<const mat&>(*this)(i, j));
    }

    // Dimension
    std::pair<std::size_t, std::size_t> dim() const {
        return std::make_pair(_count_rows, _count_cols);
    }

    std::size_t count_rows() const { return _count_rows; }
    std::size_t count_cols() const { return _count_cols; }

    // Equality
    bool operator==(const mat& rhs) const {
        return _elem == rhs._elem && _count_rows == rhs._count_rows &&
               _count_cols == rhs._count_cols;
    }
    bool operator!=(const mat& rhs) const { return !(*this == rhs); }

    // Binary arithmetic operators
    mat& operator+=(const mat& rhs) {
        std::transform(
            _elem.cbegin(), _elem.cend(), rhs._elem.cbegin(), _elem.begin(), std::plus<> {});
        return *this;
    }

    mat& operator+=(const double s) {
        std::transform(
            _elem.cbegin(), _elem.cend(), _elem.begin(), [s](const double e) { return e + s; });
        return *this;
    }

    mat& operator-=(const mat& rhs) {
        std::transform(
            _elem.cbegin(), _elem.cend(), rhs._elem.cbegin(), _elem.begin(), std::minus<> {});
        return *this;
    }

    mat& operator*=(const double& s) {
        std::transform(
            _elem.cbegin(), _elem.cend(), _elem.begin(), [s](const auto& v) { return v * s; });
        return *this;
    }

    mat& operator/=(const double& s) {
        if (s == 0.)
            throw std::runtime_error("[ERROR] Division by zero");
        std::transform(
            _elem.cbegin(), _elem.cend(), _elem.begin(), [s](const auto& v) { return v / s; });
        return *this;
    }

    // Matrix multiplication
    mat& operator*=(const mat<DIM_COLS, DIM_COLS>& m) {
        std::array<double, DIM_COLS * DIM_ROWS> elm;
        cblas_dgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    DIM_ROWS,
                    DIM_COLS,
                    DIM_COLS,
                    1.,
                    _elem.data(),
                    DIM_ROWS,
                    m.elem().data(),
                    DIM_COLS,
                    0.,
                    elm.data(),
                    DIM_ROWS);
        _elem = std::move(elm);
        return *this;
    }

    // Description (convert to string)
    std::string str(output_fmt fmt = output_fmt::nml) const {
        std::stringstream strm {};

        int width;
        int precision;

        std::ios_base::fmtflags options;
        
        switch(fmt){
            case output_fmt::sht:
                options = std::ios_base::fixed;
                precision = 2;
                width = 8;
                break;
                
            case output_fmt::nml:
                options = std::ios_base::fixed;
                precision = 4;
                width = 10;
                break;
                
            case output_fmt::ext:
                options = std::ios_base::fixed;
                precision = 8;
                width = 14;
                break;
                
            case output_fmt::sci:
                options = std::ios_base::scientific;
                precision = 4;
                width = 10;
                break;
                
            case output_fmt::scx:
                options = std::ios_base::scientific;
                precision = 8;
                width = 18;
        }
        
        strm.setf(options, std::ios_base::floatfield);
        strm.precision(precision);
        
        for (std::size_t i = 0; i < _count_rows; ++i) {
            strm << "[ ";
            
            for (std::size_t j = 0; j < _count_cols; ++j) {
                strm.width(width);

            strm << _elem[j * _count_rows + i] << ", ";
        }

            strm << "]\n";
        }

        return strm.str();
    }

protected:
    void _format(const char* fmt) {
        bool        eol                  = false;
        bool        row_parsed           = false;
        bool        numeric_rep_preceded = false;
        char        ch;
        std::size_t m, n;
        unsigned    i, j, idx;

        m = n = 0;
        idx = i = j = 0;
        while (!eol) {
            ch = fmt[idx++];
            if (ch == '\t' || ch == '\r' || ch == '\n' || ch == ' ' || ch == ',' || ch == ';' ||
                ch == '\0') {

                if (numeric_rep_preceded == true) {
                    j++;
                    numeric_rep_preceded = false;
                }

                if (ch == ';' || ch == '\0') {
                    row_parsed = true;
                    if (i == 0) {
                        n = j;
                    }
                    else {
                        if (n != j)
                            throw std::runtime_error {"Illegal construction of mat"};
                    }
                }

                if (ch == '\0') {
                    m   = ++i;
                    eol = true;
                }
            }
            else {
                if (row_parsed == true) {
                    i++;
                    j          = 0;
                    row_parsed = false;
                }
                numeric_rep_preceded = true;
            }
        }

        unsigned ddx;
        char     buf[256];
        eol = false;
        idx = ddx = i = j = 0;
        while (!eol) {
            ch = fmt[idx++];
            if (ch == '\t' || ch == '\r' || ch == '\n' || ch == ' ' || ch == ',' || ch == ';' ||
                ch == '\0') {
                if (ddx > 0) {
                    buf[ddx]         = '\0';
                    _elem[j * m + i] = std::stod(buf);
                    j++;
                    ddx = 0;
                }
                if (ch == ';' || ch == '\0') {
                    i++;
                    j = 0;
                }
                if (ch == '\0')
                    eol = true;
            }
            else
                buf[ddx++] = ch;
        }
    }
};

// -----------------------------------------------------------------------------
//                                                       Special Matrix Creation
// -----------------------------------------------------------------------------
/**
 @brief Creates a DIM_COLS x DIM_COLS identity matrix
 */
template <std::size_t DIM_COLS> mat<DIM_COLS, DIM_COLS> identity() {
    mat<DIM_COLS, DIM_COLS> I {};

    I.elem().fill(0.);
    for (std::size_t i = 0; i < DIM_COLS; ++i)
        I.elem()[i * DIM_COLS + i] = 1.;

    return I;
}

/**
 @brief Creates a square matrix with only diagonal elements
 */
template <std::size_t DIM> mat<DIM, DIM> diag(std::array<double, DIM>& val) {
    std::array<double, DIM * DIM> elm {};
    elm.fill(0.);

    for (std::size_t i = 0; i < DIM; ++i)
        elm[i * DIM + i] = val[i];

    return mat<DIM, DIM> {DIM, DIM, std::move(elm)};
}

/**
 @brief Creates a square matrix with only diagonal elements
 */
template <std::size_t DIM> mat<DIM, DIM> diag(double* val) {
    std::array<double, DIM * DIM> elm {};
    elm.fill(0.);

    for (std::size_t i = 0; i < DIM; ++i)
        elm[i * DIM + i] = val[i];

    return mat<DIM, DIM> {DIM, DIM, std::move(elm)};
}

/**
 @brief Creates a matrix with random numbers in uniform distribution
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS> mat<DIM_ROWS, DIM_COLS> rand() {
    // mat<DIM_ROWS, DIM_COLS> m {};
    std::array<double, DIM_ROWS * DIM_COLS> elm;

    std::random_device               rdu;
    std::mt19937                     genu(rdu());
    std::uniform_real_distribution<> ud(0, 1);
    for (std::size_t i = 0; i < DIM_ROWS * DIM_COLS; ++i) {
        elm[i] = ud(genu);
    }

    return mat<DIM_ROWS, DIM_COLS> {DIM_ROWS, DIM_COLS, std::move(elm)};
}

/**
 @brief Creates a matrix with random numbers in normal distribution
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS> mat<DIM_ROWS, DIM_COLS> randn() {
    std::array<double, DIM_ROWS * DIM_COLS> elm;

    std::random_device         rdn;
    std::mt19937               genn(rdn());
    std::normal_distribution<> nd(0, 1);
    for (std::size_t i = 0; i < DIM_ROWS * DIM_COLS; ++i) {
        elm[i] = nd(genn);
    }

    return mat<DIM_ROWS, DIM_COLS> {DIM_ROWS, DIM_COLS, std::move(elm)};
}

// -----------------------------------------------------------------------------
//                                                             Matrix Operations
// -----------------------------------------------------------------------------
/**
 @brief Transposes a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_COLS, DIM_ROWS> transpose(mat<DIM_ROWS, DIM_COLS>& m) {
    mat<DIM_COLS, DIM_ROWS> t {};

    // The number of columns to copy simultaneously (to utilize cache)
    constexpr std::size_t job_size {16};
    const std::size_t     count_sets {DIM_COLS / job_size};

    // Copy 'job_size' columns simultaneously
    for (std::size_t s = 0; s < count_sets; ++s)
        for (std::size_t j = 0; j < job_size; ++j)
            for (std::size_t i = 0; i < DIM_ROWS; ++i)
                t.elem()[j + DIM_COLS * (s * job_size + i)] =
                    m.elem()[(s * job_size + j) * DIM_ROWS + i];

    // Copy the remaining columns (if any)
    const std::size_t count_remains {DIM_COLS - count_sets * job_size};
    for (std::size_t j = 0; j < count_remains; ++j)
        for (std::size_t i = 0; i < DIM_ROWS; ++i)
            t.elem()[j + DIM_COLS * (count_sets * job_size + i)] =
                m.elem()[(count_sets * job_size + j) * DIM_ROWS + i];

    return t;
}

/**
 @brief Transposes a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_COLS, DIM_ROWS> transpose_naive(mat<DIM_ROWS, DIM_COLS>& m) {
    mat<DIM_COLS, DIM_ROWS> t {};

    // Copy rows one by one
    for (std::size_t j = 0; j < DIM_COLS; ++j)
        for (std::size_t i = 0; i < DIM_ROWS; ++i)
            t.elem()[i * DIM_COLS + j] = m.elem()[j * DIM_ROWS + i];

    return t;
}

/**
 @brief Negates a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator-(const mat<DIM_ROWS, DIM_COLS>& m) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result *= -1.;
    return result;
}

/**
 @brief Adds a scalar to a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator+(const mat<DIM_ROWS, DIM_COLS>& m, const double s) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result += s;
    return result;
}

template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator+(const double s, const mat<DIM_ROWS, DIM_COLS>& m) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result += s;
    return result;
}

/**
 @brief Subtracts a scalar from a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator-(const mat<DIM_ROWS, DIM_COLS>& m, const double s) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result -= s;
    return result;
}

/**
 @breif Subtracts a matrix from a scalar
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator-(const double s, const mat<DIM_ROWS, DIM_COLS>& m) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result *= -1.;
    result += s;
    return result;
}

/**
 @brief Multiplies a scalar to a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator*(const mat<DIM_ROWS, DIM_COLS>& m, const double s) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result *= s;
    return result;
}

/**
 @brief Multiplies a scalar to a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator*(const double s, const mat<DIM_ROWS, DIM_COLS>& m) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result *= s;
    return result;
}

/**
 @brief Divides a matrix with a scalar
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator/(const mat<DIM_ROWS, DIM_COLS>& m, const double s) {
    mat<DIM_ROWS, DIM_COLS> result {m};
    result /= s;
    return result;
}

/**
 @brief Multiplies two matrices using BLAS
 */
template <std::size_t DIM_ROWS, std::size_t DIM, std::size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS> operator*(const mat<DIM_ROWS, DIM>& m1, const mat<DIM, DIM_COLS>& m2) {
    std::array<double, DIM_ROWS * DIM_COLS> elm;
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                DIM_ROWS,
                DIM_COLS,
                DIM,
                1.,
                m1.elem().data(),
                DIM_ROWS,
                m2.elem().data(),
                DIM,
                0.,
                elm.data(),
                DIM_ROWS);
    return mat<DIM_ROWS, DIM_COLS> {DIM_ROWS, DIM_COLS, std::move(elm)};
}

/**
 @brief Post-multiplies a vector to a matrix
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
vec<DIM_COLS> operator*(const mat<DIM_ROWS, DIM_COLS>& m, const vec<DIM_COLS>& v) {
    std::array<double, DIM_COLS> elm;
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                DIM_ROWS,
                1,
                DIM_COLS,
                1.,
                m.elem().data(),
                DIM_ROWS,
                v.elem().data(),
                DIM_COLS,
                0.,
                elm.data(),
                DIM_COLS);
    return vec<DIM_COLS> {std::move(elm)};
}

/**
 @brief Pre-multiplies a vector to a matrix

 @details This function implicitly assumes that the first argument is a row
 vector.
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
vec<DIM_COLS> operator*(const vec<DIM_ROWS>& v, const mat<DIM_ROWS, DIM_COLS>& m) {
    std::array<double, DIM_COLS> elm;
    cblas_dgemm(CblasColMajor,
                CblasNoTrans,
                CblasNoTrans,
                1,
                DIM_ROWS,
                DIM_COLS,
                1.,
                v.elem().data(),
                1,
                m.elem().data(),
                DIM_ROWS,
                0.,
                elm.data(),
                1);
    return vec<DIM_COLS> {std::move(elm)};
}

// -----------------------------------------------------------------------------
//                                                  Matrix Operations: Algebraic
// -----------------------------------------------------------------------------
/**
 @brief Calculates the trace of a square matrix
 */
template <std::size_t DIM> double tr(const mat<DIM, DIM>& M) {
    double tr {0.0};
    for (std::size_t i = 0; i < DIM; ++i) {
        tr += M.elem()[i * DIM + i];
    }
    return tr;
}

/**
 @brief Calculates the determinant of a square matrix

 @details
 Lapack's dgetrf() computes a A=P*L*U decomposition for a general M-by-N matrix A.
 Assuming an invertible square matrix A, its determinant can be computed as a product:

 - U is an upper triangular matrix.  Hence, its determinant is the product of the
 diagonal elements, which happens to be the diagonal elements of the output A.
 Indeed, see how the output A is defined:

 On exit, the factors L and U from the factorization A = P*L*U; the unit diagonal
 elements of L are not stored.

 - L is a lower triangular matrix featuring unit diagonal elements which are not
 stored.  Hence, its determinant is always 1.

 - P is a permutation matrix coded as a product of transpositions, i.e., 2-cycles
 or swap.  Indeed, see dgetri() to understand how it is used.  Hence, its
 determinant is either 1 or -1, depending on whether the number of transpositions
 is even or odd.  As a result, the determinant of P can be computed as:

 int j;
 double detp=1.;
 for (j=0; j<n; j++) {
     if (j+1 != ipiv[j]) {
         // j+1 : ipiv is from Fortran, hence starts at 1.
         // hey ! This is a transpose !
         detp = -detp;
     }
 }
 */
template <std::size_t DIM> double det(const mat<DIM, DIM>& M) {
    auto LU = std::make_unique<real_t[]>(DIM * DIM);
    std::memcpy(LU.get(), M.elem().data(), sizeof(real_t) * DIM * DIM);

    integer_t N {static_cast<integer_t>(DIM)};
    integer_t INFO;

    auto IPIV = std::make_unique<integer_t[]>(DIM);

    dgetrf_(&N, &N, LU.get(), &N, IPIV.get(), &INFO);

    double d = 0.0;
    if (INFO != 0) {
        return d;
    }

    d = 1.0;
    for (std::size_t i = 0; i < DIM; ++i) {
        if (IPIV[i] != static_cast<integer_t>(i + 1)) {
            d *= -LU[i * DIM + i];
        }
        else {
            d *= LU[i * DIM + i];
        }
    }

    return d;
}

/**
 @brief Invert a square matrix
 */
template <std::size_t DIM> mat<DIM, DIM> inv(const mat<DIM, DIM>& m) {
    integer_t     n = {DIM};
    mat<DIM, DIM> result {m};
    integer_t     INFO;

    auto IPIV = std::make_unique<integer_t[]>(DIM);

    dgetrf_(&n, &n, result.elem().data(), &n, IPIV.get(), &INFO);

    double d = 1.;
    for (std::size_t i = 0; i < DIM; ++i)
        d *= result.elem()[i * DIM + i];

    if (std::fabs(d) < EPS)
        throw std::runtime_error {"The mat is singular."};

    n              = static_cast<integer_t>(DIM);
    integer_t prod = static_cast<integer_t>(DIM * DIM);
    auto      WORK = std::make_unique<real_t[]>(DIM);

    dgetri_(&n, result.elem().data(), &n, IPIV.get(), WORK.get(), &prod, &INFO);

    return result;
}

/**
 @brief Decomposes a matrix into two unitary matrices and a diagonal matrix (SVD)
 */
template <std::size_t DIM_ROWS, std::size_t DIM_COLS>
vec<std::min(DIM_ROWS, DIM_COLS)>
svd(const mat<DIM_ROWS, DIM_COLS>& M, mat<DIM_ROWS, DIM_ROWS>& U, mat<DIM_COLS, DIM_COLS>& Vt) {
    integer_t m {static_cast<integer_t>(DIM_ROWS)};
    integer_t n {static_cast<integer_t>(DIM_COLS)};
    integer_t lda  = m;
    integer_t ldu  = m;
    integer_t ldvt = n;
    integer_t ds   = std::min(m, n);

    char      jobz = 'A';
    integer_t lwork =
        3 * ds * ds +
        std::max(std::max(m, n), 5 * std::min(m, n) * std::min(m, n) + 4 * std::min(m, n));

    auto el = std::make_unique<real_t[]>(DIM_ROWS * DIM_COLS);
    std::memcpy(el.get(), M.elem().data(), sizeof(real_t) * DIM_ROWS * DIM_COLS);

    auto      s     = std::make_unique<real_t[]>(ds);
    auto      u     = std::make_unique<real_t[]>(ldu * m);
    auto      vt    = std::make_unique<real_t[]>(ldvt * n);
    auto      work  = std::make_unique<real_t[]>(std::max(1, lwork));
    auto      iwork = std::make_unique<integer_t[]>(8 * ds);
    integer_t info;

    dgesdd_(&jobz,
            &m,
            &n,
            el.get(),
            &lda,
            s.get(),
            u.get(),
            &ldu,
            vt.get(),
            &ldvt,
            work.get(),
            &lwork,
            iwork.get(),
            &info);

    if (info > 0) {
        throw std::runtime_error {"The algorithm for SVD failed to converge."};
    }

    // Copy u to the matrix U
    std::memcpy(U.elem().data(), u.get(), sizeof(real_t) * DIM_ROWS * DIM_ROWS);

    // Copy vt to the matrix Vt and transpose it
    mat<DIM_COLS, DIM_COLS> V {};
    std::memcpy(V.elem().data(), vt.get(), sizeof(real_t) * DIM_COLS * DIM_COLS);
    Vt = transpose(V);

    return vec<std::min(DIM_ROWS, DIM_COLS)>(s.get());
}

} // namespace tls::blat
#endif

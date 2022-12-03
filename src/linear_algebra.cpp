//===------------------------------------------------------------*- C++ -*-===//
//
//  linear_algebra.cpp
//  LinearAlgebra
//
//  Created by Changmook Chun on 2022/11/25.
//  Copyright © 2022 Teaeles.com. All rights reserved.
//
//===----------------------------------------------------------------------===//
#include "linear_algebra.hpp"

#include <cstring>

using namespace linear_algebra;

void linear_algebra::_copy_elements(const linear_algebra::size_t& sz,
                                    const real_t*                 src,
                                    real_t*                       des) {
    std::memcpy(des, src, sz * sizeof(real_t));
}

void linear_algebra::_copy_elements(const linear_algebra::size_t&  sz,
                                    const std::unique_ptr<real_t>& src,
                                    std::unique_ptr<real_t>&       des) {
    std::memcpy(des.get(), src.get(), sz * sizeof(real_t));
}

void linear_algebra::_copy_elements(const linear_algebra::size_t&  sz,
                                    const std::unique_ptr<real_t>& src,
                                    real_t*                        des) {
    std::memcpy(des, src.get(), sz * sizeof(real_t));
}

void linear_algebra::_copy_elements(const linear_algebra::size_t& sz,
                                    const real_t*                 src,
                                    std::unique_ptr<real_t>&      des) {
    std::memcpy(des.get(), src, sz * sizeof(real_t));
}

void linear_algebra::_add_elements(const linear_algebra::size_t& sz,
                                   const real_t*                 src1,
                                   const real_t*                 src2,
                                   real_t*                       des) {
    for (index_t i = 0; i != sz; i++)
        des[i] = src1[i] + src2[i];
}

void linear_algebra::_add_elements(const linear_algebra::size_t& sz,
                                   const real_t*                 src,
                                   const real_t&                 s,
                                   real_t*                       des) {
    for (index_t i = 0; i != sz; i++)
        des[i] = src[i] + s;
}

void linear_algebra::_add_elements(const linear_algebra::size_t& sz, real_t* src, const real_t& s) {
    for (index_t i = 0; i != sz; i++)
        src[i] += s;
}

void linear_algebra::_add_elements(const linear_algebra::size_t&  sz,
                                   const std::unique_ptr<real_t>& src1,
                                   const std::unique_ptr<real_t>& src2,
                                   std::unique_ptr<real_t>&       des) {
    linear_algebra::_add_elements(sz, src1.get(), src2.get(), des.get());
}

void linear_algebra::_add_elements(const linear_algebra::size_t&  sz,
                                   const std::unique_ptr<real_t>& src,
                                   const real_t&                  s,
                                   std::unique_ptr<real_t>&       des) {
    linear_algebra::_add_elements(sz, src.get(), s, des.get());
}

void linear_algebra::_add_elements(const linear_algebra::size_t& sz,
                                   std::unique_ptr<real_t>&      src,
                                   const real_t&                 s) {
    linear_algebra::_add_elements(sz, src.get(), s);
}

void linear_algebra::_subtract_elements(const linear_algebra::size_t& sz,
                                        const real_t*                 src1,
                                        const real_t*                 src2,
                                        real_t*                       des) {
    for (index_t i = 0; i != sz; i++)
        des[i] = src1[i] - src2[i];
}

void linear_algebra::_subtract_elements(const linear_algebra::size_t& sz,
                                        const real_t*                 src,
                                        const real_t&                 s,
                                        real_t*                       des) {
    for (index_t i = 0; i != sz; i++)
        des[i] = src[i] - s;
}

void linear_algebra::_subtract_elements(const linear_algebra::size_t& sz,
                                        real_t*                       src,
                                        const real_t&                 s) {
    for (index_t i = 0; i != sz; i++)
        src[i] -= s;
}

void linear_algebra::_subtract_elements(const linear_algebra::size_t&  sz,
                                        const std::unique_ptr<real_t>& src1,
                                        const std::unique_ptr<real_t>& src2,
                                        std::unique_ptr<real_t>&       des) {
    linear_algebra::_subtract_elements(sz, src1.get(), src2.get(), des.get());
}

void linear_algebra::_subtract_elements(const linear_algebra::size_t&  sz,
                                        const std::unique_ptr<real_t>& src,
                                        const real_t&                  s,
                                        std::unique_ptr<real_t>&       des) {
    linear_algebra::_subtract_elements(sz, src.get(), s, des.get());
}

void linear_algebra::_subtract_elements(const linear_algebra::size_t& sz,
                                        std::unique_ptr<real_t>&      src,
                                        const real_t&                 s) {
    linear_algebra::_subtract_elements(sz, src.get(), s);
}

void linear_algebra::_negate_elements(const linear_algebra::size_t& sz,
                                      const real_t*                 src,
                                      real_t*                       des) {
    for (index_t i = 0; i != sz; i++)
        des[i] = -1.0 * src[i];
}

void linear_algebra::_negate_elements(const linear_algebra::size_t&  sz,
                                      const std::unique_ptr<real_t>& src,
                                      std::unique_ptr<real_t>&       des) {
    linear_algebra::_negate_elements(sz, src.get(), des.get());
}

void linear_algebra::_multiply_scalar_to_elements(const linear_algebra::size_t& sz,
                                                  const real_t*                 src,
                                                  const real_t&                 s,
                                                  real_t*                       des) {
    for (index_t i = 0; i != sz; i++)
        des[i] = src[i] * s;
}

void linear_algebra::_multiply_scalar_to_elements(const linear_algebra::size_t&  sz,
                                                  const std::unique_ptr<real_t>& src,
                                                  const real_t&                  s,
                                                  std::unique_ptr<real_t>&       des) {
    linear_algebra::_multiply_scalar_to_elements(sz, src.get(), s, des.get());
}

void linear_algebra::_divide_elements_by_scalar(const linear_algebra::size_t& sz,
                                                const real_t*                 src,
                                                const real_t&                 s,
                                                real_t*                       des) {
    if (std::fabs(s) < EPS)
        throw std::runtime_error {"Division by zero."};

    for (index_t i = 0; i != sz; i++)
        des[i] = src[i] / s;
}

void linear_algebra::_divide_elements_by_scalar(const linear_algebra::size_t&  sz,
                                                const std::unique_ptr<real_t>& src,
                                                const real_t&                  s,
                                                std::unique_ptr<real_t>&       des) {
    linear_algebra::_divide_elements_by_scalar(sz, src.get(), s, des.get());
}

bool linear_algebra::_exclusiveOR(const bool& a, const bool& b) {
    return ((a && !b) || (!a && b));
}

linear_algebra::size_t linear_algebra::_min(const linear_algebra::size_t& a,
                                            const linear_algebra::size_t& b) {
    return (a > b) ? b : a;
}

linear_algebra::size_t linear_algebra::_max(const linear_algebra::size_t& a,
                                            const linear_algebra::size_t& b) {
    return (a > b) ? a : b;
}

void linear_algebra::_generate_uniform_random_numbers(real_t*                       _el,
                                                      const linear_algebra::size_t& _s) {
#if __cplusplus >= 201103L
    std::random_device               rdu;
    std::mt19937                     genu(rdu());
    std::uniform_real_distribution<> ud(0, 1);
    for (index_t i = 0; i != _s; i++) {
        _el[i] = ud(genu);
    }
#elif defined(__APPLE__) || defined(__linux__)
    srandom((unsigned)time(NULL));
    for (index_t i = 0; i != _s; i++) {
        _el[i] = real_t(random()) / real_t(RAND_MAX);
    }
#elif defined(_WIN32) || defined(_WIN64)
    srand((unsigned int)time(NULL));
    for (index_t i = 0; i != _s; i++) {
        _el[i] = real_t(rand()) / real_t(RAND_MAX);
    }
#endif
}

void linear_algebra::_generate_normal_random_numbers(real_t*                       _el,
                                                     const linear_algebra::size_t& _s) {
#if __cplusplus >= 201103L
    std::random_device         rdn;
    std::mt19937               genn(rdn());
    std::normal_distribution<> nd(0, 1);
    for (index_t i = 0; i != _s; i++) {
        _el[i] = nd(genn);
    }
#else
    throw std::runtime_error {"C++11 is required."};
#endif
}

index_t linear_algebra::index_max_absolute_elem(linear_algebra::size_t sz, double* values) {
    index_t index_tOfMax = 0UL;
    for (index_t i = 1UL; i != sz; i++) {
        if (std::fabs(values[i]) > std::fabs(values[index_tOfMax])) {
            index_tOfMax = i;
        }
    }
    return index_tOfMax;
}

index_t linear_algebra::index_min_absolute_elem(linear_algebra::size_t sz, double* values) {
    index_t index_tOfMin = 0UL;
    for (index_t i = 1UL; i != sz; i++) {
        if (std::fabs(values[i]) < std::fabs(values[index_tOfMin])) {
            index_tOfMin = i;
        }
    }
    return index_tOfMin;
}

index_t linear_algebra::index_max_elem(linear_algebra::size_t sz, double* values) {
    index_t index_tOfMax = 0UL;
    for (index_t i = 1UL; i != sz; i++) {
        if (values[i] > values[index_tOfMax]) {
            index_tOfMax = i;
        }
    }
    return index_tOfMax;
}

index_t linear_algebra::index_min_elem(linear_algebra::size_t sz, double* values) {
    index_t index_tOfMin = 0UL;
    for (index_t i = 1UL; i != sz; i++) {
        if (values[i] > values[index_tOfMin]) {
            index_tOfMin = i;
        }
    }
    return index_tOfMin;
}

mat linear_algebra::eye(const linear_algebra::size_t n) {
    return mat::identity(n, n);
}

mat linear_algebra::eye(const linear_algebra::size_t m, const linear_algebra::size_t n) {
    return mat::identity(m, n);
}

mat linear_algebra::zeros(const linear_algebra::size_t n) {
    return mat(n, n, initial_values_t::ZEROS);
}

mat linear_algebra::zeros(const linear_algebra::size_t m, const linear_algebra::size_t n) {
    return mat(m, n, initial_values_t::ZEROS);
}

mat linear_algebra::ones(const linear_algebra::size_t n) {
    return mat(n, n, initial_values_t::ONES);
}

mat linear_algebra::ones(const linear_algebra::size_t m, const linear_algebra::size_t n) {
    return mat(m, n, initial_values_t::ONES);
}

mat linear_algebra::diag(const vec& v, const int k) {
    return mat::diag(v, k);
}

linear_algebra::size_t array::size() const {
    return _size;
}

linear_algebra::size_t array::count() const {
    return _size;
}

real_t* array::elements() const {
    return _elements.get();
}

void array::set_size(linear_algebra::size_t s) {
    _size = s;
}

void array::set_elements(real_t* v) {
    if (_elements.get() != v) {
        _elements.reset(v);
    }
}

void array::copyValuesTo(double* val) const {
    _copy_elements(_size, _elements.get(), val);
}

array::array()
    : _size {0}
    , _elements {nullptr} { }

array::array(const linear_algebra::size_t thesize)
    : _size {thesize}
    , _elements {std::unique_ptr<real_t>(new real_t[_size])} {
    for (index_t i = 0; i != _size; i++) {
        _elements.get()[i] = 0.0;
    }
}

array::array(linear_algebra::size_t thesize, initial_values_t iv)
    : _size {thesize}
    , _elements {std::unique_ptr<real_t>(new real_t[_size])} {

    switch (iv) {
    case initial_values_t::ZEROS:
        for (index_t i = 0; i != _size; i++) {
            _elements.get()[i] = 0.0;
        }
        break;

    case initial_values_t::ONES:
        for (index_t i = 0; i != _size; i++) {
            _elements.get()[i] = 1.0;
        }
        break;

    case initial_values_t::RAND:
        _generate_uniform_random_numbers(_elements.get(), _size);
        break;

    case initial_values_t::RANDN:
        _generate_normal_random_numbers(_elements.get(), _size);
        break;
    }
}

array::array(linear_algebra::size_t thesize, double* values)
    : _size {thesize}
    , _elements {std::unique_ptr<real_t>(new real_t[_size])} {
    for (index_t i = 0; i != _size; i++) {
        _elements.get()[i] = real_t(values[i]);
    }
    _copy_elements(_size, values, _elements.get());
}

array::array(const array& src)
    : _size {src._size}
    , _elements {std::unique_ptr<real_t>(new real_t[_size])} {
    _copy_elements(_size, src._elements.get(), _elements.get());
}

array::~array() { }

bool array::operator==(const array& rhs) const {
    if (this == &rhs)
        return true;

    if (_size != rhs._size)
        return false;

    for (index_t i = 0; i != _size; i++) {
        if (_elements.get()[i] != rhs._elements.get()[i]) {
            return false;
        }
    }
    return true;
}

bool array::isSimilarTo(const array& a) const {
    if (*this == a)
        return true;

    if (_size != a._size)
        return false;

    for (index_t i = 0; i != _size; i++) {
        if (std::fabs(_elements.get()[i] - a._elements.get()[i]) > TOLERANCE) {
            return false;
        }
    }
    return true;
}

bool array::operator!=(const array& rhs) const {
    if (*this == rhs)
        return false;

    return true;
}

void array::print(print_format_t fmt, std::ostream& strm) const {
    strm << std::endl;
    strm << " array @ " << (void*)this << std::endl;
    strm << " -size: " << _size << std::endl;
    char prev = strm.fill('-');
    strm << std::setw(80) << '-' << std::endl;
    strm.fill(prev);

    int width = _set_print_format(fmt, strm);

    for (index_t i = 0; i != _size; i++) {
        strm.width(width);
        strm << _elements.get()[i] << std::endl;
    }

    _restore_print_format(strm);
}

int array::_set_print_format(print_format_t fmt, std::ostream& strm) const {
    int                     precision;
    int                     width;
    std::ios_base::fmtflags options;

    switch (fmt) {
    case print_format_t::SHORT:
        options   = std::ios_base::fixed;
        precision = 2;
        width     = 8;
        break;

    case print_format_t::NORMAL:
        options   = std::ios_base::fixed;
        precision = 4;
        width     = 10;
        break;

    case print_format_t::EXTENDED:
        options   = std::ios_base::fixed;
        precision = 8;
        width     = 14;
        break;

    case print_format_t::SCIENTIFIC:
        options   = std::ios_base::scientific;
        precision = 4;
        width     = 10;
        break;

    case print_format_t::EXTSCIENTIFIC:
        options   = std::ios_base::scientific;
        precision = 8;
        width     = 18;
    }

    strm.setf(options, std::ios_base::floatfield);
    strm.precision(precision);
    return width;
}

void array::_restore_print_format(std::ostream& strm) const {
    strm.setf(std::ios_base::fmtflags(0), std::ios_base::floatfield);
    strm.precision(6);
    strm.width(0);
}

std::ostream& linear_algebra::operator<<(std::ostream& strm, const array& thearray) {
    thearray.print(print_format_t::SHORT, strm);
    return strm;
}

linear_algebra::size_t vec::dim() const {
    return _size;
}

linear_algebra::size_t vec::dimension() const {
    return _size;
}

vec::vec() { }

vec::vec(linear_algebra::size_t thesize)
    : array(thesize) { }

vec::vec(linear_algebra::size_t thesize, initial_values_t iv)
    : array(thesize, iv) { }

vec::vec(linear_algebra::size_t thesize, double* values)
    : array(thesize, values) { }

vec::vec(const vec& src)
    : array(src) { }

vec::vec(const char* fmt) {
    _format(fmt);
}

vec::~vec() { }

const real_t& vec::operator()(const index_t& i) const {
    if (doesindex_tExceedvecDimension(i))
        throw std::runtime_error {"The index_t is out of range."};

    return _elements.get()[i - 1];
}

real_t& vec::operator()(const index_t& i) {
    return const_cast<real_t&>(static_cast<const vec&>(*this)(i));
}

bool vec::doesindex_tExceedvecDimension(const index_t& i) const {
    if ((i < 1) || (_size < i))
        return true;

    return false;
}

vec& vec::operator=(const vec& src) {
    if (this == &src)
        return *this;

    _elements.reset();
    _size     = src._size;
    _elements = std::unique_ptr<real_t>(new real_t[_size]);
    _copy_elements(_size, src._elements, _elements);

    return *this;
}

bool vec::operator==(const vec& rhs) const {
    if (this == &rhs)
        return true;

    return array::operator==(rhs);
}

bool vec::isSimilarTo(const vec& v) const {
    if (*this == v)
        return true;

    return array::isSimilarTo(v);
}

bool vec::operator!=(const vec& rhs) const {
    if (*this == rhs)
        return false;

    return true;
}

vec vec::operator+(const vec& rhs) const {
    if (_size != rhs._size)
        throw std::runtime_error {"The dimensions not matched."};

    vec result(_size);
    _add_elements(_size, _elements, rhs._elements, result._elements);
    return result;
}

vec vec::operator-(const vec& rhs) const {
    if (_size != rhs._size)
        throw std::runtime_error {"The dimensions not matched."};

    vec result(_size);
    _subtract_elements(_size, _elements, rhs._elements, result._elements);
    return result;
}

vec& vec::operator+=(const vec& rhs) {
    if (_size != rhs._size)
        throw std::runtime_error {"The dimensions not matched."};

    _add_elements(_size, _elements, rhs._elements, _elements);
    return *this;
}

vec& vec::operator-=(const vec& rhs) {
    if (_size != rhs._size)
        throw std::runtime_error {"The dimensions not matched."};

    _subtract_elements(_size, _elements, rhs._elements, _elements);
    return *this;
}

vec vec::operator+(const double s) const {
    vec result(_size);
    for (index_t i = 0; i != _size; i++)
        result._elements.get()[i] = _elements.get()[i] + s;

    return result;
}

vec vec::operator-(const double s) const {
    vec result(_size);
    for (index_t i = 0; i != _size; i++)
        result._elements.get()[i] = _elements.get()[i] - s;

    return result;
}

vec& vec::operator+=(const double s) {
    for (index_t i = 0; i != _size; i++)
        _elements.get()[i] += s;

    return *this;
}

vec& vec::operator-=(const double s) {
    for (index_t i = 0; i != _size; i++)
        _elements.get()[i] -= s;

    return *this;
}

vec vec::operator-() const {
    vec result(_size);
    _negate_elements(_size, _elements, result._elements);
    return result;
}

vec vec::operator*(const double& s) const {
    vec result(_size);
    _multiply_scalar_to_elements(_size, _elements, real_t(s), result._elements);
    return result;
}

vec vec::operator/(const double& s) const {
    if (fabs(s) < TOLERANCE)
        throw std::runtime_error {"Attempted to divide by zero."};

    vec result(_size);
    _divide_elements_by_scalar(_size, _elements, real_t(s), result._elements);
    return result;
}

vec& vec::operator*=(const double& s) {
    _multiply_scalar_to_elements(_size, _elements, real_t(s), _elements);
    return *this;
}

vec& vec::operator/=(const double& s) {
    if (fabs(s) < TOLERANCE) {
        throw std::runtime_error {"Attempted to divide by zero."};
    }
    _divide_elements_by_scalar(_size, _elements, real_t(s), _elements);
    return *this;
}

vec vec::operator*(const mat& m) const {
    if (_size != m.height())
        throw std::runtime_error {"The dimensions not matched."};

    linear_algebra::size_t h  = m.height();
    linear_algebra::size_t w  = m.width();
    real_t*                _m = m.elements();
    real_t*                r  = new real_t[w];
    double                 a  = 0.;

    for (index_t j = 0; j != w; j++) {
        for (index_t i = 0; i != h; i++) {
            a += _elements.get()[i] * _m[j * h + i];
        }
        r[j] = a;
        a    = 0.;
    }

    vec R(w, r);
    delete[] r;
    return R;
}

void vec::insert(const double& v, const index_t idx) {
    if ((_size == 0) && (idx == 1)) {
        append(v);
        return;
    }
    else if (doesindex_tExceedvecDimension(idx))
        throw std::runtime_error {"The index_t is out of range."};

    real_t* nel = new real_t[_size + 1];

    _copy_elements(idx - 1, _elements, nel);

    nel[idx - 1] = v;

    _copy_elements(_size - idx + 1, &_elements.get()[idx - 1], &nel[idx]);

    _elements.reset(nel);
    _size++;
}

void vec::append(const double& v) {
    real_t* nel = new real_t[_size + 1];
    _copy_elements(_size, _elements, nel);
    nel[_size] = v;
    _elements.reset(nel);
    _size++;
}

void vec::insert_vec(const vec& v, const index_t idx) {
    if ((_size == 0) && (idx == 1)) {
        append_vec(v);
        return;
    }
    else if (doesindex_tExceedvecDimension(idx))
        throw std::runtime_error {"The index_t is out of range."};

    real_t* nel = new real_t[_size + v._size];
    _copy_elements(idx - 1, _elements, nel);
    _copy_elements(v._size, v._elements, &nel[idx - 1]);
    _copy_elements(_size - idx + 1, &_elements.get()[idx - 1], &nel[idx - 1 + v._size]);

    _elements.reset(nel);
    _size += v._size;
}

void vec::append_vec(const vec& v) {
    real_t* nel = new real_t[_size + v._size];
    _copy_elements(_size, _elements, nel);
    _copy_elements(v._size, v._elements, &nel[_size]);

    _elements.reset(nel);
    _size += v._size;
}

void vec::remove(const index_t idx) {
    if (doesindex_tExceedvecDimension(idx))
        throw std::runtime_error {"The index_t is out of range."};

    real_t* nel;
    if (_size == 1) {
        nel = NULL;
    }
    else {
        nel = new real_t[_size - 1];
        _copy_elements(idx - 1, _elements, nel);
        _copy_elements(_size - idx, &_elements.get()[idx], &nel[idx - 1]);
    }
    _elements.reset(nel);
    _size--;
}

void vec::remove_elements(const index_t begin, const index_t end) {
    if (doesindex_tExceedvecDimension(begin) || doesindex_tExceedvecDimension(end))
        throw std::runtime_error {"The index_t is out of range."};

    if (begin > end)
        throw std::runtime_error {"Operation not permitted."};

    real_t* nel = new real_t[_size - end + begin - 1];
    _copy_elements(begin - 1, _elements, nel);

    _copy_elements(_size - end, &_elements.get()[end], &nel[begin - 1]);
    _elements.reset(nel);
    _size -= (end - begin + 1);
}

double vec::norm(const unsigned p) const {
    if (p == 0)
        throw std::runtime_error {"Operation not permitted."};

    double normOfvec;
    if (p == Inf) {
        index_t maxindex_tInvec = linear_algebra::index_max_absolute_elem(_size, _elements.get());
        normOfvec               = _elements.get()[maxindex_tInvec];
    }
    else {
        double reciprocalP          = 1.0 / double(p);
        double poweredSumOfElements = 0.0;
        for (index_t i = 0; i != _size; i++)
            poweredSumOfElements += std::pow(_elements.get()[i], double(p));

        normOfvec = std::pow(poweredSumOfElements, reciprocalP);
    }
    return normOfvec;
}

vec vec::unit(const unsigned p) const {
    vec    u = (*this);
    double n = this->norm(p);
    if (n > TOLERANCE) {
        u /= n;
    }
    return u;
}

vec vec::normalize(const unsigned p) {
    double n = this->norm(p);
    if (n > TOLERANCE) {
        (*this) /= n;
    }
    return (*this);
}

vec vec::transpose() const {
    vec transposed(*this);
    return transposed;
}

vec vec::T() const {
    return this->transpose();
}

double vec::operator*(const vec& v) const {
    if (_size != v._size)
        throw std::runtime_error {"The dimensions not matched."};

    double ip = 0.0;
    for (index_t i = 0; i != _size; i++)
        ip += _elements.get()[i] * v._elements.get()[i];

    return ip;
}

void vec::print(print_format_t fmt, std::ostream& strm) const {
    strm << std::endl;
    if (_size == 0) {
        strm << " Empty vec" << std::endl;
    }
    else {
        strm << " " << _size << " dimensional vec\n";
    }
    char prev = strm.fill('-');
    strm << std::setw(80) << '-' << std::endl;
    strm.fill(prev);

    int width = _set_print_format(fmt, strm);

    for (index_t i = 0; i != _size; i++) {
        strm.width(width);
        strm << _elements.get()[i] << std::endl;
    }
    strm << std::endl << std::endl;

    _restore_print_format(strm);
}

vec linear_algebra::operator+(const double s, const vec& v) {
    return v + s;
}

vec linear_algebra::operator-(const double s, const vec& v) {
    return -v + s;
}

vec linear_algebra::operator*(const double& s, const vec& A) {
    linear_algebra::size_t sz = A.size();
    vec                    result(sz);
    linear_algebra::_multiply_scalar_to_elements(sz, A.elements(), real_t(s), result.elements());
    return result;
}

double linear_algebra::norm(const vec& v, const unsigned p) {
    return v.norm(p);
}

vec linear_algebra::unit(const vec& v, const unsigned p) {
    return v.unit();
}

vec linear_algebra::normalize(vec& v, const unsigned p) {
    return v.normalize(p);
}

vec linear_algebra::transpose(const vec& v) {
    return v.transpose();
}

double linear_algebra::dot(const vec& v1, const vec& v2) {
    if (v1.dim() != v2.dim())
        throw std::runtime_error {"The dimensions not matched."};

    real_t* _v1 = v1.elements();
    real_t* _v2 = v2.elements();
    double  ip  = 0.0;

    for (index_t i = 0; i != v1.dim(); i++)
        ip += _v1[i] * _v2[i];

    return ip;
}

void vec::_format(const char* fmt) {
    bool                   isColumnvec            = false;
    bool                   isRowvec               = false;
    bool                   delimiterPreceded      = false;
    bool                   onlyWhiteSpacePreceded = false;
    bool                   numericRepPreceded     = false;
    bool                   endOfLine              = false;
    char                   ch;
    linear_algebra::size_t count = 0;
    unsigned               idx   = 0;

    while (!endOfLine) {
        ch = fmt[idx++];
        if (ch == ',' || ch == ';') {
            delimiterPreceded      = true;
            onlyWhiteSpacePreceded = false;
        }
        if ((ch == '\t' || ch == ' ') && delimiterPreceded == false)
            onlyWhiteSpacePreceded = true;
        if (ch == '\t' || ch == ' ' || ch == ',' || ch == ';' || ch == '\0') {

            if (ch == ';' && isRowvec == true)
                throw std::runtime_error {"The construction string not adequate."};

            if (ch == ',' && isColumnvec == true)
                throw std::runtime_error {"The construction string not adequate."};

            if (numericRepPreceded == true) {
                if (ch == ';')
                    isColumnvec = true;
                if (ch == ',')
                    isRowvec = true;
                count++;
                numericRepPreceded = false;
            }
            if (ch == '\0') {
                endOfLine = true;
            }
        }
        else {
            if (count == 0 && delimiterPreceded == true)
                throw std::runtime_error {"The construction string not adequate."};

            if (onlyWhiteSpacePreceded == true) {
                if (isColumnvec == true)
                    throw std::runtime_error {"The construction string not adequate."};

                isRowvec = true;
            }
            numericRepPreceded     = true;
            delimiterPreceded      = false;
            onlyWhiteSpacePreceded = false;
        }
    }
    _size = count;
    if (_size > 0) {
        _elements = std::unique_ptr<real_t>(new real_t[_size]);
        for (index_t i = 0; i != _size; i++)
            _elements.get()[i] = 0.0;
    }
    else
        _elements = nullptr;

    char buf[128];
    endOfLine    = false;
    idx          = 0;
    unsigned ddx = 0;
    unsigned i   = 0;
    while (!endOfLine) {
        ch = fmt[idx++];
        if (ch == '\t' || ch == ' ' || ch == ',' || ch == ';' || ch == '\0') {
            if (ddx > 0) {
                buf[ddx]           = '\0';
                _elements.get()[i] = atof(buf);
                i++;
                ddx = 0;
            }
            if (ch == '\0')
                endOfLine = true;
        }
        else
            buf[ddx++] = ch;
    }
}

linear_algebra::size_t mat::width() const {
    return _width;
}

linear_algebra::size_t mat::count_cols() const {
    return _width;
}

linear_algebra::size_t mat::height() const {
    return _height;
}

linear_algebra::size_t mat::count_rows() const {
    return _height;
}

void mat::set_dimension(linear_algebra::size_t h, linear_algebra::size_t w) {
    _width  = w;
    _height = h;
    this->recount();
}

mat::mat()
    : array()
    , _width(0)
    , _height(0) { }

mat::mat(const linear_algebra::size_t h, const linear_algebra::size_t w)
    : array(h * w)
    , _width(w)
    , _height(h) { }

mat::mat(const linear_algebra::size_t h, const linear_algebra::size_t w, const initial_values_t iv)
    : array(h * w, iv)
    , _width(w)
    , _height(h) { }

mat::mat(const linear_algebra::size_t h, const linear_algebra::size_t w, double* values)
    : array(h * w, values)
    , _width(w)
    , _height(h) { }

mat::mat(const mat& src)
    : array(src)
    , _width(src._width)
    , _height(src._height) { }

mat::mat(const char* fmt) {
    _format(fmt);
}

const real_t& mat::operator()(const index_t& i, const index_t& j) const {
    if (indices_exceed_dim(i, j))
        throw std::runtime_error {"The index_t is out of range."};

    return _elements.get()[(j - 1) * _height + (i - 1)];
}

real_t& mat::operator()(const index_t& i, const index_t& j) {
    return const_cast<real_t&>(static_cast<const mat&>(*this)(i, j));
}

mat& mat::operator=(const mat& src) {
    if (this == &src)
        return *this;

    _elements.reset(new real_t[src.size()]);
    _height = src.height();
    _width  = src.width();
    recount();
    _copy_elements(_size, src._elements, _elements);

    return *this;
}

bool mat::operator==(const mat& m) const {
    if (this == &m)
        return true;

    if (this->same_dim(m) == false)
        return false;

    return array::operator==(m);
}

bool mat::operator!=(const mat& m) const {
    if ((*this) == m)
        return false;

    return true;
}

bool mat::isSimilarTo(const mat& m) const {
    if (*this == m)
        return true;

    if (this->same_dim(m) == false)
        return false;

    return array::isSimilarTo(m);
}

mat mat::operator+(const mat& m) const {
    if (this->same_dim(m) == false)
        throw std::runtime_error {"The dimensions of matrices not match."};

    mat r(_height, _width);
    _add_elements(_size, _elements, m._elements, r._elements);
    return r;
}

mat mat::operator-(const mat& m) const {
    if (this->same_dim(m) == false)
        throw std::runtime_error {"The dimensions of matrices not match."};

    mat r(_height, _width);
    _subtract_elements(_size, _elements, m._elements, r._elements);
    return r;
}

mat& mat::operator+=(const mat& m) {
    if (this->same_dim(m) == false)
        throw std::runtime_error {"The dimensions of matrices not match."};

    _add_elements(_size, _elements, m._elements, _elements);
    return *this;
}

mat& mat::operator-=(const mat& m) {
    if (this->same_dim(m) == false)
        throw std::runtime_error {"The dimensions of matrices not match."};

    _subtract_elements(_size, _elements, m._elements, _elements);
    return *this;
}

mat mat::operator+(const double s) const {
    mat r(_height, _width);
    _add_elements(_size, _elements, s, r._elements);
    return r;
}

mat mat::operator-(const double s) const {
    mat r(_height, _width);
    _subtract_elements(_size, _elements, s, r._elements);
    return r;
}

mat& mat::operator+=(const double s) {
    _add_elements(_size, _elements, s);
    return *this;
}

mat& mat::operator-=(const double s) {
    _subtract_elements(_size, _elements, s);
    return *this;
}

mat mat::operator-() const {
    mat r(_height, _width);
    _negate_elements(_size, _elements, r._elements);
    return r;
}

mat mat::operator*(const double& s) const {
    mat r(_height, _width);
    _multiply_scalar_to_elements(_size, _elements, s, r._elements);
    return r;
}

mat mat::operator*(const mat& m) const {
    if (_width != m._height)
        throw std::runtime_error {"The dimensions of matrices not match."};

    real_t* r = new real_t[_height * m._width];

    double a = 0.;

    for (index_t i = 0; i != _height; i++) {
        for (index_t j = 0; j != m._width; j++) {
            for (index_t k = 0; k != _width; k++) {
                a += _elements.get()[k * _height + i] * m._elements.get()[j * m._height + k];
            }
            r[j * _height + i] = a;
            a                  = 0.0;
        }
    }

    mat R(_height, m._width, r);
    delete[] r;
    return R;
}

vec mat::operator*(const vec& v) const {
    if (_width != v.dim())
        throw std::runtime_error {"The dimensions of matrices not match."};

    real_t* r  = new real_t[_height];
    real_t* _v = v.elements();
    double  a  = 0.;

    for (index_t i = 0; i != _height; i++) {
        for (index_t j = 0; j != _width; j++) {
            a += _elements.get()[j * _height + i] * _v[j];
        }
        r[i] = a;
        a    = 0.;
    }

    vec R(_height, r);
    delete[] r;
    return R;
}

mat& mat::operator*=(const double& s) {
    _multiply_scalar_to_elements(_size, _elements, s, _elements);
    return *this;
}

mat& mat::operator*=(const mat& m) {
    if (_width != m._height)
        throw std::runtime_error {"The dimensions of matrices not match."};

    real_t* r = new real_t[_height * m._width];

    double a = 0.0;

    for (index_t i = 0; i != _height; i++) {
        for (index_t j = 0; j != m._width; j++) {
            for (index_t k = 0; k != _width; k++) {
                a += _elements.get()[k * _height + i] * m._elements.get()[j * m._height + k];
            }
            r[j * _height + i] = a;
            a                  = 0.0;
        }
    }
    this->set_dimension(_height, m._width);
    _copy_elements(_size, r, _elements);
    delete[] r;
    return *this;
}

mat mat::operator/(const double& s) const {
    if (s == 0.)
        throw std::runtime_error {"Division by zero."};

    return (*this) * (1.0 / s);
}

void mat::insert_row(const vec& v, const index_t p) {
    if (_size == 0) {
        append_row(v);
        return;
    }

    if (v.dim() != _width)
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (index_exceeds_rows(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    if (_width == 0)
        throw std::runtime_error {"Operation not permitted."};

    real_t* nel = new real_t[(_height + 1) * _width];

    _replace_rows_with_mat(nel, _height + 1, _width, 1, p - 1, *this, 1);

    real_t* _v = v.elements();
    for (index_t i = 0; i != v.dim(); i++)
        nel[i * (_height + 1) + p - 1] = _v[i];

    _replace_rows_with_mat(nel, _height + 1, _width, p + 1, _height - p + 1, *this, p);

    _height++;
    recount();
    _elements.reset(nel);
}

void mat::insert_rows(const mat& m, const index_t p) {
    if (_size == 0) {
        append_rows(m);
        return;
    }

    if (m._width != _width)
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (index_exceeds_rows(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    if (_width == 0)
        throw std::runtime_error {"Operation not permitted."};

    linear_algebra::size_t inc = m._height;
    real_t*                nel = new real_t[(_height + m._height) * _width];

    _replace_rows_with_mat(nel, _height + inc, _width, 1, p - 1, *this, 1);
    _replace_rows_with_mat(nel, _height + inc, _width, p, inc, m, 1);
    _replace_rows_with_mat(nel, _height + inc, _width, p + inc, _height - p + 1, *this, p);

    _height += inc;
    recount();
    _elements.reset(nel);
}

void mat::append_row(const vec& v) {
    if (_size == 0) {
        _size = _width = v.dim();
        _height        = 1;
        _elements.reset(new real_t[_size]);
        _copy_elements(_size, v.elements(), _elements);
        return;
    }

    if (v.dim() != _width)
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (_width == 0)
        throw std::runtime_error {"Operation not permitted."};

    real_t* nel = new real_t[(_height + 1) * _width];

    _replace_rows_with_mat(nel, _height + 1, _width, 1, _height, *this, 1);

    real_t* _v = v.elements();
    for (index_t i = 0; i != v.dim(); i++)
        nel[i * (_height + 1) + _height] = _v[i];

    _height++;
    recount();
    _elements.reset(nel);
}

void mat::append_rows(const mat& m) {
    if (_size == 0) {
        _size   = m.size();
        _width  = m.width();
        _height = m.height();
        _elements.reset(new real_t[_size]);
        _copy_elements(_size, m.elements(), _elements);
        return;
    }

    if (_width != m.width())
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (_width == 0)
        throw std::runtime_error {"Operation not permitted."};

    linear_algebra::size_t inc = m.height();
    real_t*                nel = new real_t[(_height + inc) * _width];

    _replace_rows_with_mat(nel, _height + inc, _width, 1, _height, *this, 1);
    _replace_rows_with_mat(nel, _height + inc, _width, _height + 1, inc, m, 1);

    _height += inc;
    recount();
    _elements.reset(nel);
}

void mat::delete_row(const index_t p) {
    if (index_exceeds_rows(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    if (_height == 1) {
        _size = _height = _width = 0;
        _elements.reset();
        return;
    }

    real_t* nel = new real_t[(_height - 1) * _width];
    _replace_rows_with_mat(nel, _height - 1, _width, 1, p - 1, *this, 1);
    _replace_rows_with_mat(nel, _height - 1, _width, p, _height - p, *this, p + 1);

    _height--;
    recount();
    _elements.reset(nel);
}

void mat::swap_rows(const index_t p, const index_t q) {
    if ((p == q) || (_width == 0))
        return;

    if ((index_exceeds_rows(p) == true) || (index_exceeds_rows(q) == true))
        throw std::runtime_error {"The index_t is out of range."};

    for (index_t j = 0; j != _width; j++) {
        real_t temp                          = _elements.get()[_height * j + p - 1];
        _elements.get()[_height * j + p - 1] = _elements.get()[_height * j + q - 1];
        _elements.get()[_height * j + q - 1] = temp;
    }
}

void mat::insert_col(const vec& v, const index_t p) {
    if (_size == 0) {
        append_col(v);
        return;
    }

    if (v.dim() != _height)
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (index_exceeds_cols(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    if (_height == 0)
        throw std::runtime_error {"Operation not permitted."};

    real_t* nel = new real_t[_height * (_width + 1)];

    _replace_columns_with_mat(nel, _height, _width + 1, 1, p - 1, *this, 1);

    real_t* _v = v.elements();
    for (index_t j = 0; j != v.dim(); j++)
        nel[(p - 1) * _height + j] = _v[j];

    _replace_columns_with_mat(nel, _height, _width + 1, p + 1, _width - p + 1, *this, p);

    _width++;
    recount();
    _elements.reset(nel);
}

void mat::insert_cols(const mat& m, const index_t p) {
    if (_size == 0) {
        append_cols(m);
        return;
    }

    if (m._height != _height)
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (index_exceeds_cols(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    if (_height == 0)
        throw std::runtime_error {"Operation not permitted."};

    linear_algebra::size_t inc = m._width;
    real_t*                nel = new real_t[_height * (_width + m._width)];

    _replace_columns_with_mat(nel, _height, _width + inc, 1, p - 1, *this, 1);
    _replace_columns_with_mat(nel, _height, _width + inc, p, inc, m, 1);
    _replace_columns_with_mat(nel, _height, _width + inc, p + inc, _width - p + 1, *this, p);

    _width += inc;
    recount();
    _elements.reset(nel);
}

void mat::append_col(const vec& v) {
    if (_size == 0) {
        _height = _size = v.dim();
        _width          = 1;
        _elements.reset(new real_t[_size]);
        _copy_elements(_size, v.elements(), _elements);
        return;
    }

    if (v.dim() != _height)
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (_height == 0)
        throw std::runtime_error {"Operation not permitted."};

    real_t* nel = new real_t[_height * (_width + 1)];
    _replace_columns_with_mat(nel, _height, _width + 1, 1, _width, *this, 1);

    real_t* _v = v.elements();
    for (index_t j = 0; j != v.dim(); j++)
        nel[_height * _width + j] = _v[j];

    _width++;
    recount();
    _elements.reset(nel);
}

void mat::append_cols(const mat& m) {
    if (_size == 0) {
        _size   = m.size();
        _width  = m.width();
        _height = m.height();
        _elements.reset(new real_t[_size]);
        _copy_elements(_size, m.elements(), _elements);
        return;
    }

    if (_height != m.height())
        throw std::runtime_error {"The dimensions of matrices not match."};

    if (_height == 0)
        throw std::runtime_error {"Operation not permitted."};

    linear_algebra::size_t inc = m.width();
    real_t*                nel = new real_t[_height * (_width + inc)];

    _replace_columns_with_mat(nel, _height, _width + inc, 1, _width, *this, 1);

    _replace_columns_with_mat(nel, _height, _width + inc, _width + 1, inc, m, 1);

    _width += inc;
    recount();
    _elements.reset(nel);
}

void mat::delete_col(const index_t p) {
    if (index_exceeds_cols(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    if (_width == 1) {
        _size = _height = _width = 0;
        _elements.reset();
        return;
    }

    real_t* nel = new real_t[_height * (_width - 1)];

    _replace_columns_with_mat(nel, _height, _width - 1, 1, p - 1, *this, 1);

    _replace_columns_with_mat(nel, _height, _width - 1, p, _width - p, *this, p + 1);

    _width--;
    recount();
    _elements.reset(nel);
}

void mat::swap_cols(const index_t p, const index_t q) {
    if ((p == q) || (_height == 0))
        return;

    if ((index_exceeds_cols(p) == true) || (index_exceeds_cols(q) == true))
        throw std::runtime_error {"The index_t is out of range."};

    for (index_t i = 0; i != _height; i++) {
        real_t temp                            = _elements.get()[_height * (p - 1) + i];
        _elements.get()[_height * (p - 1) + i] = _elements.get()[_height * (q - 1) + i];
        _elements.get()[_height * (q - 1) + i] = temp;
    }
}

vec mat::row(const index_t p) const {
    if (index_exceeds_rows(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    real_t* el = new real_t[_width];
    for (index_t j = 0; j != _width; j++)
        el[j] = _elements.get()[_height * j + p - 1];

    vec v(_width, el);
    delete[] el;

    return v;
}

vec mat::column(const index_t p) const {
    if (index_exceeds_cols(p) == true)
        throw std::runtime_error {"The index_t is out of range."};

    real_t* el = new real_t[_height];
    for (index_t i = 0; i != _height; i++)
        el[i] = _elements.get()[_height * (p - 1) + i];

    vec v(_height, el);
    delete[] el;

    return v;
}

vec mat::diag() const {
    integer_t n  = _min(_width, _height);
    real_t*   el = new real_t[n];

    for (index_t i = 0; i < n; i++)
        el[i] = _elements.get()[_height * i + i];

    vec v(n, el);
    delete[] el;

    return v;
}

mat mat::submat(const index_t top,
                const index_t bottom,
                const index_t left,
                const index_t right) const {
    if ((index_exceeds_rows(top) == true) || (index_exceeds_rows(bottom) == true) ||
        (index_exceeds_cols(left) == true) || (index_exceeds_cols(right) == true))
        throw std::runtime_error {"The index_t is out of range."};

    if ((right < left) || (bottom < top))
        throw std::runtime_error {"Operation not permitted."};

    mat     m(bottom - top + 1, right - left + 1);
    real_t* el = m.elements();

    for (index_t j = left - 1; j != right; j++) {
        for (index_t i = top - 1; i != bottom; i++) {
            *el = _elements.get()[_height * j + i];
            el++;
        }
    }
    return m;
}

mat mat::rows(const index_t top, const index_t bottom) const {
    return submat(top, bottom, 1, _width);
}

mat mat::columns(const index_t left, const index_t right) const {
    return submat(1, _height, left, right);
}

mat mat::transpose() const {
    real_t* nel = new real_t[_size];

    for (index_t i = 0; i != _height; i++) {
        for (index_t j = 0; j != _width; j++) {
            nel[i * _width + j] = _elements.get()[j * _height + i];
        }
    }
    mat m = mat(_width, _height, nel);
    return m;
}

mat mat::T() const {
    return this->transpose();
}

mat mat::inv() const {
    if (_height != _width)
        throw std::runtime_error {"Operation not permitted."};

    integer_t  N = _height;
    mat        m(*this);
    integer_t  INFO;
    integer_t* IPIV = new integer_t[_height];

    dgetrf_(&N, &N, m.elements(), &N, IPIV, &INFO);

    double d = m._prod_diagonal();
    if (std::fabs(d) < TOLERANCE)
        throw std::runtime_error {"The mat is singular."};

    N              = _height;
    integer_t prod = _height * _height;
    real_t*   WORK = new real_t[_height];

    dgetri_(&N, m.elements(), &N, IPIV, WORK, &prod, &INFO);

    delete[] WORK;

    delete[] IPIV;
    return m;
}

bool mat::square() const {
    if (_width == _height) {
        return true;
    }
    else {
        return false;
    }
}

bool mat::symmetric() const {
    if (*this == this->transpose()) {
        return true;
    }
    else {
        return false;
    }
}

bool mat::skew_symmetric() const {
    if (*this == -(this->transpose())) {
        return true;
    }
    else {
        return false;
    }
}

bool mat::pseudo_symmetric() const {
    if (this->isSimilarTo(this->transpose())) {
        return true;
    }
    else {
        return false;
    }
}

bool mat::pseudo_skew_symmetric() const {
    if (this->isSimilarTo(-(this->transpose()))) {
        return true;
    }
    else {
        return false;
    }
}

double mat::trace() const {
    if (_width != _height) {
        throw std::runtime_error {"Operation not permitted."};
    }
    double tr = 0.0;
    for (index_t i = 0; i != _height; i++) {
        tr += _elements.get()[i * _height + i];
    }
    return tr;
}

double mat::det() const {
    if (_width != _height) {
        throw std::runtime_error {"Operation not permitted."};
    }

    real_t* LU = new real_t[_size];
    this->copyValuesTo(LU);
    integer_t  N = _height;
    integer_t  INFO;
    integer_t* IPIV = new integer_t[_height];

    dgetrf_(&N, &N, LU, &N, IPIV, &INFO);

    double d = 0.0;
    if (INFO != 0) {
        return d;
    }

    d = 1.0;
    for (index_t i = 0; i != _height; i++) {
        if (IPIV[i] < (i + 1)) {
            d *= -LU[i * _height + i];
        }
        else {
            d *= LU[i * _height + i];
        }
    }
    delete[] LU;
    return d;
}

vec mat::svd() const {
    mat U(_height, _height);
    mat V(_width, _width);
    return svd(U, V);
}

vec mat::svd(mat& U, mat& V) const {
    integer_t m    = _height;
    integer_t n    = _width;
    integer_t lda  = m;
    integer_t ldu  = m;
    integer_t ldvt = n;
    integer_t ds   = _min(m, n);

    char      jobz  = 'A';
    integer_t lwork = 3 * ds * ds + _max(_max(m, n), 5 * _min(m, n) * _min(m, n) + 4 * _min(m, n));
    real_t*   el    = new real_t[_size];
    this->copyValuesTo(el);
    real_t*    s     = new real_t[ds];
    real_t*    u     = new real_t[ldu * m];
    real_t*    vt    = new real_t[ldvt * n];
    real_t*    work  = new real_t[_max(1, lwork)];
    integer_t* iwork = new integer_t[8 * ds];
    integer_t  info;

    dgesdd_(&jobz, &m, &n, el, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, iwork, &info);

    if (info > 0) {
        throw std::runtime_error {"The algorithm failed to converge."};
    }

    vec S(ds, s);
    U = mat(m, m, u);
    V = mat(n, n, vt).transpose();

    return S;
}

mat mat::eig(job_spec_t js) const {
    mat e(1, _width);

    integer_t N = _width;
    mat       A(*this);
    integer_t LDA = N;
    integer_t INFO;

    if (this->symmetric() == true) {

        char JOBZ;
        if (js == job_spec_t::EIGENVALUE_ONLY) {
            JOBZ = 'N';
        }
        else {
            JOBZ = 'V';
        }
        char      UPLO  = 'U';
        real_t*   W     = new real_t[N];
        integer_t LWORK = 3 * N;
        real_t*   WORK  = new real_t[LWORK];

        dsyev_(&JOBZ, &UPLO, &N, A.elements(), &LDA, W, WORK, &LWORK, &INFO);

        if (INFO == 0) {
            for (index_t i = 0; i < N; i++) {
                e(1, i + 1) = W[i];
            }
            if (js != job_spec_t::EIGENVALUE_ONLY) {
                e.append_rows(A);
            }
            delete[] W;
        }
        else {
            delete[] W;
            throw std::runtime_error {"Failed to calculate eigen_values."};
        }

        ;
    }
    else {

        mat       VL, VR;
        char      JOBVL, JOBVR;
        integer_t LDVL = N;
        integer_t LDVR = N;
        switch (js) {
        case job_spec_t::EIGENVALUE_ONLY:
            JOBVL = JOBVR = 'N';
            break;

        case job_spec_t::EIGENVALUE_AND_EIGENVEC:
            JOBVL = JOBVR = 'V';
            VL            = mat(LDVL, N);
            VR            = mat(LDVR, N);
            break;

        case job_spec_t::EIGENVALUE_AND_LEFT_EIGENVEC:
            JOBVL = 'V';
            JOBVR = 'N';
            VL    = mat(LDVL, N);
            break;

        case job_spec_t::EIGENVALUE_AND_RIGHT_EIGENVEC:
            JOBVL = 'N';
            JOBVR = 'V';
            VR    = mat(LDVR, N);
            break;
        }
        real_t* WR = new real_t[N];
        real_t* WI = new real_t[N];

        integer_t LWORK = 4 * N;
        real_t*   WORK  = new real_t[LWORK];

        dgeev_(&JOBVL,
               &JOBVR,
               &N,
               A.elements(),
               &LDA,
               WR,
               WI,
               VL.elements(),
               &LDVL,
               VR.elements(),
               &LDVR,
               WORK,
               &LWORK,
               &INFO);

        if (INFO == 0) {

            for (index_t i = 0; i < N; i++) {
                e(1, i + 1) = WR[i];
            }
            vec imaginery = vec(N, WI);
            e.append_row(imaginery);

            switch (js) {
            case job_spec_t::EIGENVALUE_ONLY:
                break;

            case job_spec_t::EIGENVALUE_AND_EIGENVEC:
                e.append_rows(VL);
                e.append_rows(VR);
                break;

            case job_spec_t::EIGENVALUE_AND_LEFT_EIGENVEC:
                e.append_rows(VL);
                break;

            case job_spec_t::EIGENVALUE_AND_RIGHT_EIGENVEC:
                e.append_rows(VR);
                break;
            }

            delete[] WR;
            delete[] WI;
            delete[] WORK;
        }
        else {
            delete[] WR;
            delete[] WI;
            delete[] WORK;

            throw std::runtime_error {"Failed to calculate eigen_values."};
        }
    }
    return e;
}

mat mat::eigen_values() const {
    if (this->square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    return eig(job_spec_t::EIGENVALUE_ONLY);
}

mat mat::eigen_vecs() const {
    if (this->square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    return eig(job_spec_t::EIGENVALUE_AND_RIGHT_EIGENVEC);
}

mat mat::left_eigen_vecs() const {
    if (this->square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    return eig(job_spec_t::EIGENVALUE_AND_LEFT_EIGENVEC);
}

mat mat::left_right_eigen_vecs() const {
    if (this->square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    return eig(job_spec_t::EIGENVALUE_AND_EIGENVEC);
}

vec mat::real_eigen_values_symmetric_part() const {
    vec e(_width);
    if (symmetric() == false) {
        mat As = .5 * (*this + (this->transpose()));
        e      = As.eigen_values().row(1);
    }
    else {
        e = eigen_values().row(1);
    }
    return e;
}

bool mat::positive_definite() const {
    if (square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    vec e = real_eigen_values_symmetric_part();

    bool positiveDefiniteness = true;
    for (index_t i = 1; i != (_width + 1); i++) {
        if (e(i) <= 0.0) {
            positiveDefiniteness = false;
            break;
        }
    }
    return positiveDefiniteness;
}

bool mat::positive_semi_definite() const {
    if (square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    vec e = real_eigen_values_symmetric_part();

    bool positiveSemiDefiniteness = true;
    for (index_t i = 1; i != (_width + 1); i++) {
        if (e(i) < 0.0) {
            positiveSemiDefiniteness = false;
            break;
        }
    }

    return positiveSemiDefiniteness;
}

bool mat::pseudo_positive_semi_definite() const {
    if (positive_semi_definite() == true) {
        return true;
    }

    if (square() == false) {
        throw std::runtime_error {"mat is not square."};
    }

    vec e = real_eigen_values_symmetric_part();

    bool positiveSemiDefiniteness = true;
    for (index_t i = 1; i != (_width + 1); i++) {
        if (e(i) < -TOLERANCE) {
            positiveSemiDefiniteness = false;
            break;
        }
    }

    return positiveSemiDefiniteness;
}

void mat::print_eigen_value_vec(print_format_t fmt, std::ostream& strm) const {
    int                    width = _set_print_format(fmt, strm);
    linear_algebra::size_t n     = _width;
    if ((_height == 1) || (_height == 1 + n)) {
        strm << std::endl << "real_t eigen_values:" << std::endl;
        for (index_t j = 1; j != n + 1; j++) {
            strm.width(width);
            strm << "[" << (*this)(1, j) << " ]" << std::endl;
        }
    }
    else if ((_height == 2) || (_height == 2 + n) || (_height == 2 + 2 * n)) {
        strm << std::endl << "Complex eigen_values:" << std::endl;
        for (index_t j = 1; j != n + 1; j++) {
            strm << "[";
            strm.width(width);
            strm << (*this)(1, j) << " ";
            if ((*this)(2, j) >= 0) {
                strm << "+";
            }
            else {
                strm << " ";
            }
            strm.width(width);
            strm << (*this)(2, j) << "i"
                 << " ]" << std::endl;
        }
    }

    if (_height == 1 + n) {
        strm << std::endl << "real_t eigen_vecs:" << std::endl;
        for (index_t i = 2; i != n + 2; i++) {
            strm << "[";
            for (index_t j = 1; j != n + 1; j++) {
                strm.width(width);
                strm << (*this)(i, j);
            }
            strm << " ]" << std::endl;
        }
    }
    else if (_height == 2 + n) {
        strm << std::endl << "Complex eigen_vecs:" << std::endl;
        for (index_t j = 1; j != n + 1; j++) {
            if ((*this)(2, j) == 0.0) {
                strm << "Eigenvec " << j << " is real_t." << std::endl;
                for (index_t i = 3; i != 3 + n; i++) {
                    strm << "[";
                    strm.width(width);
                    strm << (*this)(i, j) << " ]" << std::endl;
                }
                strm << std::endl;
            }
            else {
                strm << "Eigenvecs " << j << " and " << j + 1 << " are complex conjugate."
                     << std::endl;
                print_complex_conjugate_eigen_vecs(3, j, n, width, strm);
                j++;
            }
        }
    }
    else if (_height == 2 + 2 * n) {
        strm << std::endl << "Complex left eigen_vecs:" << std::endl;
        for (index_t j = 1; j != n + 1; j++) {
            if ((*this)(2, j) == 0.0) {
                strm << "Eigenvec " << j << " is real_t." << std::endl;
                for (index_t i = 3; i != 3 + n; i++) {
                    strm << "[";
                    strm.width(width);
                    strm << (*this)(i, j) << " ]" << std::endl;
                }
                strm << std::endl;
            }
            else {
                strm << "Eigenvecs " << j << " and " << j + 1 << " are complex conjugate."
                     << std::endl;
                print_complex_conjugate_eigen_vecs(3, j, n, width, strm);
                j++;
            }
        }

        strm << std::endl << "Complex right eigen_vecs:" << std::endl;
        for (index_t j = 1; j != n + 1; j++) {
            if ((*this)(2, j) == 0.0) {
                strm << "Eigenvec " << j << " is real_t." << std::endl;
                for (index_t i = 3 + n; i != 3 + 2 * n; i++) {
                    strm << "[";
                    strm.width(width);
                    strm << (*this)(i, j) << " ]" << std::endl;
                }
                strm << std::endl;
            }
            else {

                strm << "Eigenvecs " << j << " and " << j + 1 << " are complex conjugate."
                     << std::endl;
                print_complex_conjugate_eigen_vecs(3 + n, j, n, width, strm);
                j++;
            }
        }
    }
    _restore_print_format(strm);
}

void mat::print(print_format_t fmt, std::ostream& strm) const {
    strm << std::endl;
    if (_size == 0) {
        strm << " Empty mat" << std::endl;
    }
    else {
        strm << " " << _height << " x " << _width << " mat" << std::endl;
    }
    char prev = strm.fill('-');
    strm << std::setw(80) << '-' << std::endl;
    strm.fill(prev);
    int width = _set_print_format(fmt, strm);

    for (index_t i = 0; i != _height; i++) {
        strm << "[";
        for (index_t j = 0; j != _width; j++) {
            strm.width(width);
            strm << _elements.get()[j * _height + i];
        }
        strm << " ]" << std::endl;
    }
    strm << std::endl << std::endl;
    _restore_print_format(strm);
}

void mat::recount() {
    _size = _width * _height;
}

void mat::_format(const char* fmt) {
    bool                   endOfLine          = false;
    bool                   rowEnded           = false;
    bool                   numericRepPreceded = false;
    char                   ch;
    linear_algebra::size_t m, n;
    unsigned               i, j, idx;

    m = n = 0;
    idx = i = j = 0;
    while (!endOfLine) {
        ch = fmt[idx++];
        if (ch == '\t' || ch == '\r' || ch == '\n' || ch == ' ' || ch == ',' || ch == ';' ||
            ch == '\0') {

            if (numericRepPreceded == true) {
                j++;
                numericRepPreceded = false;
            }

            if (ch == ';' || ch == '\0') {
                rowEnded = true;
                if (i == 0) {
                    n = j;
                }
                else {
                    if (n != j)
                        throw std::runtime_error {"Illegal construction of mat"};
                }
            }

            if (ch == '\0') {
                m         = ++i;
                endOfLine = true;
            }
        }
        else {
            if (rowEnded == true) {
                i++;
                j        = 0;
                rowEnded = false;
            }
            numericRepPreceded = true;
        }
    }
    _height = m;
    _width  = n;
    _size   = m * n;

    if (_size > 0) {
        _elements.reset(new real_t[_size]);
        for (index_t i = 0; i != _size; i++)
            _elements.get()[i] = 0.0;
    }
    else
        _elements = nullptr;

    unsigned ddx;
    char     buf[128];
    endOfLine = false;
    idx = ddx = i = j = 0;
    while (!endOfLine) {
        ch = fmt[idx++];
        if (ch == '\t' || ch == '\r' || ch == '\n' || ch == ' ' || ch == ',' || ch == ';' ||
            ch == '\0') {
            if (ddx > 0) {
                buf[ddx]                   = '\0';
                _elements.get()[j * m + i] = atof(buf);
                j++;
                ddx = 0;
            }
            if (ch == ';' || ch == '\0') {
                i++;
                j = 0;
            }
            if (ch == '\0')
                endOfLine = true;
        }
        else
            buf[ddx++] = ch;
    }
}

bool mat::indices_exceed_dim(const index_t& i, const index_t& j) const {
    if ((i < 1) || (_height < i) || (j < 1) || (_width < j)) {
        return true;
    }
    else {
        return false;
    }
}

bool mat::same_dim(const mat& m) const {
    if ((_width == m._width) && (_height == m._height)) {
        return true;
    }
    else {
        return false;
    }
}

void mat::_replace_rows_with_mat(real_t*                      el,
                                 const linear_algebra::size_t h,
                                 const linear_algebra::size_t w,
                                 const index_t                p,
                                 const linear_algebra::size_t n,
                                 const mat&                   m,
                                 const index_t                q) {

    if (p < 1 || h < (p + n - 1) || q < 1 || m.height() < (q + n - 1)) {
        throw std::runtime_error {"Operation not permitted."};
    }

    for (index_t i = 0; i != n; i++) {
        for (index_t j = 0; j != w; j++) {
            el[j * h + i + p - 1] = m._elements.get()[j * m._height + i + q - 1];
        }
    }
}

bool mat::index_exceeds_rows(const index_t i) const {
    if ((i < 1) || (_height < i)) {
        return true;
    }
    else {
        return false;
    }
}

void mat::_replace_columns_with_mat(real_t*                      el,
                                    const linear_algebra::size_t h,
                                    const linear_algebra::size_t w,
                                    const index_t                p,
                                    const linear_algebra::size_t n,
                                    const mat&                   m,
                                    const index_t                q) {
    if (p < 1 || w < (p + n - 1) || q < 1 || m.width() < (q + n - 1)) {
        throw std::runtime_error {"Operation not permitted."};
    }
    index_t offset1 = h * (p - 1);
    index_t offset2 = h * (q - 1);
    for (index_t k = 0; k != h * n; k++) {
        el[offset1 + k] = m._elements.get()[offset2 + k];
    }
}

bool mat::index_exceeds_cols(const index_t j) const {
    if ((j < 1) || (_width < j)) {
        return true;
    }
    else {
        return false;
    }
}

double mat::_prod_diagonal() const {
    double d = 1.0;
    for (index_t i = 0; i != _height; i++) {
        d *= _elements.get()[i * _height + i];
    }
    return d;
}

void mat::print_complex_conjugate_eigen_vecs(index_t                row,
                                             index_t                col,
                                             linear_algebra::size_t len,
                                             int                    width,
                                             std::ostream&          strm) const {
    for (index_t i = row; i != row + len; i++) {
        char sgn;
        strm << "[";
        strm.width(width);
        strm << (*this)(i, col) << " ";
        if ((*this)(i, col + 1) >= 0) {
            strm << "+";
            sgn = ' ';
        }
        else {
            strm << " ";
            sgn = '+';
        }
        strm.width(width);
        strm << (*this)(i, col + 1) << "i"
             << " ] [";

        strm.width(width);
        strm << (*this)(i, col) << " " << sgn;

        strm.width(width);
        strm << -(*this)(i, col + 1) << "i"
             << " ]" << std::endl;
    }
}

mat linear_algebra::operator+(const double s, const mat& m) {
    return m + s;
}

mat linear_algebra::operator-(const double s, const mat& m) {
    return -m + s;
}

mat linear_algebra::operator*(const double& s, const mat& m) {
    return m * s;
}

vec linear_algebra::diag(const mat& m) {
    return m.diag();
}

mat linear_algebra::transpose(const mat& m) {
    return m.transpose();
}

mat linear_algebra::inv(const mat& m) {
    return m.inv();
}

bool linear_algebra::square(const mat& m) {
    return m.square();
}

bool linear_algebra::symmetric(const mat& m) {
    return m.symmetric();
}

bool linear_algebra::skew_symmetric(const mat& m) {
    return m.skew_symmetric();
}

bool linear_algebra::pseudo_symmetric(const mat& m) {
    return m.pseudo_symmetric();
}

bool linear_algebra::pseudo_skew_symmetric(const mat& m) {
    return m.pseudo_skew_symmetric();
}

double linear_algebra::trace(const mat& m) {
    return m.trace();
}

double linear_algebra::det(const mat& m) {
    return m.det();
}

vec linear_algebra::svd(const mat& m) {
    return m.svd();
}

mat linear_algebra::eigen_values(const mat& m) {
    return m.eigen_values();
}

mat linear_algebra::eigen_vecs(const mat& m) {
    return m.eigen_vecs();
}

mat linear_algebra::left_eigen_vecs(const mat& m) {
    return m.left_eigen_vecs();
}

mat linear_algebra::left_right_eigen_vecs(const mat& m) {
    return m.left_right_eigen_vecs();
}

void linear_algebra::print_eigen_value_vec(const mat& evm, print_format_t fmt, std::ostream& strm) {
    evm.print_eigen_value_vec(fmt, strm);
}

bool linear_algebra::positive_definite(const mat& m) {
    return m.positive_definite();
}

bool linear_algebra::positive_semi_definite(const mat& m) {
    return m.positive_semi_definite();
}

bool linear_algebra::pseudo_positive_semi_definite(const mat& m) {
    return m.pseudo_positive_semi_definite();
}

mat mat::identity(const linear_algebra::size_t m, const linear_algebra::size_t n) {
    mat                          mat(m, n);
    const linear_algebra::size_t smaller = (m > n) ? n : m;
    real_t*                      el      = mat.elements();

    for (index_t i = 0; i != smaller; i++) {
        el[m * i + i] = 1.0;
    }

    return mat;
}

mat mat::diag(const vec& v, const int k) {
    index_t                offset = abs(k);
    linear_algebra::size_t sz     = v.dim() + offset;
    mat                    m(sz, sz);
    real_t*                mel = m.elements();
    real_t*                vel = v.elements();
    if (k > 0) {
        for (index_t i = 0; i != v.dim(); i++) {
            mel[(i + offset) * sz + i] = vel[i];
        }
    }
    else if (k < 0) {
        for (index_t i = 0; i != v.dim(); i++) {
            mel[i * sz + offset + i] = vel[i];
        }
    }
    else {
        for (index_t i = 0; i != v.dim(); i++) {
            mel[i * sz + i] = vel[i];
        }
    }
    return m;
}


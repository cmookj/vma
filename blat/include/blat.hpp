//===------------------------------------------------------------*- C++ -*-===//
//
//  blat.hpp
//  Basic Linear Algebra Toolkit
//
//  Created by Changmook Chun on 2022/11/25.
//  Copyright © 2022 Teaeles.com. All rights reserved.
//
//===----------------------------------------------------------------------===//
#ifndef __LINEAR_ALGEBRA_H_
#define __LINEAR_ALGEBRA_H_

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>  // for memcpy
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "../internal/concurrency.hpp"

#if defined(__APPLE__)
#if defined(USE_OPENBLAS)
#include <Accelerate/Accelerate.h>
using integer_t = __CLPK_integer;
using real_t    = __CLPK_doublereal;
#else
#define ACCELERATE_NEW_LAPACK
// #define ACCELERATE_LAPACK_ILP64
#include <Accelerate/Accelerate.h>
using integer_t = __LAPACK_int;
using real_t    = double;
#endif
#endif

#if defined(_WIN32) || defined(_WIN64)
#include <cblas.h>
#include <clapack.h>
#include <f2c.h>
#undef abs

#include <cstdint>
#include <ctime>
using integer_t = integer;
using real_t    = doublereal;
#endif

#if defined(__linux) || defined(__linux__)
#include <cblas.h>
#include <clapack.h>
#include <f2c.h>
#undef abs
#undef min
#undef max

using integer_t = integer;
using real_t    = doublereal;

// clang-format off
extern "C" {
extern void
dgetrf_ (const int*, const int*, double*, const int*, int*, int*);

extern int
dgesdd_ (char*, integer*, integer*, doublereal*, integer*, doublereal*, 
         doublereal*, integer*, doublereal*, integer*, doublereal*, integer*, 
         integer*, integer*);

extern int
dgetri_ (integer*, doublereal*, integer*, integer*, doublereal*, integer*, 
         integer*);

extern int
dsyev_ (char*, char*, integer*, doublereal*, integer*, doublereal*, doublereal*, 
        integer*, integer*);

extern int
dgeev_ (char*, char*, integer*, doublereal*, integer*, doublereal*, doublereal*, 
        doublereal*, integer*, doublereal*, integer*, doublereal*, integer*, 
        integer*);
}
// clang-format on

#endif

namespace gpw::blat {

// Constants
const double TOL = 2.2204e-16;
const double EPS = std::numeric_limits<double>::epsilon();

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

using std::size_t;

// =============================================================================
//                                                                  Enumerations
// =============================================================================
#pragma mark - Enumerations

// Format specification to generate string representation
enum class output_fmt { sht, nml, ext, sci, scx };

// Eigenvalue problem specification to control the problem setup and solution
enum class eigen { val, vec, lvec, rvec };

using complex_t = std::complex<double>;

// =============================================================================
//                                                                     Utilities
// =============================================================================
#pragma mark - Utilities
int
set_format (std::stringstream& strm, output_fmt fmt = output_fmt::nml);

// =============================================================================
//                                                     C L A S S  :  V E C T O R
// =============================================================================
#pragma mark - Vector

/**
 @brief Vector
 */
template <size_t DIM, typename T = double> class vec {
protected:
  std::vector<T> _elem;

public:
  // Default constructor and destructor
  vec()
      : _elem{std::vector<T> (DIM, 0.)} {}
  virtual ~vec() = default;

  // Copy constructor & assignment
  vec (const vec&) = default;
  vec&
  operator= (const vec&) = default;

  // Move constructor & assignment
  vec (vec&&) noexcept = default;
  vec&
  operator= (vec&&) noexcept = default;

  // Other constructors
  vec (std::vector<T>&& elm)
      : _elem{std::move (elm)} {}

  vec (const T v)
      : _elem{std::vector<T> (DIM, v)} {}

  vec (const double* vp)
      : _elem{std::vector<T> (DIM, 0.)} {
    std::for_each (_elem.begin(), _elem.end(), [&vp] (T& e) { e = *(vp++); });
  }

  vec (const std::initializer_list<T>& il)
      : _elem{il} {}

  // Access methods
  std::vector<T>&
  elem () {
    return _elem;
  }
  const std::vector<T>&
  elem () const {
    return _elem;
  }

  using iterator = T*;
  T*
  begin () {
    return &_elem[0];
  }
  T*
  end () {
    return &_elem[0] + DIM;
  }
  constexpr const T*
  cbegin () const noexcept {
    return &_elem[0];
  }
  constexpr const T*
  cend () const noexcept {
    return &_elem[0] + DIM;
  }

  constexpr T*
  data () noexcept {
    return _elem.data();
  }
  constexpr const T*
  data () const noexcept {
    return _elem.data();
  }

  // Subscript operators
  T&
  operator[] (const size_t n) {
    return const_cast<T&> (static_cast<const vec&> (*this)[n]);
  }
  T&
  operator() (const size_t n) {
    return const_cast<T&> (static_cast<const vec&> (*this) (n));
  }
  const T&
  operator[] (const size_t n) const {
    return _elem[n];
  }
  const T&
  operator() (const size_t n) const {
    return _elem[n - 1];
  }

  // Dimension
  size_t
  dim () const {
    return _elem.size();
  }

  // Equality
  bool
  operator== (const vec& rhs) const {
    // Calculate difference
    std::vector<T> diff (DIM);
    std::transform (cbegin(), cend(), rhs.cbegin(), diff.begin(), std::minus<>{});

    if constexpr (std::is_floating_point<T>::value) {
      const T eps              = std::numeric_limits<T>::epsilon();
      auto    greater_than_eps = [&eps] (const T& v) { return std::abs (v) > eps; };
      return std::find_if (diff.cbegin(), diff.cend(), greater_than_eps) == diff.cend();
    } else if constexpr (std::is_same_v<T, complex_t>) {
      const double eps              = std::numeric_limits<double>::epsilon();
      auto         greater_than_eps = [&eps] (const T& v) { return std::abs (v) > eps; };
      return std::find_if (diff.cbegin(), diff.cend(), greater_than_eps) == diff.end();
    } else return _elem == rhs._elem;
  }

  bool
  operator!= (const vec& rhs) const {
    return !(_elem == rhs._elem);
  }

  // Binary arithmetic operators
  vec&
  operator+= (const vec& rhs) {
    std::transform (_elem.cbegin(), _elem.cend(), rhs.cbegin(), _elem.begin(), std::plus<>{});
    return *this;
  }

  vec&
  operator-= (const vec& rhs) {
    std::transform (_elem.cbegin(), _elem.cend(), rhs.cbegin(), _elem.begin(), std::minus<>{});
    return *this;
  }

  vec&
  operator*= (const double& s) {
    std::transform (_elem.cbegin(), _elem.cend(), _elem.begin(), [s] (const auto& v) {
      return v * s;
    });
    return *this;
  }

  vec&
  operator/= (const double& s) {
    std::transform (_elem.cbegin(), _elem.cend(), _elem.begin(), [s] (const auto& v) {
      return v / s;
    });
    return *this;
  }
};

/**
 @brief Represents a vector as a string
 */
template <size_t DIM, typename T = double>
std::string
to_string (const vec<DIM, T>& v, output_fmt fmt = output_fmt::nml) {
  std::stringstream strm{};

  int  width = set_format (strm, fmt);
  auto print = [&strm, &width] (const T& v) {
    strm.width (width);
    strm << v << ", ";
  };

  strm << "[ ";
  std::for_each (v.cbegin(), v.cend(), print);
  strm << "]";

  return strm.str();
}

// -----------------------------------------------------------------------------
//                                                       Special Vector Creation
// -----------------------------------------------------------------------------
/**
 @brief Creates a vector with random numbers in uniform distribution
 */
template <size_t DIM>
vec<DIM>
rand () {
  std::vector<double> elm (DIM, 0.);

  std::random_device               rdu;
  std::mt19937                     genu (rdu());
  std::uniform_real_distribution<> ud (0, 1);
  for (size_t i = 0; i < DIM; ++i) {
    elm[i] = ud (genu);
  }

  return vec<DIM>{std::move (elm)};
}

/**
 @brief Creates a vector with random numbers in normal distribution
 */
template <size_t DIM>
vec<DIM>
randn () {
  std::vector<double> elm (DIM, 0.);

  std::random_device         rdn;
  std::mt19937               genn (rdn());
  std::normal_distribution<> nd (0, 1);
  for (size_t i = 0; i < DIM; ++i) {
    elm[i] = nd (genn);
  }

  return vec<DIM>{std::move (elm)};
}

/**
 @brief Creates a complex conjugate of a vector

 @details Note that when the input argument is a real vector, this function
 creates a complex vector.
 */
template <size_t DIM, typename T>
vec<DIM, complex_t>
conj (const vec<DIM, T>& v) {
  std::vector<complex_t> elm (DIM);

  auto conj = [] (const auto& c) { return std::conj (c); };
  std::transform (v.cbegin(), v.cend(), elm.begin(), conj);

  return vec<DIM, complex_t>{std::move (elm)};
}

/**
 @brief Creates a new complex vector from two old-fashioned arrays of doubles

 @details This function assumes the arrays are in row major order.
 */
template <size_t DIM>
vec<DIM, complex_t>
cvec (const double* re, const double* im) {
  std::vector<complex_t> elm (DIM);

  for (size_t i = 0; i < DIM; ++i)
    elm[i] = complex_t{re[i], im[i]};

  return vec<DIM, complex_t>{std::move (elm)};
}

/**
 @brief Creates a new complex vector from two real vectors
 */
template <size_t DIM>
vec<DIM, complex_t>
cvec (const vec<DIM>& re, const vec<DIM>& im) {
  std::vector<complex_t> elm (DIM);

  for (size_t i = 0; i < DIM; ++i)
    elm[i] = complex_t{re.elem()[i], im.elem()[i]};

  return vec<DIM, complex_t>{std::move (elm)};
}

/**
 @brief Creates a new complex vector from a real vector

 @details The imaginary parts are all set to 0.
 */
template <size_t DIM>
vec<DIM, complex_t>
cvec (const vec<DIM>& re) {
  std::vector<complex_t> elm (DIM);

  for (size_t i = 0; i < DIM; ++i)
    elm[i] = complex_t{re.elem()[i], 0.};

  return vec<DIM, complex_t>{std::move (elm)};
}

// -----------------------------------------------------------------------------
//                                                             Vector Operations
// -----------------------------------------------------------------------------
/**
 @brief Calculates inner product of two vectors
 */
template <size_t DIM>
double
inner (const vec<DIM>& a, const vec<DIM>& b) {
  return std::transform_reduce (a.cbegin(), a.cend(), b.cbegin(), 0.);
}

/**
 @brief Calculates p-norm of a vector

 @details The vector norm |x|_p for p = 1, 2, ... is defined as
     |x|_p = (sum |x_i|^p)^(1/p).
 */
template <size_t DIM>
double
norm (const vec<DIM>& v, const unsigned p = 2) {
  auto powered_fold = [p] (const double a, const double b) {
    return a + std::pow (std::abs (b), double (p));
  };
  return std::pow (std::accumulate (v.cbegin(), v.cend(), 0., powered_fold), 1. / double (p));
}

/**
 @brief Calculates infinite vector norm

 @details The special case |x|_inf is defined as
     |x|_inf = max |x_i|
 */
template <size_t DIM>
double
norm_inf (const vec<DIM>& v) {
  auto index = std::max_element (v.cbegin(), v.cend(), [] (double a, double b) {
    return std::abs (a) < std::abs (b);
  });

  return *index;
}

/**
 @brief Normalizes a vector
 */
template <size_t DIM>
vec<DIM>
normalize (const vec<DIM>& v, const unsigned p = 2) {
  double n = norm (v, p);
  if (n < TOL) n = 1.0;

  return vec<DIM>{v / n};
}

template <size_t DIM>
double
abs (const vec<DIM>& v) {
  return norm (v);
}

/**
 @brief Calculates the distance between two vectors
 */
template <size_t DIM>
double
dist (const vec<DIM>& a, const vec<DIM>& b) {
  return norm (a - b);
}

/**
 @brief Adds two vectors
 */
template <typename T, size_t DIM>
vec<DIM, T>
operator+ (const vec<DIM, T>& a, const vec<DIM, T>& b) {
  vec result{a};
  result += b;
  return result;
}

/**
 @brief Subtracts a vector from another
 */
template <typename T, size_t DIM>
vec<DIM, T>
operator- (const vec<DIM, T>& a, const vec<DIM, T>& b) {
  vec result{a};
  result -= b;
  return result;
}

/**
 @brief Negate a vector
 */
template <typename T, size_t DIM>
vec<DIM, T>
operator- (const vec<DIM, T>& a) {
  vec result{a};
  result *= -1.;
  return result;
}

/**
 @brief Multiplies a scalar to a vector
 */
template <typename T, size_t DIM>
vec<DIM, T>
operator* (const vec<DIM, T>& a, const double s) {
  vec result{a};
  result *= s;
  return result;
}

/**
 @brief Multiplies a scalar to a vector
 */
template <typename T, size_t DIM>
vec<DIM, T>
operator* (const double s, const vec<DIM, T>& a) {
  vec result{a};
  result *= s;
  return result;
}

/**
 @brief Divides a vector by a scalar
 */
template <typename T, size_t DIM>
vec<DIM, T>
operator/ (const vec<DIM, T>& a, const double s) {
  vec result{a};
  result /= s;
  return result;
}

/**
 @brief Extracts real part of a complex vector as a new real vector
 */
template <size_t DIM>
vec<DIM>
real (const vec<DIM, complex_t>& v) {
  std::vector<double> elm (DIM);

  auto re = [] (const complex_t& c) { return c.real(); };
  std::transform (v.cbegin(), v.cend(), elm.begin(), re);

  return vec<DIM>{std::move (elm)};
}

/**
 @brief Extracts imaginary part of a complex vector as a new real vector
 */
template <size_t DIM>
vec<DIM>
imag (const vec<DIM, complex_t>& v) {
  std::vector<double> elm (DIM);

  auto im = [] (const complex_t& c) { return c.imag(); };
  std::transform (v.cbegin(), v.cend(), elm.begin(), im);

  return vec<DIM>{std::move (elm)};
}

/**
 @brief Determines whether two vectors are close to each other
 */
template <size_t DIM, typename T>
bool
similar (const vec<DIM, T>& a, const vec<DIM, T>& b, double tol = TOL) {
  auto diff       = a - b;
  auto abs_square = [] (const auto& v) { return std::abs (v * v); };

  std::transform (diff.cbegin(), diff.cend(), diff.begin(), abs_square);
  double sum = std::accumulate (diff.cbegin(), diff.cend(), 0.);

  if (std::sqrt (sum / static_cast<double> (DIM)) < tol) return true;
  else return false;
}

/**
 @brief Determines whether two real vectors have the same direction
 */
template <size_t DIM>
bool
collinear (const vec<DIM>& a, const vec<DIM>& b, double tol = TOL) {
  std::vector<double> ratio (DIM);
  std::transform (a.cbegin(), a.cend(), b.cbegin(), ratio.begin(), std::divides<>{});

  // Originally, the following expression in if statement should be either
  //   if (std::adjacent_find(ratio.begin(), ratio.end(),
  //   std::not_equal_to<>())
  //       == ratio.end())
  // or,
  //   if (std::equal(ratio.begin() + 1, ratio.end(), ratio.begin()))
  //
  // But, to ignore 'nan' which is the result of 0./0., we need a special
  // predicate.

  auto nan_skipping_not_equal_to = [&tol] (const double& a, const double& b) {
    if (std::isnan (a) || std::isnan (b)) return false;
    else return std::abs (a - b) > tol;
  };

  return std::adjacent_find (ratio.begin(), ratio.end(), nan_skipping_not_equal_to) == ratio.end();
}

/**
 @brief Determines whether two compelx vectors have the same direction
 */
template <size_t DIM>
bool
collinear (const vec<DIM, complex_t>& a, const vec<DIM, complex_t>& b, double tol = TOL) {
  std::vector<complex_t> ratio (DIM);
  std::transform (a.cbegin(), a.cend(), b.cbegin(), ratio.begin(), std::divides<>{});

  // Originally, the following expression in if statement should be either
  //   if (std::adjacent_find(ratio.begin(), ratio.end(),
  //   std::not_equal_to<>())
  //       == ratio.end())
  // or,
  //   if (std::equal(ratio.begin() + 1, ratio.end(), ratio.begin()))
  //
  // But, to ignore 'nan' which is the result of 0./0., we need a special
  // predicate.

  auto nan_skipping_not_equal_to = [&tol] (const complex_t& a, const complex_t& b) {
    if ((std::isnan (a.real()) && std::isnan (b.real())) ||
        (std::isnan (a.imag()) && std::isnan (b.imag())))
      return false;
    else return std::abs (a - b) > tol;
  };

  return std::adjacent_find (ratio.begin(), ratio.end(), nan_skipping_not_equal_to) == ratio.end();
}

// Comparison (by the norm of the vectors)
template <size_t DIM>
bool
operator< (const vec<DIM>& l, const vec<DIM>& r) {
  return norm (l) < norm (r);
}

// =============================================================================
//                                                     C L A S S  :  M A T R I X
// =============================================================================
#pragma mark - Matrix

/**
 @brief Matrix

 @details This class stores the elements of a matrix in column major order,
 for easier and efficient integration with LAPACK.
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T = double> class mat {
protected:
  static constexpr size_t SZ = DIM_ROWS * DIM_COLS;
  std::vector<T>          _elem;

public:
  // Default constructor and destructor
  mat()
      : _elem{std::vector<T> (SZ, 0.)} {}
  virtual ~mat() = default;

  // Copy constructor & assignment
  mat (const mat&) = default;
  mat&
  operator= (const mat&) = default;

  // Move constructor & assignment
  mat (mat&&) noexcept = default;
  mat&
  operator= (mat&&) noexcept = default;

  // Other constructors
  mat (std::vector<T>&& elm)
      : _elem{std::move (elm)} {}

  mat (const T v)
      : _elem{std::vector<T> (SZ, 0.)} {
    std::for_each (_elem.begin(), _elem.end(), [&v] (T& e) { e = v; });
  }

  /**
   @brief Constructs a new matrix from an old-fashioned array of doubles

   @details This function assumes the array is in row major order.
   */
  mat (const double* vp)
      : _elem{std::vector<T> (SZ)} {
    size_t idx{0};
    for (size_t j = 0; j < DIM_COLS; ++j)
      for (size_t i = 0; i < DIM_ROWS; ++i)
        _elem[j * DIM_ROWS + i] = vp[idx++];
  }

  /**
   @brief Constructs a new matrix from an initializer_list of Ts

   @details This function assumes the list is in row major order.
   */
  mat (const std::initializer_list<std::initializer_list<T>>& il)
      : _elem{std::vector<T> (SZ, 0.)} {
    size_t          idx{0};
    size_t          i{0};
    size_t          count{SZ};
    std::vector<T>& elm{_elem};

    auto extract_row = [&i, &idx, &count, &elm] (const std::initializer_list<T>& row) {
      for (auto it = row.begin(); it < row.end(); ++it) {
        elm[idx] = *it;
        idx      = (idx + DIM_ROWS);
        if (idx >= count) idx = ++i;
      }
    };
    std::for_each (il.begin(), il.end(), extract_row);
  }

  // Access methods
  std::vector<T>&
  elem () {
    return _elem;
  }
  const std::vector<T>&
  elem () const {
    return _elem;
  }

  using iterator = T*;
  T*
  begin () {
    return &_elem[0];
  }
  T*
  end () {
    return &_elem[0] + SZ;
  }
  constexpr const T*
  cbegin () const noexcept {
    return &_elem[0];
  }
  constexpr const T*
  cend () const noexcept {
    return &_elem[0] + SZ;
  }

  constexpr T*
  data () noexcept {
    return _elem.data();
  }
  constexpr const T*
  data () const noexcept {
    return _elem.data();
  }

  /**
   @brief Index operator

   @details Note that the elements of the matrix are in column major order.
   */
  const T&
  operator() (const size_t i, const size_t j) const {
    return _elem[(i - 1) + (j - 1) * DIM_ROWS];
  }

  T&
  operator() (const size_t i, const size_t j) {
    return const_cast<T&> (static_cast<const mat&> (*this) (i, j));
  }

  // Dimension
  std::pair<size_t, size_t>
  dim () const {
    return std::make_pair (DIM_ROWS, DIM_COLS);
  }

  size_t
  count_rows () const {
    return DIM_ROWS;
  }
  size_t
  count_cols () const {
    return DIM_COLS;
  }

  // Extraction of a column or a row as a vector
  vec<DIM_ROWS, T>
  col (const size_t j) const {
    std::vector<T> el (DIM_ROWS);

    auto head = _elem.cbegin();
    if ((1 <= j) && (j <= DIM_COLS)) std::copy_n (head + (j - 1) * DIM_ROWS, DIM_ROWS, el.begin());

    return vec<DIM_ROWS, T>{std::move (el)};
  }

  vec<DIM_COLS, T>
  row (const size_t i) const {
    std::vector<T> el (DIM_COLS);

    if ((1 <= i) && (i <= DIM_ROWS)) {
      auto it = _elem.cbegin() + (i - 1);
      for (size_t j = 0; j < DIM_COLS; ++j) {
        std::copy_n (it, 1, el.begin() + j);
        it += DIM_ROWS;
      }
    }

    return vec<DIM_COLS, T>{std::move (el)};
  }

  // Replaces a column
  void
  set_col (const size_t j, const vec<DIM_ROWS, T>& v) {
    for (size_t i = 0; i < DIM_ROWS; ++i)
      _elem[(j - 1) * DIM_ROWS + i] = v (i + 1);
  }

  // Replaces a row
  void
  set_row (const size_t i, const vec<DIM_COLS, T>& v) {
    for (size_t j = 0; j < DIM_COLS; ++j)
      _elem[j * DIM_ROWS + (i - 1)] = v (j + 1);
  }

  // Equality
  bool
  operator== (const mat& rhs) const {
    // Calculate difference
    std::vector<T> diff (SZ);
    std::transform (cbegin(), cend(), rhs.cbegin(), diff.begin(), std::minus<>{});

    if constexpr (std::is_floating_point<T>::value) {
      const T eps              = std::numeric_limits<T>::epsilon();
      auto    greater_than_eps = [&eps] (const T& v) { return std::abs (v) > eps; };
      return std::find_if (diff.cbegin(), diff.cend(), greater_than_eps) == diff.end();
    } else if constexpr (std::is_same_v<T, complex_t>) {
      const double eps              = std::numeric_limits<double>::epsilon();
      auto         greater_than_eps = [&eps] (const T& v) { return std::abs (v) > eps; };
      return std::find_if (diff.cbegin(), diff.cend(), greater_than_eps) == diff.end();
    } else return _elem == rhs._elem;
  }

  bool
  operator!= (const mat& rhs) const {
    return !(*this == rhs);
  }

  // Binary arithmetic operators
  mat&
  operator+= (const mat& rhs) {
    std::transform (_elem.cbegin(), _elem.cend(), rhs.cbegin(), _elem.begin(), std::plus<>{});
    return *this;
  }

  mat&
  operator+= (const T s) {
    std::transform (_elem.cbegin(), _elem.cend(), _elem.begin(), [s] (const auto& e) {
      return e + s;
    });
    return *this;
  }

  mat&
  operator-= (const mat& rhs) {
    std::transform (_elem.cbegin(), _elem.cend(), rhs.cbegin(), _elem.begin(), std::minus<>{});
    return *this;
  }

  mat&
  operator*= (const T& s) {
    std::transform (_elem.cbegin(), _elem.cend(), _elem.begin(), [s] (const auto& v) {
      return v * s;
    });
    return *this;
  }

  mat&
  operator/= (const T& s) {
    std::transform (_elem.cbegin(), _elem.cend(), _elem.begin(), [s] (const auto& v) {
      return v / s;
    });
    return *this;
  }

  // Matrix multiplication
  mat&
  operator*= (const mat<DIM_COLS, DIM_COLS, T>& m) {
    std::vector<T> elm (SZ);

    if constexpr (std::is_same_v<T, double>)
      cblas_dgemm (
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          DIM_ROWS,
          DIM_COLS,
          DIM_COLS,
          1.,
          _elem.data(),
          DIM_ROWS,
          m.data(),
          DIM_COLS,
          0.,
          elm.data(),
          DIM_ROWS
      );
    else if constexpr (std::is_same_v<T, complex_t>)
      cblas_zgemm (
          CblasColMajor,
          CblasNoTrans,
          CblasNoTrans,
          DIM_ROWS,
          DIM_COLS,
          DIM_COLS,
          std::complex<double>{1., 0.},
          _elem.data(),
          DIM_ROWS,
          m.data(),
          DIM_COLS,
          std::complex<double>{0., 0.},
          elm.data(),
          DIM_ROWS
      );

    _elem = std::move (elm);
    return *this;
  }
};

/**
 @brief Represents a matrix as a string
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
std::string
to_string (const mat<DIM_ROWS, DIM_COLS, T>& M, output_fmt fmt = output_fmt::nml) {
  std::stringstream strm{};

  int width = set_format (strm, fmt);
  for (size_t i = 0; i < DIM_ROWS; ++i) {
    strm << "[ ";
    for (size_t j = 0; j < DIM_COLS; ++j) {
      strm.width (width);
      strm << M.elem()[j * DIM_ROWS + i] << ", ";
    }
    strm << "]\n";
  }

  return strm.str();
}

/**
 @brief Extracts real part of a complex matrix as a new real matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS>
real (const mat<DIM_ROWS, DIM_COLS, complex_t>& M) {
  constexpr size_t    SZ{DIM_ROWS * DIM_COLS};
  std::vector<double> elm (SZ);

  auto re = [] (const complex_t& c) { return c.real(); };
  std::transform (M.cbegin(), M.cend(), elm.begin(), re);

  return mat<DIM_ROWS, DIM_COLS>{std::move (elm)};
}

/**
 @brief Extracts imaginary part of a complex matrix as a new real matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS>
imag (const mat<DIM_ROWS, DIM_COLS, complex_t>& M) {
  constexpr size_t    SZ{DIM_ROWS * DIM_COLS};
  std::vector<double> elm (SZ);

  auto im = [] (const complex_t& c) { return c.imag(); };
  std::transform (M.cbegin(), M.cend(), elm.begin(), im);

  return mat<DIM_ROWS, DIM_COLS>{std::move (elm)};
}

// -----------------------------------------------------------------------------
//                                                       Special Matrix Creation
// -----------------------------------------------------------------------------
/**
 @brief Creates a DIM x DIM identity matrix
 */
template <size_t DIM>
mat<DIM, DIM>
identity () {
  mat<DIM, DIM> I{};

  for (size_t i = 0; i < DIM; ++i)
    I.elem()[i * DIM + i] = 1.;

  return I;
}

/**
 @brief Creates a square matrix with only diagonal elements
 */
template <size_t DIM>
mat<DIM, DIM>
diag (std::vector<double>& val) {
  std::vector<double> elm (DIM * DIM, 0.);

  for (size_t i = 0; i < DIM; ++i)
    elm[i * DIM + i] = val[i];

  return mat<DIM, DIM>{std::move (elm)};
}

/**
 @brief Creates a square matrix with only diagonal elements
 */
template <size_t DIM>
mat<DIM, DIM>
diag (std::initializer_list<double>& il) {
  std::vector<double> elm (DIM * DIM, 0.);

  std::vector<double> val{il};

  for (size_t i = 0; i < val.size(); ++i)
    elm[i * DIM + i] = val[i];

  return mat<DIM, DIM>{DIM, DIM, std::move (elm)};
}

/**
 @brief Creates a square matrix with only diagonal elements
 */
template <size_t DIM>
mat<DIM, DIM>
diag (double* val) {
  std::vector<double> elm (DIM * DIM, 0.);

  for (size_t i = 0; i < DIM; ++i)
    elm[i * DIM + i] = val[i];

  return mat<DIM, DIM>{std::move (elm)};
}

/**
 @brief Creates a matrix with only diagonal elements
 */
template <size_t DIM_ROWS, size_t DIM_COLS, size_t DIM>
mat<DIM_ROWS, DIM_COLS>
diag (const vec<DIM>& v) {
  constexpr size_t    SZ{DIM_ROWS * DIM_COLS};
  std::vector<double> elm (SZ, 0.);

  for (size_t i = 0; i < DIM; ++i)
    elm[i * DIM_ROWS + i] = v.elem()[i];

  return mat<DIM_ROWS, DIM_COLS>{std::move (elm)};
}

/**
 @brief Creates a matrix with random numbers in uniform distribution
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS>
rand () {
  std::vector<double> elm (DIM_ROWS * DIM_COLS);

  std::random_device               rdu;
  std::mt19937                     genu (rdu());
  std::uniform_real_distribution<> ud (0, 1);
  for (size_t i = 0; i < DIM_ROWS * DIM_COLS; ++i) {
    elm[i] = ud (genu);
  }

  return mat<DIM_ROWS, DIM_COLS>{std::move (elm)};
}

/**
 @brief Creates a matrix with random numbers in normal distribution
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS>
randn () {
  std::vector<double> elm (DIM_ROWS * DIM_COLS);

  std::random_device         rdn;
  std::mt19937               genn (rdn());
  std::normal_distribution<> nd (0, 1);
  for (size_t i = 0; i < DIM_ROWS * DIM_COLS; ++i) {
    elm[i] = nd (genn);
  }

  return mat<DIM_ROWS, DIM_COLS>{std::move (elm)};
}

/**
 @brief Creates a complex conjugate of a matrix

 @details Note that when the input argument is a real matrix, this function
 creates a complex matrix.
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, complex_t>
conj (const mat<DIM_ROWS, DIM_COLS, T>& M) {
  std::vector<complex_t> elm (DIM_ROWS * DIM_COLS);

  auto conj = [] (const auto& c) { return std::conj (c); };
  std::transform (M.cbegin(), M.cend(), elm.begin(), conj);

  return mat<DIM_ROWS, DIM_COLS, complex_t>{std::move (elm)};
}

/**
 @brief Creates a new complex matrix from two old-fashioned arrays of doubles

 @details This function assumes the arrays are in row major order.
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
mat<DIM_ROWS, DIM_COLS, complex_t>
cmat (const double* re, const double* im) {
  std::vector<complex_t> elm (DIM_ROWS * DIM_COLS);

  size_t idx{0};
  for (size_t i = 0; i < DIM_ROWS; ++i)
    for (size_t j = 0; j < DIM_COLS; ++j) {
      elm[j * DIM_ROWS + i] = complex_t{re[idx], im[idx]};
      idx++;
    }

  return mat<DIM_ROWS, DIM_COLS, complex_t>{std::move (elm)};
}

// -----------------------------------------------------------------------------
//                                                             Matrix Operations
// -----------------------------------------------------------------------------
/**
 @brief Transposes a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_COLS, DIM_ROWS, T>
transpose (const mat<DIM_ROWS, DIM_COLS, T>& m) {
  mat<DIM_COLS, DIM_ROWS, T> t{};

  using namespace gpw::concurrency;

  thread_pool tp;

  // Transpose each column independently, in parallel.
  for (int j = 0; j < DIM_COLS; ++j) {
    tp.queue_job ([j, &m, &t] () {
      for (size_t i = 0; i < DIM_ROWS; ++i)
        t.elem()[i * DIM_COLS + j] = m.elem()[j * DIM_ROWS + i];
    });
  }

  tp.start();

  while (tp.busy())
    ;

  tp.stop();

  return t;
}

/**
 @brief Negates a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator- (const mat<DIM_ROWS, DIM_COLS, T>& m) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result *= -1.;
  return result;
}

/**
 @brief Adds two matrices
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator+ (const mat<DIM_ROWS, DIM_COLS, T>& a, const mat<DIM_ROWS, DIM_COLS, T>& b) {
  auto result{a};
  result += b;
  return result;
}

/**
 @brief Adds a scalar to a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator+ (const mat<DIM_ROWS, DIM_COLS, T>& m, const T s) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result += s;
  return result;
}

template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator+ (const T s, const mat<DIM_ROWS, DIM_COLS, T>& m) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result += s;
  return result;
}

/**
 @brief Subtracts a matrix from another
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator- (const mat<DIM_ROWS, DIM_COLS, T>& a, const mat<DIM_ROWS, DIM_COLS, T>& b) {
  auto result{a};
  result -= b;
  return result;
}

/**
 @brief Subtracts a scalar from a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator- (const mat<DIM_ROWS, DIM_COLS, T>& m, const T s) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result -= s;
  return result;
}

/**
 @breif Subtracts a matrix from a scalar
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator- (const T s, const mat<DIM_ROWS, DIM_COLS, T>& m) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result *= -1.;
  result += s;
  return result;
}

/**
 @brief Multiplies a scalar to a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator* (const mat<DIM_ROWS, DIM_COLS, T>& m, const T s) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result *= s;
  return result;
}

/**
 @brief Multiplies a scalar to a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator* (const T s, const mat<DIM_ROWS, DIM_COLS, T>& m) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result *= s;
  return result;
}

/**
 @brief Divides a matrix with a scalar
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator/ (const mat<DIM_ROWS, DIM_COLS, T>& m, const T s) {
  mat<DIM_ROWS, DIM_COLS, T> result{m};
  result /= s;
  return result;
}

/**
 @brief Multiplies two matrices using BLAS
 */
template <size_t DIM_ROWS, size_t DIM, size_t DIM_COLS, typename T>
mat<DIM_ROWS, DIM_COLS, T>
operator* (const mat<DIM_ROWS, DIM, T>& m1, const mat<DIM, DIM_COLS, T>& m2) {
  std::vector<T> elm (DIM_ROWS * DIM_COLS);

  if constexpr (std::is_same_v<T, double>)
    cblas_dgemm (
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        DIM_ROWS,
        DIM_COLS,
        DIM,
        1,
        m1.data(),
        DIM_ROWS,
        m2.data(),
        DIM,
        0.,
        elm.data(),
        DIM_ROWS
    );
  else if constexpr (std::is_same_v<T, complex_t>) {
    std::complex<double> alpha{1., 0.};
    std::complex<double> beta{0., 0.};
    cblas_zgemm (
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        DIM_ROWS,
        DIM_COLS,
        DIM,
        &alpha,
        m1.data(),
        DIM_ROWS,
        m2.data(),
        DIM,
        &beta,
        elm.data(),
        DIM_ROWS
    );
  }

  return mat<DIM_ROWS, DIM_COLS, T>{std::move (elm)};
}

/**
 @brief Post-multiplies a vector to a matrix
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
vec<DIM_COLS, T>
operator* (const mat<DIM_ROWS, DIM_COLS, T>& m, const vec<DIM_COLS, T>& v) {
  std::vector<T> elm (DIM_COLS);

  if constexpr (std::is_same_v<T, double>)
    cblas_dgemm (
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        DIM_ROWS,
        1,
        DIM_COLS,
        1,
        m.data(),
        DIM_ROWS,
        v.data(),
        DIM_COLS,
        0.,
        elm.data(),
        DIM_COLS
    );

  else if constexpr (std::is_same_v<T, complex_t>) {
    std::complex<double> alpha{1., 0};
    std::complex<double> beta{0., 0.};
    cblas_zgemm (
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        DIM_ROWS,
        1,
        DIM_COLS,
        &alpha,
        m.data(),
        DIM_ROWS,
        v.data(),
        DIM_COLS,
        &beta,
        elm.data(),
        DIM_COLS
    );
  }

  return vec<DIM_COLS, T>{std::move (elm)};
}

/**
 @brief Pre-multiplies a vector to a matrix

 @details This function implicitly assumes that the first argument is a row
 vector.
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
vec<DIM_COLS, T>
operator* (const vec<DIM_ROWS, T>& v, const mat<DIM_ROWS, DIM_COLS, T>& m) {
  std::vector<T> elm (DIM_COLS);

  if constexpr (std::is_same_v<T, double>)
    cblas_dgemm (
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        1,
        DIM_ROWS,
        DIM_COLS,
        1,
        v.data(),
        1,
        m.data(),
        DIM_ROWS,
        0.,
        elm.data(),
        1
    );
  else if constexpr (std::is_same_v<T, complex_t>) {
    std::complex<double> alpha{1., 0.};
    std::complex<double> beta{0., 0.};
    cblas_zgemm (
        CblasColMajor,
        CblasNoTrans,
        CblasNoTrans,
        1,
        DIM_ROWS,
        DIM_COLS,
        &alpha,
        v.data(),
        1,
        m.data(),
        DIM_ROWS,
        &beta,
        elm.data(),
        1
    );
  }

  return vec<DIM_COLS, T>{std::move (elm)};
}

/**
 @brief Determines whether two matrices are similar to each other element-wise.
 */
template <size_t DIM_ROWS, size_t DIM_COLS, typename T>
bool
similar (
    const mat<DIM_ROWS, DIM_COLS, T>& M1,
    const mat<DIM_ROWS, DIM_COLS, T>& M2,
    const double                      tol = std::numeric_limits<T>::epsilon()
) {
  auto diff{M1 - M2};

  return std::find_if (diff.cbegin(), diff.cend(), [&tol] (const auto& v) {
           return std::abs (v) > tol;
         }) == diff.end();
}

// -----------------------------------------------------------------------------
//                                                  Matrix Operations: Algebraic
// -----------------------------------------------------------------------------
/**
 @brief Calculates the trace of a square matrix
 */
template <size_t DIM, typename T>
T
tr (const mat<DIM, DIM, T>& M) {
  T tr{0.0};
  for (size_t i = 0; i < DIM; ++i) {
    tr += M.elem()[i * DIM + i];
  }
  return tr;
}

/**
 @brief Calculates the determinant of a square matrix

 @details
 Lapack's dgetrf() computes a A=P*L*U decomposition for a general M-by-N matrix
 A. Assuming an invertible square matrix A, its determinant can be computed as a
 product:

 - U is an upper triangular matrix.  Hence, its determinant is the product of
 the diagonal elements, which happens to be the diagonal elements of the output
 A. Indeed, see how the output A is defined:

 On exit, the factors L and U from the factorization A = P*L*U; the unit
 diagonal elements of L are not stored.

 - L is a lower triangular matrix featuring unit diagonal elements which are not
 stored.  Hence, its determinant is always 1.

 - P is a permutation matrix coded as a product of transpositions, i.e.,
 2-cycles or swap.  Indeed, see dgetri() to understand how it is used.  Hence,
 its determinant is either 1 or -1, depending on whether the number of
 transpositions is even or odd.  As a result, the determinant of P can be
 computed as:

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
template <size_t DIM>
double
det (const mat<DIM, DIM>& M) {
  auto LU = std::make_unique<real_t[]> (DIM * DIM);
  std::memcpy (LU.get(), M.data(), sizeof (real_t) * DIM * DIM);

  integer_t N{static_cast<integer_t> (DIM)};
  integer_t INFO;

  auto IPIV = std::make_unique<integer_t[]> (DIM);

  dgetrf_ (&N, &N, LU.get(), &N, IPIV.get(), &INFO);

  double d = 0.0;
  if (INFO != 0) {
    return d;
  }

  d = 1.0;
  for (size_t i = 0; i < DIM; ++i) {
    if (IPIV[i] != static_cast<integer_t> (i + 1)) {
      d *= -LU[i * DIM + i];
    } else {
      d *= LU[i * DIM + i];
    }
  }

  return d;
}

/**
 @brief Invert a square matrix
 */
template <size_t DIM>
mat<DIM, DIM>
inv (const mat<DIM, DIM>& m) {
  integer_t     n = {DIM};
  mat<DIM, DIM> result{m};
  integer_t     INFO;

  auto IPIV = std::make_unique<integer_t[]> (DIM);

  dgetrf_ (&n, &n, result.data(), &n, IPIV.get(), &INFO);

  double d = 1.;
  for (size_t i = 0; i < DIM; ++i)
    d *= result.elem()[i * DIM + i];

  if (std::abs (d) < TOL) throw std::runtime_error{"The mat is singular."};

  n              = static_cast<integer_t> (DIM);
  integer_t prod = static_cast<integer_t> (DIM * DIM);
  auto      WORK = std::make_unique<real_t[]> (DIM);

  dgetri_ (&n, result.data(), &n, IPIV.get(), WORK.get(), &prod, &INFO);

  return result;
}

template <size_t DIM_ROWS, size_t DIM_COLS> struct svd_t {
  mat<DIM_ROWS, DIM_ROWS>            U;
  vec<std::min (DIM_ROWS, DIM_COLS)> S;
  mat<DIM_COLS, DIM_COLS>            V;
};

/**
 @brief Decomposes a matrix into two unitary matrices and a diagonal matrix
 (SVD)

 @details This function decomposes a m x n matrix into m x m unitary matrix,
 m x n diagonal matrix of singular values, and n x n unitary matrix.

 One possible SVD of the a matrix M:
   M = U.Σ.V^†
   where
   M = ( 7.52   -1.1    -7.95    1.08
        -0.76    0.62    9.34   -7.1
         5.13    6.62   -5.66    0.87
        -4.75    8.52    5.75    5.3
         1.33    4.91   -5.49   -3.52
        -2.4    -6.77    2.34    3.95 )
   U = (-0.572674    0.177563    0.0056271    0.529022    0.58299    -0.144023
         0.459422   -0.107528   -0.724027     0.417373    0.167946    0.225273
        -0.450447   -0.413957    0.00417222   0.36286    -0.532307    0.459023
         0.334096   -0.692623    0.494818     0.185129    0.358495   -0.0318806
        -0.317397   -0.308371   -0.280347    -0.60983     0.437689    0.402626
         0.213804    0.459053    0.390253     0.0900183   0.168744    0.744771 )
   Σ = (18.366   0      0       0
         0      13.63   0       0
         0       0     10.8533  0
         0       0      0       4.49157
         0       0      0       0
         0       0      0       0 )
   V = (-0.516645    0.0786131  -0.280639   0.805071
        -0.121232   -0.992329   -0.0212036  0.0117076
         0.847064   -0.0945254  -0.141271   0.503578
        -0.0293912  -0.0129938   0.949123   0.313262 )

 Note that because the matrix Σ has zeros in the 5'th and 6'th rows, the 5'th
 and 6'th columns of the unitary matrix U might be different from the result of
 this function.
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
svd_t<DIM_ROWS, DIM_COLS>
svd (const mat<DIM_ROWS, DIM_COLS>& M) {
  integer_t m{static_cast<integer_t> (DIM_ROWS)};
  integer_t n{static_cast<integer_t> (DIM_COLS)};
  integer_t lda  = m;
  integer_t ldu  = m;
  integer_t ldvt = n;
  integer_t ds   = std::min (m, n);

  char      jobz = 'A';
  integer_t lwork =
      3 * ds * ds +
      std::max (std::max (m, n), 5 * std::min (m, n) * std::min (m, n) + 4 * std::min (m, n));

  auto el = std::make_unique<real_t[]> (DIM_ROWS * DIM_COLS);
  std::memcpy (el.get(), M.data(), sizeof (real_t) * DIM_ROWS * DIM_COLS);

  auto      s     = std::make_unique<real_t[]> (ds);
  auto      u     = std::make_unique<real_t[]> (ldu * m);
  auto      vt    = std::make_unique<real_t[]> (ldvt * n);
  auto      work  = std::make_unique<real_t[]> (lwork);  // std::max(1, lwork));
  auto      iwork = std::make_unique<integer_t[]> (8 * ds);
  integer_t info;

  dgesdd_ (
      &jobz,
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
      &info
  );

  if (info > 0) {
    throw std::runtime_error{"The algorithm for SVD failed to converge."};
  }

  // Create matrix U
  mat<DIM_ROWS, DIM_ROWS> U{u.get()};

  // Copy vt to the matrix Vt and transpose it
  mat<DIM_COLS, DIM_COLS> Vt{vt.get()};

  return svd_t<DIM_ROWS, DIM_COLS>{U, vec<std::min (DIM_ROWS, DIM_COLS)> (s.get()), transpose (Vt)};
}

/**
 @brief Checks whether the matrix is symmetric
 */
template <size_t DIM>
bool
is_symmetric (const mat<DIM, DIM>& M) {
  return M == transpose (M);
}

/**
 @brief Class to store eigenvalues and eigenvectors
 */
template <size_t DIM> struct eigensystem {
  // Eigenvalues
  vec<DIM, complex_t> eigvals;

  // Eigenvectors
  mat<DIM, DIM, complex_t> eigvecs_rt;
  mat<DIM, DIM, complex_t> eigvecs_lft;
};

/**
 @brief Solves Eigensystem problem
 */
template <size_t DIM>
eigensystem<DIM>
eigen (const mat<DIM, DIM>& M, eigen js = eigen::val) {
  eigensystem<DIM> es;

  integer_t     N{DIM};
  mat<DIM, DIM> A{M};
  integer_t     LDA{N};
  integer_t     INFO;

  if (is_symmetric (M)) {
    // Symmetric matrix has real eigenvalues and eigenvectors
    char JOBZ;
    if (js == eigen::val) {
      JOBZ = 'N';
    } else {
      JOBZ = 'V';
    }
    char UPLO = 'U';

    auto W = std::make_unique<real_t[]> (N);

    integer_t LWORK{3 * N};
    auto      WORK = std::make_unique<real_t[]> (LWORK);

    dsyev_ (&JOBZ, &UPLO, &N, A.data(), &LDA, W.get(), WORK.get(), &LWORK, &INFO);

    if (INFO == 0) {
      es.eigvals = vec<DIM, complex_t>{W.get()};

      if (js != eigen::val) es.eigvecs_rt = mat<DIM, DIM, complex_t>{std::move (A.data())};
    } else throw std::runtime_error{"Failed to calculate eigenvalues."};
  } else {
    char          JOBVL, JOBVR;
    integer_t     LDVL{N};
    integer_t     LDVR{N};
    mat<DIM, DIM> VL{};
    mat<DIM, DIM> VR{};
    size_t        UN{static_cast<size_t> (N)};

    switch (js) {
    case eigen::val: JOBVL = JOBVR = 'N'; break;
    case eigen::vec: JOBVL = JOBVR = 'V'; break;
    case eigen::lvec:
      JOBVL = 'V';
      JOBVR = 'N';
      break;
    case eigen::rvec:
      JOBVL = 'N';
      JOBVR = 'V';
      break;
    }
    auto WR = std::make_unique<double[]> (N);
    auto WI = std::make_unique<double[]> (N);

    integer_t LWORK{4 * N};
    auto      WORK = std::make_unique<double[]> (LWORK);

    dgeev_ (
        &JOBVL,
        &JOBVR,
        &N,
        A.data(),
        &LDA,
        WR.get(),
        WI.get(),
        VL.data(),
        &LDVL,
        VR.data(),
        &LDVR,
        WORK.get(),
        &LWORK,
        &INFO
    );

    if (INFO == 0) {
      for (size_t j = 1; j <= UN; ++j) {
        // Eigenvalue
        es.eigvals[j - 1] = complex_t{WR[j - 1], WI[j - 1]};

        switch (js) {
        case eigen::val: break;

        case eigen::vec:
          if (WI[j - 1] != 0. && j < UN && WI[j - 1] == -WI[j]) {
            // Complex conjugate

            // Next Eigenvalue
            es.eigvals[j] = complex_t{WR[j], WI[j]};

            // Right Eigenvectors
            es.eigvecs_rt.set_col (j, cvec (VR.col (j), VR.col (j + 1)));
            es.eigvecs_rt.set_col (j + 1, cvec (VR.col (j), -VR.col (j + 1)));

            // Left Eigenvectors
            es.eigvecs_lft.set_col (j, cvec (VL.col (j), VL.col (j + 1)));
            es.eigvecs_lft.set_col (j + 1, cvec (VL.col (j), -VL.col (j + 1)));

            // Skip the next Eigenvalue & Eigenvectors
            j++;
          } else {  // Real
            // Right Eigenvectors
            es.eigvecs_rt.set_col (j, cvec (VR.col (j)));

            // Left Eigenvectors
            es.eigvecs_lft.set_col (j, cvec (VL.col (j)));
          }
          break;

        case eigen::lvec:
          if (WI[j - 1] != 0. && j < UN && WI[j - 1] == -WI[j]) {
            // Complex conjugate

            // Next Eigenvalue
            es.eigvals[j] = complex_t{WR[j], WI[j]};

            // Left Eigenvectors
            es.eigvecs_lft.set_col (j, cvec (VL.col (j), VL.col (j + 1)));
            es.eigvecs_lft.set_col (j + 1, cvec (VL.col (j), -VL.col (j + 1)));

            // Skip the next Eigenvalue & Eigenvectors
            j++;
          } else {  // Real
            // Left Eigenvectors
            es.eigvecs_lft.set_col (j, cvec (VL.col (j)));
          }
          break;

        case eigen::rvec:
          if (WI[j - 1] != 0. && j < UN && WI[j - 1] == -WI[j]) {
            // Complex conjugate

            // Next Eigenvalue
            es.eigvals[j] = complex_t{WR[j], WI[j]};

            // Right Eigenvectors
            es.eigvecs_rt.set_col (j, cvec (VR.col (j), VR.col (j + 1)));
            es.eigvecs_rt.set_col (j + 1, cvec (VR.col (j), -VR.col (j + 1)));

            // Skip the next Eigenvalue & Eigenvectors
            j++;
          } else {  // Real
            // Right Eigenvectors
            es.eigvecs_rt.set_col (j, cvec (VR.col (j)));
          }
          break;
        }
      }
    } else throw std::runtime_error{"Failed to calculate eigenvalues."};
  }

  return es;
}

/**
 @brief Calculates Frobenius norm
 */
template <size_t DIM_ROWS, size_t DIM_COLS>
double
norm_frobenius (const mat<DIM_ROWS, DIM_COLS>& M) {
  return std::sqrt (tr (transpose (M) * M));
}

}  // namespace gpw::blat

#endif

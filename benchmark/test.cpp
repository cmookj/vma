//
//  main.cpp
//  benchmark
//
//  Created by Changmook Chun on 12/15/24.
//

#include "vma.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <iterator>
#include <utility>
#include <vector>

// Measure time of execution for a function
using time_point = std::chrono::high_resolution_clock::time_point;

time_point
now () {
    return std::chrono::high_resolution_clock::now();
}

double
duration (time_point since) {
    return std::chrono::duration_cast<std::chrono::milliseconds> (now() - since).count();
}

template <typename F, typename... Args>
double
measure_time (F func, Args&&... args) {
    time_point t1 = now();
    func (std::forward<Args> (args)...);
    return duration (t1);
}

using namespace gpw::vma;

//
//                         Matrix Transpose Test
//
template <size_t sz_r, size_t sz_c>
void
populate (mat<sz_r, sz_c>& m) {
    for (std::size_t i = 1; i <= m.count_rows(); ++i)
        for (std::size_t j = 1; j <= m.count_cols(); ++j)
            m (i, j) = 0.5 * i + 0.3 * j;
}

template <size_t sz_r, size_t sz_c>
bool
verify (mat<sz_r, sz_c>& m1, mat<sz_c, sz_r>& m2) {
    if (m1.count_rows() != m2.count_cols() || m1.count_cols() != m2.count_rows()) return false;

    for (std::size_t i = 1; i <= m2.count_rows(); ++i)
        for (std::size_t j = 1; j <= m2.count_cols(); ++j)
            if (abs (m2 (i, j) - (0.5 * j + 0.3 * i)) > 0.000000001) return false;

    return true;
}

template <size_t sz_r, size_t sz_c>
void
test_transpose (mat<sz_r, sz_c>& m) {
    auto tr = transpose (m);

    if (verify (m, tr)) std::cout << "OK!\n";
    else std::cout << "Something wrong...\n";
}

template <size_t sz_r, size_t sz_c>
double
transpose_heavy () {
    mat<sz_r, sz_c> m{};
    populate (m);

    return measure_time (test_transpose<sz_r, sz_c>, m);
}

double
transpose_4GiB () {
    return transpose_heavy<16384, 16384>();
}

double
transpose_8GiB () {
    return transpose_heavy<32768, 16384>();
}

double
transpose_16GiB () {
    return transpose_heavy<32768, 32768>();
}

void
test_heavy_matrix_transpose (const unsigned count) {
    std::vector<double> times_4 (count);
    std::vector<double> times_8 (count);
    std::vector<double> times_16 (count);

    std::generate_n (times_4.begin(), count, transpose_4GiB);
    std::generate_n (times_8.begin(), count, transpose_8GiB);
    std::generate_n (times_16.begin(), count, transpose_16GiB);

    auto time_4  = std::accumulate (times_4.cbegin(), times_4.cend(), 0.0) / double (count);
    auto time_8  = std::accumulate (times_8.cbegin(), times_8.cend(), 0.0) / double (count);
    auto time_16 = std::accumulate (times_16.cbegin(), times_16.cend(), 0.0) / double (count);

    std::cout << "---- Transpose ----\n";
    std::cout << "Average of " << count << " times measurements\n";
    std::cout << "4GiB " << time_4 << " msec\n";
    std::cout << "8GiB " << time_8 << " msec\n";
    std::cout << "16GiB " << time_16 << " msec\n";

    std::cout << "\nRaw measurement data\n";
    std::copy (times_4.cbegin(), times_4.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
    std::copy (times_8.cbegin(), times_8.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
    std::copy (times_16.cbegin(), times_16.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
}

//
//                      Matrix Multiplication Test
//
template <size_t sz>
void
post_multiply (mat<sz, sz>& m1, mat<sz, sz>& m2) {
    m1 *= m2;
}

template <size_t sz>
void
multiply_matrices (mat<sz, sz>& ab, mat<sz, sz>& a, mat<sz, sz>& b) {
    ab = a * b;
}

void
print_message (const std::string& msg) {
    std::cout << msg << '\n';
}

template <size_t sz>
double
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

    // Multiplication case #1
    auto time_1 = measure_time (post_multiply<sz>, mm0, mm1);

    // Verify
    if (mm0.count_rows() != prev_count_rows) print_message ("[Error] dimension mismatch");
    if (mm0.count_cols() != mm2.count_cols()) print_message ("[Error] dimension mismatch");

    const std::size_t p{mm0.count_cols()};
    for (std::size_t i = 1; i <= mm0.count_rows(); i = i << 1)
        for (std::size_t j = 1; j <= mm0.count_cols(); j = j << 1)
            if (abs (mm0 (i, j) - i * p * (p + 1) / 2) > 0.000001) {
                print_message ("[Error] incorrect computation in multiplication #1");
                return 0.0;
            }

    // Multiplication case #2
    mat<sz, sz> mm12;
    auto        time_2 = measure_time (multiply_matrices<sz>, mm12, mm1, mm2);

    // Verify
    for (std::size_t i = 1; i <= mm12.count_rows(); i = i << 1)
        for (std::size_t j = 1; j <= mm12.count_cols(); j = j << 1)
            if (abs (mm12 (i, j) - mm1.count_cols() * i * j) > 0.000001) {
                print_message ("[Error] incorrect computation in multiplication #2");
                return 0.0;
            }

    print_message ("OK!");
    return time_1 + time_2;
}

double
test_heavy_multiplication_512MiB () {
    mat<4096, 4096> mm0;
    mat<4096, 4096> mm1;
    mat<4096, 4096> mm2;

    return heavy_multiplication (mm0, mm1, mm2);
}

double
test_heavy_multiplication_2GiB () {
    mat<8192, 8192> mm0;
    mat<8192, 8192> mm1;
    mat<8192, 8192> mm2;

    return heavy_multiplication (mm0, mm1, mm2);
}

double
test_heavy_multiplication_8GiB () {
    mat<16384, 16384> mm0;
    mat<16384, 16384> mm1;
    mat<16384, 16384> mm2;

    return heavy_multiplication (mm0, mm1, mm2);
}

double
test_heavy_multiplication_32GiB () {
    mat<32768, 32768> mm0;
    mat<32768, 32768> mm1;
    mat<32768, 32768> mm2;

    return heavy_multiplication (mm0, mm1, mm2);
}

void
test_heavy_matrix_multiplication (const size_t count) {
    std::vector<double> times_h (count);
    std::vector<double> times_2 (count);
    std::vector<double> times_8 (count);
    std::vector<double> times_32 (count);

    std::generate_n (times_h.begin(), count, test_heavy_multiplication_512MiB);
    std::generate_n (times_2.begin(), count, test_heavy_multiplication_2GiB);
    std::generate_n (times_8.begin(), count, test_heavy_multiplication_8GiB);
    std::generate_n (times_32.begin(), count, test_heavy_multiplication_32GiB);

    auto time_h  = std::accumulate (times_h.cbegin(), times_h.cend(), 0.0) / double (count);
    auto time_2  = std::accumulate (times_2.cbegin(), times_2.cend(), 0.0) / double (count);
    auto time_8  = std::accumulate (times_8.cbegin(), times_8.cend(), 0.0) / double (count);
    auto time_32 = std::accumulate (times_32.cbegin(), times_32.cend(), 0.0) / double (count);

    std::cout << "---- Multiplication ----\n";
    std::cout << "Average of " << count << " times measurements\n";
    std::cout << "512MiB " << time_h << " msec\n";
    std::cout << "2GiB " << time_2 << " msec\n";
    std::cout << "8GiB " << time_8 << " msec\n";
    std::cout << "32GiB " << time_32 << " msec\n";

    std::cout << "\nRaw measurement data\n";
    std::copy (times_h.cbegin(), times_h.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
    std::copy (times_2.cbegin(), times_2.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
    std::copy (times_8.cbegin(), times_8.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
    std::copy (times_32.cbegin(), times_32.cend(), std::ostream_iterator<double> (std::cout, ", "));
    std::cout << "\n";
}

int
main (int argc, const char* argv[]) {
    test_heavy_matrix_transpose (10);
    std::cout << "\n";
    test_heavy_matrix_multiplication (10);
    std::cout << "\n";

    return 0;
}

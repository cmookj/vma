# Vector Matrix Arithmetic Library

This is a very simple linear algebra library.  Currently, it supports real/complex vectors and matrices.

## System Requirement

### macOS

Tested on macOS Ventura 13.1 throught macOS SEquoia 15.7.1.

### Ubuntu Linux

Tested on Ubuntu 22.04 in QEMU virtual machine (ARM 64), which requires
```shell
sudo apt install -y libf2c2-dev libatlas-base-dev libgfortran-11-dev
```

## Test

To test (with `c++17`):
```
GTEST_COLOR=1 bazel test --cxxopt=-std=c++17 --test_output=all //test:point_test
```

## License

MIT License

Copyright (c) 2025 Changmook Chun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


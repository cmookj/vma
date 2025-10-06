load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_library(
    name = "vma",
    srcs = ["vma.cc"],
    hdrs = ["vma.h"],
    copts = ["-framework", "Accelerate"],
    linkopts = ["-framework", "Accelerate"],
    visibility = ["//test:__pkg__"],
)

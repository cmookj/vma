load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")

cc_test(
    name = "vma_test",
    size = "small",
    srcs = ["vma_test.cc"],
    deps = [
        "@googletest//:gtest",
        "@googletest//:gtest_main",
        "//core:vma",
    ],
    linkopts = ["-framework", "Accelerate"],
)

load("@rules_cc//cc:cc_library.bzl", "cc_library")

# cc_library(
#     name = "vma",
#     srcs = ["vma.cc"],
#     hdrs = ["vma.h"],
#     copts = ["-framework", "Accelerate"],
#     linkopts = ["-framework", "Accelerate"],
#     visibility = ["//visibility:public"],
# )

cc_library(
    name = "vma",
    srcs = ["vma.cc"],
    hdrs = ["vma.h", "concurrency.h" ],
    copts = select({
        "@platforms//os:macos": ["-framework", "Accelerate"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "@platforms//os:macos": ["-framework", "Accelerate"],
        "@platforms//os:linux": ["-lopenblas", "-lgfortran"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)


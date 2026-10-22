//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_DPCT_HELPER_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DPCT_HELPER_HPP

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <map>

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/half_type.hpp>
#ಅನ್ತರ್ಭಾವ <oneapi/mkl.hpp>

#ಆದರೆ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/mman.h>
#ಅಥವಾಆದರೆ defined(_WIN64)
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NOMINMAX
#ವ್ಯಾಖ್ಯಾನಿಸು NOMINMAX
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <windows.h>
#ಇಲ್ಲದಿದ್ದರೆ
#ದೋಷ "Only support Windows and Linux."
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <sys/syscall.h>
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ defined(_WIN64)
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ NOMINMAX
#ವ್ಯಾಖ್ಯಾನಿಸು NOMINMAX
#ಆದರೆ_ಅಂತ್ಯ
#ಅನ್ತರ್ಭಾವ <windows.h>
#ಆದರೆ_ಅಂತ್ಯ


#ವ್ಯಾಖ್ಯಾನಿಸು DPCT_COMPATIBILITY_ತಾತ್ಕಾಲಿಕ (900)

#ಆದರೆ defined(_MSC_VER)
#ವ್ಯಾಖ್ಯಾನಿಸು __dpct_align__(n) __declspec(align(n))
#ವ್ಯಾಖ್ಯಾನಿಸು __dpct_inline__ __forceinline
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು __dpct_align__(n) __attribute__((aligned(n)))
#ವ್ಯಾಖ್ಯಾನಿಸು __dpct_inline__ __inline__ __attribute__((always_inline))
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(_MSC_VER)
#ವ್ಯಾಖ್ಯಾನಿಸು __dpct_noinline__ __declspec(noinline)
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು __dpct_noinline__ __attribute__((noinline))
#ಆದರೆ_ಅಂತ್ಯ

#ವ್ಯಾಖ್ಯಾನಿಸು DPCT_ಬಳಸದ(x) (ಶೂನ್ಯ)(x)

[[noreturn]] ಸಾಲೊಳಗೆ ಶೂನ್ಯ _abort(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * str) {
    std::cerr << str << std::ಸಾಲು_ಅಂತ್ಯ;
    std::abort();
}

ಸಾಲೊಳಗೆ std::string get_device_type_name(ಸ್ಥಿರಾಂಕ sycl::device &Device) {
    ಸ್ವಯಂಚಾಲಿತ DeviceType = Device.get_info<sycl::info::device::device_type>();
    ಆಯ್ಕೆ (DeviceType) {
    ಸ್ಥಿತಿ sycl::info::device_type::cpu:
        ಹಿಂತಿರುಗಿ "cpu";
    ಸ್ಥಿತಿ sycl::info::device_type::gpu:
        ಹಿಂತಿರುಗಿ "gpu";
    ಸ್ಥಿತಿ sycl::info::device_type::host:
        ಹಿಂತಿರುಗಿ "host";
    ಸ್ಥಿತಿ sycl::info::device_type::accelerator:
        ಹಿಂತಿರುಗಿ "acc";
    ಪೂರ್ವನಿಯೋಜಿತ:
        ಹಿಂತಿರುಗಿ "unknown";
    }
}

ಸಾಲೊಳಗೆ std::string get_device_backend_and_type(ಸ್ಥಿರಾಂಕ sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    ಹಿಂತಿರುಗಿ device_type.str();
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Ts> ರಚನೆ ಮಾತೃಕೆ_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     ಮೌಲ್ಯ_info[2];
    std::int64_t           ಗಾತ್ರ_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

ನಾಮಾಕಾಶ dpct
{
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::queue *ಸರತಿ_ptr;
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::event *event_ptr;
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಅಕ್ಷರ *device_ptr;
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint8_t byte_t;
    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ sycl::ಬಫರ್<byte_t> ಬಫರ್_t;

    /// SYCL default exception handler
    ಸಾಲೊಳಗೆ ಸ್ವಯಂಚಾಲಿತ exception_handler = [](sycl::exception_list exceptions)
    {
        ಗಾಗಿ (std::exception_ptr ಸ್ಥಿರಾಂಕ &e : exceptions)
        {
            ಪ್ರಯತ್ನಿಸು
            {
                std::rethrow_exception(e);
            }
            ಹಿಡಿ (sycl::exception ಸ್ಥಿರಾಂಕ &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::ಸಾಲು_ಅಂತ್ಯ
                          << e.what() << std::ಸಾಲು_ಅಂತ್ಯ
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::ಸಾಲು_ಅಂತ್ಯ;
            }
        }
    };

    ಎಣಿಕೆ error_code
    {
        success = 0,
        default_error = 999
    };

    ಎಣಿಕೆ memcpy_direction
    {
        host_ಗಮ್ಯ_host,
        host_ಗಮ್ಯ_device,
        device_ಗಮ್ಯ_host,
        device_ಗಮ್ಯ_device,
        automatic
    };

    ಎಣಿಕೆ ಸ್ಮರಣೆ_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    ಎಣಿಕೆ ವರ್ಗ library_ದತ್ತಾಂಶ_t : ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ
    {
        real_float = 0,
        complex_float,
        real_double,
        complex_double,
        real_half,
        complex_half,
        real_bfloat16,
        complex_bfloat16,
        real_int4,
        complex_int4,
        real_uint4,
        complex_uint4,
        real_int8,
        complex_int8,
        real_uint8,
        complex_uint8,
        real_int16,
        complex_int16,
        real_uint16,
        complex_uint16,
        real_int32,
        complex_int32,
        real_uint32,
        complex_uint32,
        real_int64,
        complex_int64,
        real_uint64,
        complex_uint64,
        real_int8_4,
        real_int8_32,
        real_uint8_4,
        library_ದತ್ತಾಂಶ_t_ಗಾತ್ರ
    };

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ರಚನೆ ದತ್ತಾಂಶ_Type_2
    {
        ಬಳಸಿ T2 = T;
    };
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ರಚನೆ ದತ್ತಾಂಶ_Type_2<sycl::vec<T, 2>>
    {
        ಬಳಸಿ T2 = std::complex<T>;
    };

    ಸ್ಥಿರ ಶೂನ್ಯ destroy_event(event_ptr event)
    {
        ಅಳಿಸು event;
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ get_tid()
    {
#ಆದರೆ defined(__linux__)
        ಹಿಂತಿರುಗಿ syscall(SYS_gettid);
#ಅಥವಾಆದರೆ defined(_WIN64)
        ಹಿಂತಿರುಗಿ GetCurrentThreadId();
#ಇಲ್ಲದಿದ್ದರೆ
#ದೋಷ "Only support Windows and Linux."
#ಆದರೆ_ಅಂತ್ಯ
    }

    ನಾಮಾಕಾಶ detail
    {
        ಸ್ಥಿರ ಶೂನ್ಯ get_ಆವೃತ್ತಿ(ಸ್ಥಿರಾಂಕ sycl::device &dev, ಪೂರ್ಣಾಂಕ &major, ಪೂರ್ಣಾಂಕ &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::ಆವೃತ್ತಿ_2>();
            std::string::size_type i = 0;
            ಆಗ (i < ver.size()) {
              ಆದರೆ (ಅಂಕಿಯೇ(ver[i]))
                ನಿಲ್ಲಿಸು;
              i++;
            }
            major = std::stoi(&(ver[i]));
            ಆಗ (i < ver.size()) {
              ಆದರೆ (ver[i] == '.')
                ನಿಲ್ಲಿಸು;
              i++;
            }
            ಆದರೆ (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } ಇಲ್ಲದಿದ್ದರೆ {
              // c.
              minor = 0;
            }
        }

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಟ್ಯಾಗ್, ಬಗೆ_ಹೆಸರು T>
        ವರ್ಗ generic_error_type
        {
        ಸಾರ್ವಜನಿಕ:
            generic_error_type() = ಪೂರ್ವನಿಯೋಜಿತ;
            generic_error_type(T value) : value{value} {}
            ಕ್ರಿಯಾಕಾರಕ T() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ value; }

        ಖಾಸಗಿ:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    ವರ್ಗ dim3 {
        ಸಾರ್ವಜನಿಕ:
        ಚಿಹ್ನೆರಹಿತ x, y, z;

        constexpr dim3(ಚಿಹ್ನೆರಹಿತ x = 1, ಚಿಹ್ನೆರಹಿತ y = 1, ಚಿಹ್ನೆರಹಿತ z = 1)
            : x(x), y(y), z(z) {}

        dim3(ಸ್ಥಿರಾಂಕ sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        ಕ್ರಿಯಾಕಾರಕ sycl::range<3>() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ sycl::range<3>(z, y, x); }
    }; // namespace dim3

    ಸಾಲೊಳಗೆ dim3 ಕ್ರಿಯಾಕಾರಕ*(ಸ್ಥಿರಾಂಕ dim3 &a, ಸ್ಥಿರಾಂಕ dim3 &b) {
    ಹಿಂತಿರುಗಿ dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    ವರ್ಗ pitched_ದತ್ತಾಂಶ
    {
    ಸಾರ್ವಜನಿಕ:
        pitched_ದತ್ತಾಂಶ() : pitched_ದತ್ತಾಂಶ(ಶೂನ್ಯಸೂಚಕ, 0, 0, 0) {}
        pitched_ದತ್ತಾಂಶ(ಶೂನ್ಯ *ದತ್ತಾಂಶ, size_t pitch, size_t x, size_t y)
            : ದತ್ತಾಂಶ_2(ದತ್ತಾಂಶ), _pitch(pitch), _x(x), _y(y) {}

        ಶೂನ್ಯ *get_ದತ್ತಾಂಶ_ptr() { ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ_2; }
        ಶೂನ್ಯ ಗಣ_ದತ್ತಾಂಶ_ptr(ಶೂನ್ಯ *ದತ್ತಾಂಶ) { ದತ್ತಾಂಶ_2 = ದತ್ತಾಂಶ; }

        size_t get_pitch() { ಹಿಂತಿರುಗಿ _pitch; }
        ಶೂನ್ಯ ಗಣ_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { ಹಿಂತಿರುಗಿ _x; }
        ಶೂನ್ಯ ಗಣ_x(size_t x) { _x = x; }

        size_t get_y() { ಹಿಂತಿರುಗಿ _y; }
        ಶೂನ್ಯ ಗಣ_y(size_t y) { _y = y; }

    ಖಾಸಗಿ:
        ಶೂನ್ಯ *ದತ್ತಾಂಶ_2;
        size_t _pitch, _x, _y;
    };

    ವರ್ಗ device_info
    {
    ಸಾರ್ವಜನಿಕ:
        // get interface
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *get_name() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _name; }
        ಅಕ್ಷರ *get_name() { ಹಿಂತಿರುಗಿ _name; }
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Work_ಅಂಶ_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_ಅಂಶ_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_ಅಂಶ_Sizes_Ty, ಪೂರ್ಣಾಂಕ *>,
                                   ಪೂರ್ಣಾಂಕ> = 0>
        ಸ್ವಯಂಚಾಲಿತ get_max_work_ಅಂಶ_sizes() ಸ್ಥಿರಾಂಕ
        {
            ಆದರೆ constexpr (std::is_same_v<Work_ಅಂಶ_Sizes_Ty, sycl::range<3>>)
                ಹಿಂತಿರುಗಿ sycl::range<3>(max_work_ಅಂಶ_sizes_i[0],
                                      max_work_ಅಂಶ_sizes_i[1],
                                      max_work_ಅಂಶ_sizes_i[2]);
            ಇಲ್ಲದಿದ್ದರೆ
            {
                ಹಿಂತಿರುಗಿ max_work_ಅಂಶ_sizes_i;
            }
        }
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು Work_ಅಂಶ_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_ಅಂಶ_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_ಅಂಶ_Sizes_Ty, ಪೂರ್ಣಾಂಕ *>,
                                   ಪೂರ್ಣಾಂಕ> = 0>
        ಸ್ವಯಂಚಾಲಿತ get_max_work_ಅಂಶ_sizes()
        {
            ಆದರೆ constexpr (std::is_same_v<Work_ಅಂಶ_Sizes_Ty, sycl::range<3>>)
                ಹಿಂತಿರುಗಿ sycl::range<3>(max_work_ಅಂಶ_sizes_i[0],
                                      max_work_ಅಂಶ_sizes_i[1],
                                      max_work_ಅಂಶ_sizes_i[2]);
            ಇಲ್ಲದಿದ್ದರೆ
            {
                ಹಿಂತಿರುಗಿ max_work_ಅಂಶ_sizes_i;
            }
        }
        ತಾರ್ಕಿಕ get_host_unified_ಸ್ಮರಣೆ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ host_unified_ಸ್ಮರಣೆ; }
        ಪೂರ್ಣಾಂಕ get_major_ಆವೃತ್ತಿ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _major; }
        ಪೂರ್ಣಾಂಕ get_minor_ಆವೃತ್ತಿ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _minor; }
        ಪೂರ್ಣಾಂಕ get_integrated() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _integrated; }
        ಪೂರ್ಣಾಂಕ get_max_clock_frequency() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _frequency; }
        ಪೂರ್ಣಾಂಕ get_max_compute_units() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _max_compute_units; }
        ಪೂರ್ಣಾಂಕ get_max_work_ಗುಂಪು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ max_work_ಗುಂಪು_ಗಾತ್ರ_2; }
        ಪೂರ್ಣಾಂಕ get_max_sub_ಗುಂಪು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ max_sub_ಗುಂಪು_ಗಾತ್ರ; }
        ಪೂರ್ಣಾಂಕ get_max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit() ಸ್ಥಿರಾಂಕ
        {
            ಹಿಂತಿರುಗಿ max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit;
        }
        ಪೂರ್ಣಾಂಕ get_max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು() ಸ್ಥಿರಾಂಕ
        {
            ಹಿಂತಿರುಗಿ max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು;
        }
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು NDRange_ಗಾತ್ರ_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_ಗಾತ್ರ_Ty, size_t *> ||
                                       std::is_same_v<NDRange_ಗಾತ್ರ_Ty, ಪೂರ್ಣಾಂಕ *>,
                                   ಪೂರ್ಣಾಂಕ> = 0>
        ಸ್ವಯಂಚಾಲಿತ get_max_nd_range_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ
        {
            ಆದರೆ constexpr (std::is_same_v<NDRange_ಗಾತ್ರ_Ty, size_t *>)
                ಹಿಂತಿರುಗಿ max_nd_range_ಗಾತ್ರ;
            ಇಲ್ಲದಿದ್ದರೆ
                ಹಿಂತಿರುಗಿ max_nd_range_ಗಾತ್ರ_i;
        }
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು NDRange_ಗಾತ್ರ_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_ಗಾತ್ರ_Ty, size_t *> ||
                                       std::is_same_v<NDRange_ಗಾತ್ರ_Ty, ಪೂರ್ಣಾಂಕ *>,
                                   ಪೂರ್ಣಾಂಕ> = 0>
        ಸ್ವಯಂಚಾಲಿತ get_max_nd_range_ಗಾತ್ರ()
        {
            ಆದರೆ constexpr (std::is_same_v<NDRange_ಗಾತ್ರ_Ty, size_t *>)
                ಹಿಂತಿರುಗಿ max_nd_range_ಗಾತ್ರ;
            ಇಲ್ಲದಿದ್ದರೆ
                ಹಿಂತಿರುಗಿ max_nd_range_ಗಾತ್ರ_i;
        }
        size_t get_global_mem_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ global_mem_ಗಾತ್ರ_2; }
        size_t get_local_mem_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ local_mem_ಗಾತ್ರ; }
        size_t get_max_mem_alloc_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ max_mem_alloc_ಗಾತ್ರ; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ get_ಸ್ಮರಣೆ_clock_ದರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಸ್ಮರಣೆ_clock_ದರ; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ get_ಸ್ಮರಣೆ_bus_ಅಗಲ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಸ್ಮರಣೆ_bus_ಅಗಲ; }
        uint32_t get_device_id() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _device_id; }
        std::array<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ, 16> get_uuid() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _uuid; }
        /// Returns global memory cache size in bytes.
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ get_global_mem_cache_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ
        {
            ಹಿಂತಿರುಗಿ global_mem_cache_ಗಾತ್ರ;
        }

        // set interface
        ಶೂನ್ಯ ಗಣ_name(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *name)
        {
            size_t ಉದ್ದಳತೆ = ಅಕ್ಷರಸರಣಿ_ಉದ್ದ(name);
            ಆದರೆ (ಉದ್ದಳತೆ < 256)
            {
                std::ಸ್ಮರಣೆ_ನಕಲು(_name, name, ಉದ್ದಳತೆ + 1);
            }
            ಇಲ್ಲದಿದ್ದರೆ
            {
                std::ಸ್ಮರಣೆ_ನಕಲು(_name, name, 255);
                _name[255] = '\0';
            }
        }
        ಶೂನ್ಯ ಗಣ_max_work_ಅಂಶ_sizes(ಸ್ಥಿರಾಂಕ sycl::range<3> max_work_ಅಂಶ_sizes)
        {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 3; ++i)
                max_work_ಅಂಶ_sizes_i[i] = max_work_ಅಂಶ_sizes[i];
        }
        [[deprecated]] ಶೂನ್ಯ
        ಗಣ_max_work_ಅಂಶ_sizes(ಸ್ಥಿರಾಂಕ sycl::id<3> max_work_ಅಂಶ_sizes)
        {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 3; ++i)
            {
                max_work_ಅಂಶ_sizes_i[i] = max_work_ಅಂಶ_sizes[i];
            }
        }
        ಶೂನ್ಯ ಗಣ_host_unified_ಸ್ಮರಣೆ(ತಾರ್ಕಿಕ host_unified_ಸ್ಮರಣೆ_2)
        {
            host_unified_ಸ್ಮರಣೆ = host_unified_ಸ್ಮರಣೆ_2;
        }
        ಶೂನ್ಯ ಗಣ_major_ಆವೃತ್ತಿ(ಪೂರ್ಣಾಂಕ major) { _major = major; }
        ಶೂನ್ಯ ಗಣ_minor_ಆವೃತ್ತಿ(ಪೂರ್ಣಾಂಕ minor) { _minor = minor; }
        ಶೂನ್ಯ ಗಣ_integrated(ಪೂರ್ಣಾಂಕ integrated) { _integrated = integrated; }
        ಶೂನ್ಯ ಗಣ_max_clock_frequency(ಪೂರ್ಣಾಂಕ frequency) { _frequency = frequency; }
        ಶೂನ್ಯ ಗಣ_max_compute_units(ಪೂರ್ಣಾಂಕ max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        ಶೂನ್ಯ ಗಣ_global_mem_ಗಾತ್ರ(size_t global_mem_ಗಾತ್ರ)
        {
            global_mem_ಗಾತ್ರ_2 = global_mem_ಗಾತ್ರ;
        }
        ಶೂನ್ಯ ಗಣ_local_mem_ಗಾತ್ರ(size_t local_mem_ಗಾತ್ರ_2)
        {
            local_mem_ಗಾತ್ರ = local_mem_ಗಾತ್ರ_2;
        }
        ಶೂನ್ಯ ಗಣ_max_mem_alloc_ಗಾತ್ರ(size_t max_mem_alloc_ಗಾತ್ರ_2)
        {
            max_mem_alloc_ಗಾತ್ರ = max_mem_alloc_ಗಾತ್ರ_2;
        }
        ಶೂನ್ಯ ಗಣ_max_work_ಗುಂಪು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ max_work_ಗುಂಪು_ಗಾತ್ರ)
        {
            max_work_ಗುಂಪು_ಗಾತ್ರ_2 = max_work_ಗುಂಪು_ಗಾತ್ರ;
        }
        ಶೂನ್ಯ ಗಣ_max_sub_ಗುಂಪು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ max_sub_ಗುಂಪು_ಗಾತ್ರ_2)
        {
            max_sub_ಗುಂಪು_ಗಾತ್ರ = max_sub_ಗುಂಪು_ಗಾತ್ರ_2;
        }
        ಶೂನ್ಯ
        ಗಣ_max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit(ಪೂರ್ಣಾಂಕ max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit_2)
        {
            max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit = max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit_2;
        }
        ಶೂನ್ಯ ಗಣ_max_nd_range_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ max_nd_range_ಗಾತ್ರ_2[])
        {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 3; i++)
            {
                max_nd_range_ಗಾತ್ರ[i] = max_nd_range_ಗಾತ್ರ_2[i];
                max_nd_range_ಗಾತ್ರ_i[i] = max_nd_range_ಗಾತ್ರ_2[i];
            }
        }
        ಶೂನ್ಯ ಗಣ_ಸ್ಮರಣೆ_clock_ದರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಸ್ಮರಣೆ_clock_ದರ_2)
        {
            ಸ್ಮರಣೆ_clock_ದರ = ಸ್ಮರಣೆ_clock_ದರ_2;
        }
        ಶೂನ್ಯ ಗಣ_ಸ್ಮರಣೆ_bus_ಅಗಲ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಸ್ಮರಣೆ_bus_ಅಗಲ_2)
        {
            ಸ್ಮರಣೆ_bus_ಅಗಲ = ಸ್ಮರಣೆ_bus_ಅಗಲ_2;
        }
        ಶೂನ್ಯ
        ಗಣ_max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು(ಪೂರ್ಣಾಂಕ max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು_2)
        {
            max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು = max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು_2;
        }
        ಶೂನ್ಯ ಗಣ_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        ಶೂನ್ಯ ಗಣ_uuid(std::array<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        ಶೂನ್ಯ ಗಣ_global_mem_cache_ಗಾತ್ರ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ global_mem_cache_ಗಾತ್ರ_2)
        {
            global_mem_cache_ಗಾತ್ರ = global_mem_cache_ಗಾತ್ರ_2;
        }

    ಖಾಸಗಿ:
        ಅಕ್ಷರ _name[256];
        ಪೂರ್ಣಾಂಕ max_work_ಅಂಶ_sizes_i[3];
        ತಾರ್ಕಿಕ host_unified_ಸ್ಮರಣೆ = ಸುಳ್ಳು;
        ಪೂರ್ಣಾಂಕ _major;
        ಪೂರ್ಣಾಂಕ _minor;
        ಪೂರ್ಣಾಂಕ _integrated = 0;
        ಪೂರ್ಣಾಂಕ _frequency;
        // Set estimated value 3200000 kHz as default value.
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಸ್ಮರಣೆ_clock_ದರ = 3200000;
        // Set estimated value 64 bits as default value.
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಸ್ಮರಣೆ_bus_ಅಗಲ = 64;
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ global_mem_cache_ಗಾತ್ರ;
        ಪೂರ್ಣಾಂಕ _max_compute_units;
        ಪೂರ್ಣಾಂಕ max_work_ಗುಂಪು_ಗಾತ್ರ_2;
        ಪೂರ್ಣಾಂಕ max_sub_ಗುಂಪು_ಗಾತ್ರ;
        ಪೂರ್ಣಾಂಕ max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit;
        ಪೂರ್ಣಾಂಕ max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು;
        size_t global_mem_ಗಾತ್ರ_2;
        size_t local_mem_ಗಾತ್ರ;
        size_t max_mem_alloc_ಗಾತ್ರ;
        size_t max_nd_range_ಗಾತ್ರ[3];
        ಪೂರ್ಣಾಂಕ max_nd_range_ಗಾತ್ರ_i[3];
        uint32_t _device_id;
        std::array<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ, 16> _uuid;
    };

    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ get_major_ಆವೃತ್ತಿ(ಸ್ಥಿರಾಂಕ sycl::device &dev)
    {
        ಪೂರ್ಣಾಂಕ major, minor;
        detail::get_ಆವೃತ್ತಿ(dev, major, minor);
        ಹಿಂತಿರುಗಿ major;
    }

    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ get_minor_ಆವೃತ್ತಿ(ಸ್ಥಿರಾಂಕ sycl::device &dev)
    {
        ಪೂರ್ಣಾಂಕ major, minor;
        detail::get_ಆವೃತ್ತಿ(dev, major, minor);
        ಹಿಂತಿರುಗಿ minor;
    }

    ಸ್ಥಿರ ಶೂನ್ಯ get_device_info(device_info &ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ sycl::device &dev)
    {
        device_info prop;
        prop.ಗಣ_name(dev.get_info<sycl::info::device::name>().c_str());

        ಪೂರ್ಣಾಂಕ major, minor;
        detail::get_ಆವೃತ್ತಿ(dev, major, minor);
        prop.ಗಣ_major_ಆವೃತ್ತಿ(major);
        prop.ಗಣ_minor_ಆವೃತ್ತಿ(minor);

        prop.ಗಣ_max_work_ಅಂಶ_sizes(
#ಆದರೆ (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_ಅಂಶ_sizes>());
#ಇಲ್ಲದಿದ್ದರೆ
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_ಅಂಶ_sizes<3>>());
#ಆದರೆ_ಅಂತ್ಯ
        prop.ಗಣ_host_unified_ಸ್ಮರಣೆ(dev.has(sycl::aspect::usm_host_allocations));

        prop.ಗಣ_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.ಗಣ_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.ಗಣ_max_work_ಗುಂಪು_ಗಾತ್ರ(
            dev.get_info<sycl::info::device::max_work_ಗುಂಪು_ಗಾತ್ರ>());
        prop.ಗಣ_global_mem_ಗಾತ್ರ(dev.get_info<sycl::info::device::global_mem_ಗಾತ್ರ>());
        prop.ಗಣ_local_mem_ಗಾತ್ರ(dev.get_info<sycl::info::device::local_mem_ಗಾತ್ರ_2>());
        prop.ಗಣ_max_mem_alloc_ಗಾತ್ರ(dev.get_info<sycl::info::device::max_mem_alloc_ಗಾತ್ರ_2>());

#ಆದರೆ (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        ಆದರೆ (dev.has(sycl::aspect::ext_intel_ಸ್ಮರಣೆ_clock_ದರ))
        {
            ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ tmp =
                dev.get_info<sycl::ext::intel::info::device::ಸ್ಮರಣೆ_clock_ದರ_2>();
            ಆದರೆ (tmp != 0)
                prop.ಗಣ_ಸ್ಮರಣೆ_clock_ದರ(1000 * tmp);
        }
        ಆದರೆ (dev.has(sycl::aspect::ext_intel_ಸ್ಮರಣೆ_bus_ಅಗಲ))
        {
            prop.ಗಣ_ಸ್ಮರಣೆ_bus_ಅಗಲ(
                dev.get_info<sycl::ext::intel::info::device::ಸ್ಮರಣೆ_bus_ಅಗಲ_2>());
        }
        ಆದರೆ (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.ಗಣ_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        ಆದರೆ (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.ಗಣ_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#ಅಥವಾಆದರೆ defined(_MSC_VER) && !defined(__clang__)
#ಸೂಚನೆ ಸಂದೇಶ("get_device_info: querying ಸ್ಮರಣೆ_clock_ದರ_2 and \
        ಸ್ಮರಣೆ_bus_ಅಗಲ_2 are not supported by the compiler ಬಳಸಿದ. \
        Use 3200000 kHz as ಸ್ಮರಣೆ_clock_ದರ_2 ಪೂರ್ವನಿಯೋಜಿತ value. \
        Use 64 ಬಿಟ್ಟುಗಳು as ಸ್ಮರಣೆ_bus_ಅಗಲ_2 ಪೂರ್ವನಿಯೋಜಿತ value.")
#ಇಲ್ಲದಿದ್ದರೆ
#warning "get_device_info: querying ಸ್ಮರಣೆ_clock_ದರ_2 and \
        ಸ್ಮರಣೆ_bus_ಅಗಲ_2 are not supported by the compiler ಬಳಸಿದ. \
        Use 3200000 kHz as ಸ್ಮರಣೆ_clock_ದರ_2 ಪೂರ್ವನಿಯೋಜಿತ value. \
        Use 64 ಬಿಟ್ಟುಗಳು as ಸ್ಮರಣೆ_bus_ಅಗಲ_2 ಪೂರ್ವನಿಯೋಜಿತ value."
#ಆದರೆ_ಅಂತ್ಯ

        size_t max_sub_ಗುಂಪು_ಗಾತ್ರ_2 = 1;
        std::vector<size_t> sub_ಗುಂಪು_sizes =
            dev.get_info<sycl::info::device::sub_ಗುಂಪು_sizes>();

        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &sub_ಗುಂಪು_ಗಾತ್ರ : sub_ಗುಂಪು_sizes)
        {
            ಆದರೆ (max_sub_ಗುಂಪು_ಗಾತ್ರ_2 < sub_ಗುಂಪು_ಗಾತ್ರ)
                max_sub_ಗುಂಪು_ಗಾತ್ರ_2 = sub_ಗುಂಪು_ಗಾತ್ರ;
        }

        prop.ಗಣ_max_sub_ಗುಂಪು_ಗಾತ್ರ(max_sub_ಗುಂಪು_ಗಾತ್ರ_2);

        prop.ಗಣ_max_work_ವಸ್ತುಗಳು_ಪ್ರತಿ_compute_unit(
            dev.get_info<sycl::info::device::max_work_ಗುಂಪು_ಗಾತ್ರ>());
        ಪೂರ್ಣಾಂಕ max_nd_range_ಗಾತ್ರ_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.ಗಣ_max_nd_range_ಗಾತ್ರ(max_nd_range_ಗಾತ್ರ_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.ಗಣ_max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು(65536);

        prop.ಗಣ_global_mem_cache_ಗಾತ್ರ(
            dev.get_info<sycl::info::device::global_mem_cache_ಗಾತ್ರ_2>());
        ಹೊರಹೋಗು = prop;
    }

    /// dpct device extension
    ವರ್ಗ device_ext : ಸಾರ್ವಜನಿಕ sycl::device {
      ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ std::mutex mutex_type;

     ಸಾರ್ವಜನಿಕ:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        clear_queues();
      }
      device_ext(ಸ್ಥಿರಾಂಕ sycl::device &ತಳ) : sycl::device(ತಳ) {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        init_queues();
      }

      ಪೂರ್ಣಾಂಕ is_native_atomic_supported() { ಹಿಂತಿರುಗಿ 0; }
      ಪೂರ್ಣಾಂಕ get_major_ಆವೃತ್ತಿ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ dpct::get_major_ಆವೃತ್ತಿ(*ಇದು); }

      ಪೂರ್ಣಾಂಕ get_minor_ಆವೃತ್ತಿ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ dpct::get_minor_ಆವೃತ್ತಿ(*ಇದು); }

      ಪೂರ್ಣಾಂಕ get_max_compute_units() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      ಪೂರ್ಣಾಂಕ get_max_clock_frequency() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_max_clock_frequency();
      }

      ಪೂರ್ಣಾಂಕ get_integrated() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ get_device_info().get_integrated(); }

      ಪೂರ್ಣಾಂಕ get_max_sub_ಗುಂಪು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_max_sub_ಗುಂಪು_ಗಾತ್ರ();
      }

      ಪೂರ್ಣಾಂಕ get_max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_max_ರಿಜಿಸ್ಟರ್_ಗಾತ್ರ_ಪ್ರತಿ_work_ಗುಂಪು();
      }

      ಪೂರ್ಣಾಂಕ get_max_work_ಗುಂಪು_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_max_work_ಗುಂಪು_ಗಾತ್ರ();
      }

      ಪೂರ್ಣಾಂಕ get_mem_ತಳ_addr_align() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_info<sycl::info::device::mem_ತಳ_addr_align>();
      }

      size_t get_global_mem_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_global_mem_ಗಾತ್ರ();
      }

      size_t get_max_mem_alloc_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ {
        ಹಿಂತಿರುಗಿ get_device_info().get_max_mem_alloc_ಗಾತ್ರ();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      ಶೂನ್ಯ get_ಸ್ಮರಣೆ_info(size_t &free_ಸ್ಮರಣೆ, size_t &ಒಟ್ಟು_ಸ್ಮರಣೆ) {
        ಒಟ್ಟು_ಸ್ಮರಣೆ = get_device_info().get_global_mem_ಗಾತ್ರ();
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#ಆದರೆ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        ಆದರೆ (!has(sycl::aspect::ext_intel_free_ಸ್ಮರಣೆ)) {
          std::cerr << warning_info << std::ಸಾಲು_ಅಂತ್ಯ;
          free_ಸ್ಮರಣೆ = ಒಟ್ಟು_ಸ್ಮರಣೆ;
        } ಇಲ್ಲದಿದ್ದರೆ {
          free_ಸ್ಮರಣೆ = get_info<sycl::ext::intel::info::device::free_ಸ್ಮರಣೆ>();
        }
#ಇಲ್ಲದಿದ್ದರೆ
        std::cerr << warning_info << std::ಸಾಲು_ಅಂತ್ಯ;
        free_ಸ್ಮರಣೆ = ಒಟ್ಟು_ಸ್ಮರಣೆ;
#ಆದರೆ defined(_MSC_VER) && !defined(__clang__)
#ಸೂಚನೆ ಸಂದೇಶ("Querying the number of bytes of free memory is not supported")
#ಇಲ್ಲದಿದ್ದರೆ
#warning "Querying the number of bytes of free memory is not supported"
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ
      }

      ಶೂನ್ಯ get_device_info(device_info &ಹೊರಹೋಗು) ಸ್ಥಿರಾಂಕ {
        dpct::get_device_info(ಹೊರಹೋಗು, *ಇದು);
      }

      device_info get_device_info() ಸ್ಥಿರಾಂಕ {
        device_info prop;
        dpct::get_device_info(prop, *ಇದು);
        ಹಿಂತಿರುಗಿ prop;
      }

      ಶೂನ್ಯ reset() {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &ಒಳ_ಕ್ರಮ_ಸರತಿ() { ಹಿಂತಿರುಗಿ q_ಒಳ_ಕ್ರಮ; }

      sycl::queue &ಹೊರಹೋಗು_ದ_ಕ್ರಮ_ಸರತಿ() { ಹಿಂತಿರುಗಿ q_ಹೊರಹೋಗು_ದ_ಕ್ರಮ; }

      sycl::queue &default_ಸರತಿ() { ಹಿಂತಿರುಗಿ ಒಳ_ಕ್ರಮ_ಸರತಿ(); }

      ಶೂನ್ಯ queues_wait_and_throw() {
        std::unique_lock<mutex_type> ಬೀಗ(m_mutex);
        ಬೀಗ.unlock();
        ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        ಬೀಗ.ಬೀಗ();
      }

      sycl::queue create_ಸರತಿ(ತಾರ್ಕಿಕ enable_exception_handler = ಸುಳ್ಳು) {
        ಹಿಂತಿರುಗಿ create_ಒಳ_ಕ್ರಮ_ಸರತಿ(enable_exception_handler);
      }

      sycl::queue create_ಸರತಿ(sycl::device device,
                               ತಾರ್ಕಿಕ enable_exception_handler = ಸುಳ್ಳು) {
        ಹಿಂತಿರುಗಿ create_ಒಳ_ಕ್ರಮ_ಸರತಿ(device, enable_exception_handler);
      }

      sycl::queue create_ಒಳ_ಕ್ರಮ_ಸರತಿ(ತಾರ್ಕಿಕ enable_exception_handler = ಸುಳ್ಳು) {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        ಹಿಂತಿರುಗಿ create_ಸರತಿ_impl(enable_exception_handler,
                                 sycl::property::queue::ಒಳ_ಕ್ರಮ());
      }

      sycl::queue create_ಒಳ_ಕ್ರಮ_ಸರತಿ(sycl::device device,
                                        ತಾರ್ಕಿಕ enable_exception_handler = ಸುಳ್ಳು) {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        ಹಿಂತಿರುಗಿ create_ಸರತಿ_impl(device, enable_exception_handler,
                                 sycl::property::queue::ಒಳ_ಕ್ರಮ());
      }

      sycl::queue create_ಹೊರಹೋಗು_ದ_ಕ್ರಮ_ಸರತಿ(
          ತಾರ್ಕಿಕ enable_exception_handler = ಸುಳ್ಳು) {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        ಹಿಂತಿರುಗಿ create_ಸರತಿ_impl(enable_exception_handler);
      }

      ಶೂನ್ಯ destroy_ಸರತಿ(sycl::queue queue) {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ಸ್ಥಿರಾಂಕ sycl::queue &q) -> ತಾರ್ಕಿಕ
                                    {
                                        ಹಿಂತಿರುಗಿ q == queue;
                                    }),
                    _queues.end());
      }
      ಶೂನ್ಯ ಗಣ_saved_ಸರತಿ(sycl::queue q) {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        saved_ಸರತಿ = q;
      }
      sycl::queue get_saved_ಸರತಿ() ಸ್ಥಿರಾಂಕ {
        std::lock_guard<mutex_type> ಬೀಗ(m_mutex);
        ಹಿಂತಿರುಗಿ saved_ಸರತಿ;
      }

     ಖಾಸಗಿ:
      ಶೂನ್ಯ clear_queues() { _queues.clear(); }

      ಶೂನ್ಯ init_queues() {
        q_ಒಳ_ಕ್ರಮ =
            create_ಸರತಿ_impl(ನಿಜ, sycl::property::queue::ಒಳ_ಕ್ರಮ());
        q_ಹೊರಹೋಗು_ದ_ಕ್ರಮ = create_ಸರತಿ_impl(ನಿಜ);
        saved_ಸರತಿ = default_ಸರತಿ();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      ಮಾದರಿ <ವರ್ಗ... Properties>
      sycl::queue create_ಸರತಿ_impl(ತಾರ್ಕಿಕ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ಆದರೆ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *ಇದು, eh,
            sycl::property_list(
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#ಆದರೆ_ಅಂತ್ಯ
                properties...)));

        ಹಿಂತಿರುಗಿ _queues.back();
      }

      ಮಾದರಿ <ವರ್ಗ... Properties>
      sycl::queue create_ಸರತಿ_impl(sycl::device device,
                                    ತಾರ್ಕಿಕ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ಆದರೆ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#ಆದರೆ_ಅಂತ್ಯ
                            properties...)));

        ಹಿಂತಿರುಗಿ _queues.back();
      }

      ಶೂನ್ಯ get_ಆವೃತ್ತಿ(ಪೂರ್ಣಾಂಕ &major, ಪೂರ್ಣಾಂಕ &minor) ಸ್ಥಿರಾಂಕ {
        detail::get_ಆವೃತ್ತಿ(*ಇದು, major, minor);
      }
      sycl::queue q_ಒಳ_ಕ್ರಮ, q_ಹೊರಹೋಗು_ದ_ಕ್ರಮ;
      sycl::queue saved_ಸರತಿ;
      std::vector<sycl::queue> _queues;
      ಬದಲಾಯಿಸಬಹುದಾದ mutex_type m_mutex;
    };


    /// device manager
    ವರ್ಗ dev_mgr
    {
    ಸಾರ್ವಜನಿಕ:
        device_ext &current_device()
        {
            ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ dev_id = current_device_id();
            check_id(dev_id);
            ಹಿಂತಿರುಗಿ *_devs[dev_id];
        }
        device_ext &cpu_device() ಸ್ಥಿರಾಂಕ
        {
            std::lock_guard<std::recursive_mutex> ಬೀಗ(m_mutex);
            ಆದರೆ (_cpu_device == -1)
            {
                ಎಸೆ std::runtime_error("no valid cpu device");
            }
            ಇಲ್ಲದಿದ್ದರೆ
            {
                ಹಿಂತಿರುಗಿ *_devs[_cpu_device];
            }
        }
        device_ext &get_device(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id) ಸ್ಥಿರಾಂಕ
        {
            std::lock_guard<std::recursive_mutex> ಬೀಗ(m_mutex);
            check_id(id);
            ಹಿಂತಿರುಗಿ *_devs[id];
        }
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ current_device_id() ಸ್ಥಿರಾಂಕ
        {
            std::lock_guard<std::recursive_mutex> ಬೀಗ(m_mutex);
            ಸ್ವಯಂಚಾಲಿತ it = _thread2dev_map.find(get_tid());
            ಆದರೆ (it != _thread2dev_map.end())
                ಹಿಂತಿರುಗಿ it->second;
            ಹಿಂತಿರುಗಿ DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        ಶೂನ್ಯ select_device(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id)
        {
            std::lock_guard<std::recursive_mutex> ಬೀಗ(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ device_ಎಣಿಕೆ() { ಹಿಂತಿರುಗಿ _devs.size(); }

        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ get_device_id(ಸ್ಥಿರಾಂಕ sycl::device &dev)
        {
            ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id = 0;
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &dev_ಅಂಶ : _devs)
            {
                ಆದರೆ (*dev_ಅಂಶ == dev)
                {
                    ಹಿಂತಿರುಗಿ id;
                }
                id++;
            }
            ಹಿಂತಿರುಗಿ -1;
        }

        ಸಾಲೊಳಗೆ std::string get_preferred_gpu_platform_name() {
            std::string ಫಲಿತಾಂಶ;

            std::string filter = "";
            ಅಕ್ಷರ* env = getenv("ONEAPI_DEVICE_SELECTOR");
            ಆದರೆ (env) {
                ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(env, "level_zero")) {
                    filter = "level-zero";
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(env, "opencl")) {
                    filter = "opencl";
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(env, "cuda")) {
                    filter = "cuda";
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(env, "hip")) {
                    filter = "hip";
                }
                ಇಲ್ಲದಿದ್ದರೆ {
                    ಎಸೆ std::runtime_error("invalid device filter: " + std::string(env));
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ವಯಂಚಾಲಿತ default_device = sycl::device(sycl::default_selector_v);
                ಸ್ವಯಂಚಾಲಿತ default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (std::ಅಕ್ಷರಸರಣಿ_ಹುಡುಕು(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            ಸ್ವಯಂಚಾಲಿತ platform_list = sycl::platform::get_platforms();

            ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ& platform : platform_list) {
                ಸ್ವಯಂಚಾಲಿತ devices = platform.get_devices();
                ಸ್ವಯಂಚಾಲಿತ gpu_dev = std::find_if(devices.begin(), devices.end(), [](ಸ್ಥಿರಾಂಕ sycl::device& d) {
                    ಹಿಂತಿರುಗಿ d.is_gpu();
                });

                ಆದರೆ (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    ಮುಂದುವರಿ;
                }

                ಸ್ವಯಂಚಾಲಿತ platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_ಕೆಳಗಿನ_case;
                platform_name_ಕೆಳಗಿನ_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_ಕೆಳಗಿನ_case.begin(), ::tolower);

                ಆದರೆ (platform_name_ಕೆಳಗಿನ_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    ಮುಂದುವರಿ;
                }

                ಫಲಿತಾಂಶ = platform_name;
            }

            ಆದರೆ (ಫಲಿತಾಂಶ.ಖಾಲಿಗಳು())
                ಎಸೆ std::runtime_error("can not find preferred GPU platform");

            ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
        }

        ಮಾದರಿ <ವರ್ಗ DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<ಪೂರ್ಣಾಂಕ, DeviceSelector, ಸ್ಥಿರಾಂಕ sycl::device &>>
        select_device(ಸ್ಥಿರಾಂಕ DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ಸ್ಥಿರ dev_mgr &instance()
        {
            ಸ್ಥಿರ dev_mgr d_m;
            ಹಿಂತಿರುಗಿ d_m;
        }
        dev_mgr(ಸ್ಥಿರಾಂಕ dev_mgr &) = ಅಳಿಸು;
        dev_mgr &ಕ್ರಿಯಾಕಾರಕ=(ಸ್ಥಿರಾಂಕ dev_mgr &) = ಅಳಿಸು;
        dev_mgr(dev_mgr &&) = ಅಳಿಸು;
        dev_mgr &ಕ್ರಿಯಾಕಾರಕ=(dev_mgr &&) = ಅಳಿಸು;

    ಖಾಸಗಿ:
        ಬದಲಾಯಿಸಬಹುದಾದ std::recursive_mutex m_mutex;
        ಸ್ಥಿರ ತಾರ್ಕಿಕ compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            ಆದರೆ(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) ಹಿಂತಿರುಗಿ ನಿಜ;
            ಆದರೆ(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            ಹಿಂತಿರುಗಿ prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ convert_backend_ಸೂಚ್ಯಂಕ(std::string & backend) {
            ಆದರೆ (backend == "ext_oneapi_level_zero:gpu") ಹಿಂತಿರುಗಿ 0;
            ಆದರೆ (backend == "opencl:gpu") ಹಿಂತಿರುಗಿ 1;
            ಆದರೆ (backend == "ext_oneapi_cuda:gpu") ಹಿಂತಿರುಗಿ 2;
            ಆದರೆ (backend == "ext_oneapi_hip:gpu") ಹಿಂತಿರುಗಿ 3;
            ಆದರೆ (backend == "opencl:cpu") ಹಿಂತಿರುಗಿ 4;
            ಆದರೆ (backend == "opencl:acc") ಹಿಂತಿರುಗಿ 5;
            ಬರೆ("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ಸ್ಥಿರ ತಾರ್ಕಿಕ compare_backend(std::string &backend1, std::string &backend2) {
            ಹಿಂತಿರುಗಿ convert_backend_ಸೂಚ್ಯಂಕ(backend1) < convert_backend_ಸೂಚ್ಯಂಕ(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            ಆದರೆ (default_device.is_cpu())
                _cpu_device = 0;

            ಸ್ವಯಂಚಾಲಿತ Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            ಸ್ವಯಂಚಾಲಿತ preferred_platform_name = get_preferred_gpu_platform_name();

            ಆಗ (!Platforms.ಖಾಲಿಗಳು()) {
                ಸ್ವಯಂಚಾಲಿತ Platform = Platforms.back();
                Platforms.pop_back();
                ಸ್ವಯಂಚಾಲಿತ platform_name = Platform.get_info<sycl::info::platform::name>();
                ಆದರೆ (platform_name.compare(preferred_platform_name) != 0) {
                    ಮುಂದುವರಿ;
                }
                ಸ್ವಯಂಚಾಲಿತ devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> ಕೀಲಿಗಳು;
            ಗಾಗಿ(ಸ್ವಯಂಚಾಲಿತ it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                ಕೀಲಿಗಳು.push_back(it->ಮೊದಲ);
            }
            std::sort(ಕೀಲಿಗಳು.begin(), ಕೀಲಿಗಳು.end(), compare_backend);

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &ಕೀಲಿ : ಕೀಲಿಗಳು) {
                std::vector<sycl::device> devs = backend_devices[ಕೀಲಿ];
                std::sort(devs.begin(), devs.end(), compare_dev);
                ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ &dev : sycl_all_devs)
            {
                ಆದರೆ (dev == default_device)
                {
                    ಮುಂದುವರಿ;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                ಆದರೆ (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        ಶೂನ್ಯ check_id(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id) ಸ್ಥಿರಾಂಕ
        {
            ಆದರೆ (id >= _devs.size())
            {
                ಎಸೆ std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ> _thread2dev_map;
        ಪೂರ್ಣಾಂಕ _cpu_device = -1;
    };

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ sycl::queue &get_default_ಸರತಿ()
    {
        ಹಿಂತಿರುಗಿ dev_mgr::instance().current_device().default_ಸರತಿ();
    }

    ನಾಮಾಕಾಶ detail
    {
        ಎಣಿಕೆ ವರ್ಗ ಸೂಚಕ_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ಸ್ಥಿರ ಸೂಚಕ_access_attribute get_ಸೂಚಕ_attribute(sycl::queue &q,
                                                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr)
        {
            ಆಯ್ಕೆ (sycl::get_ಸೂಚಕ_type(ptr, q.get_context()))
            {
            ಸ್ಥಿತಿ sycl::usm::alloc::unknown:
                ಹಿಂತಿರುಗಿ ಸೂಚಕ_access_attribute::host_only;
            ಸ್ಥಿತಿ sycl::usm::alloc::device:
                ಹಿಂತಿರುಗಿ ಸೂಚಕ_access_attribute::device_only;
            ಸ್ಥಿತಿ sycl::usm::alloc::shared:
            ಸ್ಥಿತಿ sycl::usm::alloc::host:
                ಹಿಂತಿರುಗಿ ಸೂಚಕ_access_attribute::host_device;
            }
        }

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ArgT>
        ಸಾಲೊಳಗೆ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ)library_ದತ್ತಾಂಶ_t::library_ದತ್ತಾಂಶ_t_ಗಾತ್ರ <=
                              std::numeric_limits<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_ದತ್ತಾಂಶ_t>, "Unsupported ArgT");
            ಹಿಂತಿರುಗಿ (std::uint64_t)Val;
        }

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಮೊದಲ_T, ಬಗೆ_ಹೆಸರು... RestT>
        ಸಾಲೊಳಗೆ constexpr std::uint64_t get_type_combination_id(ಮೊದಲ_T ಮೊದಲ_Val,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_ದತ್ತಾಂಶ_t::library_ದತ್ತಾಂಶ_t_ಗಾತ್ರ <=
                              std::numeric_limits<ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(ಗಾತ್ರ...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<ಮೊದಲ_T, library_ದತ್ತಾಂಶ_t>, "Unsupported FirstT");
            ಹಿಂತಿರುಗಿ get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)ಮೊದಲ_Val);
        }

        ವರ್ಗ mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#ಆದರೆ defined(__linux__)
                mapped_ವಿಳಾಸ_space =
                    (byte_t *)mmap(ಶೂನ್ಯಸೂಚಕ, mapped_region_ಗಾತ್ರ, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ಅಥವಾಆದರೆ defined(_WIN64)
                mapped_ವಿಳಾಸ_space = (byte_t *)VirtualAlloc(
                    ಖಾಲಿ,               // NULL specified as the base address parameter
                    mapped_region_ಗಾತ್ರ, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    ಪುಟ_NOACCESS);     // Protection = no access
#ಇಲ್ಲದಿದ್ದರೆ
#ದೋಷ "Only support Windows and Linux."
#ಆದರೆ_ಅಂತ್ಯ
                ಮುಂದಿನ_free = mapped_ವಿಳಾಸ_space;
            }

        ಸಾರ್ವಜನಿಕ:
            ಬಳಸಿ ಬಫರ್_id_t = ಪೂರ್ಣಾಂಕ;

            ರಚನೆ ಹಂಚಿಕೆ
            {
                ಬಫರ್_t ಬಫರ್;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#ಆದರೆ defined(__linux__)
                munmap(mapped_ವಿಳಾಸ_space, mapped_region_ಗಾತ್ರ);
#ಅಥವಾಆದರೆ defined(_WIN64)
                VirtualFree(mapped_ವಿಳಾಸ_space, 0, MEM_RELEASE);
#ಇಲ್ಲದಿದ್ದರೆ
#ದೋಷ "Only support Windows and Linux."
#ಆದರೆ_ಅಂತ್ಯ
            }

            mem_mgr(ಸ್ಥಿರಾಂಕ mem_mgr &) = ಅಳಿಸು;
            mem_mgr &ಕ್ರಿಯಾಕಾರಕ=(ಸ್ಥಿರಾಂಕ mem_mgr &) = ಅಳಿಸು;
            mem_mgr(mem_mgr &&) = ಅಳಿಸು;
            mem_mgr &ಕ್ರಿಯಾಕಾರಕ=(mem_mgr &&) = ಅಳಿಸು;

            /// Allocate
            ಶೂನ್ಯ *mem_alloc(size_t size)
            {
                ಆದರೆ (!size)
                    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
                std::lock_guard<std::mutex> ಬೀಗ(m_mutex);
                ಆದರೆ (ಮುಂದಿನ_free + size > mapped_ವಿಳಾಸ_space + mapped_region_ಗಾತ್ರ)
                {
                    ಎಸೆ std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                //ಹಂಚಿಕೆ
                sycl::range<1> r(size);
                ಬಫರ್_t buf(r);
                ಹಂಚಿಕೆ A{buf, ಮುಂದಿನ_free, size};
                // Map allocation to device pointer
                ಶೂನ್ಯ *ಫಲಿತಾಂಶ = ಮುಂದಿನ_free;
                m_map.emplace(ಮುಂದಿನ_free + size, A);
                // Update pointer to the next free space.
                ಮುಂದಿನ_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
            }

            /// Deallocate
            ಶೂನ್ಯ mem_free(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr)
            {
                ಆದರೆ (!ptr)
                    ಹಿಂತಿರುಗಿ;
                std::lock_guard<std::mutex> ಬೀಗ(m_mutex);
                ಸ್ವಯಂಚಾಲಿತ it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            ಹಂಚಿಕೆ translate_ptr(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr)
            {
                std::lock_guard<std::mutex> ಬೀಗ(m_mutex);
                ಸ್ವಯಂಚಾಲಿತ it = get_map_iterator(ptr);
                ಹಿಂತಿರುಗಿ it->second;
            }

            /// Check if the pointer represents device pointer or not.
            ತಾರ್ಕಿಕ is_device_ptr(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr) ಸ್ಥಿರಾಂಕ
            {
                std::lock_guard<std::mutex> ಬೀಗ(m_mutex);
                ಹಿಂತಿರುಗಿ (mapped_ವಿಳಾಸ_space <= ptr) &&
                       (ptr < mapped_ವಿಳಾಸ_space + mapped_region_ಗಾತ್ರ);
            }

            /// Returns the instance of memory manager singleton.
            ಸ್ಥಿರ mem_mgr &instance()
            {
                ಸ್ಥಿರ mem_mgr m;
                ಹಿಂತಿರುಗಿ m;
            }

        ಖಾಸಗಿ:
            std::map<byte_t *, ಹಂಚಿಕೆ> m_map;
            ಬದಲಾಯಿಸಬಹುದಾದ std::mutex m_mutex;
            byte_t *mapped_ವಿಳಾಸ_space;
            byte_t *ಮುಂದಿನ_free;
            ಸ್ಥಿರಾಂಕ size_t mapped_region_ಗಾತ್ರ = 128ull * 1024 * 1024 * 1024;
            ಸ್ಥಿರಾಂಕ size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ಸ್ಥಿರಾಂಕ size_t extra_padding = 0;

            std::map<byte_t *, ಹಂಚಿಕೆ>::iterator get_map_iterator(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *ptr)
            {
                ಸ್ವಯಂಚಾಲಿತ it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ಸ್ಥಿರಾಂಕ byte_t *>(ptr)));
                ಆದರೆ (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ಎಸೆ std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ಸ್ಥಿರಾಂಕ ಹಂಚಿಕೆ &alloc = it->second;
                ಆದರೆ (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ಎಸೆ std::runtime_error("invalid virtual pointer");
                }
                ಹಿಂತಿರುಗಿ it;
            }
        };

        ಮಾದರಿ <ವರ್ಗ T, ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2, size_t Dimension>
        ವರ್ಗ accessor;
        ಮಾದರಿ <ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2, ವರ್ಗ T = byte_t>
        ವರ್ಗ ಸ್ಮರಣೆ_traits
        {
        ಸಾರ್ವಜನಿಕ:
            ಸ್ಥಿರ constexpr sycl::access::ಗುರಿ ಗುರಿ =
                sycl::access::ಗುರಿ::device;
            ಸ್ಥಿರ constexpr sycl::access_ಬಹುಲಕ mode =
                (ಸ್ಮರಣೆ_2 == constant) ? sycl::access_ಬಹುಲಕ::ಪಠ
                                     : sycl::access_ಬಹುಲಕ::read_write;
            ಸ್ಥಿರ constexpr size_t type_ಗಾತ್ರ = ಗಾತ್ರ(T);
            ಬಳಸಿ element_t =
                ಬಗೆ_ಹೆಸರು std::conditional<ಸ್ಮರಣೆ_2 == constant, ಸ್ಥಿರಾಂಕ T, T>::type;
            ಬಳಸಿ ಮೌಲ್ಯ_t = ಬಗೆ_ಹೆಸರು std::remove_cv<T>::type;
            ಮಾದರಿ <size_t Dimension = 1>
            ಬಳಸಿ accessor_t = ಬಗೆ_ಹೆಸರು std::conditional<
                ಸ್ಮರಣೆ_2 == local, sycl::local_accessor<ಮೌಲ್ಯ_t, Dimension>,
                sycl::accessor<T, Dimension, mode, ಗುರಿ>>::type;
            ಬಳಸಿ ಸೂಚಕ_t = T *;
        };

        ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ *dpct_malloc(size_t size, sycl::queue &q)
        {
            ಹಿಂತಿರುಗಿ sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#ವ್ಯಾಖ್ಯಾನಿಸು PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            ಹಿಂತಿರುಗಿ dpct_malloc(pitch * y * z, q);
        }

        /**
         * @brief Sets \p value to the first \p size elements starting from \p dev_ptr in \p q.
         * @tparam valueT The type of the element to be set.
         * @param [in] q The queue in which the operation is done.
         * @param [in] dev_ptr Pointer to the virtual device memory address.
         * @param [in] value The value to be set.
         * @param [in] size Number of elements to be set to the value.
         * @return An event representing the memset operation.
         */
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಮೌಲ್ಯ_T>
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ sycl::event dpct_memset(sycl::queue &q, ಶೂನ್ಯ *dev_ptr,
                                              ಮೌಲ್ಯ_T value, size_t size)
        {
            ಹಿಂತಿರುಗಿ q.fill(dev_ptr, value, size);
        }

        /**
         * @brief Sets \p value to the 3D memory region pointed by \p data in \p q.
         * @tparam valueT The type of the element to be set.
         * @param [in] q The queue in which the operation is done.
         * @param [in] data Pointer to the pitched device memory region.
         * @param [in] value The value to be set.
         * @param [in] size 3D memory region by number of elements.
         * @return An event list representing the memset operations.
         */
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಮೌಲ್ಯ_T>
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_ದತ್ತಾಂಶ ದತ್ತಾಂಶ, ಮೌಲ್ಯ_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t ತುಂಡು_2 = ದತ್ತಾಂಶ.get_pitch() * ದತ್ತಾಂಶ.get_y();
            ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *ದತ್ತಾಂಶ_surface = (ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *)ದತ್ತಾಂಶ.get_ದತ್ತಾಂಶ_ptr();
            ಗಾಗಿ (size_t z = 0; z < size.get(2); ++z)
            {
                ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *ದತ್ತಾಂಶ_ptr = ದತ್ತಾಂಶ_surface;
                ಗಾಗಿ (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, ದತ್ತಾಂಶ_ptr, value, size.get(0)));
                    ದತ್ತಾಂಶ_ptr += ದತ್ತಾಂಶ.get_pitch();
                }
                ದತ್ತಾಂಶ_surface += ತುಂಡು_2;
            }
            ಹಿಂತಿರುಗಿ event_list;
        }

        /**
         * @brief Sets \p val to the pitched 2D memory region pointed by \p ptr in \p q.
         * @tparam valueT The type of the element to be set.
         * @param [in] q The queue in which the operation is done.
         * @param [in] ptr Pointer to the virtual device memory.
         * @param [in] pitch The pitch size by number of elements, including padding.
         * @param [in] val The value to be set.
         * @param [in] x The width of memory region by number of elements.
         * @param [in] y The height of memory region by number of elements.
         * @return An event list representing the memset operations.
         */
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು ಮೌಲ್ಯ_T>
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, ಶೂನ್ಯ *ptr, size_t pitch, ಮೌಲ್ಯ_T val, size_t x,
                    size_t y)
        {
            ಹಿಂತಿರುಗಿ dpct_memset(q, pitched_ದತ್ತಾಂಶ(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ಸ್ಥಿರ memcpy_direction deduce_memcpy_direction(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr,
                                                        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr,
                                                        memcpy_direction dir)
        {
            ಆಯ್ಕೆ (dir)
            {
            ಸ್ಥಿತಿ memcpy_direction::host_ಗಮ್ಯ_host:
            ಸ್ಥಿತಿ memcpy_direction::host_ಗಮ್ಯ_device:
            ಸ್ಥಿತಿ memcpy_direction::device_ಗಮ್ಯ_host:
            ಸ್ಥಿತಿ memcpy_direction::device_ಗಮ್ಯ_device:
                ಹಿಂತಿರುಗಿ dir;
            ಸ್ಥಿತಿ memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ memcpy_direction
                    direction_ಕೋಷ್ಟಕ[static_cast<ಚಿಹ್ನೆರಹಿತ>(ಸೂಚಕ_access_attribute::end)]
                                   [static_cast<ಚಿಹ್ನೆರಹಿತ>(ಸೂಚಕ_access_attribute::end)] =
                                       {{memcpy_direction::host_ಗಮ್ಯ_host,
                                         memcpy_direction::device_ಗಮ್ಯ_host,
                                         memcpy_direction::host_ಗಮ್ಯ_host},
                                        {memcpy_direction::host_ಗಮ್ಯ_device,
                                         memcpy_direction::device_ಗಮ್ಯ_device,
                                         memcpy_direction::device_ಗಮ್ಯ_device},
                                        {memcpy_direction::host_ಗಮ್ಯ_host,
                                         memcpy_direction::device_ಗಮ್ಯ_device,
                                         memcpy_direction::device_ಗಮ್ಯ_device}};
                ಹಿಂತಿರುಗಿ direction_ಕೋಷ್ಟಕ[static_cast<ಚಿಹ್ನೆರಹಿತ>(get_ಸೂಚಕ_attribute(
                    q, ಗಮ್ಯ_ptr))][static_cast<ಚಿಹ್ನೆರಹಿತ>(get_ಸೂಚಕ_attribute(q, from_ptr))];
            }
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಎಸೆ std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ಸ್ಥಿರ sycl::event
        dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr, size_t size,
                    memcpy_direction direction,
                    ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &dep_events = {})
        {
            ಆದರೆ (!size)
                ಹಿಂತಿರುಗಿ sycl::event{};
            ಹಿಂತಿರುಗಿ q.ಸ್ಮರಣೆ_ನಕಲು(ಗಮ್ಯ_ptr, from_ptr, size, dep_events);
            DPCT_ಬಳಸದ(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t get_copy_range(sycl::range<3> size, size_t ತುಂಡು_2,
                                            size_t pitch)
        {
            ಹಿಂತಿರುಗಿ ತುಂಡು_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t get_ಸ್ಥಾನಾಂತರ(sycl::id<3> id, size_t ತುಂಡು_2,
                                        size_t pitch)
        {
            ಹಿಂತಿರುಗಿ ತುಂಡು_2 * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr,
                    sycl::range<3> ಗಮ್ಯ_range, sycl::range<3> from_range,
                    sycl::id<3> ಗಮ್ಯ_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            ವರ್ಗ host_ಬಫರ್
            {
                ಶೂನ್ಯ *_buf;
                size_t ಗಾತ್ರ_3;
                sycl::queue &_q;
                ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &_deps; // free operation depends

            ಸಾರ್ವಜನಿಕ:
                host_ಬಫರ್(size_t size, sycl::queue &q,
                            ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &deps)
                    : _buf(std::ಸ್ಮರಣೆ_ಕೊಡು(size)), ಗಾತ್ರ_3(size), _q(q), _deps(deps) {}
                ಶೂನ್ಯ *get_ptr() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _buf; }
                size_t get_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಗಾತ್ರ_3; }
                ~host_ಬಫರ್()
                {
                    ಆದರೆ (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::ಬಿಡುಗಡೆ(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t ಗಮ್ಯ_ತುಂಡು = ಗಮ್ಯ_range.get(1) * ಗಮ್ಯ_range.get(0),
                   from_ತುಂಡು = from_range.get(1) * from_range.get(0);
            ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *ಗಮ್ಯ_surface =
                (ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *)ಗಮ್ಯ_ptr + get_ಸ್ಥಾನಾಂತರ(ಗಮ್ಯ_id, ಗಮ್ಯ_ತುಂಡು, ಗಮ್ಯ_range.get(0));
            ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *from_surface =
                (ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *)from_ptr +
                get_ಸ್ಥಾನಾಂತರ(from_id, from_ತುಂಡು, from_range.get(0));

            ಆದರೆ (ಗಮ್ಯ_ತುಂಡು == from_ತುಂಡು && ಗಮ್ಯ_ತುಂಡು == size.get(1) * size.get(0))
            {
                ಹಿಂತಿರುಗಿ {dpct_memcpy(q, ಗಮ್ಯ_surface, from_surface, ಗಮ್ಯ_ತುಂಡು * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, ಗಮ್ಯ_ptr, from_ptr, direction);
            size_t ಗಾತ್ರ_ತುಂಡು = size.get(1) * size.get(0);
            ಆಯ್ಕೆ (direction)
            {
            ಸ್ಥಿತಿ host_ಗಮ್ಯ_host:
                ಗಾಗಿ (size_t z = 0; z < size.get(2); ++z)
                {
                    ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *ಗಮ್ಯ_ptr = ಗಮ್ಯ_surface;
                    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *from_ptr = from_surface;
                    ಆದರೆ (ಗಮ್ಯ_range.get(0) == from_range.get(0) &&
                        ಗಮ್ಯ_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, ಗಾತ್ರ_ತುಂಡು,
                                                         direction, dep_events));
                    }
                    ಇಲ್ಲದಿದ್ದರೆ
                    {
                        ಗಾಗಿ (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            ಗಮ್ಯ_ptr += ಗಮ್ಯ_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    ಗಮ್ಯ_surface += ಗಮ್ಯ_ತುಂಡು;
                    from_surface += from_ತುಂಡು;
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ host_ಗಮ್ಯ_device:
            {
                host_ಬಫರ್ buf(get_copy_range(size, ಗಮ್ಯ_ತುಂಡು, ಗಮ್ಯ_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                ಆದರೆ (ಗಮ್ಯ_ತುಂಡು == ಗಾತ್ರ_ತುಂಡು)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, ಗಮ್ಯ_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_ಗಮ್ಯ_host, dep_events);
                }
                ಇಲ್ಲದಿದ್ದರೆ
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, ಗಮ್ಯ_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_ಗಮ್ಯ_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), ಗಮ್ಯ_surface, buf.get_ಗಾತ್ರ(),
                                        device_ಗಮ್ಯ_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, ಗಮ್ಯ_surface, buf.get_ptr(),
                                                 buf.get_ಗಾತ್ರ(), host_ಗಮ್ಯ_device,
                                                 host_events));
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ device_ಗಮ್ಯ_host:
            {
                host_ಬಫರ್ buf(get_copy_range(size, from_ತುಂಡು, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, ಗಮ್ಯ_surface, buf.get_ptr(), ಗಮ್ಯ_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_ಗಮ್ಯ_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_ಗಾತ್ರ(),
                                                         device_ಗಮ್ಯ_host, dep_events)});
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ device_ಗಮ್ಯ_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<ವರ್ಗ dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        ಗಮ್ಯ_surface[get_ಸ್ಥಾನಾಂತರ(id, ಗಮ್ಯ_ತುಂಡು, ಗಮ್ಯ_range.get(0))] =
                            from_surface[get_ಸ್ಥಾನಾಂತರ(id, from_ತುಂಡು, from_range.get(0))];
                    }); }));
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                ಎಸೆ std::runtime_error("dpct_memcpy: invalid direction value");
            }
            ಹಿಂತಿರುಗಿ event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_ದತ್ತಾಂಶ ಗಮ್ಯ, sycl::id<3> ಗಮ್ಯ_id,
                    pitched_ದತ್ತಾಂಶ from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            ಹಿಂತಿರುಗಿ dpct_memcpy(q, ಗಮ್ಯ.get_ದತ್ತಾಂಶ_ptr(), from.get_ದತ್ತಾಂಶ_ptr(),
                               sycl::range<3>(ಗಮ್ಯ.get_pitch(), ಗಮ್ಯ.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), ಗಮ್ಯ_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr,
                    size_t ಗಮ್ಯ_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            ಹಿಂತಿರುಗಿ dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, sycl::range<3>(ಗಮ್ಯ_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        ನಾಮಾಕಾಶ deprecated
        {

            ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, sycl::usm::alloc Alloc_ಬಗೆ>
            ವರ್ಗ usm_allocator
            {
            ಖಾಸಗಿ:
                ಬಳಸಿ Alloc = sycl::usm_allocator<T, Alloc_ಬಗೆ>;
                Alloc _impl;

            ಸಾರ್ವಜನಿಕ:
                ಬಳಸಿ value_type = ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::value_type;
                ಬಳಸಿ pointer = ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::pointer;
                ಬಳಸಿ const_pointer = ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::const_pointer;
                ಬಳಸಿ void_pointer = ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::void_pointer;
                ಬಳಸಿ const_void_pointer =
                    ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::const_void_pointer;
                ಬಳಸಿ reference = ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::value_type &;
                ಬಳಸಿ const_reference =
                    ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::value_type &;
                ಬಳಸಿ difference_type =
                    ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::difference_type;
                ಬಳಸಿ size_type = ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::size_type;
                ಬಳಸಿ propagate_on_container_copy_assignment = ಬಗೆ_ಹೆಸರು std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                ಬಳಸಿ propagate_on_container_move_assignment = ಬಗೆ_ಹೆಸರು std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                ಬಳಸಿ propagate_on_container_swap =
                    ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::propagate_on_container_swap;
                ಬಳಸಿ is_always_equal =
                    ಬಗೆ_ಹೆಸರು std::allocator_traits<Alloc>::is_always_equal;

                ಮಾದರಿ <ಬಗೆ_ಹೆಸರು U>
                ರಚನೆ rebind
                {
                    ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ usm_allocator<U, Alloc_ಬಗೆ> other;
                };

                usm_allocator() : _impl(dpct::get_default_ಸರತಿ()) {}
                ~usm_allocator() {}
                usm_allocator(ಸ್ಥಿರಾಂಕ usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { ಹಿಂತಿರುಗಿ &r; }
                const_pointer address(const_reference r) { ಹಿಂತಿರುಗಿ &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = ಶೂನ್ಯಸೂಚಕ)
                {
                    ಹಿಂತಿರುಗಿ std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                ಶೂನ್ಯ deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ಸ್ಥಿರಾಂಕ
                {
                    ಹಿಂತಿರುಗಿ std::allocator_traits<Alloc>::max_size(_impl);
                }
                ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ==(ಸ್ಥಿರಾಂಕ usm_allocator &other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _impl == other._impl; }
                ತಾರ್ಕಿಕ ಕ್ರಿಯಾಕಾರಕ!=(ಸ್ಥಿರಾಂಕ usm_allocator &other) ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _impl != other._impl; }
            };

        } // namespace deprecated

        ಸಾಲೊಳಗೆ ಶೂನ್ಯ dpct_free(ಶೂನ್ಯ *ptr,
                              ಸ್ಥಿರಾಂಕ sycl::queue &q)
        {
            ಆದರೆ (ptr)
            {
                sycl::ಬಿಡುಗಡೆ(ptr, q.get_context());
            }
        }

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
        ಸಾಲೊಳಗೆ ಸ್ವಯಂಚಾಲಿತ get_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<ಶೂನ್ಯ *>(x));
            ಹಿಂತಿರುಗಿ new_x;
        }

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
        ಸಾಲೊಳಗೆ ಬಗೆ_ಹೆಸರು ದತ್ತಾಂಶ_Type_2<T>::T2 get_ಮೌಲ್ಯ(ಸ್ಥಿರಾಂಕ T *s, sycl::queue &q)
        {
            ಬಳಸಿ Ty = ಬಗೆ_ಹೆಸರು ದತ್ತಾಂಶ_Type_2<T>::T2;
            Ty s_h;
            ಆದರೆ (get_ಸೂಚಕ_attribute(q, s) == ಸೂಚಕ_access_attribute::device_only)
                detail::dpct_memcpy(q, (ಶೂನ್ಯ *)&s_h, (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)s, ಗಾತ್ರ(T), device_ಗಮ್ಯ_host)
                    .wait();
            ಇಲ್ಲದಿದ್ದರೆ
                s_h = *reinterpret_cast<ಸ್ಥಿರಾಂಕ Ty *>(s);
            ಹಿಂತಿರುಗಿ s_h;
        }

    } // namespace detail

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ಸಾಲೊಳಗೆ ಸ್ವಯಂಚಾಲಿತ get_ಮೌಲ್ಯ(ಸ್ಥಿರಾಂಕ T *s, sycl::queue &q)
    {
        ಹಿಂತಿರುಗಿ detail::get_ಮೌಲ್ಯ(s, q);
    }

    ನಾಮಾಕಾಶ detail
    {
    ಮಾದರಿ <ವರ್ಗ Ta, ವರ್ಗ Tb, ವರ್ಗ Tc, ವರ್ಗ Ts>
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಂಕ m,
                          ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b, ಪೂರ್ಣಾಂಕ ldb,
                          ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, ಪೂರ್ಣಾಂಕ ldc) {
        Ts   alpha_ಮೌಲ್ಯ_2 = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ Ts *>(alpha), q);
        Ts   beta_ಮೌಲ್ಯ  = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ Ts *>(beta), q);
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_a      = get_ಸ್ಮರಣೆ<ಸ್ಥಿರಾಂಕ Ta>(a);
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_b      = get_ಸ್ಮರಣೆ<ಸ್ಥಿರಾಂಕ Tb>(b);
        ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_c      = get_ಸ್ಮರಣೆ<Tc>(c);
        oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm(q, a_trans, b_trans, m, n, k, alpha_ಮೌಲ್ಯ_2, ದತ್ತಾಂಶ_a,
                                               lda, ದತ್ತಾಂಶ_b, ldb, beta_ಮೌಲ್ಯ, ದತ್ತಾಂಶ_c, ldc);
    }

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು VecT, ವರ್ಗ ದ್ವಿಮಾನ_Operation, ವರ್ಗ = ಶೂನ್ಯ>
        ವರ್ಗ vectorized_ದ್ವಿಮಾನ
        {
        ಸಾರ್ವಜನಿಕ:
            ಸಾಲೊಳಗೆ VecT ಕ್ರಿಯಾಕಾರಕ()(VecT a, VecT b, ಸ್ಥಿರಾಂಕ ದ್ವಿಮಾನ_Operation ದ್ವಿಮಾನ_op)
            {
                VecT v4;
                ಗಾಗಿ (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ದ್ವಿಮಾನ_op(a[i], b[i]);
                }
                ಹಿಂತಿರುಗಿ v4;
            }
        };

        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು VecT, ವರ್ಗ ದ್ವಿಮಾನ_Operation>
        ವರ್ಗ vectorized_ದ್ವಿಮಾನ<
            VecT, ದ್ವಿಮಾನ_Operation,
            std::void_t<std::invoke_result_t<ದ್ವಿಮಾನ_Operation, VecT, VecT>>>
        {
        ಸಾರ್ವಜನಿಕ:
            ಸಾಲೊಳಗೆ VecT ಕ್ರಿಯಾಕಾರಕ()(VecT a, VecT b, ಸ್ಥಿರಾಂಕ ದ್ವಿಮಾನ_Operation ದ್ವಿಮಾನ_op)
            {
                ಹಿಂತಿರುಗಿ ದ್ವಿಮಾನ_op(a, b).ಮಾದರಿ as<VecT>();
            }
        };

        ಮಾದರಿ <ವರ್ಗ Ta, ವರ್ಗ Tb, ವರ್ಗ Tc, ವರ್ಗ Ts>
        ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ ** a, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ ** b,
                                    ಪೂರ್ಣಾಂಕ ldb, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta, ಶೂನ್ಯ ** c, ಪೂರ್ಣಾಂಕ ldc, ಪೂರ್ಣಾಂಕ batch_ಗಾತ್ರ,
                                    ಮಾತೃಕೆ_info_t<ದಶಮಾಂಶ> * ಮಾತೃಕೆ_info) {
            Ts alpha_ಮೌಲ್ಯ_2 = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ Ts *>(alpha), q);
            Ts beta_ಮೌಲ್ಯ = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ Ts *>(beta), q);

            ಮಾತೃಕೆ_info->transpose_info[0] = a_trans;
            ಮಾತೃಕೆ_info->transpose_info[1] = b_trans;
            ಮಾತೃಕೆ_info->ಮೌಲ್ಯ_info[0] = alpha_ಮೌಲ್ಯ_2;
            ಮಾತೃಕೆ_info->ಮೌಲ್ಯ_info[1] = beta_ಮೌಲ್ಯ;
            ಮಾತೃಕೆ_info->ಗಾತ್ರ_info[0] = m;
            ಮಾತೃಕೆ_info->ಗಾತ್ರ_info[1] = n;
            ಮಾತೃಕೆ_info->ಗಾತ್ರ_info[2] = k;
            ಮಾತೃಕೆ_info->ld_info[0] = lda;
            ಮಾತೃಕೆ_info->ld_info[1] = ldb;
            ಮಾತೃಕೆ_info->ld_info[2] = ldc;
            ಮಾತೃಕೆ_info->groupsize_info = batch_ಗಾತ್ರ;

            sycl::event e = oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm_batch(
                q, ಮಾತೃಕೆ_info->transpose_info, ಮಾತೃಕೆ_info->transpose_info + 1,
                ಮಾತೃಕೆ_info->ಗಾತ್ರ_info, ಮಾತೃಕೆ_info->ಗಾತ್ರ_info + 1, ಮಾತೃಕೆ_info->ಗಾತ್ರ_info + 2,
                reinterpret_cast<Ts *>(ಮಾತೃಕೆ_info->ಮೌಲ್ಯ_info), reinterpret_cast<ಸ್ಥಿರಾಂಕ Ta **>(a), ಮಾತೃಕೆ_info->ld_info,
                reinterpret_cast<ಸ್ಥಿರಾಂಕ Tb **>(b), ಮಾತೃಕೆ_info->ld_info + 1,
                reinterpret_cast<Ts *>(ಮಾತೃಕೆ_info->ಮೌಲ್ಯ_info + 1), reinterpret_cast<Tc **>(c),
                ಮಾತೃಕೆ_info->ld_info + 2, 1, &(ಮಾತೃಕೆ_info->groupsize_info));
        }

        ಮಾದರಿ <ವರ್ಗ Ta, ವರ್ಗ Tb, ವರ್ಗ Tc, ವರ್ಗ Ts>
        ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a, ಪೂರ್ಣಾಂಕ lda,
                                    ದೀರ್ಘ ದೀರ್ಘ ಪೂರ್ಣಾಂಕ stride_a, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b, ಪೂರ್ಣಾಂಕ ldb, ದೀರ್ಘ ದೀರ್ಘ ಪೂರ್ಣಾಂಕ stride_b,
                                    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, ಪೂರ್ಣಾಂಕ ldc, ದೀರ್ಘ ದೀರ್ಘ ಪೂರ್ಣಾಂಕ stride_c, ಪೂರ್ಣಾಂಕ batch_ಗಾತ್ರ) {
            Ts alpha_ಮೌಲ್ಯ_2 = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ Ts *>(alpha), q);
            Ts beta_ಮೌಲ್ಯ = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ Ts *>(beta), q);
            ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_a = get_ಸ್ಮರಣೆ<ಸ್ಥಿರಾಂಕ Ta>(a);
            ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_b = get_ಸ್ಮರಣೆ<ಸ್ಥಿರಾಂಕ Tb>(b);
            ಸ್ವಯಂಚಾಲಿತ ದತ್ತಾಂಶ_c = get_ಸ್ಮರಣೆ<Tc>(c);
            oneapi::mkl::blas::ಸ್ತಂಭ_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_ಮೌಲ್ಯ_2,
                                                         ದತ್ತಾಂಶ_a, lda, stride_a, ದತ್ತಾಂಶ_b, ldb, stride_b, beta_ಮೌಲ್ಯ,
                                                         ದತ್ತಾಂಶ_c, ldc, stride_c, batch_ಗಾತ್ರ);
        }

    } // namespace detail

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು VecT, ವರ್ಗ ದ್ವಿಮಾನ_Operation>
    ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ vectorized_ದ್ವಿಮಾನ(ಚಿಹ್ನೆರಹಿತ a, ಚಿಹ್ನೆರಹಿತ b,
                                      ಸ್ಥಿರಾಂಕ ದ್ವಿಮಾನ_Operation ದ್ವಿಮಾನ_op)
    {
        sycl::vec<ಚಿಹ್ನೆರಹಿತ, 1> v0{a}, v1{b};
        ಸ್ವಯಂಚಾಲಿತ v2 = v0.as<VecT>();
        ಸ್ವಯಂಚಾಲಿತ v3 = v1.as<VecT>();
        ಸ್ವಯಂಚಾಲಿತ v4 =
            detail::vectorized_ದ್ವಿಮಾನ<VecT, ದ್ವಿಮಾನ_Operation>()(v2, v3, ದ್ವಿಮಾನ_op);
        v0 = v4.ಮಾದರಿ as<sycl::vec<ಚಿಹ್ನೆರಹಿತ, 1>>();
        ಹಿಂತಿರುಗಿ v0;
    }

    ಸ್ಥಿರ ಶೂನ್ಯ async_dpct_memcpy(ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_ಸರತಿ())
    {
        detail::dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, size, direction);
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ select_device(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id)
    {
        dev_mgr::instance().select_device(id);
        ಹಿಂತಿರುಗಿ id;
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    T ಕ್ರಮಬದಲಾವಣೆ_sub_ಗುಂಪು_by_xor(sycl::sub_ಗುಂಪು g, T x, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಮುಖವಾಡ,
                               ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ logical_sub_ಗುಂಪು_ಗಾತ್ರ = 32)
    {
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id = g.get_local_ರೇಖೀಯ_id();
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಸೂಚ್ಯಂಕ =
            id / logical_sub_ಗುಂಪು_ಗಾತ್ರ * logical_sub_ಗುಂಪು_ಗಾತ್ರ;
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಗುರಿ_ಸ್ಥಾನಾಂತರ = (id % logical_sub_ಗುಂಪು_ಗಾತ್ರ) ^ ಮುಖವಾಡ;
        ಹಿಂತಿರುಗಿ sycl::select_from_ಗುಂಪು(g, x,
                                       ಗುರಿ_ಸ್ಥಾನಾಂತರ < logical_sub_ಗುಂಪು_ಗಾತ್ರ
                                           ? ಆರಂಭ_ಸೂಚ್ಯಂಕ + ಗುರಿ_ಸ್ಥಾನಾಂತರ
                                           : id);
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T1, ಬಗೆ_ಹೆಸರು T2>
    ಬಳಸಿ dot_ಗುಣಲಬ್ಧ_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    sycl::vec<T, 4> ಹೊರತೆಗೆ_and_sign_or_zero_extend4(T val) {
      ಹಿಂತಿರುಗಿ sycl::vec<T, 1>(val)
          .ಮಾದರಿ as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .ಮಾದರಿ convert<T>();
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T1, ಬಗೆ_ಹೆಸರು T2, ಬಗೆ_ಹೆಸರು T3>
    ಸಾಲೊಳಗೆ ಸ್ವಯಂಚಾಲಿತ dp4a(T1 a, T2 b, T3 c) {
      dot_ಗುಣಲಬ್ಧ_acc_t<T1, T2> res = c;
      ಸ್ವಯಂಚಾಲಿತ va = ಹೊರತೆಗೆ_and_sign_or_zero_extend4(a);
      ಸ್ವಯಂಚಾಲಿತ vb = ಹೊರತೆಗೆ_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      ಹಿಂತಿರುಗಿ res;
    }

    ರಚನೆ sub_sat
    {
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
        ಸ್ವಯಂಚಾಲಿತ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ T x, ಸ್ಥಿರಾಂಕ T y) ಸ್ಥಿರಾಂಕ
        {
            ಹಿಂತಿರುಗಿ sycl::sub_sat(x, y);
        }
    };

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು S, ಬಗೆ_ಹೆಸರು T>
    ಸಾಲೊಳಗೆ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        ಸ್ವಯಂಚಾಲಿತ v2 = v0.ಮಾದರಿ as<S>();
        ಸ್ವಯಂಚಾಲಿತ v3 = v1.ಮಾದರಿ as<S>();
        ಸ್ವಯಂಚಾಲಿತ v4 = sycl::min(v2, v3);
        v0 = v4.ಮಾದರಿ as<sycl::vec<T, 1>>();
        ಹಿಂತಿರುಗಿ v0;
    }

    ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ಘಾತ(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b) { ಹಿಂತಿರುಗಿ sycl::pown(a, b); }
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ ಘಾತ(ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ a, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b) { ಹಿಂತಿರುಗಿ sycl::pown(a, b); }
    ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ಘಾತ(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) { ಹಿಂತಿರುಗಿ sycl::ಘಾತ(a, b); }
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ ಘಾತ(ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ a, ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ b) { ಹಿಂತಿರುಗಿ sycl::ಘಾತ(a, b); }
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು U>
    ಸಾಲೊಳಗೆ ಬಗೆ_ಹೆಸರು std::enable_if_t<std::is_floating_point_v<T>, T>
    ಘಾತ(ಸ್ಥಿರಾಂಕ T a, ಸ್ಥಿರಾಂಕ U b)
    {
        ಹಿಂತಿರುಗಿ sycl::ಘಾತ(a, static_cast<T>(b));
    }
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಬಗೆ_ಹೆಸರು U>
    ಸಾಲೊಳಗೆ ಬಗೆ_ಹೆಸರು std::enable_if_t<!std::is_floating_point_v<T>, ದ್ವಿಗುಣ>
    ಘಾತ(ಸ್ಥಿರಾಂಕ T a, ಸ್ಥಿರಾಂಕ U b)
    {
        ಹಿಂತಿರುಗಿ sycl::ಘಾತ(static_cast<ದ್ವಿಗುಣ>(a), static_cast<ದ್ವಿಗುಣ>(b));
    }

    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ min(ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b)
    {
        ಹಿಂತಿರುಗಿ sycl::fmin(a, static_cast<ದ್ವಿಗುಣ>(b));
    }
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ min(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ b)
    {
        ಹಿಂತಿರುಗಿ sycl::fmin(static_cast<ದ್ವಿಗುಣ>(a), b);
    }
    ಸಾಲೊಳಗೆ ದಶಮಾಂಶ min(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) { ಹಿಂತಿರುಗಿ sycl::fmin(a, b); }
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ min(ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ a, ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ b) { ಹಿಂತಿರುಗಿ sycl::fmin(a, b); }
    ಸಾಲೊಳಗೆ std::uint32_t min(ಸ್ಥಿರಾಂಕ std::uint32_t a, ಸ್ಥಿರಾಂಕ std::int32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, static_cast<std::uint32_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint32_t min(ಸ್ಥಿರಾಂಕ std::int32_t a, ಸ್ಥಿರಾಂಕ std::uint32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(static_cast<std::uint32_t>(a), b);
    }
    ಸಾಲೊಳಗೆ std::int32_t min(ಸ್ಥಿರಾಂಕ std::int32_t a, ಸ್ಥಿರಾಂಕ std::int32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint32_t min(ಸ್ಥಿರಾಂಕ std::uint32_t a, ಸ್ಥಿರಾಂಕ std::uint32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::int64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::int64_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ಸಾಲೊಳಗೆ std::int64_t min(ಸ್ಥಿರಾಂಕ std::int64_t a, ಸ್ಥಿರಾಂಕ std::int64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::int32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::int32_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::uint32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint64_t min(ಸ್ಥಿರಾಂಕ std::uint32_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ max(ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b)
    {
        ಹಿಂತಿರುಗಿ sycl::fmax(a, static_cast<ದ್ವಿಗುಣ>(b));
    }
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ max(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ b)
    {
        ಹಿಂತಿರುಗಿ sycl::fmax(static_cast<ದ್ವಿಗುಣ>(a), b);
    }
    ಸಾಲೊಳಗೆ ದಶಮಾಂಶ max(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ a, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b) { ಹಿಂತಿರುಗಿ sycl::fmax(a, b); }
    ಸಾಲೊಳಗೆ ದ್ವಿಗುಣ max(ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ a, ಸ್ಥಿರಾಂಕ ದ್ವಿಗುಣ b) { ಹಿಂತಿರುಗಿ sycl::fmax(a, b); }
    ಸಾಲೊಳಗೆ std::uint32_t max(ಸ್ಥಿರಾಂಕ std::uint32_t a, ಸ್ಥಿರಾಂಕ std::int32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, static_cast<std::uint32_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint32_t max(ಸ್ಥಿರಾಂಕ std::int32_t a, ಸ್ಥಿರಾಂಕ std::uint32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(static_cast<std::uint32_t>(a), b);
    }
    ಸಾಲೊಳಗೆ std::int32_t max(ಸ್ಥಿರಾಂಕ std::int32_t a, ಸ್ಥಿರಾಂಕ std::int32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint32_t max(ಸ್ಥಿರಾಂಕ std::uint32_t a, ಸ್ಥಿರಾಂಕ std::uint32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::int64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::int64_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ಸಾಲೊಳಗೆ std::int64_t max(ಸ್ಥಿರಾಂಕ std::int64_t a, ಸ್ಥಿರಾಂಕ std::int64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::int32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::int32_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::uint64_t a, ಸ್ಥಿರಾಂಕ std::uint32_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ಸಾಲೊಳಗೆ std::uint64_t max(ಸ್ಥಿರಾಂಕ std::uint32_t a, ಸ್ಥಿರಾಂಕ std::uint64_t b)
    {
        ಹಿಂತಿರುಗಿ sycl::max(static_cast<std::uint64_t>(a), b);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ
    has_capability_or_fail(ಸ್ಥಿರಾಂಕ sycl::device &dev,
                           ಸ್ಥಿರಾಂಕ std::initializer_list<sycl::aspect> &props)
    {
        ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ &it : props)
        {
            ಆದರೆ (dev.has(it))
                ಮುಂದುವರಿ;
            ಆಯ್ಕೆ (it)
            {
            ಸ್ಥಿತಿ sycl::aspect::fp64:
                ಎಸೆ std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ sycl::aspect::fp16:
                ಎಸೆ std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
#ವ್ಯಾಖ್ಯಾನಿಸು __SYCL_ASPECT(ASPECT, ID) \
    ಸ್ಥಿತಿ sycl::aspect::ASPECT:    \
        ಹಿಂತಿರುಗಿ #ASPECT;
#ವ್ಯಾಖ್ಯಾನಿಸು __SYCL_ASPECT_DEPRECATED(ASPECT, ID, ಸಂದೇಶ_3) __SYCL_ASPECT(ASPECT, ID)
#ವ್ಯಾಖ್ಯಾನಿಸು __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, ಸಂದೇಶ_3)
                ಸ್ವಯಂಚಾಲಿತ getAspectNameStr = [](sycl::aspect Aspect_ಅಂಕೆ) -> std::string
                {
                    ಆಯ್ಕೆ (Aspect_ಅಂಕೆ)
                    {
#ಅನ್ತರ್ಭಾವ <sycl/info/aspects.def>
#ಅನ್ತರ್ಭಾವ <sycl/info/aspects_deprecated.def>
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ಹಿಂತಿರುಗಿ "unknown aspect";
                    }
                };
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು __SYCL_ASPECT_DEPRECATED_ALIAS
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು __SYCL_ASPECT_DEPRECATED
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು __SYCL_ASPECT
                ಎಸೆ std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            ನಿಲ್ಲಿಸು;
        }
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ get_current_device_id()
    {
        ಹಿಂತಿರುಗಿ dev_mgr::instance().current_device_id();
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ device_ext &get_current_device()
    {
        ಹಿಂತಿರುಗಿ dev_mgr::instance().current_device();
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ device_ext &get_device(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ id)
    {
        ಹಿಂತಿರುಗಿ dev_mgr::instance().get_device(id);
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ sycl::queue &get_ಒಳ_ಕ್ರಮ_ಸರತಿ()
    {
        ಹಿಂತಿರುಗಿ dev_mgr::instance().current_device().ಒಳ_ಕ್ರಮ_ಸರತಿ();
    }

    ಸ್ಥಿರ sycl::event
    dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr, size_t size,
                memcpy_direction direction,
                ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &dep_events = {})
    {
        ಆದರೆ (!size)
            ಹಿಂತಿರುಗಿ sycl::event{};
        ಹಿಂತಿರುಗಿ q.ಸ್ಮರಣೆ_ನಕಲು(ಗಮ್ಯ_ptr, from_ptr, size, dep_events);
        DPCT_ಬಳಸದ(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t get_copy_range(sycl::range<3> size, size_t ತುಂಡು_2,
                                        size_t pitch)
    {
        ಹಿಂತಿರುಗಿ ತುಂಡು_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t get_ಸ್ಥಾನಾಂತರ(sycl::id<3> id, size_t ತುಂಡು_2,
                                    size_t pitch)
    {
        ಹಿಂತಿರುಗಿ ತುಂಡು_2 * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr,
                sycl::range<3> ಗಮ್ಯ_range, sycl::range<3> from_range,
                sycl::id<3> ಗಮ್ಯ_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        ವರ್ಗ host_ಬಫರ್
        {
            ಶೂನ್ಯ *_buf;
            size_t ಗಾತ್ರ_3;
            sycl::queue &_q;
            ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &_deps; // free operation depends

        ಸಾರ್ವಜನಿಕ:
            host_ಬಫರ್(size_t size, sycl::queue &q,
                        ಸ್ಥಿರಾಂಕ std::vector<sycl::event> &deps)
                : _buf(std::ಸ್ಮರಣೆ_ಕೊಡು(size)), ಗಾತ್ರ_3(size), _q(q), _deps(deps) {}
            ಶೂನ್ಯ *get_ptr() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ _buf; }
            size_t get_ಗಾತ್ರ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ಗಾತ್ರ_3; }
            ~host_ಬಫರ್()
            {
                ಆದರೆ (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::ಬಿಡುಗಡೆ(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t ಗಮ್ಯ_ತುಂಡು = ಗಮ್ಯ_range.get(1) * ಗಮ್ಯ_range.get(0),
               from_ತುಂಡು = from_range.get(1) * from_range.get(0);
        ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *ಗಮ್ಯ_surface =
            (ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *)ಗಮ್ಯ_ptr + get_ಸ್ಥಾನಾಂತರ(ಗಮ್ಯ_id, ಗಮ್ಯ_ತುಂಡು, ಗಮ್ಯ_range.get(0));
        ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *from_surface =
            (ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *)from_ptr +
            get_ಸ್ಥಾನಾಂತರ(from_id, from_ತುಂಡು, from_range.get(0));

        ಆದರೆ (ಗಮ್ಯ_ತುಂಡು == from_ತುಂಡು && ಗಮ್ಯ_ತುಂಡು == size.get(1) * size.get(0))
        {
            ಹಿಂತಿರುಗಿ {dpct_memcpy(q, ಗಮ್ಯ_surface, from_surface, ಗಮ್ಯ_ತುಂಡು * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, ಗಮ್ಯ_ptr, from_ptr, direction);
        size_t ಗಾತ್ರ_ತುಂಡು = size.get(1) * size.get(0);
        ಆಯ್ಕೆ (direction)
        {
        ಸ್ಥಿತಿ host_ಗಮ್ಯ_host:
            ಗಾಗಿ (size_t z = 0; z < size.get(2); ++z)
            {
                ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *ಗಮ್ಯ_ptr = ಗಮ್ಯ_surface;
                ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ಅಕ್ಷರ *from_ptr = from_surface;
                ಆದರೆ (ಗಮ್ಯ_range.get(0) == from_range.get(0) &&
                    ಗಮ್ಯ_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, ಗಾತ್ರ_ತುಂಡು,
                                                     direction, dep_events));
                }
                ಇಲ್ಲದಿದ್ದರೆ
                {
                    ಗಾಗಿ (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        ಗಮ್ಯ_ptr += ಗಮ್ಯ_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                ಗಮ್ಯ_surface += ಗಮ್ಯ_ತುಂಡು;
                from_surface += from_ತುಂಡು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ host_ಗಮ್ಯ_device:
        {
            host_ಬಫರ್ buf(get_copy_range(size, ಗಮ್ಯ_ತುಂಡು, ಗಮ್ಯ_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            ಆದರೆ (ಗಮ್ಯ_ತುಂಡು == ಗಾತ್ರ_ತುಂಡು)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, ಗಮ್ಯ_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_ಗಮ್ಯ_host, dep_events);
            }
            ಇಲ್ಲದಿದ್ದರೆ
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, ಗಮ್ಯ_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_ಗಮ್ಯ_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), ಗಮ್ಯ_surface, buf.get_ಗಾತ್ರ(),
                                    device_ಗಮ್ಯ_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, ಗಮ್ಯ_surface, buf.get_ptr(),
                                             buf.get_ಗಾತ್ರ(), host_ಗಮ್ಯ_device,
                                             host_events));
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ device_ಗಮ್ಯ_host:
        {
            host_ಬಫರ್ buf(get_copy_range(size, from_ತುಂಡು, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, ಗಮ್ಯ_surface, buf.get_ptr(), ಗಮ್ಯ_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_ಗಮ್ಯ_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_ಗಾತ್ರ(),
                                                     device_ಗಮ್ಯ_host, dep_events)});
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ device_ಗಮ್ಯ_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<ವರ್ಗ dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                ಗಮ್ಯ_surface[get_ಸ್ಥಾನಾಂತರ(id, ಗಮ್ಯ_ತುಂಡು, ಗಮ್ಯ_range.get(0))] =
                    from_surface[get_ಸ್ಥಾನಾಂತರ(id, from_ತುಂಡು, from_range.get(0))];
            }); }));
        ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಎಸೆ std::runtime_error("dpct_memcpy: invalid direction value");
        }
        ಹಿಂತಿರುಗಿ event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_ದತ್ತಾಂಶ ಗಮ್ಯ, sycl::id<3> ಗಮ್ಯ_id,
                pitched_ದತ್ತಾಂಶ from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        ಹಿಂತಿರುಗಿ dpct_memcpy(q, ಗಮ್ಯ.get_ದತ್ತಾಂಶ_ptr(), from.get_ದತ್ತಾಂಶ_ptr(),
                           sycl::range<3>(ಗಮ್ಯ.get_pitch(), ಗಮ್ಯ.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), ಗಮ್ಯ_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *ಗಮ್ಯ_ptr, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr,
                size_t ಗಮ್ಯ_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        ಹಿಂತಿರುಗಿ dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, sycl::range<3>(ಗಮ್ಯ_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಂಕ m, ಪೂರ್ಣಾಂಕ n,
                     ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a, library_ದತ್ತಾಂಶ_t a_type, ಪೂರ್ಣಾಂಕ lda, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b,
                     library_ದತ್ತಾಂಶ_t b_type, ಪೂರ್ಣಾಂಕ ldb, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, library_ದತ್ತಾಂಶ_t c_type, ಪೂರ್ಣಾಂಕ ldc,
                     library_ದತ್ತಾಂಶ_t scaling_type) {
        ಆದರೆ (scaling_type == library_ದತ್ತಾಂಶ_t::real_float &&
            c_type == library_ದತ್ತಾಂಶ_t::complex_float)
        {
            scaling_type = library_ದತ್ತಾಂಶ_t::complex_float;
        }
        ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (scaling_type == library_ದತ್ತಾಂಶ_t::real_double &&
                 c_type == library_ದತ್ತಾಂಶ_t::complex_double)
        {
            scaling_type = library_ದತ್ತಾಂಶ_t::complex_double;
        }

        std::uint64_t ಕೀಲಿ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ಆಯ್ಕೆ (ಕೀಲಿ)
        {
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_impl<ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_double, library_ದತ್ತಾಂಶ_t::real_double,
            library_ದತ್ತಾಂಶ_t::real_double, library_ದತ್ತಾಂಶ_t::real_double):
        {
            detail::gemm_impl<ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::complex_float, library_ದತ್ತಾಂಶ_t::complex_float,
            library_ದತ್ತಾಂಶ_t::complex_float, library_ದತ್ತಾಂಶ_t::complex_float):
        {
            detail::gemm_impl<std::complex<ದಶಮಾಂಶ>, std::complex<ದಶಮಾಂಶ>,
                              std::complex<ದಶಮಾಂಶ>, std::complex<ದಶಮಾಂಶ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::complex_double, library_ದತ್ತಾಂಶ_t::complex_double,
            library_ದತ್ತಾಂಶ_t::complex_double, library_ದತ್ತಾಂಶ_t::complex_double):
        {
            detail::gemm_impl<std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>,
                              std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __INTEL_MKL__
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_bfloat16,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_float):
        {
            ದಶಮಾಂಶ alpha_ಮೌಲ್ಯ_2 =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(alpha), q);
            ದಶಮಾಂಶ beta_ಮೌಲ್ಯ =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(beta), q);
            sycl::half alpha_half(alpha_ಮೌಲ್ಯ_2);
            sycl::half beta_half(beta_ಮೌಲ್ಯ);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_int8, library_ದತ್ತಾಂಶ_t::real_int8,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_bfloat16,
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_int8, library_ದತ್ತಾಂಶ_t::real_int8,
            library_ದತ್ತಾಂಶ_t::real_int32, library_ದತ್ತಾಂಶ_t::real_int32):
        {
            ದಶಮಾಂಶ alpha_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ std::int32_t *>(alpha), q);
            ದಶಮಾಂಶ beta_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            ನಿಲ್ಲಿಸು;
        }
#ಆದರೆ_ಅಂತ್ಯ // __INTEL_MKL__
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಎಸೆ std::runtime_error("the combination of data type is unsupported");
        }
    }  // gemm()

    /// Computes a batch of matrix-matrix product with general matrices.
    /// \param [in] q The queue where the routine should be executed.
    /// \param [in] a_trans Specifies the operation applied to A.
    /// \param [in] b_trans Specifies the operation applied to B.
    /// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
    /// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
    /// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
    /// \param [in] alpha Scaling factor for the matrix-matrix product.
    /// \param [in] a Input matrix A.
    /// \param [in] a_type Data type of the matrix A.
    /// \param [in] lda Leading dimension of A.
    /// \param [in] b Input matrix B.
    /// \param [in] b_type Data type of the matrix B.
    /// \param [in] ldb Leading dimension of B.
    /// \param [in] beta Scaling factor for matrix C.
    /// \param [in, out] c Input/Output matrix C.
    /// \param [in] c_type Data type of the matrix C.
    /// \param [in] ldc Leading dimension of C.
    /// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
    /// \param [in] scaling_type Data type of the scaling factors.
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಂಕ m,
                           ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a[], library_ದತ್ತಾಂಶ_t a_type, ಪೂರ್ಣಾಂಕ lda,
                           ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b[], library_ದತ್ತಾಂಶ_t b_type, ಪೂರ್ಣಾಂಕ ldb, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c[],
                           library_ದತ್ತಾಂಶ_t c_type, ಪೂರ್ಣಾಂಕ ldc, ಪೂರ್ಣಾಂಕ batch_ಗಾತ್ರ, library_ದತ್ತಾಂಶ_t scaling_type,
                           ಮಾತೃಕೆ_info_t<ದಶಮಾಂಶ> * ಮಾತೃಕೆ_info) {
        std::uint64_t ಕೀಲಿ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ಆಯ್ಕೆ (ಕೀಲಿ)
        {
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_double, library_ದತ್ತಾಂಶ_t::real_double,
            library_ದತ್ತಾಂಶ_t::real_double, library_ದತ್ತಾಂಶ_t::real_double):
        {
            detail::gemm_batch_impl<ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __INTEL_MKL__
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_bfloat16,
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_bfloat16,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_int8, library_ದತ್ತಾಂಶ_t::real_int8,
            library_ದತ್ತಾಂಶ_t::real_int32, library_ದತ್ತಾಂಶ_t::real_int32):
        {
            ದಶಮಾಂಶ alpha_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ std::int32_t *>(alpha), q);
            ದಶಮಾಂಶ beta_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_ಗಾತ್ರ,
                ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_int8, library_ದತ್ತಾಂಶ_t::real_int8,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_float):
        {
            ದಶಮಾಂಶ alpha_ಮೌಲ್ಯ_2 =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(alpha), q);
            ದಶಮಾಂಶ beta_ಮೌಲ್ಯ =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(beta), q);
            sycl::half alpha_half(alpha_ಮೌಲ್ಯ_2);
            sycl::half beta_half(beta_ಮೌಲ್ಯ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_ಗಾತ್ರ, ಮಾತೃಕೆ_info);
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಎಸೆ std::runtime_error("the combination of data type is unsupported");
        }
    }

    /// Computes a batch of matrix-matrix product with general matrices.
    /// \param [in] q The queue where the routine should be executed.
    /// \param [in] a_trans Specifies the operation applied to A.
    /// \param [in] b_trans Specifies the operation applied to B.
    /// \param [in] m Specifies the number of rows of the matrix op(A) and of the matrix C.
    /// \param [in] n Specifies the number of columns of the matrix op(B) and of the matrix C.
    /// \param [in] k Specifies the number of columns of the matrix op(A) and the number of rows of the matrix op(B).
    /// \param [in] alpha Scaling factor for the matrix-matrix product.
    /// \param [in] a Input matrix A.
    /// \param [in] a_type Data type of the matrix A.
    /// \param [in] lda Leading dimension of A.
    /// \param [in] stride_a Stride between the different A matrices.
    /// \param [in] b Input matrix B.
    /// \param [in] b_type Data type of the matrix B.
    /// \param [in] ldb Leading dimension of B.
    /// \param [in] stride_b Stride between the different B matrices.
    /// \param [in] beta Scaling factor for matrix C.
    /// \param [in, out] c Input/Output matrix C.
    /// \param [in] c_type Data type of the matrix C.
    /// \param [in] ldc Leading dimension of C.
    /// \param [in] stride_c Stride between the different C matrices.
    /// \param [in] batch_size Specifies the number of matrix multiply operations to perform.
    /// \param [in] scaling_type Data type of the scaling factors.
    ಸಾಲೊಳಗೆ ಶೂನ್ಯ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಂಕ m,
                           ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ k, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * alpha, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * a, library_ದತ್ತಾಂಶ_t a_type, ಪೂರ್ಣಾಂಕ lda,
                           ದೀರ್ಘ ದೀರ್ಘ ಪೂರ್ಣಾಂಕ stride_a, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * b, library_ದತ್ತಾಂಶ_t b_type, ಪೂರ್ಣಾಂಕ ldb,
                           ದೀರ್ಘ ದೀರ್ಘ ಪೂರ್ಣಾಂಕ stride_b, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, library_ದತ್ತಾಂಶ_t c_type, ಪೂರ್ಣಾಂಕ ldc,
                           ದೀರ್ಘ ದೀರ್ಘ ಪೂರ್ಣಾಂಕ stride_c, ಪೂರ್ಣಾಂಕ batch_ಗಾತ್ರ, library_ದತ್ತಾಂಶ_t scaling_type) {
        ಆದರೆ (scaling_type == library_ದತ್ತಾಂಶ_t::real_float &&
            c_type == library_ದತ್ತಾಂಶ_t::complex_float)
        {
            scaling_type = library_ದತ್ತಾಂಶ_t::complex_float;
        }
        ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (scaling_type == library_ದತ್ತಾಂಶ_t::real_double &&
                 c_type == library_ದತ್ತಾಂಶ_t::complex_double)
        {
            scaling_type = library_ದತ್ತಾಂಶ_t::complex_double;
        }

        std::uint64_t ಕೀಲಿ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ಆಯ್ಕೆ (ಕೀಲಿ)
        {
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_double, library_ದತ್ತಾಂಶ_t::real_double,
            library_ದತ್ತಾಂಶ_t::real_double, library_ದತ್ತಾಂಶ_t::real_double):
        {
            detail::gemm_batch_impl<ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::complex_float, library_ದತ್ತಾಂಶ_t::complex_float,
            library_ದತ್ತಾಂಶ_t::complex_float, library_ದತ್ತಾಂಶ_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<ದಶಮಾಂಶ>, std::complex<ದಶಮಾಂಶ>,
                                    std::complex<ದಶಮಾಂಶ>, std::complex<ದಶಮಾಂಶ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::complex_double, library_ದತ್ತಾಂಶ_t::complex_double,
            library_ದತ್ತಾಂಶ_t::complex_double, library_ದತ್ತಾಂಶ_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>,
                                    std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __INTEL_MKL__
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_bfloat16,
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_bfloat16, library_ದತ್ತಾಂಶ_t::real_bfloat16,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
#ಆದರೆ_ಅಂತ್ಯ
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_int8, library_ದತ್ತಾಂಶ_t::real_int8,
            library_ದತ್ತಾಂಶ_t::real_int32, library_ದತ್ತಾಂಶ_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_int8, library_ದತ್ತಾಂಶ_t::real_int8,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_float, library_ದತ್ತಾಂಶ_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ದಶಮಾಂಶ, ದಶಮಾಂಶ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಸ್ಥಿತಿ detail::get_type_combination_id(
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_half,
            library_ದತ್ತಾಂಶ_t::real_half, library_ದತ್ತಾಂಶ_t::real_float):
        {
            ದಶಮಾಂಶ alpha_ಮೌಲ್ಯ_2 =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(alpha), q);
            ದಶಮಾಂಶ beta_ಮೌಲ್ಯ =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(beta), q);
            sycl::half alpha_half(alpha_ಮೌಲ್ಯ_2);
            sycl::half beta_half(beta_ಮೌಲ್ಯ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_ಗಾತ್ರ);
            ನಿಲ್ಲಿಸು;
        }
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಎಸೆ std::runtime_error("the combination of data type is unsupported");
        }
    }

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ
    async_dpct_memcpy(ಶೂನ್ಯ *ಗಮ್ಯ_ptr, size_t ಗಮ್ಯ_pitch, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_ಸರತಿ())
    {
        detail::dpct_memcpy(q, ಗಮ್ಯ_ptr, from_ptr, ಗಮ್ಯ_pitch, from_pitch, x, y,
                            direction);
    }

    ಬಳಸಿ err0 = detail::generic_error_type<ರಚನೆ err0_ಟ್ಯಾಗ್, ಪೂರ್ಣಾಂಕ>;
    ಬಳಸಿ err1 = detail::generic_error_type<ರಚನೆ err1_ಟ್ಯಾಗ್, ಪೂರ್ಣಾಂಕ>;

    ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dpct_free(ಶೂನ್ಯ *ptr, sycl::queue &q = get_default_ಸರತಿ()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    ಮಾದರಿ <ವರ್ಗ T, ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2, size_t Dimension> ವರ್ಗ accessor;
    ಮಾದರಿ <ವರ್ಗ T, ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2> ವರ್ಗ accessor<T, ಸ್ಮರಣೆ_2, 3> {
    ಸಾರ್ವಜನಿಕ:
        ಬಳಸಿ ಸ್ಮರಣೆ_t = detail::ಸ್ಮರಣೆ_traits<ಸ್ಮರಣೆ_2, T>;
        ಬಳಸಿ element_t = ಬಗೆ_ಹೆಸರು ಸ್ಮರಣೆ_t::element_t;
        ಬಳಸಿ ಸೂಚಕ_t = ಬಗೆ_ಹೆಸರು ಸ್ಮರಣೆ_t::ಸೂಚಕ_t;
        ಬಳಸಿ accessor_t = ಬಗೆ_ಹೆಸರು ಸ್ಮರಣೆ_t::ಮಾದರಿ accessor_t<3>;
        accessor(ಸೂಚಕ_t ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ sycl::range<3> &ಒಳ_range)
            : ದತ್ತಾಂಶ_2(ದತ್ತಾಂಶ), _range(ಒಳ_range) {}
        ಮಾದರಿ <ಸ್ಮರಣೆ_region M = ಸ್ಮರಣೆ_2>
        accessor(ಬಗೆ_ಹೆಸರು std::enable_if<M != local, ಸ್ಥಿರಾಂಕ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ಸ್ಥಿರಾಂಕ accessor_t &acc, ಸ್ಥಿರಾಂಕ sycl::range<3> &ಒಳ_range)
            : accessor(acc.get_ಸೂಚಕ(), ಒಳ_range) {}
        accessor<T, ಸ್ಮರಣೆ_2, 2> ಕ್ರಿಯಾಕಾರಕ[](size_t ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            ಹಿಂತಿರುಗಿ accessor<T, ಸ್ಮರಣೆ_2, 2>(ದತ್ತಾಂಶ_2 + ಸೂಚ್ಯಂಕ * sub.size(), sub);
        }

        ಸೂಚಕ_t get_ptr() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ_2; }

    ಖಾಸಗಿ:
        ಸೂಚಕ_t ದತ್ತಾಂಶ_2;
        sycl::range<3> _range;
    };
    ಮಾದರಿ <ವರ್ಗ T, ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2> ವರ್ಗ accessor<T, ಸ್ಮರಣೆ_2, 2> {
    ಸಾರ್ವಜನಿಕ:
        ಬಳಸಿ ಸ್ಮರಣೆ_t = detail::ಸ್ಮರಣೆ_traits<ಸ್ಮರಣೆ_2, T>;
        ಬಳಸಿ element_t = ಬಗೆ_ಹೆಸರು ಸ್ಮರಣೆ_t::element_t;
        ಬಳಸಿ ಸೂಚಕ_t = ಬಗೆ_ಹೆಸರು ಸ್ಮರಣೆ_t::ಸೂಚಕ_t;
        ಬಳಸಿ accessor_t = ಬಗೆ_ಹೆಸರು ಸ್ಮರಣೆ_t::ಮಾದರಿ accessor_t<2>;
        accessor(ಸೂಚಕ_t ದತ್ತಾಂಶ, ಸ್ಥಿರಾಂಕ sycl::range<2> &ಒಳ_range)
            : ದತ್ತಾಂಶ_2(ದತ್ತಾಂಶ), _range(ಒಳ_range) {}
        ಮಾದರಿ <ಸ್ಮರಣೆ_region M = ಸ್ಮರಣೆ_2>
        accessor(ಬಗೆ_ಹೆಸರು std::enable_if<M != local, ಸ್ಥಿರಾಂಕ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ಸ್ಥಿರಾಂಕ accessor_t &acc, ಸ್ಥಿರಾಂಕ sycl::range<2> &ಒಳ_range)
            : accessor(acc.get_ಸೂಚಕ(), ಒಳ_range) {}

        ಸೂಚಕ_t ಕ್ರಿಯಾಕಾರಕ[](size_t ಸೂಚ್ಯಂಕ) ಸ್ಥಿರಾಂಕ {
            ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ_2 + _range.get(1) * ಸೂಚ್ಯಂಕ;
        }

        ಸೂಚಕ_t get_ptr() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ_2; }

    ಖಾಸಗಿ:
        ಸೂಚಕ_t ದತ್ತಾಂಶ_2;
        sycl::range<2> _range;
    };

    ನಾಮಾಕಾಶ detail {
        /// Device variable with address space of shared, global or constant.
        ಮಾದರಿ <ವರ್ಗ T, ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2, size_t Dimension> ವರ್ಗ device_ಸ್ಮರಣೆ {
        ಸಾರ್ವಜನಿಕ:
            ಬಳಸಿ accessor_t =
                ಬಗೆ_ಹೆಸರು detail::ಸ್ಮರಣೆ_traits<ಸ್ಮರಣೆ_2,
                                            T>::ಮಾದರಿ accessor_t<Dimension>;
            ಬಳಸಿ ಮೌಲ್ಯ_t = ಬಗೆ_ಹೆಸರು detail::ಸ್ಮರಣೆ_traits<ಸ್ಮರಣೆ_2, T>::ಮೌಲ್ಯ_t;
            ಬಳಸಿ dpct_accessor_t = dpct::accessor<T, ಸ್ಮರಣೆ_2, Dimension>;

            device_ಸ್ಮರಣೆ() : device_ಸ್ಮರಣೆ(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ sycl::range<Dimension> &ಒಳ_range,
                        std::initializer_list<ಮೌಲ್ಯ_t> &&init_list)
                : device_ಸ್ಮರಣೆ(ಒಳ_range) {
                ಪ್ರತಿಜ್ಞಾ(init_list.size() <= ಒಳ_range.size());
                _host_ptr = (ಮೌಲ್ಯ_t *)std::ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ_3);
                std::memset(_host_ptr, 0, ಗಾತ್ರ_3);
                std::ಸ್ಮರಣೆ_ನಕಲು(_host_ptr, init_list.begin(), init_list.size() * ಗಾತ್ರ(T));
            }

            /// Constructor of 2-D array with initializer list
            ಮಾದರಿ <size_t D = Dimension>
            device_ಸ್ಮರಣೆ(
                ಸ್ಥಿರಾಂಕ ಬಗೆ_ಹೆಸರು std::enable_if<D == 2, sycl::range<2>>::type &ಒಳ_range,
                std::initializer_list<std::initializer_list<ಮೌಲ್ಯ_t>> &&init_list)
                : device_ಸ್ಮರಣೆ(ಒಳ_range) {
                ಪ್ರತಿಜ್ಞಾ(init_list.size() <= ಒಳ_range[0]);
                _host_ptr = (ಮೌಲ್ಯ_t *)std::ಸ್ಮರಣೆ_ಕೊಡು(ಗಾತ್ರ_3);
                std::memset(_host_ptr, 0, ಗಾತ್ರ_3);
                ಸ್ವಯಂಚಾಲಿತ tmp_ದತ್ತಾಂಶ = _host_ptr;
                ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ sub_list : init_list) {
                    ಪ್ರತಿಜ್ಞಾ(sub_list.size() <= ಒಳ_range[1]);
                    std::ಸ್ಮರಣೆ_ನಕಲು(tmp_ದತ್ತಾಂಶ, sub_list.begin(),
                                sub_list.size() * ಗಾತ್ರ(T));
                    tmp_ದತ್ತಾಂಶ += ಒಳ_range[1];
                }
            }

            /// Constructor with range
            device_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ sycl::range<Dimension> &range_ಒಳ)
                : ಗಾತ್ರ_3(range_ಒಳ.size() * ಗಾತ್ರ(T)), _range(range_ಒಳ),
                _reference(ಸುಳ್ಳು), _host_ptr(ಶೂನ್ಯಸೂಚಕ), _device_ptr(ಶೂನ್ಯಸೂಚಕ) {
                static_assert(
                    (ಸ್ಮರಣೆ_2 == global) || (ಸ್ಮರಣೆ_2 == constant) || (ಸ್ಮರಣೆ_2 == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            ಮಾದರಿ <ವರ್ಗ... ವಾದಗಳು>
            device_ಸ್ಮರಣೆ(ವಾದಗಳು... Arguments)
                : device_ಸ್ಮರಣೆ(sycl::range<Dimension>(Arguments...)) {}

            ~device_ಸ್ಮರಣೆ() {
                ಆದರೆ (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                ಆದರೆ (_host_ptr)
                    std::ಬಿಡುಗಡೆ(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            ಶೂನ್ಯ init() { init(dpct::get_default_ಸರತಿ()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            ಶೂನ್ಯ init(sycl::queue &q) {
                ಆದರೆ (_device_ptr)
                    ಹಿಂತಿರುಗಿ;
                ಆದರೆ (!ಗಾತ್ರ_3)
                    ಹಿಂತಿರುಗಿ;
                allocate_device(q);
                ಆದರೆ (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, ಗಾತ್ರ_3,
                                        host_ಗಮ್ಯ_device);
            }

            /// The variable is assigned to a device pointer.
            ಶೂನ್ಯ ನಿಯೋಜಿಸು(ಮೌಲ್ಯ_t *src, size_t size) {
                ಇದು->~device_ಸ್ಮರಣೆ();
                ಹೊಸ (ಇದು) device_ಸ್ಮರಣೆ(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ಮೌಲ್ಯ_t *get_ptr() { ಹಿಂತಿರುಗಿ get_ptr(get_default_ಸರತಿ()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ಮೌಲ್ಯ_t *get_ptr(sycl::queue &q) {
                init(q);
                ಹಿಂತಿರುಗಿ _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_ಗಾತ್ರ() { ಹಿಂತಿರುಗಿ ಗಾತ್ರ_3; }

            ಮಾದರಿ <size_t D = Dimension>
            ಬಗೆ_ಹೆಸರು std::enable_if<D == 1, T>::type &ಕ್ರಿಯಾಕಾರಕ[](size_t ಸೂಚ್ಯಂಕ) {
                init();
                ಹಿಂತಿರುಗಿ _device_ptr[ಸೂಚ್ಯಂಕ];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            ಮಾದರಿ <size_t D = Dimension>
            ಬಗೆ_ಹೆಸರು std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                ಹಿಂತಿರುಗಿ dpct_accessor_t((T *)_device_ptr, _range);
            }

        ಖಾಸಗಿ:
            device_ಸ್ಮರಣೆ(ಮೌಲ್ಯ_t *ಸ್ಮರಣೆ_ptr, size_t size)
                : ಗಾತ್ರ_3(size), _range(size / ಗಾತ್ರ(T)), _reference(ನಿಜ),
                _device_ptr(ಸ್ಮರಣೆ_ptr) {}

            ಶೂನ್ಯ allocate_device(sycl::queue &q) {
        #ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ DPCT_USM_LEVEL_NONE
                ಆದರೆ (ಸ್ಮರಣೆ_2 == shared) {
                    _device_ptr = (ಮೌಲ್ಯ_t *)sycl::malloc_shared(ಗಾತ್ರ_3, q.get_device(),
                                                                q.get_context());
                    ಹಿಂತಿರುಗಿ;
                }
        #ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                ಆದರೆ (ಸ್ಮರಣೆ_2 == constant) {
                    _device_ptr = (ಮೌಲ್ಯ_t *)sycl::malloc_device(
                        ಗಾತ್ರ_3, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    ಹಿಂತಿರುಗಿ;
                }
        #ಆದರೆ_ಅಂತ್ಯ
        #ಆದರೆ_ಅಂತ್ಯ
                _device_ptr = (ಮೌಲ್ಯ_t *)detail::dpct_malloc(ಗಾತ್ರ_3, q);
            }

            size_t ಗಾತ್ರ_3;
            sycl::range<Dimension> _range;
            ತಾರ್ಕಿಕ _reference;
            ಮೌಲ್ಯ_t *_host_ptr;
            ಮೌಲ್ಯ_t *_device_ptr;
        };
        ಮಾದರಿ <ವರ್ಗ T, ಸ್ಮರಣೆ_region ಸ್ಮರಣೆ_2>
        ವರ್ಗ device_ಸ್ಮರಣೆ<T, ಸ್ಮರಣೆ_2, 0> : ಸಾರ್ವಜನಿಕ device_ಸ್ಮರಣೆ<T, ಸ್ಮರಣೆ_2, 1> {
        ಸಾರ್ವಜನಿಕ:
            ಬಳಸಿ ತಳ = device_ಸ್ಮರಣೆ<T, ಸ್ಮರಣೆ_2, 1>;
            ಬಳಸಿ ಮೌಲ್ಯ_t = ಬಗೆ_ಹೆಸರು ತಳ::ಮೌಲ್ಯ_t;
            ಬಳಸಿ accessor_t =
                ಬಗೆ_ಹೆಸರು detail::ಸ್ಮರಣೆ_traits<ಸ್ಮರಣೆ_2, T>::ಮಾದರಿ accessor_t<0>;

            /// Constructor with initial value.
            device_ಸ್ಮರಣೆ(ಸ್ಥಿರಾಂಕ ಮೌಲ್ಯ_t &val) : ತಳ(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_ಸ್ಮರಣೆ() : ತಳ(1) {}
        };
        } // namespace detail

    ಮಾದರಿ <ವರ್ಗ T, size_t Dimension>
    ಬಳಸಿ global_ಸ್ಮರಣೆ = detail::device_ಸ್ಮರಣೆ<T, global, Dimension>;
    ಮಾದರಿ <ವರ್ಗ T, size_t Dimension>
    ಬಳಸಿ constant_ಸ್ಮರಣೆ = detail::device_ಸ್ಮರಣೆ<T, constant, Dimension>;
    ಮಾದರಿ <ವರ್ಗ T, size_t Dimension>
    ಬಳಸಿ shared_ಸ್ಮರಣೆ = detail::device_ಸ್ಮರಣೆ<T, shared, Dimension>;


    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T,
            sycl::access::ವಿಳಾಸ_space ವಿಳಾಸ_Space =
                sycl::access::ವಿಳಾಸ_space::global_space,
            sycl::memory_order ಸ್ಮರಣೆ_ಕ್ರಮ = sycl::memory_order::relaxed,
            sycl::ಸ್ಮರಣೆ_scope ಸ್ಮರಣೆ_Scope = sycl::ಸ್ಮರಣೆ_scope::device>
    ಸಾಲೊಳಗೆ T atomic_fetch_add(T *addr, T ಕಾರ್ಯಾಂಶ) {
    ಸ್ವಯಂಚಾಲಿತ atm =
        sycl::atomic_ref<T, ಸ್ಮರಣೆ_ಕ್ರಮ, ಸ್ಮರಣೆ_Scope, ವಿಳಾಸ_Space>(addr[0]);
    ಹಿಂತಿರುಗಿ atm.fetch_add(ಕಾರ್ಯಾಂಶ);
    }

    ಮಾದರಿ <sycl::access::ವಿಳಾಸ_space ವಿಳಾಸ_Space =
                sycl::access::ವಿಳಾಸ_space::global_space,
            sycl::memory_order ಸ್ಮರಣೆ_ಕ್ರಮ = sycl::memory_order::relaxed,
            sycl::ಸ್ಮರಣೆ_scope ಸ್ಮರಣೆ_Scope = sycl::ಸ್ಮರಣೆ_scope::device,
            ಬಗೆ_ಹೆಸರು T1, ಬಗೆ_ಹೆಸರು T2>
    ಸಾಲೊಳಗೆ T1 atomic_fetch_add(T1 *addr, T2 ಕಾರ್ಯಾಂಶ) {
    ಸ್ವಯಂಚಾಲಿತ atm =
        sycl::atomic_ref<T1, ಸ್ಮರಣೆ_ಕ್ರಮ, ಸ್ಮರಣೆ_Scope, ವಿಳಾಸ_Space>(addr[0]);
    ಹಿಂತಿರುಗಿ atm.fetch_add(ಕಾರ್ಯಾಂಶ);
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, sycl::access::ವಿಳಾಸ_space ವಿಳಾಸ_Space =
                            sycl::access::ವಿಳಾಸ_space::global_space>
    ಸಾಲೊಳಗೆ T atomic_fetch_add(T *addr, T ಕಾರ್ಯಾಂಶ,
                            sycl::memory_order ಸ್ಮರಣೆ_ಕ್ರಮ) {
    ಆಯ್ಕೆ (ಸ್ಮರಣೆ_ಕ್ರಮ) {
        ಸ್ಥಿತಿ sycl::memory_order::relaxed:
            ಹಿಂತಿರುಗಿ atomic_fetch_add<T, ವಿಳಾಸ_Space, sycl::memory_order::relaxed,
                                    sycl::ಸ್ಮರಣೆ_scope::device>(addr, ಕಾರ್ಯಾಂಶ);
        ಸ್ಥಿತಿ sycl::memory_order::acq_rel:
            ಹಿಂತಿರುಗಿ atomic_fetch_add<T, ವಿಳಾಸ_Space, sycl::memory_order::acq_rel,
                                    sycl::ಸ್ಮರಣೆ_scope::device>(addr, ಕಾರ್ಯಾಂಶ);
        ಸ್ಥಿತಿ sycl::memory_order::seq_cst:
            ಹಿಂತಿರುಗಿ atomic_fetch_add<T, ವಿಳಾಸ_Space, sycl::memory_order::seq_cst,
                                    sycl::ಸ್ಮರಣೆ_scope::device>(addr, ಕಾರ್ಯಾಂಶ);
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    ಮಾದರಿ <sycl::access::ವಿಳಾಸ_space ವಿಳಾಸ_Space =
                sycl::access::ವಿಳಾಸ_space::global_space,
            ಬಗೆ_ಹೆಸರು T1, ಬಗೆ_ಹೆಸರು T2>
    ಸಾಲೊಳಗೆ T1 atomic_fetch_add(T1 *addr, T2 ಕಾರ್ಯಾಂಶ,
                            sycl::memory_order ಸ್ಮರಣೆ_ಕ್ರಮ) {
    atomic_fetch_add<T1, ವಿಳಾಸ_Space>(addr, ಕಾರ್ಯಾಂಶ, ಸ್ಮರಣೆ_ಕ್ರಮ);
    }

    ಸಾಲೊಳಗೆ ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ byte_level_ಕ್ರಮಬದಲಾವಣೆ(
        ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ a, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ b, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ s) {
      ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      ಹಿಂತಿರುಗಿ ret;
    }

    ಸಾಲೊಳಗೆ uint32_t byte_level_ಕ್ರಮಬದಲಾವಣೆ_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, ಪೂರ್ಣಾಂಕ mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      ಆದರೆ (mode >= 1 && mode <= 6) {
        ಹಿಂತಿರುಗಿ byte_level_ಕ್ರಮಬದಲಾವಣೆ(low32, high32, lookup[mode - 1][sel & 0x3]);
      } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (!mode) {
        ಹಿಂತಿರುಗಿ byte_level_ಕ್ರಮಬದಲಾವಣೆ(low32, high32, sel);
      }
      ಹಿಂತಿರುಗಿ 0;
    }

    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n_nondefault_params, ಪೂರ್ಣಾಂಕ n_default_params, ಬಗೆ_ಹೆಸರು T>
    ವರ್ಗ ವಾದಗಳು_selector;

    /// args_selector is a helper class for extracting arguments from an
    /// array of pointers to arguments or buffer of arguments to pass to a
    /// kernel function.
    ///
    /// \param R(Ts...) The type of the kernel
    /// \param n_nondefault_params The number of nondefault parameters of the
    /// kernel (excluding parameters that like sycl::nd_item, etc.) \param
    /// n_default_params The number of default parameters of the kernel
    ///
    /// Example usage:
    /// With the following kernel:
    ///   void foo(sycl::float2 *x, int n, sycl::nd_item<3> item_ct1, float
    ///   f=.1) {}
    /// and with the declaration:
    ///   args_selector<2, 1, decltype(foo)> selector(kernelParams, extra);
    /// we have:
    ///   selector.get<0>() returns a reference to sycl::float*,
    ///   selector.get<1>() returns a reference to int,
    ///   selector.get<2>() returns a reference to float
    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ n_nondefault_params, ಪೂರ್ಣಾಂಕ n_default_params, ಬಗೆ_ಹೆಸರು R,
              ಬಗೆ_ಹೆಸರು... Ts>
    ವರ್ಗ ವಾದಗಳು_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      ಖಾಸಗಿ:
        ಶೂನ್ಯ **kernel_params;
        ಅಕ್ಷರ *ವಾದಗಳು_ಬಫರ್;

        ಮಾದರಿ <ಪೂರ್ಣಾಂಕ i> ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ account_for_default_params() {
            constexpr ಪೂರ್ಣಾಂಕ n_ಒಟ್ಟು_params = ಗಾತ್ರ...(Ts);
            ಆದರೆ constexpr (i >= n_nondefault_params) {
                ಹಿಂತಿರುಗಿ n_ಒಟ್ಟು_params - n_default_params +
                       (i - n_nondefault_params);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ i;
            }
        }

      ಸಾರ್ವಜನಿಕ:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        ಮಾದರಿ <ಪೂರ್ಣಾಂಕ i>
        ಬಳಸಿ arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ params_ಅಂಕೆ = ಗಾತ್ರ...(Ts);

      ಖಾಸಗಿ:
        ಮಾದರಿ <ಪೂರ್ಣಾಂಕ i> ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ get_ಸ್ಥಾನಾಂತರ() {
            ಆದರೆ constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                ಹಿಂತಿರುಗಿ 0;
            } ಇಲ್ಲದಿದ್ದರೆ {
                constexpr ಪೂರ್ಣಾಂಕ prev_off = get_ಸ್ಥಾನಾಂತರ<i - 1>();
                constexpr ಪೂರ್ಣಾಂಕ prev_past_ಕೊನೆ =
                    prev_off + ಗಾತ್ರ(arg_type<i - 1>);
                ಬಳಸಿ T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                ಆದರೆ constexpr (prev_past_ಕೊನೆ % alignof(T) == 0) {
                    ಹಿಂತಿರುಗಿ prev_past_ಕೊನೆ;
                }
                // otherwise bump prev_past_end to match alignment
                ಇಲ್ಲದಿದ್ದರೆ {
                    ಹಿಂತಿರುಗಿ prev_past_ಕೊನೆ +
                           (alignof(T) - (prev_past_ಕೊನೆ % alignof(T)));
                }
            }
        }

        ಸ್ಥಿರ ಅಕ್ಷರ *get_ವಾದಗಳು_ಬಫರ್(ಶೂನ್ಯ **extra) {
            ಆದರೆ (!extra)
                ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
            ಗಾಗಿ (; (std::size_t)*extra != 0; ++extra) {
                ಆದರೆ ((std::size_t)*extra == 1) {
                    ಹಿಂತಿರುಗಿ static_cast<ಅಕ್ಷರ *>(*(extra + 1));
                }
            }
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

      ಸಾರ್ವಜನಿಕ:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        ವಾದಗಳು_selector(ಶೂನ್ಯ **kernel_params, ಶೂನ್ಯ **extra)
            : kernel_params(kernel_params),
              ವಾದಗಳು_ಬಫರ್(get_ವಾದಗಳು_ಬಫರ್(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        ಮಾದರಿ <ಪೂರ್ಣಾಂಕ i> arg_type<i> &get() {
            ಆದರೆ (kernel_params) {
                ಹಿಂತಿರುಗಿ *static_cast<arg_type<i> *>(kernel_params[i]);
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಹಿಂತಿರುಗಿ *reinterpret_cast<arg_type<i> *>(ವಾದಗಳು_ಬಫರ್ +
                                                        get_ಸ್ಥಾನಾಂತರ<i>());
            }
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp

    /// Utility class for launching SYCL kernels through kernel
    /// function wrapper.
    /// For example:
    /// A SYCL kernel function:
    ///   void kernel_func(int *ptr, sycl::nd_item<3> item);
    /// Kernel function wrapper:
    ///   void kernel_func_wrapper(int *ptr) {
    ///     sycl::queue queue = *dpct::kernel_launcher::_que;
    ///     unsigned int localMemSize = dpct::kernel_launcher::_local_mem_size;
    ///     sycl::nd_range<3> nr = dpct::kernel_launcher::_nr;
    ///     queue.parallel_for(
    ///       nr,
    ///       [=](sycl::nd_item<3> item_ct1) {
    ///         kernel_func(ptr, item_ct1);
    ///       });
    ///   }
    /// Then launch the kernel through wrapper like:
    ///   typedef void(*fpt)(int *);
    ///   fpt fp = kernel_func_wrapper;
    ///   dpct::kernel_launcher::launch(fp, dpct::dim3(1), dpct::dim3(1), 0, 0,
    ///   device_ptr);
    /// If the origin function type is erased, then need to register it first:
    ///   void *fp = (void *)wrapper_register(&kernel_func_wrapper).get();
    ///   dpct::kernel_launcher::launch(fp, dpct::dim3(1), dpct::dim3(1), args,
    ///   0, 0);
    ವರ್ಗ kernel_launcher {
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು FuncT, ಬಗೆ_ಹೆಸರು ArgSelector, std::size_t... ಸೂಚ್ಯಂಕ_2>
        ಸ್ಥಿರ ಶೂನ್ಯ launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<ಸೂಚ್ಯಂಕ_2...>) {
            func(selector.ಮಾದರಿ get<ಸೂಚ್ಯಂಕ_2>()...);
        }
        ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_execution_config(dim3 ಗುಂಪು_range, dim3 local_range,
                                         ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ local_mem_ಗಾತ್ರ_2,
                                         ಸರತಿ_ptr que) {
            ಆದರೆ (que) {
                _que = que;
            } ಇಲ್ಲದಿದ್ದರೆ {
                _que = &get_default_ಸರತಿ();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(ಗುಂಪು_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_ಗಾತ್ರ = local_mem_ಗಾತ್ರ_2;


        }
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::mutex kernel_function_ptr_map_mutex;

      ಸಾರ್ವಜನಿಕ:
        /// Variables for storing execution configuration.
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ thread_local sycl::queue *_que = ಶೂನ್ಯಸೂಚಕ;
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ thread_local ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ local_mem_ಗಾತ್ರ = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ಸ್ಥಿರ ಸಾಲೊಳಗೆ std::map<
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *,
            std::function<ಶೂನ್ಯ(dim3, dim3, ಶೂನ್ಯ **, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ, ಸರತಿ_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ಸ್ಥಿರ ಶೂನ್ಯ ರಿಜಿಸ್ಟರ್_kernel_ptr(
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *func,
            std::function<ಶೂನ್ಯ(dim3, dim3, ಶೂನ್ಯ **, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ, ಸರತಿ_ptr)>
                launcher) {
            std::lock_guard<std::mutex> ಬೀಗ(kernel_function_ptr_map_mutex);
            kernel_function_ptr_map[func] = std::move(launcher);
        }
        /// Launches a kernel function with arguments provided directly through
        /// kernel function wrapper.
        /// \tparam FuncT Type of the kernel function wrapper.
        /// \tparam ArgsT Types of kernel arguments.
        /// \param [in] func Pointer to the kernel function wrapper.
        /// \param [in] group_range SYCL group range.
        /// \param [in] local_range SYCL local range.
        /// \param [in] local_mem_size The size of local memory required by the
        /// kernel function. \param [in] que SYCL queue used to execute kernel.
        /// \param [in] args Kernel arguments.
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು FuncT, ಬಗೆ_ಹೆಸರು... ವಾದಗಳು_T>
        ಸ್ಥಿರ std::enable_if_t<std::is_invocable_v<FuncT *, ವಾದಗಳು_T...>, ಶೂನ್ಯ>
        launch(FuncT *func, dim3 ಗುಂಪು_range, dim3 local_range,
               ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ local_mem_ಗಾತ್ರ_2, ಸರತಿ_ptr que, ವಾದಗಳು_T... ವಾದಗಳು_2) {
            ಗಣ_execution_config(ಗುಂಪು_range, local_range, local_mem_ಗಾತ್ರ_2, que);
            func(ವಾದಗಳು_2...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        ಸ್ಥಿರ ಶೂನ್ಯ launch(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *func, dim3 ಗುಂಪು_range, dim3 local_range,
                           ಶೂನ್ಯ **ವಾದಗಳು_2, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ local_mem_ಗಾತ್ರ_2,
                           ಸರತಿ_ptr que) {
            std::lock_guard<std::mutex> ಬೀಗ(kernel_function_ptr_map_mutex);
            ಸ್ವಯಂಚಾಲಿತ Iter = kernel_function_ptr_map.find(func);
            ಆದರೆ (Iter == kernel_function_ptr_map.end()) {
                ಎಸೆ std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(ಗುಂಪು_range, local_range, ವಾದಗಳು_2, local_mem_ಗಾತ್ರ_2, que);
        }
        /// Launches a kernel function with packed arguments through kernel
        /// function wrapper.
        /// \tparam FuncT Type of the kernel function wrapper.
        /// \param [in] func Pointer to the kernel function wrapper.
        /// \param [in] group_range SYCL group range.
        /// \param [in] local_range SYCL local range.
        /// \param [in] args Array of pointers to kernel arguments.
        /// \param [in] local_mem_size The size of local memory required by the
        /// kernel function. \param [in] que SYCL queue used to execute kernel.
        ಮಾದರಿ <ಬಗೆ_ಹೆಸರು FuncT>
        ಸ್ಥಿರ std::enable_if_t<std::is_function_v<FuncT>, ಶೂನ್ಯ>
        launch(FuncT *func, dim3 ಗುಂಪು_range, dim3 local_range, ಶೂನ್ಯ **ವಾದಗಳು_2,
               ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ local_mem_ಗಾತ್ರ_2, ಸರತಿ_ptr que) {
            constexpr size_t p_ಅಂಕೆ = ವಾದಗಳು_selector<0, 0, FuncT>::params_ಅಂಕೆ;
            ಗಣ_execution_config(ಗುಂಪು_range, local_range, local_mem_ಗಾತ್ರ_2, que);
            ವಾದಗಳು_selector<p_ಅಂಕೆ, p_ಅಂಕೆ, FuncT> selector(ವಾದಗಳು_2, ಶೂನ್ಯಸೂಚಕ);
            launch_helper(func, selector, std::make_index_sequence<p_ಅಂಕೆ>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    T select_from_sub_ಗುಂಪು(
        sycl::sub_ಗುಂಪು g,
        T x,
        ಪೂರ್ಣಾಂಕ remote_local_id,
        ಪೂರ್ಣಾಂಕ logical_sub_ಗುಂಪು_ಗಾತ್ರ = 32) {
      ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ಆರಂಭ_ಸೂಚ್ಯಂಕ = g.get_local_ರೇಖೀಯ_id() /
                                 logical_sub_ಗುಂಪು_ಗಾತ್ರ *
                                 logical_sub_ಗುಂಪು_ಗಾತ್ರ;
      ಹಿಂತಿರುಗಿ sycl::select_from_ಗುಂಪು(
          g, x, ಆರಂಭ_ಸೂಚ್ಯಂಕ + remote_local_id % logical_sub_ಗುಂಪು_ಗಾತ್ರ);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ಶೂನ್ಯ ldmatrix(uintptr_t addr, T* m, ತಾರ್ಕಿಕ trans = ಸುಳ್ಳು, ಚಿಹ್ನೆರಹಿತ mat = 0) {
      ಸ್ವಯಂಚಾಲಿತ sg = sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು();
      ಪೂರ್ಣಾಂಕ lane = sg.get_local_ರೇಖೀಯ_id();

      ಪೂರ್ಣಾಂಕ lane_group8_ಅಡ್ಡಸಾಲು = lane / 8;
      ಪೂರ್ಣಾಂಕ lane_group8_ಲಂಬಸಾಲು = lane % 8;

      ಆದರೆ (!trans) {
        // calculate the source lane
        ಪೂರ್ಣಾಂಕ src_lane = 2 * lane_group8_ಅಡ್ಡಸಾಲು;
        ಆದರೆ (lane_group8_ಲಂಬಸಾಲು >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        ಸ್ವಯಂಚಾಲಿತ recv_addr_uintp =
            dpct::select_from_sub_ಗುಂಪು(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        ಸ್ವಯಂಚಾಲಿತ recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_ಲಂಬಸಾಲು % 4];
      } ಇಲ್ಲದಿದ್ದರೆ {
        // calculate the source lane
        ಪೂರ್ಣಾಂಕ src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        ಸ್ವಯಂಚಾಲಿತ recv_addr_uintp_1 =
            dpct::select_from_sub_ಗುಂಪು(sg, addr, mat * 8 + src_lane);
        ಸ್ವಯಂಚಾಲಿತ recv_addr_uintp_2 =
            dpct::select_from_sub_ಗುಂಪು(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        ಸ್ವಯಂಚಾಲಿತ recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        ಸ್ವಯಂಚಾಲಿತ recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        ಪೂರ್ಣಾಂಕ ಸೂಚ್ಯಂಕ = lane / 4;
        sycl::half val0 = recv_addr_1[ಸೂಚ್ಯಂಕ];
        sycl::half val1 = recv_addr_2[ಸೂಚ್ಯಂಕ];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ಶೂನ್ಯ ldmatrix(uintptr_t addr, T* m1, T* m2, ತಾರ್ಕಿಕ trans = ಸುಳ್ಳು) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ಶೂನ್ಯ ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, ತಾರ್ಕಿಕ trans = ಸುಳ್ಳು) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
      // Load 3rd matrix
      ldmatrix(addr, m3, trans, 2);
      // Load 4th matrix
      ldmatrix(addr, m4, trans, 3);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp

    /// A helper struct that defines the pack type for the input matrix
    /// fragments
    /// of mma() function based on the type of input matrix fragments.
    /// The MMAType struct is specialized for different types of input matrices.
    /// Currently, the specialization for f16, bf16 and s8 types is defined
    /// below. \tparam [in] T The type of the input matrix fragments
    ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
    ರಚನೆ MMAType {
      ಬಳಸಿ PackType = uint32_t;
    };

    /// Each work item of a sub-group (limited to size 32) calling this function
    /// calculates a subset fragment for the output matrix D using MAD operation
    /// on A, B & C matrix fragments (D = A * B + C). Current supported shapes &
    /// types:
    /// - m8n8k4 (f32.f16.f16.f32)
    /// - m8n8k16 (s32.s8.s8.s32)
    /// - m16n8k8 (f32.f16.f16.f32 & f32.bf16.bf16.f32)
    /// - m16n8k16 (f32.f16.f16.f32 & s32.s8.s8.s32)
    /// - m16n8k32 (s32.s8.s8.s32)
    /// Here, m, n & k define the shapes of A, B & C matrices respectively
    /// (A = [m x k], B = [k x n], C = [m x n]).
    /// \tparam [in] M The rows of A, C & D matrices
    /// \tparam [in] N The columns of B, C, D matrices
    /// \tparam [in] K The columns & rows of A & B matrices respectively
    /// \tparam [in] ABType The type of the input matrix (A & B) fragment
    /// \tparam [in] CDType The type of the output matrix (C & D) fragment
    /// \param [out] d_mat_frag The fragment of the output matrix D to store the
    /// result of A * B + C
    /// \param [in] a_mat_frag The fragment of the input matrix A to be
    /// multiplied with B matrix fragment \param [in] b_mat_frag The fragment of
    /// the input matrix B to be multiplied with A matrix fragment \param [in]
    /// c_mat_frag The fragment of the input matrix C to be added with the
    /// result of A * B fragments
    ಮಾದರಿ <ಪೂರ್ಣಾಂಕ M, ಪೂರ್ಣಾಂಕ N, ಪೂರ್ಣಾಂಕ K, ಬಗೆ_ಹೆಸರು ABType, ಬಗೆ_ಹೆಸರು CDType>
    ಶೂನ್ಯ mma(
        ಅಸ್ಥಿರ ಶೂನ್ಯ** d_mat_frag,
        ಶೂನ್ಯ* a_mat_frag,
        ಶೂನ್ಯ* b_mat_frag,
        ಶೂನ್ಯ* c_mat_frag) {
      ಸ್ವಯಂಚಾಲಿತ d = reinterpret_cast<ಅಸ್ಥಿರ CDType**>(d_mat_frag);
      ಸ್ವಯಂಚಾಲಿತ a =
          reinterpret_cast<ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType*>(a_mat_frag);
      ಸ್ವಯಂಚಾಲಿತ b =
          reinterpret_cast<ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType*>(b_mat_frag);
      ಸ್ವಯಂಚಾಲಿತ c = reinterpret_cast<CDType*>(c_mat_frag);

      ಸ್ವಯಂಚಾಲಿತ sg = sycl::ext::oneapi::this_work_ಅಂಶ::get_sub_ಗುಂಪು();
      ಪೂರ್ಣಾಂಕ lane = sg.get_local_ರೇಖೀಯ_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      ಗಿಡ್ಡ ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ = 4 * (lane >> 2);
      ಗಿಡ್ಡ ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ = 8 * (lane % 4);

      ಆದರೆ constexpr (M == 8 && N == 8 && K == 4) {
        ಆದರೆ constexpr (std::is_floating_point_v<CDType>) {
          ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ = ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ % 16;

          // Init D matrix with fragments of C matrix
          *d[0] = c[0];
          *d[1] = c[1];
          *d[2] = c[2];
          *d[3] = c[3];
          *d[4] = c[4];
          *d[5] = c[5];
          *d[6] = c[6];
          *d[7] = c[7];

          // Calculate the row and col offset indices to iterate through the row
          // & col fragments of A & B matrices
          ಪೂರ್ಣಾಂಕ r_ind = (lane % 2) ? 1 : 0;
          ಪೂರ್ಣಾಂಕ c_ind = ((lane % 4) / 2) ? 2 : 0;

          // Each sub-group is responsible for computing a fragment size of 8*8
          // elements of matrix D for each of 4 MMA computations.
          // Each work item computes 8 elements of matrix D by gathering
          // their corresponding col & row matrix fragments of length k (4)
          // from A & B matrices respectively using below mapping logic:
          // row0 = (i % 4) if (lane < 16) else (i % 4) + 4
          // col0 = (lane % 4)
          // As each row & col fragment of A & B matrices is distributed across
          // 4 work items, each iteration of below loop loads a partial fragment
          // of matrix A (row) and matrix B (col) using the row & col offsets.
          ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a[2], recv_b[2];

          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[0], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[1], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[1], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);

            ಸ್ವಯಂಚಾಲಿತ ra = reinterpret_cast<ABType*>(recv_a);
            ಸ್ವಯಂಚಾಲಿತ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment (for
            // even work item indices) d0 += col0{ a0 } * row0{ b0 } d1 += col0{
            // a0 } * row0{ b1 } d2 += col1{ a2 } * row0{ b0 } d3 += col1{ a2 }
            // * row0{ b1 } (for odd work item indices) d0 += col0{ a1 } * row0{
            // b2 } d1 += col0{ a1 } * row0{ b3 } d2 += col1{ a3 } * row0{ b2 }
            // d3 += col1{ a3 } * row0{ b3 }
            *d[0] +=
                static_cast<ದಶಮಾಂಶ>(ra[r_ind]) * static_cast<ದಶಮಾಂಶ>(rb[c_ind]);
            *d[1] += static_cast<ದಶಮಾಂಶ>(ra[r_ind]) *
                     static_cast<ದಶಮಾಂಶ>(rb[c_ind + 1]);
            *d[2] += static_cast<ದಶಮಾಂಶ>(ra[r_ind + 2]) *
                     static_cast<ದಶಮಾಂಶ>(rb[c_ind]);
            *d[3] += static_cast<ದಶಮಾಂಶ>(ra[r_ind + 2]) *
                     static_cast<ದಶಮಾಂಶ>(rb[c_ind + 1]);

            // Load partial fragment from row1 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[1], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 16);

            // (for even work item indices)
            // d0 += col0{ a0 } * row1{ b0 }
            // d1 += col0{ a0 } * row1{ b1 }
            // d2 += col1{ a2 } * row1{ b0 }
            // d3 += col1{ a2 } * row1{ b1 }
            // (for odd work item indices)
            // d0 += col0{ a1 } * row1{ b2 }
            // d1 += col0{ a1 } * row1{ b3 }
            // d2 += col1{ a3 } * row1{ b2 }
            // d3 += col1{ a3 } * row1{ b3 }
            *d[4] +=
                static_cast<ದಶಮಾಂಶ>(ra[r_ind]) * static_cast<ದಶಮಾಂಶ>(rb[c_ind]);
            *d[5] += static_cast<ದಶಮಾಂಶ>(ra[r_ind]) *
                     static_cast<ದಶಮಾಂಶ>(rb[c_ind + 1]);
            *d[6] += static_cast<ದಶಮಾಂಶ>(ra[r_ind + 2]) *
                     static_cast<ದಶಮಾಂಶ>(rb[c_ind]);
            *d[7] += static_cast<ದಶಮಾಂಶ>(ra[r_ind + 2]) *
                     static_cast<ದಶಮಾಂಶ>(rb[c_ind + 1]);
          }
        }
      } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (M == 8 && N == 8 && K == 16) {
        ಆದರೆ constexpr (std::is_integral_v<ABType>) {
          // Init D matrix with fragments of C matrix
          *d[0] = c[0];
          *d[1] = c[1];

          // Each sub-group is responsible for computing a fragment size of 16*8
          // elements of matrix D.
          // Each work item computes 2 elements of matrix D by gathering
          // their corresponding row & col matrix fragments of length k (16)
          // from A & B matrices respectively using below mapping logic:
          // row0 = ((lane % 4) * 4) + i
          // col0 = (lane >> 2)
          // As each row & col fragment of A & B matrices is distributed across
          // 4 work items, each iteration of below loop loads a partial fragment
          // of matrix A (row) and matrix B (col) using the row & col offsets.
          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_ಗುಂಪು(sg, a[0], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 4);

            ಸ್ವಯಂಚಾಲಿತ a = reinterpret_cast<ABType*>(&recv_a);
            ಸ್ವಯಂಚಾಲಿತ b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (M == 16 && N == 8 && K == 8) {
        ಆದರೆ constexpr (std::is_floating_point_v<CDType>) {
          // Init D matrix fragment with C matrix fragment
          *d[0] = c[0];
          *d[1] = c[1];
          *d[2] = c[2];
          *d[3] = c[3];

          // Each sub-group is responsible for computing a fragment size of 16*8
          // elements of matrix D.
          // Each work item computes 4 elements of matrix D by gathering
          // their corresponding row & col matrix fragments of length k (8)
          // from A & B matrices respectively using below mapping logic:
          // row0 = (lane >> 2) & row1 = (lane >> 2) + 8
          // col0 = (lane % 4) * 2 + (i & 0x1)
          // As each row & col fragment of A & B matrices is distributed across
          // 4 work items, each iteration of below loop loads a partial fragment
          // of matrix A (row) and matrix B (col) using the row & col offsets.
          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[0], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[1], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 4);

            ಸ್ವಯಂಚಾಲಿತ ra = reinterpret_cast<ABType*>(recv_a);
            ಸ್ವಯಂಚಾಲಿತ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 2; j++) {
              *d[0] += static_cast<ದಶಮಾಂಶ>(ra[j]) * static_cast<ದಶಮಾಂಶ>(rb[j]);
              *d[1] +=
                  static_cast<ದಶಮಾಂಶ>(ra[j]) * static_cast<ದಶಮಾಂಶ>(rb[j + 2]);
              *d[2] +=
                  static_cast<ದಶಮಾಂಶ>(ra[j + 2]) * static_cast<ದಶಮಾಂಶ>(rb[j]);
              *d[3] +=
                  static_cast<ದಶಮಾಂಶ>(ra[j + 2]) * static_cast<ದಶಮಾಂಶ>(rb[j + 2]);
            }
          }
        }
      } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (M == 16 && N == 8 && K == 16) {
        ಆದರೆ constexpr (std::is_floating_point_v<CDType>) {
          // Init D matrix fragment with C matrix fragment
          *d[0] = c[0];
          *d[1] = c[1];
          *d[2] = c[2];
          *d[3] = c[3];

          // Each sub-group is responsible for computing a fragment size of 16*8
          // elements of matrix D.
          // Each work item computes 4 elements of matrix D by gathering
          // their corresponding row & col matrix fragments of length k (8)
          // from A & B matrices respectively using below mapping logic:
          // row0 = (lane >> 2)    & row1 = (lane >> 2) + 8
          // col0 = (lane % 4) * 2 & col1 = (lane % 4) * 2 + 1
          // As each row & col fragment of A & B matrices is distributed across
          // 4 work items, each iteration of below loop loads a partial fragment
          // of matrix A (row) and matrix B (col) using the row & col offsets.
          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[0], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[2], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[1], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[3], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[1], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[1], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + 4 + i);

            ಸ್ವಯಂಚಾಲಿತ ra = reinterpret_cast<ABType*>(recv_a);
            ಸ್ವಯಂಚಾಲಿತ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (std::is_integral_v<ABType>) {
          // Init D matrix with fragments of C matrix
          *d[0] = c[0];
          *d[1] = c[1];
          *d[2] = c[2];
          *d[3] = c[3];

          // Each sub-group is responsible for computing a fragment size of 16*8
          // elements of matrix D.
          // Each work item computes 4 elements of matrix D by gathering
          // their corresponding row & col matrix fragments of length k (8)
          // from A & B matrices respectively using below mapping logic:
          // row0 = (lane >> 2)    & row1 = (lane >> 2) + 8
          // col0 = (lane % 4) * 2 & col1 = (lane % 4) * 2 + 1
          // As each row & col fragment of A & B matrices is distributed across
          // 4 work items, each iteration of below loop loads a partial fragment
          // of matrix A (row) and matrix B (col) using the row & col offsets.
          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[0], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[1], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 4);

            ಸ್ವಯಂಚಾಲಿತ ra = reinterpret_cast<ABType*>(recv_a);
            ಸ್ವಯಂಚಾಲಿತ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ constexpr (M == 16 && N == 8 && K == 32) {
        ಆದರೆ constexpr (std::is_integral_v<ABType>) {
          // Init D matrix with fragments of C matrix
          *d[0] = c[0];
          *d[1] = c[1];
          *d[2] = c[2];
          *d[3] = c[3];

          // Each sub-group is responsible for computing a fragment size of 16*8
          // elements of matrix D.
          // Each work item computes 4 elements of matrix D by gathering
          // their corresponding row & col matrix fragments of length k (32)
          // from A & B matrices respectively using below mapping logic:
          // row0 = (lane >> 2)    & row1 = (lane >> 2) + 8
          // col0 = ((lane % 4) * 4) + (i & 0x3) & col1 = ((lane % 4) * 4) + (i
          // & 0x3) As each row & col fragment of A & B matrices is distributed
          // across 4 work items, each iteration of below loop loads a partial
          // fragment of matrix A (row) and matrix B (col) using the row & col
          // offsets.
          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[0], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[1], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[0], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 4);

            ಸ್ವಯಂಚಾಲಿತ a = reinterpret_cast<ABType*>(recv_a);
            ಸ್ವಯಂಚಾಲಿತ b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            ಬಗೆ_ಹೆಸರು MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[2], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, a[3], ಅಡ್ಡಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[1], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_ಗುಂಪು(sg, b[1], ಲಂಬಸಾಲು_ತುಂಬು_ಸ್ಥಾನಾಂತರ + i + 4);

            ಸ್ವಯಂಚಾಲಿತ a = reinterpret_cast<ABType*>(recv_a);
            ಸ್ವಯಂಚಾಲಿತ b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }
        }
      }
    }
} // COPY from DPCT head files

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_DPCT_HELPER_HPP

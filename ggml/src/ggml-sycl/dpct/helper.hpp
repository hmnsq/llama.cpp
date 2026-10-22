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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DPCT_HELPER_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DPCT_HELPER_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <iostream>
#ਅਨ੍ਤਰ੍ਭਾਵ <map>

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/half_type.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <oneapi/mkl.hpp>

#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/mman.h>
#ਜਾਂਜੇ defined(_WIN64)
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NOMINMAX
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOMINMAX
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>
#ਨਹੀਂਤਾਂ
#ਗਲਤੀ "Only support Windows and Linux."
#ਜੇ_ਅੰਤ

#ਜੇ defined(__linux__)
#ਅਨ੍ਤਰ੍ਭਾਵ <unistd.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <sys/syscall.h>
#ਜੇ_ਅੰਤ
#ਜੇ defined(_WIN64)
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ NOMINMAX
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ NOMINMAX
#ਜੇ_ਅੰਤ
#ਅਨ੍ਤਰ੍ਭਾਵ <windows.h>
#ਜੇ_ਅੰਤ


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DPCT_COMPATIBILITY_ਆਰਜ਼ੀ (900)

#ਜੇ defined(_MSC_VER)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __dpct_align__(n) __declspec(align(n))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __dpct_inline__ __forceinline
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __dpct_align__(n) __attribute__((aligned(n)))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __dpct_inline__ __inline__ __attribute__((always_inline))
#ਜੇ_ਅੰਤ

#ਜੇ defined(_MSC_VER)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __dpct_noinline__ __declspec(noinline)
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __dpct_noinline__ __attribute__((noinline))
#ਜੇ_ਅੰਤ

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DPCT_ਅਣਵਰਤਿਆ(x) (ਖਾਲੀ)(x)

[[noreturn]] ਸਤਰਅੰਦਰ ਖਾਲੀ _abort(ਅਚਲ ਅੱਖਰ * str) {
    std::cerr << str << std::ਸਤਰ_ਅੰਤ;
    std::abort();
}

ਸਤਰਅੰਦਰ std::string get_device_type_name(ਅਚਲ sycl::device &Device) {
    ਸਵੈਚਾਲਿਤ DeviceType = Device.get_info<sycl::info::device::device_type>();
    ਚੋਣ (DeviceType) {
    ਹਾਲਤ sycl::info::device_type::cpu:
        ਵਾਪਸ "cpu";
    ਹਾਲਤ sycl::info::device_type::gpu:
        ਵਾਪਸ "gpu";
    ਹਾਲਤ sycl::info::device_type::host:
        ਵਾਪਸ "host";
    ਹਾਲਤ sycl::info::device_type::accelerator:
        ਵਾਪਸ "acc";
    ਮੂਲ:
        ਵਾਪਸ "unknown";
    }
}

ਸਤਰਅੰਦਰ std::string get_device_backend_and_type(ਅਚਲ sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    ਵਾਪਸ device_type.str();
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Ts> ਬਣਤਰ ਮੈਟ੍ਰਿਕਸ_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     ਮੁੱਲ_info[2];
    std::int64_t           ਮਾਪ_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

ਨਾਮਾਕਾਸ਼ dpct
{
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::queue *ਕਤਾਰ_ptr;
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::event *event_ptr;
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਅੱਖਰ *device_ptr;
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint8_t byte_t;
    ਕਿਸਮਪਰਿਭਾਸ਼ਾ sycl::ਬਫਰ<byte_t> ਬਫਰ_t;

    /// SYCL default exception handler
    ਸਤਰਅੰਦਰ ਸਵੈਚਾਲਿਤ exception_handler = [](sycl::exception_list exceptions)
    {
        ਲਈ (std::exception_ptr ਅਚਲ &e : exceptions)
        {
            ਕੋਸ਼ਿਸ਼
            {
                std::rethrow_exception(e);
            }
            ਫੜੋ (sycl::exception ਅਚਲ &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::ਸਤਰ_ਅੰਤ
                          << e.what() << std::ਸਤਰ_ਅੰਤ
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::ਸਤਰ_ਅੰਤ;
            }
        }
    };

    ਗਿਣਤੀ error_code
    {
        success = 0,
        default_error = 999
    };

    ਗਿਣਤੀ memcpy_direction
    {
        host_ਮੰਜ਼ਿਲ_host,
        host_ਮੰਜ਼ਿਲ_device,
        device_ਮੰਜ਼ਿਲ_host,
        device_ਮੰਜ਼ਿਲ_device,
        automatic
    };

    ਗਿਣਤੀ ਸਮ੍ਰਿਤੀ_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    ਗਿਣਤੀ ਵਰਗ library_ਅੰਕੜੇ_t : ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ
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
        library_ਅੰਕੜੇ_t_ਮਾਪ
    };

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਬਣਤਰ ਅੰਕੜੇ_Type_2
    {
        ਵਰਤੋਂ T2 = T;
    };
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਬਣਤਰ ਅੰਕੜੇ_Type_2<sycl::vec<T, 2>>
    {
        ਵਰਤੋਂ T2 = std::complex<T>;
    };

    ਸਥਿਰ ਖਾਲੀ destroy_event(event_ptr event)
    {
        ਮਿਟਾਓ event;
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ get_tid()
    {
#ਜੇ defined(__linux__)
        ਵਾਪਸ syscall(SYS_gettid);
#ਜਾਂਜੇ defined(_WIN64)
        ਵਾਪਸ GetCurrentThreadId();
#ਨਹੀਂਤਾਂ
#ਗਲਤੀ "Only support Windows and Linux."
#ਜੇ_ਅੰਤ
    }

    ਨਾਮਾਕਾਸ਼ detail
    {
        ਸਥਿਰ ਖਾਲੀ get_ਸੰਸਕਰਣ(ਅਚਲ sycl::device &dev, ਪੂਰਨ &major, ਪੂਰਨ &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::ਸੰਸਕਰਣ_2>();
            std::string::size_type i = 0;
            ਜਦੋਂ (i < ver.size()) {
              ਜੇ (ਅੰਕ_ਹੈ(ver[i]))
                ਰੋਕੋ;
              i++;
            }
            major = std::stoi(&(ver[i]));
            ਜਦੋਂ (i < ver.size()) {
              ਜੇ (ver[i] == '.')
                ਰੋਕੋ;
              i++;
            }
            ਜੇ (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } ਨਹੀਂਤਾਂ {
              // c.
              minor = 0;
            }
        }

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਟੈਗ, ਕਿਸਮ_ਨਾਂ T>
        ਵਰਗ generic_error_type
        {
        ਜਨਤਕ:
            generic_error_type() = ਮੂਲ;
            generic_error_type(T value) : value{value} {}
            ਸੰਚਾਲਕ T() ਅਚਲ { ਵਾਪਸ value; }

        ਨਿਜੀ:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    ਵਰਗ dim3 {
        ਜਨਤਕ:
        ਬਿਨਾਨਿਸ਼ਾਨੀ x, y, z;

        constexpr dim3(ਬਿਨਾਨਿਸ਼ਾਨੀ x = 1, ਬਿਨਾਨਿਸ਼ਾਨੀ y = 1, ਬਿਨਾਨਿਸ਼ਾਨੀ z = 1)
            : x(x), y(y), z(z) {}

        dim3(ਅਚਲ sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        ਸੰਚਾਲਕ sycl::range<3>() ਅਚਲ { ਵਾਪਸ sycl::range<3>(z, y, x); }
    }; // namespace dim3

    ਸਤਰਅੰਦਰ dim3 ਸੰਚਾਲਕ*(ਅਚਲ dim3 &a, ਅਚਲ dim3 &b) {
    ਵਾਪਸ dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    ਵਰਗ pitched_ਅੰਕੜੇ
    {
    ਜਨਤਕ:
        pitched_ਅੰਕੜੇ() : pitched_ਅੰਕੜੇ(ਖਾਲੀ_ਸੂਚਕ, 0, 0, 0) {}
        pitched_ਅੰਕੜੇ(ਖਾਲੀ *ਅੰਕੜੇ, size_t pitch, size_t x, size_t y)
            : ਅੰਕੜੇ_2(ਅੰਕੜੇ), _pitch(pitch), _x(x), _y(y) {}

        ਖਾਲੀ *get_ਅੰਕੜੇ_ptr() { ਵਾਪਸ ਅੰਕੜੇ_2; }
        ਖਾਲੀ ਸੈੱਟ_ਅੰਕੜੇ_ptr(ਖਾਲੀ *ਅੰਕੜੇ) { ਅੰਕੜੇ_2 = ਅੰਕੜੇ; }

        size_t get_pitch() { ਵਾਪਸ _pitch; }
        ਖਾਲੀ ਸੈੱਟ_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { ਵਾਪਸ _x; }
        ਖਾਲੀ ਸੈੱਟ_x(size_t x) { _x = x; }

        size_t get_y() { ਵਾਪਸ _y; }
        ਖਾਲੀ ਸੈੱਟ_y(size_t y) { _y = y; }

    ਨਿਜੀ:
        ਖਾਲੀ *ਅੰਕੜੇ_2;
        size_t _pitch, _x, _y;
    };

    ਵਰਗ device_info
    {
    ਜਨਤਕ:
        // get interface
        ਅਚਲ ਅੱਖਰ *get_name() ਅਚਲ { ਵਾਪਸ _name; }
        ਅੱਖਰ *get_name() { ਵਾਪਸ _name; }
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Work_ਵਸਤੂ_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_ਵਸਤੂ_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_ਵਸਤੂ_Sizes_Ty, ਪੂਰਨ *>,
                                   ਪੂਰਨ> = 0>
        ਸਵੈਚਾਲਿਤ get_max_work_ਵਸਤੂ_sizes() ਅਚਲ
        {
            ਜੇ constexpr (std::is_same_v<Work_ਵਸਤੂ_Sizes_Ty, sycl::range<3>>)
                ਵਾਪਸ sycl::range<3>(max_work_ਵਸਤੂ_sizes_i[0],
                                      max_work_ਵਸਤੂ_sizes_i[1],
                                      max_work_ਵਸਤੂ_sizes_i[2]);
            ਨਹੀਂਤਾਂ
            {
                ਵਾਪਸ max_work_ਵਸਤੂ_sizes_i;
            }
        }
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ Work_ਵਸਤੂ_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_ਵਸਤੂ_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_ਵਸਤੂ_Sizes_Ty, ਪੂਰਨ *>,
                                   ਪੂਰਨ> = 0>
        ਸਵੈਚਾਲਿਤ get_max_work_ਵਸਤੂ_sizes()
        {
            ਜੇ constexpr (std::is_same_v<Work_ਵਸਤੂ_Sizes_Ty, sycl::range<3>>)
                ਵਾਪਸ sycl::range<3>(max_work_ਵਸਤੂ_sizes_i[0],
                                      max_work_ਵਸਤੂ_sizes_i[1],
                                      max_work_ਵਸਤੂ_sizes_i[2]);
            ਨਹੀਂਤਾਂ
            {
                ਵਾਪਸ max_work_ਵਸਤੂ_sizes_i;
            }
        }
        ਤਾਰਕਿਕ get_host_unified_ਸਮ੍ਰਿਤੀ() ਅਚਲ { ਵਾਪਸ host_unified_ਸਮ੍ਰਿਤੀ; }
        ਪੂਰਨ get_major_ਸੰਸਕਰਣ() ਅਚਲ { ਵਾਪਸ _major; }
        ਪੂਰਨ get_minor_ਸੰਸਕਰਣ() ਅਚਲ { ਵਾਪਸ _minor; }
        ਪੂਰਨ get_integrated() ਅਚਲ { ਵਾਪਸ _integrated; }
        ਪੂਰਨ get_max_clock_frequency() ਅਚਲ { ਵਾਪਸ _frequency; }
        ਪੂਰਨ get_max_compute_units() ਅਚਲ { ਵਾਪਸ _max_compute_units; }
        ਪੂਰਨ get_max_work_ਸਮੂਹ_ਮਾਪ() ਅਚਲ { ਵਾਪਸ max_work_ਸਮੂਹ_ਮਾਪ_2; }
        ਪੂਰਨ get_max_sub_ਸਮੂਹ_ਮਾਪ() ਅਚਲ { ਵਾਪਸ max_sub_ਸਮੂਹ_ਮਾਪ; }
        ਪੂਰਨ get_max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit() ਅਚਲ
        {
            ਵਾਪਸ max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit;
        }
        ਪੂਰਨ get_max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ() ਅਚਲ
        {
            ਵਾਪਸ max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ;
        }
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ NDRange_ਮਾਪ_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_ਮਾਪ_Ty, size_t *> ||
                                       std::is_same_v<NDRange_ਮਾਪ_Ty, ਪੂਰਨ *>,
                                   ਪੂਰਨ> = 0>
        ਸਵੈਚਾਲਿਤ get_max_nd_range_ਮਾਪ() ਅਚਲ
        {
            ਜੇ constexpr (std::is_same_v<NDRange_ਮਾਪ_Ty, size_t *>)
                ਵਾਪਸ max_nd_range_ਮਾਪ;
            ਨਹੀਂਤਾਂ
                ਵਾਪਸ max_nd_range_ਮਾਪ_i;
        }
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ NDRange_ਮਾਪ_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_ਮਾਪ_Ty, size_t *> ||
                                       std::is_same_v<NDRange_ਮਾਪ_Ty, ਪੂਰਨ *>,
                                   ਪੂਰਨ> = 0>
        ਸਵੈਚਾਲਿਤ get_max_nd_range_ਮਾਪ()
        {
            ਜੇ constexpr (std::is_same_v<NDRange_ਮਾਪ_Ty, size_t *>)
                ਵਾਪਸ max_nd_range_ਮਾਪ;
            ਨਹੀਂਤਾਂ
                ਵਾਪਸ max_nd_range_ਮਾਪ_i;
        }
        size_t get_global_mem_ਮਾਪ() ਅਚਲ { ਵਾਪਸ global_mem_ਮਾਪ_2; }
        size_t get_local_mem_ਮਾਪ() ਅਚਲ { ਵਾਪਸ local_mem_ਮਾਪ; }
        size_t get_max_mem_alloc_ਮਾਪ() ਅਚਲ { ਵਾਪਸ max_mem_alloc_ਮਾਪ; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ get_ਸਮ੍ਰਿਤੀ_clock_ਦਰ() ਅਚਲ { ਵਾਪਸ ਸਮ੍ਰਿਤੀ_clock_ਦਰ; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ get_ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ() ਅਚਲ { ਵਾਪਸ ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ; }
        uint32_t get_device_id() ਅਚਲ { ਵਾਪਸ _device_id; }
        std::array<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ, 16> get_uuid() ਅਚਲ { ਵਾਪਸ _uuid; }
        /// Returns global memory cache size in bytes.
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ get_global_mem_cache_ਮਾਪ() ਅਚਲ
        {
            ਵਾਪਸ global_mem_cache_ਮਾਪ;
        }

        // set interface
        ਖਾਲੀ ਸੈੱਟ_name(ਅਚਲ ਅੱਖਰ *name)
        {
            size_t ਲੰਮਾਈ = ਲੜੀ_ਲੰਬਾਈ(name);
            ਜੇ (ਲੰਮਾਈ < 256)
            {
                std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(_name, name, ਲੰਮਾਈ + 1);
            }
            ਨਹੀਂਤਾਂ
            {
                std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(_name, name, 255);
                _name[255] = '\0';
            }
        }
        ਖਾਲੀ ਸੈੱਟ_max_work_ਵਸਤੂ_sizes(ਅਚਲ sycl::range<3> max_work_ਵਸਤੂ_sizes)
        {
            ਲਈ (ਪੂਰਨ i = 0; i < 3; ++i)
                max_work_ਵਸਤੂ_sizes_i[i] = max_work_ਵਸਤੂ_sizes[i];
        }
        [[deprecated]] ਖਾਲੀ
        ਸੈੱਟ_max_work_ਵਸਤੂ_sizes(ਅਚਲ sycl::id<3> max_work_ਵਸਤੂ_sizes)
        {
            ਲਈ (ਪੂਰਨ i = 0; i < 3; ++i)
            {
                max_work_ਵਸਤੂ_sizes_i[i] = max_work_ਵਸਤੂ_sizes[i];
            }
        }
        ਖਾਲੀ ਸੈੱਟ_host_unified_ਸਮ੍ਰਿਤੀ(ਤਾਰਕਿਕ host_unified_ਸਮ੍ਰਿਤੀ_2)
        {
            host_unified_ਸਮ੍ਰਿਤੀ = host_unified_ਸਮ੍ਰਿਤੀ_2;
        }
        ਖਾਲੀ ਸੈੱਟ_major_ਸੰਸਕਰਣ(ਪੂਰਨ major) { _major = major; }
        ਖਾਲੀ ਸੈੱਟ_minor_ਸੰਸਕਰਣ(ਪੂਰਨ minor) { _minor = minor; }
        ਖਾਲੀ ਸੈੱਟ_integrated(ਪੂਰਨ integrated) { _integrated = integrated; }
        ਖਾਲੀ ਸੈੱਟ_max_clock_frequency(ਪੂਰਨ frequency) { _frequency = frequency; }
        ਖਾਲੀ ਸੈੱਟ_max_compute_units(ਪੂਰਨ max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        ਖਾਲੀ ਸੈੱਟ_global_mem_ਮਾਪ(size_t global_mem_ਮਾਪ)
        {
            global_mem_ਮਾਪ_2 = global_mem_ਮਾਪ;
        }
        ਖਾਲੀ ਸੈੱਟ_local_mem_ਮਾਪ(size_t local_mem_ਮਾਪ_2)
        {
            local_mem_ਮਾਪ = local_mem_ਮਾਪ_2;
        }
        ਖਾਲੀ ਸੈੱਟ_max_mem_alloc_ਮਾਪ(size_t max_mem_alloc_ਮਾਪ_2)
        {
            max_mem_alloc_ਮਾਪ = max_mem_alloc_ਮਾਪ_2;
        }
        ਖਾਲੀ ਸੈੱਟ_max_work_ਸਮੂਹ_ਮਾਪ(ਪੂਰਨ max_work_ਸਮੂਹ_ਮਾਪ)
        {
            max_work_ਸਮੂਹ_ਮਾਪ_2 = max_work_ਸਮੂਹ_ਮਾਪ;
        }
        ਖਾਲੀ ਸੈੱਟ_max_sub_ਸਮੂਹ_ਮਾਪ(ਪੂਰਨ max_sub_ਸਮੂਹ_ਮਾਪ_2)
        {
            max_sub_ਸਮੂਹ_ਮਾਪ = max_sub_ਸਮੂਹ_ਮਾਪ_2;
        }
        ਖਾਲੀ
        ਸੈੱਟ_max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit(ਪੂਰਨ max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit_2)
        {
            max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit = max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit_2;
        }
        ਖਾਲੀ ਸੈੱਟ_max_nd_range_ਮਾਪ(ਪੂਰਨ max_nd_range_ਮਾਪ_2[])
        {
            ਲਈ (ਪੂਰਨ i = 0; i < 3; i++)
            {
                max_nd_range_ਮਾਪ[i] = max_nd_range_ਮਾਪ_2[i];
                max_nd_range_ਮਾਪ_i[i] = max_nd_range_ਮਾਪ_2[i];
            }
        }
        ਖਾਲੀ ਸੈੱਟ_ਸਮ੍ਰਿਤੀ_clock_ਦਰ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2)
        {
            ਸਮ੍ਰਿਤੀ_clock_ਦਰ = ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2;
        }
        ਖਾਲੀ ਸੈੱਟ_ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2)
        {
            ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ = ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2;
        }
        ਖਾਲੀ
        ਸੈੱਟ_max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ(ਪੂਰਨ max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ_2)
        {
            max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ = max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ_2;
        }
        ਖਾਲੀ ਸੈੱਟ_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        ਖਾਲੀ ਸੈੱਟ_uuid(std::array<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        ਖਾਲੀ ਸੈੱਟ_global_mem_cache_ਮਾਪ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ global_mem_cache_ਮਾਪ_2)
        {
            global_mem_cache_ਮਾਪ = global_mem_cache_ਮਾਪ_2;
        }

    ਨਿਜੀ:
        ਅੱਖਰ _name[256];
        ਪੂਰਨ max_work_ਵਸਤੂ_sizes_i[3];
        ਤਾਰਕਿਕ host_unified_ਸਮ੍ਰਿਤੀ = ਝੂਠ;
        ਪੂਰਨ _major;
        ਪੂਰਨ _minor;
        ਪੂਰਨ _integrated = 0;
        ਪੂਰਨ _frequency;
        // Set estimated value 3200000 kHz as default value.
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਸਮ੍ਰਿਤੀ_clock_ਦਰ = 3200000;
        // Set estimated value 64 bits as default value.
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ = 64;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ global_mem_cache_ਮਾਪ;
        ਪੂਰਨ _max_compute_units;
        ਪੂਰਨ max_work_ਸਮੂਹ_ਮਾਪ_2;
        ਪੂਰਨ max_sub_ਸਮੂਹ_ਮਾਪ;
        ਪੂਰਨ max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit;
        ਪੂਰਨ max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ;
        size_t global_mem_ਮਾਪ_2;
        size_t local_mem_ਮਾਪ;
        size_t max_mem_alloc_ਮਾਪ;
        size_t max_nd_range_ਮਾਪ[3];
        ਪੂਰਨ max_nd_range_ਮਾਪ_i[3];
        uint32_t _device_id;
        std::array<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ, 16> _uuid;
    };

    ਸਥਿਰ ਪੂਰਨ get_major_ਸੰਸਕਰਣ(ਅਚਲ sycl::device &dev)
    {
        ਪੂਰਨ major, minor;
        detail::get_ਸੰਸਕਰਣ(dev, major, minor);
        ਵਾਪਸ major;
    }

    ਸਥਿਰ ਪੂਰਨ get_minor_ਸੰਸਕਰਣ(ਅਚਲ sycl::device &dev)
    {
        ਪੂਰਨ major, minor;
        detail::get_ਸੰਸਕਰਣ(dev, major, minor);
        ਵਾਪਸ minor;
    }

    ਸਥਿਰ ਖਾਲੀ get_device_info(device_info &ਬਾਹਰਲਾ, ਅਚਲ sycl::device &dev)
    {
        device_info prop;
        prop.ਸੈੱਟ_name(dev.get_info<sycl::info::device::name>().c_str());

        ਪੂਰਨ major, minor;
        detail::get_ਸੰਸਕਰਣ(dev, major, minor);
        prop.ਸੈੱਟ_major_ਸੰਸਕਰਣ(major);
        prop.ਸੈੱਟ_minor_ਸੰਸਕਰਣ(minor);

        prop.ਸੈੱਟ_max_work_ਵਸਤੂ_sizes(
#ਜੇ (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_ਵਸਤੂ_sizes>());
#ਨਹੀਂਤਾਂ
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_ਵਸਤੂ_sizes<3>>());
#ਜੇ_ਅੰਤ
        prop.ਸੈੱਟ_host_unified_ਸਮ੍ਰਿਤੀ(dev.has(sycl::aspect::usm_host_allocations));

        prop.ਸੈੱਟ_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.ਸੈੱਟ_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.ਸੈੱਟ_max_work_ਸਮੂਹ_ਮਾਪ(
            dev.get_info<sycl::info::device::max_work_ਸਮੂਹ_ਮਾਪ>());
        prop.ਸੈੱਟ_global_mem_ਮਾਪ(dev.get_info<sycl::info::device::global_mem_ਮਾਪ>());
        prop.ਸੈੱਟ_local_mem_ਮਾਪ(dev.get_info<sycl::info::device::local_mem_ਮਾਪ_2>());
        prop.ਸੈੱਟ_max_mem_alloc_ਮਾਪ(dev.get_info<sycl::info::device::max_mem_alloc_ਮਾਪ_2>());

#ਜੇ (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        ਜੇ (dev.has(sycl::aspect::ext_intel_ਸਮ੍ਰਿਤੀ_clock_ਦਰ))
        {
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ tmp =
                dev.get_info<sycl::ext::intel::info::device::ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2>();
            ਜੇ (tmp != 0)
                prop.ਸੈੱਟ_ਸਮ੍ਰਿਤੀ_clock_ਦਰ(1000 * tmp);
        }
        ਜੇ (dev.has(sycl::aspect::ext_intel_ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ))
        {
            prop.ਸੈੱਟ_ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ(
                dev.get_info<sycl::ext::intel::info::device::ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2>());
        }
        ਜੇ (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.ਸੈੱਟ_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        ਜੇ (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.ਸੈੱਟ_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#ਜਾਂਜੇ defined(_MSC_VER) && !defined(__clang__)
#ਹਦਾਇਤ ਸੁਨੇਹਾ("get_device_info: querying ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2 and \
        ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2 are not supported by the compiler ਵਰਤਿਆ. \
        Use 3200000 kHz as ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2 ਮੂਲ value. \
        Use 64 ਬਿੱਟ as ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2 ਮੂਲ value.")
#ਨਹੀਂਤਾਂ
#warning "get_device_info: querying ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2 and \
        ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2 are not supported by the compiler ਵਰਤਿਆ. \
        Use 3200000 kHz as ਸਮ੍ਰਿਤੀ_clock_ਦਰ_2 ਮੂਲ value. \
        Use 64 ਬਿੱਟ as ਸਮ੍ਰਿਤੀ_bus_ਚੌੜਾਈ_2 ਮੂਲ value."
#ਜੇ_ਅੰਤ

        size_t max_sub_ਸਮੂਹ_ਮਾਪ_2 = 1;
        std::vector<size_t> sub_ਸਮੂਹ_sizes =
            dev.get_info<sycl::info::device::sub_ਸਮੂਹ_sizes>();

        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &sub_ਸਮੂਹ_ਮਾਪ : sub_ਸਮੂਹ_sizes)
        {
            ਜੇ (max_sub_ਸਮੂਹ_ਮਾਪ_2 < sub_ਸਮੂਹ_ਮਾਪ)
                max_sub_ਸਮੂਹ_ਮਾਪ_2 = sub_ਸਮੂਹ_ਮਾਪ;
        }

        prop.ਸੈੱਟ_max_sub_ਸਮੂਹ_ਮਾਪ(max_sub_ਸਮੂਹ_ਮਾਪ_2);

        prop.ਸੈੱਟ_max_work_ਚੀਜ਼ਾਂ_ਪ੍ਰਤੀ_compute_unit(
            dev.get_info<sycl::info::device::max_work_ਸਮੂਹ_ਮਾਪ>());
        ਪੂਰਨ max_nd_range_ਮਾਪ_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.ਸੈੱਟ_max_nd_range_ਮਾਪ(max_nd_range_ਮਾਪ_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.ਸੈੱਟ_max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ(65536);

        prop.ਸੈੱਟ_global_mem_cache_ਮਾਪ(
            dev.get_info<sycl::info::device::global_mem_cache_ਮਾਪ_2>());
        ਬਾਹਰਲਾ = prop;
    }

    /// dpct device extension
    ਵਰਗ device_ext : ਜਨਤਕ sycl::device {
      ਕਿਸਮਪਰਿਭਾਸ਼ਾ std::mutex mutex_type;

     ਜਨਤਕ:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        clear_queues();
      }
      device_ext(ਅਚਲ sycl::device &ਅਧਾਰ) : sycl::device(ਅਧਾਰ) {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        init_queues();
      }

      ਪੂਰਨ is_native_atomic_supported() { ਵਾਪਸ 0; }
      ਪੂਰਨ get_major_ਸੰਸਕਰਣ() ਅਚਲ { ਵਾਪਸ dpct::get_major_ਸੰਸਕਰਣ(*ਇਹ); }

      ਪੂਰਨ get_minor_ਸੰਸਕਰਣ() ਅਚਲ { ਵਾਪਸ dpct::get_minor_ਸੰਸਕਰਣ(*ਇਹ); }

      ਪੂਰਨ get_max_compute_units() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      ਪੂਰਨ get_max_clock_frequency() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_max_clock_frequency();
      }

      ਪੂਰਨ get_integrated() ਅਚਲ { ਵਾਪਸ get_device_info().get_integrated(); }

      ਪੂਰਨ get_max_sub_ਸਮੂਹ_ਮਾਪ() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_max_sub_ਸਮੂਹ_ਮਾਪ();
      }

      ਪੂਰਨ get_max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_max_ਰਜਿਸਟਰ_ਮਾਪ_ਪ੍ਰਤੀ_work_ਸਮੂਹ();
      }

      ਪੂਰਨ get_max_work_ਸਮੂਹ_ਮਾਪ() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_max_work_ਸਮੂਹ_ਮਾਪ();
      }

      ਪੂਰਨ get_mem_ਅਧਾਰ_addr_align() ਅਚਲ {
        ਵਾਪਸ get_info<sycl::info::device::mem_ਅਧਾਰ_addr_align>();
      }

      size_t get_global_mem_ਮਾਪ() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_global_mem_ਮਾਪ();
      }

      size_t get_max_mem_alloc_ਮਾਪ() ਅਚਲ {
        ਵਾਪਸ get_device_info().get_max_mem_alloc_ਮਾਪ();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      ਖਾਲੀ get_ਸਮ੍ਰਿਤੀ_info(size_t &free_ਸਮ੍ਰਿਤੀ, size_t &ਕੁੱਲ_ਸਮ੍ਰਿਤੀ) {
        ਕੁੱਲ_ਸਮ੍ਰਿਤੀ = get_device_info().get_global_mem_ਮਾਪ();
        ਅਚਲ ਅੱਖਰ *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#ਜੇ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        ਜੇ (!has(sycl::aspect::ext_intel_free_ਸਮ੍ਰਿਤੀ)) {
          std::cerr << warning_info << std::ਸਤਰ_ਅੰਤ;
          free_ਸਮ੍ਰਿਤੀ = ਕੁੱਲ_ਸਮ੍ਰਿਤੀ;
        } ਨਹੀਂਤਾਂ {
          free_ਸਮ੍ਰਿਤੀ = get_info<sycl::ext::intel::info::device::free_ਸਮ੍ਰਿਤੀ>();
        }
#ਨਹੀਂਤਾਂ
        std::cerr << warning_info << std::ਸਤਰ_ਅੰਤ;
        free_ਸਮ੍ਰਿਤੀ = ਕੁੱਲ_ਸਮ੍ਰਿਤੀ;
#ਜੇ defined(_MSC_VER) && !defined(__clang__)
#ਹਦਾਇਤ ਸੁਨੇਹਾ("Querying the number of bytes of free memory is not supported")
#ਨਹੀਂਤਾਂ
#warning "Querying the number of bytes of free memory is not supported"
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ
      }

      ਖਾਲੀ get_device_info(device_info &ਬਾਹਰਲਾ) ਅਚਲ {
        dpct::get_device_info(ਬਾਹਰਲਾ, *ਇਹ);
      }

      device_info get_device_info() ਅਚਲ {
        device_info prop;
        dpct::get_device_info(prop, *ਇਹ);
        ਵਾਪਸ prop;
      }

      ਖਾਲੀ reset() {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ() { ਵਾਪਸ q_ਵਿੱਚ_ਕ੍ਰਮ; }

      sycl::queue &ਬਾਹਰਲਾ_ਦਾ_ਕ੍ਰਮ_ਕਤਾਰ() { ਵਾਪਸ q_ਬਾਹਰਲਾ_ਦਾ_ਕ੍ਰਮ; }

      sycl::queue &default_ਕਤਾਰ() { ਵਾਪਸ ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ(); }

      ਖਾਲੀ queues_wait_and_throw() {
        std::unique_lock<mutex_type> ਤਾਲਾ(m_mutex);
        ਤਾਲਾ.unlock();
        ਲਈ (ਸਵੈਚਾਲਿਤ &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        ਤਾਲਾ.ਤਾਲਾ();
      }

      sycl::queue create_ਕਤਾਰ(ਤਾਰਕਿਕ enable_exception_handler = ਝੂਠ) {
        ਵਾਪਸ create_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ(enable_exception_handler);
      }

      sycl::queue create_ਕਤਾਰ(sycl::device device,
                               ਤਾਰਕਿਕ enable_exception_handler = ਝੂਠ) {
        ਵਾਪਸ create_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ(device, enable_exception_handler);
      }

      sycl::queue create_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ(ਤਾਰਕਿਕ enable_exception_handler = ਝੂਠ) {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        ਵਾਪਸ create_ਕਤਾਰ_impl(enable_exception_handler,
                                 sycl::property::queue::ਵਿੱਚ_ਕ੍ਰਮ());
      }

      sycl::queue create_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ(sycl::device device,
                                        ਤਾਰਕਿਕ enable_exception_handler = ਝੂਠ) {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        ਵਾਪਸ create_ਕਤਾਰ_impl(device, enable_exception_handler,
                                 sycl::property::queue::ਵਿੱਚ_ਕ੍ਰਮ());
      }

      sycl::queue create_ਬਾਹਰਲਾ_ਦਾ_ਕ੍ਰਮ_ਕਤਾਰ(
          ਤਾਰਕਿਕ enable_exception_handler = ਝੂਠ) {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        ਵਾਪਸ create_ਕਤਾਰ_impl(enable_exception_handler);
      }

      ਖਾਲੀ destroy_ਕਤਾਰ(sycl::queue queue) {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ਅਚਲ sycl::queue &q) -> ਤਾਰਕਿਕ
                                    {
                                        ਵਾਪਸ q == queue;
                                    }),
                    _queues.end());
      }
      ਖਾਲੀ ਸੈੱਟ_saved_ਕਤਾਰ(sycl::queue q) {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        saved_ਕਤਾਰ = q;
      }
      sycl::queue get_saved_ਕਤਾਰ() ਅਚਲ {
        std::lock_guard<mutex_type> ਤਾਲਾ(m_mutex);
        ਵਾਪਸ saved_ਕਤਾਰ;
      }

     ਨਿਜੀ:
      ਖਾਲੀ clear_queues() { _queues.clear(); }

      ਖਾਲੀ init_queues() {
        q_ਵਿੱਚ_ਕ੍ਰਮ =
            create_ਕਤਾਰ_impl(ਸੱਚ, sycl::property::queue::ਵਿੱਚ_ਕ੍ਰਮ());
        q_ਬਾਹਰਲਾ_ਦਾ_ਕ੍ਰਮ = create_ਕਤਾਰ_impl(ਸੱਚ);
        saved_ਕਤਾਰ = default_ਕਤਾਰ();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      ਸਾਂਚਾ <ਵਰਗ... Properties>
      sycl::queue create_ਕਤਾਰ_impl(ਤਾਰਕਿਕ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ਜੇ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *ਇਹ, eh,
            sycl::property_list(
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#ਜੇ_ਅੰਤ
                properties...)));

        ਵਾਪਸ _queues.back();
      }

      ਸਾਂਚਾ <ਵਰਗ... Properties>
      sycl::queue create_ਕਤਾਰ_impl(sycl::device device,
                                    ਤਾਰਕਿਕ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ਜੇ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#ਜੇ_ਅੰਤ
                            properties...)));

        ਵਾਪਸ _queues.back();
      }

      ਖਾਲੀ get_ਸੰਸਕਰਣ(ਪੂਰਨ &major, ਪੂਰਨ &minor) ਅਚਲ {
        detail::get_ਸੰਸਕਰਣ(*ਇਹ, major, minor);
      }
      sycl::queue q_ਵਿੱਚ_ਕ੍ਰਮ, q_ਬਾਹਰਲਾ_ਦਾ_ਕ੍ਰਮ;
      sycl::queue saved_ਕਤਾਰ;
      std::vector<sycl::queue> _queues;
      ਬਦਲਣਯੋਗ mutex_type m_mutex;
    };


    /// device manager
    ਵਰਗ dev_mgr
    {
    ਜਨਤਕ:
        device_ext &current_device()
        {
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ dev_id = current_device_id();
            check_id(dev_id);
            ਵਾਪਸ *_devs[dev_id];
        }
        device_ext &cpu_device() ਅਚਲ
        {
            std::lock_guard<std::recursive_mutex> ਤਾਲਾ(m_mutex);
            ਜੇ (_cpu_device == -1)
            {
                ਸੁੱਟੋ std::runtime_error("no valid cpu device");
            }
            ਨਹੀਂਤਾਂ
            {
                ਵਾਪਸ *_devs[_cpu_device];
            }
        }
        device_ext &get_device(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id) ਅਚਲ
        {
            std::lock_guard<std::recursive_mutex> ਤਾਲਾ(m_mutex);
            check_id(id);
            ਵਾਪਸ *_devs[id];
        }
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ current_device_id() ਅਚਲ
        {
            std::lock_guard<std::recursive_mutex> ਤਾਲਾ(m_mutex);
            ਸਵੈਚਾਲਿਤ it = _thread2dev_map.find(get_tid());
            ਜੇ (it != _thread2dev_map.end())
                ਵਾਪਸ it->second;
            ਵਾਪਸ DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        ਖਾਲੀ select_device(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id)
        {
            std::lock_guard<std::recursive_mutex> ਤਾਲਾ(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ device_ਗਿਣਤੀ() { ਵਾਪਸ _devs.size(); }

        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ get_device_id(ਅਚਲ sycl::device &dev)
        {
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id = 0;
            ਲਈ (ਸਵੈਚਾਲਿਤ &dev_ਵਸਤੂ : _devs)
            {
                ਜੇ (*dev_ਵਸਤੂ == dev)
                {
                    ਵਾਪਸ id;
                }
                id++;
            }
            ਵਾਪਸ -1;
        }

        ਸਤਰਅੰਦਰ std::string get_preferred_gpu_platform_name() {
            std::string ਨਤੀਜਾ;

            std::string filter = "";
            ਅੱਖਰ* env = getenv("ONEAPI_DEVICE_SELECTOR");
            ਜੇ (env) {
                ਜੇ (std::ਲੜੀ_ਖੋਜ(env, "level_zero")) {
                    filter = "level-zero";
                }
                ਨਹੀਂਤਾਂ ਜੇ (std::ਲੜੀ_ਖੋਜ(env, "opencl")) {
                    filter = "opencl";
                }
                ਨਹੀਂਤਾਂ ਜੇ (std::ਲੜੀ_ਖੋਜ(env, "cuda")) {
                    filter = "cuda";
                }
                ਨਹੀਂਤਾਂ ਜੇ (std::ਲੜੀ_ਖੋਜ(env, "hip")) {
                    filter = "hip";
                }
                ਨਹੀਂਤਾਂ {
                    ਸੁੱਟੋ std::runtime_error("invalid device filter: " + std::string(env));
                }
            } ਨਹੀਂਤਾਂ {
                ਸਵੈਚਾਲਿਤ default_device = sycl::device(sycl::default_selector_v);
                ਸਵੈਚਾਲਿਤ default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                ਜੇ (std::ਲੜੀ_ਖੋਜ(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                ਨਹੀਂਤਾਂ ਜੇ (std::ਲੜੀ_ਖੋਜ(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                ਨਹੀਂਤਾਂ ਜੇ (std::ਲੜੀ_ਖੋਜ(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            ਸਵੈਚਾਲਿਤ platform_list = sycl::platform::get_platforms();

            ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ& platform : platform_list) {
                ਸਵੈਚਾਲਿਤ devices = platform.get_devices();
                ਸਵੈਚਾਲਿਤ gpu_dev = std::find_if(devices.begin(), devices.end(), [](ਅਚਲ sycl::device& d) {
                    ਵਾਪਸ d.is_gpu();
                });

                ਜੇ (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    ਜਾਰੀ;
                }

                ਸਵੈਚਾਲਿਤ platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_ਨੀਵਾਂ_case;
                platform_name_ਨੀਵਾਂ_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_ਨੀਵਾਂ_case.begin(), ::tolower);

                ਜੇ (platform_name_ਨੀਵਾਂ_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    ਜਾਰੀ;
                }

                ਨਤੀਜਾ = platform_name;
            }

            ਜੇ (ਨਤੀਜਾ.ਖਾਲੀ_ਥਾਵਾਂ())
                ਸੁੱਟੋ std::runtime_error("can not find preferred GPU platform");

            ਵਾਪਸ ਨਤੀਜਾ;
        }

        ਸਾਂਚਾ <ਵਰਗ DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<ਪੂਰਨ, DeviceSelector, ਅਚਲ sycl::device &>>
        select_device(ਅਚਲ DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ਸਥਿਰ dev_mgr &instance()
        {
            ਸਥਿਰ dev_mgr d_m;
            ਵਾਪਸ d_m;
        }
        dev_mgr(ਅਚਲ dev_mgr &) = ਮਿਟਾਓ;
        dev_mgr &ਸੰਚਾਲਕ=(ਅਚਲ dev_mgr &) = ਮਿਟਾਓ;
        dev_mgr(dev_mgr &&) = ਮਿਟਾਓ;
        dev_mgr &ਸੰਚਾਲਕ=(dev_mgr &&) = ਮਿਟਾਓ;

    ਨਿਜੀ:
        ਬਦਲਣਯੋਗ std::recursive_mutex m_mutex;
        ਸਥਿਰ ਤਾਰਕਿਕ compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            ਜੇ(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) ਵਾਪਸ ਸੱਚ;
            ਜੇ(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) ਵਾਪਸ ਝੂਠ;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            ਵਾਪਸ prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ਸਥਿਰ ਪੂਰਨ convert_backend_ਸੂਚਕਾਂਕ(std::string & backend) {
            ਜੇ (backend == "ext_oneapi_level_zero:gpu") ਵਾਪਸ 0;
            ਜੇ (backend == "opencl:gpu") ਵਾਪਸ 1;
            ਜੇ (backend == "ext_oneapi_cuda:gpu") ਵਾਪਸ 2;
            ਜੇ (backend == "ext_oneapi_hip:gpu") ਵਾਪਸ 3;
            ਜੇ (backend == "opencl:cpu") ਵਾਪਸ 4;
            ਜੇ (backend == "opencl:acc") ਵਾਪਸ 5;
            ਲਿਖੋ("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ਸਥਿਰ ਤਾਰਕਿਕ compare_backend(std::string &backend1, std::string &backend2) {
            ਵਾਪਸ convert_backend_ਸੂਚਕਾਂਕ(backend1) < convert_backend_ਸੂਚਕਾਂਕ(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            ਜੇ (default_device.is_cpu())
                _cpu_device = 0;

            ਸਵੈਚਾਲਿਤ Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            ਸਵੈਚਾਲਿਤ preferred_platform_name = get_preferred_gpu_platform_name();

            ਜਦੋਂ (!Platforms.ਖਾਲੀ_ਥਾਵਾਂ()) {
                ਸਵੈਚਾਲਿਤ Platform = Platforms.back();
                Platforms.pop_back();
                ਸਵੈਚਾਲਿਤ platform_name = Platform.get_info<sycl::info::platform::name>();
                ਜੇ (platform_name.compare(preferred_platform_name) != 0) {
                    ਜਾਰੀ;
                }
                ਸਵੈਚਾਲਿਤ devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> ਕੁੰਜੀਆਂ;
            ਲਈ(ਸਵੈਚਾਲਿਤ it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                ਕੁੰਜੀਆਂ.push_back(it->ਪਹਿਲਾ);
            }
            std::sort(ਕੁੰਜੀਆਂ.begin(), ਕੁੰਜੀਆਂ.end(), compare_backend);

            ਲਈ (ਸਵੈਚਾਲਿਤ &ਕੁੰਜੀ : ਕੁੰਜੀਆਂ) {
                std::vector<sycl::device> devs = backend_devices[ਕੁੰਜੀ];
                std::sort(devs.begin(), devs.end(), compare_dev);
                ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            ਲਈ (ਸਵੈਚਾਲਿਤ &dev : sycl_all_devs)
            {
                ਜੇ (dev == default_device)
                {
                    ਜਾਰੀ;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                ਜੇ (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        ਖਾਲੀ check_id(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id) ਅਚਲ
        {
            ਜੇ (id >= _devs.size())
            {
                ਸੁੱਟੋ std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ> _thread2dev_map;
        ਪੂਰਨ _cpu_device = -1;
    };

    ਸਥਿਰ ਸਤਰਅੰਦਰ sycl::queue &get_default_ਕਤਾਰ()
    {
        ਵਾਪਸ dev_mgr::instance().current_device().default_ਕਤਾਰ();
    }

    ਨਾਮਾਕਾਸ਼ detail
    {
        ਗਿਣਤੀ ਵਰਗ ਸੂਚਕ_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ਸਥਿਰ ਸੂਚਕ_access_attribute get_ਸੂਚਕ_attribute(sycl::queue &q,
                                                              ਅਚਲ ਖਾਲੀ *ptr)
        {
            ਚੋਣ (sycl::get_ਸੂਚਕ_type(ptr, q.get_context()))
            {
            ਹਾਲਤ sycl::usm::alloc::unknown:
                ਵਾਪਸ ਸੂਚਕ_access_attribute::host_only;
            ਹਾਲਤ sycl::usm::alloc::device:
                ਵਾਪਸ ਸੂਚਕ_access_attribute::device_only;
            ਹਾਲਤ sycl::usm::alloc::shared:
            ਹਾਲਤ sycl::usm::alloc::host:
                ਵਾਪਸ ਸੂਚਕ_access_attribute::host_device;
            }
        }

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ArgT>
        ਸਤਰਅੰਦਰ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ)library_ਅੰਕੜੇ_t::library_ਅੰਕੜੇ_t_ਮਾਪ <=
                              std::numeric_limits<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_ਅੰਕੜੇ_t>, "Unsupported ArgT");
            ਵਾਪਸ (std::uint64_t)Val;
        }

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਪਹਿਲਾ_T, ਕਿਸਮ_ਨਾਂ... RestT>
        ਸਤਰਅੰਦਰ constexpr std::uint64_t get_type_combination_id(ਪਹਿਲਾ_T ਪਹਿਲਾ_Val,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_ਅੰਕੜੇ_t::library_ਅੰਕੜੇ_t_ਮਾਪ <=
                              std::numeric_limits<ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(ਆਕਾਰ...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<ਪਹਿਲਾ_T, library_ਅੰਕੜੇ_t>, "Unsupported FirstT");
            ਵਾਪਸ get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)ਪਹਿਲਾ_Val);
        }

        ਵਰਗ mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#ਜੇ defined(__linux__)
                mapped_ਪਤਾ_space =
                    (byte_t *)mmap(ਖਾਲੀ_ਸੂਚਕ, mapped_region_ਮਾਪ, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ਜਾਂਜੇ defined(_WIN64)
                mapped_ਪਤਾ_space = (byte_t *)VirtualAlloc(
                    ਨਿਰਾ,               // NULL specified as the base address parameter
                    mapped_region_ਮਾਪ, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    ਪੰਨਾ_NOACCESS);     // Protection = no access
#ਨਹੀਂਤਾਂ
#ਗਲਤੀ "Only support Windows and Linux."
#ਜੇ_ਅੰਤ
                ਅਗਲਾ_free = mapped_ਪਤਾ_space;
            }

        ਜਨਤਕ:
            ਵਰਤੋਂ ਬਫਰ_id_t = ਪੂਰਨ;

            ਬਣਤਰ ਵੰਡ
            {
                ਬਫਰ_t ਬਫਰ;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#ਜੇ defined(__linux__)
                munmap(mapped_ਪਤਾ_space, mapped_region_ਮਾਪ);
#ਜਾਂਜੇ defined(_WIN64)
                VirtualFree(mapped_ਪਤਾ_space, 0, MEM_RELEASE);
#ਨਹੀਂਤਾਂ
#ਗਲਤੀ "Only support Windows and Linux."
#ਜੇ_ਅੰਤ
            }

            mem_mgr(ਅਚਲ mem_mgr &) = ਮਿਟਾਓ;
            mem_mgr &ਸੰਚਾਲਕ=(ਅਚਲ mem_mgr &) = ਮਿਟਾਓ;
            mem_mgr(mem_mgr &&) = ਮਿਟਾਓ;
            mem_mgr &ਸੰਚਾਲਕ=(mem_mgr &&) = ਮਿਟਾਓ;

            /// Allocate
            ਖਾਲੀ *mem_alloc(size_t size)
            {
                ਜੇ (!size)
                    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
                std::lock_guard<std::mutex> ਤਾਲਾ(m_mutex);
                ਜੇ (ਅਗਲਾ_free + size > mapped_ਪਤਾ_space + mapped_region_ਮਾਪ)
                {
                    ਸੁੱਟੋ std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                //ਵੰਡ
                sycl::range<1> r(size);
                ਬਫਰ_t buf(r);
                ਵੰਡ A{buf, ਅਗਲਾ_free, size};
                // Map allocation to device pointer
                ਖਾਲੀ *ਨਤੀਜਾ = ਅਗਲਾ_free;
                m_map.emplace(ਅਗਲਾ_free + size, A);
                // Update pointer to the next free space.
                ਅਗਲਾ_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                ਵਾਪਸ ਨਤੀਜਾ;
            }

            /// Deallocate
            ਖਾਲੀ mem_free(ਅਚਲ ਖਾਲੀ *ptr)
            {
                ਜੇ (!ptr)
                    ਵਾਪਸ;
                std::lock_guard<std::mutex> ਤਾਲਾ(m_mutex);
                ਸਵੈਚਾਲਿਤ it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            ਵੰਡ translate_ptr(ਅਚਲ ਖਾਲੀ *ptr)
            {
                std::lock_guard<std::mutex> ਤਾਲਾ(m_mutex);
                ਸਵੈਚਾਲਿਤ it = get_map_iterator(ptr);
                ਵਾਪਸ it->second;
            }

            /// Check if the pointer represents device pointer or not.
            ਤਾਰਕਿਕ is_device_ptr(ਅਚਲ ਖਾਲੀ *ptr) ਅਚਲ
            {
                std::lock_guard<std::mutex> ਤਾਲਾ(m_mutex);
                ਵਾਪਸ (mapped_ਪਤਾ_space <= ptr) &&
                       (ptr < mapped_ਪਤਾ_space + mapped_region_ਮਾਪ);
            }

            /// Returns the instance of memory manager singleton.
            ਸਥਿਰ mem_mgr &instance()
            {
                ਸਥਿਰ mem_mgr m;
                ਵਾਪਸ m;
            }

        ਨਿਜੀ:
            std::map<byte_t *, ਵੰਡ> m_map;
            ਬਦਲਣਯੋਗ std::mutex m_mutex;
            byte_t *mapped_ਪਤਾ_space;
            byte_t *ਅਗਲਾ_free;
            ਅਚਲ size_t mapped_region_ਮਾਪ = 128ull * 1024 * 1024 * 1024;
            ਅਚਲ size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ਅਚਲ size_t extra_padding = 0;

            std::map<byte_t *, ਵੰਡ>::iterator get_map_iterator(ਅਚਲ ਖਾਲੀ *ptr)
            {
                ਸਵੈਚਾਲਿਤ it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ਅਚਲ byte_t *>(ptr)));
                ਜੇ (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ਸੁੱਟੋ std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ਅਚਲ ਵੰਡ &alloc = it->second;
                ਜੇ (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ਸੁੱਟੋ std::runtime_error("invalid virtual pointer");
                }
                ਵਾਪਸ it;
            }
        };

        ਸਾਂਚਾ <ਵਰਗ T, ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2, size_t Dimension>
        ਵਰਗ accessor;
        ਸਾਂਚਾ <ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2, ਵਰਗ T = byte_t>
        ਵਰਗ ਸਮ੍ਰਿਤੀ_traits
        {
        ਜਨਤਕ:
            ਸਥਿਰ constexpr sycl::access::ਨਿਸ਼ਾਨਾ ਨਿਸ਼ਾਨਾ =
                sycl::access::ਨਿਸ਼ਾਨਾ::device;
            ਸਥਿਰ constexpr sycl::access_ਬਹੁਲਕ mode =
                (ਸਮ੍ਰਿਤੀ_2 == constant) ? sycl::access_ਬਹੁਲਕ::ਪਠ
                                     : sycl::access_ਬਹੁਲਕ::read_write;
            ਸਥਿਰ constexpr size_t type_ਮਾਪ = ਆਕਾਰ(T);
            ਵਰਤੋਂ element_t =
                ਕਿਸਮ_ਨਾਂ std::conditional<ਸਮ੍ਰਿਤੀ_2 == constant, ਅਚਲ T, T>::type;
            ਵਰਤੋਂ ਮੁੱਲ_t = ਕਿਸਮ_ਨਾਂ std::remove_cv<T>::type;
            ਸਾਂਚਾ <size_t Dimension = 1>
            ਵਰਤੋਂ accessor_t = ਕਿਸਮ_ਨਾਂ std::conditional<
                ਸਮ੍ਰਿਤੀ_2 == local, sycl::local_accessor<ਮੁੱਲ_t, Dimension>,
                sycl::accessor<T, Dimension, mode, ਨਿਸ਼ਾਨਾ>>::type;
            ਵਰਤੋਂ ਸੂਚਕ_t = T *;
        };

        ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ *dpct_malloc(size_t size, sycl::queue &q)
        {
            ਵਾਪਸ sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            ਵਾਪਸ dpct_malloc(pitch * y * z, q);
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
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਮੁੱਲ_T>
        ਸਥਿਰ ਸਤਰਅੰਦਰ sycl::event dpct_memset(sycl::queue &q, ਖਾਲੀ *dev_ptr,
                                              ਮੁੱਲ_T value, size_t size)
        {
            ਵਾਪਸ q.fill(dev_ptr, value, size);
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
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਮੁੱਲ_T>
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_ਅੰਕੜੇ ਅੰਕੜੇ, ਮੁੱਲ_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t ਟੁਕੜਾ_2 = ਅੰਕੜੇ.get_pitch() * ਅੰਕੜੇ.get_y();
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *ਅੰਕੜੇ_surface = (ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *)ਅੰਕੜੇ.get_ਅੰਕੜੇ_ptr();
            ਲਈ (size_t z = 0; z < size.get(2); ++z)
            {
                ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *ਅੰਕੜੇ_ptr = ਅੰਕੜੇ_surface;
                ਲਈ (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, ਅੰਕੜੇ_ptr, value, size.get(0)));
                    ਅੰਕੜੇ_ptr += ਅੰਕੜੇ.get_pitch();
                }
                ਅੰਕੜੇ_surface += ਟੁਕੜਾ_2;
            }
            ਵਾਪਸ event_list;
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
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ ਮੁੱਲ_T>
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, ਖਾਲੀ *ptr, size_t pitch, ਮੁੱਲ_T val, size_t x,
                    size_t y)
        {
            ਵਾਪਸ dpct_memset(q, pitched_ਅੰਕੜੇ(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ਸਥਿਰ memcpy_direction deduce_memcpy_direction(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr,
                                                        ਅਚਲ ਖਾਲੀ *from_ptr,
                                                        memcpy_direction dir)
        {
            ਚੋਣ (dir)
            {
            ਹਾਲਤ memcpy_direction::host_ਮੰਜ਼ਿਲ_host:
            ਹਾਲਤ memcpy_direction::host_ਮੰਜ਼ਿਲ_device:
            ਹਾਲਤ memcpy_direction::device_ਮੰਜ਼ਿਲ_host:
            ਹਾਲਤ memcpy_direction::device_ਮੰਜ਼ਿਲ_device:
                ਵਾਪਸ dir;
            ਹਾਲਤ memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ਸਥਿਰ ਅਚਲ memcpy_direction
                    direction_ਸਾਰਣੀ[static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ>(ਸੂਚਕ_access_attribute::end)]
                                   [static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ>(ਸੂਚਕ_access_attribute::end)] =
                                       {{memcpy_direction::host_ਮੰਜ਼ਿਲ_host,
                                         memcpy_direction::device_ਮੰਜ਼ਿਲ_host,
                                         memcpy_direction::host_ਮੰਜ਼ਿਲ_host},
                                        {memcpy_direction::host_ਮੰਜ਼ਿਲ_device,
                                         memcpy_direction::device_ਮੰਜ਼ਿਲ_device,
                                         memcpy_direction::device_ਮੰਜ਼ਿਲ_device},
                                        {memcpy_direction::host_ਮੰਜ਼ਿਲ_host,
                                         memcpy_direction::device_ਮੰਜ਼ਿਲ_device,
                                         memcpy_direction::device_ਮੰਜ਼ਿਲ_device}};
                ਵਾਪਸ direction_ਸਾਰਣੀ[static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ>(get_ਸੂਚਕ_attribute(
                    q, ਮੰਜ਼ਿਲ_ptr))][static_cast<ਬਿਨਾਨਿਸ਼ਾਨੀ>(get_ਸੂਚਕ_attribute(q, from_ptr))];
            }
            ਮੂਲ:
                ਸੁੱਟੋ std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ਸਥਿਰ sycl::event
        dpct_memcpy(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr, size_t size,
                    memcpy_direction direction,
                    ਅਚਲ std::vector<sycl::event> &dep_events = {})
        {
            ਜੇ (!size)
                ਵਾਪਸ sycl::event{};
            ਵਾਪਸ q.ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਮੰਜ਼ਿਲ_ptr, from_ptr, size, dep_events);
            DPCT_ਅਣਵਰਤਿਆ(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ਸਥਿਰ ਸਤਰਅੰਦਰ size_t get_copy_range(sycl::range<3> size, size_t ਟੁਕੜਾ_2,
                                            size_t pitch)
        {
            ਵਾਪਸ ਟੁਕੜਾ_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ਸਥਿਰ ਸਤਰਅੰਦਰ size_t get_ਵਿੱਥ(sycl::id<3> id, size_t ਟੁਕੜਾ_2,
                                        size_t pitch)
        {
            ਵਾਪਸ ਟੁਕੜਾ_2 * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr,
                    sycl::range<3> ਮੰਜ਼ਿਲ_range, sycl::range<3> from_range,
                    sycl::id<3> ਮੰਜ਼ਿਲ_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ਅਚਲ std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            ਵਰਗ host_ਬਫਰ
            {
                ਖਾਲੀ *_buf;
                size_t ਮਾਪ_2;
                sycl::queue &_q;
                ਅਚਲ std::vector<sycl::event> &_deps; // free operation depends

            ਜਨਤਕ:
                host_ਬਫਰ(size_t size, sycl::queue &q,
                            ਅਚਲ std::vector<sycl::event> &deps)
                    : _buf(std::ਸਮ੍ਰਿਤੀ_ਦਿਓ(size)), ਮਾਪ_2(size), _q(q), _deps(deps) {}
                ਖਾਲੀ *get_ptr() ਅਚਲ { ਵਾਪਸ _buf; }
                size_t get_ਮਾਪ() ਅਚਲ { ਵਾਪਸ ਮਾਪ_2; }
                ~host_ਬਫਰ()
                {
                    ਜੇ (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::ਮੁਕਤ(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t ਮੰਜ਼ਿਲ_ਟੁਕੜਾ = ਮੰਜ਼ਿਲ_range.get(1) * ਮੰਜ਼ਿਲ_range.get(0),
                   from_ਟੁਕੜਾ = from_range.get(1) * from_range.get(0);
            ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *ਮੰਜ਼ਿਲ_surface =
                (ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *)ਮੰਜ਼ਿਲ_ptr + get_ਵਿੱਥ(ਮੰਜ਼ਿਲ_id, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ, ਮੰਜ਼ਿਲ_range.get(0));
            ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *from_surface =
                (ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *)from_ptr +
                get_ਵਿੱਥ(from_id, from_ਟੁਕੜਾ, from_range.get(0));

            ਜੇ (ਮੰਜ਼ਿਲ_ਟੁਕੜਾ == from_ਟੁਕੜਾ && ਮੰਜ਼ਿਲ_ਟੁਕੜਾ == size.get(1) * size.get(0))
            {
                ਵਾਪਸ {dpct_memcpy(q, ਮੰਜ਼ਿਲ_surface, from_surface, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, direction);
            size_t ਮਾਪ_ਟੁਕੜਾ = size.get(1) * size.get(0);
            ਚੋਣ (direction)
            {
            ਹਾਲਤ host_ਮੰਜ਼ਿਲ_host:
                ਲਈ (size_t z = 0; z < size.get(2); ++z)
                {
                    ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *ਮੰਜ਼ਿਲ_ptr = ਮੰਜ਼ਿਲ_surface;
                    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *from_ptr = from_surface;
                    ਜੇ (ਮੰਜ਼ਿਲ_range.get(0) == from_range.get(0) &&
                        ਮੰਜ਼ਿਲ_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, ਮਾਪ_ਟੁਕੜਾ,
                                                         direction, dep_events));
                    }
                    ਨਹੀਂਤਾਂ
                    {
                        ਲਈ (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            ਮੰਜ਼ਿਲ_ptr += ਮੰਜ਼ਿਲ_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    ਮੰਜ਼ਿਲ_surface += ਮੰਜ਼ਿਲ_ਟੁਕੜਾ;
                    from_surface += from_ਟੁਕੜਾ;
                }
                ਰੋਕੋ;
            ਹਾਲਤ host_ਮੰਜ਼ਿਲ_device:
            {
                host_ਬਫਰ buf(get_copy_range(size, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ, ਮੰਜ਼ਿਲ_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                ਜੇ (ਮੰਜ਼ਿਲ_ਟੁਕੜਾ == ਮਾਪ_ਟੁਕੜਾ)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, ਮੰਜ਼ਿਲ_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_ਮੰਜ਼ਿਲ_host, dep_events);
                }
                ਨਹੀਂਤਾਂ
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, ਮੰਜ਼ਿਲ_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_ਮੰਜ਼ਿਲ_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), ਮੰਜ਼ਿਲ_surface, buf.get_ਮਾਪ(),
                                        device_ਮੰਜ਼ਿਲ_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, ਮੰਜ਼ਿਲ_surface, buf.get_ptr(),
                                                 buf.get_ਮਾਪ(), host_ਮੰਜ਼ਿਲ_device,
                                                 host_events));
                ਰੋਕੋ;
            }
            ਹਾਲਤ device_ਮੰਜ਼ਿਲ_host:
            {
                host_ਬਫਰ buf(get_copy_range(size, from_ਟੁਕੜਾ, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, ਮੰਜ਼ਿਲ_surface, buf.get_ptr(), ਮੰਜ਼ਿਲ_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_ਮੰਜ਼ਿਲ_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_ਮਾਪ(),
                                                         device_ਮੰਜ਼ਿਲ_host, dep_events)});
                ਰੋਕੋ;
            }
            ਹਾਲਤ device_ਮੰਜ਼ਿਲ_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<ਵਰਗ dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        ਮੰਜ਼ਿਲ_surface[get_ਵਿੱਥ(id, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ, ਮੰਜ਼ਿਲ_range.get(0))] =
                            from_surface[get_ਵਿੱਥ(id, from_ਟੁਕੜਾ, from_range.get(0))];
                    }); }));
                ਰੋਕੋ;
            ਮੂਲ:
                ਸੁੱਟੋ std::runtime_error("dpct_memcpy: invalid direction value");
            }
            ਵਾਪਸ event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_ਅੰਕੜੇ ਮੰਜ਼ਿਲ, sycl::id<3> ਮੰਜ਼ਿਲ_id,
                    pitched_ਅੰਕੜੇ from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            ਵਾਪਸ dpct_memcpy(q, ਮੰਜ਼ਿਲ.get_ਅੰਕੜੇ_ptr(), from.get_ਅੰਕੜੇ_ptr(),
                               sycl::range<3>(ਮੰਜ਼ਿਲ.get_pitch(), ਮੰਜ਼ਿਲ.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), ਮੰਜ਼ਿਲ_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr,
                    size_t ਮੰਜ਼ਿਲ_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            ਵਾਪਸ dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, sycl::range<3>(ਮੰਜ਼ਿਲ_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        ਨਾਮਾਕਾਸ਼ deprecated
        {

            ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, sycl::usm::alloc Alloc_ਕਿਸਮ>
            ਵਰਗ usm_allocator
            {
            ਨਿਜੀ:
                ਵਰਤੋਂ Alloc = sycl::usm_allocator<T, Alloc_ਕਿਸਮ>;
                Alloc _impl;

            ਜਨਤਕ:
                ਵਰਤੋਂ value_type = ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::value_type;
                ਵਰਤੋਂ pointer = ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::pointer;
                ਵਰਤੋਂ const_pointer = ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::const_pointer;
                ਵਰਤੋਂ void_pointer = ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::void_pointer;
                ਵਰਤੋਂ const_void_pointer =
                    ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::const_void_pointer;
                ਵਰਤੋਂ reference = ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::value_type &;
                ਵਰਤੋਂ const_reference =
                    ਅਚਲ ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::value_type &;
                ਵਰਤੋਂ difference_type =
                    ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::difference_type;
                ਵਰਤੋਂ size_type = ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::size_type;
                ਵਰਤੋਂ propagate_on_container_copy_assignment = ਕਿਸਮ_ਨਾਂ std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                ਵਰਤੋਂ propagate_on_container_move_assignment = ਕਿਸਮ_ਨਾਂ std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                ਵਰਤੋਂ propagate_on_container_swap =
                    ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::propagate_on_container_swap;
                ਵਰਤੋਂ is_always_equal =
                    ਕਿਸਮ_ਨਾਂ std::allocator_traits<Alloc>::is_always_equal;

                ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ U>
                ਬਣਤਰ rebind
                {
                    ਕਿਸਮਪਰਿਭਾਸ਼ਾ usm_allocator<U, Alloc_ਕਿਸਮ> other;
                };

                usm_allocator() : _impl(dpct::get_default_ਕਤਾਰ()) {}
                ~usm_allocator() {}
                usm_allocator(ਅਚਲ usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { ਵਾਪਸ &r; }
                const_pointer address(const_reference r) { ਵਾਪਸ &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = ਖਾਲੀ_ਸੂਚਕ)
                {
                    ਵਾਪਸ std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                ਖਾਲੀ deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ਅਚਲ
                {
                    ਵਾਪਸ std::allocator_traits<Alloc>::max_size(_impl);
                }
                ਤਾਰਕਿਕ ਸੰਚਾਲਕ==(ਅਚਲ usm_allocator &other) ਅਚਲ { ਵਾਪਸ _impl == other._impl; }
                ਤਾਰਕਿਕ ਸੰਚਾਲਕ!=(ਅਚਲ usm_allocator &other) ਅਚਲ { ਵਾਪਸ _impl != other._impl; }
            };

        } // namespace deprecated

        ਸਤਰਅੰਦਰ ਖਾਲੀ dpct_free(ਖਾਲੀ *ptr,
                              ਅਚਲ sycl::queue &q)
        {
            ਜੇ (ptr)
            {
                sycl::ਮੁਕਤ(ptr, q.get_context());
            }
        }

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
        ਸਤਰਅੰਦਰ ਸਵੈਚਾਲਿਤ get_ਸਮ੍ਰਿਤੀ(ਅਚਲ ਖਾਲੀ *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<ਖਾਲੀ *>(x));
            ਵਾਪਸ new_x;
        }

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
        ਸਤਰਅੰਦਰ ਕਿਸਮ_ਨਾਂ ਅੰਕੜੇ_Type_2<T>::T2 get_ਮੁੱਲ(ਅਚਲ T *s, sycl::queue &q)
        {
            ਵਰਤੋਂ Ty = ਕਿਸਮ_ਨਾਂ ਅੰਕੜੇ_Type_2<T>::T2;
            Ty s_h;
            ਜੇ (get_ਸੂਚਕ_attribute(q, s) == ਸੂਚਕ_access_attribute::device_only)
                detail::dpct_memcpy(q, (ਖਾਲੀ *)&s_h, (ਅਚਲ ਖਾਲੀ *)s, ਆਕਾਰ(T), device_ਮੰਜ਼ਿਲ_host)
                    .wait();
            ਨਹੀਂਤਾਂ
                s_h = *reinterpret_cast<ਅਚਲ Ty *>(s);
            ਵਾਪਸ s_h;
        }

    } // namespace detail

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਸਤਰਅੰਦਰ ਸਵੈਚਾਲਿਤ get_ਮੁੱਲ(ਅਚਲ T *s, sycl::queue &q)
    {
        ਵਾਪਸ detail::get_ਮੁੱਲ(s, q);
    }

    ਨਾਮਾਕਾਸ਼ detail
    {
    ਸਾਂਚਾ <ਵਰਗ Ta, ਵਰਗ Tb, ਵਰਗ Tc, ਵਰਗ Ts>
    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ਪੂਰਨ m,
                          ਪੂਰਨ n, ਪੂਰਨ k, ਅਚਲ ਖਾਲੀ * alpha, ਅਚਲ ਖਾਲੀ * a, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * b, ਪੂਰਨ ldb,
                          ਅਚਲ ਖਾਲੀ * beta, ਖਾਲੀ * c, ਪੂਰਨ ldc) {
        Ts   alpha_ਮੁੱਲ_2 = dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ Ts *>(alpha), q);
        Ts   beta_ਮੁੱਲ  = dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ Ts *>(beta), q);
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_a      = get_ਸਮ੍ਰਿਤੀ<ਅਚਲ Ta>(a);
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_b      = get_ਸਮ੍ਰਿਤੀ<ਅਚਲ Tb>(b);
        ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_c      = get_ਸਮ੍ਰਿਤੀ<Tc>(c);
        oneapi::mkl::blas::ਕਾਲਮ_major::gemm(q, a_trans, b_trans, m, n, k, alpha_ਮੁੱਲ_2, ਅੰਕੜੇ_a,
                                               lda, ਅੰਕੜੇ_b, ldb, beta_ਮੁੱਲ, ਅੰਕੜੇ_c, ldc);
    }

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ VecT, ਵਰਗ ਦੁਆਧਾਰੀ_Operation, ਵਰਗ = ਖਾਲੀ>
        ਵਰਗ vectorized_ਦੁਆਧਾਰੀ
        {
        ਜਨਤਕ:
            ਸਤਰਅੰਦਰ VecT ਸੰਚਾਲਕ()(VecT a, VecT b, ਅਚਲ ਦੁਆਧਾਰੀ_Operation ਦੁਆਧਾਰੀ_op)
            {
                VecT v4;
                ਲਈ (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ਦੁਆਧਾਰੀ_op(a[i], b[i]);
                }
                ਵਾਪਸ v4;
            }
        };

        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ VecT, ਵਰਗ ਦੁਆਧਾਰੀ_Operation>
        ਵਰਗ vectorized_ਦੁਆਧਾਰੀ<
            VecT, ਦੁਆਧਾਰੀ_Operation,
            std::void_t<std::invoke_result_t<ਦੁਆਧਾਰੀ_Operation, VecT, VecT>>>
        {
        ਜਨਤਕ:
            ਸਤਰਅੰਦਰ VecT ਸੰਚਾਲਕ()(VecT a, VecT b, ਅਚਲ ਦੁਆਧਾਰੀ_Operation ਦੁਆਧਾਰੀ_op)
            {
                ਵਾਪਸ ਦੁਆਧਾਰੀ_op(a, b).ਸਾਂਚਾ as<VecT>();
            }
        };

        ਸਾਂਚਾ <ਵਰਗ Ta, ਵਰਗ Tb, ਵਰਗ Tc, ਵਰਗ Ts>
        ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ਪੂਰਨ m, ਪੂਰਨ n, ਪੂਰਨ k, ਅਚਲ ਖਾਲੀ * alpha, ਅਚਲ ਖਾਲੀ ** a, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ ** b,
                                    ਪੂਰਨ ldb, ਅਚਲ ਖਾਲੀ * beta, ਖਾਲੀ ** c, ਪੂਰਨ ldc, ਪੂਰਨ batch_ਮਾਪ,
                                    ਮੈਟ੍ਰਿਕਸ_info_t<ਦਸ਼ਮਲਵ> * ਮੈਟ੍ਰਿਕਸ_info) {
            Ts alpha_ਮੁੱਲ_2 = dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ Ts *>(alpha), q);
            Ts beta_ਮੁੱਲ = dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ Ts *>(beta), q);

            ਮੈਟ੍ਰਿਕਸ_info->transpose_info[0] = a_trans;
            ਮੈਟ੍ਰਿਕਸ_info->transpose_info[1] = b_trans;
            ਮੈਟ੍ਰਿਕਸ_info->ਮੁੱਲ_info[0] = alpha_ਮੁੱਲ_2;
            ਮੈਟ੍ਰਿਕਸ_info->ਮੁੱਲ_info[1] = beta_ਮੁੱਲ;
            ਮੈਟ੍ਰਿਕਸ_info->ਮਾਪ_info[0] = m;
            ਮੈਟ੍ਰਿਕਸ_info->ਮਾਪ_info[1] = n;
            ਮੈਟ੍ਰਿਕਸ_info->ਮਾਪ_info[2] = k;
            ਮੈਟ੍ਰਿਕਸ_info->ld_info[0] = lda;
            ਮੈਟ੍ਰਿਕਸ_info->ld_info[1] = ldb;
            ਮੈਟ੍ਰਿਕਸ_info->ld_info[2] = ldc;
            ਮੈਟ੍ਰਿਕਸ_info->groupsize_info = batch_ਮਾਪ;

            sycl::event e = oneapi::mkl::blas::ਕਾਲਮ_major::gemm_batch(
                q, ਮੈਟ੍ਰਿਕਸ_info->transpose_info, ਮੈਟ੍ਰਿਕਸ_info->transpose_info + 1,
                ਮੈਟ੍ਰਿਕਸ_info->ਮਾਪ_info, ਮੈਟ੍ਰਿਕਸ_info->ਮਾਪ_info + 1, ਮੈਟ੍ਰਿਕਸ_info->ਮਾਪ_info + 2,
                reinterpret_cast<Ts *>(ਮੈਟ੍ਰਿਕਸ_info->ਮੁੱਲ_info), reinterpret_cast<ਅਚਲ Ta **>(a), ਮੈਟ੍ਰਿਕਸ_info->ld_info,
                reinterpret_cast<ਅਚਲ Tb **>(b), ਮੈਟ੍ਰਿਕਸ_info->ld_info + 1,
                reinterpret_cast<Ts *>(ਮੈਟ੍ਰਿਕਸ_info->ਮੁੱਲ_info + 1), reinterpret_cast<Tc **>(c),
                ਮੈਟ੍ਰਿਕਸ_info->ld_info + 2, 1, &(ਮੈਟ੍ਰਿਕਸ_info->groupsize_info));
        }

        ਸਾਂਚਾ <ਵਰਗ Ta, ਵਰਗ Tb, ਵਰਗ Tc, ਵਰਗ Ts>
        ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ਪੂਰਨ m, ਪੂਰਨ n, ਪੂਰਨ k, ਅਚਲ ਖਾਲੀ * alpha, ਅਚਲ ਖਾਲੀ * a, ਪੂਰਨ lda,
                                    ਲੰਮਾ ਲੰਮਾ ਪੂਰਨ stride_a, ਅਚਲ ਖਾਲੀ * b, ਪੂਰਨ ldb, ਲੰਮਾ ਲੰਮਾ ਪੂਰਨ stride_b,
                                    ਅਚਲ ਖਾਲੀ * beta, ਖਾਲੀ * c, ਪੂਰਨ ldc, ਲੰਮਾ ਲੰਮਾ ਪੂਰਨ stride_c, ਪੂਰਨ batch_ਮਾਪ) {
            Ts alpha_ਮੁੱਲ_2 = dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ Ts *>(alpha), q);
            Ts beta_ਮੁੱਲ = dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ Ts *>(beta), q);
            ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_a = get_ਸਮ੍ਰਿਤੀ<ਅਚਲ Ta>(a);
            ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_b = get_ਸਮ੍ਰਿਤੀ<ਅਚਲ Tb>(b);
            ਸਵੈਚਾਲਿਤ ਅੰਕੜੇ_c = get_ਸਮ੍ਰਿਤੀ<Tc>(c);
            oneapi::mkl::blas::ਕਾਲਮ_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_ਮੁੱਲ_2,
                                                         ਅੰਕੜੇ_a, lda, stride_a, ਅੰਕੜੇ_b, ldb, stride_b, beta_ਮੁੱਲ,
                                                         ਅੰਕੜੇ_c, ldc, stride_c, batch_ਮਾਪ);
        }

    } // namespace detail

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ VecT, ਵਰਗ ਦੁਆਧਾਰੀ_Operation>
    ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ vectorized_ਦੁਆਧਾਰੀ(ਬਿਨਾਨਿਸ਼ਾਨੀ a, ਬਿਨਾਨਿਸ਼ਾਨੀ b,
                                      ਅਚਲ ਦੁਆਧਾਰੀ_Operation ਦੁਆਧਾਰੀ_op)
    {
        sycl::vec<ਬਿਨਾਨਿਸ਼ਾਨੀ, 1> v0{a}, v1{b};
        ਸਵੈਚਾਲਿਤ v2 = v0.as<VecT>();
        ਸਵੈਚਾਲਿਤ v3 = v1.as<VecT>();
        ਸਵੈਚਾਲਿਤ v4 =
            detail::vectorized_ਦੁਆਧਾਰੀ<VecT, ਦੁਆਧਾਰੀ_Operation>()(v2, v3, ਦੁਆਧਾਰੀ_op);
        v0 = v4.ਸਾਂਚਾ as<sycl::vec<ਬਿਨਾਨਿਸ਼ਾਨੀ, 1>>();
        ਵਾਪਸ v0;
    }

    ਸਥਿਰ ਖਾਲੀ async_dpct_memcpy(ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_ਕਤਾਰ())
    {
        detail::dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, size, direction);
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ select_device(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id)
    {
        dev_mgr::instance().select_device(id);
        ਵਾਪਸ id;
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    T ਕ੍ਰਮ_ਬਦਲੋ_sub_ਸਮੂਹ_by_xor(sycl::sub_ਸਮੂਹ g, T x, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਨਕਾਬ,
                               ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ logical_sub_ਸਮੂਹ_ਮਾਪ = 32)
    {
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id = g.get_local_ਰੇਖਿਕ_id();
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ =
            id / logical_sub_ਸਮੂਹ_ਮਾਪ * logical_sub_ਸਮੂਹ_ਮਾਪ;
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਨਿਸ਼ਾਨਾ_ਵਿੱਥ = (id % logical_sub_ਸਮੂਹ_ਮਾਪ) ^ ਨਕਾਬ;
        ਵਾਪਸ sycl::select_from_ਸਮੂਹ(g, x,
                                       ਨਿਸ਼ਾਨਾ_ਵਿੱਥ < logical_sub_ਸਮੂਹ_ਮਾਪ
                                           ? ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ + ਨਿਸ਼ਾਨਾ_ਵਿੱਥ
                                           : id);
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T1, ਕਿਸਮ_ਨਾਂ T2>
    ਵਰਤੋਂ dot_ਗੁਣਨਫਲ_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    sycl::vec<T, 4> ਕੱਢੋ_and_sign_or_zero_extend4(T val) {
      ਵਾਪਸ sycl::vec<T, 1>(val)
          .ਸਾਂਚਾ as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .ਸਾਂਚਾ convert<T>();
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T1, ਕਿਸਮ_ਨਾਂ T2, ਕਿਸਮ_ਨਾਂ T3>
    ਸਤਰਅੰਦਰ ਸਵੈਚਾਲਿਤ dp4a(T1 a, T2 b, T3 c) {
      dot_ਗੁਣਨਫਲ_acc_t<T1, T2> res = c;
      ਸਵੈਚਾਲਿਤ va = ਕੱਢੋ_and_sign_or_zero_extend4(a);
      ਸਵੈਚਾਲਿਤ vb = ਕੱਢੋ_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      ਵਾਪਸ res;
    }

    ਬਣਤਰ sub_sat
    {
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
        ਸਵੈਚਾਲਿਤ ਸੰਚਾਲਕ()(ਅਚਲ T x, ਅਚਲ T y) ਅਚਲ
        {
            ਵਾਪਸ sycl::sub_sat(x, y);
        }
    };

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ S, ਕਿਸਮ_ਨਾਂ T>
    ਸਤਰਅੰਦਰ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        ਸਵੈਚਾਲਿਤ v2 = v0.ਸਾਂਚਾ as<S>();
        ਸਵੈਚਾਲਿਤ v3 = v1.ਸਾਂਚਾ as<S>();
        ਸਵੈਚਾਲਿਤ v4 = sycl::min(v2, v3);
        v0 = v4.ਸਾਂਚਾ as<sycl::vec<T, 1>>();
        ਵਾਪਸ v0;
    }

    ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ਘਾਤ(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਪੂਰਨ b) { ਵਾਪਸ sycl::pown(a, b); }
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ ਘਾਤ(ਅਚਲ ਦੁੱਗਣਾ a, ਅਚਲ ਪੂਰਨ b) { ਵਾਪਸ sycl::pown(a, b); }
    ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ਘਾਤ(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) { ਵਾਪਸ sycl::ਘਾਤ(a, b); }
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ ਘਾਤ(ਅਚਲ ਦੁੱਗਣਾ a, ਅਚਲ ਦੁੱਗਣਾ b) { ਵਾਪਸ sycl::ਘਾਤ(a, b); }
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ U>
    ਸਤਰਅੰਦਰ ਕਿਸਮ_ਨਾਂ std::enable_if_t<std::is_floating_point_v<T>, T>
    ਘਾਤ(ਅਚਲ T a, ਅਚਲ U b)
    {
        ਵਾਪਸ sycl::ਘਾਤ(a, static_cast<T>(b));
    }
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਕਿਸਮ_ਨਾਂ U>
    ਸਤਰਅੰਦਰ ਕਿਸਮ_ਨਾਂ std::enable_if_t<!std::is_floating_point_v<T>, ਦੁੱਗਣਾ>
    ਘਾਤ(ਅਚਲ T a, ਅਚਲ U b)
    {
        ਵਾਪਸ sycl::ਘਾਤ(static_cast<ਦੁੱਗਣਾ>(a), static_cast<ਦੁੱਗਣਾ>(b));
    }

    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ min(ਅਚਲ ਦੁੱਗਣਾ a, ਅਚਲ ਦਸ਼ਮਲਵ b)
    {
        ਵਾਪਸ sycl::fmin(a, static_cast<ਦੁੱਗਣਾ>(b));
    }
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ min(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦੁੱਗਣਾ b)
    {
        ਵਾਪਸ sycl::fmin(static_cast<ਦੁੱਗਣਾ>(a), b);
    }
    ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ min(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) { ਵਾਪਸ sycl::fmin(a, b); }
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ min(ਅਚਲ ਦੁੱਗਣਾ a, ਅਚਲ ਦੁੱਗਣਾ b) { ਵਾਪਸ sycl::fmin(a, b); }
    ਸਤਰਅੰਦਰ std::uint32_t min(ਅਚਲ std::uint32_t a, ਅਚਲ std::int32_t b)
    {
        ਵਾਪਸ sycl::min(a, static_cast<std::uint32_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint32_t min(ਅਚਲ std::int32_t a, ਅਚਲ std::uint32_t b)
    {
        ਵਾਪਸ sycl::min(static_cast<std::uint32_t>(a), b);
    }
    ਸਤਰਅੰਦਰ std::int32_t min(ਅਚਲ std::int32_t a, ਅਚਲ std::int32_t b)
    {
        ਵਾਪਸ sycl::min(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint32_t min(ਅਚਲ std::uint32_t a, ਅਚਲ std::uint32_t b)
    {
        ਵਾਪਸ sycl::min(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::uint64_t a, ਅਚਲ std::int64_t b)
    {
        ਵਾਪਸ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::int64_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ਸਤਰਅੰਦਰ std::int64_t min(ਅਚਲ std::int64_t a, ਅਚਲ std::int64_t b)
    {
        ਵਾਪਸ sycl::min(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::uint64_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::min(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::uint64_t a, ਅਚਲ std::int32_t b)
    {
        ਵਾਪਸ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::int32_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::uint64_t a, ਅਚਲ std::uint32_t b)
    {
        ਵਾਪਸ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint64_t min(ਅਚਲ std::uint32_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ max(ਅਚਲ ਦੁੱਗਣਾ a, ਅਚਲ ਦਸ਼ਮਲਵ b)
    {
        ਵਾਪਸ sycl::fmax(a, static_cast<ਦੁੱਗਣਾ>(b));
    }
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ max(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦੁੱਗਣਾ b)
    {
        ਵਾਪਸ sycl::fmax(static_cast<ਦੁੱਗਣਾ>(a), b);
    }
    ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ max(ਅਚਲ ਦਸ਼ਮਲਵ a, ਅਚਲ ਦਸ਼ਮਲਵ b) { ਵਾਪਸ sycl::fmax(a, b); }
    ਸਤਰਅੰਦਰ ਦੁੱਗਣਾ max(ਅਚਲ ਦੁੱਗਣਾ a, ਅਚਲ ਦੁੱਗਣਾ b) { ਵਾਪਸ sycl::fmax(a, b); }
    ਸਤਰਅੰਦਰ std::uint32_t max(ਅਚਲ std::uint32_t a, ਅਚਲ std::int32_t b)
    {
        ਵਾਪਸ sycl::max(a, static_cast<std::uint32_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint32_t max(ਅਚਲ std::int32_t a, ਅਚਲ std::uint32_t b)
    {
        ਵਾਪਸ sycl::max(static_cast<std::uint32_t>(a), b);
    }
    ਸਤਰਅੰਦਰ std::int32_t max(ਅਚਲ std::int32_t a, ਅਚਲ std::int32_t b)
    {
        ਵਾਪਸ sycl::max(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint32_t max(ਅਚਲ std::uint32_t a, ਅਚਲ std::uint32_t b)
    {
        ਵਾਪਸ sycl::max(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::uint64_t a, ਅਚਲ std::int64_t b)
    {
        ਵਾਪਸ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::int64_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ਸਤਰਅੰਦਰ std::int64_t max(ਅਚਲ std::int64_t a, ਅਚਲ std::int64_t b)
    {
        ਵਾਪਸ sycl::max(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::uint64_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::max(a, b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::uint64_t a, ਅਚਲ std::int32_t b)
    {
        ਵਾਪਸ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::int32_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::uint64_t a, ਅਚਲ std::uint32_t b)
    {
        ਵਾਪਸ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ਸਤਰਅੰਦਰ std::uint64_t max(ਅਚਲ std::uint32_t a, ਅਚਲ std::uint64_t b)
    {
        ਵਾਪਸ sycl::max(static_cast<std::uint64_t>(a), b);
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ
    has_capability_or_fail(ਅਚਲ sycl::device &dev,
                           ਅਚਲ std::initializer_list<sycl::aspect> &props)
    {
        ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ &it : props)
        {
            ਜੇ (dev.has(it))
                ਜਾਰੀ;
            ਚੋਣ (it)
            {
            ਹਾਲਤ sycl::aspect::fp64:
                ਸੁੱਟੋ std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ਰੋਕੋ;
            ਹਾਲਤ sycl::aspect::fp16:
                ਸੁੱਟੋ std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ਰੋਕੋ;
            ਮੂਲ:
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __SYCL_ASPECT(ASPECT, ID) \
    ਹਾਲਤ sycl::aspect::ASPECT:    \
        ਵਾਪਸ #ASPECT;
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __SYCL_ASPECT_DEPRECATED(ASPECT, ID, ਸੁਨੇਹਾ_3) __SYCL_ASPECT(ASPECT, ID)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, ਸੁਨੇਹਾ_3)
                ਸਵੈਚਾਲਿਤ getAspectNameStr = [](sycl::aspect Aspect_ਅੰਕ) -> std::string
                {
                    ਚੋਣ (Aspect_ਅੰਕ)
                    {
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/info/aspects.def>
#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/info/aspects_deprecated.def>
                    ਮੂਲ:
                        ਵਾਪਸ "unknown aspect";
                    }
                };
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ __SYCL_ASPECT_DEPRECATED_ALIAS
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ __SYCL_ASPECT_DEPRECATED
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ __SYCL_ASPECT
                ਸੁੱਟੋ std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            ਰੋਕੋ;
        }
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ get_current_device_id()
    {
        ਵਾਪਸ dev_mgr::instance().current_device_id();
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ device_ext &get_current_device()
    {
        ਵਾਪਸ dev_mgr::instance().current_device();
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ device_ext &get_device(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ id)
    {
        ਵਾਪਸ dev_mgr::instance().get_device(id);
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ sycl::queue &get_ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ()
    {
        ਵਾਪਸ dev_mgr::instance().current_device().ਵਿੱਚ_ਕ੍ਰਮ_ਕਤਾਰ();
    }

    ਸਥਿਰ sycl::event
    dpct_memcpy(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr, size_t size,
                memcpy_direction direction,
                ਅਚਲ std::vector<sycl::event> &dep_events = {})
    {
        ਜੇ (!size)
            ਵਾਪਸ sycl::event{};
        ਵਾਪਸ q.ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਮੰਜ਼ਿਲ_ptr, from_ptr, size, dep_events);
        DPCT_ਅਣਵਰਤਿਆ(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ਸਥਿਰ ਸਤਰਅੰਦਰ size_t get_copy_range(sycl::range<3> size, size_t ਟੁਕੜਾ_2,
                                        size_t pitch)
    {
        ਵਾਪਸ ਟੁਕੜਾ_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ size_t get_ਵਿੱਥ(sycl::id<3> id, size_t ਟੁਕੜਾ_2,
                                    size_t pitch)
    {
        ਵਾਪਸ ਟੁਕੜਾ_2 * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr,
                sycl::range<3> ਮੰਜ਼ਿਲ_range, sycl::range<3> from_range,
                sycl::id<3> ਮੰਜ਼ਿਲ_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ਅਚਲ std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        ਵਰਗ host_ਬਫਰ
        {
            ਖਾਲੀ *_buf;
            size_t ਮਾਪ_2;
            sycl::queue &_q;
            ਅਚਲ std::vector<sycl::event> &_deps; // free operation depends

        ਜਨਤਕ:
            host_ਬਫਰ(size_t size, sycl::queue &q,
                        ਅਚਲ std::vector<sycl::event> &deps)
                : _buf(std::ਸਮ੍ਰਿਤੀ_ਦਿਓ(size)), ਮਾਪ_2(size), _q(q), _deps(deps) {}
            ਖਾਲੀ *get_ptr() ਅਚਲ { ਵਾਪਸ _buf; }
            size_t get_ਮਾਪ() ਅਚਲ { ਵਾਪਸ ਮਾਪ_2; }
            ~host_ਬਫਰ()
            {
                ਜੇ (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::ਮੁਕਤ(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t ਮੰਜ਼ਿਲ_ਟੁਕੜਾ = ਮੰਜ਼ਿਲ_range.get(1) * ਮੰਜ਼ਿਲ_range.get(0),
               from_ਟੁਕੜਾ = from_range.get(1) * from_range.get(0);
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *ਮੰਜ਼ਿਲ_surface =
            (ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *)ਮੰਜ਼ਿਲ_ptr + get_ਵਿੱਥ(ਮੰਜ਼ਿਲ_id, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ, ਮੰਜ਼ਿਲ_range.get(0));
        ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *from_surface =
            (ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *)from_ptr +
            get_ਵਿੱਥ(from_id, from_ਟੁਕੜਾ, from_range.get(0));

        ਜੇ (ਮੰਜ਼ਿਲ_ਟੁਕੜਾ == from_ਟੁਕੜਾ && ਮੰਜ਼ਿਲ_ਟੁਕੜਾ == size.get(1) * size.get(0))
        {
            ਵਾਪਸ {dpct_memcpy(q, ਮੰਜ਼ਿਲ_surface, from_surface, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, direction);
        size_t ਮਾਪ_ਟੁਕੜਾ = size.get(1) * size.get(0);
        ਚੋਣ (direction)
        {
        ਹਾਲਤ host_ਮੰਜ਼ਿਲ_host:
            ਲਈ (size_t z = 0; z < size.get(2); ++z)
            {
                ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *ਮੰਜ਼ਿਲ_ptr = ਮੰਜ਼ਿਲ_surface;
                ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਅੱਖਰ *from_ptr = from_surface;
                ਜੇ (ਮੰਜ਼ਿਲ_range.get(0) == from_range.get(0) &&
                    ਮੰਜ਼ਿਲ_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, ਮਾਪ_ਟੁਕੜਾ,
                                                     direction, dep_events));
                }
                ਨਹੀਂਤਾਂ
                {
                    ਲਈ (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        ਮੰਜ਼ਿਲ_ptr += ਮੰਜ਼ਿਲ_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                ਮੰਜ਼ਿਲ_surface += ਮੰਜ਼ਿਲ_ਟੁਕੜਾ;
                from_surface += from_ਟੁਕੜਾ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ host_ਮੰਜ਼ਿਲ_device:
        {
            host_ਬਫਰ buf(get_copy_range(size, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ, ਮੰਜ਼ਿਲ_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            ਜੇ (ਮੰਜ਼ਿਲ_ਟੁਕੜਾ == ਮਾਪ_ਟੁਕੜਾ)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, ਮੰਜ਼ਿਲ_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_ਮੰਜ਼ਿਲ_host, dep_events);
            }
            ਨਹੀਂਤਾਂ
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, ਮੰਜ਼ਿਲ_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_ਮੰਜ਼ਿਲ_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), ਮੰਜ਼ਿਲ_surface, buf.get_ਮਾਪ(),
                                    device_ਮੰਜ਼ਿਲ_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, ਮੰਜ਼ਿਲ_surface, buf.get_ptr(),
                                             buf.get_ਮਾਪ(), host_ਮੰਜ਼ਿਲ_device,
                                             host_events));
            ਰੋਕੋ;
        }
        ਹਾਲਤ device_ਮੰਜ਼ਿਲ_host:
        {
            host_ਬਫਰ buf(get_copy_range(size, from_ਟੁਕੜਾ, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, ਮੰਜ਼ਿਲ_surface, buf.get_ptr(), ਮੰਜ਼ਿਲ_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_ਮੰਜ਼ਿਲ_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_ਮਾਪ(),
                                                     device_ਮੰਜ਼ਿਲ_host, dep_events)});
            ਰੋਕੋ;
        }
        ਹਾਲਤ device_ਮੰਜ਼ਿਲ_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<ਵਰਗ dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                ਮੰਜ਼ਿਲ_surface[get_ਵਿੱਥ(id, ਮੰਜ਼ਿਲ_ਟੁਕੜਾ, ਮੰਜ਼ਿਲ_range.get(0))] =
                    from_surface[get_ਵਿੱਥ(id, from_ਟੁਕੜਾ, from_range.get(0))];
            }); }));
        ਰੋਕੋ;
        ਮੂਲ:
            ਸੁੱਟੋ std::runtime_error("dpct_memcpy: invalid direction value");
        }
        ਵਾਪਸ event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_ਅੰਕੜੇ ਮੰਜ਼ਿਲ, sycl::id<3> ਮੰਜ਼ਿਲ_id,
                pitched_ਅੰਕੜੇ from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        ਵਾਪਸ dpct_memcpy(q, ਮੰਜ਼ਿਲ.get_ਅੰਕੜੇ_ptr(), from.get_ਅੰਕੜੇ_ptr(),
                           sycl::range<3>(ਮੰਜ਼ਿਲ.get_pitch(), ਮੰਜ਼ਿਲ.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), ਮੰਜ਼ਿਲ_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ਸਥਿਰ ਸਤਰਅੰਦਰ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, ਅਚਲ ਖਾਲੀ *from_ptr,
                size_t ਮੰਜ਼ਿਲ_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        ਵਾਪਸ dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, sycl::range<3>(ਮੰਜ਼ਿਲ_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ਪੂਰਨ m, ਪੂਰਨ n,
                     ਪੂਰਨ k, ਅਚਲ ਖਾਲੀ * alpha, ਅਚਲ ਖਾਲੀ * a, library_ਅੰਕੜੇ_t a_type, ਪੂਰਨ lda, ਅਚਲ ਖਾਲੀ * b,
                     library_ਅੰਕੜੇ_t b_type, ਪੂਰਨ ldb, ਅਚਲ ਖਾਲੀ * beta, ਖਾਲੀ * c, library_ਅੰਕੜੇ_t c_type, ਪੂਰਨ ldc,
                     library_ਅੰਕੜੇ_t scaling_type) {
        ਜੇ (scaling_type == library_ਅੰਕੜੇ_t::real_float &&
            c_type == library_ਅੰਕੜੇ_t::complex_float)
        {
            scaling_type = library_ਅੰਕੜੇ_t::complex_float;
        }
        ਨਹੀਂਤਾਂ ਜੇ (scaling_type == library_ਅੰਕੜੇ_t::real_double &&
                 c_type == library_ਅੰਕੜੇ_t::complex_double)
        {
            scaling_type = library_ਅੰਕੜੇ_t::complex_double;
        }

        std::uint64_t ਕੁੰਜੀ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ਚੋਣ (ਕੁੰਜੀ)
        {
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_impl<ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_double, library_ਅੰਕੜੇ_t::real_double,
            library_ਅੰਕੜੇ_t::real_double, library_ਅੰਕੜੇ_t::real_double):
        {
            detail::gemm_impl<ਦੁੱਗਣਾ, ਦੁੱਗਣਾ, ਦੁੱਗਣਾ, ਦੁੱਗਣਾ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::complex_float, library_ਅੰਕੜੇ_t::complex_float,
            library_ਅੰਕੜੇ_t::complex_float, library_ਅੰਕੜੇ_t::complex_float):
        {
            detail::gemm_impl<std::complex<ਦਸ਼ਮਲਵ>, std::complex<ਦਸ਼ਮਲਵ>,
                              std::complex<ਦਸ਼ਮਲਵ>, std::complex<ਦਸ਼ਮਲਵ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::complex_double, library_ਅੰਕੜੇ_t::complex_double,
            library_ਅੰਕੜੇ_t::complex_double, library_ਅੰਕੜੇ_t::complex_double):
        {
            detail::gemm_impl<std::complex<ਦੁੱਗਣਾ>, std::complex<ਦੁੱਗਣਾ>,
                              std::complex<ਦੁੱਗਣਾ>, std::complex<ਦੁੱਗਣਾ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __INTEL_MKL__
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_bfloat16,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_float):
        {
            ਦਸ਼ਮਲਵ alpha_ਮੁੱਲ_2 =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(alpha), q);
            ਦਸ਼ਮਲਵ beta_ਮੁੱਲ =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(beta), q);
            sycl::half alpha_half(alpha_ਮੁੱਲ_2);
            sycl::half beta_half(beta_ਮੁੱਲ);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_int8, library_ਅੰਕੜੇ_t::real_int8,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_bfloat16,
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_int8, library_ਅੰਕੜੇ_t::real_int8,
            library_ਅੰਕੜੇ_t::real_int32, library_ਅੰਕੜੇ_t::real_int32):
        {
            ਦਸ਼ਮਲਵ alpha_float =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ std::int32_t *>(alpha), q);
            ਦਸ਼ਮਲਵ beta_float =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            ਰੋਕੋ;
        }
#ਜੇ_ਅੰਤ // __INTEL_MKL__
        ਮੂਲ:
            ਸੁੱਟੋ std::runtime_error("the combination of data type is unsupported");
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
    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ਪੂਰਨ m,
                           ਪੂਰਨ n, ਪੂਰਨ k, ਅਚਲ ਖਾਲੀ * alpha, ਅਚਲ ਖਾਲੀ * a[], library_ਅੰਕੜੇ_t a_type, ਪੂਰਨ lda,
                           ਅਚਲ ਖਾਲੀ * b[], library_ਅੰਕੜੇ_t b_type, ਪੂਰਨ ldb, ਅਚਲ ਖਾਲੀ * beta, ਖਾਲੀ * c[],
                           library_ਅੰਕੜੇ_t c_type, ਪੂਰਨ ldc, ਪੂਰਨ batch_ਮਾਪ, library_ਅੰਕੜੇ_t scaling_type,
                           ਮੈਟ੍ਰਿਕਸ_info_t<ਦਸ਼ਮਲਵ> * ਮੈਟ੍ਰਿਕਸ_info) {
        std::uint64_t ਕੁੰਜੀ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ਚੋਣ (ਕੁੰਜੀ)
        {
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_double, library_ਅੰਕੜੇ_t::real_double,
            library_ਅੰਕੜੇ_t::real_double, library_ਅੰਕੜੇ_t::real_double):
        {
            detail::gemm_batch_impl<ਦੁੱਗਣਾ, ਦੁੱਗਣਾ, ਦੁੱਗਣਾ, ਦੁੱਗਣਾ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __INTEL_MKL__
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_bfloat16,
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_bfloat16,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
#ਜੇ_ਅੰਤ
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_int8, library_ਅੰਕੜੇ_t::real_int8,
            library_ਅੰਕੜੇ_t::real_int32, library_ਅੰਕੜੇ_t::real_int32):
        {
            ਦਸ਼ਮਲਵ alpha_float =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ std::int32_t *>(alpha), q);
            ਦਸ਼ਮਲਵ beta_float =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_ਮਾਪ,
                ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_int8, library_ਅੰਕੜੇ_t::real_int8,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_float):
        {
            ਦਸ਼ਮਲਵ alpha_ਮੁੱਲ_2 =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(alpha), q);
            ਦਸ਼ਮਲਵ beta_ਮੁੱਲ =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(beta), q);
            sycl::half alpha_half(alpha_ਮੁੱਲ_2);
            sycl::half beta_half(beta_ਮੁੱਲ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_ਮਾਪ, ਮੈਟ੍ਰਿਕਸ_info);
            ਰੋਕੋ;
        }
        ਮੂਲ:
            ਸੁੱਟੋ std::runtime_error("the combination of data type is unsupported");
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
    ਸਤਰਅੰਦਰ ਖਾਲੀ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ਪੂਰਨ m,
                           ਪੂਰਨ n, ਪੂਰਨ k, ਅਚਲ ਖਾਲੀ * alpha, ਅਚਲ ਖਾਲੀ * a, library_ਅੰਕੜੇ_t a_type, ਪੂਰਨ lda,
                           ਲੰਮਾ ਲੰਮਾ ਪੂਰਨ stride_a, ਅਚਲ ਖਾਲੀ * b, library_ਅੰਕੜੇ_t b_type, ਪੂਰਨ ldb,
                           ਲੰਮਾ ਲੰਮਾ ਪੂਰਨ stride_b, ਅਚਲ ਖਾਲੀ * beta, ਖਾਲੀ * c, library_ਅੰਕੜੇ_t c_type, ਪੂਰਨ ldc,
                           ਲੰਮਾ ਲੰਮਾ ਪੂਰਨ stride_c, ਪੂਰਨ batch_ਮਾਪ, library_ਅੰਕੜੇ_t scaling_type) {
        ਜੇ (scaling_type == library_ਅੰਕੜੇ_t::real_float &&
            c_type == library_ਅੰਕੜੇ_t::complex_float)
        {
            scaling_type = library_ਅੰਕੜੇ_t::complex_float;
        }
        ਨਹੀਂਤਾਂ ਜੇ (scaling_type == library_ਅੰਕੜੇ_t::real_double &&
                 c_type == library_ਅੰਕੜੇ_t::complex_double)
        {
            scaling_type = library_ਅੰਕੜੇ_t::complex_double;
        }

        std::uint64_t ਕੁੰਜੀ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ਚੋਣ (ਕੁੰਜੀ)
        {
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_double, library_ਅੰਕੜੇ_t::real_double,
            library_ਅੰਕੜੇ_t::real_double, library_ਅੰਕੜੇ_t::real_double):
        {
            detail::gemm_batch_impl<ਦੁੱਗਣਾ, ਦੁੱਗਣਾ, ਦੁੱਗਣਾ, ਦੁੱਗਣਾ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::complex_float, library_ਅੰਕੜੇ_t::complex_float,
            library_ਅੰਕੜੇ_t::complex_float, library_ਅੰਕੜੇ_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<ਦਸ਼ਮਲਵ>, std::complex<ਦਸ਼ਮਲਵ>,
                                    std::complex<ਦਸ਼ਮਲਵ>, std::complex<ਦਸ਼ਮਲਵ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::complex_double, library_ਅੰਕੜੇ_t::complex_double,
            library_ਅੰਕੜੇ_t::complex_double, library_ਅੰਕੜੇ_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<ਦੁੱਗਣਾ>, std::complex<ਦੁੱਗਣਾ>,
                                    std::complex<ਦੁੱਗਣਾ>, std::complex<ਦੁੱਗਣਾ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __INTEL_MKL__
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_bfloat16,
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_bfloat16, library_ਅੰਕੜੇ_t::real_bfloat16,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_ਮਾਪ);
            ਰੋਕੋ;
        }
#ਜੇ_ਅੰਤ
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_int8, library_ਅੰਕੜੇ_t::real_int8,
            library_ਅੰਕੜੇ_t::real_int32, library_ਅੰਕੜੇ_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_int8, library_ਅੰਕੜੇ_t::real_int8,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_float, library_ਅੰਕੜੇ_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ਦਸ਼ਮਲਵ, ਦਸ਼ਮਲਵ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਹਾਲਤ detail::get_type_combination_id(
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_half,
            library_ਅੰਕੜੇ_t::real_half, library_ਅੰਕੜੇ_t::real_float):
        {
            ਦਸ਼ਮਲਵ alpha_ਮੁੱਲ_2 =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(alpha), q);
            ਦਸ਼ਮਲਵ beta_ਮੁੱਲ =
                dpct::get_ਮੁੱਲ(reinterpret_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(beta), q);
            sycl::half alpha_half(alpha_ਮੁੱਲ_2);
            sycl::half beta_half(beta_ਮੁੱਲ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_ਮਾਪ);
            ਰੋਕੋ;
        }
        ਮੂਲ:
            ਸੁੱਟੋ std::runtime_error("the combination of data type is unsupported");
        }
    }

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ
    async_dpct_memcpy(ਖਾਲੀ *ਮੰਜ਼ਿਲ_ptr, size_t ਮੰਜ਼ਿਲ_pitch, ਅਚਲ ਖਾਲੀ *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_ਕਤਾਰ())
    {
        detail::dpct_memcpy(q, ਮੰਜ਼ਿਲ_ptr, from_ptr, ਮੰਜ਼ਿਲ_pitch, from_pitch, x, y,
                            direction);
    }

    ਵਰਤੋਂ err0 = detail::generic_error_type<ਬਣਤਰ err0_ਟੈਗ, ਪੂਰਨ>;
    ਵਰਤੋਂ err1 = detail::generic_error_type<ਬਣਤਰ err1_ਟੈਗ, ਪੂਰਨ>;

    ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dpct_free(ਖਾਲੀ *ptr, sycl::queue &q = get_default_ਕਤਾਰ()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    ਸਾਂਚਾ <ਵਰਗ T, ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2, size_t Dimension> ਵਰਗ accessor;
    ਸਾਂਚਾ <ਵਰਗ T, ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2> ਵਰਗ accessor<T, ਸਮ੍ਰਿਤੀ_2, 3> {
    ਜਨਤਕ:
        ਵਰਤੋਂ ਸਮ੍ਰਿਤੀ_t = detail::ਸਮ੍ਰਿਤੀ_traits<ਸਮ੍ਰਿਤੀ_2, T>;
        ਵਰਤੋਂ element_t = ਕਿਸਮ_ਨਾਂ ਸਮ੍ਰਿਤੀ_t::element_t;
        ਵਰਤੋਂ ਸੂਚਕ_t = ਕਿਸਮ_ਨਾਂ ਸਮ੍ਰਿਤੀ_t::ਸੂਚਕ_t;
        ਵਰਤੋਂ accessor_t = ਕਿਸਮ_ਨਾਂ ਸਮ੍ਰਿਤੀ_t::ਸਾਂਚਾ accessor_t<3>;
        accessor(ਸੂਚਕ_t ਅੰਕੜੇ, ਅਚਲ sycl::range<3> &ਵਿੱਚ_range)
            : ਅੰਕੜੇ_2(ਅੰਕੜੇ), _range(ਵਿੱਚ_range) {}
        ਸਾਂਚਾ <ਸਮ੍ਰਿਤੀ_region M = ਸਮ੍ਰਿਤੀ_2>
        accessor(ਕਿਸਮ_ਨਾਂ std::enable_if<M != local, ਅਚਲ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ਅਚਲ accessor_t &acc, ਅਚਲ sycl::range<3> &ਵਿੱਚ_range)
            : accessor(acc.get_ਸੂਚਕ(), ਵਿੱਚ_range) {}
        accessor<T, ਸਮ੍ਰਿਤੀ_2, 2> ਸੰਚਾਲਕ[](size_t ਸੂਚਕਾਂਕ) ਅਚਲ {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            ਵਾਪਸ accessor<T, ਸਮ੍ਰਿਤੀ_2, 2>(ਅੰਕੜੇ_2 + ਸੂਚਕਾਂਕ * sub.size(), sub);
        }

        ਸੂਚਕ_t get_ptr() ਅਚਲ { ਵਾਪਸ ਅੰਕੜੇ_2; }

    ਨਿਜੀ:
        ਸੂਚਕ_t ਅੰਕੜੇ_2;
        sycl::range<3> _range;
    };
    ਸਾਂਚਾ <ਵਰਗ T, ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2> ਵਰਗ accessor<T, ਸਮ੍ਰਿਤੀ_2, 2> {
    ਜਨਤਕ:
        ਵਰਤੋਂ ਸਮ੍ਰਿਤੀ_t = detail::ਸਮ੍ਰਿਤੀ_traits<ਸਮ੍ਰਿਤੀ_2, T>;
        ਵਰਤੋਂ element_t = ਕਿਸਮ_ਨਾਂ ਸਮ੍ਰਿਤੀ_t::element_t;
        ਵਰਤੋਂ ਸੂਚਕ_t = ਕਿਸਮ_ਨਾਂ ਸਮ੍ਰਿਤੀ_t::ਸੂਚਕ_t;
        ਵਰਤੋਂ accessor_t = ਕਿਸਮ_ਨਾਂ ਸਮ੍ਰਿਤੀ_t::ਸਾਂਚਾ accessor_t<2>;
        accessor(ਸੂਚਕ_t ਅੰਕੜੇ, ਅਚਲ sycl::range<2> &ਵਿੱਚ_range)
            : ਅੰਕੜੇ_2(ਅੰਕੜੇ), _range(ਵਿੱਚ_range) {}
        ਸਾਂਚਾ <ਸਮ੍ਰਿਤੀ_region M = ਸਮ੍ਰਿਤੀ_2>
        accessor(ਕਿਸਮ_ਨਾਂ std::enable_if<M != local, ਅਚਲ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ਅਚਲ accessor_t &acc, ਅਚਲ sycl::range<2> &ਵਿੱਚ_range)
            : accessor(acc.get_ਸੂਚਕ(), ਵਿੱਚ_range) {}

        ਸੂਚਕ_t ਸੰਚਾਲਕ[](size_t ਸੂਚਕਾਂਕ) ਅਚਲ {
            ਵਾਪਸ ਅੰਕੜੇ_2 + _range.get(1) * ਸੂਚਕਾਂਕ;
        }

        ਸੂਚਕ_t get_ptr() ਅਚਲ { ਵਾਪਸ ਅੰਕੜੇ_2; }

    ਨਿਜੀ:
        ਸੂਚਕ_t ਅੰਕੜੇ_2;
        sycl::range<2> _range;
    };

    ਨਾਮਾਕਾਸ਼ detail {
        /// Device variable with address space of shared, global or constant.
        ਸਾਂਚਾ <ਵਰਗ T, ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2, size_t Dimension> ਵਰਗ device_ਸਮ੍ਰਿਤੀ {
        ਜਨਤਕ:
            ਵਰਤੋਂ accessor_t =
                ਕਿਸਮ_ਨਾਂ detail::ਸਮ੍ਰਿਤੀ_traits<ਸਮ੍ਰਿਤੀ_2,
                                            T>::ਸਾਂਚਾ accessor_t<Dimension>;
            ਵਰਤੋਂ ਮੁੱਲ_t = ਕਿਸਮ_ਨਾਂ detail::ਸਮ੍ਰਿਤੀ_traits<ਸਮ੍ਰਿਤੀ_2, T>::ਮੁੱਲ_t;
            ਵਰਤੋਂ dpct_accessor_t = dpct::accessor<T, ਸਮ੍ਰਿਤੀ_2, Dimension>;

            device_ਸਮ੍ਰਿਤੀ() : device_ਸਮ੍ਰਿਤੀ(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_ਸਮ੍ਰਿਤੀ(ਅਚਲ sycl::range<Dimension> &ਵਿੱਚ_range,
                        std::initializer_list<ਮੁੱਲ_t> &&init_list)
                : device_ਸਮ੍ਰਿਤੀ(ਵਿੱਚ_range) {
                ਪ੍ਰਤਿਜ੍ਞਾ(init_list.size() <= ਵਿੱਚ_range.size());
                _host_ptr = (ਮੁੱਲ_t *)std::ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਮਾਪ_2);
                std::memset(_host_ptr, 0, ਮਾਪ_2);
                std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(_host_ptr, init_list.begin(), init_list.size() * ਆਕਾਰ(T));
            }

            /// Constructor of 2-D array with initializer list
            ਸਾਂਚਾ <size_t D = Dimension>
            device_ਸਮ੍ਰਿਤੀ(
                ਅਚਲ ਕਿਸਮ_ਨਾਂ std::enable_if<D == 2, sycl::range<2>>::type &ਵਿੱਚ_range,
                std::initializer_list<std::initializer_list<ਮੁੱਲ_t>> &&init_list)
                : device_ਸਮ੍ਰਿਤੀ(ਵਿੱਚ_range) {
                ਪ੍ਰਤਿਜ੍ਞਾ(init_list.size() <= ਵਿੱਚ_range[0]);
                _host_ptr = (ਮੁੱਲ_t *)std::ਸਮ੍ਰਿਤੀ_ਦਿਓ(ਮਾਪ_2);
                std::memset(_host_ptr, 0, ਮਾਪ_2);
                ਸਵੈਚਾਲਿਤ tmp_ਅੰਕੜੇ = _host_ptr;
                ਲਈ (ਸਵੈਚਾਲਿਤ sub_list : init_list) {
                    ਪ੍ਰਤਿਜ੍ਞਾ(sub_list.size() <= ਵਿੱਚ_range[1]);
                    std::ਸਮ੍ਰਿਤੀ_ਨਕਲ(tmp_ਅੰਕੜੇ, sub_list.begin(),
                                sub_list.size() * ਆਕਾਰ(T));
                    tmp_ਅੰਕੜੇ += ਵਿੱਚ_range[1];
                }
            }

            /// Constructor with range
            device_ਸਮ੍ਰਿਤੀ(ਅਚਲ sycl::range<Dimension> &range_ਵਿੱਚ)
                : ਮਾਪ_2(range_ਵਿੱਚ.size() * ਆਕਾਰ(T)), _range(range_ਵਿੱਚ),
                _reference(ਝੂਠ), _host_ptr(ਖਾਲੀ_ਸੂਚਕ), _device_ptr(ਖਾਲੀ_ਸੂਚਕ) {
                static_assert(
                    (ਸਮ੍ਰਿਤੀ_2 == global) || (ਸਮ੍ਰਿਤੀ_2 == constant) || (ਸਮ੍ਰਿਤੀ_2 == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            ਸਾਂਚਾ <ਵਰਗ... ਦਲੀਲਾਂ>
            device_ਸਮ੍ਰਿਤੀ(ਦਲੀਲਾਂ... Arguments)
                : device_ਸਮ੍ਰਿਤੀ(sycl::range<Dimension>(Arguments...)) {}

            ~device_ਸਮ੍ਰਿਤੀ() {
                ਜੇ (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                ਜੇ (_host_ptr)
                    std::ਮੁਕਤ(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            ਖਾਲੀ init() { init(dpct::get_default_ਕਤਾਰ()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            ਖਾਲੀ init(sycl::queue &q) {
                ਜੇ (_device_ptr)
                    ਵਾਪਸ;
                ਜੇ (!ਮਾਪ_2)
                    ਵਾਪਸ;
                allocate_device(q);
                ਜੇ (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, ਮਾਪ_2,
                                        host_ਮੰਜ਼ਿਲ_device);
            }

            /// The variable is assigned to a device pointer.
            ਖਾਲੀ ਸੌਂਪੋ(ਮੁੱਲ_t *src, size_t size) {
                ਇਹ->~device_ਸਮ੍ਰਿਤੀ();
                ਨਵਾਂ (ਇਹ) device_ਸਮ੍ਰਿਤੀ(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ਮੁੱਲ_t *get_ptr() { ਵਾਪਸ get_ptr(get_default_ਕਤਾਰ()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ਮੁੱਲ_t *get_ptr(sycl::queue &q) {
                init(q);
                ਵਾਪਸ _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_ਮਾਪ() { ਵਾਪਸ ਮਾਪ_2; }

            ਸਾਂਚਾ <size_t D = Dimension>
            ਕਿਸਮ_ਨਾਂ std::enable_if<D == 1, T>::type &ਸੰਚਾਲਕ[](size_t ਸੂਚਕਾਂਕ) {
                init();
                ਵਾਪਸ _device_ptr[ਸੂਚਕਾਂਕ];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            ਸਾਂਚਾ <size_t D = Dimension>
            ਕਿਸਮ_ਨਾਂ std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                ਵਾਪਸ dpct_accessor_t((T *)_device_ptr, _range);
            }

        ਨਿਜੀ:
            device_ਸਮ੍ਰਿਤੀ(ਮੁੱਲ_t *ਸਮ੍ਰਿਤੀ_ptr, size_t size)
                : ਮਾਪ_2(size), _range(size / ਆਕਾਰ(T)), _reference(ਸੱਚ),
                _device_ptr(ਸਮ੍ਰਿਤੀ_ptr) {}

            ਖਾਲੀ allocate_device(sycl::queue &q) {
        #ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ DPCT_USM_LEVEL_NONE
                ਜੇ (ਸਮ੍ਰਿਤੀ_2 == shared) {
                    _device_ptr = (ਮੁੱਲ_t *)sycl::malloc_shared(ਮਾਪ_2, q.get_device(),
                                                                q.get_context());
                    ਵਾਪਸ;
                }
        #ਜੇ_ਪਰਿਭਾਸ਼ਿਤ SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                ਜੇ (ਸਮ੍ਰਿਤੀ_2 == constant) {
                    _device_ptr = (ਮੁੱਲ_t *)sycl::malloc_device(
                        ਮਾਪ_2, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    ਵਾਪਸ;
                }
        #ਜੇ_ਅੰਤ
        #ਜੇ_ਅੰਤ
                _device_ptr = (ਮੁੱਲ_t *)detail::dpct_malloc(ਮਾਪ_2, q);
            }

            size_t ਮਾਪ_2;
            sycl::range<Dimension> _range;
            ਤਾਰਕਿਕ _reference;
            ਮੁੱਲ_t *_host_ptr;
            ਮੁੱਲ_t *_device_ptr;
        };
        ਸਾਂਚਾ <ਵਰਗ T, ਸਮ੍ਰਿਤੀ_region ਸਮ੍ਰਿਤੀ_2>
        ਵਰਗ device_ਸਮ੍ਰਿਤੀ<T, ਸਮ੍ਰਿਤੀ_2, 0> : ਜਨਤਕ device_ਸਮ੍ਰਿਤੀ<T, ਸਮ੍ਰਿਤੀ_2, 1> {
        ਜਨਤਕ:
            ਵਰਤੋਂ ਅਧਾਰ = device_ਸਮ੍ਰਿਤੀ<T, ਸਮ੍ਰਿਤੀ_2, 1>;
            ਵਰਤੋਂ ਮੁੱਲ_t = ਕਿਸਮ_ਨਾਂ ਅਧਾਰ::ਮੁੱਲ_t;
            ਵਰਤੋਂ accessor_t =
                ਕਿਸਮ_ਨਾਂ detail::ਸਮ੍ਰਿਤੀ_traits<ਸਮ੍ਰਿਤੀ_2, T>::ਸਾਂਚਾ accessor_t<0>;

            /// Constructor with initial value.
            device_ਸਮ੍ਰਿਤੀ(ਅਚਲ ਮੁੱਲ_t &val) : ਅਧਾਰ(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_ਸਮ੍ਰਿਤੀ() : ਅਧਾਰ(1) {}
        };
        } // namespace detail

    ਸਾਂਚਾ <ਵਰਗ T, size_t Dimension>
    ਵਰਤੋਂ global_ਸਮ੍ਰਿਤੀ = detail::device_ਸਮ੍ਰਿਤੀ<T, global, Dimension>;
    ਸਾਂਚਾ <ਵਰਗ T, size_t Dimension>
    ਵਰਤੋਂ constant_ਸਮ੍ਰਿਤੀ = detail::device_ਸਮ੍ਰਿਤੀ<T, constant, Dimension>;
    ਸਾਂਚਾ <ਵਰਗ T, size_t Dimension>
    ਵਰਤੋਂ shared_ਸਮ੍ਰਿਤੀ = detail::device_ਸਮ੍ਰਿਤੀ<T, shared, Dimension>;


    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T,
            sycl::access::ਪਤਾ_space ਪਤਾ_Space =
                sycl::access::ਪਤਾ_space::global_space,
            sycl::memory_order ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ = sycl::memory_order::relaxed,
            sycl::ਸਮ੍ਰਿਤੀ_scope ਸਮ੍ਰਿਤੀ_Scope = sycl::ਸਮ੍ਰਿਤੀ_scope::device>
    ਸਤਰਅੰਦਰ T atomic_fetch_add(T *addr, T ਸੰਕਾਰਜ) {
    ਸਵੈਚਾਲਿਤ atm =
        sycl::atomic_ref<T, ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ, ਸਮ੍ਰਿਤੀ_Scope, ਪਤਾ_Space>(addr[0]);
    ਵਾਪਸ atm.fetch_add(ਸੰਕਾਰਜ);
    }

    ਸਾਂਚਾ <sycl::access::ਪਤਾ_space ਪਤਾ_Space =
                sycl::access::ਪਤਾ_space::global_space,
            sycl::memory_order ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ = sycl::memory_order::relaxed,
            sycl::ਸਮ੍ਰਿਤੀ_scope ਸਮ੍ਰਿਤੀ_Scope = sycl::ਸਮ੍ਰਿਤੀ_scope::device,
            ਕਿਸਮ_ਨਾਂ T1, ਕਿਸਮ_ਨਾਂ T2>
    ਸਤਰਅੰਦਰ T1 atomic_fetch_add(T1 *addr, T2 ਸੰਕਾਰਜ) {
    ਸਵੈਚਾਲਿਤ atm =
        sycl::atomic_ref<T1, ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ, ਸਮ੍ਰਿਤੀ_Scope, ਪਤਾ_Space>(addr[0]);
    ਵਾਪਸ atm.fetch_add(ਸੰਕਾਰਜ);
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, sycl::access::ਪਤਾ_space ਪਤਾ_Space =
                            sycl::access::ਪਤਾ_space::global_space>
    ਸਤਰਅੰਦਰ T atomic_fetch_add(T *addr, T ਸੰਕਾਰਜ,
                            sycl::memory_order ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ) {
    ਚੋਣ (ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ) {
        ਹਾਲਤ sycl::memory_order::relaxed:
            ਵਾਪਸ atomic_fetch_add<T, ਪਤਾ_Space, sycl::memory_order::relaxed,
                                    sycl::ਸਮ੍ਰਿਤੀ_scope::device>(addr, ਸੰਕਾਰਜ);
        ਹਾਲਤ sycl::memory_order::acq_rel:
            ਵਾਪਸ atomic_fetch_add<T, ਪਤਾ_Space, sycl::memory_order::acq_rel,
                                    sycl::ਸਮ੍ਰਿਤੀ_scope::device>(addr, ਸੰਕਾਰਜ);
        ਹਾਲਤ sycl::memory_order::seq_cst:
            ਵਾਪਸ atomic_fetch_add<T, ਪਤਾ_Space, sycl::memory_order::seq_cst,
                                    sycl::ਸਮ੍ਰਿਤੀ_scope::device>(addr, ਸੰਕਾਰਜ);
        ਮੂਲ:
            ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    ਸਾਂਚਾ <sycl::access::ਪਤਾ_space ਪਤਾ_Space =
                sycl::access::ਪਤਾ_space::global_space,
            ਕਿਸਮ_ਨਾਂ T1, ਕਿਸਮ_ਨਾਂ T2>
    ਸਤਰਅੰਦਰ T1 atomic_fetch_add(T1 *addr, T2 ਸੰਕਾਰਜ,
                            sycl::memory_order ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ) {
    atomic_fetch_add<T1, ਪਤਾ_Space>(addr, ਸੰਕਾਰਜ, ਸਮ੍ਰਿਤੀ_ਕ੍ਰਮ);
    }

    ਸਤਰਅੰਦਰ ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ byte_level_ਕ੍ਰਮ_ਬਦਲੋ(
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ a, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ b, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ s) {
      ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      ਵਾਪਸ ret;
    }

    ਸਤਰਅੰਦਰ uint32_t byte_level_ਕ੍ਰਮ_ਬਦਲੋ_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, ਪੂਰਨ mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      ਜੇ (mode >= 1 && mode <= 6) {
        ਵਾਪਸ byte_level_ਕ੍ਰਮ_ਬਦਲੋ(low32, high32, lookup[mode - 1][sel & 0x3]);
      } ਨਹੀਂਤਾਂ ਜੇ (!mode) {
        ਵਾਪਸ byte_level_ਕ੍ਰਮ_ਬਦਲੋ(low32, high32, sel);
      }
      ਵਾਪਸ 0;
    }

    ਸਾਂਚਾ <ਪੂਰਨ n_nondefault_params, ਪੂਰਨ n_default_params, ਕਿਸਮ_ਨਾਂ T>
    ਵਰਗ ਦਲੀਲਾਂ_selector;

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
    ਸਾਂਚਾ <ਪੂਰਨ n_nondefault_params, ਪੂਰਨ n_default_params, ਕਿਸਮ_ਨਾਂ R,
              ਕਿਸਮ_ਨਾਂ... Ts>
    ਵਰਗ ਦਲੀਲਾਂ_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      ਨਿਜੀ:
        ਖਾਲੀ **kernel_params;
        ਅੱਖਰ *ਦਲੀਲਾਂ_ਬਫਰ;

        ਸਾਂਚਾ <ਪੂਰਨ i> ਸਥਿਰ constexpr ਪੂਰਨ account_for_default_params() {
            constexpr ਪੂਰਨ n_ਕੁੱਲ_params = ਆਕਾਰ...(Ts);
            ਜੇ constexpr (i >= n_nondefault_params) {
                ਵਾਪਸ n_ਕੁੱਲ_params - n_default_params +
                       (i - n_nondefault_params);
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ i;
            }
        }

      ਜਨਤਕ:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        ਸਾਂਚਾ <ਪੂਰਨ i>
        ਵਰਤੋਂ arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ਸਥਿਰ constexpr ਪੂਰਨ params_ਅੰਕ = ਆਕਾਰ...(Ts);

      ਨਿਜੀ:
        ਸਾਂਚਾ <ਪੂਰਨ i> ਸਥਿਰ constexpr ਪੂਰਨ get_ਵਿੱਥ() {
            ਜੇ constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                ਵਾਪਸ 0;
            } ਨਹੀਂਤਾਂ {
                constexpr ਪੂਰਨ prev_off = get_ਵਿੱਥ<i - 1>();
                constexpr ਪੂਰਨ prev_past_ਸਿਰਾ =
                    prev_off + ਆਕਾਰ(arg_type<i - 1>);
                ਵਰਤੋਂ T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                ਜੇ constexpr (prev_past_ਸਿਰਾ % alignof(T) == 0) {
                    ਵਾਪਸ prev_past_ਸਿਰਾ;
                }
                // otherwise bump prev_past_end to match alignment
                ਨਹੀਂਤਾਂ {
                    ਵਾਪਸ prev_past_ਸਿਰਾ +
                           (alignof(T) - (prev_past_ਸਿਰਾ % alignof(T)));
                }
            }
        }

        ਸਥਿਰ ਅੱਖਰ *get_ਦਲੀਲਾਂ_ਬਫਰ(ਖਾਲੀ **extra) {
            ਜੇ (!extra)
                ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
            ਲਈ (; (std::size_t)*extra != 0; ++extra) {
                ਜੇ ((std::size_t)*extra == 1) {
                    ਵਾਪਸ static_cast<ਅੱਖਰ *>(*(extra + 1));
                }
            }
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

      ਜਨਤਕ:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        ਦਲੀਲਾਂ_selector(ਖਾਲੀ **kernel_params, ਖਾਲੀ **extra)
            : kernel_params(kernel_params),
              ਦਲੀਲਾਂ_ਬਫਰ(get_ਦਲੀਲਾਂ_ਬਫਰ(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        ਸਾਂਚਾ <ਪੂਰਨ i> arg_type<i> &get() {
            ਜੇ (kernel_params) {
                ਵਾਪਸ *static_cast<arg_type<i> *>(kernel_params[i]);
            } ਨਹੀਂਤਾਂ {
                ਵਾਪਸ *reinterpret_cast<arg_type<i> *>(ਦਲੀਲਾਂ_ਬਫਰ +
                                                        get_ਵਿੱਥ<i>());
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
    ਵਰਗ kernel_launcher {
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ FuncT, ਕਿਸਮ_ਨਾਂ ArgSelector, std::size_t... ਸੂਚਕਾਂਕ_2>
        ਸਥਿਰ ਖਾਲੀ launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<ਸੂਚਕਾਂਕ_2...>) {
            func(selector.ਸਾਂਚਾ get<ਸੂਚਕਾਂਕ_2>()...);
        }
        ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_execution_config(dim3 ਸਮੂਹ_range, dim3 local_range,
                                         ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ local_mem_ਮਾਪ_2,
                                         ਕਤਾਰ_ptr que) {
            ਜੇ (que) {
                _que = que;
            } ਨਹੀਂਤਾਂ {
                _que = &get_default_ਕਤਾਰ();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(ਸਮੂਹ_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_ਮਾਪ = local_mem_ਮਾਪ_2;


        }
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::mutex kernel_function_ptr_map_mutex;

      ਜਨਤਕ:
        /// Variables for storing execution configuration.
        ਸਥਿਰ ਸਤਰਅੰਦਰ thread_local sycl::queue *_que = ਖਾਲੀ_ਸੂਚਕ;
        ਸਥਿਰ ਸਤਰਅੰਦਰ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ਸਥਿਰ ਸਤਰਅੰਦਰ thread_local ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ local_mem_ਮਾਪ = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ਸਥਿਰ ਸਤਰਅੰਦਰ std::map<
            ਅਚਲ ਖਾਲੀ *,
            std::function<ਖਾਲੀ(dim3, dim3, ਖਾਲੀ **, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ, ਕਤਾਰ_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ਸਥਿਰ ਖਾਲੀ ਰਜਿਸਟਰ_kernel_ptr(
            ਅਚਲ ਖਾਲੀ *func,
            std::function<ਖਾਲੀ(dim3, dim3, ਖਾਲੀ **, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ, ਕਤਾਰ_ptr)>
                launcher) {
            std::lock_guard<std::mutex> ਤਾਲਾ(kernel_function_ptr_map_mutex);
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
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ FuncT, ਕਿਸਮ_ਨਾਂ... ਦਲੀਲਾਂ_T>
        ਸਥਿਰ std::enable_if_t<std::is_invocable_v<FuncT *, ਦਲੀਲਾਂ_T...>, ਖਾਲੀ>
        launch(FuncT *func, dim3 ਸਮੂਹ_range, dim3 local_range,
               ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ local_mem_ਮਾਪ_2, ਕਤਾਰ_ptr que, ਦਲੀਲਾਂ_T... ਦਲੀਲਾਂ_2) {
            ਸੈੱਟ_execution_config(ਸਮੂਹ_range, local_range, local_mem_ਮਾਪ_2, que);
            func(ਦਲੀਲਾਂ_2...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        ਸਥਿਰ ਖਾਲੀ launch(ਅਚਲ ਖਾਲੀ *func, dim3 ਸਮੂਹ_range, dim3 local_range,
                           ਖਾਲੀ **ਦਲੀਲਾਂ_2, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ local_mem_ਮਾਪ_2,
                           ਕਤਾਰ_ptr que) {
            std::lock_guard<std::mutex> ਤਾਲਾ(kernel_function_ptr_map_mutex);
            ਸਵੈਚਾਲਿਤ Iter = kernel_function_ptr_map.find(func);
            ਜੇ (Iter == kernel_function_ptr_map.end()) {
                ਸੁੱਟੋ std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(ਸਮੂਹ_range, local_range, ਦਲੀਲਾਂ_2, local_mem_ਮਾਪ_2, que);
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
        ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ FuncT>
        ਸਥਿਰ std::enable_if_t<std::is_function_v<FuncT>, ਖਾਲੀ>
        launch(FuncT *func, dim3 ਸਮੂਹ_range, dim3 local_range, ਖਾਲੀ **ਦਲੀਲਾਂ_2,
               ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ local_mem_ਮਾਪ_2, ਕਤਾਰ_ptr que) {
            constexpr size_t p_ਅੰਕ = ਦਲੀਲਾਂ_selector<0, 0, FuncT>::params_ਅੰਕ;
            ਸੈੱਟ_execution_config(ਸਮੂਹ_range, local_range, local_mem_ਮਾਪ_2, que);
            ਦਲੀਲਾਂ_selector<p_ਅੰਕ, p_ਅੰਕ, FuncT> selector(ਦਲੀਲਾਂ_2, ਖਾਲੀ_ਸੂਚਕ);
            launch_helper(func, selector, std::make_index_sequence<p_ਅੰਕ>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    T select_from_sub_ਸਮੂਹ(
        sycl::sub_ਸਮੂਹ g,
        T x,
        ਪੂਰਨ remote_local_id,
        ਪੂਰਨ logical_sub_ਸਮੂਹ_ਮਾਪ = 32) {
      ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ = g.get_local_ਰੇਖਿਕ_id() /
                                 logical_sub_ਸਮੂਹ_ਮਾਪ *
                                 logical_sub_ਸਮੂਹ_ਮਾਪ;
      ਵਾਪਸ sycl::select_from_ਸਮੂਹ(
          g, x, ਸ਼ੁਰੂ_ਸੂਚਕਾਂਕ + remote_local_id % logical_sub_ਸਮੂਹ_ਮਾਪ);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਖਾਲੀ ldmatrix(uintptr_t addr, T* m, ਤਾਰਕਿਕ trans = ਝੂਠ, ਬਿਨਾਨਿਸ਼ਾਨੀ mat = 0) {
      ਸਵੈਚਾਲਿਤ sg = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ();
      ਪੂਰਨ lane = sg.get_local_ਰੇਖਿਕ_id();

      ਪੂਰਨ lane_group8_ਪੰਕਤੀ = lane / 8;
      ਪੂਰਨ lane_group8_ਖਾਨਾ = lane % 8;

      ਜੇ (!trans) {
        // calculate the source lane
        ਪੂਰਨ src_lane = 2 * lane_group8_ਪੰਕਤੀ;
        ਜੇ (lane_group8_ਖਾਨਾ >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        ਸਵੈਚਾਲਿਤ recv_addr_uintp =
            dpct::select_from_sub_ਸਮੂਹ(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        ਸਵੈਚਾਲਿਤ recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_ਖਾਨਾ % 4];
      } ਨਹੀਂਤਾਂ {
        // calculate the source lane
        ਪੂਰਨ src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        ਸਵੈਚਾਲਿਤ recv_addr_uintp_1 =
            dpct::select_from_sub_ਸਮੂਹ(sg, addr, mat * 8 + src_lane);
        ਸਵੈਚਾਲਿਤ recv_addr_uintp_2 =
            dpct::select_from_sub_ਸਮੂਹ(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        ਸਵੈਚਾਲਿਤ recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        ਸਵੈਚਾਲਿਤ recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        ਪੂਰਨ ਸੂਚਕਾਂਕ = lane / 4;
        sycl::half val0 = recv_addr_1[ਸੂਚਕਾਂਕ];
        sycl::half val1 = recv_addr_2[ਸੂਚਕਾਂਕ];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਖਾਲੀ ldmatrix(uintptr_t addr, T* m1, T* m2, ਤਾਰਕਿਕ trans = ਝੂਠ) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਖਾਲੀ ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, ਤਾਰਕਿਕ trans = ਝੂਠ) {
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
    ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
    ਬਣਤਰ MMAType {
      ਵਰਤੋਂ PackType = uint32_t;
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
    ਸਾਂਚਾ <ਪੂਰਨ M, ਪੂਰਨ N, ਪੂਰਨ K, ਕਿਸਮ_ਨਾਂ ABType, ਕਿਸਮ_ਨਾਂ CDType>
    ਖਾਲੀ mma(
        ਅਸਥਿਰ ਖਾਲੀ** d_mat_frag,
        ਖਾਲੀ* a_mat_frag,
        ਖਾਲੀ* b_mat_frag,
        ਖਾਲੀ* c_mat_frag) {
      ਸਵੈਚਾਲਿਤ d = reinterpret_cast<ਅਸਥਿਰ CDType**>(d_mat_frag);
      ਸਵੈਚਾਲਿਤ a =
          reinterpret_cast<ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType*>(a_mat_frag);
      ਸਵੈਚਾਲਿਤ b =
          reinterpret_cast<ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType*>(b_mat_frag);
      ਸਵੈਚਾਲਿਤ c = reinterpret_cast<CDType*>(c_mat_frag);

      ਸਵੈਚਾਲਿਤ sg = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_sub_ਸਮੂਹ();
      ਪੂਰਨ lane = sg.get_local_ਰੇਖਿਕ_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      ਛੋਟਾ ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ = 4 * (lane >> 2);
      ਛੋਟਾ ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ = 8 * (lane % 4);

      ਜੇ constexpr (M == 8 && N == 8 && K == 4) {
        ਜੇ constexpr (std::is_floating_point_v<CDType>) {
          ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ = ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ % 16;

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
          ਪੂਰਨ r_ind = (lane % 2) ? 1 : 0;
          ਪੂਰਨ c_ind = ((lane % 4) / 2) ? 2 : 0;

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
          ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a[2], recv_b[2];

          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[0], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[1], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[1], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);

            ਸਵੈਚਾਲਿਤ ra = reinterpret_cast<ABType*>(recv_a);
            ਸਵੈਚਾਲਿਤ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment (for
            // even work item indices) d0 += col0{ a0 } * row0{ b0 } d1 += col0{
            // a0 } * row0{ b1 } d2 += col1{ a2 } * row0{ b0 } d3 += col1{ a2 }
            // * row0{ b1 } (for odd work item indices) d0 += col0{ a1 } * row0{
            // b2 } d1 += col0{ a1 } * row0{ b3 } d2 += col1{ a3 } * row0{ b2 }
            // d3 += col1{ a3 } * row0{ b3 }
            *d[0] +=
                static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind]) * static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind]);
            *d[1] += static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind]) *
                     static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind + 1]);
            *d[2] += static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind + 2]) *
                     static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind]);
            *d[3] += static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind + 2]) *
                     static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind + 1]);

            // Load partial fragment from row1 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[1], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 16);

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
                static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind]) * static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind]);
            *d[5] += static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind]) *
                     static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind + 1]);
            *d[6] += static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind + 2]) *
                     static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind]);
            *d[7] += static_cast<ਦਸ਼ਮਲਵ>(ra[r_ind + 2]) *
                     static_cast<ਦਸ਼ਮਲਵ>(rb[c_ind + 1]);
          }
        }
      } ਨਹੀਂਤਾਂ ਜੇ constexpr (M == 8 && N == 8 && K == 16) {
        ਜੇ constexpr (std::is_integral_v<ABType>) {
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
          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_ਸਮੂਹ(sg, a[0], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 4);

            ਸਵੈਚਾਲਿਤ a = reinterpret_cast<ABType*>(&recv_a);
            ਸਵੈਚਾਲਿਤ b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } ਨਹੀਂਤਾਂ ਜੇ constexpr (M == 16 && N == 8 && K == 8) {
        ਜੇ constexpr (std::is_floating_point_v<CDType>) {
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
          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[0], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[1], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 4);

            ਸਵੈਚਾਲਿਤ ra = reinterpret_cast<ABType*>(recv_a);
            ਸਵੈਚਾਲਿਤ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            ਲਈ (ਪੂਰਨ j = 0; j < 2; j++) {
              *d[0] += static_cast<ਦਸ਼ਮਲਵ>(ra[j]) * static_cast<ਦਸ਼ਮਲਵ>(rb[j]);
              *d[1] +=
                  static_cast<ਦਸ਼ਮਲਵ>(ra[j]) * static_cast<ਦਸ਼ਮਲਵ>(rb[j + 2]);
              *d[2] +=
                  static_cast<ਦਸ਼ਮਲਵ>(ra[j + 2]) * static_cast<ਦਸ਼ਮਲਵ>(rb[j]);
              *d[3] +=
                  static_cast<ਦਸ਼ਮਲਵ>(ra[j + 2]) * static_cast<ਦਸ਼ਮਲਵ>(rb[j + 2]);
            }
          }
        }
      } ਨਹੀਂਤਾਂ ਜੇ constexpr (M == 16 && N == 8 && K == 16) {
        ਜੇ constexpr (std::is_floating_point_v<CDType>) {
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
          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[0], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[2], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[1], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[3], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[1], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[1], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + 4 + i);

            ਸਵੈਚਾਲਿਤ ra = reinterpret_cast<ABType*>(recv_a);
            ਸਵੈਚਾਲਿਤ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } ਨਹੀਂਤਾਂ ਜੇ constexpr (std::is_integral_v<ABType>) {
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
          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[0], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[1], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 4);

            ਸਵੈਚਾਲਿਤ ra = reinterpret_cast<ABType*>(recv_a);
            ਸਵੈਚਾਲਿਤ rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } ਨਹੀਂਤਾਂ ਜੇ constexpr (M == 16 && N == 8 && K == 32) {
        ਜੇ constexpr (std::is_integral_v<ABType>) {
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
          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[0], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[1], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[0], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 4);

            ਸਵੈਚਾਲਿਤ a = reinterpret_cast<ABType*>(recv_a);
            ਸਵੈਚਾਲਿਤ b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            ਕਿਸਮ_ਨਾਂ MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[2], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, a[3], ਪੰਕਤੀ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[1], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_ਸਮੂਹ(sg, b[1], ਖਾਨਾ_ਲੱਦੋ_ਵਿੱਥ + i + 4);

            ਸਵੈਚਾਲਿਤ a = reinterpret_cast<ABType*>(recv_a);
            ਸਵੈਚਾਲਿਤ b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
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

#ਜੇ_ਅੰਤ // GGML_SYCL_DPCT_HELPER_HPP

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

#ifndef GGML_SYCL_DPCT_HELPER_HPP
#define GGML_SYCL_DPCT_HELPER_HPP

#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <iostream>
#ಅನ್ತರ್ಭಾವ <map>

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <sycl/half_type.hpp>
#ಅನ್ತರ್ಭಾವ <oneapi/mkl.hpp>

#ಯದಿ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <sys/mman.h>
#ನೋಚೇದ್ಯದಿ defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ಅನ್ತರ್ಭಾವ <windows.h>
#ಅನ್ಯಥಾ
#error "Only support Windows and Linux."
#endif

#ಯದಿ defined(__linux__)
#ಅನ್ತರ್ಭಾವ <unistd.h>
#ಅನ್ತರ್ಭಾವ <sys/syscall.h>
#endif
#ಯದಿ defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ಅನ್ತರ್ಭಾವ <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#ಯದಿ defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#ಅನ್ಯಥಾ
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#ಯದಿ defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#ಅನ್ಯಥಾ
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (ಶೂನ್ಯ)(x)

[[noreturn]] ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ _abort(ಧ್ರುವ ವರ್ಣ * str) {
    std::cerr << str << std::endl;
    std::abort();
}

ಅನ್ತಃಸ್ಥ std::string get_device_type_name(ಧ್ರುವ sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    ವಿಕಲ್ಪ (DeviceType) {
    ಪಕ್ಷ sycl::info::device_type::cpu:
        ಪ್ರತ್ಯಾಗಮ "cpu";
    ಪಕ್ಷ sycl::info::device_type::gpu:
        ಪ್ರತ್ಯಾಗಮ "gpu";
    ಪಕ್ಷ sycl::info::device_type::host:
        ಪ್ರತ್ಯಾಗಮ "host";
    ಪಕ್ಷ sycl::info::device_type::accelerator:
        ಪ್ರತ್ಯಾಗಮ "acc";
    ಸ್ವಭಾವ:
        ಪ್ರತ್ಯಾಗಮ "unknown";
    }
}

ಅನ್ತಃಸ್ಥ std::string get_device_backend_and_type(ಧ್ರುವ sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    ಪ್ರತ್ಯಾಗಮ device_type.str();
}

template <typename Ts> ಸಂರಚನಾ matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     ಮೌಲ್ಯ_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

ನಾಮಾಕಾಶ dpct
{
    ಪ್ರಕಾರನಿರ್ದೇಶ sycl::queue *queue_ptr;
    ಪ್ರಕಾರನಿರ್ದೇಶ sycl::event *event_ptr;
    ಪ್ರಕಾರನಿರ್ದೇಶ ವರ್ಣ *device_ptr;
    ಪ್ರಕಾರನಿರ್ದೇಶ uint8_t byte_t;
    ಪ್ರಕಾರನಿರ್ದೇಶ sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    ಅನ್ತಃಸ್ಥ auto exception_handler = [](sycl::exception_list exceptions)
    {
        ಪ್ರತಿ (std::exception_ptr ಧ್ರುವ &e : exceptions)
        {
            ಯತ್ನ
            {
                std::rethrow_exception(e);
            }
            ಗೃಹಾಣ (sycl::exception ಧ್ರುವ &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    ಗಣನಾ error_code
    {
        success = 0,
        default_error = 999
    };

    ಗಣನಾ memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    ಗಣನಾ memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    ಗಣನಾ ವರ್ಗ library_data_t : unsigned ವರ್ಣ
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
        library_data_t_size
    };

    template <typename T>
    ಸಂರಚನಾ DataType
    {
        using T2 = T;
    };
    template <typename T>
    ಸಂರಚನಾ DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    ಸ್ಥಿರ ಶೂನ್ಯ destroy_event(event_ptr event)
    {
        delete event;
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ unsigned ಪೂರ್ಣಾಙ್ಕ get_tid()
    {
#ಯದಿ defined(__linux__)
        ಪ್ರತ್ಯಾಗಮ syscall(SYS_gettid);
#ನೋಚೇದ್ಯದಿ defined(_WIN64)
        ಪ್ರತ್ಯಾಗಮ GetCurrentThreadId();
#ಅನ್ಯಥಾ
#error "Only support Windows and Linux."
#endif
    }

    ನಾಮಾಕಾಶ detail
    {
        ಸ್ಥಿರ ಶೂನ್ಯ get_version(ಧ್ರುವ sycl::device &dev, ಪೂರ್ಣಾಙ್ಕ &major, ಪೂರ್ಣಾಙ್ಕ &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            ಯಾವತ್ (i < ver.size()) {
              ಯದಿ (isdigit(ver[i]))
                ಭಙ್ಗ;
              i++;
            }
            major = std::stoi(&(ver[i]));
            ಯಾವತ್ (i < ver.size()) {
              ಯದಿ (ver[i] == '.')
                ಭಙ್ಗ;
              i++;
            }
            ಯದಿ (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } ಅನ್ಯಥಾ {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        ವರ್ಗ generic_error_type
        {
        ಸಾರ್ವಜನಿಕ:
            generic_error_type() = ಸ್ವಭಾವ;
            generic_error_type(T value) : value{value} {}
            operator T() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ value; }

        ಗುಪ್ತ:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    ವರ್ಗ dim3 {
        ಸಾರ್ವಜನಿಕ:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(ಧ್ರುವ sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ sycl::range<3>(z, y, x); }
    }; // namespace dim3

    ಅನ್ತಃಸ್ಥ dim3 operator*(ಧ್ರುವ dim3 &a, ಧ್ರುವ dim3 &b) {
    ಪ್ರತ್ಯಾಗಮ dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    ವರ್ಗ pitched_data
    {
    ಸಾರ್ವಜನಿಕ:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(ಶೂನ್ಯ *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        ಶೂನ್ಯ *get_data_ptr() { ಪ್ರತ್ಯಾಗಮ _data; }
        ಶೂನ್ಯ set_data_ptr(ಶೂನ್ಯ *data) { _data = data; }

        size_t get_pitch() { ಪ್ರತ್ಯಾಗಮ _pitch; }
        ಶೂನ್ಯ set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { ಪ್ರತ್ಯಾಗಮ _x; }
        ಶೂನ್ಯ set_x(size_t x) { _x = x; }

        size_t get_y() { ಪ್ರತ್ಯಾಗಮ _y; }
        ಶೂನ್ಯ set_y(size_t y) { _y = y; }

    ಗುಪ್ತ:
        ಶೂನ್ಯ *_data;
        size_t _pitch, _x, _y;
    };

    ವರ್ಗ device_info
    {
    ಸಾರ್ವಜನಿಕ:
        // get interface
        ಧ್ರುವ ವರ್ಣ *get_name() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _name; }
        ವರ್ಣ *get_name() { ಪ್ರತ್ಯಾಗಮ _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, ಪೂರ್ಣಾಙ್ಕ *>,
                                   ಪೂರ್ಣಾಙ್ಕ> = 0>
        auto get_max_work_item_sizes() ಧ್ರುವ
        {
            ಯದಿ constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ಪ್ರತ್ಯಾಗಮ sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            ಅನ್ಯಥಾ
            {
                ಪ್ರತ್ಯಾಗಮ _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, ಪೂರ್ಣಾಙ್ಕ *>,
                                   ಪೂರ್ಣಾಙ್ಕ> = 0>
        auto get_max_work_item_sizes()
        {
            ಯದಿ constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ಪ್ರತ್ಯಾಗಮ sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            ಅನ್ಯಥಾ
            {
                ಪ್ರತ್ಯಾಗಮ _max_work_item_sizes_i;
            }
        }
        ತಾರ್ಕಿಕ get_host_unified_memory() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _host_unified_memory; }
        ಪೂರ್ಣಾಙ್ಕ get_major_version() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _major; }
        ಪೂರ್ಣಾಙ್ಕ get_minor_version() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _minor; }
        ಪೂರ್ಣಾಙ್ಕ get_integrated() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _integrated; }
        ಪೂರ್ಣಾಙ್ಕ get_max_clock_frequency() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _frequency; }
        ಪೂರ್ಣಾಙ್ಕ get_max_compute_units() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _max_compute_units; }
        ಪೂರ್ಣಾಙ್ಕ get_max_work_group_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _max_work_group_size; }
        ಪೂರ್ಣಾಙ್ಕ get_max_sub_group_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _max_sub_group_size; }
        ಪೂರ್ಣಾಙ್ಕ get_max_work_ವಸ್ತುಗಳು_per_compute_unit() ಧ್ರುವ
        {
            ಪ್ರತ್ಯಾಗಮ max_work_ವಸ್ತುಗಳು_per_compute_unit;
        }
        ಪೂರ್ಣಾಙ್ಕ get_max_register_size_per_work_group() ಧ್ರುವ
        {
            ಪ್ರತ್ಯಾಗಮ _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, ಪೂರ್ಣಾಙ್ಕ *>,
                                   ಪೂರ್ಣಾಙ್ಕ> = 0>
        auto get_max_nd_range_size() ಧ್ರುವ
        {
            ಯದಿ constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                ಪ್ರತ್ಯಾಗಮ _max_nd_range_size;
            ಅನ್ಯಥಾ
                ಪ್ರತ್ಯಾಗಮ _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, ಪೂರ್ಣಾಙ್ಕ *>,
                                   ಪೂರ್ಣಾಙ್ಕ> = 0>
        auto get_max_nd_range_size()
        {
            ಯದಿ constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                ಪ್ರತ್ಯಾಗಮ _max_nd_range_size;
            ಅನ್ಯಥಾ
                ಪ್ರತ್ಯಾಗಮ _max_nd_range_size_i;
        }
        size_t get_global_mem_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _global_mem_size; }
        size_t get_local_mem_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _local_mem_size; }
        size_t get_max_mem_alloc_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned ಪೂರ್ಣಾಙ್ಕ get_memory_clock_rate() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned ಪೂರ್ಣಾಙ್ಕ get_memory_bus_width() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _memory_bus_width; }
        uint32_t get_device_id() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _device_id; }
        std::array<unsigned ವರ್ಣ, 16> get_uuid() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned ಪೂರ್ಣಾಙ್ಕ get_global_mem_cache_size() ಧ್ರುವ
        {
            ಪ್ರತ್ಯಾಗಮ _global_mem_cache_size;
        }

        // set interface
        ಶೂನ್ಯ set_name(ಧ್ರುವ ವರ್ಣ *name)
        {
            size_t length = strlen(name);
            ಯದಿ (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            ಅನ್ಯಥಾ
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        ಶೂನ್ಯ set_max_work_item_sizes(ಧ್ರುವ sycl::range<3> max_work_item_sizes)
        {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] ಶೂನ್ಯ
        set_max_work_item_sizes(ಧ್ರುವ sycl::id<3> max_work_item_sizes)
        {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        ಶೂನ್ಯ set_host_unified_memory(ತಾರ್ಕಿಕ host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        ಶೂನ್ಯ set_major_version(ಪೂರ್ಣಾಙ್ಕ major) { _major = major; }
        ಶೂನ್ಯ set_minor_version(ಪೂರ್ಣಾಙ್ಕ minor) { _minor = minor; }
        ಶೂನ್ಯ set_integrated(ಪೂರ್ಣಾಙ್ಕ integrated) { _integrated = integrated; }
        ಶೂನ್ಯ set_max_clock_frequency(ಪೂರ್ಣಾಙ್ಕ frequency) { _frequency = frequency; }
        ಶೂನ್ಯ set_max_compute_units(ಪೂರ್ಣಾಙ್ಕ max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        ಶೂನ್ಯ set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        ಶೂನ್ಯ set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        ಶೂನ್ಯ set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        ಶೂನ್ಯ set_max_work_group_size(ಪೂರ್ಣಾಙ್ಕ max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        ಶೂನ್ಯ set_max_sub_group_size(ಪೂರ್ಣಾಙ್ಕ max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        ಶೂನ್ಯ
        set_max_work_ವಸ್ತುಗಳು_per_compute_unit(ಪೂರ್ಣಾಙ್ಕ max_work_ವಸ್ತುಗಳು_per_compute_unit_2)
        {
            max_work_ವಸ್ತುಗಳು_per_compute_unit = max_work_ವಸ್ತುಗಳು_per_compute_unit_2;
        }
        ಶೂನ್ಯ set_max_nd_range_size(ಪೂರ್ಣಾಙ್ಕ max_nd_range_size[])
        {
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        ಶೂನ್ಯ set_memory_clock_rate(unsigned ಪೂರ್ಣಾಙ್ಕ memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        ಶೂನ್ಯ set_memory_bus_width(unsigned ಪೂರ್ಣಾಙ್ಕ memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        ಶೂನ್ಯ
        set_max_register_size_per_work_group(ಪೂರ್ಣಾಙ್ಕ max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        ಶೂನ್ಯ set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        ಶೂನ್ಯ set_uuid(std::array<unsigned ವರ್ಣ, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        ಶೂನ್ಯ set_global_mem_cache_size(unsigned ಪೂರ್ಣಾಙ್ಕ global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    ಗುಪ್ತ:
        ವರ್ಣ _name[256];
        ಪೂರ್ಣಾಙ್ಕ _max_work_item_sizes_i[3];
        ತಾರ್ಕಿಕ _host_unified_memory = ಅಸತ್ಯ;
        ಪೂರ್ಣಾಙ್ಕ _major;
        ಪೂರ್ಣಾಙ್ಕ _minor;
        ಪೂರ್ಣಾಙ್ಕ _integrated = 0;
        ಪೂರ್ಣಾಙ್ಕ _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned ಪೂರ್ಣಾಙ್ಕ _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned ಪೂರ್ಣಾಙ್ಕ _memory_bus_width = 64;
        unsigned ಪೂರ್ಣಾಙ್ಕ _global_mem_cache_size;
        ಪೂರ್ಣಾಙ್ಕ _max_compute_units;
        ಪೂರ್ಣಾಙ್ಕ _max_work_group_size;
        ಪೂರ್ಣಾಙ್ಕ _max_sub_group_size;
        ಪೂರ್ಣಾಙ್ಕ max_work_ವಸ್ತುಗಳು_per_compute_unit;
        ಪೂರ್ಣಾಙ್ಕ _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        ಪೂರ್ಣಾಙ್ಕ _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned ವರ್ಣ, 16> _uuid;
    };

    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ get_major_version(ಧ್ರುವ sycl::device &dev)
    {
        ಪೂರ್ಣಾಙ್ಕ major, minor;
        detail::get_version(dev, major, minor);
        ಪ್ರತ್ಯಾಗಮ major;
    }

    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ get_minor_version(ಧ್ರುವ sycl::device &dev)
    {
        ಪೂರ್ಣಾಙ್ಕ major, minor;
        detail::get_version(dev, major, minor);
        ಪ್ರತ್ಯಾಗಮ minor;
    }

    ಸ್ಥಿರ ಶೂನ್ಯ get_device_info(device_info &out, ಧ್ರುವ sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        ಪೂರ್ಣಾಙ್ಕ major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#ಯದಿ (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#ಅನ್ಯಥಾ
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_item_sizes<3>>());
#endif
        prop.set_host_unified_memory(dev.has(sycl::aspect::usm_host_allocations));

        prop.set_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.set_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.set_max_work_group_size(
            dev.get_info<sycl::info::device::max_work_group_size>());
        prop.set_global_mem_size(dev.get_info<sycl::info::device::global_mem_size>());
        prop.set_local_mem_size(dev.get_info<sycl::info::device::local_mem_size>());
        prop.set_max_mem_alloc_size(dev.get_info<sycl::info::device::max_mem_alloc_size>());

#ಯದಿ (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        ಯದಿ (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned ಪೂರ್ಣಾಙ್ಕ tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            ಯದಿ (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        ಯದಿ (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        ಯದಿ (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        ಯದಿ (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#ನೋಚೇದ್ಯದಿ defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate ಸ್ವಭಾವ value. \
        Use 64 bits as memory_bus_width ಸ್ವಭಾವ value.")
#ಅನ್ಯಥಾ
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate ಸ್ವಭಾವ value. \
        Use 64 bits as memory_bus_width ಸ್ವಭಾವ value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        ಪ್ರತಿ (ಧ್ರುವ auto &sub_group_size : sub_group_sizes)
        {
            ಯದಿ (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_ವಸ್ತುಗಳು_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        ಪೂರ್ಣಾಙ್ಕ max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    ವರ್ಗ device_ext : ಸಾರ್ವಜನಿಕ sycl::device {
      ಪ್ರಕಾರನಿರ್ದೇಶ std::mutex mutex_type;

     ಸಾರ್ವಜನಿಕ:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(ಧ್ರುವ sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      ಪೂರ್ಣಾಙ್ಕ is_native_atomic_supported() { ಪ್ರತ್ಯಾಗಮ 0; }
      ಪೂರ್ಣಾಙ್ಕ get_major_version() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ dpct::get_major_version(*ಏತತ್); }

      ಪೂರ್ಣಾಙ್ಕ get_minor_version() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ dpct::get_minor_version(*ಏತತ್); }

      ಪೂರ್ಣಾಙ್ಕ get_max_compute_units() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      ಪೂರ್ಣಾಙ್ಕ get_max_clock_frequency() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_max_clock_frequency();
      }

      ಪೂರ್ಣಾಙ್ಕ get_integrated() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ get_device_info().get_integrated(); }

      ಪೂರ್ಣಾಙ್ಕ get_max_sub_group_size() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_max_sub_group_size();
      }

      ಪೂರ್ಣಾಙ್ಕ get_max_register_size_per_work_group() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_max_register_size_per_work_group();
      }

      ಪೂರ್ಣಾಙ್ಕ get_max_work_group_size() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_max_work_group_size();
      }

      ಪೂರ್ಣಾಙ್ಕ get_mem_base_addr_align() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() ಧ್ರುವ {
        ಪ್ರತ್ಯಾಗಮ get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      ಶೂನ್ಯ get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        ಧ್ರುವ ವರ್ಣ *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#ಯದಿ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        ಯದಿ (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } ಅನ್ಯಥಾ {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#ಅನ್ಯಥಾ
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#ಯದಿ defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#ಅನ್ಯಥಾ
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      ಶೂನ್ಯ get_device_info(device_info &out) ಧ್ರುವ {
        dpct::get_device_info(out, *ಏತತ್);
      }

      device_info get_device_info() ಧ್ರುವ {
        device_info prop;
        dpct::get_device_info(prop, *ಏತತ್);
        ಪ್ರತ್ಯಾಗಮ prop;
      }

      ಶೂನ್ಯ reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { ಪ್ರತ್ಯಾಗಮ _q_in_order; }

      sycl::queue &out_of_order_queue() { ಪ್ರತ್ಯಾಗಮ _q_out_of_order; }

      sycl::queue &default_queue() { ಪ್ರತ್ಯಾಗಮ in_order_queue(); }

      ಶೂನ್ಯ queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        ಪ್ರತಿ (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(ತಾರ್ಕಿಕ enable_exception_handler = ಅಸತ್ಯ) {
        ಪ್ರತ್ಯಾಗಮ create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               ತಾರ್ಕಿಕ enable_exception_handler = ಅಸತ್ಯ) {
        ಪ್ರತ್ಯಾಗಮ create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(ತಾರ್ಕಿಕ enable_exception_handler = ಅಸತ್ಯ) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ಪ್ರತ್ಯಾಗಮ create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        ತಾರ್ಕಿಕ enable_exception_handler = ಅಸತ್ಯ) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ಪ್ರತ್ಯಾಗಮ create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          ತಾರ್ಕಿಕ enable_exception_handler = ಅಸತ್ಯ) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ಪ್ರತ್ಯಾಗಮ create_queue_impl(enable_exception_handler);
      }

      ಶೂನ್ಯ destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ಧ್ರುವ sycl::queue &q) -> ತಾರ್ಕಿಕ
                                    {
                                        ಪ್ರತ್ಯಾಗಮ q == queue;
                                    }),
                    _queues.end());
      }
      ಶೂನ್ಯ set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() ಧ್ರುವ {
        std::lock_guard<mutex_type> lock(m_mutex);
        ಪ್ರತ್ಯಾಗಮ _saved_queue;
      }

     ಗುಪ್ತ:
      ಶೂನ್ಯ clear_queues() { _queues.clear(); }

      ಶೂನ್ಯ init_queues() {
        _q_in_order =
            create_queue_impl(ಸತ್ಯ, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(ಸತ್ಯ);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <ವರ್ಗ... Properties>
      sycl::queue create_queue_impl(ತಾರ್ಕಿಕ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ಯದಿ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *ಏತತ್, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        ಪ್ರತ್ಯಾಗಮ _queues.back();
      }

      template <ವರ್ಗ... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    ತಾರ್ಕಿಕ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ಯದಿ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        ಪ್ರತ್ಯಾಗಮ _queues.back();
      }

      ಶೂನ್ಯ get_version(ಪೂರ್ಣಾಙ್ಕ &major, ಪೂರ್ಣಾಙ್ಕ &minor) ಧ್ರುವ {
        detail::get_version(*ಏತತ್, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    ವರ್ಗ dev_mgr
    {
    ಸಾರ್ವಜನಿಕ:
        device_ext &current_device()
        {
            unsigned ಪೂರ್ಣಾಙ್ಕ dev_id = current_device_id();
            check_id(dev_id);
            ಪ್ರತ್ಯಾಗಮ *_devs[dev_id];
        }
        device_ext &cpu_device() ಧ್ರುವ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            ಯದಿ (_cpu_device == -1)
            {
                ಕ್ಷಿಪ std::runtime_error("no valid cpu device");
            }
            ಅನ್ಯಥಾ
            {
                ಪ್ರತ್ಯಾಗಮ *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned ಪೂರ್ಣಾಙ್ಕ id) ಧ್ರುವ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            ಪ್ರತ್ಯಾಗಮ *_devs[id];
        }
        unsigned ಪೂರ್ಣಾಙ್ಕ current_device_id() ಧ್ರುವ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            ಯದಿ (it != _thread2dev_map.end())
                ಪ್ರತ್ಯಾಗಮ it->second;
            ಪ್ರತ್ಯಾಗಮ DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        ಶೂನ್ಯ select_device(unsigned ಪೂರ್ಣಾಙ್ಕ id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned ಪೂರ್ಣಾಙ್ಕ device_ಎಣಿಕೆ() { ಪ್ರತ್ಯಾಗಮ _devs.size(); }

        unsigned ಪೂರ್ಣಾಙ್ಕ get_device_id(ಧ್ರುವ sycl::device &dev)
        {
            unsigned ಪೂರ್ಣಾಙ್ಕ id = 0;
            ಪ್ರತಿ (auto &dev_item : _devs)
            {
                ಯದಿ (*dev_item == dev)
                {
                    ಪ್ರತ್ಯಾಗಮ id;
                }
                id++;
            }
            ಪ್ರತ್ಯಾಗಮ -1;
        }

        ಅನ್ತಃಸ್ಥ std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            ವರ್ಣ* env = getenv("ONEAPI_DEVICE_SELECTOR");
            ಯದಿ (env) {
                ಯದಿ (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                ಅನ್ಯಥಾ ಯದಿ (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                ಅನ್ಯಥಾ ಯದಿ (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                ಅನ್ಯಥಾ ಯದಿ (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                ಅನ್ಯಥಾ {
                    ಕ್ಷಿಪ std::runtime_error("invalid device filter: " + std::string(env));
                }
            } ಅನ್ಯಥಾ {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                ಯದಿ (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                ಅನ್ಯಥಾ ಯದಿ (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                ಅನ್ಯಥಾ ಯದಿ (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            ಪ್ರತಿ (ಧ್ರುವ auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](ಧ್ರುವ sycl::device& d) {
                    ಪ್ರತ್ಯಾಗಮ d.is_gpu();
                });

                ಯದಿ (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    ಅನುವರ್ತ;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_ಕೆಳಗಿನ_case;
                platform_name_ಕೆಳಗಿನ_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_ಕೆಳಗಿನ_case.begin(), ::tolower);

                ಯದಿ (platform_name_ಕೆಳಗಿನ_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    ಅನುವರ್ತ;
                }

                result = platform_name;
            }

            ಯದಿ (result.empty())
                ಕ್ಷಿಪ std::runtime_error("can not find preferred GPU platform");

            ಪ್ರತ್ಯಾಗಮ result;
        }

        template <ವರ್ಗ DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<ಪೂರ್ಣಾಙ್ಕ, DeviceSelector, ಧ್ರುವ sycl::device &>>
        select_device(ಧ್ರುವ DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned ಪೂರ್ಣಾಙ್ಕ selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ಸ್ಥಿರ dev_mgr &instance()
        {
            ಸ್ಥಿರ dev_mgr d_m;
            ಪ್ರತ್ಯಾಗಮ d_m;
        }
        dev_mgr(ಧ್ರುವ dev_mgr &) = delete;
        dev_mgr &operator=(ಧ್ರುವ dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    ಗುಪ್ತ:
        mutable std::recursive_mutex m_mutex;
        ಸ್ಥಿರ ತಾರ್ಕಿಕ compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            ಯದಿ(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) ಪ್ರತ್ಯಾಗಮ ಸತ್ಯ;
            ಯದಿ(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) ಪ್ರತ್ಯಾಗಮ ಅಸತ್ಯ;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            ಪ್ರತ್ಯಾಗಮ prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ convert_backend_index(std::string & backend) {
            ಯದಿ (backend == "ext_oneapi_level_zero:gpu") ಪ್ರತ್ಯಾಗಮ 0;
            ಯದಿ (backend == "opencl:gpu") ಪ್ರತ್ಯಾಗಮ 1;
            ಯದಿ (backend == "ext_oneapi_cuda:gpu") ಪ್ರತ್ಯಾಗಮ 2;
            ಯದಿ (backend == "ext_oneapi_hip:gpu") ಪ್ರತ್ಯಾಗಮ 3;
            ಯದಿ (backend == "opencl:cpu") ಪ್ರತ್ಯಾಗಮ 4;
            ಯದಿ (backend == "opencl:acc") ಪ್ರತ್ಯಾಗಮ 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ಸ್ಥಿರ ತಾರ್ಕಿಕ compare_backend(std::string &backend1, std::string &backend2) {
            ಪ್ರತ್ಯಾಗಮ convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            ಯದಿ (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            ಯಾವತ್ (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                ಯದಿ (platform_name.compare(preferred_platform_name) != 0) {
                    ಅನುವರ್ತ;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                ಪ್ರತಿ (ಧ್ರುವ auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> ಕೀಲಿಗಳು;
            ಪ್ರತಿ(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                ಕೀಲಿಗಳು.push_back(it->first);
            }
            std::sort(ಕೀಲಿಗಳು.begin(), ಕೀಲಿಗಳು.end(), compare_backend);

            ಪ್ರತಿ (auto &ಕೀಲಿ : ಕೀಲಿಗಳು) {
                std::vector<sycl::device> devs = backend_devices[ಕೀಲಿ];
                std::sort(devs.begin(), devs.end(), compare_dev);
                ಪ್ರತಿ (ಧ್ರುವ auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            ಪ್ರತಿ (auto &dev : sycl_all_devs)
            {
                ಯದಿ (dev == default_device)
                {
                    ಅನುವರ್ತ;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                ಯದಿ (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        ಶೂನ್ಯ check_id(unsigned ಪೂರ್ಣಾಙ್ಕ id) ಧ್ರುವ
        {
            ಯದಿ (id >= _devs.size())
            {
                ಕ್ಷಿಪ std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ಧ್ರುವ unsigned ಪೂರ್ಣಾಙ್ಕ DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned ಪೂರ್ಣಾಙ್ಕ, unsigned ಪೂರ್ಣಾಙ್ಕ> _thread2dev_map;
        ಪೂರ್ಣಾಙ್ಕ _cpu_device = -1;
    };

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ sycl::queue &get_default_queue()
    {
        ಪ್ರತ್ಯಾಗಮ dev_mgr::instance().current_device().default_queue();
    }

    ನಾಮಾಕಾಶ detail
    {
        ಗಣನಾ ವರ್ಗ pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ಸ್ಥಿರ pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              ಧ್ರುವ ಶೂನ್ಯ *ptr)
        {
            ವಿಕಲ್ಪ (sycl::get_pointer_type(ptr, q.get_context()))
            {
            ಪಕ್ಷ sycl::usm::alloc::unknown:
                ಪ್ರತ್ಯಾಗಮ pointer_access_attribute::host_only;
            ಪಕ್ಷ sycl::usm::alloc::device:
                ಪ್ರತ್ಯಾಗಮ pointer_access_attribute::device_only;
            ಪಕ್ಷ sycl::usm::alloc::shared:
            ಪಕ್ಷ sycl::usm::alloc::host:
                ಪ್ರತ್ಯಾಗಮ pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        ಅನ್ತಃಸ್ಥ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned ವರ್ಣ)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned ವರ್ಣ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            ಪ್ರತ್ಯಾಗಮ (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        ಅನ್ತಃಸ್ಥ constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned ವರ್ಣ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(ಪರಿಮಾಣ...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            ಪ್ರತ್ಯಾಗಮ get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        ವರ್ಗ mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#ಯದಿ defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ನೋಚೇದ್ಯದಿ defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    ರಿಕ್ತ,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#ಅನ್ಯಥಾ
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        ಸಾರ್ವಜನಿಕ:
            using buffer_id_t = ಪೂರ್ಣಾಙ್ಕ;

            ಸಂರಚನಾ allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#ಯದಿ defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#ನೋಚೇದ್ಯದಿ defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#ಅನ್ಯಥಾ
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(ಧ್ರುವ mem_mgr &) = delete;
            mem_mgr &operator=(ಧ್ರುವ mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            ಶೂನ್ಯ *mem_alloc(size_t size)
            {
                ಯದಿ (!size)
                    ಪ್ರತ್ಯಾಗಮ nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                ಯದಿ (next_free + size > mapped_address_space + mapped_region_size)
                {
                    ಕ್ಷಿಪ std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                ಶೂನ್ಯ *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                ಪ್ರತ್ಯಾಗಮ result;
            }

            /// Deallocate
            ಶೂನ್ಯ mem_free(ಧ್ರುವ ಶೂನ್ಯ *ptr)
            {
                ಯದಿ (!ptr)
                    ಪ್ರತ್ಯಾಗಮ;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(ಧ್ರುವ ಶೂನ್ಯ *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                ಪ್ರತ್ಯಾಗಮ it->second;
            }

            /// Check if the pointer represents device pointer or not.
            ತಾರ್ಕಿಕ is_device_ptr(ಧ್ರುವ ಶೂನ್ಯ *ptr) ಧ್ರುವ
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ಪ್ರತ್ಯಾಗಮ (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            ಸ್ಥಿರ mem_mgr &instance()
            {
                ಸ್ಥಿರ mem_mgr m;
                ಪ್ರತ್ಯಾಗಮ m;
            }

        ಗುಪ್ತ:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            ಧ್ರುವ size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            ಧ್ರುವ size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ಧ್ರುವ size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(ಧ್ರುವ ಶೂನ್ಯ *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ಧ್ರುವ byte_t *>(ptr)));
                ಯದಿ (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ಕ್ಷಿಪ std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ಧ್ರುವ allocation &alloc = it->second;
                ಯದಿ (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ಕ್ಷಿಪ std::runtime_error("invalid virtual pointer");
                }
                ಪ್ರತ್ಯಾಗಮ it;
            }
        };

        template <ವರ್ಗ T, memory_region Memory, size_t Dimension>
        ವರ್ಗ accessor;
        template <memory_region Memory, ವರ್ಗ T = byte_t>
        ವರ್ಗ memory_traits
        {
        ಸಾರ್ವಜನಿಕ:
            ಸ್ಥಿರ constexpr sycl::access::target target =
                sycl::access::target::device;
            ಸ್ಥಿರ constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::ಪಠ
                                     : sycl::access_mode::read_write;
            ಸ್ಥಿರ constexpr size_t type_size = ಪರಿಮಾಣ(T);
            using element_t =
                typename std::conditional<Memory == constant, ಧ್ರುವ T, T>::type;
            using ಮೌಲ್ಯ_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<ಮೌಲ್ಯ_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ *dpct_malloc(size_t size, sycl::queue &q)
        {
            ಪ್ರತ್ಯಾಗಮ sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            ಪ್ರತ್ಯಾಗಮ dpct_malloc(pitch * y * z, q);
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
        template <typename ಮೌಲ್ಯ_T>
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ sycl::event dpct_memset(sycl::queue &q, ಶೂನ್ಯ *dev_ptr,
                                              ಮೌಲ್ಯ_T value, size_t size)
        {
            ಪ್ರತ್ಯಾಗಮ q.fill(dev_ptr, value, size);
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
        template <typename ಮೌಲ್ಯ_T>
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, ಮೌಲ್ಯ_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned ವರ್ಣ *data_surface = (unsigned ವರ್ಣ *)data.get_data_ptr();
            ಪ್ರತಿ (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned ವರ್ಣ *data_ptr = data_surface;
                ಪ್ರತಿ (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            ಪ್ರತ್ಯಾಗಮ event_list;
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
        template <typename ಮೌಲ್ಯ_T>
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, ಶೂನ್ಯ *ptr, size_t pitch, ಮೌಲ್ಯ_T val, size_t x,
                    size_t y)
        {
            ಪ್ರತ್ಯಾಗಮ dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ಸ್ಥಿರ memcpy_direction deduce_memcpy_direction(sycl::queue &q, ಶೂನ್ಯ *to_ptr,
                                                        ಧ್ರುವ ಶೂನ್ಯ *from_ptr,
                                                        memcpy_direction dir)
        {
            ವಿಕಲ್ಪ (dir)
            {
            ಪಕ್ಷ memcpy_direction::host_to_host:
            ಪಕ್ಷ memcpy_direction::host_to_device:
            ಪಕ್ಷ memcpy_direction::device_to_host:
            ಪಕ್ಷ memcpy_direction::device_to_device:
                ಪ್ರತ್ಯಾಗಮ dir;
            ಪಕ್ಷ memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ಸ್ಥಿರ ಧ್ರುವ memcpy_direction
                    direction_table[static_cast<unsigned>(pointer_access_attribute::end)]
                                   [static_cast<unsigned>(pointer_access_attribute::end)] =
                                       {{memcpy_direction::host_to_host,
                                         memcpy_direction::device_to_host,
                                         memcpy_direction::host_to_host},
                                        {memcpy_direction::host_to_device,
                                         memcpy_direction::device_to_device,
                                         memcpy_direction::device_to_device},
                                        {memcpy_direction::host_to_host,
                                         memcpy_direction::device_to_device,
                                         memcpy_direction::device_to_device}};
                ಪ್ರತ್ಯಾಗಮ direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            ಸ್ವಭಾವ:
                ಕ್ಷಿಪ std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ಸ್ಥಿರ sycl::event
        dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr, size_t size,
                    memcpy_direction direction,
                    ಧ್ರುವ std::vector<sycl::event> &dep_events = {})
        {
            ಯದಿ (!size)
                ಪ್ರತ್ಯಾಗಮ sycl::event{};
            ಪ್ರತ್ಯಾಗಮ q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            ಪ್ರತ್ಯಾಗಮ slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            ಪ್ರತ್ಯಾಗಮ slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ಧ್ರುವ std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            ವರ್ಗ host_buffer
            {
                ಶೂನ್ಯ *_buf;
                size_t _size;
                sycl::queue &_q;
                ಧ್ರುವ std::vector<sycl::event> &_deps; // free operation depends

            ಸಾರ್ವಜನಿಕ:
                host_buffer(size_t size, sycl::queue &q,
                            ಧ್ರುವ std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                ಶೂನ್ಯ *get_ptr() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _buf; }
                size_t get_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _size; }
                ~host_buffer()
                {
                    ಯದಿ (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::ಮುಕ್ತ(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned ವರ್ಣ *to_surface =
                (unsigned ವರ್ಣ *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            ಧ್ರುವ unsigned ವರ್ಣ *from_surface =
                (ಧ್ರುವ unsigned ವರ್ಣ *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            ಯದಿ (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                ಪ್ರತ್ಯಾಗಮ {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            ವಿಕಲ್ಪ (direction)
            {
            ಪಕ್ಷ host_to_host:
                ಪ್ರತಿ (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned ವರ್ಣ *to_ptr = to_surface;
                    ಧ್ರುವ unsigned ವರ್ಣ *from_ptr = from_surface;
                    ಯದಿ (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    ಅನ್ಯಥಾ
                    {
                        ಪ್ರತಿ (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            to_ptr += to_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    to_surface += to_slice;
                    from_surface += from_slice;
                }
                ಭಙ್ಗ;
            ಪಕ್ಷ host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                ಯದಿ (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                ಅನ್ಯಥಾ
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, to_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                                        device_to_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                                 buf.get_size(), host_to_device,
                                                 host_events));
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ device_to_host:
            {
                host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_to_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_size(),
                                                         device_to_host, dep_events)});
                ಭಙ್ಗ;
            }
            ಪಕ್ಷ device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<ವರ್ಗ dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
                ಕ್ಷಿಪ std::runtime_error("dpct_memcpy: invalid direction value");
            }
            ಪ್ರತ್ಯಾಗಮ event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            ಪ್ರತ್ಯಾಗಮ dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            ಪ್ರತ್ಯಾಗಮ dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        ನಾಮಾಕಾಶ deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            ವರ್ಗ usm_allocator
            {
            ಗುಪ್ತ:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            ಸಾರ್ವಜನಿಕ:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    ಧ್ರುವ typename std::allocator_traits<Alloc>::value_type &;
                using difference_type =
                    typename std::allocator_traits<Alloc>::difference_type;
                using size_type = typename std::allocator_traits<Alloc>::size_type;
                using propagate_on_container_copy_assignment = typename std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                using propagate_on_container_move_assignment = typename std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                using propagate_on_container_swap =
                    typename std::allocator_traits<Alloc>::propagate_on_container_swap;
                using is_always_equal =
                    typename std::allocator_traits<Alloc>::is_always_equal;

                template <typename U>
                ಸಂರಚನಾ rebind
                {
                    ಪ್ರಕಾರನಿರ್ದೇಶ usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(ಧ್ರುವ usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { ಪ್ರತ್ಯಾಗಮ &r; }
                const_pointer address(const_reference r) { ಪ್ರತ್ಯಾಗಮ &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    ಪ್ರತ್ಯಾಗಮ std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                ಶೂನ್ಯ deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ಧ್ರುವ
                {
                    ಪ್ರತ್ಯಾಗಮ std::allocator_traits<Alloc>::max_size(_impl);
                }
                ತಾರ್ಕಿಕ operator==(ಧ್ರುವ usm_allocator &other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _impl == other._impl; }
                ತಾರ್ಕಿಕ operator!=(ಧ್ರುವ usm_allocator &other) ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _impl != other._impl; }
            };

        } // namespace deprecated

        ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dpct_free(ಶೂನ್ಯ *ptr,
                              ಧ್ರುವ sycl::queue &q)
        {
            ಯದಿ (ptr)
            {
                sycl::ಮುಕ್ತ(ptr, q.get_context());
            }
        }

        template <typename T>
        ಅನ್ತಃಸ್ಥ auto get_memory(ಧ್ರುವ ಶೂನ್ಯ *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<ಶೂನ್ಯ *>(x));
            ಪ್ರತ್ಯಾಗಮ new_x;
        }

        template <typename T>
        ಅನ್ತಃಸ್ಥ typename DataType<T>::T2 get_ಮೌಲ್ಯ(ಧ್ರುವ T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            ಯದಿ (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (ಶೂನ್ಯ *)&s_h, (ಧ್ರುವ ಶೂನ್ಯ *)s, ಪರಿಮಾಣ(T), device_to_host)
                    .wait();
            ಅನ್ಯಥಾ
                s_h = *reinterpret_cast<ಧ್ರುವ Ty *>(s);
            ಪ್ರತ್ಯಾಗಮ s_h;
        }

    } // namespace detail

    template <typename T>
    ಅನ್ತಃಸ್ಥ auto get_ಮೌಲ್ಯ(ಧ್ರುವ T *s, sycl::queue &q)
    {
        ಪ್ರತ್ಯಾಗಮ detail::get_ಮೌಲ್ಯ(s, q);
    }

    ನಾಮಾಕಾಶ detail
    {
    template <ವರ್ಗ Ta, ವರ್ಗ Tb, ವರ್ಗ Tc, ವರ್ಗ Ts>
    ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಙ್ಕ m,
                          ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಶೂನ್ಯ * alpha, ಧ್ರುವ ಶೂನ್ಯ * a, ಪೂರ್ಣಾಙ್ಕ lda, ಧ್ರುವ ಶೂನ್ಯ * b, ಪೂರ್ಣಾಙ್ಕ ldb,
                          ಧ್ರುವ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, ಪೂರ್ಣಾಙ್ಕ ldc) {
        Ts   alpha_ಮೌಲ್ಯ_2 = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ Ts *>(alpha), q);
        Ts   beta_ಮೌಲ್ಯ  = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ Ts *>(beta), q);
        auto data_a      = get_memory<ಧ್ರುವ Ta>(a);
        auto data_b      = get_memory<ಧ್ರುವ Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_ಮೌಲ್ಯ_2, data_a,
                                               lda, data_b, ldb, beta_ಮೌಲ್ಯ, data_c, ldc);
    }

        template <typename VecT, ವರ್ಗ ದ್ವಿಮಾನ_Operation, ವರ್ಗ = ಶೂನ್ಯ>
        ವರ್ಗ vectorized_ದ್ವಿಮಾನ
        {
        ಸಾರ್ವಜನಿಕ:
            ಅನ್ತಃಸ್ಥ VecT operator()(VecT a, VecT b, ಧ್ರುವ ದ್ವಿಮಾನ_Operation ದ್ವಿಮಾನ_op)
            {
                VecT v4;
                ಪ್ರತಿ (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ದ್ವಿಮಾನ_op(a[i], b[i]);
                }
                ಪ್ರತ್ಯಾಗಮ v4;
            }
        };

        template <typename VecT, ವರ್ಗ ದ್ವಿಮಾನ_Operation>
        ವರ್ಗ vectorized_ದ್ವಿಮಾನ<
            VecT, ದ್ವಿಮಾನ_Operation,
            std::void_t<std::invoke_result_t<ದ್ವಿಮಾನ_Operation, VecT, VecT>>>
        {
        ಸಾರ್ವಜನಿಕ:
            ಅನ್ತಃಸ್ಥ VecT operator()(VecT a, VecT b, ಧ್ರುವ ದ್ವಿಮಾನ_Operation ದ್ವಿಮಾನ_op)
            {
                ಪ್ರತ್ಯಾಗಮ ದ್ವಿಮಾನ_op(a, b).template as<VecT>();
            }
        };

        template <ವರ್ಗ Ta, ವರ್ಗ Tb, ವರ್ಗ Tc, ವರ್ಗ Ts>
        ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ಪೂರ್ಣಾಙ್ಕ m, ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಶೂನ್ಯ * alpha, ಧ್ರುವ ಶೂನ್ಯ ** a, ಪೂರ್ಣಾಙ್ಕ lda, ಧ್ರುವ ಶೂನ್ಯ ** b,
                                    ಪೂರ್ಣಾಙ್ಕ ldb, ಧ್ರುವ ಶೂನ್ಯ * beta, ಶೂನ್ಯ ** c, ಪೂರ್ಣಾಙ್ಕ ldc, ಪೂರ್ಣಾಙ್ಕ batch_size,
                                    matrix_info_t<ಭಿನ್ನ> * matrix_info) {
            Ts alpha_ಮೌಲ್ಯ_2 = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ Ts *>(alpha), q);
            Ts beta_ಮೌಲ್ಯ = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->ಮೌಲ್ಯ_info[0] = alpha_ಮೌಲ್ಯ_2;
            matrix_info->ಮೌಲ್ಯ_info[1] = beta_ಮೌಲ್ಯ;
            matrix_info->size_info[0] = m;
            matrix_info->size_info[1] = n;
            matrix_info->size_info[2] = k;
            matrix_info->ld_info[0] = lda;
            matrix_info->ld_info[1] = ldb;
            matrix_info->ld_info[2] = ldc;
            matrix_info->groupsize_info = batch_size;

            sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
                q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
                matrix_info->size_info, matrix_info->size_info + 1, matrix_info->size_info + 2,
                reinterpret_cast<Ts *>(matrix_info->ಮೌಲ್ಯ_info), reinterpret_cast<ಧ್ರುವ Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<ಧ್ರುವ Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->ಮೌಲ್ಯ_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <ವರ್ಗ Ta, ವರ್ಗ Tb, ವರ್ಗ Tc, ವರ್ಗ Ts>
        ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ಪೂರ್ಣಾಙ್ಕ m, ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಶೂನ್ಯ * alpha, ಧ್ರುವ ಶೂನ್ಯ * a, ಪೂರ್ಣಾಙ್ಕ lda,
                                    long long ಪೂರ್ಣಾಙ್ಕ stride_a, ಧ್ರುವ ಶೂನ್ಯ * b, ಪೂರ್ಣಾಙ್ಕ ldb, long long ಪೂರ್ಣಾಙ್ಕ stride_b,
                                    ಧ್ರುವ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, ಪೂರ್ಣಾಙ್ಕ ldc, long long ಪೂರ್ಣಾಙ್ಕ stride_c, ಪೂರ್ಣಾಙ್ಕ batch_size) {
            Ts alpha_ಮೌಲ್ಯ_2 = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ Ts *>(alpha), q);
            Ts beta_ಮೌಲ್ಯ = dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ Ts *>(beta), q);
            auto data_a = get_memory<ಧ್ರುವ Ta>(a);
            auto data_b = get_memory<ಧ್ರುವ Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_ಮೌಲ್ಯ_2,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_ಮೌಲ್ಯ,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, ವರ್ಗ ದ್ವಿಮಾನ_Operation>
    ಅನ್ತಃಸ್ಥ unsigned vectorized_ದ್ವಿಮಾನ(unsigned a, unsigned b,
                                      ಧ್ರುವ ದ್ವಿಮಾನ_Operation ದ್ವಿಮಾನ_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_ದ್ವಿಮಾನ<VecT, ದ್ವಿಮಾನ_Operation>()(v2, v3, ದ್ವಿಮಾನ_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        ಪ್ರತ್ಯಾಗಮ v0;
    }

    ಸ್ಥಿರ ಶೂನ್ಯ async_dpct_memcpy(ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ unsigned ಪೂರ್ಣಾಙ್ಕ select_device(unsigned ಪೂರ್ಣಾಙ್ಕ id)
    {
        dev_mgr::instance().select_device(id);
        ಪ್ರತ್ಯಾಗಮ id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned ಪೂರ್ಣಾಙ್ಕ mask,
                               unsigned ಪೂರ್ಣಾಙ್ಕ logical_sub_group_size = 32)
    {
        unsigned ಪೂರ್ಣಾಙ್ಕ id = g.get_local_linear_id();
        unsigned ಪೂರ್ಣಾಙ್ಕ start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned ಪೂರ್ಣಾಙ್ಕ target_offset = (id % logical_sub_group_size) ^ mask;
        ಪ್ರತ್ಯಾಗಮ sycl::select_from_group(g, x,
                                       target_offset < logical_sub_group_size
                                           ? start_index + target_offset
                                           : id);
    }

    template <typename T1, typename T2>
    using dot_product_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    template <typename T>
    sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
      ಪ್ರತ್ಯಾಗಮ sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    ಅನ್ತಃಸ್ಥ auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      ಪ್ರತ್ಯಾಗಮ res;
    }

    ಸಂರಚನಾ sub_sat
    {
        template <typename T>
        auto operator()(ಧ್ರುವ T x, ಧ್ರುವ T y) ಧ್ರುವ
        {
            ಪ್ರತ್ಯಾಗಮ sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    ಅನ್ತಃಸ್ಥ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        ಪ್ರತ್ಯಾಗಮ v0;
    }

    ಅನ್ತಃಸ್ಥ ಭಿನ್ನ pow(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b) { ಪ್ರತ್ಯಾಗಮ sycl::pown(a, b); }
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ pow(ಧ್ರುವ ದ್ವಿಗುಣ a, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b) { ಪ್ರತ್ಯಾಗಮ sycl::pown(a, b); }
    ಅನ್ತಃಸ್ಥ ಭಿನ್ನ pow(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) { ಪ್ರತ್ಯಾಗಮ sycl::pow(a, b); }
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ pow(ಧ್ರುವ ದ್ವಿಗುಣ a, ಧ್ರುವ ದ್ವಿಗುಣ b) { ಪ್ರತ್ಯಾಗಮ sycl::pow(a, b); }
    template <typename T, typename U>
    ಅನ್ತಃಸ್ಥ typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(ಧ್ರುವ T a, ಧ್ರುವ U b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    ಅನ್ತಃಸ್ಥ typename std::enable_if_t<!std::is_floating_point_v<T>, ದ್ವಿಗುಣ>
    pow(ಧ್ರುವ T a, ಧ್ರುವ U b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::pow(static_cast<ದ್ವಿಗುಣ>(a), static_cast<ದ್ವಿಗುಣ>(b));
    }

    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ min(ಧ್ರುವ ದ್ವಿಗುಣ a, ಧ್ರುವ ಭಿನ್ನ b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::fmin(a, static_cast<ದ್ವಿಗುಣ>(b));
    }
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ min(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ದ್ವಿಗುಣ b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::fmin(static_cast<ದ್ವಿಗುಣ>(a), b);
    }
    ಅನ್ತಃಸ್ಥ ಭಿನ್ನ min(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) { ಪ್ರತ್ಯಾಗಮ sycl::fmin(a, b); }
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ min(ಧ್ರುವ ದ್ವಿಗುಣ a, ಧ್ರುವ ದ್ವಿಗುಣ b) { ಪ್ರತ್ಯಾಗಮ sycl::fmin(a, b); }
    ಅನ್ತಃಸ್ಥ std::uint32_t min(ಧ್ರುವ std::uint32_t a, ಧ್ರುವ std::int32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, static_cast<std::uint32_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint32_t min(ಧ್ರುವ std::int32_t a, ಧ್ರುವ std::uint32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(static_cast<std::uint32_t>(a), b);
    }
    ಅನ್ತಃಸ್ಥ std::int32_t min(ಧ್ರುವ std::int32_t a, ಧ್ರುವ std::int32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint32_t min(ಧ್ರುವ std::uint32_t a, ಧ್ರುವ std::uint32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::int64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::int64_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ಅನ್ತಃಸ್ಥ std::int64_t min(ಧ್ರುವ std::int64_t a, ಧ್ರುವ std::int64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::int32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::int32_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::uint32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t min(ಧ್ರುವ std::uint32_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ max(ಧ್ರುವ ದ್ವಿಗುಣ a, ಧ್ರುವ ಭಿನ್ನ b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::fmax(a, static_cast<ದ್ವಿಗುಣ>(b));
    }
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ max(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ದ್ವಿಗುಣ b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::fmax(static_cast<ದ್ವಿಗುಣ>(a), b);
    }
    ಅನ್ತಃಸ್ಥ ಭಿನ್ನ max(ಧ್ರುವ ಭಿನ್ನ a, ಧ್ರುವ ಭಿನ್ನ b) { ಪ್ರತ್ಯಾಗಮ sycl::fmax(a, b); }
    ಅನ್ತಃಸ್ಥ ದ್ವಿಗುಣ max(ಧ್ರುವ ದ್ವಿಗುಣ a, ಧ್ರುವ ದ್ವಿಗುಣ b) { ಪ್ರತ್ಯಾಗಮ sycl::fmax(a, b); }
    ಅನ್ತಃಸ್ಥ std::uint32_t max(ಧ್ರುವ std::uint32_t a, ಧ್ರುವ std::int32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, static_cast<std::uint32_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint32_t max(ಧ್ರುವ std::int32_t a, ಧ್ರುವ std::uint32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(static_cast<std::uint32_t>(a), b);
    }
    ಅನ್ತಃಸ್ಥ std::int32_t max(ಧ್ರುವ std::int32_t a, ಧ್ರುವ std::int32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint32_t max(ಧ್ರುವ std::uint32_t a, ಧ್ರುವ std::uint32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::int64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::int64_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ಅನ್ತಃಸ್ಥ std::int64_t max(ಧ್ರುವ std::int64_t a, ಧ್ರುವ std::int64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::int32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::int32_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::uint64_t a, ಧ್ರುವ std::uint32_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ಅನ್ತಃಸ್ಥ std::uint64_t max(ಧ್ರುವ std::uint32_t a, ಧ್ರುವ std::uint64_t b)
    {
        ಪ್ರತ್ಯಾಗಮ sycl::max(static_cast<std::uint64_t>(a), b);
    }

    ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ
    has_capability_or_fail(ಧ್ರುವ sycl::device &dev,
                           ಧ್ರುವ std::initializer_list<sycl::aspect> &props)
    {
        ಪ್ರತಿ (ಧ್ರುವ auto &it : props)
        {
            ಯದಿ (dev.has(it))
                ಅನುವರ್ತ;
            ವಿಕಲ್ಪ (it)
            {
            ಪಕ್ಷ sycl::aspect::fp64:
                ಕ್ಷಿಪ std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ಭಙ್ಗ;
            ಪಕ್ಷ sycl::aspect::fp16:
                ಕ್ಷಿಪ std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ಭಙ್ಗ;
            ಸ್ವಭಾವ:
#define __SYCL_ASPECT(ASPECT, ID) \
    ಪಕ್ಷ sycl::aspect::ASPECT:    \
        ಪ್ರತ್ಯಾಗಮ #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect Aspect_ಅಂಕೆ) -> std::string
                {
                    ವಿಕಲ್ಪ (Aspect_ಅಂಕೆ)
                    {
#ಅನ್ತರ್ಭಾವ <sycl/info/aspects.def>
#ಅನ್ತರ್ಭಾವ <sycl/info/aspects_deprecated.def>
                    ಸ್ವಭಾವ:
                        ಪ್ರತ್ಯಾಗಮ "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                ಕ್ಷಿಪ std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            ಭಙ್ಗ;
        }
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ unsigned ಪೂರ್ಣಾಙ್ಕ get_current_device_id()
    {
        ಪ್ರತ್ಯಾಗಮ dev_mgr::instance().current_device_id();
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ device_ext &get_current_device()
    {
        ಪ್ರತ್ಯಾಗಮ dev_mgr::instance().current_device();
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ device_ext &get_device(unsigned ಪೂರ್ಣಾಙ್ಕ id)
    {
        ಪ್ರತ್ಯಾಗಮ dev_mgr::instance().get_device(id);
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ sycl::queue &get_in_order_queue()
    {
        ಪ್ರತ್ಯಾಗಮ dev_mgr::instance().current_device().in_order_queue();
    }

    ಸ್ಥಿರ sycl::event
    dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr, size_t size,
                memcpy_direction direction,
                ಧ್ರುವ std::vector<sycl::event> &dep_events = {})
    {
        ಯದಿ (!size)
            ಪ್ರತ್ಯಾಗಮ sycl::event{};
        ಪ್ರತ್ಯಾಗಮ q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        ಪ್ರತ್ಯಾಗಮ slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        ಪ್ರತ್ಯಾಗಮ slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ಧ್ರುವ std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        ವರ್ಗ host_buffer
        {
            ಶೂನ್ಯ *_buf;
            size_t _size;
            sycl::queue &_q;
            ಧ್ರುವ std::vector<sycl::event> &_deps; // free operation depends

        ಸಾರ್ವಜನಿಕ:
            host_buffer(size_t size, sycl::queue &q,
                        ಧ್ರುವ std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            ಶೂನ್ಯ *get_ptr() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _buf; }
            size_t get_size() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _size; }
            ~host_buffer()
            {
                ಯದಿ (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::ಮುಕ್ತ(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned ವರ್ಣ *to_surface =
            (unsigned ವರ್ಣ *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        ಧ್ರುವ unsigned ವರ್ಣ *from_surface =
            (ಧ್ರುವ unsigned ವರ್ಣ *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        ಯದಿ (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            ಪ್ರತ್ಯಾಗಮ {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        ವಿಕಲ್ಪ (direction)
        {
        ಪಕ್ಷ host_to_host:
            ಪ್ರತಿ (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned ವರ್ಣ *to_ptr = to_surface;
                ಧ್ರುವ unsigned ವರ್ಣ *from_ptr = from_surface;
                ಯದಿ (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                ಅನ್ಯಥಾ
                {
                    ಪ್ರತಿ (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        to_ptr += to_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                to_surface += to_slice;
                from_surface += from_slice;
            }
            ಭಙ್ಗ;
        ಪಕ್ಷ host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            ಯದಿ (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            ಅನ್ಯಥಾ
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, to_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_size(),
                                    device_to_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                             buf.get_size(), host_to_device,
                                             host_events));
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ device_to_host:
        {
            host_buffer buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_to_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_size(),
                                                     device_to_host, dep_events)});
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<ವರ್ಗ dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            ಕ್ಷಿಪ std::runtime_error("dpct_memcpy: invalid direction value");
        }
        ಪ್ರತ್ಯಾಗಮ event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        ಪ್ರತ್ಯಾಗಮ dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ಶೂನ್ಯ *to_ptr, ಧ್ರುವ ಶೂನ್ಯ *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        ಪ್ರತ್ಯಾಗಮ dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಙ್ಕ m, ಪೂರ್ಣಾಙ್ಕ n,
                     ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಶೂನ್ಯ * alpha, ಧ್ರುವ ಶೂನ್ಯ * a, library_data_t a_type, ಪೂರ್ಣಾಙ್ಕ lda, ಧ್ರುವ ಶೂನ್ಯ * b,
                     library_data_t b_type, ಪೂರ್ಣಾಙ್ಕ ldb, ಧ್ರುವ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, library_data_t c_type, ಪೂರ್ಣಾಙ್ಕ ldc,
                     library_data_t scaling_type) {
        ಯದಿ (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        ಅನ್ಯಥಾ ಯದಿ (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t ಕೀಲಿ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ವಿಕಲ್ಪ (ಕೀಲಿ)
        {
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<ಭಿನ್ನ>, std::complex<ಭಿನ್ನ>,
                              std::complex<ಭಿನ್ನ>, std::complex<ಭಿನ್ನ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>,
                              std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
#ifdef __INTEL_MKL__
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ಭಿನ್ನ alpha_ಮೌಲ್ಯ_2 =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(alpha), q);
            ಭಿನ್ನ beta_ಮೌಲ್ಯ =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(beta), q);
            sycl::half alpha_half(alpha_ಮೌಲ್ಯ_2);
            sycl::half beta_half(beta_ಮೌಲ್ಯ);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ಭಿನ್ನ alpha_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ std::int32_t *>(alpha), q);
            ಭಿನ್ನ beta_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            ಭಙ್ಗ;
        }
#endif // __INTEL_MKL__
        ಸ್ವಭಾವ:
            ಕ್ಷಿಪ std::runtime_error("the combination of data type is unsupported");
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
    ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಙ್ಕ m,
                           ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಶೂನ್ಯ * alpha, ಧ್ರುವ ಶೂನ್ಯ * a[], library_data_t a_type, ಪೂರ್ಣಾಙ್ಕ lda,
                           ಧ್ರುವ ಶೂನ್ಯ * b[], library_data_t b_type, ಪೂರ್ಣಾಙ್ಕ ldb, ಧ್ರುವ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c[],
                           library_data_t c_type, ಪೂರ್ಣಾಙ್ಕ ldc, ಪೂರ್ಣಾಙ್ಕ batch_size, library_data_t scaling_type,
                           matrix_info_t<ಭಿನ್ನ> * matrix_info) {
        std::uint64_t ಕೀಲಿ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ವಿಕಲ್ಪ (ಕೀಲಿ)
        {
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
#ifdef __INTEL_MKL__
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
#endif
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ಭಿನ್ನ alpha_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ std::int32_t *>(alpha), q);
            ಭಿನ್ನ beta_float =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ಭಿನ್ನ alpha_ಮೌಲ್ಯ_2 =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(alpha), q);
            ಭಿನ್ನ beta_ಮೌಲ್ಯ =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(beta), q);
            sycl::half alpha_half(alpha_ಮೌಲ್ಯ_2);
            sycl::half beta_half(beta_ಮೌಲ್ಯ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            ಕ್ಷಿಪ std::runtime_error("the combination of data type is unsupported");
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
    ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ಪೂರ್ಣಾಙ್ಕ m,
                           ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ k, ಧ್ರುವ ಶೂನ್ಯ * alpha, ಧ್ರುವ ಶೂನ್ಯ * a, library_data_t a_type, ಪೂರ್ಣಾಙ್ಕ lda,
                           long long ಪೂರ್ಣಾಙ್ಕ stride_a, ಧ್ರುವ ಶೂನ್ಯ * b, library_data_t b_type, ಪೂರ್ಣಾಙ್ಕ ldb,
                           long long ಪೂರ್ಣಾಙ್ಕ stride_b, ಧ್ರುವ ಶೂನ್ಯ * beta, ಶೂನ್ಯ * c, library_data_t c_type, ಪೂರ್ಣಾಙ್ಕ ldc,
                           long long ಪೂರ್ಣಾಙ್ಕ stride_c, ಪೂರ್ಣಾಙ್ಕ batch_size, library_data_t scaling_type) {
        ಯದಿ (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        ಅನ್ಯಥಾ ಯದಿ (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t ಕೀಲಿ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ವಿಕಲ್ಪ (ಕೀಲಿ)
        {
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ, ದ್ವಿಗುಣ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<ಭಿನ್ನ>, std::complex<ಭಿನ್ನ>,
                                    std::complex<ಭಿನ್ನ>, std::complex<ಭಿನ್ನ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>,
                                    std::complex<ದ್ವಿಗುಣ>, std::complex<ದ್ವಿಗುಣ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
#ifdef __INTEL_MKL__
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            ಭಙ್ಗ;
        }
#endif
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ಭಿನ್ನ, ಭಿನ್ನ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಪಕ್ಷ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ಭಿನ್ನ alpha_ಮೌಲ್ಯ_2 =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(alpha), q);
            ಭಿನ್ನ beta_ಮೌಲ್ಯ =
                dpct::get_ಮೌಲ್ಯ(reinterpret_cast<ಧ್ರುವ ಭಿನ್ನ *>(beta), q);
            sycl::half alpha_half(alpha_ಮೌಲ್ಯ_2);
            sycl::half beta_half(beta_ಮೌಲ್ಯ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            ಭಙ್ಗ;
        }
        ಸ್ವಭಾವ:
            ಕ್ಷಿಪ std::runtime_error("the combination of data type is unsupported");
        }
    }

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ
    async_dpct_memcpy(ಶೂನ್ಯ *to_ptr, size_t to_pitch, ಧ್ರುವ ಶೂನ್ಯ *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<ಸಂರಚನಾ err0_tag, ಪೂರ್ಣಾಙ್ಕ>;
    using err1 = detail::generic_error_type<ಸಂರಚನಾ err1_tag, ಪೂರ್ಣಾಙ್ಕ>;

    ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dpct_free(ಶೂನ್ಯ *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <ವರ್ಗ T, memory_region Memory, size_t Dimension> ವರ್ಗ accessor;
    template <ವರ್ಗ T, memory_region Memory> ವರ್ಗ accessor<T, Memory, 3> {
    ಸಾರ್ವಜನಿಕ:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, ಧ್ರುವ sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ಧ್ರುವ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ಧ್ರುವ accessor_t &acc, ಧ್ರುವ sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) ಧ್ರುವ {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            ಪ್ರತ್ಯಾಗಮ accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _data; }

    ಗುಪ್ತ:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <ವರ್ಗ T, memory_region Memory> ವರ್ಗ accessor<T, Memory, 2> {
    ಸಾರ್ವಜನಿಕ:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, ಧ್ರುವ sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ಧ್ರುವ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ಧ್ರುವ accessor_t &acc, ಧ್ರುವ sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) ಧ್ರುವ {
            ಪ್ರತ್ಯಾಗಮ _data + _range.get(1) * index;
        }

        pointer_t get_ptr() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ _data; }

    ಗುಪ್ತ:
        pointer_t _data;
        sycl::range<2> _range;
    };

    ನಾಮಾಕಾಶ detail {
        /// Device variable with address space of shared, global or constant.
        template <ವರ್ಗ T, memory_region Memory, size_t Dimension> ವರ್ಗ device_memory {
        ಸಾರ್ವಜನಿಕ:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using ಮೌಲ್ಯ_t = typename detail::memory_traits<Memory, T>::ಮೌಲ್ಯ_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(ಧ್ರುವ sycl::range<Dimension> &in_range,
                        std::initializer_list<ಮೌಲ್ಯ_t> &&init_list)
                : device_memory(in_range) {
                ಪ್ರತಿಜ್ಞಾ(init_list.size() <= in_range.size());
                _host_ptr = (ಮೌಲ್ಯ_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * ಪರಿಮಾಣ(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                ಧ್ರುವ typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<ಮೌಲ್ಯ_t>> &&init_list)
                : device_memory(in_range) {
                ಪ್ರತಿಜ್ಞಾ(init_list.size() <= in_range[0]);
                _host_ptr = (ಮೌಲ್ಯ_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                ಪ್ರತಿ (auto sub_list : init_list) {
                    ಪ್ರತಿಜ್ಞಾ(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * ಪರಿಮಾಣ(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(ಧ್ರುವ sycl::range<Dimension> &range_in)
                : _size(range_in.size() * ಪರಿಮಾಣ(T)), _range(range_in),
                _reference(ಅಸತ್ಯ), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <ವರ್ಗ... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                ಯದಿ (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                ಯದಿ (_host_ptr)
                    std::ಮುಕ್ತ(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            ಶೂನ್ಯ init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            ಶೂನ್ಯ init(sycl::queue &q) {
                ಯದಿ (_device_ptr)
                    ಪ್ರತ್ಯಾಗಮ;
                ಯದಿ (!_size)
                    ಪ್ರತ್ಯಾಗಮ;
                allocate_device(q);
                ಯದಿ (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            ಶೂನ್ಯ assign(ಮೌಲ್ಯ_t *src, size_t size) {
                ಏತತ್->~device_memory();
                ನೂತನ (ಏತತ್) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ಮೌಲ್ಯ_t *get_ptr() { ಪ್ರತ್ಯಾಗಮ get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ಮೌಲ್ಯ_t *get_ptr(sycl::queue &q) {
                init(q);
                ಪ್ರತ್ಯಾಗಮ _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { ಪ್ರತ್ಯಾಗಮ _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                ಪ್ರತ್ಯಾಗಮ _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                ಪ್ರತ್ಯಾಗಮ dpct_accessor_t((T *)_device_ptr, _range);
            }

        ಗುಪ್ತ:
            device_memory(ಮೌಲ್ಯ_t *memory_ptr, size_t size)
                : _size(size), _range(size / ಪರಿಮಾಣ(T)), _reference(ಸತ್ಯ),
                _device_ptr(memory_ptr) {}

            ಶೂನ್ಯ allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                ಯದಿ (Memory == shared) {
                    _device_ptr = (ಮೌಲ್ಯ_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    ಪ್ರತ್ಯಾಗಮ;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                ಯದಿ (Memory == constant) {
                    _device_ptr = (ಮೌಲ್ಯ_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    ಪ್ರತ್ಯಾಗಮ;
                }
        #endif
        #endif
                _device_ptr = (ಮೌಲ್ಯ_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            ತಾರ್ಕಿಕ _reference;
            ಮೌಲ್ಯ_t *_host_ptr;
            ಮೌಲ್ಯ_t *_device_ptr;
        };
        template <ವರ್ಗ T, memory_region Memory>
        ವರ್ಗ device_memory<T, Memory, 0> : ಸಾರ್ವಜನಿಕ device_memory<T, Memory, 1> {
        ಸಾರ್ವಜನಿಕ:
            using base = device_memory<T, Memory, 1>;
            using ಮೌಲ್ಯ_t = typename base::ಮೌಲ್ಯ_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(ಧ್ರುವ ಮೌಲ್ಯ_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <ವರ್ಗ T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <ವರ್ಗ T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <ವರ್ಗ T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    ಅನ್ತಃಸ್ಥ T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ಪ್ರತ್ಯಾಗಮ atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    ಅನ್ತಃಸ್ಥ T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ಪ್ರತ್ಯಾಗಮ atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    ಅನ್ತಃಸ್ಥ T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    ವಿಕಲ್ಪ (memoryOrder) {
        ಪಕ್ಷ sycl::memory_order::relaxed:
            ಪ್ರತ್ಯಾಗಮ atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        ಪಕ್ಷ sycl::memory_order::acq_rel:
            ಪ್ರತ್ಯಾಗಮ atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        ಪಕ್ಷ sycl::memory_order::seq_cst:
            ಪ್ರತ್ಯಾಗಮ atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        ಸ್ವಭಾವ:
            ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    ಅನ್ತಃಸ್ಥ T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    ಅನ್ತಃಸ್ಥ unsigned ಪೂರ್ಣಾಙ್ಕ byte_level_permute(
        unsigned ಪೂರ್ಣಾಙ್ಕ a, unsigned ಪೂರ್ಣಾಙ್ಕ b, unsigned ಪೂರ್ಣಾಙ್ಕ s) {
      unsigned ಪೂರ್ಣಾಙ್ಕ ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      ಪ್ರತ್ಯಾಗಮ ret;
    }

    ಅನ್ತಃಸ್ಥ uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, ಪೂರ್ಣಾಙ್ಕ mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      ಯದಿ (mode >= 1 && mode <= 6) {
        ಪ್ರತ್ಯಾಗಮ byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } ಅನ್ಯಥಾ ಯದಿ (!mode) {
        ಪ್ರತ್ಯಾಗಮ byte_level_permute(low32, high32, sel);
      }
      ಪ್ರತ್ಯಾಗಮ 0;
    }

    template <ಪೂರ್ಣಾಙ್ಕ n_nondefault_params, ಪೂರ್ಣಾಙ್ಕ n_default_params, typename T>
    ವರ್ಗ args_selector;

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
    template <ಪೂರ್ಣಾಙ್ಕ n_nondefault_params, ಪೂರ್ಣಾಙ್ಕ n_default_params, typename R,
              typename... Ts>
    ವರ್ಗ args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      ಗುಪ್ತ:
        ಶೂನ್ಯ **kernel_params;
        ವರ್ಣ *args_buffer;

        template <ಪೂರ್ಣಾಙ್ಕ i> ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ account_for_default_params() {
            constexpr ಪೂರ್ಣಾಙ್ಕ n_total_params = ಪರಿಮಾಣ...(Ts);
            ಯದಿ constexpr (i >= n_nondefault_params) {
                ಪ್ರತ್ಯಾಗಮ n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ i;
            }
        }

      ಸಾರ್ವಜನಿಕ:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <ಪೂರ್ಣಾಙ್ಕ i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ params_ಅಂಕೆ = ಪರಿಮಾಣ...(Ts);

      ಗುಪ್ತ:
        template <ಪೂರ್ಣಾಙ್ಕ i> ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ get_offset() {
            ಯದಿ constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                ಪ್ರತ್ಯಾಗಮ 0;
            } ಅನ್ಯಥಾ {
                constexpr ಪೂರ್ಣಾಙ್ಕ prev_off = get_offset<i - 1>();
                constexpr ಪೂರ್ಣಾಙ್ಕ prev_past_end =
                    prev_off + ಪರಿಮಾಣ(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                ಯದಿ constexpr (prev_past_end % alignof(T) == 0) {
                    ಪ್ರತ್ಯಾಗಮ prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                ಅನ್ಯಥಾ {
                    ಪ್ರತ್ಯಾಗಮ prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        ಸ್ಥಿರ ವರ್ಣ *get_args_buffer(ಶೂನ್ಯ **extra) {
            ಯದಿ (!extra)
                ಪ್ರತ್ಯಾಗಮ nullptr;
            ಪ್ರತಿ (; (std::size_t)*extra != 0; ++extra) {
                ಯದಿ ((std::size_t)*extra == 1) {
                    ಪ್ರತ್ಯಾಗಮ static_cast<ವರ್ಣ *>(*(extra + 1));
                }
            }
            ಪ್ರತ್ಯಾಗಮ nullptr;
        }

      ಸಾರ್ವಜನಿಕ:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(ಶೂನ್ಯ **kernel_params, ಶೂನ್ಯ **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <ಪೂರ್ಣಾಙ್ಕ i> arg_type<i> &get() {
            ಯದಿ (kernel_params) {
                ಪ್ರತ್ಯಾಗಮ *static_cast<arg_type<i> *>(kernel_params[i]);
            } ಅನ್ಯಥಾ {
                ಪ್ರತ್ಯಾಗಮ *reinterpret_cast<arg_type<i> *>(args_buffer +
                                                        get_offset<i>());
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
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        ಸ್ಥಿರ ಶೂನ್ಯ launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        ಸ್ಥಿರ ಶೂನ್ಯ set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned ಪೂರ್ಣಾಙ್ಕ local_mem_size,
                                         queue_ptr que) {
            ಯದಿ (que) {
                _que = que;
            } ಅನ್ಯಥಾ {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::mutex kernel_function_ptr_map_mutex;

      ಸಾರ್ವಜನಿಕ:
        /// Variables for storing execution configuration.
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ thread_local sycl::queue *_que = nullptr;
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ thread_local unsigned ಪೂರ್ಣಾಙ್ಕ _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ std::map<
            ಧ್ರುವ ಶೂನ್ಯ *,
            std::function<ಶೂನ್ಯ(dim3, dim3, ಶೂನ್ಯ **, unsigned ಪೂರ್ಣಾಙ್ಕ, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ಸ್ಥಿರ ಶೂನ್ಯ register_kernel_ptr(
            ಧ್ರುವ ಶೂನ್ಯ *func,
            std::function<ಶೂನ್ಯ(dim3, dim3, ಶೂನ್ಯ **, unsigned ಪೂರ್ಣಾಙ್ಕ, queue_ptr)>
                launcher) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
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
        template <typename FuncT, typename... ArgsT>
        ಸ್ಥಿರ std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, ಶೂನ್ಯ>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned ಪೂರ್ಣಾಙ್ಕ local_mem_size, queue_ptr que, ArgsT... args) {
            set_execution_config(group_range, local_range, local_mem_size, que);
            func(args...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        ಸ್ಥಿರ ಶೂನ್ಯ launch(ಧ್ರುವ ಶೂನ್ಯ *func, dim3 group_range, dim3 local_range,
                           ಶೂನ್ಯ **args, unsigned ಪೂರ್ಣಾಙ್ಕ local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            ಯದಿ (Iter == kernel_function_ptr_map.end()) {
                ಕ್ಷಿಪ std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(group_range, local_range, args, local_mem_size, que);
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
        template <typename FuncT>
        ಸ್ಥಿರ std::enable_if_t<std::is_function_v<FuncT>, ಶೂನ್ಯ>
        launch(FuncT *func, dim3 group_range, dim3 local_range, ಶೂನ್ಯ **args,
               unsigned ಪೂರ್ಣಾಙ್ಕ local_mem_size, queue_ptr que) {
            constexpr size_t p_ಅಂಕೆ = args_selector<0, 0, FuncT>::params_ಅಂಕೆ;
            set_execution_config(group_range, local_range, local_mem_size, que);
            args_selector<p_ಅಂಕೆ, p_ಅಂಕೆ, FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_ಅಂಕೆ>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    template <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        ಪೂರ್ಣಾಙ್ಕ remote_local_id,
        ಪೂರ್ಣಾಙ್ಕ logical_sub_group_size = 32) {
      unsigned ಪೂರ್ಣಾಙ್ಕ start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      ಪ್ರತ್ಯಾಗಮ sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    ಶೂನ್ಯ ldmatrix(uintptr_t addr, T* m, ತಾರ್ಕಿಕ trans = ಅಸತ್ಯ, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      ಪೂರ್ಣಾಙ್ಕ lane = sg.get_local_linear_id();

      ಪೂರ್ಣಾಙ್ಕ lane_group8_row = lane / 8;
      ಪೂರ್ಣಾಙ್ಕ lane_group8_col = lane % 8;

      ಯದಿ (!trans) {
        // calculate the source lane
        ಪೂರ್ಣಾಙ್ಕ src_lane = 2 * lane_group8_row;
        ಯದಿ (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } ಅನ್ಯಥಾ {
        // calculate the source lane
        ಪೂರ್ಣಾಙ್ಕ src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        ಪೂರ್ಣಾಙ್ಕ index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    ಶೂನ್ಯ ldmatrix(uintptr_t addr, T* m1, T* m2, ತಾರ್ಕಿಕ trans = ಅಸತ್ಯ) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    ಶೂನ್ಯ ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, ತಾರ್ಕಿಕ trans = ಅಸತ್ಯ) {
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
    template <typename T>
    ಸಂರಚನಾ MMAType {
      using PackType = uint32_t;
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
    template <ಪೂರ್ಣಾಙ್ಕ M, ಪೂರ್ಣಾಙ್ಕ N, ಪೂರ್ಣಾಙ್ಕ K, typename ABType, typename CDType>
    ಶೂನ್ಯ mma(
        ಚಞ್ಚಲ ಶೂನ್ಯ** d_mat_frag,
        ಶೂನ್ಯ* a_mat_frag,
        ಶೂನ್ಯ* b_mat_frag,
        ಶೂನ್ಯ* c_mat_frag) {
      auto d = reinterpret_cast<ಚಞ್ಚಲ CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      ಪೂರ್ಣಾಙ್ಕ lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      ಯದಿ constexpr (M == 8 && N == 8 && K == 4) {
        ಯದಿ constexpr (std::is_floating_point_v<CDType>) {
          col_load_offset = row_load_offset % 16;

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
          ಪೂರ್ಣಾಙ್ಕ r_ind = (lane % 2) ? 1 : 0;
          ಪೂರ್ಣಾಙ್ಕ c_ind = ((lane % 4) / 2) ? 2 : 0;

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
          typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_group(sg, a[1], row_load_offset + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[1], col_load_offset + i);

            auto ra = reinterpret_cast<ABType*>(recv_a);
            auto rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment (for
            // even work item indices) d0 += col0{ a0 } * row0{ b0 } d1 += col0{
            // a0 } * row0{ b1 } d2 += col1{ a2 } * row0{ b0 } d3 += col1{ a2 }
            // * row0{ b1 } (for odd work item indices) d0 += col0{ a1 } * row0{
            // b2 } d1 += col0{ a1 } * row0{ b3 } d2 += col1{ a3 } * row0{ b2 }
            // d3 += col1{ a3 } * row0{ b3 }
            *d[0] +=
                static_cast<ಭಿನ್ನ>(ra[r_ind]) * static_cast<ಭಿನ್ನ>(rb[c_ind]);
            *d[1] += static_cast<ಭಿನ್ನ>(ra[r_ind]) *
                     static_cast<ಭಿನ್ನ>(rb[c_ind + 1]);
            *d[2] += static_cast<ಭಿನ್ನ>(ra[r_ind + 2]) *
                     static_cast<ಭಿನ್ನ>(rb[c_ind]);
            *d[3] += static_cast<ಭಿನ್ನ>(ra[r_ind + 2]) *
                     static_cast<ಭಿನ್ನ>(rb[c_ind + 1]);

            // Load partial fragment from row1 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[1], col_load_offset + i + 16);

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
                static_cast<ಭಿನ್ನ>(ra[r_ind]) * static_cast<ಭಿನ್ನ>(rb[c_ind]);
            *d[5] += static_cast<ಭಿನ್ನ>(ra[r_ind]) *
                     static_cast<ಭಿನ್ನ>(rb[c_ind + 1]);
            *d[6] += static_cast<ಭಿನ್ನ>(ra[r_ind + 2]) *
                     static_cast<ಭಿನ್ನ>(rb[c_ind]);
            *d[7] += static_cast<ಭಿನ್ನ>(ra[r_ind + 2]) *
                     static_cast<ಭಿನ್ನ>(rb[c_ind + 1]);
          }
        }
      } ಅನ್ಯಥಾ ಯದಿ constexpr (M == 8 && N == 8 && K == 16) {
        ಯದಿ constexpr (std::is_integral_v<ABType>) {
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
          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

            auto a = reinterpret_cast<ABType*>(&recv_a);
            auto b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } ಅನ್ಯಥಾ ಯದಿ constexpr (M == 16 && N == 8 && K == 8) {
        ಯದಿ constexpr (std::is_floating_point_v<CDType>) {
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
          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

            auto ra = reinterpret_cast<ABType*>(recv_a);
            auto rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 2; j++) {
              *d[0] += static_cast<ಭಿನ್ನ>(ra[j]) * static_cast<ಭಿನ್ನ>(rb[j]);
              *d[1] +=
                  static_cast<ಭಿನ್ನ>(ra[j]) * static_cast<ಭಿನ್ನ>(rb[j + 2]);
              *d[2] +=
                  static_cast<ಭಿನ್ನ>(ra[j + 2]) * static_cast<ಭಿನ್ನ>(rb[j]);
              *d[3] +=
                  static_cast<ಭಿನ್ನ>(ra[j + 2]) * static_cast<ಭಿನ್ನ>(rb[j + 2]);
            }
          }
        }
      } ಅನ್ಯಥಾ ಯದಿ constexpr (M == 16 && N == 8 && K == 16) {
        ಯದಿ constexpr (std::is_floating_point_v<CDType>) {
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
          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_group(sg, a[2], row_load_offset + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_group(sg, a[3], row_load_offset + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[1], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_group(sg, b[1], col_load_offset + 4 + i);

            auto ra = reinterpret_cast<ABType*>(recv_a);
            auto rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } ಅನ್ಯಥಾ ಯದಿ constexpr (std::is_integral_v<ABType>) {
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
          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

            auto ra = reinterpret_cast<ABType*>(recv_a);
            auto rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } ಅನ್ಯಥಾ ಯದಿ constexpr (M == 16 && N == 8 && K == 32) {
        ಯದಿ constexpr (std::is_integral_v<ABType>) {
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
          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_group(sg, a[1], row_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

            auto a = reinterpret_cast<ABType*>(recv_a);
            auto b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_group(sg, a[2], row_load_offset + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_group(sg, a[3], row_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[1], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[1], col_load_offset + i + 4);

            auto a = reinterpret_cast<ABType*>(recv_a);
            auto b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; j++) {
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

#endif // GGML_SYCL_DPCT_HELPER_HPP

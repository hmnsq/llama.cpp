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

#অন্তর্ভাৱ <cstdlib>
#অন্তর্ভাৱ <iostream>
#অন্তর্ভাৱ <map>

#অন্তর্ভাৱ <sycl/sycl.hpp>
#অন্তর্ভাৱ <sycl/half_type.hpp>
#অন্তর্ভাৱ <oneapi/mkl.hpp>

#যদি defined(__linux__)
#অন্তর্ভাৱ <sys/mman.h>
#নোচেদ্যদি defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#অন্তর্ভাৱ <windows.h>
#অন্যথা
#error "Only support Windows and Linux."
#endif

#যদি defined(__linux__)
#অন্তর্ভাৱ <unistd.h>
#অন্তর্ভাৱ <sys/syscall.h>
#endif
#যদি defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#অন্তর্ভাৱ <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#যদি defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#অন্যথা
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#যদি defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#অন্যথা
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (শূন্য)(x)

[[noreturn]] অন্তঃস্থ শূন্য _abort(ধ্রুৱ ৱর্ণ * str) {
    std::cerr << str << std::endl;
    std::abort();
}

অন্তঃস্থ std::string get_device_type_name(ধ্রুৱ sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    ৱিকল্প (DeviceType) {
    পক্ষ sycl::info::device_type::cpu:
        ঘূৰাই "cpu";
    পক্ষ sycl::info::device_type::gpu:
        ঘূৰাই "gpu";
    পক্ষ sycl::info::device_type::host:
        ঘূৰাই "host";
    পক্ষ sycl::info::device_type::accelerator:
        ঘূৰাই "acc";
    স্ৱভাৱ:
        ঘূৰাই "unknown";
    }
}

অন্তঃস্থ std::string get_device_backend_and_type(ধ্রুৱ sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    ঘূৰাই device_type.str();
}

template <typename Ts> সংরচনা matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     মান_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

নামাকাশ dpct
{
    প্রকারনির্দেশ sycl::queue *queue_ptr;
    প্রকারনির্দেশ sycl::event *event_ptr;
    প্রকারনির্দেশ ৱর্ণ *device_ptr;
    প্রকারনির্দেশ uint8_t byte_t;
    প্রকারনির্দেশ sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    অন্তঃস্থ auto exception_handler = [](sycl::exception_list exceptions)
    {
        বাবে (std::exception_ptr ধ্রুৱ &e : exceptions)
        {
            যত্ন
            {
                std::rethrow_exception(e);
            }
            গৃহাণ (sycl::exception ধ্রুৱ &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    গণনা error_code
    {
        success = 0,
        default_error = 999
    };

    গণনা memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    গণনা memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    গণনা শ্ৰেণী library_data_t : unsigned ৱর্ণ
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
    সংরচনা DataType
    {
        using T2 = T;
    };
    template <typename T>
    সংরচনা DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    স্থির শূন্য destroy_event(event_ptr event)
    {
        delete event;
    }

    স্থির অন্তঃস্থ unsigned পূর্ণাঙ্ক get_tid()
    {
#যদি defined(__linux__)
        ঘূৰাই syscall(SYS_gettid);
#নোচেদ্যদি defined(_WIN64)
        ঘূৰাই GetCurrentThreadId();
#অন্যথা
#error "Only support Windows and Linux."
#endif
    }

    নামাকাশ detail
    {
        স্থির শূন্য get_version(ধ্রুৱ sycl::device &dev, পূর্ণাঙ্ক &major, পূর্ণাঙ্ক &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            যেতিয়ালৈ (i < ver.size()) {
              যদি (isdigit(ver[i]))
                ভঙ্গ;
              i++;
            }
            major = std::stoi(&(ver[i]));
            যেতিয়ালৈ (i < ver.size()) {
              যদি (ver[i] == '.')
                ভঙ্গ;
              i++;
            }
            যদি (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } অন্যথা {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        শ্ৰেণী generic_error_type
        {
        সার্ৱজনিক:
            generic_error_type() = স্ৱভাৱ;
            generic_error_type(T value) : value{value} {}
            operator T() ধ্রুৱ { ঘূৰাই value; }

        গুপ্ত:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    শ্ৰেণী dim3 {
        সার্ৱজনিক:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(ধ্রুৱ sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() ধ্রুৱ { ঘূৰাই sycl::range<3>(z, y, x); }
    }; // namespace dim3

    অন্তঃস্থ dim3 operator*(ধ্রুৱ dim3 &a, ধ্রুৱ dim3 &b) {
    ঘূৰাই dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    শ্ৰেণী pitched_data
    {
    সার্ৱজনিক:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(শূন্য *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        শূন্য *get_data_ptr() { ঘূৰাই _data; }
        শূন্য set_data_ptr(শূন্য *data) { _data = data; }

        size_t get_pitch() { ঘূৰাই _pitch; }
        শূন্য set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { ঘূৰাই _x; }
        শূন্য set_x(size_t x) { _x = x; }

        size_t get_y() { ঘূৰাই _y; }
        শূন্য set_y(size_t y) { _y = y; }

    গুপ্ত:
        শূন্য *_data;
        size_t _pitch, _x, _y;
    };

    শ্ৰেণী device_info
    {
    সার্ৱজনিক:
        // get interface
        ধ্রুৱ ৱর্ণ *get_name() ধ্রুৱ { ঘূৰাই _name; }
        ৱর্ণ *get_name() { ঘূৰাই _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, পূর্ণাঙ্ক *>,
                                   পূর্ণাঙ্ক> = 0>
        auto get_max_work_item_sizes() ধ্রুৱ
        {
            যদি constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ঘূৰাই sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            অন্যথা
            {
                ঘূৰাই _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, পূর্ণাঙ্ক *>,
                                   পূর্ণাঙ্ক> = 0>
        auto get_max_work_item_sizes()
        {
            যদি constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ঘূৰাই sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            অন্যথা
            {
                ঘূৰাই _max_work_item_sizes_i;
            }
        }
        তার্কিক get_host_unified_memory() ধ্রুৱ { ঘূৰাই _host_unified_memory; }
        পূর্ণাঙ্ক get_major_version() ধ্রুৱ { ঘূৰাই _major; }
        পূর্ণাঙ্ক get_minor_version() ধ্রুৱ { ঘূৰাই _minor; }
        পূর্ণাঙ্ক get_integrated() ধ্রুৱ { ঘূৰাই _integrated; }
        পূর্ণাঙ্ক get_max_clock_frequency() ধ্রুৱ { ঘূৰাই _frequency; }
        পূর্ণাঙ্ক get_max_compute_units() ধ্রুৱ { ঘূৰাই _max_compute_units; }
        পূর্ণাঙ্ক get_max_work_group_size() ধ্রুৱ { ঘূৰাই _max_work_group_size; }
        পূর্ণাঙ্ক get_max_sub_group_size() ধ্রুৱ { ঘূৰাই _max_sub_group_size; }
        পূর্ণাঙ্ক get_max_work_বস্তুসমূহ_per_compute_unit() ধ্রুৱ
        {
            ঘূৰাই max_work_বস্তুসমূহ_per_compute_unit;
        }
        পূর্ণাঙ্ক get_max_register_size_per_work_group() ধ্রুৱ
        {
            ঘূৰাই _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, পূর্ণাঙ্ক *>,
                                   পূর্ণাঙ্ক> = 0>
        auto get_max_nd_range_size() ধ্রুৱ
        {
            যদি constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                ঘূৰাই _max_nd_range_size;
            অন্যথা
                ঘূৰাই _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, পূর্ণাঙ্ক *>,
                                   পূর্ণাঙ্ক> = 0>
        auto get_max_nd_range_size()
        {
            যদি constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                ঘূৰাই _max_nd_range_size;
            অন্যথা
                ঘূৰাই _max_nd_range_size_i;
        }
        size_t get_global_mem_size() ধ্রুৱ { ঘূৰাই _global_mem_size; }
        size_t get_local_mem_size() ধ্রুৱ { ঘূৰাই _local_mem_size; }
        size_t get_max_mem_alloc_size() ধ্রুৱ { ঘূৰাই _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned পূর্ণাঙ্ক get_memory_clock_rate() ধ্রুৱ { ঘূৰাই _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned পূর্ণাঙ্ক get_memory_bus_width() ধ্রুৱ { ঘূৰাই _memory_bus_width; }
        uint32_t get_device_id() ধ্রুৱ { ঘূৰাই _device_id; }
        std::array<unsigned ৱর্ণ, 16> get_uuid() ধ্রুৱ { ঘূৰাই _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned পূর্ণাঙ্ক get_global_mem_cache_size() ধ্রুৱ
        {
            ঘূৰাই _global_mem_cache_size;
        }

        // set interface
        শূন্য set_name(ধ্রুৱ ৱর্ণ *name)
        {
            size_t length = strlen(name);
            যদি (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            অন্যথা
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        শূন্য set_max_work_item_sizes(ধ্রুৱ sycl::range<3> max_work_item_sizes)
        {
            বাবে (পূর্ণাঙ্ক i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] শূন্য
        set_max_work_item_sizes(ধ্রুৱ sycl::id<3> max_work_item_sizes)
        {
            বাবে (পূর্ণাঙ্ক i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        শূন্য set_host_unified_memory(তার্কিক host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        শূন্য set_major_version(পূর্ণাঙ্ক major) { _major = major; }
        শূন্য set_minor_version(পূর্ণাঙ্ক minor) { _minor = minor; }
        শূন্য set_integrated(পূর্ণাঙ্ক integrated) { _integrated = integrated; }
        শূন্য set_max_clock_frequency(পূর্ণাঙ্ক frequency) { _frequency = frequency; }
        শূন্য set_max_compute_units(পূর্ণাঙ্ক max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        শূন্য set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        শূন্য set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        শূন্য set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        শূন্য set_max_work_group_size(পূর্ণাঙ্ক max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        শূন্য set_max_sub_group_size(পূর্ণাঙ্ক max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        শূন্য
        set_max_work_বস্তুসমূহ_per_compute_unit(পূর্ণাঙ্ক max_work_বস্তুসমূহ_per_compute_unit_2)
        {
            max_work_বস্তুসমূহ_per_compute_unit = max_work_বস্তুসমূহ_per_compute_unit_2;
        }
        শূন্য set_max_nd_range_size(পূর্ণাঙ্ক max_nd_range_size[])
        {
            বাবে (পূর্ণাঙ্ক i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        শূন্য set_memory_clock_rate(unsigned পূর্ণাঙ্ক memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        শূন্য set_memory_bus_width(unsigned পূর্ণাঙ্ক memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        শূন্য
        set_max_register_size_per_work_group(পূর্ণাঙ্ক max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        শূন্য set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        শূন্য set_uuid(std::array<unsigned ৱর্ণ, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        শূন্য set_global_mem_cache_size(unsigned পূর্ণাঙ্ক global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    গুপ্ত:
        ৱর্ণ _name[256];
        পূর্ণাঙ্ক _max_work_item_sizes_i[3];
        তার্কিক _host_unified_memory = মিছা;
        পূর্ণাঙ্ক _major;
        পূর্ণাঙ্ক _minor;
        পূর্ণাঙ্ক _integrated = 0;
        পূর্ণাঙ্ক _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned পূর্ণাঙ্ক _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned পূর্ণাঙ্ক _memory_bus_width = 64;
        unsigned পূর্ণাঙ্ক _global_mem_cache_size;
        পূর্ণাঙ্ক _max_compute_units;
        পূর্ণাঙ্ক _max_work_group_size;
        পূর্ণাঙ্ক _max_sub_group_size;
        পূর্ণাঙ্ক max_work_বস্তুসমূহ_per_compute_unit;
        পূর্ণাঙ্ক _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        পূর্ণাঙ্ক _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned ৱর্ণ, 16> _uuid;
    };

    স্থির পূর্ণাঙ্ক get_major_version(ধ্রুৱ sycl::device &dev)
    {
        পূর্ণাঙ্ক major, minor;
        detail::get_version(dev, major, minor);
        ঘূৰাই major;
    }

    স্থির পূর্ণাঙ্ক get_minor_version(ধ্রুৱ sycl::device &dev)
    {
        পূর্ণাঙ্ক major, minor;
        detail::get_version(dev, major, minor);
        ঘূৰাই minor;
    }

    স্থির শূন্য get_device_info(device_info &out, ধ্রুৱ sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        পূর্ণাঙ্ক major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#যদি (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#অন্যথা
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

#যদি (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        যদি (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned পূর্ণাঙ্ক tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            যদি (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        যদি (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        যদি (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        যদি (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#নোচেদ্যদি defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate স্ৱভাৱ value. \
        Use 64 bits as memory_bus_width স্ৱভাৱ value.")
#অন্যথা
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate স্ৱভাৱ value. \
        Use 64 bits as memory_bus_width স্ৱভাৱ value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        বাবে (ধ্রুৱ auto &sub_group_size : sub_group_sizes)
        {
            যদি (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_বস্তুসমূহ_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        পূর্ণাঙ্ক max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    শ্ৰেণী device_ext : সার্ৱজনিক sycl::device {
      প্রকারনির্দেশ std::mutex mutex_type;

     সার্ৱজনিক:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(ধ্রুৱ sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      পূর্ণাঙ্ক is_native_atomic_supported() { ঘূৰাই 0; }
      পূর্ণাঙ্ক get_major_version() ধ্রুৱ { ঘূৰাই dpct::get_major_version(*এতত্); }

      পূর্ণাঙ্ক get_minor_version() ধ্রুৱ { ঘূৰাই dpct::get_minor_version(*এতত্); }

      পূর্ণাঙ্ক get_max_compute_units() ধ্রুৱ {
        ঘূৰাই get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      পূর্ণাঙ্ক get_max_clock_frequency() ধ্রুৱ {
        ঘূৰাই get_device_info().get_max_clock_frequency();
      }

      পূর্ণাঙ্ক get_integrated() ধ্রুৱ { ঘূৰাই get_device_info().get_integrated(); }

      পূর্ণাঙ্ক get_max_sub_group_size() ধ্রুৱ {
        ঘূৰাই get_device_info().get_max_sub_group_size();
      }

      পূর্ণাঙ্ক get_max_register_size_per_work_group() ধ্রুৱ {
        ঘূৰাই get_device_info().get_max_register_size_per_work_group();
      }

      পূর্ণাঙ্ক get_max_work_group_size() ধ্রুৱ {
        ঘূৰাই get_device_info().get_max_work_group_size();
      }

      পূর্ণাঙ্ক get_mem_base_addr_align() ধ্রুৱ {
        ঘূৰাই get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() ধ্রুৱ {
        ঘূৰাই get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() ধ্রুৱ {
        ঘূৰাই get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      শূন্য get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        ধ্রুৱ ৱর্ণ *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#যদি (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        যদি (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } অন্যথা {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#অন্যথা
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#যদি defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#অন্যথা
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      শূন্য get_device_info(device_info &out) ধ্রুৱ {
        dpct::get_device_info(out, *এতত্);
      }

      device_info get_device_info() ধ্রুৱ {
        device_info prop;
        dpct::get_device_info(prop, *এতত্);
        ঘূৰাই prop;
      }

      শূন্য reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { ঘূৰাই _q_in_order; }

      sycl::queue &out_of_order_queue() { ঘূৰাই _q_out_of_order; }

      sycl::queue &default_queue() { ঘূৰাই in_order_queue(); }

      শূন্য queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        বাবে (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(তার্কিক enable_exception_handler = মিছা) {
        ঘূৰাই create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               তার্কিক enable_exception_handler = মিছা) {
        ঘূৰাই create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(তার্কিক enable_exception_handler = মিছা) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ঘূৰাই create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        তার্কিক enable_exception_handler = মিছা) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ঘূৰাই create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          তার্কিক enable_exception_handler = মিছা) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ঘূৰাই create_queue_impl(enable_exception_handler);
      }

      শূন্য destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ধ্রুৱ sycl::queue &q) -> তার্কিক
                                    {
                                        ঘূৰাই q == queue;
                                    }),
                    _queues.end());
      }
      শূন্য set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() ধ্রুৱ {
        std::lock_guard<mutex_type> lock(m_mutex);
        ঘূৰাই _saved_queue;
      }

     গুপ্ত:
      শূন্য clear_queues() { _queues.clear(); }

      শূন্য init_queues() {
        _q_in_order =
            create_queue_impl(সঁচা, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(সঁচা);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <শ্ৰেণী... Properties>
      sycl::queue create_queue_impl(তার্কিক enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        যদি (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *এতত্, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        ঘূৰাই _queues.back();
      }

      template <শ্ৰেণী... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    তার্কিক enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        যদি (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        ঘূৰাই _queues.back();
      }

      শূন্য get_version(পূর্ণাঙ্ক &major, পূর্ণাঙ্ক &minor) ধ্রুৱ {
        detail::get_version(*এতত্, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    শ্ৰেণী dev_mgr
    {
    সার্ৱজনিক:
        device_ext &current_device()
        {
            unsigned পূর্ণাঙ্ক dev_id = current_device_id();
            check_id(dev_id);
            ঘূৰাই *_devs[dev_id];
        }
        device_ext &cpu_device() ধ্রুৱ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            যদি (_cpu_device == -1)
            {
                ক্ষিপ std::runtime_error("no valid cpu device");
            }
            অন্যথা
            {
                ঘূৰাই *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned পূর্ণাঙ্ক id) ধ্রুৱ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            ঘূৰাই *_devs[id];
        }
        unsigned পূর্ণাঙ্ক current_device_id() ধ্রুৱ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            যদি (it != _thread2dev_map.end())
                ঘূৰাই it->second;
            ঘূৰাই DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        শূন্য select_device(unsigned পূর্ণাঙ্ক id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned পূর্ণাঙ্ক device_গুনতি() { ঘূৰাই _devs.size(); }

        unsigned পূর্ণাঙ্ক get_device_id(ধ্রুৱ sycl::device &dev)
        {
            unsigned পূর্ণাঙ্ক id = 0;
            বাবে (auto &dev_item : _devs)
            {
                যদি (*dev_item == dev)
                {
                    ঘূৰাই id;
                }
                id++;
            }
            ঘূৰাই -1;
        }

        অন্তঃস্থ std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            ৱর্ণ* env = getenv("ONEAPI_DEVICE_SELECTOR");
            যদি (env) {
                যদি (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                অন্যথা যদি (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                অন্যথা যদি (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                অন্যথা যদি (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                অন্যথা {
                    ক্ষিপ std::runtime_error("invalid device filter: " + std::string(env));
                }
            } অন্যথা {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                যদি (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                অন্যথা যদি (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                অন্যথা যদি (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            বাবে (ধ্রুৱ auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](ধ্রুৱ sycl::device& d) {
                    ঘূৰাই d.is_gpu();
                });

                যদি (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    অনুৱর্ত;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_নিচেৰ_case;
                platform_name_নিচেৰ_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_নিচেৰ_case.begin(), ::tolower);

                যদি (platform_name_নিচেৰ_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    অনুৱর্ত;
                }

                result = platform_name;
            }

            যদি (result.empty())
                ক্ষিপ std::runtime_error("can not find preferred GPU platform");

            ঘূৰাই result;
        }

        template <শ্ৰেণী DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<পূর্ণাঙ্ক, DeviceSelector, ধ্রুৱ sycl::device &>>
        select_device(ধ্রুৱ DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned পূর্ণাঙ্ক selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        স্থির dev_mgr &instance()
        {
            স্থির dev_mgr d_m;
            ঘূৰাই d_m;
        }
        dev_mgr(ধ্রুৱ dev_mgr &) = delete;
        dev_mgr &operator=(ধ্রুৱ dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    গুপ্ত:
        mutable std::recursive_mutex m_mutex;
        স্থির তার্কিক compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            যদি(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) ঘূৰাই সঁচা;
            যদি(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) ঘূৰাই মিছা;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            ঘূৰাই prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        স্থির পূর্ণাঙ্ক convert_backend_index(std::string & backend) {
            যদি (backend == "ext_oneapi_level_zero:gpu") ঘূৰাই 0;
            যদি (backend == "opencl:gpu") ঘূৰাই 1;
            যদি (backend == "ext_oneapi_cuda:gpu") ঘূৰাই 2;
            যদি (backend == "ext_oneapi_hip:gpu") ঘূৰাই 3;
            যদি (backend == "opencl:cpu") ঘূৰাই 4;
            যদি (backend == "opencl:acc") ঘূৰাই 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        স্থির তার্কিক compare_backend(std::string &backend1, std::string &backend2) {
            ঘূৰাই convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            যদি (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            যেতিয়ালৈ (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                যদি (platform_name.compare(preferred_platform_name) != 0) {
                    অনুৱর্ত;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                বাবে (ধ্রুৱ auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> চাবিগুলি;
            বাবে(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                চাবিগুলি.push_back(it->first);
            }
            std::sort(চাবিগুলি.begin(), চাবিগুলি.end(), compare_backend);

            বাবে (auto &চাবি : চাবিগুলি) {
                std::vector<sycl::device> devs = backend_devices[চাবি];
                std::sort(devs.begin(), devs.end(), compare_dev);
                বাবে (ধ্রুৱ auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            বাবে (auto &dev : sycl_all_devs)
            {
                যদি (dev == default_device)
                {
                    অনুৱর্ত;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                যদি (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        শূন্য check_id(unsigned পূর্ণাঙ্ক id) ধ্রুৱ
        {
            যদি (id >= _devs.size())
            {
                ক্ষিপ std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ধ্রুৱ unsigned পূর্ণাঙ্ক DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned পূর্ণাঙ্ক, unsigned পূর্ণাঙ্ক> _thread2dev_map;
        পূর্ণাঙ্ক _cpu_device = -1;
    };

    স্থির অন্তঃস্থ sycl::queue &get_default_queue()
    {
        ঘূৰাই dev_mgr::instance().current_device().default_queue();
    }

    নামাকাশ detail
    {
        গণনা শ্ৰেণী pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        স্থির pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              ধ্রুৱ শূন্য *ptr)
        {
            ৱিকল্প (sycl::get_pointer_type(ptr, q.get_context()))
            {
            পক্ষ sycl::usm::alloc::unknown:
                ঘূৰাই pointer_access_attribute::host_only;
            পক্ষ sycl::usm::alloc::device:
                ঘূৰাই pointer_access_attribute::device_only;
            পক্ষ sycl::usm::alloc::shared:
            পক্ষ sycl::usm::alloc::host:
                ঘূৰাই pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        অন্তঃস্থ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned ৱর্ণ)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned ৱর্ণ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            ঘূৰাই (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        অন্তঃস্থ constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned ৱর্ণ>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(পরিমাণ...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            ঘূৰাই get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        শ্ৰেণী mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#যদি defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#নোচেদ্যদি defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    রিক্ত,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#অন্যথা
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        সার্ৱজনিক:
            using buffer_id_t = পূর্ণাঙ্ক;

            সংরচনা allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#যদি defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#নোচেদ্যদি defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#অন্যথা
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(ধ্রুৱ mem_mgr &) = delete;
            mem_mgr &operator=(ধ্রুৱ mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            শূন্য *mem_alloc(size_t size)
            {
                যদি (!size)
                    ঘূৰাই nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                যদি (next_free + size > mapped_address_space + mapped_region_size)
                {
                    ক্ষিপ std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                শূন্য *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                ঘূৰাই result;
            }

            /// Deallocate
            শূন্য mem_free(ধ্রুৱ শূন্য *ptr)
            {
                যদি (!ptr)
                    ঘূৰাই;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(ধ্রুৱ শূন্য *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                ঘূৰাই it->second;
            }

            /// Check if the pointer represents device pointer or not.
            তার্কিক is_device_ptr(ধ্রুৱ শূন্য *ptr) ধ্রুৱ
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ঘূৰাই (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            স্থির mem_mgr &instance()
            {
                স্থির mem_mgr m;
                ঘূৰাই m;
            }

        গুপ্ত:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            ধ্রুৱ size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            ধ্রুৱ size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ধ্রুৱ size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(ধ্রুৱ শূন্য *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ধ্রুৱ byte_t *>(ptr)));
                যদি (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ক্ষিপ std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ধ্রুৱ allocation &alloc = it->second;
                যদি (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ক্ষিপ std::runtime_error("invalid virtual pointer");
                }
                ঘূৰাই it;
            }
        };

        template <শ্ৰেণী T, memory_region Memory, size_t Dimension>
        শ্ৰেণী accessor;
        template <memory_region Memory, শ্ৰেণী T = byte_t>
        শ্ৰেণী memory_traits
        {
        সার্ৱজনিক:
            স্থির constexpr sycl::access::target target =
                sycl::access::target::device;
            স্থির constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::পঠ
                                     : sycl::access_mode::read_write;
            স্থির constexpr size_t type_size = পরিমাণ(T);
            using element_t =
                typename std::conditional<Memory == constant, ধ্রুৱ T, T>::type;
            using মান_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<মান_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        স্থির অন্তঃস্থ শূন্য *dpct_malloc(size_t size, sycl::queue &q)
        {
            ঘূৰাই sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        স্থির অন্তঃস্থ শূন্য *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            ঘূৰাই dpct_malloc(pitch * y * z, q);
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
        template <typename মান_T>
        স্থির অন্তঃস্থ sycl::event dpct_memset(sycl::queue &q, শূন্য *dev_ptr,
                                              মান_T value, size_t size)
        {
            ঘূৰাই q.fill(dev_ptr, value, size);
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
        template <typename মান_T>
        স্থির অন্তঃস্থ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, মান_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned ৱর্ণ *data_surface = (unsigned ৱর্ণ *)data.get_data_ptr();
            বাবে (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned ৱর্ণ *data_ptr = data_surface;
                বাবে (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            ঘূৰাই event_list;
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
        template <typename মান_T>
        স্থির অন্তঃস্থ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, শূন্য *ptr, size_t pitch, মান_T val, size_t x,
                    size_t y)
        {
            ঘূৰাই dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        স্থির memcpy_direction deduce_memcpy_direction(sycl::queue &q, শূন্য *to_ptr,
                                                        ধ্রুৱ শূন্য *from_ptr,
                                                        memcpy_direction dir)
        {
            ৱিকল্প (dir)
            {
            পক্ষ memcpy_direction::host_to_host:
            পক্ষ memcpy_direction::host_to_device:
            পক্ষ memcpy_direction::device_to_host:
            পক্ষ memcpy_direction::device_to_device:
                ঘূৰাই dir;
            পক্ষ memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                স্থির ধ্রুৱ memcpy_direction
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
                ঘূৰাই direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            স্ৱভাৱ:
                ক্ষিপ std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        স্থির sycl::event
        dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr, size_t size,
                    memcpy_direction direction,
                    ধ্রুৱ std::vector<sycl::event> &dep_events = {})
        {
            যদি (!size)
                ঘূৰাই sycl::event{};
            ঘূৰাই q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        স্থির অন্তঃস্থ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            ঘূৰাই slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        স্থির অন্তঃস্থ size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            ঘূৰাই slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        স্থির অন্তঃস্থ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ধ্রুৱ std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            শ্ৰেণী host_buffer
            {
                শূন্য *_buf;
                size_t _size;
                sycl::queue &_q;
                ধ্রুৱ std::vector<sycl::event> &_deps; // free operation depends

            সার্ৱজনিক:
                host_buffer(size_t size, sycl::queue &q,
                            ধ্রুৱ std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                শূন্য *get_ptr() ধ্রুৱ { ঘূৰাই _buf; }
                size_t get_size() ধ্রুৱ { ঘূৰাই _size; }
                ~host_buffer()
                {
                    যদি (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::মুক্ত(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned ৱর্ণ *to_surface =
                (unsigned ৱর্ণ *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            ধ্রুৱ unsigned ৱর্ণ *from_surface =
                (ধ্রুৱ unsigned ৱর্ণ *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            যদি (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                ঘূৰাই {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            ৱিকল্প (direction)
            {
            পক্ষ host_to_host:
                বাবে (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned ৱর্ণ *to_ptr = to_surface;
                    ধ্রুৱ unsigned ৱর্ণ *from_ptr = from_surface;
                    যদি (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    অন্যথা
                    {
                        বাবে (size_t y = 0; y < size.get(1); ++y)
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
                ভঙ্গ;
            পক্ষ host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                যদি (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                অন্যথা
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
                ভঙ্গ;
            }
            পক্ষ device_to_host:
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
                ভঙ্গ;
            }
            পক্ষ device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<শ্ৰেণী dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                ভঙ্গ;
            স্ৱভাৱ:
                ক্ষিপ std::runtime_error("dpct_memcpy: invalid direction value");
            }
            ঘূৰাই event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        স্থির অন্তঃস্থ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            ঘূৰাই dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        স্থির অন্তঃস্থ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            ঘূৰাই dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        নামাকাশ deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            শ্ৰেণী usm_allocator
            {
            গুপ্ত:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            সার্ৱজনিক:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    ধ্রুৱ typename std::allocator_traits<Alloc>::value_type &;
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
                সংরচনা rebind
                {
                    প্রকারনির্দেশ usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(ধ্রুৱ usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { ঘূৰাই &r; }
                const_pointer address(const_reference r) { ঘূৰাই &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    ঘূৰাই std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                শূন্য deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ধ্রুৱ
                {
                    ঘূৰাই std::allocator_traits<Alloc>::max_size(_impl);
                }
                তার্কিক operator==(ধ্রুৱ usm_allocator &other) ধ্রুৱ { ঘূৰাই _impl == other._impl; }
                তার্কিক operator!=(ধ্রুৱ usm_allocator &other) ধ্রুৱ { ঘূৰাই _impl != other._impl; }
            };

        } // namespace deprecated

        অন্তঃস্থ শূন্য dpct_free(শূন্য *ptr,
                              ধ্রুৱ sycl::queue &q)
        {
            যদি (ptr)
            {
                sycl::মুক্ত(ptr, q.get_context());
            }
        }

        template <typename T>
        অন্তঃস্থ auto get_memory(ধ্রুৱ শূন্য *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<শূন্য *>(x));
            ঘূৰাই new_x;
        }

        template <typename T>
        অন্তঃস্থ typename DataType<T>::T2 get_মান(ধ্রুৱ T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            যদি (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (শূন্য *)&s_h, (ধ্রুৱ শূন্য *)s, পরিমাণ(T), device_to_host)
                    .wait();
            অন্যথা
                s_h = *reinterpret_cast<ধ্রুৱ Ty *>(s);
            ঘূৰাই s_h;
        }

    } // namespace detail

    template <typename T>
    অন্তঃস্থ auto get_মান(ধ্রুৱ T *s, sycl::queue &q)
    {
        ঘূৰাই detail::get_মান(s, q);
    }

    নামাকাশ detail
    {
    template <শ্ৰেণী Ta, শ্ৰেণী Tb, শ্ৰেণী Tc, শ্ৰেণী Ts>
    অন্তঃস্থ শূন্য gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণাঙ্ক m,
                          পূর্ণাঙ্ক n, পূর্ণাঙ্ক k, ধ্রুৱ শূন্য * alpha, ধ্রুৱ শূন্য * a, পূর্ণাঙ্ক lda, ধ্রুৱ শূন্য * b, পূর্ণাঙ্ক ldb,
                          ধ্রুৱ শূন্য * beta, শূন্য * c, পূর্ণাঙ্ক ldc) {
        Ts   alpha_মান_2 = dpct::get_মান(reinterpret_cast<ধ্রুৱ Ts *>(alpha), q);
        Ts   beta_মান  = dpct::get_মান(reinterpret_cast<ধ্রুৱ Ts *>(beta), q);
        auto data_a      = get_memory<ধ্রুৱ Ta>(a);
        auto data_b      = get_memory<ধ্রুৱ Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_মান_2, data_a,
                                               lda, data_b, ldb, beta_মান, data_c, ldc);
    }

        template <typename VecT, শ্ৰেণী দ্বিমিক_Operation, শ্ৰেণী = শূন্য>
        শ্ৰেণী vectorized_দ্বিমিক
        {
        সার্ৱজনিক:
            অন্তঃস্থ VecT operator()(VecT a, VecT b, ধ্রুৱ দ্বিমিক_Operation দ্বিমিক_op)
            {
                VecT v4;
                বাবে (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = দ্বিমিক_op(a[i], b[i]);
                }
                ঘূৰাই v4;
            }
        };

        template <typename VecT, শ্ৰেণী দ্বিমিক_Operation>
        শ্ৰেণী vectorized_দ্বিমিক<
            VecT, দ্বিমিক_Operation,
            std::void_t<std::invoke_result_t<দ্বিমিক_Operation, VecT, VecT>>>
        {
        সার্ৱজনিক:
            অন্তঃস্থ VecT operator()(VecT a, VecT b, ধ্রুৱ দ্বিমিক_Operation দ্বিমিক_op)
            {
                ঘূৰাই দ্বিমিক_op(a, b).template as<VecT>();
            }
        };

        template <শ্ৰেণী Ta, শ্ৰেণী Tb, শ্ৰেণী Tc, শ্ৰেণী Ts>
        অন্তঃস্থ শূন্য gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    পূর্ণাঙ্ক m, পূর্ণাঙ্ক n, পূর্ণাঙ্ক k, ধ্রুৱ শূন্য * alpha, ধ্রুৱ শূন্য ** a, পূর্ণাঙ্ক lda, ধ্রুৱ শূন্য ** b,
                                    পূর্ণাঙ্ক ldb, ধ্রুৱ শূন্য * beta, শূন্য ** c, পূর্ণাঙ্ক ldc, পূর্ণাঙ্ক batch_size,
                                    matrix_info_t<ভিন্ন> * matrix_info) {
            Ts alpha_মান_2 = dpct::get_মান(reinterpret_cast<ধ্রুৱ Ts *>(alpha), q);
            Ts beta_মান = dpct::get_মান(reinterpret_cast<ধ্রুৱ Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->মান_info[0] = alpha_মান_2;
            matrix_info->মান_info[1] = beta_মান;
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
                reinterpret_cast<Ts *>(matrix_info->মান_info), reinterpret_cast<ধ্রুৱ Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<ধ্রুৱ Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->মান_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <শ্ৰেণী Ta, শ্ৰেণী Tb, শ্ৰেণী Tc, শ্ৰেণী Ts>
        অন্তঃস্থ শূন্য gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    পূর্ণাঙ্ক m, পূর্ণাঙ্ক n, পূর্ণাঙ্ক k, ধ্রুৱ শূন্য * alpha, ধ্রুৱ শূন্য * a, পূর্ণাঙ্ক lda,
                                    long long পূর্ণাঙ্ক stride_a, ধ্রুৱ শূন্য * b, পূর্ণাঙ্ক ldb, long long পূর্ণাঙ্ক stride_b,
                                    ধ্রুৱ শূন্য * beta, শূন্য * c, পূর্ণাঙ্ক ldc, long long পূর্ণাঙ্ক stride_c, পূর্ণাঙ্ক batch_size) {
            Ts alpha_মান_2 = dpct::get_মান(reinterpret_cast<ধ্রুৱ Ts *>(alpha), q);
            Ts beta_মান = dpct::get_মান(reinterpret_cast<ধ্রুৱ Ts *>(beta), q);
            auto data_a = get_memory<ধ্রুৱ Ta>(a);
            auto data_b = get_memory<ধ্রুৱ Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_মান_2,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_মান,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, শ্ৰেণী দ্বিমিক_Operation>
    অন্তঃস্থ unsigned vectorized_দ্বিমিক(unsigned a, unsigned b,
                                      ধ্রুৱ দ্বিমিক_Operation দ্বিমিক_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_দ্বিমিক<VecT, দ্বিমিক_Operation>()(v2, v3, দ্বিমিক_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        ঘূৰাই v0;
    }

    স্থির শূন্য async_dpct_memcpy(শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    স্থির অন্তঃস্থ unsigned পূর্ণাঙ্ক select_device(unsigned পূর্ণাঙ্ক id)
    {
        dev_mgr::instance().select_device(id);
        ঘূৰাই id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned পূর্ণাঙ্ক mask,
                               unsigned পূর্ণাঙ্ক logical_sub_group_size = 32)
    {
        unsigned পূর্ণাঙ্ক id = g.get_local_linear_id();
        unsigned পূর্ণাঙ্ক start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned পূর্ণাঙ্ক target_offset = (id % logical_sub_group_size) ^ mask;
        ঘূৰাই sycl::select_from_group(g, x,
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
      ঘূৰাই sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    অন্তঃস্থ auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      ঘূৰাই res;
    }

    সংরচনা sub_sat
    {
        template <typename T>
        auto operator()(ধ্রুৱ T x, ধ্রুৱ T y) ধ্রুৱ
        {
            ঘূৰাই sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    অন্তঃস্থ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        ঘূৰাই v0;
    }

    অন্তঃস্থ ভিন্ন pow(ধ্রুৱ ভিন্ন a, ধ্রুৱ পূর্ণাঙ্ক b) { ঘূৰাই sycl::pown(a, b); }
    অন্তঃস্থ দ্ৱিগুণ pow(ধ্রুৱ দ্ৱিগুণ a, ধ্রুৱ পূর্ণাঙ্ক b) { ঘূৰাই sycl::pown(a, b); }
    অন্তঃস্থ ভিন্ন pow(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) { ঘূৰাই sycl::pow(a, b); }
    অন্তঃস্থ দ্ৱিগুণ pow(ধ্রুৱ দ্ৱিগুণ a, ধ্রুৱ দ্ৱিগুণ b) { ঘূৰাই sycl::pow(a, b); }
    template <typename T, typename U>
    অন্তঃস্থ typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(ধ্রুৱ T a, ধ্রুৱ U b)
    {
        ঘূৰাই sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    অন্তঃস্থ typename std::enable_if_t<!std::is_floating_point_v<T>, দ্ৱিগুণ>
    pow(ধ্রুৱ T a, ধ্রুৱ U b)
    {
        ঘূৰাই sycl::pow(static_cast<দ্ৱিগুণ>(a), static_cast<দ্ৱিগুণ>(b));
    }

    অন্তঃস্থ দ্ৱিগুণ min(ধ্রুৱ দ্ৱিগুণ a, ধ্রুৱ ভিন্ন b)
    {
        ঘূৰাই sycl::fmin(a, static_cast<দ্ৱিগুণ>(b));
    }
    অন্তঃস্থ দ্ৱিগুণ min(ধ্রুৱ ভিন্ন a, ধ্রুৱ দ্ৱিগুণ b)
    {
        ঘূৰাই sycl::fmin(static_cast<দ্ৱিগুণ>(a), b);
    }
    অন্তঃস্থ ভিন্ন min(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) { ঘূৰাই sycl::fmin(a, b); }
    অন্তঃস্থ দ্ৱিগুণ min(ধ্রুৱ দ্ৱিগুণ a, ধ্রুৱ দ্ৱিগুণ b) { ঘূৰাই sycl::fmin(a, b); }
    অন্তঃস্থ std::uint32_t min(ধ্রুৱ std::uint32_t a, ধ্রুৱ std::int32_t b)
    {
        ঘূৰাই sycl::min(a, static_cast<std::uint32_t>(b));
    }
    অন্তঃস্থ std::uint32_t min(ধ্রুৱ std::int32_t a, ধ্রুৱ std::uint32_t b)
    {
        ঘূৰাই sycl::min(static_cast<std::uint32_t>(a), b);
    }
    অন্তঃস্থ std::int32_t min(ধ্রুৱ std::int32_t a, ধ্রুৱ std::int32_t b)
    {
        ঘূৰাই sycl::min(a, b);
    }
    অন্তঃস্থ std::uint32_t min(ধ্রুৱ std::uint32_t a, ধ্রুৱ std::uint32_t b)
    {
        ঘূৰাই sycl::min(a, b);
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::int64_t b)
    {
        ঘূৰাই sycl::min(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::int64_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::min(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃস্থ std::int64_t min(ধ্রুৱ std::int64_t a, ধ্রুৱ std::int64_t b)
    {
        ঘূৰাই sycl::min(a, b);
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::min(a, b);
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::int32_t b)
    {
        ঘূৰাই sycl::min(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::int32_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::min(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::uint32_t b)
    {
        ঘূৰাই sycl::min(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃস্থ std::uint64_t min(ধ্রুৱ std::uint32_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    অন্তঃস্থ দ্ৱিগুণ max(ধ্রুৱ দ্ৱিগুণ a, ধ্রুৱ ভিন্ন b)
    {
        ঘূৰাই sycl::fmax(a, static_cast<দ্ৱিগুণ>(b));
    }
    অন্তঃস্থ দ্ৱিগুণ max(ধ্রুৱ ভিন্ন a, ধ্রুৱ দ্ৱিগুণ b)
    {
        ঘূৰাই sycl::fmax(static_cast<দ্ৱিগুণ>(a), b);
    }
    অন্তঃস্থ ভিন্ন max(ধ্রুৱ ভিন্ন a, ধ্রুৱ ভিন্ন b) { ঘূৰাই sycl::fmax(a, b); }
    অন্তঃস্থ দ্ৱিগুণ max(ধ্রুৱ দ্ৱিগুণ a, ধ্রুৱ দ্ৱিগুণ b) { ঘূৰাই sycl::fmax(a, b); }
    অন্তঃস্থ std::uint32_t max(ধ্রুৱ std::uint32_t a, ধ্রুৱ std::int32_t b)
    {
        ঘূৰাই sycl::max(a, static_cast<std::uint32_t>(b));
    }
    অন্তঃস্থ std::uint32_t max(ধ্রুৱ std::int32_t a, ধ্রুৱ std::uint32_t b)
    {
        ঘূৰাই sycl::max(static_cast<std::uint32_t>(a), b);
    }
    অন্তঃস্থ std::int32_t max(ধ্রুৱ std::int32_t a, ধ্রুৱ std::int32_t b)
    {
        ঘূৰাই sycl::max(a, b);
    }
    অন্তঃস্থ std::uint32_t max(ধ্রুৱ std::uint32_t a, ধ্রুৱ std::uint32_t b)
    {
        ঘূৰাই sycl::max(a, b);
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::int64_t b)
    {
        ঘূৰাই sycl::max(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::int64_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::max(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃস্থ std::int64_t max(ধ্রুৱ std::int64_t a, ধ্রুৱ std::int64_t b)
    {
        ঘূৰাই sycl::max(a, b);
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::max(a, b);
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::int32_t b)
    {
        ঘূৰাই sycl::max(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::int32_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::max(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::uint64_t a, ধ্রুৱ std::uint32_t b)
    {
        ঘূৰাই sycl::max(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃস্থ std::uint64_t max(ধ্রুৱ std::uint32_t a, ধ্রুৱ std::uint64_t b)
    {
        ঘূৰাই sycl::max(static_cast<std::uint64_t>(a), b);
    }

    অন্তঃস্থ শূন্য
    has_capability_or_fail(ধ্রুৱ sycl::device &dev,
                           ধ্রুৱ std::initializer_list<sycl::aspect> &props)
    {
        বাবে (ধ্রুৱ auto &it : props)
        {
            যদি (dev.has(it))
                অনুৱর্ত;
            ৱিকল্প (it)
            {
            পক্ষ sycl::aspect::fp64:
                ক্ষিপ std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ভঙ্গ;
            পক্ষ sycl::aspect::fp16:
                ক্ষিপ std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ভঙ্গ;
            স্ৱভাৱ:
#define __SYCL_ASPECT(ASPECT, ID) \
    পক্ষ sycl::aspect::ASPECT:    \
        ঘূৰাই #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect Aspect_অঙ্ক) -> std::string
                {
                    ৱিকল্প (Aspect_অঙ্ক)
                    {
#অন্তর্ভাৱ <sycl/info/aspects.def>
#অন্তর্ভাৱ <sycl/info/aspects_deprecated.def>
                    স্ৱভাৱ:
                        ঘূৰাই "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                ক্ষিপ std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            ভঙ্গ;
        }
    }

    স্থির অন্তঃস্থ unsigned পূর্ণাঙ্ক get_current_device_id()
    {
        ঘূৰাই dev_mgr::instance().current_device_id();
    }

    স্থির অন্তঃস্থ device_ext &get_current_device()
    {
        ঘূৰাই dev_mgr::instance().current_device();
    }

    স্থির অন্তঃস্থ device_ext &get_device(unsigned পূর্ণাঙ্ক id)
    {
        ঘূৰাই dev_mgr::instance().get_device(id);
    }

    স্থির অন্তঃস্থ sycl::queue &get_in_order_queue()
    {
        ঘূৰাই dev_mgr::instance().current_device().in_order_queue();
    }

    স্থির sycl::event
    dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr, size_t size,
                memcpy_direction direction,
                ধ্রুৱ std::vector<sycl::event> &dep_events = {})
    {
        যদি (!size)
            ঘূৰাই sycl::event{};
        ঘূৰাই q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    স্থির অন্তঃস্থ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        ঘূৰাই slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    স্থির অন্তঃস্থ size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        ঘূৰাই slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    স্থির অন্তঃস্থ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ধ্রুৱ std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        শ্ৰেণী host_buffer
        {
            শূন্য *_buf;
            size_t _size;
            sycl::queue &_q;
            ধ্রুৱ std::vector<sycl::event> &_deps; // free operation depends

        সার্ৱজনিক:
            host_buffer(size_t size, sycl::queue &q,
                        ধ্রুৱ std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            শূন্য *get_ptr() ধ্রুৱ { ঘূৰাই _buf; }
            size_t get_size() ধ্রুৱ { ঘূৰাই _size; }
            ~host_buffer()
            {
                যদি (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::মুক্ত(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned ৱর্ণ *to_surface =
            (unsigned ৱর্ণ *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        ধ্রুৱ unsigned ৱর্ণ *from_surface =
            (ধ্রুৱ unsigned ৱর্ণ *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        যদি (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            ঘূৰাই {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        ৱিকল্প (direction)
        {
        পক্ষ host_to_host:
            বাবে (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned ৱর্ণ *to_ptr = to_surface;
                ধ্রুৱ unsigned ৱর্ণ *from_ptr = from_surface;
                যদি (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                অন্যথা
                {
                    বাবে (size_t y = 0; y < size.get(1); ++y)
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
            ভঙ্গ;
        পক্ষ host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            যদি (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            অন্যথা
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
            ভঙ্গ;
        }
        পক্ষ device_to_host:
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
            ভঙ্গ;
        }
        পক্ষ device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<শ্ৰেণী dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        ভঙ্গ;
        স্ৱভাৱ:
            ক্ষিপ std::runtime_error("dpct_memcpy: invalid direction value");
        }
        ঘূৰাই event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    স্থির অন্তঃস্থ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        ঘূৰাই dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    স্থির অন্তঃস্থ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুৱ শূন্য *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        ঘূৰাই dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    অন্তঃস্থ শূন্য gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণাঙ্ক m, পূর্ণাঙ্ক n,
                     পূর্ণাঙ্ক k, ধ্রুৱ শূন্য * alpha, ধ্রুৱ শূন্য * a, library_data_t a_type, পূর্ণাঙ্ক lda, ধ্রুৱ শূন্য * b,
                     library_data_t b_type, পূর্ণাঙ্ক ldb, ধ্রুৱ শূন্য * beta, শূন্য * c, library_data_t c_type, পূর্ণাঙ্ক ldc,
                     library_data_t scaling_type) {
        যদি (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        অন্যথা যদি (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t চাবি =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ৱিকল্প (চাবি)
        {
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<ভিন্ন, ভিন্ন, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<দ্ৱিগুণ, দ্ৱিগুণ, দ্ৱিগুণ, দ্ৱিগুণ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<ভিন্ন>, std::complex<ভিন্ন>,
                              std::complex<ভিন্ন>, std::complex<ভিন্ন>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<দ্ৱিগুণ>, std::complex<দ্ৱিগুণ>,
                              std::complex<দ্ৱিগুণ>, std::complex<দ্ৱিগুণ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
#ifdef __INTEL_MKL__
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ভিন্ন alpha_মান_2 =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ ভিন্ন *>(alpha), q);
            ভিন্ন beta_মান =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ ভিন্ন *>(beta), q);
            sycl::half alpha_half(alpha_মান_2);
            sycl::half beta_half(beta_মান);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ভিন্ন alpha_float =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ std::int32_t *>(alpha), q);
            ভিন্ন beta_float =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            ভঙ্গ;
        }
#endif // __INTEL_MKL__
        স্ৱভাৱ:
            ক্ষিপ std::runtime_error("the combination of data type is unsupported");
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
    অন্তঃস্থ শূন্য gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণাঙ্ক m,
                           পূর্ণাঙ্ক n, পূর্ণাঙ্ক k, ধ্রুৱ শূন্য * alpha, ধ্রুৱ শূন্য * a[], library_data_t a_type, পূর্ণাঙ্ক lda,
                           ধ্রুৱ শূন্য * b[], library_data_t b_type, পূর্ণাঙ্ক ldb, ধ্রুৱ শূন্য * beta, শূন্য * c[],
                           library_data_t c_type, পূর্ণাঙ্ক ldc, পূর্ণাঙ্ক batch_size, library_data_t scaling_type,
                           matrix_info_t<ভিন্ন> * matrix_info) {
        std::uint64_t চাবি =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ৱিকল্প (চাবি)
        {
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ভিন্ন, ভিন্ন, ভিন্ন, ভিন্ন>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<দ্ৱিগুণ, দ্ৱিগুণ, দ্ৱিগুণ, দ্ৱিগুণ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
#ifdef __INTEL_MKL__
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
#endif
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ভিন্ন alpha_float =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ std::int32_t *>(alpha), q);
            ভিন্ন beta_float =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ভিন্ন alpha_মান_2 =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ ভিন্ন *>(alpha), q);
            ভিন্ন beta_মান =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ ভিন্ন *>(beta), q);
            sycl::half alpha_half(alpha_মান_2);
            sycl::half beta_half(beta_মান);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            ভঙ্গ;
        }
        স্ৱভাৱ:
            ক্ষিপ std::runtime_error("the combination of data type is unsupported");
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
    অন্তঃস্থ শূন্য gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণাঙ্ক m,
                           পূর্ণাঙ্ক n, পূর্ণাঙ্ক k, ধ্রুৱ শূন্য * alpha, ধ্রুৱ শূন্য * a, library_data_t a_type, পূর্ণাঙ্ক lda,
                           long long পূর্ণাঙ্ক stride_a, ধ্রুৱ শূন্য * b, library_data_t b_type, পূর্ণাঙ্ক ldb,
                           long long পূর্ণাঙ্ক stride_b, ধ্রুৱ শূন্য * beta, শূন্য * c, library_data_t c_type, পূর্ণাঙ্ক ldc,
                           long long পূর্ণাঙ্ক stride_c, পূর্ণাঙ্ক batch_size, library_data_t scaling_type) {
        যদি (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        অন্যথা যদি (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t চাবি =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ৱিকল্প (চাবি)
        {
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ভিন্ন, ভিন্ন, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<দ্ৱিগুণ, দ্ৱিগুণ, দ্ৱিগুণ, দ্ৱিগুণ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<ভিন্ন>, std::complex<ভিন্ন>,
                                    std::complex<ভিন্ন>, std::complex<ভিন্ন>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<দ্ৱিগুণ>, std::complex<দ্ৱিগুণ>,
                                    std::complex<দ্ৱিগুণ>, std::complex<দ্ৱিগুণ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
#ifdef __INTEL_MKL__
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            ভঙ্গ;
        }
#endif
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ভিন্ন, ভিন্ন>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        পক্ষ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ভিন্ন alpha_মান_2 =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ ভিন্ন *>(alpha), q);
            ভিন্ন beta_মান =
                dpct::get_মান(reinterpret_cast<ধ্রুৱ ভিন্ন *>(beta), q);
            sycl::half alpha_half(alpha_মান_2);
            sycl::half beta_half(beta_মান);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            ভঙ্গ;
        }
        স্ৱভাৱ:
            ক্ষিপ std::runtime_error("the combination of data type is unsupported");
        }
    }

    স্থির অন্তঃস্থ শূন্য
    async_dpct_memcpy(শূন্য *to_ptr, size_t to_pitch, ধ্রুৱ শূন্য *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<সংরচনা err0_tag, পূর্ণাঙ্ক>;
    using err1 = detail::generic_error_type<সংরচনা err1_tag, পূর্ণাঙ্ক>;

    স্থির অন্তঃস্থ শূন্য dpct_free(শূন্য *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <শ্ৰেণী T, memory_region Memory, size_t Dimension> শ্ৰেণী accessor;
    template <শ্ৰেণী T, memory_region Memory> শ্ৰেণী accessor<T, Memory, 3> {
    সার্ৱজনিক:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, ধ্রুৱ sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ধ্রুৱ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ধ্রুৱ accessor_t &acc, ধ্রুৱ sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) ধ্রুৱ {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            ঘূৰাই accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() ধ্রুৱ { ঘূৰাই _data; }

    গুপ্ত:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <শ্ৰেণী T, memory_region Memory> শ্ৰেণী accessor<T, Memory, 2> {
    সার্ৱজনিক:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, ধ্রুৱ sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ধ্রুৱ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ধ্রুৱ accessor_t &acc, ধ্রুৱ sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) ধ্রুৱ {
            ঘূৰাই _data + _range.get(1) * index;
        }

        pointer_t get_ptr() ধ্রুৱ { ঘূৰাই _data; }

    গুপ্ত:
        pointer_t _data;
        sycl::range<2> _range;
    };

    নামাকাশ detail {
        /// Device variable with address space of shared, global or constant.
        template <শ্ৰেণী T, memory_region Memory, size_t Dimension> শ্ৰেণী device_memory {
        সার্ৱজনিক:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using মান_t = typename detail::memory_traits<Memory, T>::মান_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(ধ্রুৱ sycl::range<Dimension> &in_range,
                        std::initializer_list<মান_t> &&init_list)
                : device_memory(in_range) {
                প্রতিজ্ঞা(init_list.size() <= in_range.size());
                _host_ptr = (মান_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * পরিমাণ(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                ধ্রুৱ typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<মান_t>> &&init_list)
                : device_memory(in_range) {
                প্রতিজ্ঞা(init_list.size() <= in_range[0]);
                _host_ptr = (মান_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                বাবে (auto sub_list : init_list) {
                    প্রতিজ্ঞা(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * পরিমাণ(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(ধ্রুৱ sycl::range<Dimension> &range_in)
                : _size(range_in.size() * পরিমাণ(T)), _range(range_in),
                _reference(মিছা), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <শ্ৰেণী... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                যদি (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                যদি (_host_ptr)
                    std::মুক্ত(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            শূন্য init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            শূন্য init(sycl::queue &q) {
                যদি (_device_ptr)
                    ঘূৰাই;
                যদি (!_size)
                    ঘূৰাই;
                allocate_device(q);
                যদি (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            শূন্য assign(মান_t *src, size_t size) {
                এতত্->~device_memory();
                নূতন (এতত্) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            মান_t *get_ptr() { ঘূৰাই get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            মান_t *get_ptr(sycl::queue &q) {
                init(q);
                ঘূৰাই _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { ঘূৰাই _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                ঘূৰাই _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                ঘূৰাই dpct_accessor_t((T *)_device_ptr, _range);
            }

        গুপ্ত:
            device_memory(মান_t *memory_ptr, size_t size)
                : _size(size), _range(size / পরিমাণ(T)), _reference(সঁচা),
                _device_ptr(memory_ptr) {}

            শূন্য allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                যদি (Memory == shared) {
                    _device_ptr = (মান_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    ঘূৰাই;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                যদি (Memory == constant) {
                    _device_ptr = (মান_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    ঘূৰাই;
                }
        #endif
        #endif
                _device_ptr = (মান_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            তার্কিক _reference;
            মান_t *_host_ptr;
            মান_t *_device_ptr;
        };
        template <শ্ৰেণী T, memory_region Memory>
        শ্ৰেণী device_memory<T, Memory, 0> : সার্ৱজনিক device_memory<T, Memory, 1> {
        সার্ৱজনিক:
            using base = device_memory<T, Memory, 1>;
            using মান_t = typename base::মান_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(ধ্রুৱ মান_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <শ্ৰেণী T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <শ্ৰেণী T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <শ্ৰেণী T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    অন্তঃস্থ T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ঘূৰাই atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    অন্তঃস্থ T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ঘূৰাই atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    অন্তঃস্থ T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    ৱিকল্প (memoryOrder) {
        পক্ষ sycl::memory_order::relaxed:
            ঘূৰাই atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        পক্ষ sycl::memory_order::acq_rel:
            ঘূৰাই atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        পক্ষ sycl::memory_order::seq_cst:
            ঘূৰাই atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        স্ৱভাৱ:
            প্রতিজ্ঞা(মিছা && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    অন্তঃস্থ T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    অন্তঃস্থ unsigned পূর্ণাঙ্ক byte_level_permute(
        unsigned পূর্ণাঙ্ক a, unsigned পূর্ণাঙ্ক b, unsigned পূর্ণাঙ্ক s) {
      unsigned পূর্ণাঙ্ক ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      ঘূৰাই ret;
    }

    অন্তঃস্থ uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, পূর্ণাঙ্ক mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      যদি (mode >= 1 && mode <= 6) {
        ঘূৰাই byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } অন্যথা যদি (!mode) {
        ঘূৰাই byte_level_permute(low32, high32, sel);
      }
      ঘূৰাই 0;
    }

    template <পূর্ণাঙ্ক n_nondefault_params, পূর্ণাঙ্ক n_default_params, typename T>
    শ্ৰেণী args_selector;

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
    template <পূর্ণাঙ্ক n_nondefault_params, পূর্ণাঙ্ক n_default_params, typename R,
              typename... Ts>
    শ্ৰেণী args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      গুপ্ত:
        শূন্য **kernel_params;
        ৱর্ণ *args_buffer;

        template <পূর্ণাঙ্ক i> স্থির constexpr পূর্ণাঙ্ক account_for_default_params() {
            constexpr পূর্ণাঙ্ক n_total_params = পরিমাণ...(Ts);
            যদি constexpr (i >= n_nondefault_params) {
                ঘূৰাই n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } অন্যথা {
                ঘূৰাই i;
            }
        }

      সার্ৱজনিক:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <পূর্ণাঙ্ক i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        স্থির constexpr পূর্ণাঙ্ক params_অঙ্ক = পরিমাণ...(Ts);

      গুপ্ত:
        template <পূর্ণাঙ্ক i> স্থির constexpr পূর্ণাঙ্ক get_offset() {
            যদি constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                ঘূৰাই 0;
            } অন্যথা {
                constexpr পূর্ণাঙ্ক prev_off = get_offset<i - 1>();
                constexpr পূর্ণাঙ্ক prev_past_end =
                    prev_off + পরিমাণ(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                যদি constexpr (prev_past_end % alignof(T) == 0) {
                    ঘূৰাই prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                অন্যথা {
                    ঘূৰাই prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        স্থির ৱর্ণ *get_args_buffer(শূন্য **extra) {
            যদি (!extra)
                ঘূৰাই nullptr;
            বাবে (; (std::size_t)*extra != 0; ++extra) {
                যদি ((std::size_t)*extra == 1) {
                    ঘূৰাই static_cast<ৱর্ণ *>(*(extra + 1));
                }
            }
            ঘূৰাই nullptr;
        }

      সার্ৱজনিক:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(শূন্য **kernel_params, শূন্য **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <পূর্ণাঙ্ক i> arg_type<i> &get() {
            যদি (kernel_params) {
                ঘূৰাই *static_cast<arg_type<i> *>(kernel_params[i]);
            } অন্যথা {
                ঘূৰাই *reinterpret_cast<arg_type<i> *>(args_buffer +
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
    শ্ৰেণী kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        স্থির শূন্য launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        স্থির শূন্য set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned পূর্ণাঙ্ক local_mem_size,
                                         queue_ptr que) {
            যদি (que) {
                _que = que;
            } অন্যথা {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        স্থির অন্তঃস্থ std::mutex kernel_function_ptr_map_mutex;

      সার্ৱজনিক:
        /// Variables for storing execution configuration.
        স্থির অন্তঃস্থ thread_local sycl::queue *_que = nullptr;
        স্থির অন্তঃস্থ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        স্থির অন্তঃস্থ thread_local unsigned পূর্ণাঙ্ক _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        স্থির অন্তঃস্থ std::map<
            ধ্রুৱ শূন্য *,
            std::function<শূন্য(dim3, dim3, শূন্য **, unsigned পূর্ণাঙ্ক, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        স্থির শূন্য register_kernel_ptr(
            ধ্রুৱ শূন্য *func,
            std::function<শূন্য(dim3, dim3, শূন্য **, unsigned পূর্ণাঙ্ক, queue_ptr)>
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
        স্থির std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, শূন্য>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned পূর্ণাঙ্ক local_mem_size, queue_ptr que, ArgsT... args) {
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
        স্থির শূন্য launch(ধ্রুৱ শূন্য *func, dim3 group_range, dim3 local_range,
                           শূন্য **args, unsigned পূর্ণাঙ্ক local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            যদি (Iter == kernel_function_ptr_map.end()) {
                ক্ষিপ std::runtime_error("dpct::launch() : no registered "
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
        স্থির std::enable_if_t<std::is_function_v<FuncT>, শূন্য>
        launch(FuncT *func, dim3 group_range, dim3 local_range, শূন্য **args,
               unsigned পূর্ণাঙ্ক local_mem_size, queue_ptr que) {
            constexpr size_t p_অঙ্ক = args_selector<0, 0, FuncT>::params_অঙ্ক;
            set_execution_config(group_range, local_range, local_mem_size, que);
            args_selector<p_অঙ্ক, p_অঙ্ক, FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_অঙ্ক>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    template <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        পূর্ণাঙ্ক remote_local_id,
        পূর্ণাঙ্ক logical_sub_group_size = 32) {
      unsigned পূর্ণাঙ্ক start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      ঘূৰাই sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    শূন্য ldmatrix(uintptr_t addr, T* m, তার্কিক trans = মিছা, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      পূর্ণাঙ্ক lane = sg.get_local_linear_id();

      পূর্ণাঙ্ক lane_group8_row = lane / 8;
      পূর্ণাঙ্ক lane_group8_col = lane % 8;

      যদি (!trans) {
        // calculate the source lane
        পূর্ণাঙ্ক src_lane = 2 * lane_group8_row;
        যদি (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } অন্যথা {
        // calculate the source lane
        পূর্ণাঙ্ক src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        পূর্ণাঙ্ক index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    শূন্য ldmatrix(uintptr_t addr, T* m1, T* m2, তার্কিক trans = মিছা) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    শূন্য ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, তার্কিক trans = মিছা) {
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
    সংরচনা MMAType {
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
    template <পূর্ণাঙ্ক M, পূর্ণাঙ্ক N, পূর্ণাঙ্ক K, typename ABType, typename CDType>
    শূন্য mma(
        চঞ্চল শূন্য** d_mat_frag,
        শূন্য* a_mat_frag,
        শূন্য* b_mat_frag,
        শূন্য* c_mat_frag) {
      auto d = reinterpret_cast<চঞ্চল CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      পূর্ণাঙ্ক lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      যদি constexpr (M == 8 && N == 8 && K == 4) {
        যদি constexpr (std::is_floating_point_v<CDType>) {
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
          পূর্ণাঙ্ক r_ind = (lane % 2) ? 1 : 0;
          পূর্ণাঙ্ক c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
                static_cast<ভিন্ন>(ra[r_ind]) * static_cast<ভিন্ন>(rb[c_ind]);
            *d[1] += static_cast<ভিন্ন>(ra[r_ind]) *
                     static_cast<ভিন্ন>(rb[c_ind + 1]);
            *d[2] += static_cast<ভিন্ন>(ra[r_ind + 2]) *
                     static_cast<ভিন্ন>(rb[c_ind]);
            *d[3] += static_cast<ভিন্ন>(ra[r_ind + 2]) *
                     static_cast<ভিন্ন>(rb[c_ind + 1]);

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
                static_cast<ভিন্ন>(ra[r_ind]) * static_cast<ভিন্ন>(rb[c_ind]);
            *d[5] += static_cast<ভিন্ন>(ra[r_ind]) *
                     static_cast<ভিন্ন>(rb[c_ind + 1]);
            *d[6] += static_cast<ভিন্ন>(ra[r_ind + 2]) *
                     static_cast<ভিন্ন>(rb[c_ind]);
            *d[7] += static_cast<ভিন্ন>(ra[r_ind + 2]) *
                     static_cast<ভিন্ন>(rb[c_ind + 1]);
          }
        }
      } অন্যথা যদি constexpr (M == 8 && N == 8 && K == 16) {
        যদি constexpr (std::is_integral_v<ABType>) {
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
          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } অন্যথা যদি constexpr (M == 16 && N == 8 && K == 8) {
        যদি constexpr (std::is_floating_point_v<CDType>) {
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
          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
            বাবে (পূর্ণাঙ্ক j = 0; j < 2; j++) {
              *d[0] += static_cast<ভিন্ন>(ra[j]) * static_cast<ভিন্ন>(rb[j]);
              *d[1] +=
                  static_cast<ভিন্ন>(ra[j]) * static_cast<ভিন্ন>(rb[j + 2]);
              *d[2] +=
                  static_cast<ভিন্ন>(ra[j + 2]) * static_cast<ভিন্ন>(rb[j]);
              *d[3] +=
                  static_cast<ভিন্ন>(ra[j + 2]) * static_cast<ভিন্ন>(rb[j + 2]);
            }
          }
        }
      } অন্যথা যদি constexpr (M == 16 && N == 8 && K == 16) {
        যদি constexpr (std::is_floating_point_v<CDType>) {
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
          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } অন্যথা যদি constexpr (std::is_integral_v<ABType>) {
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
          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
            বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } অন্যথা যদি constexpr (M == 16 && N == 8 && K == 32) {
        যদি constexpr (std::is_integral_v<ABType>) {
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
          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          বাবে (পূর্ণাঙ্ক i = 0; i < 4; i++) {
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
            বাবে (পূর্ণাঙ্ক j = 0; j < 4; j++) {
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

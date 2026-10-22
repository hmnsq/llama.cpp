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

#অন্তর্ভাব <cstdlib>
#অন্তর্ভাব <iostream>
#অন্তর্ভাব <map>

#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব <sycl/half_type.hpp>
#অন্তর্ভাব <oneapi/mkl.hpp>

#যদি defined(__linux__)
#অন্তর্ভাব <sys/mman.h>
#নতুবা_যদি defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#অন্তর্ভাব <windows.h>
#অন্যথা
#error "Only support Windows and Linux."
#endif

#যদি defined(__linux__)
#অন্তর্ভাব <unistd.h>
#অন্তর্ভাব <sys/syscall.h>
#endif
#যদি defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#অন্তর্ভাব <windows.h>
#endif


#define DPCT_COMPATIBILITY_অস্থায়ী (900)

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

[[noreturn]] অন্তঃরেখ শূন্য _abort(ধ্রুবক অক্ষর * str) {
    std::cerr << str << std::endl;
    std::abort();
}

অন্তঃরেখ std::string get_device_type_name(ধ্রুবক sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    নির্বাচন (DeviceType) {
    ক্ষেত্র sycl::info::device_type::cpu:
        ফেরত "cpu";
    ক্ষেত্র sycl::info::device_type::gpu:
        ফেরত "gpu";
    ক্ষেত্র sycl::info::device_type::host:
        ফেরত "host";
    ক্ষেত্র sycl::info::device_type::accelerator:
        ফেরত "acc";
    পূর্বনির্ধারিত:
        ফেরত "unknown";
    }
}

অন্তঃরেখ std::string get_device_backend_and_type(ধ্রুবক sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    ফেরত device_type.str();
}

template <typename Ts> কাঠামো matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     মান_info[2];
    std::int64_t           আকার_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

নামাকাশ dpct
{
    প্রকারলেখ sycl::queue *queue_ptr;
    প্রকারলেখ sycl::event *event_ptr;
    প্রকারলেখ অক্ষর *device_ptr;
    প্রকারলেখ uint8_t byte_t;
    প্রকারলেখ sycl::বাফার<byte_t> বাফার_t;

    /// SYCL default exception handler
    অন্তঃরেখ auto exception_handler = [](sycl::exception_list exceptions)
    {
        জন্য (std::exception_ptr ধ্রুবক &e : exceptions)
        {
            চেষ্টা
            {
                std::rethrow_exception(e);
            }
            ধরো (sycl::exception ধ্রুবক &e)
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

    গণনা শ্রেণী library_তথ্য_t : unsigned অক্ষর
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
        library_তথ্য_t_আকার
    };

    template <typename T>
    কাঠামো তথ্য_Type_2
    {
        using T2 = T;
    };
    template <typename T>
    কাঠামো তথ্য_Type_2<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    স্থির শূন্য destroy_event(event_ptr event)
    {
        delete event;
    }

    স্থির অন্তঃরেখ unsigned পূর্ণসংখ্যা get_tid()
    {
#যদি defined(__linux__)
        ফেরত syscall(SYS_gettid);
#নতুবা_যদি defined(_WIN64)
        ফেরত GetCurrentThreadId();
#অন্যথা
#error "Only support Windows and Linux."
#endif
    }

    নামাকাশ detail
    {
        স্থির শূন্য get_version(ধ্রুবক sycl::device &dev, পূর্ণসংখ্যা &major, পূর্ণসংখ্যা &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            যখন (i < ver.size()) {
              যদি (isdigit(ver[i]))
                বিরতি;
              i++;
            }
            major = std::stoi(&(ver[i]));
            যখন (i < ver.size()) {
              যদি (ver[i] == '.')
                বিরতি;
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
        শ্রেণী generic_error_type
        {
        সর্বজনীন:
            generic_error_type() = পূর্বনির্ধারিত;
            generic_error_type(T value) : value{value} {}
            operator T() ধ্রুবক { ফেরত value; }

        ব্যক্তিগত:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    শ্রেণী dim3 {
        সর্বজনীন:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(ধ্রুবক sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() ধ্রুবক { ফেরত sycl::range<3>(z, y, x); }
    }; // namespace dim3

    অন্তঃরেখ dim3 operator*(ধ্রুবক dim3 &a, ধ্রুবক dim3 &b) {
    ফেরত dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    শ্রেণী pitched_তথ্য
    {
    সর্বজনীন:
        pitched_তথ্য() : pitched_তথ্য(nullptr, 0, 0, 0) {}
        pitched_তথ্য(শূন্য *তথ্য, size_t pitch, size_t x, size_t y)
            : তথ্য_2(তথ্য), _pitch(pitch), _x(x), _y(y) {}

        শূন্য *get_তথ্য_ptr() { ফেরত তথ্য_2; }
        শূন্য set_তথ্য_ptr(শূন্য *তথ্য) { তথ্য_2 = তথ্য; }

        size_t get_pitch() { ফেরত _pitch; }
        শূন্য set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { ফেরত _x; }
        শূন্য set_x(size_t x) { _x = x; }

        size_t get_y() { ফেরত _y; }
        শূন্য set_y(size_t y) { _y = y; }

    ব্যক্তিগত:
        শূন্য *তথ্য_2;
        size_t _pitch, _x, _y;
    };

    শ্রেণী device_info
    {
    সর্বজনীন:
        // get interface
        ধ্রুবক অক্ষর *get_name() ধ্রুবক { ফেরত _name; }
        অক্ষর *get_name() { ফেরত _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, পূর্ণসংখ্যা *>,
                                   পূর্ণসংখ্যা> = 0>
        auto get_max_work_item_sizes() ধ্রুবক
        {
            যদি constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ফেরত sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            অন্যথা
            {
                ফেরত _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, পূর্ণসংখ্যা *>,
                                   পূর্ণসংখ্যা> = 0>
        auto get_max_work_item_sizes()
        {
            যদি constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ফেরত sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            অন্যথা
            {
                ফেরত _max_work_item_sizes_i;
            }
        }
        যৌক্তিক get_host_unified_memory() ধ্রুবক { ফেরত _host_unified_memory; }
        পূর্ণসংখ্যা get_major_version() ধ্রুবক { ফেরত _major; }
        পূর্ণসংখ্যা get_minor_version() ধ্রুবক { ফেরত _minor; }
        পূর্ণসংখ্যা get_integrated() ধ্রুবক { ফেরত _integrated; }
        পূর্ণসংখ্যা get_max_clock_frequency() ধ্রুবক { ফেরত _frequency; }
        পূর্ণসংখ্যা get_max_compute_units() ধ্রুবক { ফেরত _max_compute_units; }
        পূর্ণসংখ্যা get_max_work_group_আকার() ধ্রুবক { ফেরত max_work_group_আকার_2; }
        পূর্ণসংখ্যা get_max_sub_group_আকার() ধ্রুবক { ফেরত max_sub_group_আকার; }
        পূর্ণসংখ্যা get_max_work_বস্তুসমূহ_per_compute_unit() ধ্রুবক
        {
            ফেরত max_work_বস্তুসমূহ_per_compute_unit;
        }
        পূর্ণসংখ্যা get_max_register_আকার_per_work_group() ধ্রুবক
        {
            ফেরত max_register_আকার_per_work_group;
        }
        template <typename NDRange_আকার_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_আকার_Ty, size_t *> ||
                                       std::is_same_v<NDRange_আকার_Ty, পূর্ণসংখ্যা *>,
                                   পূর্ণসংখ্যা> = 0>
        auto get_max_nd_range_আকার() ধ্রুবক
        {
            যদি constexpr (std::is_same_v<NDRange_আকার_Ty, size_t *>)
                ফেরত max_nd_range_আকার;
            অন্যথা
                ফেরত max_nd_range_আকার_i;
        }
        template <typename NDRange_আকার_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_আকার_Ty, size_t *> ||
                                       std::is_same_v<NDRange_আকার_Ty, পূর্ণসংখ্যা *>,
                                   পূর্ণসংখ্যা> = 0>
        auto get_max_nd_range_আকার()
        {
            যদি constexpr (std::is_same_v<NDRange_আকার_Ty, size_t *>)
                ফেরত max_nd_range_আকার;
            অন্যথা
                ফেরত max_nd_range_আকার_i;
        }
        size_t get_global_mem_আকার() ধ্রুবক { ফেরত global_mem_আকার_2; }
        size_t get_local_mem_আকার() ধ্রুবক { ফেরত local_mem_আকার; }
        size_t get_max_mem_alloc_আকার() ধ্রুবক { ফেরত max_mem_alloc_আকার; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned পূর্ণসংখ্যা get_memory_clock_rate() ধ্রুবক { ফেরত _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned পূর্ণসংখ্যা get_memory_bus_width() ধ্রুবক { ফেরত _memory_bus_width; }
        uint32_t get_device_id() ধ্রুবক { ফেরত _device_id; }
        std::array<unsigned অক্ষর, 16> get_uuid() ধ্রুবক { ফেরত _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned পূর্ণসংখ্যা get_global_mem_cache_আকার() ধ্রুবক
        {
            ফেরত global_mem_cache_আকার;
        }

        // set interface
        শূন্য set_name(ধ্রুবক অক্ষর *name)
        {
            size_t দৈর্ঘ্য = strlen(name);
            যদি (দৈর্ঘ্য < 256)
            {
                std::memcpy(_name, name, দৈর্ঘ্য + 1);
            }
            অন্যথা
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        শূন্য set_max_work_item_sizes(ধ্রুবক sycl::range<3> max_work_item_sizes)
        {
            জন্য (পূর্ণসংখ্যা i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] শূন্য
        set_max_work_item_sizes(ধ্রুবক sycl::id<3> max_work_item_sizes)
        {
            জন্য (পূর্ণসংখ্যা i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        শূন্য set_host_unified_memory(যৌক্তিক host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        শূন্য set_major_version(পূর্ণসংখ্যা major) { _major = major; }
        শূন্য set_minor_version(পূর্ণসংখ্যা minor) { _minor = minor; }
        শূন্য set_integrated(পূর্ণসংখ্যা integrated) { _integrated = integrated; }
        শূন্য set_max_clock_frequency(পূর্ণসংখ্যা frequency) { _frequency = frequency; }
        শূন্য set_max_compute_units(পূর্ণসংখ্যা max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        শূন্য set_global_mem_আকার(size_t global_mem_আকার)
        {
            global_mem_আকার_2 = global_mem_আকার;
        }
        শূন্য set_local_mem_আকার(size_t local_mem_আকার_2)
        {
            local_mem_আকার = local_mem_আকার_2;
        }
        শূন্য set_max_mem_alloc_আকার(size_t max_mem_alloc_আকার_2)
        {
            max_mem_alloc_আকার = max_mem_alloc_আকার_2;
        }
        শূন্য set_max_work_group_আকার(পূর্ণসংখ্যা max_work_group_আকার)
        {
            max_work_group_আকার_2 = max_work_group_আকার;
        }
        শূন্য set_max_sub_group_আকার(পূর্ণসংখ্যা max_sub_group_আকার_2)
        {
            max_sub_group_আকার = max_sub_group_আকার_2;
        }
        শূন্য
        set_max_work_বস্তুসমূহ_per_compute_unit(পূর্ণসংখ্যা max_work_বস্তুসমূহ_per_compute_unit_2)
        {
            max_work_বস্তুসমূহ_per_compute_unit = max_work_বস্তুসমূহ_per_compute_unit_2;
        }
        শূন্য set_max_nd_range_আকার(পূর্ণসংখ্যা max_nd_range_আকার_2[])
        {
            জন্য (পূর্ণসংখ্যা i = 0; i < 3; i++)
            {
                max_nd_range_আকার[i] = max_nd_range_আকার_2[i];
                max_nd_range_আকার_i[i] = max_nd_range_আকার_2[i];
            }
        }
        শূন্য set_memory_clock_rate(unsigned পূর্ণসংখ্যা memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        শূন্য set_memory_bus_width(unsigned পূর্ণসংখ্যা memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        শূন্য
        set_max_register_আকার_per_work_group(পূর্ণসংখ্যা max_register_আকার_per_work_group_2)
        {
            max_register_আকার_per_work_group = max_register_আকার_per_work_group_2;
        }
        শূন্য set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        শূন্য set_uuid(std::array<unsigned অক্ষর, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        শূন্য set_global_mem_cache_আকার(unsigned পূর্ণসংখ্যা global_mem_cache_আকার_2)
        {
            global_mem_cache_আকার = global_mem_cache_আকার_2;
        }

    ব্যক্তিগত:
        অক্ষর _name[256];
        পূর্ণসংখ্যা _max_work_item_sizes_i[3];
        যৌক্তিক _host_unified_memory = মিথ্যা;
        পূর্ণসংখ্যা _major;
        পূর্ণসংখ্যা _minor;
        পূর্ণসংখ্যা _integrated = 0;
        পূর্ণসংখ্যা _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned পূর্ণসংখ্যা _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned পূর্ণসংখ্যা _memory_bus_width = 64;
        unsigned পূর্ণসংখ্যা global_mem_cache_আকার;
        পূর্ণসংখ্যা _max_compute_units;
        পূর্ণসংখ্যা max_work_group_আকার_2;
        পূর্ণসংখ্যা max_sub_group_আকার;
        পূর্ণসংখ্যা max_work_বস্তুসমূহ_per_compute_unit;
        পূর্ণসংখ্যা max_register_আকার_per_work_group;
        size_t global_mem_আকার_2;
        size_t local_mem_আকার;
        size_t max_mem_alloc_আকার;
        size_t max_nd_range_আকার[3];
        পূর্ণসংখ্যা max_nd_range_আকার_i[3];
        uint32_t _device_id;
        std::array<unsigned অক্ষর, 16> _uuid;
    };

    স্থির পূর্ণসংখ্যা get_major_version(ধ্রুবক sycl::device &dev)
    {
        পূর্ণসংখ্যা major, minor;
        detail::get_version(dev, major, minor);
        ফেরত major;
    }

    স্থির পূর্ণসংখ্যা get_minor_version(ধ্রুবক sycl::device &dev)
    {
        পূর্ণসংখ্যা major, minor;
        detail::get_version(dev, major, minor);
        ফেরত minor;
    }

    স্থির শূন্য get_device_info(device_info &out, ধ্রুবক sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        পূর্ণসংখ্যা major, minor;
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
        prop.set_max_work_group_আকার(
            dev.get_info<sycl::info::device::max_work_group_আকার>());
        prop.set_global_mem_আকার(dev.get_info<sycl::info::device::global_mem_আকার>());
        prop.set_local_mem_আকার(dev.get_info<sycl::info::device::local_mem_আকার_2>());
        prop.set_max_mem_alloc_আকার(dev.get_info<sycl::info::device::max_mem_alloc_আকার_2>());

#যদি (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        যদি (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned পূর্ণসংখ্যা tmp =
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
#নতুবা_যদি defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate পূর্বনির্ধারিত value. \
        Use 64 bits as memory_bus_width পূর্বনির্ধারিত value.")
#অন্যথা
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate পূর্বনির্ধারিত value. \
        Use 64 bits as memory_bus_width পূর্বনির্ধারিত value."
#endif

        size_t max_sub_group_আকার_2 = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        জন্য (ধ্রুবক auto &sub_group_আকার : sub_group_sizes)
        {
            যদি (max_sub_group_আকার_2 < sub_group_আকার)
                max_sub_group_আকার_2 = sub_group_আকার;
        }

        prop.set_max_sub_group_আকার(max_sub_group_আকার_2);

        prop.set_max_work_বস্তুসমূহ_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_আকার>());
        পূর্ণসংখ্যা max_nd_range_আকার_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_আকার(max_nd_range_আকার_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_আকার_per_work_group(65536);

        prop.set_global_mem_cache_আকার(
            dev.get_info<sycl::info::device::global_mem_cache_আকার_2>());
        out = prop;
    }

    /// dpct device extension
    শ্রেণী device_ext : সর্বজনীন sycl::device {
      প্রকারলেখ std::mutex mutex_type;

     সর্বজনীন:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(ধ্রুবক sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      পূর্ণসংখ্যা is_native_atomic_supported() { ফেরত 0; }
      পূর্ণসংখ্যা get_major_version() ধ্রুবক { ফেরত dpct::get_major_version(*এই); }

      পূর্ণসংখ্যা get_minor_version() ধ্রুবক { ফেরত dpct::get_minor_version(*এই); }

      পূর্ণসংখ্যা get_max_compute_units() ধ্রুবক {
        ফেরত get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      পূর্ণসংখ্যা get_max_clock_frequency() ধ্রুবক {
        ফেরত get_device_info().get_max_clock_frequency();
      }

      পূর্ণসংখ্যা get_integrated() ধ্রুবক { ফেরত get_device_info().get_integrated(); }

      পূর্ণসংখ্যা get_max_sub_group_আকার() ধ্রুবক {
        ফেরত get_device_info().get_max_sub_group_আকার();
      }

      পূর্ণসংখ্যা get_max_register_আকার_per_work_group() ধ্রুবক {
        ফেরত get_device_info().get_max_register_আকার_per_work_group();
      }

      পূর্ণসংখ্যা get_max_work_group_আকার() ধ্রুবক {
        ফেরত get_device_info().get_max_work_group_আকার();
      }

      পূর্ণসংখ্যা get_mem_base_addr_align() ধ্রুবক {
        ফেরত get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_আকার() ধ্রুবক {
        ফেরত get_device_info().get_global_mem_আকার();
      }

      size_t get_max_mem_alloc_আকার() ধ্রুবক {
        ফেরত get_device_info().get_max_mem_alloc_আকার();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      শূন্য get_memory_info(size_t &free_memory, size_t &যোগ_memory) {
        যোগ_memory = get_device_info().get_global_mem_আকার();
        ধ্রুবক অক্ষর *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#যদি (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        যদি (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = যোগ_memory;
        } অন্যথা {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#অন্যথা
        std::cerr << warning_info << std::endl;
        free_memory = যোগ_memory;
#যদি defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#অন্যথা
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      শূন্য get_device_info(device_info &out) ধ্রুবক {
        dpct::get_device_info(out, *এই);
      }

      device_info get_device_info() ধ্রুবক {
        device_info prop;
        dpct::get_device_info(prop, *এই);
        ফেরত prop;
      }

      শূন্য reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { ফেরত _q_in_order; }

      sycl::queue &out_of_order_queue() { ফেরত _q_out_of_order; }

      sycl::queue &default_queue() { ফেরত in_order_queue(); }

      শূন্য queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        জন্য (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(যৌক্তিক enable_exception_handler = মিথ্যা) {
        ফেরত create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               যৌক্তিক enable_exception_handler = মিথ্যা) {
        ফেরত create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(যৌক্তিক enable_exception_handler = মিথ্যা) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ফেরত create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        যৌক্তিক enable_exception_handler = মিথ্যা) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ফেরত create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          যৌক্তিক enable_exception_handler = মিথ্যা) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ফেরত create_queue_impl(enable_exception_handler);
      }

      শূন্য destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ধ্রুবক sycl::queue &q) -> যৌক্তিক
                                    {
                                        ফেরত q == queue;
                                    }),
                    _queues.end());
      }
      শূন্য set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() ধ্রুবক {
        std::lock_guard<mutex_type> lock(m_mutex);
        ফেরত _saved_queue;
      }

     ব্যক্তিগত:
      শূন্য clear_queues() { _queues.clear(); }

      শূন্য init_queues() {
        _q_in_order =
            create_queue_impl(সত্য, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(সত্য);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <শ্রেণী... Properties>
      sycl::queue create_queue_impl(যৌক্তিক enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        যদি (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *এই, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        ফেরত _queues.back();
      }

      template <শ্রেণী... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    যৌক্তিক enable_exception_handler,
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

        ফেরত _queues.back();
      }

      শূন্য get_version(পূর্ণসংখ্যা &major, পূর্ণসংখ্যা &minor) ধ্রুবক {
        detail::get_version(*এই, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    শ্রেণী dev_mgr
    {
    সর্বজনীন:
        device_ext &current_device()
        {
            unsigned পূর্ণসংখ্যা dev_id = current_device_id();
            check_id(dev_id);
            ফেরত *_devs[dev_id];
        }
        device_ext &cpu_device() ধ্রুবক
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            যদি (_cpu_device == -1)
            {
                ছোড়ো std::runtime_error("no valid cpu device");
            }
            অন্যথা
            {
                ফেরত *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned পূর্ণসংখ্যা id) ধ্রুবক
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            ফেরত *_devs[id];
        }
        unsigned পূর্ণসংখ্যা current_device_id() ধ্রুবক
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            যদি (it != _thread2dev_map.end())
                ফেরত it->second;
            ফেরত DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        শূন্য select_device(unsigned পূর্ণসংখ্যা id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned পূর্ণসংখ্যা device_গুনতি() { ফেরত _devs.size(); }

        unsigned পূর্ণসংখ্যা get_device_id(ধ্রুবক sycl::device &dev)
        {
            unsigned পূর্ণসংখ্যা id = 0;
            জন্য (auto &dev_item : _devs)
            {
                যদি (*dev_item == dev)
                {
                    ফেরত id;
                }
                id++;
            }
            ফেরত -1;
        }

        অন্তঃরেখ std::string get_preferred_gpu_platform_name() {
            std::string ফলাফল;

            std::string filter = "";
            অক্ষর* env = getenv("ONEAPI_DEVICE_SELECTOR");
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
                    ছোড়ো std::runtime_error("invalid device filter: " + std::string(env));
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

            জন্য (ধ্রুবক auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](ধ্রুবক sycl::device& d) {
                    ফেরত d.is_gpu();
                });

                যদি (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    চালু;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_নিচের_case;
                platform_name_নিচের_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_নিচের_case.begin(), ::tolower);

                যদি (platform_name_নিচের_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    চালু;
                }

                ফলাফল = platform_name;
            }

            যদি (ফলাফল.empty())
                ছোড়ো std::runtime_error("can not find preferred GPU platform");

            ফেরত ফলাফল;
        }

        template <শ্রেণী DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<পূর্ণসংখ্যা, DeviceSelector, ধ্রুবক sycl::device &>>
        select_device(ধ্রুবক DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned পূর্ণসংখ্যা selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        স্থির dev_mgr &instance()
        {
            স্থির dev_mgr d_m;
            ফেরত d_m;
        }
        dev_mgr(ধ্রুবক dev_mgr &) = delete;
        dev_mgr &operator=(ধ্রুবক dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    ব্যক্তিগত:
        mutable std::recursive_mutex m_mutex;
        স্থির যৌক্তিক compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            যদি(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) ফেরত সত্য;
            যদি(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) ফেরত মিথ্যা;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            ফেরত prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        স্থির পূর্ণসংখ্যা convert_backend_সূচক(std::string & backend) {
            যদি (backend == "ext_oneapi_level_zero:gpu") ফেরত 0;
            যদি (backend == "opencl:gpu") ফেরত 1;
            যদি (backend == "ext_oneapi_cuda:gpu") ফেরত 2;
            যদি (backend == "ext_oneapi_hip:gpu") ফেরত 3;
            যদি (backend == "opencl:cpu") ফেরত 4;
            যদি (backend == "opencl:acc") ফেরত 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        স্থির যৌক্তিক compare_backend(std::string &backend1, std::string &backend2) {
            ফেরত convert_backend_সূচক(backend1) < convert_backend_সূচক(backend2);
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

            যখন (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                যদি (platform_name.compare(preferred_platform_name) != 0) {
                    চালু;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                জন্য (ধ্রুবক auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> চাবিগুলি;
            জন্য(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                চাবিগুলি.push_back(it->first);
            }
            std::sort(চাবিগুলি.begin(), চাবিগুলি.end(), compare_backend);

            জন্য (auto &চাবি : চাবিগুলি) {
                std::vector<sycl::device> devs = backend_devices[চাবি];
                std::sort(devs.begin(), devs.end(), compare_dev);
                জন্য (ধ্রুবক auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            জন্য (auto &dev : sycl_all_devs)
            {
                যদি (dev == default_device)
                {
                    চালু;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                যদি (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        শূন্য check_id(unsigned পূর্ণসংখ্যা id) ধ্রুবক
        {
            যদি (id >= _devs.size())
            {
                ছোড়ো std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ধ্রুবক unsigned পূর্ণসংখ্যা DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned পূর্ণসংখ্যা, unsigned পূর্ণসংখ্যা> _thread2dev_map;
        পূর্ণসংখ্যা _cpu_device = -1;
    };

    স্থির অন্তঃরেখ sycl::queue &get_default_queue()
    {
        ফেরত dev_mgr::instance().current_device().default_queue();
    }

    নামাকাশ detail
    {
        গণনা শ্রেণী pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        স্থির pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              ধ্রুবক শূন্য *ptr)
        {
            নির্বাচন (sycl::get_pointer_type(ptr, q.get_context()))
            {
            ক্ষেত্র sycl::usm::alloc::unknown:
                ফেরত pointer_access_attribute::host_only;
            ক্ষেত্র sycl::usm::alloc::device:
                ফেরত pointer_access_attribute::device_only;
            ক্ষেত্র sycl::usm::alloc::shared:
            ক্ষেত্র sycl::usm::alloc::host:
                ফেরত pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        অন্তঃরেখ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned অক্ষর)library_তথ্য_t::library_তথ্য_t_আকার <=
                              std::numeric_limits<unsigned অক্ষর>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_তথ্য_t>, "Unsupported ArgT");
            ফেরত (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        অন্তঃরেখ constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_তথ্য_t::library_তথ্য_t_আকার <=
                              std::numeric_limits<unsigned অক্ষর>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(আকার...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_তথ্য_t>, "Unsupported FirstT");
            ফেরত get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        শ্রেণী mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#যদি defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_আকার, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#নতুবা_যদি defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    ফাঁকা,               // NULL specified as the base address parameter
                    mapped_region_আকার, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#অন্যথা
#error "Only support Windows and Linux."
#endif
                পরবর্তী_free = mapped_address_space;
            }

        সর্বজনীন:
            using বাফার_id_t = পূর্ণসংখ্যা;

            কাঠামো allocation
            {
                বাফার_t বাফার;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#যদি defined(__linux__)
                munmap(mapped_address_space, mapped_region_আকার);
#নতুবা_যদি defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#অন্যথা
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(ধ্রুবক mem_mgr &) = delete;
            mem_mgr &operator=(ধ্রুবক mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            শূন্য *mem_alloc(size_t size)
            {
                যদি (!size)
                    ফেরত nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                যদি (পরবর্তী_free + size > mapped_address_space + mapped_region_আকার)
                {
                    ছোড়ো std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                বাফার_t buf(r);
                allocation A{buf, পরবর্তী_free, size};
                // Map allocation to device pointer
                শূন্য *ফলাফল = পরবর্তী_free;
                m_map.emplace(পরবর্তী_free + size, A);
                // Update pointer to the next free space.
                পরবর্তী_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                ফেরত ফলাফল;
            }

            /// Deallocate
            শূন্য mem_free(ধ্রুবক শূন্য *ptr)
            {
                যদি (!ptr)
                    ফেরত;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(ধ্রুবক শূন্য *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                ফেরত it->second;
            }

            /// Check if the pointer represents device pointer or not.
            যৌক্তিক is_device_ptr(ধ্রুবক শূন্য *ptr) ধ্রুবক
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ফেরত (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_আকার);
            }

            /// Returns the instance of memory manager singleton.
            স্থির mem_mgr &instance()
            {
                স্থির mem_mgr m;
                ফেরত m;
            }

        ব্যক্তিগত:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *পরবর্তী_free;
            ধ্রুবক size_t mapped_region_আকার = 128ull * 1024 * 1024 * 1024;
            ধ্রুবক size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ধ্রুবক size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(ধ্রুবক শূন্য *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ধ্রুবক byte_t *>(ptr)));
                যদি (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ছোড়ো std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ধ্রুবক allocation &alloc = it->second;
                যদি (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ছোড়ো std::runtime_error("invalid virtual pointer");
                }
                ফেরত it;
            }
        };

        template <শ্রেণী T, memory_region Memory, size_t Dimension>
        শ্রেণী accessor;
        template <memory_region Memory, শ্রেণী T = byte_t>
        শ্রেণী memory_traits
        {
        সর্বজনীন:
            স্থির constexpr sycl::access::target target =
                sycl::access::target::device;
            স্থির constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::পঠ
                                     : sycl::access_mode::read_write;
            স্থির constexpr size_t type_আকার = আকার(T);
            using element_t =
                typename std::conditional<Memory == constant, ধ্রুবক T, T>::type;
            using মান_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<মান_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        স্থির অন্তঃরেখ শূন্য *dpct_malloc(size_t size, sycl::queue &q)
        {
            ফেরত sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        স্থির অন্তঃরেখ শূন্য *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            ফেরত dpct_malloc(pitch * y * z, q);
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
        স্থির অন্তঃরেখ sycl::event dpct_memset(sycl::queue &q, শূন্য *dev_ptr,
                                              মান_T value, size_t size)
        {
            ফেরত q.fill(dev_ptr, value, size);
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
        স্থির অন্তঃরেখ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_তথ্য তথ্য, মান_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = তথ্য.get_pitch() * তথ্য.get_y();
            unsigned অক্ষর *তথ্য_surface = (unsigned অক্ষর *)তথ্য.get_তথ্য_ptr();
            জন্য (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned অক্ষর *তথ্য_ptr = তথ্য_surface;
                জন্য (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, তথ্য_ptr, value, size.get(0)));
                    তথ্য_ptr += তথ্য.get_pitch();
                }
                তথ্য_surface += slice;
            }
            ফেরত event_list;
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
        স্থির অন্তঃরেখ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, শূন্য *ptr, size_t pitch, মান_T val, size_t x,
                    size_t y)
        {
            ফেরত dpct_memset(q, pitched_তথ্য(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        স্থির memcpy_direction deduce_memcpy_direction(sycl::queue &q, শূন্য *to_ptr,
                                                        ধ্রুবক শূন্য *from_ptr,
                                                        memcpy_direction dir)
        {
            নির্বাচন (dir)
            {
            ক্ষেত্র memcpy_direction::host_to_host:
            ক্ষেত্র memcpy_direction::host_to_device:
            ক্ষেত্র memcpy_direction::device_to_host:
            ক্ষেত্র memcpy_direction::device_to_device:
                ফেরত dir;
            ক্ষেত্র memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                স্থির ধ্রুবক memcpy_direction
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
                ফেরত direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            পূর্বনির্ধারিত:
                ছোড়ো std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        স্থির sycl::event
        dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr, size_t size,
                    memcpy_direction direction,
                    ধ্রুবক std::vector<sycl::event> &dep_events = {})
        {
            যদি (!size)
                ফেরত sycl::event{};
            ফেরত q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        স্থির অন্তঃরেখ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            ফেরত slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        স্থির অন্তঃরেখ size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            ফেরত slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        স্থির অন্তঃরেখ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ধ্রুবক std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            শ্রেণী host_বাফার
            {
                শূন্য *_buf;
                size_t আকার_3;
                sycl::queue &_q;
                ধ্রুবক std::vector<sycl::event> &_deps; // free operation depends

            সর্বজনীন:
                host_বাফার(size_t size, sycl::queue &q,
                            ধ্রুবক std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), আকার_3(size), _q(q), _deps(deps) {}
                শূন্য *get_ptr() ধ্রুবক { ফেরত _buf; }
                size_t get_আকার() ধ্রুবক { ফেরত আকার_3; }
                ~host_বাফার()
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
            unsigned অক্ষর *to_surface =
                (unsigned অক্ষর *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            ধ্রুবক unsigned অক্ষর *from_surface =
                (ধ্রুবক unsigned অক্ষর *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            যদি (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                ফেরত {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t আকার_slice = size.get(1) * size.get(0);
            নির্বাচন (direction)
            {
            ক্ষেত্র host_to_host:
                জন্য (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned অক্ষর *to_ptr = to_surface;
                    ধ্রুবক unsigned অক্ষর *from_ptr = from_surface;
                    যদি (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, আকার_slice,
                                                         direction, dep_events));
                    }
                    অন্যথা
                    {
                        জন্য (size_t y = 0; y < size.get(1); ++y)
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
                বিরতি;
            ক্ষেত্র host_to_device:
            {
                host_বাফার buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                যদি (to_slice == আকার_slice)
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
                            dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_আকার(),
                                        device_to_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                                 buf.get_আকার(), host_to_device,
                                                 host_events));
                বিরতি;
            }
            ক্ষেত্র device_to_host:
            {
                host_বাফার buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_to_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_আকার(),
                                                         device_to_host, dep_events)});
                বিরতি;
            }
            ক্ষেত্র device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<শ্রেণী dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                বিরতি;
            পূর্বনির্ধারিত:
                ছোড়ো std::runtime_error("dpct_memcpy: invalid direction value");
            }
            ফেরত event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        স্থির অন্তঃরেখ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_তথ্য to, sycl::id<3> to_id,
                    pitched_তথ্য from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            ফেরত dpct_memcpy(q, to.get_তথ্য_ptr(), from.get_তথ্য_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        স্থির অন্তঃরেখ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            ফেরত dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        নামাকাশ deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            শ্রেণী usm_allocator
            {
            ব্যক্তিগত:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            সর্বজনীন:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    ধ্রুবক typename std::allocator_traits<Alloc>::value_type &;
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
                কাঠামো rebind
                {
                    প্রকারলেখ usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(ধ্রুবক usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { ফেরত &r; }
                const_pointer address(const_reference r) { ফেরত &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    ফেরত std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                শূন্য deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ধ্রুবক
                {
                    ফেরত std::allocator_traits<Alloc>::max_size(_impl);
                }
                যৌক্তিক operator==(ধ্রুবক usm_allocator &other) ধ্রুবক { ফেরত _impl == other._impl; }
                যৌক্তিক operator!=(ধ্রুবক usm_allocator &other) ধ্রুবক { ফেরত _impl != other._impl; }
            };

        } // namespace deprecated

        অন্তঃরেখ শূন্য dpct_free(শূন্য *ptr,
                              ধ্রুবক sycl::queue &q)
        {
            যদি (ptr)
            {
                sycl::মুক্ত(ptr, q.get_context());
            }
        }

        template <typename T>
        অন্তঃরেখ auto get_memory(ধ্রুবক শূন্য *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<শূন্য *>(x));
            ফেরত new_x;
        }

        template <typename T>
        অন্তঃরেখ typename তথ্য_Type_2<T>::T2 get_মান(ধ্রুবক T *s, sycl::queue &q)
        {
            using Ty = typename তথ্য_Type_2<T>::T2;
            Ty s_h;
            যদি (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (শূন্য *)&s_h, (ধ্রুবক শূন্য *)s, আকার(T), device_to_host)
                    .wait();
            অন্যথা
                s_h = *reinterpret_cast<ধ্রুবক Ty *>(s);
            ফেরত s_h;
        }

    } // namespace detail

    template <typename T>
    অন্তঃরেখ auto get_মান(ধ্রুবক T *s, sycl::queue &q)
    {
        ফেরত detail::get_মান(s, q);
    }

    নামাকাশ detail
    {
    template <শ্রেণী Ta, শ্রেণী Tb, শ্রেণী Tc, শ্রেণী Ts>
    অন্তঃরেখ শূন্য gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণসংখ্যা m,
                          পূর্ণসংখ্যা n, পূর্ণসংখ্যা k, ধ্রুবক শূন্য * alpha, ধ্রুবক শূন্য * a, পূর্ণসংখ্যা lda, ধ্রুবক শূন্য * b, পূর্ণসংখ্যা ldb,
                          ধ্রুবক শূন্য * beta, শূন্য * c, পূর্ণসংখ্যা ldc) {
        Ts   alpha_মান_2 = dpct::get_মান(reinterpret_cast<ধ্রুবক Ts *>(alpha), q);
        Ts   beta_মান  = dpct::get_মান(reinterpret_cast<ধ্রুবক Ts *>(beta), q);
        auto তথ্য_a      = get_memory<ধ্রুবক Ta>(a);
        auto তথ্য_b      = get_memory<ধ্রুবক Tb>(b);
        auto তথ্য_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_মান_2, তথ্য_a,
                                               lda, তথ্য_b, ldb, beta_মান, তথ্য_c, ldc);
    }

        template <typename VecT, শ্রেণী দ্বিমিক_Operation, শ্রেণী = শূন্য>
        শ্রেণী vectorized_দ্বিমিক
        {
        সর্বজনীন:
            অন্তঃরেখ VecT operator()(VecT a, VecT b, ধ্রুবক দ্বিমিক_Operation দ্বিমিক_op)
            {
                VecT v4;
                জন্য (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = দ্বিমিক_op(a[i], b[i]);
                }
                ফেরত v4;
            }
        };

        template <typename VecT, শ্রেণী দ্বিমিক_Operation>
        শ্রেণী vectorized_দ্বিমিক<
            VecT, দ্বিমিক_Operation,
            std::void_t<std::invoke_result_t<দ্বিমিক_Operation, VecT, VecT>>>
        {
        সর্বজনীন:
            অন্তঃরেখ VecT operator()(VecT a, VecT b, ধ্রুবক দ্বিমিক_Operation দ্বিমিক_op)
            {
                ফেরত দ্বিমিক_op(a, b).template as<VecT>();
            }
        };

        template <শ্রেণী Ta, শ্রেণী Tb, শ্রেণী Tc, শ্রেণী Ts>
        অন্তঃরেখ শূন্য gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    পূর্ণসংখ্যা m, পূর্ণসংখ্যা n, পূর্ণসংখ্যা k, ধ্রুবক শূন্য * alpha, ধ্রুবক শূন্য ** a, পূর্ণসংখ্যা lda, ধ্রুবক শূন্য ** b,
                                    পূর্ণসংখ্যা ldb, ধ্রুবক শূন্য * beta, শূন্য ** c, পূর্ণসংখ্যা ldc, পূর্ণসংখ্যা batch_আকার,
                                    matrix_info_t<দশমিক> * matrix_info) {
            Ts alpha_মান_2 = dpct::get_মান(reinterpret_cast<ধ্রুবক Ts *>(alpha), q);
            Ts beta_মান = dpct::get_মান(reinterpret_cast<ধ্রুবক Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->মান_info[0] = alpha_মান_2;
            matrix_info->মান_info[1] = beta_মান;
            matrix_info->আকার_info[0] = m;
            matrix_info->আকার_info[1] = n;
            matrix_info->আকার_info[2] = k;
            matrix_info->ld_info[0] = lda;
            matrix_info->ld_info[1] = ldb;
            matrix_info->ld_info[2] = ldc;
            matrix_info->groupsize_info = batch_আকার;

            sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
                q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
                matrix_info->আকার_info, matrix_info->আকার_info + 1, matrix_info->আকার_info + 2,
                reinterpret_cast<Ts *>(matrix_info->মান_info), reinterpret_cast<ধ্রুবক Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<ধ্রুবক Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->মান_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <শ্রেণী Ta, শ্রেণী Tb, শ্রেণী Tc, শ্রেণী Ts>
        অন্তঃরেখ শূন্য gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    পূর্ণসংখ্যা m, পূর্ণসংখ্যা n, পূর্ণসংখ্যা k, ধ্রুবক শূন্য * alpha, ধ্রুবক শূন্য * a, পূর্ণসংখ্যা lda,
                                    long long পূর্ণসংখ্যা stride_a, ধ্রুবক শূন্য * b, পূর্ণসংখ্যা ldb, long long পূর্ণসংখ্যা stride_b,
                                    ধ্রুবক শূন্য * beta, শূন্য * c, পূর্ণসংখ্যা ldc, long long পূর্ণসংখ্যা stride_c, পূর্ণসংখ্যা batch_আকার) {
            Ts alpha_মান_2 = dpct::get_মান(reinterpret_cast<ধ্রুবক Ts *>(alpha), q);
            Ts beta_মান = dpct::get_মান(reinterpret_cast<ধ্রুবক Ts *>(beta), q);
            auto তথ্য_a = get_memory<ধ্রুবক Ta>(a);
            auto তথ্য_b = get_memory<ধ্রুবক Tb>(b);
            auto তথ্য_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_মান_2,
                                                         তথ্য_a, lda, stride_a, তথ্য_b, ldb, stride_b, beta_মান,
                                                         তথ্য_c, ldc, stride_c, batch_আকার);
        }

    } // namespace detail

    template <typename VecT, শ্রেণী দ্বিমিক_Operation>
    অন্তঃরেখ unsigned vectorized_দ্বিমিক(unsigned a, unsigned b,
                                      ধ্রুবক দ্বিমিক_Operation দ্বিমিক_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_দ্বিমিক<VecT, দ্বিমিক_Operation>()(v2, v3, দ্বিমিক_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        ফেরত v0;
    }

    স্থির শূন্য async_dpct_memcpy(শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    স্থির অন্তঃরেখ unsigned পূর্ণসংখ্যা select_device(unsigned পূর্ণসংখ্যা id)
    {
        dev_mgr::instance().select_device(id);
        ফেরত id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned পূর্ণসংখ্যা mask,
                               unsigned পূর্ণসংখ্যা logical_sub_group_আকার = 32)
    {
        unsigned পূর্ণসংখ্যা id = g.get_local_linear_id();
        unsigned পূর্ণসংখ্যা start_সূচক =
            id / logical_sub_group_আকার * logical_sub_group_আকার;
        unsigned পূর্ণসংখ্যা target_offset = (id % logical_sub_group_আকার) ^ mask;
        ফেরত sycl::select_from_group(g, x,
                                       target_offset < logical_sub_group_আকার
                                           ? start_সূচক + target_offset
                                           : id);
    }

    template <typename T1, typename T2>
    using dot_product_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    template <typename T>
    sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
      ফেরত sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    অন্তঃরেখ auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      ফেরত res;
    }

    কাঠামো sub_sat
    {
        template <typename T>
        auto operator()(ধ্রুবক T x, ধ্রুবক T y) ধ্রুবক
        {
            ফেরত sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    অন্তঃরেখ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        ফেরত v0;
    }

    অন্তঃরেখ দশমিক pow(ধ্রুবক দশমিক a, ধ্রুবক পূর্ণসংখ্যা b) { ফেরত sycl::pown(a, b); }
    অন্তঃরেখ দ্বিগুণ pow(ধ্রুবক দ্বিগুণ a, ধ্রুবক পূর্ণসংখ্যা b) { ফেরত sycl::pown(a, b); }
    অন্তঃরেখ দশমিক pow(ধ্রুবক দশমিক a, ধ্রুবক দশমিক b) { ফেরত sycl::pow(a, b); }
    অন্তঃরেখ দ্বিগুণ pow(ধ্রুবক দ্বিগুণ a, ধ্রুবক দ্বিগুণ b) { ফেরত sycl::pow(a, b); }
    template <typename T, typename U>
    অন্তঃরেখ typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(ধ্রুবক T a, ধ্রুবক U b)
    {
        ফেরত sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    অন্তঃরেখ typename std::enable_if_t<!std::is_floating_point_v<T>, দ্বিগুণ>
    pow(ধ্রুবক T a, ধ্রুবক U b)
    {
        ফেরত sycl::pow(static_cast<দ্বিগুণ>(a), static_cast<দ্বিগুণ>(b));
    }

    অন্তঃরেখ দ্বিগুণ min(ধ্রুবক দ্বিগুণ a, ধ্রুবক দশমিক b)
    {
        ফেরত sycl::fmin(a, static_cast<দ্বিগুণ>(b));
    }
    অন্তঃরেখ দ্বিগুণ min(ধ্রুবক দশমিক a, ধ্রুবক দ্বিগুণ b)
    {
        ফেরত sycl::fmin(static_cast<দ্বিগুণ>(a), b);
    }
    অন্তঃরেখ দশমিক min(ধ্রুবক দশমিক a, ধ্রুবক দশমিক b) { ফেরত sycl::fmin(a, b); }
    অন্তঃরেখ দ্বিগুণ min(ধ্রুবক দ্বিগুণ a, ধ্রুবক দ্বিগুণ b) { ফেরত sycl::fmin(a, b); }
    অন্তঃরেখ std::uint32_t min(ধ্রুবক std::uint32_t a, ধ্রুবক std::int32_t b)
    {
        ফেরত sycl::min(a, static_cast<std::uint32_t>(b));
    }
    অন্তঃরেখ std::uint32_t min(ধ্রুবক std::int32_t a, ধ্রুবক std::uint32_t b)
    {
        ফেরত sycl::min(static_cast<std::uint32_t>(a), b);
    }
    অন্তঃরেখ std::int32_t min(ধ্রুবক std::int32_t a, ধ্রুবক std::int32_t b)
    {
        ফেরত sycl::min(a, b);
    }
    অন্তঃরেখ std::uint32_t min(ধ্রুবক std::uint32_t a, ধ্রুবক std::uint32_t b)
    {
        ফেরত sycl::min(a, b);
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::uint64_t a, ধ্রুবক std::int64_t b)
    {
        ফেরত sycl::min(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::int64_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::min(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃরেখ std::int64_t min(ধ্রুবক std::int64_t a, ধ্রুবক std::int64_t b)
    {
        ফেরত sycl::min(a, b);
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::uint64_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::min(a, b);
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::uint64_t a, ধ্রুবক std::int32_t b)
    {
        ফেরত sycl::min(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::int32_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::min(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::uint64_t a, ধ্রুবক std::uint32_t b)
    {
        ফেরত sycl::min(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃরেখ std::uint64_t min(ধ্রুবক std::uint32_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    অন্তঃরেখ দ্বিগুণ max(ধ্রুবক দ্বিগুণ a, ধ্রুবক দশমিক b)
    {
        ফেরত sycl::fmax(a, static_cast<দ্বিগুণ>(b));
    }
    অন্তঃরেখ দ্বিগুণ max(ধ্রুবক দশমিক a, ধ্রুবক দ্বিগুণ b)
    {
        ফেরত sycl::fmax(static_cast<দ্বিগুণ>(a), b);
    }
    অন্তঃরেখ দশমিক max(ধ্রুবক দশমিক a, ধ্রুবক দশমিক b) { ফেরত sycl::fmax(a, b); }
    অন্তঃরেখ দ্বিগুণ max(ধ্রুবক দ্বিগুণ a, ধ্রুবক দ্বিগুণ b) { ফেরত sycl::fmax(a, b); }
    অন্তঃরেখ std::uint32_t max(ধ্রুবক std::uint32_t a, ধ্রুবক std::int32_t b)
    {
        ফেরত sycl::max(a, static_cast<std::uint32_t>(b));
    }
    অন্তঃরেখ std::uint32_t max(ধ্রুবক std::int32_t a, ধ্রুবক std::uint32_t b)
    {
        ফেরত sycl::max(static_cast<std::uint32_t>(a), b);
    }
    অন্তঃরেখ std::int32_t max(ধ্রুবক std::int32_t a, ধ্রুবক std::int32_t b)
    {
        ফেরত sycl::max(a, b);
    }
    অন্তঃরেখ std::uint32_t max(ধ্রুবক std::uint32_t a, ধ্রুবক std::uint32_t b)
    {
        ফেরত sycl::max(a, b);
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::uint64_t a, ধ্রুবক std::int64_t b)
    {
        ফেরত sycl::max(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::int64_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::max(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃরেখ std::int64_t max(ধ্রুবক std::int64_t a, ধ্রুবক std::int64_t b)
    {
        ফেরত sycl::max(a, b);
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::uint64_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::max(a, b);
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::uint64_t a, ধ্রুবক std::int32_t b)
    {
        ফেরত sycl::max(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::int32_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::max(static_cast<std::uint64_t>(a), b);
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::uint64_t a, ধ্রুবক std::uint32_t b)
    {
        ফেরত sycl::max(a, static_cast<std::uint64_t>(b));
    }
    অন্তঃরেখ std::uint64_t max(ধ্রুবক std::uint32_t a, ধ্রুবক std::uint64_t b)
    {
        ফেরত sycl::max(static_cast<std::uint64_t>(a), b);
    }

    অন্তঃরেখ শূন্য
    has_capability_or_fail(ধ্রুবক sycl::device &dev,
                           ধ্রুবক std::initializer_list<sycl::aspect> &props)
    {
        জন্য (ধ্রুবক auto &it : props)
        {
            যদি (dev.has(it))
                চালু;
            নির্বাচন (it)
            {
            ক্ষেত্র sycl::aspect::fp64:
                ছোড়ো std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                বিরতি;
            ক্ষেত্র sycl::aspect::fp16:
                ছোড়ো std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                বিরতি;
            পূর্বনির্ধারিত:
#define __SYCL_ASPECT(ASPECT, ID) \
    ক্ষেত্র sycl::aspect::ASPECT:    \
        ফেরত #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect Aspect_অঙ্ক) -> std::string
                {
                    নির্বাচন (Aspect_অঙ্ক)
                    {
#অন্তর্ভাব <sycl/info/aspects.def>
#অন্তর্ভাব <sycl/info/aspects_deprecated.def>
                    পূর্বনির্ধারিত:
                        ফেরত "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                ছোড়ো std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            বিরতি;
        }
    }

    স্থির অন্তঃরেখ unsigned পূর্ণসংখ্যা get_current_device_id()
    {
        ফেরত dev_mgr::instance().current_device_id();
    }

    স্থির অন্তঃরেখ device_ext &get_current_device()
    {
        ফেরত dev_mgr::instance().current_device();
    }

    স্থির অন্তঃরেখ device_ext &get_device(unsigned পূর্ণসংখ্যা id)
    {
        ফেরত dev_mgr::instance().get_device(id);
    }

    স্থির অন্তঃরেখ sycl::queue &get_in_order_queue()
    {
        ফেরত dev_mgr::instance().current_device().in_order_queue();
    }

    স্থির sycl::event
    dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr, size_t size,
                memcpy_direction direction,
                ধ্রুবক std::vector<sycl::event> &dep_events = {})
    {
        যদি (!size)
            ফেরত sycl::event{};
        ফেরত q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    স্থির অন্তঃরেখ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        ফেরত slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    স্থির অন্তঃরেখ size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        ফেরত slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    স্থির অন্তঃরেখ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ধ্রুবক std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        শ্রেণী host_বাফার
        {
            শূন্য *_buf;
            size_t আকার_3;
            sycl::queue &_q;
            ধ্রুবক std::vector<sycl::event> &_deps; // free operation depends

        সর্বজনীন:
            host_বাফার(size_t size, sycl::queue &q,
                        ধ্রুবক std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), আকার_3(size), _q(q), _deps(deps) {}
            শূন্য *get_ptr() ধ্রুবক { ফেরত _buf; }
            size_t get_আকার() ধ্রুবক { ফেরত আকার_3; }
            ~host_বাফার()
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
        unsigned অক্ষর *to_surface =
            (unsigned অক্ষর *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        ধ্রুবক unsigned অক্ষর *from_surface =
            (ধ্রুবক unsigned অক্ষর *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        যদি (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            ফেরত {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t আকার_slice = size.get(1) * size.get(0);
        নির্বাচন (direction)
        {
        ক্ষেত্র host_to_host:
            জন্য (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned অক্ষর *to_ptr = to_surface;
                ধ্রুবক unsigned অক্ষর *from_ptr = from_surface;
                যদি (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, আকার_slice,
                                                     direction, dep_events));
                }
                অন্যথা
                {
                    জন্য (size_t y = 0; y < size.get(1); ++y)
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
            বিরতি;
        ক্ষেত্র host_to_device:
        {
            host_বাফার buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            যদি (to_slice == আকার_slice)
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
                        dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_আকার(),
                                    device_to_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                             buf.get_আকার(), host_to_device,
                                             host_events));
            বিরতি;
        }
        ক্ষেত্র device_to_host:
        {
            host_বাফার buf(get_copy_range(size, from_slice, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_to_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_আকার(),
                                                     device_to_host, dep_events)});
            বিরতি;
        }
        ক্ষেত্র device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<শ্রেণী dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        বিরতি;
        পূর্বনির্ধারিত:
            ছোড়ো std::runtime_error("dpct_memcpy: invalid direction value");
        }
        ফেরত event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    স্থির অন্তঃরেখ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_তথ্য to, sycl::id<3> to_id,
                pitched_তথ্য from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        ফেরত dpct_memcpy(q, to.get_তথ্য_ptr(), from.get_তথ্য_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    স্থির অন্তঃরেখ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, শূন্য *to_ptr, ধ্রুবক শূন্য *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        ফেরত dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    অন্তঃরেখ শূন্য gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণসংখ্যা m, পূর্ণসংখ্যা n,
                     পূর্ণসংখ্যা k, ধ্রুবক শূন্য * alpha, ধ্রুবক শূন্য * a, library_তথ্য_t a_type, পূর্ণসংখ্যা lda, ধ্রুবক শূন্য * b,
                     library_তথ্য_t b_type, পূর্ণসংখ্যা ldb, ধ্রুবক শূন্য * beta, শূন্য * c, library_তথ্য_t c_type, পূর্ণসংখ্যা ldc,
                     library_তথ্য_t scaling_type) {
        যদি (scaling_type == library_তথ্য_t::real_float &&
            c_type == library_তথ্য_t::complex_float)
        {
            scaling_type = library_তথ্য_t::complex_float;
        }
        অন্যথা যদি (scaling_type == library_তথ্য_t::real_double &&
                 c_type == library_তথ্য_t::complex_double)
        {
            scaling_type = library_তথ্য_t::complex_double;
        }

        std::uint64_t চাবি =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        নির্বাচন (চাবি)
        {
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_float, library_তথ্য_t::real_float,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_impl<দশমিক, দশমিক, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_double, library_তথ্য_t::real_double,
            library_তথ্য_t::real_double, library_তথ্য_t::real_double):
        {
            detail::gemm_impl<দ্বিগুণ, দ্বিগুণ, দ্বিগুণ, দ্বিগুণ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::complex_float, library_তথ্য_t::complex_float,
            library_তথ্য_t::complex_float, library_তথ্য_t::complex_float):
        {
            detail::gemm_impl<std::complex<দশমিক>, std::complex<দশমিক>,
                              std::complex<দশমিক>, std::complex<দশমিক>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::complex_double, library_তথ্য_t::complex_double,
            library_তথ্য_t::complex_double, library_তথ্য_t::complex_double):
        {
            detail::gemm_impl<std::complex<দ্বিগুণ>, std::complex<দ্বিগুণ>,
                              std::complex<দ্বিগুণ>, std::complex<দ্বিগুণ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_half, library_তথ্য_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
#ifdef __INTEL_MKL__
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_bfloat16,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_half, library_তথ্য_t::real_float):
        {
            দশমিক alpha_মান_2 =
                dpct::get_মান(reinterpret_cast<ধ্রুবক দশমিক *>(alpha), q);
            দশমিক beta_মান =
                dpct::get_মান(reinterpret_cast<ধ্রুবক দশমিক *>(beta), q);
            sycl::half alpha_half(alpha_মান_2);
            sycl::half beta_half(beta_মান);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_int8, library_তথ্য_t::real_int8,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_bfloat16,
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_int8, library_তথ্য_t::real_int8,
            library_তথ্য_t::real_int32, library_তথ্য_t::real_int32):
        {
            দশমিক alpha_float =
                dpct::get_মান(reinterpret_cast<ধ্রুবক std::int32_t *>(alpha), q);
            দশমিক beta_float =
                dpct::get_মান(reinterpret_cast<ধ্রুবক std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, দশমিক>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            বিরতি;
        }
#endif // __INTEL_MKL__
        পূর্বনির্ধারিত:
            ছোড়ো std::runtime_error("the combination of data type is unsupported");
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
    অন্তঃরেখ শূন্য gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণসংখ্যা m,
                           পূর্ণসংখ্যা n, পূর্ণসংখ্যা k, ধ্রুবক শূন্য * alpha, ধ্রুবক শূন্য * a[], library_তথ্য_t a_type, পূর্ণসংখ্যা lda,
                           ধ্রুবক শূন্য * b[], library_তথ্য_t b_type, পূর্ণসংখ্যা ldb, ধ্রুবক শূন্য * beta, শূন্য * c[],
                           library_তথ্য_t c_type, পূর্ণসংখ্যা ldc, পূর্ণসংখ্যা batch_আকার, library_তথ্য_t scaling_type,
                           matrix_info_t<দশমিক> * matrix_info) {
        std::uint64_t চাবি =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        নির্বাচন (চাবি)
        {
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_float, library_তথ্য_t::real_float,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<দশমিক, দশমিক, দশমিক, দশমিক>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_double, library_তথ্য_t::real_double,
            library_তথ্য_t::real_double, library_তথ্য_t::real_double):
        {
            detail::gemm_batch_impl<দ্বিগুণ, দ্বিগুণ, দ্বিগুণ, দ্বিগুণ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_half, library_তথ্য_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
#ifdef __INTEL_MKL__
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_bfloat16,
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_bfloat16,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
#endif
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_int8, library_তথ্য_t::real_int8,
            library_তথ্য_t::real_int32, library_তথ্য_t::real_int32):
        {
            দশমিক alpha_float =
                dpct::get_মান(reinterpret_cast<ধ্রুবক std::int32_t *>(alpha), q);
            দশমিক beta_float =
                dpct::get_মান(reinterpret_cast<ধ্রুবক std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, দশমিক>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_আকার,
                matrix_info);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_int8, library_তথ্য_t::real_int8,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_half, library_তথ্য_t::real_float):
        {
            দশমিক alpha_মান_2 =
                dpct::get_মান(reinterpret_cast<ধ্রুবক দশমিক *>(alpha), q);
            দশমিক beta_মান =
                dpct::get_মান(reinterpret_cast<ধ্রুবক দশমিক *>(beta), q);
            sycl::half alpha_half(alpha_মান_2);
            sycl::half beta_half(beta_মান);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_আকার, matrix_info);
            বিরতি;
        }
        পূর্বনির্ধারিত:
            ছোড়ো std::runtime_error("the combination of data type is unsupported");
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
    অন্তঃরেখ শূন্য gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, পূর্ণসংখ্যা m,
                           পূর্ণসংখ্যা n, পূর্ণসংখ্যা k, ধ্রুবক শূন্য * alpha, ধ্রুবক শূন্য * a, library_তথ্য_t a_type, পূর্ণসংখ্যা lda,
                           long long পূর্ণসংখ্যা stride_a, ধ্রুবক শূন্য * b, library_তথ্য_t b_type, পূর্ণসংখ্যা ldb,
                           long long পূর্ণসংখ্যা stride_b, ধ্রুবক শূন্য * beta, শূন্য * c, library_তথ্য_t c_type, পূর্ণসংখ্যা ldc,
                           long long পূর্ণসংখ্যা stride_c, পূর্ণসংখ্যা batch_আকার, library_তথ্য_t scaling_type) {
        যদি (scaling_type == library_তথ্য_t::real_float &&
            c_type == library_তথ্য_t::complex_float)
        {
            scaling_type = library_তথ্য_t::complex_float;
        }
        অন্যথা যদি (scaling_type == library_তথ্য_t::real_double &&
                 c_type == library_তথ্য_t::complex_double)
        {
            scaling_type = library_তথ্য_t::complex_double;
        }

        std::uint64_t চাবি =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        নির্বাচন (চাবি)
        {
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_float, library_তথ্য_t::real_float,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<দশমিক, দশমিক, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_double, library_তথ্য_t::real_double,
            library_তথ্য_t::real_double, library_তথ্য_t::real_double):
        {
            detail::gemm_batch_impl<দ্বিগুণ, দ্বিগুণ, দ্বিগুণ, দ্বিগুণ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::complex_float, library_তথ্য_t::complex_float,
            library_তথ্য_t::complex_float, library_তথ্য_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<দশমিক>, std::complex<দশমিক>,
                                    std::complex<দশমিক>, std::complex<দশমিক>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::complex_double, library_তথ্য_t::complex_double,
            library_তথ্য_t::complex_double, library_তথ্য_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<দ্বিগুণ>, std::complex<দ্বিগুণ>,
                                    std::complex<দ্বিগুণ>, std::complex<দ্বিগুণ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_half, library_তথ্য_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
#ifdef __INTEL_MKL__
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_bfloat16,
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_bfloat16, library_তথ্য_t::real_bfloat16,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_আকার);
            বিরতি;
        }
#endif
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_int8, library_তথ্য_t::real_int8,
            library_তথ্য_t::real_int32, library_তথ্য_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_int8, library_তথ্য_t::real_int8,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_float, library_তথ্য_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, দশমিক, দশমিক>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        ক্ষেত্র detail::get_type_combination_id(
            library_তথ্য_t::real_half, library_তথ্য_t::real_half,
            library_তথ্য_t::real_half, library_তথ্য_t::real_float):
        {
            দশমিক alpha_মান_2 =
                dpct::get_মান(reinterpret_cast<ধ্রুবক দশমিক *>(alpha), q);
            দশমিক beta_মান =
                dpct::get_মান(reinterpret_cast<ধ্রুবক দশমিক *>(beta), q);
            sycl::half alpha_half(alpha_মান_2);
            sycl::half beta_half(beta_মান);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_আকার);
            বিরতি;
        }
        পূর্বনির্ধারিত:
            ছোড়ো std::runtime_error("the combination of data type is unsupported");
        }
    }

    স্থির অন্তঃরেখ শূন্য
    async_dpct_memcpy(শূন্য *to_ptr, size_t to_pitch, ধ্রুবক শূন্য *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<কাঠামো err0_tag, পূর্ণসংখ্যা>;
    using err1 = detail::generic_error_type<কাঠামো err1_tag, পূর্ণসংখ্যা>;

    স্থির অন্তঃরেখ শূন্য dpct_free(শূন্য *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <শ্রেণী T, memory_region Memory, size_t Dimension> শ্রেণী accessor;
    template <শ্রেণী T, memory_region Memory> শ্রেণী accessor<T, Memory, 3> {
    সর্বজনীন:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t তথ্য, ধ্রুবক sycl::range<3> &in_range)
            : তথ্য_2(তথ্য), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ধ্রুবক accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ধ্রুবক accessor_t &acc, ধ্রুবক sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t সূচক) ধ্রুবক {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            ফেরত accessor<T, Memory, 2>(তথ্য_2 + সূচক * sub.size(), sub);
        }

        pointer_t get_ptr() ধ্রুবক { ফেরত তথ্য_2; }

    ব্যক্তিগত:
        pointer_t তথ্য_2;
        sycl::range<3> _range;
    };
    template <শ্রেণী T, memory_region Memory> শ্রেণী accessor<T, Memory, 2> {
    সর্বজনীন:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t তথ্য, ধ্রুবক sycl::range<2> &in_range)
            : তথ্য_2(তথ্য), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ধ্রুবক accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ধ্রুবক accessor_t &acc, ধ্রুবক sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t সূচক) ধ্রুবক {
            ফেরত তথ্য_2 + _range.get(1) * সূচক;
        }

        pointer_t get_ptr() ধ্রুবক { ফেরত তথ্য_2; }

    ব্যক্তিগত:
        pointer_t তথ্য_2;
        sycl::range<2> _range;
    };

    নামাকাশ detail {
        /// Device variable with address space of shared, global or constant.
        template <শ্রেণী T, memory_region Memory, size_t Dimension> শ্রেণী device_memory {
        সর্বজনীন:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using মান_t = typename detail::memory_traits<Memory, T>::মান_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(ধ্রুবক sycl::range<Dimension> &in_range,
                        std::initializer_list<মান_t> &&init_list)
                : device_memory(in_range) {
                প্রতিজ্ঞা(init_list.size() <= in_range.size());
                _host_ptr = (মান_t *)std::malloc(আকার_3);
                std::memset(_host_ptr, 0, আকার_3);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * আকার(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                ধ্রুবক typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<মান_t>> &&init_list)
                : device_memory(in_range) {
                প্রতিজ্ঞা(init_list.size() <= in_range[0]);
                _host_ptr = (মান_t *)std::malloc(আকার_3);
                std::memset(_host_ptr, 0, আকার_3);
                auto tmp_তথ্য = _host_ptr;
                জন্য (auto sub_list : init_list) {
                    প্রতিজ্ঞা(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_তথ্য, sub_list.begin(),
                                sub_list.size() * আকার(T));
                    tmp_তথ্য += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(ধ্রুবক sycl::range<Dimension> &range_in)
                : আকার_3(range_in.size() * আকার(T)), _range(range_in),
                _reference(মিথ্যা), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <শ্রেণী... Args>
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
                    ফেরত;
                যদি (!আকার_3)
                    ফেরত;
                allocate_device(q);
                যদি (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, আকার_3,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            শূন্য assign(মান_t *src, size_t size) {
                এই->~device_memory();
                নতুন (এই) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            মান_t *get_ptr() { ফেরত get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            মান_t *get_ptr(sycl::queue &q) {
                init(q);
                ফেরত _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_আকার() { ফেরত আকার_3; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t সূচক) {
                init();
                ফেরত _device_ptr[সূচক];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                ফেরত dpct_accessor_t((T *)_device_ptr, _range);
            }

        ব্যক্তিগত:
            device_memory(মান_t *memory_ptr, size_t size)
                : আকার_3(size), _range(size / আকার(T)), _reference(সত্য),
                _device_ptr(memory_ptr) {}

            শূন্য allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                যদি (Memory == shared) {
                    _device_ptr = (মান_t *)sycl::malloc_shared(আকার_3, q.get_device(),
                                                                q.get_context());
                    ফেরত;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                যদি (Memory == constant) {
                    _device_ptr = (মান_t *)sycl::malloc_device(
                        আকার_3, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    ফেরত;
                }
        #endif
        #endif
                _device_ptr = (মান_t *)detail::dpct_malloc(আকার_3, q);
            }

            size_t আকার_3;
            sycl::range<Dimension> _range;
            যৌক্তিক _reference;
            মান_t *_host_ptr;
            মান_t *_device_ptr;
        };
        template <শ্রেণী T, memory_region Memory>
        শ্রেণী device_memory<T, Memory, 0> : সর্বজনীন device_memory<T, Memory, 1> {
        সর্বজনীন:
            using base = device_memory<T, Memory, 1>;
            using মান_t = typename base::মান_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(ধ্রুবক মান_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <শ্রেণী T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <শ্রেণী T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <শ্রেণী T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    অন্তঃরেখ T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ফেরত atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    অন্তঃরেখ T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ফেরত atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    অন্তঃরেখ T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    নির্বাচন (memoryOrder) {
        ক্ষেত্র sycl::memory_order::relaxed:
            ফেরত atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        ক্ষেত্র sycl::memory_order::acq_rel:
            ফেরত atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        ক্ষেত্র sycl::memory_order::seq_cst:
            ফেরত atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        পূর্বনির্ধারিত:
            প্রতিজ্ঞা(মিথ্যা && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    অন্তঃরেখ T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    অন্তঃরেখ unsigned পূর্ণসংখ্যা byte_level_permute(
        unsigned পূর্ণসংখ্যা a, unsigned পূর্ণসংখ্যা b, unsigned পূর্ণসংখ্যা s) {
      unsigned পূর্ণসংখ্যা ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      ফেরত ret;
    }

    অন্তঃরেখ uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, পূর্ণসংখ্যা mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      যদি (mode >= 1 && mode <= 6) {
        ফেরত byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } অন্যথা যদি (!mode) {
        ফেরত byte_level_permute(low32, high32, sel);
      }
      ফেরত 0;
    }

    template <পূর্ণসংখ্যা n_nondefault_params, পূর্ণসংখ্যা n_default_params, typename T>
    শ্রেণী args_selector;

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
    template <পূর্ণসংখ্যা n_nondefault_params, পূর্ণসংখ্যা n_default_params, typename R,
              typename... Ts>
    শ্রেণী args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      ব্যক্তিগত:
        শূন্য **kernel_params;
        অক্ষর *args_বাফার;

        template <পূর্ণসংখ্যা i> স্থির constexpr পূর্ণসংখ্যা account_for_default_params() {
            constexpr পূর্ণসংখ্যা n_যোগ_params = আকার...(Ts);
            যদি constexpr (i >= n_nondefault_params) {
                ফেরত n_যোগ_params - n_default_params +
                       (i - n_nondefault_params);
            } অন্যথা {
                ফেরত i;
            }
        }

      সর্বজনীন:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <পূর্ণসংখ্যা i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        স্থির constexpr পূর্ণসংখ্যা params_অঙ্ক = আকার...(Ts);

      ব্যক্তিগত:
        template <পূর্ণসংখ্যা i> স্থির constexpr পূর্ণসংখ্যা get_offset() {
            যদি constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                ফেরত 0;
            } অন্যথা {
                constexpr পূর্ণসংখ্যা পূর্ব_off = get_offset<i - 1>();
                constexpr পূর্ণসংখ্যা পূর্ব_past_end =
                    পূর্ব_off + আকার(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                যদি constexpr (পূর্ব_past_end % alignof(T) == 0) {
                    ফেরত পূর্ব_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                অন্যথা {
                    ফেরত পূর্ব_past_end +
                           (alignof(T) - (পূর্ব_past_end % alignof(T)));
                }
            }
        }

        স্থির অক্ষর *get_args_বাফার(শূন্য **extra) {
            যদি (!extra)
                ফেরত nullptr;
            জন্য (; (std::size_t)*extra != 0; ++extra) {
                যদি ((std::size_t)*extra == 1) {
                    ফেরত static_cast<অক্ষর *>(*(extra + 1));
                }
            }
            ফেরত nullptr;
        }

      সর্বজনীন:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(শূন্য **kernel_params, শূন্য **extra)
            : kernel_params(kernel_params),
              args_বাফার(get_args_বাফার(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <পূর্ণসংখ্যা i> arg_type<i> &get() {
            যদি (kernel_params) {
                ফেরত *static_cast<arg_type<i> *>(kernel_params[i]);
            } অন্যথা {
                ফেরত *reinterpret_cast<arg_type<i> *>(args_বাফার +
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
    শ্রেণী kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... সূচক_2>
        স্থির শূন্য launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<সূচক_2...>) {
            func(selector.template get<সূচক_2>()...);
        }
        স্থির শূন্য set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned পূর্ণসংখ্যা local_mem_আকার_2,
                                         queue_ptr que) {
            যদি (que) {
                _que = que;
            } অন্যথা {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_আকার = local_mem_আকার_2;


        }
        স্থির অন্তঃরেখ std::mutex kernel_function_ptr_map_mutex;

      সর্বজনীন:
        /// Variables for storing execution configuration.
        স্থির অন্তঃরেখ thread_local sycl::queue *_que = nullptr;
        স্থির অন্তঃরেখ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        স্থির অন্তঃরেখ thread_local unsigned পূর্ণসংখ্যা local_mem_আকার = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        স্থির অন্তঃরেখ std::map<
            ধ্রুবক শূন্য *,
            std::function<শূন্য(dim3, dim3, শূন্য **, unsigned পূর্ণসংখ্যা, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        স্থির শূন্য register_kernel_ptr(
            ধ্রুবক শূন্য *func,
            std::function<শূন্য(dim3, dim3, শূন্য **, unsigned পূর্ণসংখ্যা, queue_ptr)>
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
               unsigned পূর্ণসংখ্যা local_mem_আকার_2, queue_ptr que, ArgsT... args) {
            set_execution_config(group_range, local_range, local_mem_আকার_2, que);
            func(args...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        স্থির শূন্য launch(ধ্রুবক শূন্য *func, dim3 group_range, dim3 local_range,
                           শূন্য **args, unsigned পূর্ণসংখ্যা local_mem_আকার_2,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            যদি (Iter == kernel_function_ptr_map.end()) {
                ছোড়ো std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(group_range, local_range, args, local_mem_আকার_2, que);
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
               unsigned পূর্ণসংখ্যা local_mem_আকার_2, queue_ptr que) {
            constexpr size_t p_অঙ্ক = args_selector<0, 0, FuncT>::params_অঙ্ক;
            set_execution_config(group_range, local_range, local_mem_আকার_2, que);
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
        পূর্ণসংখ্যা remote_local_id,
        পূর্ণসংখ্যা logical_sub_group_আকার = 32) {
      unsigned পূর্ণসংখ্যা start_সূচক = g.get_local_linear_id() /
                                 logical_sub_group_আকার *
                                 logical_sub_group_আকার;
      ফেরত sycl::select_from_group(
          g, x, start_সূচক + remote_local_id % logical_sub_group_আকার);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    শূন্য ldmatrix(uintptr_t addr, T* m, যৌক্তিক trans = মিথ্যা, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      পূর্ণসংখ্যা lane = sg.get_local_linear_id();

      পূর্ণসংখ্যা lane_group8_row = lane / 8;
      পূর্ণসংখ্যা lane_group8_col = lane % 8;

      যদি (!trans) {
        // calculate the source lane
        পূর্ণসংখ্যা src_lane = 2 * lane_group8_row;
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
        পূর্ণসংখ্যা src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        পূর্ণসংখ্যা সূচক = lane / 4;
        sycl::half val0 = recv_addr_1[সূচক];
        sycl::half val1 = recv_addr_2[সূচক];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    শূন্য ldmatrix(uintptr_t addr, T* m1, T* m2, যৌক্তিক trans = মিথ্যা) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    শূন্য ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, যৌক্তিক trans = মিথ্যা) {
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
    কাঠামো MMAType {
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
    template <পূর্ণসংখ্যা M, পূর্ণসংখ্যা N, পূর্ণসংখ্যা K, typename ABType, typename CDType>
    শূন্য mma(
        অস্থির শূন্য** d_mat_frag,
        শূন্য* a_mat_frag,
        শূন্য* b_mat_frag,
        শূন্য* c_mat_frag) {
      auto d = reinterpret_cast<অস্থির CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      পূর্ণসংখ্যা lane = sg.get_local_linear_id();

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
          পূর্ণসংখ্যা r_ind = (lane % 2) ? 1 : 0;
          পূর্ণসংখ্যা c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
                static_cast<দশমিক>(ra[r_ind]) * static_cast<দশমিক>(rb[c_ind]);
            *d[1] += static_cast<দশমিক>(ra[r_ind]) *
                     static_cast<দশমিক>(rb[c_ind + 1]);
            *d[2] += static_cast<দশমিক>(ra[r_ind + 2]) *
                     static_cast<দশমিক>(rb[c_ind]);
            *d[3] += static_cast<দশমিক>(ra[r_ind + 2]) *
                     static_cast<দশমিক>(rb[c_ind + 1]);

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
                static_cast<দশমিক>(ra[r_ind]) * static_cast<দশমিক>(rb[c_ind]);
            *d[5] += static_cast<দশমিক>(ra[r_ind]) *
                     static_cast<দশমিক>(rb[c_ind + 1]);
            *d[6] += static_cast<দশমিক>(ra[r_ind + 2]) *
                     static_cast<দশমিক>(rb[c_ind]);
            *d[7] += static_cast<দশমিক>(ra[r_ind + 2]) *
                     static_cast<দশমিক>(rb[c_ind + 1]);
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
          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
            জন্য (পূর্ণসংখ্যা j = 0; j < 4; j++) {
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
          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
            জন্য (পূর্ণসংখ্যা j = 0; j < 2; j++) {
              *d[0] += static_cast<দশমিক>(ra[j]) * static_cast<দশমিক>(rb[j]);
              *d[1] +=
                  static_cast<দশমিক>(ra[j]) * static_cast<দশমিক>(rb[j + 2]);
              *d[2] +=
                  static_cast<দশমিক>(ra[j + 2]) * static_cast<দশমিক>(rb[j]);
              *d[3] +=
                  static_cast<দশমিক>(ra[j + 2]) * static_cast<দশমিক>(rb[j + 2]);
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
          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
            জন্য (পূর্ণসংখ্যা j = 0; j < 4; j++) {
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
          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
            জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
            জন্য (পূর্ণসংখ্যা j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
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
            জন্য (পূর্ণসংখ্যা j = 0; j < 4; j++) {
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

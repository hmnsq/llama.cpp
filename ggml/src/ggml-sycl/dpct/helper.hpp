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

#ଅନ୍ତର୍ଭାଵ <cstdlib>
#ଅନ୍ତର୍ଭାଵ <iostream>
#ଅନ୍ତର୍ଭାଵ <map>

#ଅନ୍ତର୍ଭାଵ <sycl/sycl.hpp>
#ଅନ୍ତର୍ଭାଵ <sycl/half_type.hpp>
#ଅନ୍ତର୍ଭାଵ <oneapi/mkl.hpp>

#ଯଦି defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <sys/mman.h>
#ନଚେତଯଦି defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ଅନ୍ତର୍ଭାଵ <windows.h>
#ଅନ୍ୟଥା
#error "Only support Windows and Linux."
#endif

#ଯଦି defined(__linux__)
#ଅନ୍ତର୍ଭାଵ <unistd.h>
#ଅନ୍ତର୍ଭାଵ <sys/syscall.h>
#endif
#ଯଦି defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ଅନ୍ତର୍ଭାଵ <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#ଯଦି defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#ଅନ୍ୟଥା
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#ଯଦି defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#ଅନ୍ୟଥା
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (ଶୂନ୍ୟ)(x)

[[noreturn]] ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ _abort(ଅଚଳ ଅକ୍ଷର * str) {
    std::cerr << str << std::endl;
    std::abort();
}

ଧାଡ଼ିଭିତରେ std::string get_device_type_name(ଅଚଳ sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    ବାଛ (DeviceType) {
    ସ୍ଥିତି sycl::info::device_type::cpu:
        ଫେରନ୍ତୁ "cpu";
    ସ୍ଥିତି sycl::info::device_type::gpu:
        ଫେରନ୍ତୁ "gpu";
    ସ୍ଥିତି sycl::info::device_type::host:
        ଫେରନ୍ତୁ "host";
    ସ୍ଥିତି sycl::info::device_type::accelerator:
        ଫେରନ୍ତୁ "acc";
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        ଫେରନ୍ତୁ "unknown";
    }
}

ଧାଡ଼ିଭିତରେ std::string get_device_backend_and_type(ଅଚଳ sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    ଫେରନ୍ତୁ device_type.str();
}

template <typename Ts> ସଂରଚନା matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     ମୂଲ୍ୟ_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

ନାମାକାଶ dpct
{
    ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::queue *queue_ptr;
    ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::event *event_ptr;
    ପ୍ରକାରବ୍ୟାଖ୍ୟା ଅକ୍ଷର *device_ptr;
    ପ୍ରକାରବ୍ୟାଖ୍ୟା uint8_t byte_t;
    ପ୍ରକାରବ୍ୟାଖ୍ୟା sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    ଧାଡ଼ିଭିତରେ auto exception_handler = [](sycl::exception_list exceptions)
    {
        ପାଇଁ (std::exception_ptr ଅଚଳ &e : exceptions)
        {
            ଚେଷ୍ଟା
            {
                std::rethrow_exception(e);
            }
            ଧର (sycl::exception ଅଚଳ &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    ଗଣନା error_code
    {
        success = 0,
        default_error = 999
    };

    ଗଣନା memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    ଗଣନା memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    ଗଣନା ଶ୍ରେଣୀ library_data_t : unsigned ଅକ୍ଷର
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
    ସଂରଚନା DataType
    {
        using T2 = T;
    };
    template <typename T>
    ସଂରଚନା DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    ସ୍ଥିର ଶୂନ୍ୟ destroy_event(event_ptr event)
    {
        delete event;
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_tid()
    {
#ଯଦି defined(__linux__)
        ଫେରନ୍ତୁ syscall(SYS_gettid);
#ନଚେତଯଦି defined(_WIN64)
        ଫେରନ୍ତୁ GetCurrentThreadId();
#ଅନ୍ୟଥା
#error "Only support Windows and Linux."
#endif
    }

    ନାମାକାଶ detail
    {
        ସ୍ଥିର ଶୂନ୍ୟ get_version(ଅଚଳ sycl::device &dev, ପୂର୍ଣ୍ଣସଂଖ୍ୟା &major, ପୂର୍ଣ୍ଣସଂଖ୍ୟା &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            ଯେତେବେଳେ (i < ver.size()) {
              ଯଦି (isdigit(ver[i]))
                ଭାଙ୍ଗ;
              i++;
            }
            major = std::stoi(&(ver[i]));
            ଯେତେବେଳେ (i < ver.size()) {
              ଯଦି (ver[i] == '.')
                ଭାଙ୍ଗ;
              i++;
            }
            ଯଦି (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } ଅନ୍ୟଥା {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        ଶ୍ରେଣୀ generic_error_type
        {
        ସର୍ବସାଧାରଣ:
            generic_error_type() = ପୂର୍ବନିର୍ଦ୍ଧାରିତ;
            generic_error_type(T value) : value{value} {}
            operator T() ଅଚଳ { ଫେରନ୍ତୁ value; }

        ବ୍ୟକ୍ତିଗତ:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    ଶ୍ରେଣୀ dim3 {
        ସର୍ବସାଧାରଣ:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(ଅଚଳ sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() ଅଚଳ { ଫେରନ୍ତୁ sycl::range<3>(z, y, x); }
    }; // namespace dim3

    ଧାଡ଼ିଭିତରେ dim3 operator*(ଅଚଳ dim3 &a, ଅଚଳ dim3 &b) {
    ଫେରନ୍ତୁ dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    ଶ୍ରେଣୀ pitched_data
    {
    ସର୍ବସାଧାରଣ:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(ଶୂନ୍ୟ *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        ଶୂନ୍ୟ *get_data_ptr() { ଫେରନ୍ତୁ _data; }
        ଶୂନ୍ୟ set_data_ptr(ଶୂନ୍ୟ *data) { _data = data; }

        size_t get_pitch() { ଫେରନ୍ତୁ _pitch; }
        ଶୂନ୍ୟ set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { ଫେରନ୍ତୁ _x; }
        ଶୂନ୍ୟ set_x(size_t x) { _x = x; }

        size_t get_y() { ଫେରନ୍ତୁ _y; }
        ଶୂନ୍ୟ set_y(size_t y) { _y = y; }

    ବ୍ୟକ୍ତିଗତ:
        ଶୂନ୍ୟ *_data;
        size_t _pitch, _x, _y;
    };

    ଶ୍ରେଣୀ device_info
    {
    ସର୍ବସାଧାରଣ:
        // get interface
        ଅଚଳ ଅକ୍ଷର *get_name() ଅଚଳ { ଫେରନ୍ତୁ _name; }
        ଅକ୍ଷର *get_name() { ଫେରନ୍ତୁ _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା> = 0>
        auto get_max_work_item_sizes() ଅଚଳ
        {
            ଯଦି constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ଫେରନ୍ତୁ sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            ଅନ୍ୟଥା
            {
                ଫେରନ୍ତୁ _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା> = 0>
        auto get_max_work_item_sizes()
        {
            ଯଦି constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                ଫେରନ୍ତୁ sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            ଅନ୍ୟଥା
            {
                ଫେରନ୍ତୁ _max_work_item_sizes_i;
            }
        }
        ତାର୍କିକ get_host_unified_memory() ଅଚଳ { ଫେରନ୍ତୁ _host_unified_memory; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_major_version() ଅଚଳ { ଫେରନ୍ତୁ _major; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_minor_version() ଅଚଳ { ଫେରନ୍ତୁ _minor; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_integrated() ଅଚଳ { ଫେରନ୍ତୁ _integrated; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_clock_frequency() ଅଚଳ { ଫେରନ୍ତୁ _frequency; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_compute_units() ଅଚଳ { ଫେରନ୍ତୁ _max_compute_units; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_work_group_size() ଅଚଳ { ଫେରନ୍ତୁ _max_work_group_size; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_sub_group_size() ଅଚଳ { ଫେରନ୍ତୁ _max_sub_group_size; }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit() ଅଚଳ
        {
            ଫେରନ୍ତୁ max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit;
        }
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_register_size_per_work_group() ଅଚଳ
        {
            ଫେରନ୍ତୁ _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା> = 0>
        auto get_max_nd_range_size() ଅଚଳ
        {
            ଯଦି constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                ଫେରନ୍ତୁ _max_nd_range_size;
            ଅନ୍ୟଥା
                ଫେରନ୍ତୁ _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, ପୂର୍ଣ୍ଣସଂଖ୍ୟା *>,
                                   ପୂର୍ଣ୍ଣସଂଖ୍ୟା> = 0>
        auto get_max_nd_range_size()
        {
            ଯଦି constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                ଫେରନ୍ତୁ _max_nd_range_size;
            ଅନ୍ୟଥା
                ଫେରନ୍ତୁ _max_nd_range_size_i;
        }
        size_t get_global_mem_size() ଅଚଳ { ଫେରନ୍ତୁ _global_mem_size; }
        size_t get_local_mem_size() ଅଚଳ { ଫେରନ୍ତୁ _local_mem_size; }
        size_t get_max_mem_alloc_size() ଅଚଳ { ଫେରନ୍ତୁ _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_memory_clock_rate() ଅଚଳ { ଫେରନ୍ତୁ _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_memory_bus_width() ଅଚଳ { ଫେରନ୍ତୁ _memory_bus_width; }
        uint32_t get_device_id() ଅଚଳ { ଫେରନ୍ତୁ _device_id; }
        std::array<unsigned ଅକ୍ଷର, 16> get_uuid() ଅଚଳ { ଫେରନ୍ତୁ _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_global_mem_cache_size() ଅଚଳ
        {
            ଫେରନ୍ତୁ _global_mem_cache_size;
        }

        // set interface
        ଶୂନ୍ୟ set_name(ଅଚଳ ଅକ୍ଷର *name)
        {
            size_t length = strlen(name);
            ଯଦି (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            ଅନ୍ୟଥା
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        ଶୂନ୍ୟ set_max_work_item_sizes(ଅଚଳ sycl::range<3> max_work_item_sizes)
        {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] ଶୂନ୍ୟ
        set_max_work_item_sizes(ଅଚଳ sycl::id<3> max_work_item_sizes)
        {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        ଶୂନ୍ୟ set_host_unified_memory(ତାର୍କିକ host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        ଶୂନ୍ୟ set_major_version(ପୂର୍ଣ୍ଣସଂଖ୍ୟା major) { _major = major; }
        ଶୂନ୍ୟ set_minor_version(ପୂର୍ଣ୍ଣସଂଖ୍ୟା minor) { _minor = minor; }
        ଶୂନ୍ୟ set_integrated(ପୂର୍ଣ୍ଣସଂଖ୍ୟା integrated) { _integrated = integrated; }
        ଶୂନ୍ୟ set_max_clock_frequency(ପୂର୍ଣ୍ଣସଂଖ୍ୟା frequency) { _frequency = frequency; }
        ଶୂନ୍ୟ set_max_compute_units(ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        ଶୂନ୍ୟ set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        ଶୂନ୍ୟ set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        ଶୂନ୍ୟ set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        ଶୂନ୍ୟ set_max_work_group_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        ଶୂନ୍ୟ set_max_sub_group_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        ଶୂନ୍ୟ
        set_max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit(ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit_2)
        {
            max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit = max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit_2;
        }
        ଶୂନ୍ୟ set_max_nd_range_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_nd_range_size[])
        {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        ଶୂନ୍ୟ set_memory_clock_rate(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        ଶୂନ୍ୟ set_memory_bus_width(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        ଶୂନ୍ୟ
        set_max_register_size_per_work_group(ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        ଶୂନ୍ୟ set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        ଶୂନ୍ୟ set_uuid(std::array<unsigned ଅକ୍ଷର, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        ଶୂନ୍ୟ set_global_mem_cache_size(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    ବ୍ୟକ୍ତିଗତ:
        ଅକ୍ଷର _name[256];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _max_work_item_sizes_i[3];
        ତାର୍କିକ _host_unified_memory = ମିଥ୍ୟା;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _major;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _minor;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _integrated = 0;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା _memory_bus_width = 64;
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା _global_mem_cache_size;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _max_compute_units;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _max_work_group_size;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _max_sub_group_size;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned ଅକ୍ଷର, 16> _uuid;
    };

    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_major_version(ଅଚଳ sycl::device &dev)
    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା major, minor;
        detail::get_version(dev, major, minor);
        ଫେରନ୍ତୁ major;
    }

    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_minor_version(ଅଚଳ sycl::device &dev)
    {
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା major, minor;
        detail::get_version(dev, major, minor);
        ଫେରନ୍ତୁ minor;
    }

    ସ୍ଥିର ଶୂନ୍ୟ get_device_info(device_info &out, ଅଚଳ sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        ପୂର୍ଣ୍ଣସଂଖ୍ୟା major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#ଯଦି (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#ଅନ୍ୟଥା
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

#ଯଦି (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        ଯଦି (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            ଯଦି (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        ଯଦି (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        ଯଦି (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        ଯଦି (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#ନଚେତଯଦି defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate ପୂର୍ବନିର୍ଦ୍ଧାରିତ value. \
        Use 64 bits as memory_bus_width ପୂର୍ବନିର୍ଦ୍ଧାରିତ value.")
#ଅନ୍ୟଥା
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate ପୂର୍ବନିର୍ଦ୍ଧାରିତ value. \
        Use 64 bits as memory_bus_width ପୂର୍ବନିର୍ଦ୍ଧାରିତ value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        ପାଇଁ (ଅଚଳ auto &sub_group_size : sub_group_sizes)
        {
            ଯଦି (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_ବସ୍ତୁଗୁଡ଼ିକ_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    ଶ୍ରେଣୀ device_ext : ସର୍ବସାଧାରଣ sycl::device {
      ପ୍ରକାରବ୍ୟାଖ୍ୟା std::mutex mutex_type;

     ସର୍ବସାଧାରଣ:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(ଅଚଳ sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା is_native_atomic_supported() { ଫେରନ୍ତୁ 0; }
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_major_version() ଅଚଳ { ଫେରନ୍ତୁ dpct::get_major_version(*ଏହି); }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_minor_version() ଅଚଳ { ଫେରନ୍ତୁ dpct::get_minor_version(*ଏହି); }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_compute_units() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_clock_frequency() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_max_clock_frequency();
      }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_integrated() ଅଚଳ { ଫେରନ୍ତୁ get_device_info().get_integrated(); }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_sub_group_size() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_max_sub_group_size();
      }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_register_size_per_work_group() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_max_register_size_per_work_group();
      }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_max_work_group_size() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_max_work_group_size();
      }

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_mem_base_addr_align() ଅଚଳ {
        ଫେରନ୍ତୁ get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() ଅଚଳ {
        ଫେରନ୍ତୁ get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      ଶୂନ୍ୟ get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        ଅଚଳ ଅକ୍ଷର *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#ଯଦି (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        ଯଦି (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } ଅନ୍ୟଥା {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#ଅନ୍ୟଥା
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#ଯଦି defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#ଅନ୍ୟଥା
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      ଶୂନ୍ୟ get_device_info(device_info &out) ଅଚଳ {
        dpct::get_device_info(out, *ଏହି);
      }

      device_info get_device_info() ଅଚଳ {
        device_info prop;
        dpct::get_device_info(prop, *ଏହି);
        ଫେରନ୍ତୁ prop;
      }

      ଶୂନ୍ୟ reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { ଫେରନ୍ତୁ _q_in_order; }

      sycl::queue &out_of_order_queue() { ଫେରନ୍ତୁ _q_out_of_order; }

      sycl::queue &default_queue() { ଫେରନ୍ତୁ in_order_queue(); }

      ଶୂନ୍ୟ queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        ପାଇଁ (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(ତାର୍କିକ enable_exception_handler = ମିଥ୍ୟା) {
        ଫେରନ୍ତୁ create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               ତାର୍କିକ enable_exception_handler = ମିଥ୍ୟା) {
        ଫେରନ୍ତୁ create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(ତାର୍କିକ enable_exception_handler = ମିଥ୍ୟା) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ଫେରନ୍ତୁ create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        ତାର୍କିକ enable_exception_handler = ମିଥ୍ୟା) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ଫେରନ୍ତୁ create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          ତାର୍କିକ enable_exception_handler = ମିଥ୍ୟା) {
        std::lock_guard<mutex_type> lock(m_mutex);
        ଫେରନ୍ତୁ create_queue_impl(enable_exception_handler);
      }

      ଶୂନ୍ୟ destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ଅଚଳ sycl::queue &q) -> ତାର୍କିକ
                                    {
                                        ଫେରନ୍ତୁ q == queue;
                                    }),
                    _queues.end());
      }
      ଶୂନ୍ୟ set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() ଅଚଳ {
        std::lock_guard<mutex_type> lock(m_mutex);
        ଫେରନ୍ତୁ _saved_queue;
      }

     ବ୍ୟକ୍ତିଗତ:
      ଶୂନ୍ୟ clear_queues() { _queues.clear(); }

      ଶୂନ୍ୟ init_queues() {
        _q_in_order =
            create_queue_impl(ସତ୍ୟ, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(ସତ୍ୟ);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <ଶ୍ରେଣୀ... Properties>
      sycl::queue create_queue_impl(ତାର୍କିକ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ଯଦି (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *ଏହି, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        ଫେରନ୍ତୁ _queues.back();
      }

      template <ଶ୍ରେଣୀ... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    ତାର୍କିକ enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ଯଦି (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        ଫେରନ୍ତୁ _queues.back();
      }

      ଶୂନ୍ୟ get_version(ପୂର୍ଣ୍ଣସଂଖ୍ୟା &major, ପୂର୍ଣ୍ଣସଂଖ୍ୟା &minor) ଅଚଳ {
        detail::get_version(*ଏହି, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    ଶ୍ରେଣୀ dev_mgr
    {
    ସର୍ବସାଧାରଣ:
        device_ext &current_device()
        {
            unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା dev_id = current_device_id();
            check_id(dev_id);
            ଫେରନ୍ତୁ *_devs[dev_id];
        }
        device_ext &cpu_device() ଅଚଳ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            ଯଦି (_cpu_device == -1)
            {
                ଫିଙ୍ଗ std::runtime_error("no valid cpu device");
            }
            ଅନ୍ୟଥା
            {
                ଫେରନ୍ତୁ *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) ଅଚଳ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            ଫେରନ୍ତୁ *_devs[id];
        }
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା current_device_id() ଅଚଳ
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            ଯଦି (it != _thread2dev_map.end())
                ଫେରନ୍ତୁ it->second;
            ଫେରନ୍ତୁ DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        ଶୂନ୍ୟ select_device(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା device_ଗଣନା() { ଫେରନ୍ତୁ _devs.size(); }

        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_device_id(ଅଚଳ sycl::device &dev)
        {
            unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = 0;
            ପାଇଁ (auto &dev_item : _devs)
            {
                ଯଦି (*dev_item == dev)
                {
                    ଫେରନ୍ତୁ id;
                }
                id++;
            }
            ଫେରନ୍ତୁ -1;
        }

        ଧାଡ଼ିଭିତରେ std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            ଅକ୍ଷର* env = getenv("ONEAPI_DEVICE_SELECTOR");
            ଯଦି (env) {
                ଯଦି (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                ଅନ୍ୟଥା ଯଦି (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                ଅନ୍ୟଥା ଯଦି (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                ଅନ୍ୟଥା ଯଦି (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                ଅନ୍ୟଥା {
                    ଫିଙ୍ଗ std::runtime_error("invalid device filter: " + std::string(env));
                }
            } ଅନ୍ୟଥା {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                ଯଦି (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                ଅନ୍ୟଥା ଯଦି (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                ଅନ୍ୟଥା ଯଦି (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            ପାଇଁ (ଅଚଳ auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](ଅଚଳ sycl::device& d) {
                    ଫେରନ୍ତୁ d.is_gpu();
                });

                ଯଦି (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    ଜାରି;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_ତଳ_case;
                platform_name_ତଳ_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_ତଳ_case.begin(), ::tolower);

                ଯଦି (platform_name_ତଳ_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    ଜାରି;
                }

                result = platform_name;
            }

            ଯଦି (result.empty())
                ଫିଙ୍ଗ std::runtime_error("can not find preferred GPU platform");

            ଫେରନ୍ତୁ result;
        }

        template <ଶ୍ରେଣୀ DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<ପୂର୍ଣ୍ଣସଂଖ୍ୟା, DeviceSelector, ଅଚଳ sycl::device &>>
        select_device(ଅଚଳ DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ସ୍ଥିର dev_mgr &instance()
        {
            ସ୍ଥିର dev_mgr d_m;
            ଫେରନ୍ତୁ d_m;
        }
        dev_mgr(ଅଚଳ dev_mgr &) = delete;
        dev_mgr &operator=(ଅଚଳ dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    ବ୍ୟକ୍ତିଗତ:
        mutable std::recursive_mutex m_mutex;
        ସ୍ଥିର ତାର୍କିକ compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            ଯଦି(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) ଫେରନ୍ତୁ ସତ୍ୟ;
            ଯଦି(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) ଫେରନ୍ତୁ ମିଥ୍ୟା;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            ଫେରନ୍ତୁ prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା convert_backend_index(std::string & backend) {
            ଯଦି (backend == "ext_oneapi_level_zero:gpu") ଫେରନ୍ତୁ 0;
            ଯଦି (backend == "opencl:gpu") ଫେରନ୍ତୁ 1;
            ଯଦି (backend == "ext_oneapi_cuda:gpu") ଫେରନ୍ତୁ 2;
            ଯଦି (backend == "ext_oneapi_hip:gpu") ଫେରନ୍ତୁ 3;
            ଯଦି (backend == "opencl:cpu") ଫେରନ୍ତୁ 4;
            ଯଦି (backend == "opencl:acc") ଫେରନ୍ତୁ 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ସ୍ଥିର ତାର୍କିକ compare_backend(std::string &backend1, std::string &backend2) {
            ଫେରନ୍ତୁ convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            ଯଦି (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            ଯେତେବେଳେ (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                ଯଦି (platform_name.compare(preferred_platform_name) != 0) {
                    ଜାରି;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                ପାଇଁ (ଅଚଳ auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> ଚାବିଗୁଡ଼ିକ;
            ପାଇଁ(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                ଚାବିଗୁଡ଼ିକ.push_back(it->first);
            }
            std::sort(ଚାବିଗୁଡ଼ିକ.begin(), ଚାବିଗୁଡ଼ିକ.end(), compare_backend);

            ପାଇଁ (auto &ଚାବି : ଚାବିଗୁଡ଼ିକ) {
                std::vector<sycl::device> devs = backend_devices[ଚାବି];
                std::sort(devs.begin(), devs.end(), compare_dev);
                ପାଇଁ (ଅଚଳ auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            ପାଇଁ (auto &dev : sycl_all_devs)
            {
                ଯଦି (dev == default_device)
                {
                    ଜାରି;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                ଯଦି (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        ଶୂନ୍ୟ check_id(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id) ଅଚଳ
        {
            ଯଦି (id >= _devs.size())
            {
                ଫିଙ୍ଗ std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ଅଚଳ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା> _thread2dev_map;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା _cpu_device = -1;
    };

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ sycl::queue &get_default_queue()
    {
        ଫେରନ୍ତୁ dev_mgr::instance().current_device().default_queue();
    }

    ନାମାକାଶ detail
    {
        ଗଣନା ଶ୍ରେଣୀ pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ସ୍ଥିର pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              ଅଚଳ ଶୂନ୍ୟ *ptr)
        {
            ବାଛ (sycl::get_pointer_type(ptr, q.get_context()))
            {
            ସ୍ଥିତି sycl::usm::alloc::unknown:
                ଫେରନ୍ତୁ pointer_access_attribute::host_only;
            ସ୍ଥିତି sycl::usm::alloc::device:
                ଫେରନ୍ତୁ pointer_access_attribute::device_only;
            ସ୍ଥିତି sycl::usm::alloc::shared:
            ସ୍ଥିତି sycl::usm::alloc::host:
                ଫେରନ୍ତୁ pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        ଧାଡ଼ିଭିତରେ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned ଅକ୍ଷର)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned ଅକ୍ଷର>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            ଫେରନ୍ତୁ (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        ଧାଡ଼ିଭିତରେ constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned ଅକ୍ଷର>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(ଆକାର...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            ଫେରନ୍ତୁ get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        ଶ୍ରେଣୀ mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#ଯଦି defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#ନଚେତଯଦି defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    ଖାଲି,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#ଅନ୍ୟଥା
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        ସର୍ବସାଧାରଣ:
            using buffer_id_t = ପୂର୍ଣ୍ଣସଂଖ୍ୟା;

            ସଂରଚନା allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#ଯଦି defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#ନଚେତଯଦି defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#ଅନ୍ୟଥା
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(ଅଚଳ mem_mgr &) = delete;
            mem_mgr &operator=(ଅଚଳ mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            ଶୂନ୍ୟ *mem_alloc(size_t size)
            {
                ଯଦି (!size)
                    ଫେରନ୍ତୁ nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                ଯଦି (next_free + size > mapped_address_space + mapped_region_size)
                {
                    ଫିଙ୍ଗ std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                ଶୂନ୍ୟ *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                ଫେରନ୍ତୁ result;
            }

            /// Deallocate
            ଶୂନ୍ୟ mem_free(ଅଚଳ ଶୂନ୍ୟ *ptr)
            {
                ଯଦି (!ptr)
                    ଫେରନ୍ତୁ;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(ଅଚଳ ଶୂନ୍ୟ *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                ଫେରନ୍ତୁ it->second;
            }

            /// Check if the pointer represents device pointer or not.
            ତାର୍କିକ is_device_ptr(ଅଚଳ ଶୂନ୍ୟ *ptr) ଅଚଳ
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                ଫେରନ୍ତୁ (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            ସ୍ଥିର mem_mgr &instance()
            {
                ସ୍ଥିର mem_mgr m;
                ଫେରନ୍ତୁ m;
            }

        ବ୍ୟକ୍ତିଗତ:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            ଅଚଳ size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            ଅଚଳ size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ଅଚଳ size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(ଅଚଳ ଶୂନ୍ୟ *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ଅଚଳ byte_t *>(ptr)));
                ଯଦି (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ଫିଙ୍ଗ std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ଅଚଳ allocation &alloc = it->second;
                ଯଦି (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ଫିଙ୍ଗ std::runtime_error("invalid virtual pointer");
                }
                ଫେରନ୍ତୁ it;
            }
        };

        template <ଶ୍ରେଣୀ T, memory_region Memory, size_t Dimension>
        ଶ୍ରେଣୀ accessor;
        template <memory_region Memory, ଶ୍ରେଣୀ T = byte_t>
        ଶ୍ରେଣୀ memory_traits
        {
        ସର୍ବସାଧାରଣ:
            ସ୍ଥିର constexpr sycl::access::target target =
                sycl::access::target::device;
            ସ୍ଥିର constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::ପଠ
                                     : sycl::access_mode::read_write;
            ସ୍ଥିର constexpr size_t type_size = ଆକାର(T);
            using element_t =
                typename std::conditional<Memory == constant, ଅଚଳ T, T>::type;
            using ମୂଲ୍ୟ_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<ମୂଲ୍ୟ_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ *dpct_malloc(size_t size, sycl::queue &q)
        {
            ଫେରନ୍ତୁ sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            ଫେରନ୍ତୁ dpct_malloc(pitch * y * z, q);
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
        template <typename ମୂଲ୍ୟ_T>
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ sycl::event dpct_memset(sycl::queue &q, ଶୂନ୍ୟ *dev_ptr,
                                              ମୂଲ୍ୟ_T value, size_t size)
        {
            ଫେରନ୍ତୁ q.fill(dev_ptr, value, size);
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
        template <typename ମୂଲ୍ୟ_T>
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, ମୂଲ୍ୟ_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned ଅକ୍ଷର *data_surface = (unsigned ଅକ୍ଷର *)data.get_data_ptr();
            ପାଇଁ (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned ଅକ୍ଷର *data_ptr = data_surface;
                ପାଇଁ (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            ଫେରନ୍ତୁ event_list;
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
        template <typename ମୂଲ୍ୟ_T>
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, ଶୂନ୍ୟ *ptr, size_t pitch, ମୂଲ୍ୟ_T val, size_t x,
                    size_t y)
        {
            ଫେରନ୍ତୁ dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ସ୍ଥିର memcpy_direction deduce_memcpy_direction(sycl::queue &q, ଶୂନ୍ୟ *to_ptr,
                                                        ଅଚଳ ଶୂନ୍ୟ *from_ptr,
                                                        memcpy_direction dir)
        {
            ବାଛ (dir)
            {
            ସ୍ଥିତି memcpy_direction::host_to_host:
            ସ୍ଥିତି memcpy_direction::host_to_device:
            ସ୍ଥିତି memcpy_direction::device_to_host:
            ସ୍ଥିତି memcpy_direction::device_to_device:
                ଫେରନ୍ତୁ dir;
            ସ୍ଥିତି memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ସ୍ଥିର ଅଚଳ memcpy_direction
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
                ଫେରନ୍ତୁ direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫିଙ୍ଗ std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ସ୍ଥିର sycl::event
        dpct_memcpy(sycl::queue &q, ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr, size_t size,
                    memcpy_direction direction,
                    ଅଚଳ std::vector<sycl::event> &dep_events = {})
        {
            ଯଦି (!size)
                ଫେରନ୍ତୁ sycl::event{};
            ଫେରନ୍ତୁ q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            ଫେରନ୍ତୁ slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            ଫେରନ୍ତୁ slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ଅଚଳ std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            ଶ୍ରେଣୀ host_buffer
            {
                ଶୂନ୍ୟ *_buf;
                size_t _size;
                sycl::queue &_q;
                ଅଚଳ std::vector<sycl::event> &_deps; // free operation depends

            ସର୍ବସାଧାରଣ:
                host_buffer(size_t size, sycl::queue &q,
                            ଅଚଳ std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                ଶୂନ୍ୟ *get_ptr() ଅଚଳ { ଫେରନ୍ତୁ _buf; }
                size_t get_size() ଅଚଳ { ଫେରନ୍ତୁ _size; }
                ~host_buffer()
                {
                    ଯଦି (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::ମୁକ୍ତ(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned ଅକ୍ଷର *to_surface =
                (unsigned ଅକ୍ଷର *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            ଅଚଳ unsigned ଅକ୍ଷର *from_surface =
                (ଅଚଳ unsigned ଅକ୍ଷର *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            ଯଦି (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                ଫେରନ୍ତୁ {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            ବାଛ (direction)
            {
            ସ୍ଥିତି host_to_host:
                ପାଇଁ (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned ଅକ୍ଷର *to_ptr = to_surface;
                    ଅଚଳ unsigned ଅକ୍ଷର *from_ptr = from_surface;
                    ଯଦି (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    ଅନ୍ୟଥା
                    {
                        ପାଇଁ (size_t y = 0; y < size.get(1); ++y)
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
                ଭାଙ୍ଗ;
            ସ୍ଥିତି host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                ଯଦି (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                ଅନ୍ୟଥା
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
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି device_to_host:
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
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<ଶ୍ରେଣୀ dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଫିଙ୍ଗ std::runtime_error("dpct_memcpy: invalid direction value");
            }
            ଫେରନ୍ତୁ event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            ଫେରନ୍ତୁ dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            ଫେରନ୍ତୁ dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        ନାମାକାଶ deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            ଶ୍ରେଣୀ usm_allocator
            {
            ବ୍ୟକ୍ତିଗତ:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            ସର୍ବସାଧାରଣ:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    ଅଚଳ typename std::allocator_traits<Alloc>::value_type &;
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
                ସଂରଚନା rebind
                {
                    ପ୍ରକାରବ୍ୟାଖ୍ୟା usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(ଅଚଳ usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { ଫେରନ୍ତୁ &r; }
                const_pointer address(const_reference r) { ଫେରନ୍ତୁ &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    ଫେରନ୍ତୁ std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                ଶୂନ୍ୟ deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ଅଚଳ
                {
                    ଫେରନ୍ତୁ std::allocator_traits<Alloc>::max_size(_impl);
                }
                ତାର୍କିକ operator==(ଅଚଳ usm_allocator &other) ଅଚଳ { ଫେରନ୍ତୁ _impl == other._impl; }
                ତାର୍କିକ operator!=(ଅଚଳ usm_allocator &other) ଅଚଳ { ଫେରନ୍ତୁ _impl != other._impl; }
            };

        } // namespace deprecated

        ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dpct_free(ଶୂନ୍ୟ *ptr,
                              ଅଚଳ sycl::queue &q)
        {
            ଯଦି (ptr)
            {
                sycl::ମୁକ୍ତ(ptr, q.get_context());
            }
        }

        template <typename T>
        ଧାଡ଼ିଭିତରେ auto get_memory(ଅଚଳ ଶୂନ୍ୟ *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<ଶୂନ୍ୟ *>(x));
            ଫେରନ୍ତୁ new_x;
        }

        template <typename T>
        ଧାଡ଼ିଭିତରେ typename DataType<T>::T2 get_ମୂଲ୍ୟ(ଅଚଳ T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            ଯଦି (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (ଶୂନ୍ୟ *)&s_h, (ଅଚଳ ଶୂନ୍ୟ *)s, ଆକାର(T), device_to_host)
                    .wait();
            ଅନ୍ୟଥା
                s_h = *reinterpret_cast<ଅଚଳ Ty *>(s);
            ଫେରନ୍ତୁ s_h;
        }

    } // namespace detail

    template <typename T>
    ଧାଡ଼ିଭିତରେ auto get_ମୂଲ୍ୟ(ଅଚଳ T *s, sycl::queue &q)
    {
        ଫେରନ୍ତୁ detail::get_ମୂଲ୍ୟ(s, q);
    }

    ନାମାକାଶ detail
    {
    template <ଶ୍ରେଣୀ Ta, ଶ୍ରେଣୀ Tb, ଶ୍ରେଣୀ Tc, ଶ୍ରେଣୀ Ts>
    ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m,
                          ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ଶୂନ୍ୟ * alpha, ଅଚଳ ଶୂନ୍ୟ * a, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda, ଅଚଳ ଶୂନ୍ୟ * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldb,
                          ଅଚଳ ଶୂନ୍ୟ * beta, ଶୂନ୍ୟ * c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc) {
        Ts   alpha_ମୂଲ୍ୟ_2 = dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ Ts *>(alpha), q);
        Ts   beta_ମୂଲ୍ୟ  = dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ Ts *>(beta), q);
        auto data_a      = get_memory<ଅଚଳ Ta>(a);
        auto data_b      = get_memory<ଅଚଳ Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_ମୂଲ୍ୟ_2, data_a,
                                               lda, data_b, ldb, beta_ମୂଲ୍ୟ, data_c, ldc);
    }

        template <typename VecT, ଶ୍ରେଣୀ ଦ୍ୱିମିକ_Operation, ଶ୍ରେଣୀ = ଶୂନ୍ୟ>
        ଶ୍ରେଣୀ vectorized_ଦ୍ୱିମିକ
        {
        ସର୍ବସାଧାରଣ:
            ଧାଡ଼ିଭିତରେ VecT operator()(VecT a, VecT b, ଅଚଳ ଦ୍ୱିମିକ_Operation ଦ୍ୱିମିକ_op)
            {
                VecT v4;
                ପାଇଁ (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ଦ୍ୱିମିକ_op(a[i], b[i]);
                }
                ଫେରନ୍ତୁ v4;
            }
        };

        template <typename VecT, ଶ୍ରେଣୀ ଦ୍ୱିମିକ_Operation>
        ଶ୍ରେଣୀ vectorized_ଦ୍ୱିମିକ<
            VecT, ଦ୍ୱିମିକ_Operation,
            std::void_t<std::invoke_result_t<ଦ୍ୱିମିକ_Operation, VecT, VecT>>>
        {
        ସର୍ବସାଧାରଣ:
            ଧାଡ଼ିଭିତରେ VecT operator()(VecT a, VecT b, ଅଚଳ ଦ୍ୱିମିକ_Operation ଦ୍ୱିମିକ_op)
            {
                ଫେରନ୍ତୁ ଦ୍ୱିମିକ_op(a, b).template as<VecT>();
            }
        };

        template <ଶ୍ରେଣୀ Ta, ଶ୍ରେଣୀ Tb, ଶ୍ରେଣୀ Tc, ଶ୍ରେଣୀ Ts>
        ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ଶୂନ୍ୟ * alpha, ଅଚଳ ଶୂନ୍ୟ ** a, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda, ଅଚଳ ଶୂନ୍ୟ ** b,
                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldb, ଅଚଳ ଶୂନ୍ୟ * beta, ଶୂନ୍ୟ ** c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc, ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_size,
                                    matrix_info_t<ଦଶମିକ> * matrix_info) {
            Ts alpha_ମୂଲ୍ୟ_2 = dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ Ts *>(alpha), q);
            Ts beta_ମୂଲ୍ୟ = dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->ମୂଲ୍ୟ_info[0] = alpha_ମୂଲ୍ୟ_2;
            matrix_info->ମୂଲ୍ୟ_info[1] = beta_ମୂଲ୍ୟ;
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
                reinterpret_cast<Ts *>(matrix_info->ମୂଲ୍ୟ_info), reinterpret_cast<ଅଚଳ Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<ଅଚଳ Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->ମୂଲ୍ୟ_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <ଶ୍ରେଣୀ Ta, ଶ୍ରେଣୀ Tb, ଶ୍ରେଣୀ Tc, ଶ୍ରେଣୀ Ts>
        ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ଶୂନ୍ୟ * alpha, ଅଚଳ ଶୂନ୍ୟ * a, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda,
                                    long long ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_a, ଅଚଳ ଶୂନ୍ୟ * b, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldb, long long ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_b,
                                    ଅଚଳ ଶୂନ୍ୟ * beta, ଶୂନ୍ୟ * c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc, long long ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_size) {
            Ts alpha_ମୂଲ୍ୟ_2 = dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ Ts *>(alpha), q);
            Ts beta_ମୂଲ୍ୟ = dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ Ts *>(beta), q);
            auto data_a = get_memory<ଅଚଳ Ta>(a);
            auto data_b = get_memory<ଅଚଳ Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_ମୂଲ୍ୟ_2,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_ମୂଲ୍ୟ,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, ଶ୍ରେଣୀ ଦ୍ୱିମିକ_Operation>
    ଧାଡ଼ିଭିତରେ unsigned vectorized_ଦ୍ୱିମିକ(unsigned a, unsigned b,
                                      ଅଚଳ ଦ୍ୱିମିକ_Operation ଦ୍ୱିମିକ_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_ଦ୍ୱିମିକ<VecT, ଦ୍ୱିମିକ_Operation>()(v2, v3, ଦ୍ୱିମିକ_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        ଫେରନ୍ତୁ v0;
    }

    ସ୍ଥିର ଶୂନ୍ୟ async_dpct_memcpy(ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା select_device(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id)
    {
        dev_mgr::instance().select_device(id);
        ଫେରନ୍ତୁ id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା mask,
                               unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା logical_sub_group_size = 32)
    {
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id = g.get_local_linear_id();
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା target_offset = (id % logical_sub_group_size) ^ mask;
        ଫେରନ୍ତୁ sycl::select_from_group(g, x,
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
      ଫେରନ୍ତୁ sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    ଧାଡ଼ିଭିତରେ auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      ଫେରନ୍ତୁ res;
    }

    ସଂରଚନା sub_sat
    {
        template <typename T>
        auto operator()(ଅଚଳ T x, ଅଚଳ T y) ଅଚଳ
        {
            ଫେରନ୍ତୁ sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    ଧାଡ଼ିଭିତରେ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        ଫେରନ୍ତୁ v0;
    }

    ଧାଡ଼ିଭିତରେ ଦଶମିକ pow(ଅଚଳ ଦଶମିକ a, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା b) { ଫେରନ୍ତୁ sycl::pown(a, b); }
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ pow(ଅଚଳ ଦ୍ୱିଗୁଣ a, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା b) { ଫେରନ୍ତୁ sycl::pown(a, b); }
    ଧାଡ଼ିଭିତରେ ଦଶମିକ pow(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) { ଫେରନ୍ତୁ sycl::pow(a, b); }
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ pow(ଅଚଳ ଦ୍ୱିଗୁଣ a, ଅଚଳ ଦ୍ୱିଗୁଣ b) { ଫେରନ୍ତୁ sycl::pow(a, b); }
    template <typename T, typename U>
    ଧାଡ଼ିଭିତରେ typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(ଅଚଳ T a, ଅଚଳ U b)
    {
        ଫେରନ୍ତୁ sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    ଧାଡ଼ିଭିତରେ typename std::enable_if_t<!std::is_floating_point_v<T>, ଦ୍ୱିଗୁଣ>
    pow(ଅଚଳ T a, ଅଚଳ U b)
    {
        ଫେରନ୍ତୁ sycl::pow(static_cast<ଦ୍ୱିଗୁଣ>(a), static_cast<ଦ୍ୱିଗୁଣ>(b));
    }

    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ min(ଅଚଳ ଦ୍ୱିଗୁଣ a, ଅଚଳ ଦଶମିକ b)
    {
        ଫେରନ୍ତୁ sycl::fmin(a, static_cast<ଦ୍ୱିଗୁଣ>(b));
    }
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ min(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦ୍ୱିଗୁଣ b)
    {
        ଫେରନ୍ତୁ sycl::fmin(static_cast<ଦ୍ୱିଗୁଣ>(a), b);
    }
    ଧାଡ଼ିଭିତରେ ଦଶମିକ min(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) { ଫେରନ୍ତୁ sycl::fmin(a, b); }
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ min(ଅଚଳ ଦ୍ୱିଗୁଣ a, ଅଚଳ ଦ୍ୱିଗୁଣ b) { ଫେରନ୍ତୁ sycl::fmin(a, b); }
    ଧାଡ଼ିଭିତରେ std::uint32_t min(ଅଚଳ std::uint32_t a, ଅଚଳ std::int32_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, static_cast<std::uint32_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint32_t min(ଅଚଳ std::int32_t a, ଅଚଳ std::uint32_t b)
    {
        ଫେରନ୍ତୁ sycl::min(static_cast<std::uint32_t>(a), b);
    }
    ଧାଡ଼ିଭିତରେ std::int32_t min(ଅଚଳ std::int32_t a, ଅଚଳ std::int32_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint32_t min(ଅଚଳ std::uint32_t a, ଅଚଳ std::uint32_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::uint64_t a, ଅଚଳ std::int64_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::int64_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ଧାଡ଼ିଭିତରେ std::int64_t min(ଅଚଳ std::int64_t a, ଅଚଳ std::int64_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::uint64_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::uint64_t a, ଅଚଳ std::int32_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::int32_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::uint64_t a, ଅଚଳ std::uint32_t b)
    {
        ଫେରନ୍ତୁ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t min(ଅଚଳ std::uint32_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ max(ଅଚଳ ଦ୍ୱିଗୁଣ a, ଅଚଳ ଦଶମିକ b)
    {
        ଫେରନ୍ତୁ sycl::fmax(a, static_cast<ଦ୍ୱିଗୁଣ>(b));
    }
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ max(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦ୍ୱିଗୁଣ b)
    {
        ଫେରନ୍ତୁ sycl::fmax(static_cast<ଦ୍ୱିଗୁଣ>(a), b);
    }
    ଧାଡ଼ିଭିତରେ ଦଶମିକ max(ଅଚଳ ଦଶମିକ a, ଅଚଳ ଦଶମିକ b) { ଫେରନ୍ତୁ sycl::fmax(a, b); }
    ଧାଡ଼ିଭିତରେ ଦ୍ୱିଗୁଣ max(ଅଚଳ ଦ୍ୱିଗୁଣ a, ଅଚଳ ଦ୍ୱିଗୁଣ b) { ଫେରନ୍ତୁ sycl::fmax(a, b); }
    ଧାଡ଼ିଭିତରେ std::uint32_t max(ଅଚଳ std::uint32_t a, ଅଚଳ std::int32_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, static_cast<std::uint32_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint32_t max(ଅଚଳ std::int32_t a, ଅଚଳ std::uint32_t b)
    {
        ଫେରନ୍ତୁ sycl::max(static_cast<std::uint32_t>(a), b);
    }
    ଧାଡ଼ିଭିତରେ std::int32_t max(ଅଚଳ std::int32_t a, ଅଚଳ std::int32_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint32_t max(ଅଚଳ std::uint32_t a, ଅଚଳ std::uint32_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::uint64_t a, ଅଚଳ std::int64_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::int64_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ଧାଡ଼ିଭିତରେ std::int64_t max(ଅଚଳ std::int64_t a, ଅଚଳ std::int64_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::uint64_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::uint64_t a, ଅଚଳ std::int32_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::int32_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::uint64_t a, ଅଚଳ std::uint32_t b)
    {
        ଫେରନ୍ତୁ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    ଧାଡ଼ିଭିତରେ std::uint64_t max(ଅଚଳ std::uint32_t a, ଅଚଳ std::uint64_t b)
    {
        ଫେରନ୍ତୁ sycl::max(static_cast<std::uint64_t>(a), b);
    }

    ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ
    has_capability_or_fail(ଅଚଳ sycl::device &dev,
                           ଅଚଳ std::initializer_list<sycl::aspect> &props)
    {
        ପାଇଁ (ଅଚଳ auto &it : props)
        {
            ଯଦି (dev.has(it))
                ଜାରି;
            ବାଛ (it)
            {
            ସ୍ଥିତି sycl::aspect::fp64:
                ଫିଙ୍ଗ std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ଭାଙ୍ଗ;
            ସ୍ଥିତି sycl::aspect::fp16:
                ଫିଙ୍ଗ std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ଭାଙ୍ଗ;
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
#define __SYCL_ASPECT(ASPECT, ID) \
    ସ୍ଥିତି sycl::aspect::ASPECT:    \
        ଫେରନ୍ତୁ #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect Aspect_ଅଙ୍କ) -> std::string
                {
                    ବାଛ (Aspect_ଅଙ୍କ)
                    {
#ଅନ୍ତର୍ଭାଵ <sycl/info/aspects.def>
#ଅନ୍ତର୍ଭାଵ <sycl/info/aspects_deprecated.def>
                    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                        ଫେରନ୍ତୁ "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                ଫିଙ୍ଗ std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            ଭାଙ୍ଗ;
        }
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_current_device_id()
    {
        ଫେରନ୍ତୁ dev_mgr::instance().current_device_id();
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ device_ext &get_current_device()
    {
        ଫେରନ୍ତୁ dev_mgr::instance().current_device();
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ device_ext &get_device(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା id)
    {
        ଫେରନ୍ତୁ dev_mgr::instance().get_device(id);
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ sycl::queue &get_in_order_queue()
    {
        ଫେରନ୍ତୁ dev_mgr::instance().current_device().in_order_queue();
    }

    ସ୍ଥିର sycl::event
    dpct_memcpy(sycl::queue &q, ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr, size_t size,
                memcpy_direction direction,
                ଅଚଳ std::vector<sycl::event> &dep_events = {})
    {
        ଯଦି (!size)
            ଫେରନ୍ତୁ sycl::event{};
        ଫେରନ୍ତୁ q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        ଫେରନ୍ତୁ slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        ଫେରନ୍ତୁ slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ଅଚଳ std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        ଶ୍ରେଣୀ host_buffer
        {
            ଶୂନ୍ୟ *_buf;
            size_t _size;
            sycl::queue &_q;
            ଅଚଳ std::vector<sycl::event> &_deps; // free operation depends

        ସର୍ବସାଧାରଣ:
            host_buffer(size_t size, sycl::queue &q,
                        ଅଚଳ std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            ଶୂନ୍ୟ *get_ptr() ଅଚଳ { ଫେରନ୍ତୁ _buf; }
            size_t get_size() ଅଚଳ { ଫେରନ୍ତୁ _size; }
            ~host_buffer()
            {
                ଯଦି (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::ମୁକ୍ତ(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned ଅକ୍ଷର *to_surface =
            (unsigned ଅକ୍ଷର *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        ଅଚଳ unsigned ଅକ୍ଷର *from_surface =
            (ଅଚଳ unsigned ଅକ୍ଷର *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        ଯଦି (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            ଫେରନ୍ତୁ {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        ବାଛ (direction)
        {
        ସ୍ଥିତି host_to_host:
            ପାଇଁ (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned ଅକ୍ଷର *to_ptr = to_surface;
                ଅଚଳ unsigned ଅକ୍ଷର *from_ptr = from_surface;
                ଯଦି (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                ଅନ୍ୟଥା
                {
                    ପାଇଁ (size_t y = 0; y < size.get(1); ++y)
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
            ଭାଙ୍ଗ;
        ସ୍ଥିତି host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            ଯଦି (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            ଅନ୍ୟଥା
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
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି device_to_host:
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
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<ଶ୍ରେଣୀ dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫିଙ୍ଗ std::runtime_error("dpct_memcpy: invalid direction value");
        }
        ଫେରନ୍ତୁ event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        ଫେରନ୍ତୁ dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ଶୂନ୍ୟ *to_ptr, ଅଚଳ ଶୂନ୍ୟ *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        ଫେରନ୍ତୁ dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n,
                     ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ଶୂନ୍ୟ * alpha, ଅଚଳ ଶୂନ୍ୟ * a, library_data_t a_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda, ଅଚଳ ଶୂନ୍ୟ * b,
                     library_data_t b_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldb, ଅଚଳ ଶୂନ୍ୟ * beta, ଶୂନ୍ୟ * c, library_data_t c_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc,
                     library_data_t scaling_type) {
        ଯଦି (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        ଅନ୍ୟଥା ଯଦି (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t ଚାବି =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ବାଛ (ଚାବି)
        {
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<ଦଶମିକ, ଦଶମିକ, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<ଦଶମିକ>, std::complex<ଦଶମିକ>,
                              std::complex<ଦଶମିକ>, std::complex<ଦଶମିକ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<ଦ୍ୱିଗୁଣ>, std::complex<ଦ୍ୱିଗୁଣ>,
                              std::complex<ଦ୍ୱିଗୁଣ>, std::complex<ଦ୍ୱିଗୁଣ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
#ifdef __INTEL_MKL__
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ଦଶମିକ alpha_ମୂଲ୍ୟ_2 =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ ଦଶମିକ *>(alpha), q);
            ଦଶମିକ beta_ମୂଲ୍ୟ =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ ଦଶମିକ *>(beta), q);
            sycl::half alpha_half(alpha_ମୂଲ୍ୟ_2);
            sycl::half beta_half(beta_ମୂଲ୍ୟ);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ଦଶମିକ alpha_float =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ std::int32_t *>(alpha), q);
            ଦଶମିକ beta_float =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            ଭାଙ୍ଗ;
        }
#endif // __INTEL_MKL__
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫିଙ୍ଗ std::runtime_error("the combination of data type is unsupported");
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
    ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m,
                           ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ଶୂନ୍ୟ * alpha, ଅଚଳ ଶୂନ୍ୟ * a[], library_data_t a_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda,
                           ଅଚଳ ଶୂନ୍ୟ * b[], library_data_t b_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldb, ଅଚଳ ଶୂନ୍ୟ * beta, ଶୂନ୍ୟ * c[],
                           library_data_t c_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc, ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_size, library_data_t scaling_type,
                           matrix_info_t<ଦଶମିକ> * matrix_info) {
        std::uint64_t ଚାବି =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ବାଛ (ଚାବି)
        {
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ଦଶମିକ, ଦଶମିକ, ଦଶମିକ, ଦଶମିକ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
#ifdef __INTEL_MKL__
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
#endif
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ଦଶମିକ alpha_float =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ std::int32_t *>(alpha), q);
            ଦଶମିକ beta_float =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ଦଶମିକ alpha_ମୂଲ୍ୟ_2 =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ ଦଶମିକ *>(alpha), q);
            ଦଶମିକ beta_ମୂଲ୍ୟ =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ ଦଶମିକ *>(beta), q);
            sycl::half alpha_half(alpha_ମୂଲ୍ୟ_2);
            sycl::half beta_half(beta_ମୂଲ୍ୟ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫିଙ୍ଗ std::runtime_error("the combination of data type is unsupported");
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
    ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, ପୂର୍ଣ୍ଣସଂଖ୍ୟା m,
                           ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା k, ଅଚଳ ଶୂନ୍ୟ * alpha, ଅଚଳ ଶୂନ୍ୟ * a, library_data_t a_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା lda,
                           long long ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_a, ଅଚଳ ଶୂନ୍ୟ * b, library_data_t b_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldb,
                           long long ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_b, ଅଚଳ ଶୂନ୍ୟ * beta, ଶୂନ୍ୟ * c, library_data_t c_type, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ldc,
                           long long ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା batch_size, library_data_t scaling_type) {
        ଯଦି (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        ଅନ୍ୟଥା ଯଦି (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t ଚାବି =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ବାଛ (ଚାବି)
        {
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ଦଶମିକ, ଦଶମିକ, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ, ଦ୍ୱିଗୁଣ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<ଦଶମିକ>, std::complex<ଦଶମିକ>,
                                    std::complex<ଦଶମିକ>, std::complex<ଦଶମିକ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<ଦ୍ୱିଗୁଣ>, std::complex<ଦ୍ୱିଗୁଣ>,
                                    std::complex<ଦ୍ୱିଗୁଣ>, std::complex<ଦ୍ୱିଗୁଣ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
#ifdef __INTEL_MKL__
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            ଭାଙ୍ଗ;
        }
#endif
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ଦଶମିକ, ଦଶମିକ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ସ୍ଥିତି detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ଦଶମିକ alpha_ମୂଲ୍ୟ_2 =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ ଦଶମିକ *>(alpha), q);
            ଦଶମିକ beta_ମୂଲ୍ୟ =
                dpct::get_ମୂଲ୍ୟ(reinterpret_cast<ଅଚଳ ଦଶମିକ *>(beta), q);
            sycl::half alpha_half(alpha_ମୂଲ୍ୟ_2);
            sycl::half beta_half(beta_ମୂଲ୍ୟ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            ଭାଙ୍ଗ;
        }
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫିଙ୍ଗ std::runtime_error("the combination of data type is unsupported");
        }
    }

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ
    async_dpct_memcpy(ଶୂନ୍ୟ *to_ptr, size_t to_pitch, ଅଚଳ ଶୂନ୍ୟ *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<ସଂରଚନା err0_tag, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>;
    using err1 = detail::generic_error_type<ସଂରଚନା err1_tag, ପୂର୍ଣ୍ଣସଂଖ୍ୟା>;

    ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ dpct_free(ଶୂନ୍ୟ *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <ଶ୍ରେଣୀ T, memory_region Memory, size_t Dimension> ଶ୍ରେଣୀ accessor;
    template <ଶ୍ରେଣୀ T, memory_region Memory> ଶ୍ରେଣୀ accessor<T, Memory, 3> {
    ସର୍ବସାଧାରଣ:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, ଅଚଳ sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ଅଚଳ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ଅଚଳ accessor_t &acc, ଅଚଳ sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) ଅଚଳ {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            ଫେରନ୍ତୁ accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() ଅଚଳ { ଫେରନ୍ତୁ _data; }

    ବ୍ୟକ୍ତିଗତ:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <ଶ୍ରେଣୀ T, memory_region Memory> ଶ୍ରେଣୀ accessor<T, Memory, 2> {
    ସର୍ବସାଧାରଣ:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, ଅଚଳ sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ଅଚଳ accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ଅଚଳ accessor_t &acc, ଅଚଳ sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) ଅଚଳ {
            ଫେରନ୍ତୁ _data + _range.get(1) * index;
        }

        pointer_t get_ptr() ଅଚଳ { ଫେରନ୍ତୁ _data; }

    ବ୍ୟକ୍ତିଗତ:
        pointer_t _data;
        sycl::range<2> _range;
    };

    ନାମାକାଶ detail {
        /// Device variable with address space of shared, global or constant.
        template <ଶ୍ରେଣୀ T, memory_region Memory, size_t Dimension> ଶ୍ରେଣୀ device_memory {
        ସର୍ବସାଧାରଣ:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using ମୂଲ୍ୟ_t = typename detail::memory_traits<Memory, T>::ମୂଲ୍ୟ_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(ଅଚଳ sycl::range<Dimension> &in_range,
                        std::initializer_list<ମୂଲ୍ୟ_t> &&init_list)
                : device_memory(in_range) {
                ପ୍ରତିଜ୍ଞା(init_list.size() <= in_range.size());
                _host_ptr = (ମୂଲ୍ୟ_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * ଆକାର(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                ଅଚଳ typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<ମୂଲ୍ୟ_t>> &&init_list)
                : device_memory(in_range) {
                ପ୍ରତିଜ୍ଞା(init_list.size() <= in_range[0]);
                _host_ptr = (ମୂଲ୍ୟ_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                ପାଇଁ (auto sub_list : init_list) {
                    ପ୍ରତିଜ୍ଞା(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * ଆକାର(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(ଅଚଳ sycl::range<Dimension> &range_in)
                : _size(range_in.size() * ଆକାର(T)), _range(range_in),
                _reference(ମିଥ୍ୟା), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <ଶ୍ରେଣୀ... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                ଯଦି (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                ଯଦି (_host_ptr)
                    std::ମୁକ୍ତ(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            ଶୂନ୍ୟ init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            ଶୂନ୍ୟ init(sycl::queue &q) {
                ଯଦି (_device_ptr)
                    ଫେରନ୍ତୁ;
                ଯଦି (!_size)
                    ଫେରନ୍ତୁ;
                allocate_device(q);
                ଯଦି (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            ଶୂନ୍ୟ assign(ମୂଲ୍ୟ_t *src, size_t size) {
                ଏହି->~device_memory();
                ନୂତନ (ଏହି) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ମୂଲ୍ୟ_t *get_ptr() { ଫେରନ୍ତୁ get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            ମୂଲ୍ୟ_t *get_ptr(sycl::queue &q) {
                init(q);
                ଫେରନ୍ତୁ _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { ଫେରନ୍ତୁ _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                ଫେରନ୍ତୁ _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                ଫେରନ୍ତୁ dpct_accessor_t((T *)_device_ptr, _range);
            }

        ବ୍ୟକ୍ତିଗତ:
            device_memory(ମୂଲ୍ୟ_t *memory_ptr, size_t size)
                : _size(size), _range(size / ଆକାର(T)), _reference(ସତ୍ୟ),
                _device_ptr(memory_ptr) {}

            ଶୂନ୍ୟ allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                ଯଦି (Memory == shared) {
                    _device_ptr = (ମୂଲ୍ୟ_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    ଫେରନ୍ତୁ;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                ଯଦି (Memory == constant) {
                    _device_ptr = (ମୂଲ୍ୟ_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    ଫେରନ୍ତୁ;
                }
        #endif
        #endif
                _device_ptr = (ମୂଲ୍ୟ_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            ତାର୍କିକ _reference;
            ମୂଲ୍ୟ_t *_host_ptr;
            ମୂଲ୍ୟ_t *_device_ptr;
        };
        template <ଶ୍ରେଣୀ T, memory_region Memory>
        ଶ୍ରେଣୀ device_memory<T, Memory, 0> : ସର୍ବସାଧାରଣ device_memory<T, Memory, 1> {
        ସର୍ବସାଧାରଣ:
            using base = device_memory<T, Memory, 1>;
            using ମୂଲ୍ୟ_t = typename base::ମୂଲ୍ୟ_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(ଅଚଳ ମୂଲ୍ୟ_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <ଶ୍ରେଣୀ T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <ଶ୍ରେଣୀ T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <ଶ୍ରେଣୀ T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    ଧାଡ଼ିଭିତରେ T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ଫେରନ୍ତୁ atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    ଧାଡ଼ିଭିତରେ T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    ଫେରନ୍ତୁ atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    ଧାଡ଼ିଭିତରେ T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    ବାଛ (memoryOrder) {
        ସ୍ଥିତି sycl::memory_order::relaxed:
            ଫେରନ୍ତୁ atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        ସ୍ଥିତି sycl::memory_order::acq_rel:
            ଫେରନ୍ତୁ atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        ସ୍ଥିତି sycl::memory_order::seq_cst:
            ଫେରନ୍ତୁ atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ପ୍ରତିଜ୍ଞା(ମିଥ୍ୟା && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    ଧାଡ଼ିଭିତରେ T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    ଧାଡ଼ିଭିତରେ unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା byte_level_permute(
        unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା a, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା b, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା s) {
      unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      ଫେରନ୍ତୁ ret;
    }

    ଧାଡ଼ିଭିତରେ uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, ପୂର୍ଣ୍ଣସଂଖ୍ୟା mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      ଯଦି (mode >= 1 && mode <= 6) {
        ଫେରନ୍ତୁ byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } ଅନ୍ୟଥା ଯଦି (!mode) {
        ଫେରନ୍ତୁ byte_level_permute(low32, high32, sel);
      }
      ଫେରନ୍ତୁ 0;
    }

    template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nondefault_params, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_default_params, typename T>
    ଶ୍ରେଣୀ args_selector;

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
    template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_nondefault_params, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_default_params, typename R,
              typename... Ts>
    ଶ୍ରେଣୀ args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      ବ୍ୟକ୍ତିଗତ:
        ଶୂନ୍ୟ **kernel_params;
        ଅକ୍ଷର *args_buffer;

        template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା i> ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା account_for_default_params() {
            constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_total_params = ଆକାର...(Ts);
            ଯଦି constexpr (i >= n_nondefault_params) {
                ଫେରନ୍ତୁ n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ i;
            }
        }

      ସର୍ବସାଧାରଣ:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା params_ଅଙ୍କ = ଆକାର...(Ts);

      ବ୍ୟକ୍ତିଗତ:
        template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା i> ସ୍ଥିର constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା get_offset() {
            ଯଦି constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                ଫେରନ୍ତୁ 0;
            } ଅନ୍ୟଥା {
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା prev_off = get_offset<i - 1>();
                constexpr ପୂର୍ଣ୍ଣସଂଖ୍ୟା prev_past_end =
                    prev_off + ଆକାର(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                ଯଦି constexpr (prev_past_end % alignof(T) == 0) {
                    ଫେରନ୍ତୁ prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                ଅନ୍ୟଥା {
                    ଫେରନ୍ତୁ prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        ସ୍ଥିର ଅକ୍ଷର *get_args_buffer(ଶୂନ୍ୟ **extra) {
            ଯଦି (!extra)
                ଫେରନ୍ତୁ nullptr;
            ପାଇଁ (; (std::size_t)*extra != 0; ++extra) {
                ଯଦି ((std::size_t)*extra == 1) {
                    ଫେରନ୍ତୁ static_cast<ଅକ୍ଷର *>(*(extra + 1));
                }
            }
            ଫେରନ୍ତୁ nullptr;
        }

      ସର୍ବସାଧାରଣ:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(ଶୂନ୍ୟ **kernel_params, ଶୂନ୍ୟ **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା i> arg_type<i> &get() {
            ଯଦି (kernel_params) {
                ଫେରନ୍ତୁ *static_cast<arg_type<i> *>(kernel_params[i]);
            } ଅନ୍ୟଥା {
                ଫେରନ୍ତୁ *reinterpret_cast<arg_type<i> *>(args_buffer +
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
    ଶ୍ରେଣୀ kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        ସ୍ଥିର ଶୂନ୍ୟ launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        ସ୍ଥିର ଶୂନ୍ୟ set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_mem_size,
                                         queue_ptr que) {
            ଯଦି (que) {
                _que = que;
            } ଅନ୍ୟଥା {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::mutex kernel_function_ptr_map_mutex;

      ସର୍ବସାଧାରଣ:
        /// Variables for storing execution configuration.
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ thread_local sycl::queue *_que = nullptr;
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ thread_local unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ସ୍ଥିର ଧାଡ଼ିଭିତରେ std::map<
            ଅଚଳ ଶୂନ୍ୟ *,
            std::function<ଶୂନ୍ୟ(dim3, dim3, ଶୂନ୍ୟ **, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ସ୍ଥିର ଶୂନ୍ୟ register_kernel_ptr(
            ଅଚଳ ଶୂନ୍ୟ *func,
            std::function<ଶୂନ୍ୟ(dim3, dim3, ଶୂନ୍ୟ **, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା, queue_ptr)>
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
        ସ୍ଥିର std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, ଶୂନ୍ୟ>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_mem_size, queue_ptr que, ArgsT... args) {
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
        ସ୍ଥିର ଶୂନ୍ୟ launch(ଅଚଳ ଶୂନ୍ୟ *func, dim3 group_range, dim3 local_range,
                           ଶୂନ୍ୟ **args, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            ଯଦି (Iter == kernel_function_ptr_map.end()) {
                ଫିଙ୍ଗ std::runtime_error("dpct::launch() : no registered "
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
        ସ୍ଥିର std::enable_if_t<std::is_function_v<FuncT>, ଶୂନ୍ୟ>
        launch(FuncT *func, dim3 group_range, dim3 local_range, ଶୂନ୍ୟ **args,
               unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା local_mem_size, queue_ptr que) {
            constexpr size_t p_ଅଙ୍କ = args_selector<0, 0, FuncT>::params_ଅଙ୍କ;
            set_execution_config(group_range, local_range, local_mem_size, que);
            args_selector<p_ଅଙ୍କ, p_ଅଙ୍କ, FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_ଅଙ୍କ>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    template <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା remote_local_id,
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା logical_sub_group_size = 32) {
      unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      ଫେରନ୍ତୁ sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    ଶୂନ୍ୟ ldmatrix(uintptr_t addr, T* m, ତାର୍କିକ trans = ମିଥ୍ୟା, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = sg.get_local_linear_id();

      ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane_group8_row = lane / 8;
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane_group8_col = lane % 8;

      ଯଦି (!trans) {
        // calculate the source lane
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_lane = 2 * lane_group8_row;
        ଯଦି (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } ଅନ୍ୟଥା {
        // calculate the source lane
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    ଶୂନ୍ୟ ldmatrix(uintptr_t addr, T* m1, T* m2, ତାର୍କିକ trans = ମିଥ୍ୟା) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    ଶୂନ୍ୟ ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, ତାର୍କିକ trans = ମିଥ୍ୟା) {
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
    ସଂରଚନା MMAType {
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
    template <ପୂର୍ଣ୍ଣସଂଖ୍ୟା M, ପୂର୍ଣ୍ଣସଂଖ୍ୟା N, ପୂର୍ଣ୍ଣସଂଖ୍ୟା K, typename ABType, typename CDType>
    ଶୂନ୍ୟ mma(
        ଅସ୍ଥିର ଶୂନ୍ୟ** d_mat_frag,
        ଶୂନ୍ୟ* a_mat_frag,
        ଶୂନ୍ୟ* b_mat_frag,
        ଶୂନ୍ୟ* c_mat_frag) {
      auto d = reinterpret_cast<ଅସ୍ଥିର CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      ପୂର୍ଣ୍ଣସଂଖ୍ୟା lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      ଯଦି constexpr (M == 8 && N == 8 && K == 4) {
        ଯଦି constexpr (std::is_floating_point_v<CDType>) {
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
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା r_ind = (lane % 2) ? 1 : 0;
          ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
                static_cast<ଦଶମିକ>(ra[r_ind]) * static_cast<ଦଶମିକ>(rb[c_ind]);
            *d[1] += static_cast<ଦଶମିକ>(ra[r_ind]) *
                     static_cast<ଦଶମିକ>(rb[c_ind + 1]);
            *d[2] += static_cast<ଦଶମିକ>(ra[r_ind + 2]) *
                     static_cast<ଦଶମିକ>(rb[c_ind]);
            *d[3] += static_cast<ଦଶମିକ>(ra[r_ind + 2]) *
                     static_cast<ଦଶମିକ>(rb[c_ind + 1]);

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
                static_cast<ଦଶମିକ>(ra[r_ind]) * static_cast<ଦଶମିକ>(rb[c_ind]);
            *d[5] += static_cast<ଦଶମିକ>(ra[r_ind]) *
                     static_cast<ଦଶମିକ>(rb[c_ind + 1]);
            *d[6] += static_cast<ଦଶମିକ>(ra[r_ind + 2]) *
                     static_cast<ଦଶମିକ>(rb[c_ind]);
            *d[7] += static_cast<ଦଶମିକ>(ra[r_ind + 2]) *
                     static_cast<ଦଶମିକ>(rb[c_ind + 1]);
          }
        }
      } ଅନ୍ୟଥା ଯଦି constexpr (M == 8 && N == 8 && K == 16) {
        ଯଦି constexpr (std::is_integral_v<ABType>) {
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
          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } ଅନ୍ୟଥା ଯଦି constexpr (M == 16 && N == 8 && K == 8) {
        ଯଦି constexpr (std::is_floating_point_v<CDType>) {
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
          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 2; j++) {
              *d[0] += static_cast<ଦଶମିକ>(ra[j]) * static_cast<ଦଶମିକ>(rb[j]);
              *d[1] +=
                  static_cast<ଦଶମିକ>(ra[j]) * static_cast<ଦଶମିକ>(rb[j + 2]);
              *d[2] +=
                  static_cast<ଦଶମିକ>(ra[j + 2]) * static_cast<ଦଶମିକ>(rb[j]);
              *d[3] +=
                  static_cast<ଦଶମିକ>(ra[j + 2]) * static_cast<ଦଶମିକ>(rb[j + 2]);
            }
          }
        }
      } ଅନ୍ୟଥା ଯଦି constexpr (M == 16 && N == 8 && K == 16) {
        ଯଦି constexpr (std::is_floating_point_v<CDType>) {
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
          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } ଅନ୍ୟଥା ଯଦି constexpr (std::is_integral_v<ABType>) {
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
          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } ଅନ୍ୟଥା ଯଦି constexpr (M == 16 && N == 8 && K == 32) {
        ଯଦି constexpr (std::is_integral_v<ABType>) {
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
          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < 4; i++) {
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
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < 4; j++) {
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

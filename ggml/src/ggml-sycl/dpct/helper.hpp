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

#அந்தர்பாவ <cstdlib>
#அந்தர்பாவ <iostream>
#அந்தர்பாவ <map>

#அந்தர்பாவ <sycl/sycl.hpp>
#அந்தர்பாவ <sycl/half_type.hpp>
#அந்தர்பாவ <oneapi/mkl.hpp>

#எனில் defined(__linux__)
#அந்தர்பாவ <sys/mman.h>
#அல்லதெனில் defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#அந்தர்பாவ <windows.h>
#இல்லையெனில்
#error "Only support Windows and Linux."
#endif

#எனில் defined(__linux__)
#அந்தர்பாவ <unistd.h>
#அந்தர்பாவ <sys/syscall.h>
#endif
#எனில் defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#அந்தர்பாவ <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#எனில் defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#இல்லையெனில்
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#எனில் defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#இல்லையெனில்
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (வெறுமை)(x)

[[noreturn]] உள்வரி வெறுமை _abort(மாறிலி எழுத்து * str) {
    std::cerr << str << std::endl;
    std::abort();
}

உள்வரி std::string get_device_type_name(மாறிலி sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    தேர்வு (DeviceType) {
    நிலை sycl::info::device_type::cpu:
        திரும்பு "cpu";
    நிலை sycl::info::device_type::gpu:
        திரும்பு "gpu";
    நிலை sycl::info::device_type::host:
        திரும்பு "host";
    நிலை sycl::info::device_type::accelerator:
        திரும்பு "acc";
    இயல்பு:
        திரும்பு "unknown";
    }
}

உள்வரி std::string get_device_backend_and_type(மாறிலி sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    திரும்பு device_type.str();
}

template <typename Ts> கட்டமைப்பு matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     மதிப்பு_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

நாமாகாஶ dpct
{
    வகைவரை sycl::queue *queue_ptr;
    வகைவரை sycl::event *event_ptr;
    வகைவரை எழுத்து *device_ptr;
    வகைவரை uint8_t byte_t;
    வகைவரை sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    உள்வரி auto exception_handler = [](sycl::exception_list exceptions)
    {
        க்கு (std::exception_ptr மாறிலி &e : exceptions)
        {
            முயல்
            {
                std::rethrow_exception(e);
            }
            பிடி (sycl::exception மாறிலி &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    எண்ணிக்கை error_code
    {
        success = 0,
        default_error = 999
    };

    எண்ணிக்கை memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    எண்ணிக்கை memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    எண்ணிக்கை வகுப்பு library_data_t : unsigned எழுத்து
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
    கட்டமைப்பு DataType
    {
        using T2 = T;
    };
    template <typename T>
    கட்டமைப்பு DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    நிலையான வெறுமை destroy_event(event_ptr event)
    {
        delete event;
    }

    நிலையான உள்வரி unsigned முழு get_tid()
    {
#எனில் defined(__linux__)
        திரும்பு syscall(SYS_gettid);
#அல்லதெனில் defined(_WIN64)
        திரும்பு GetCurrentThreadId();
#இல்லையெனில்
#error "Only support Windows and Linux."
#endif
    }

    நாமாகாஶ detail
    {
        நிலையான வெறுமை get_version(மாறிலி sycl::device &dev, முழு &major, முழு &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            போது (i < ver.size()) {
              எனில் (isdigit(ver[i]))
                நிறுத்து;
              i++;
            }
            major = std::stoi(&(ver[i]));
            போது (i < ver.size()) {
              எனில் (ver[i] == '.')
                நிறுத்து;
              i++;
            }
            எனில் (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } இல்லையெனில் {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        வகுப்பு generic_error_type
        {
        பொது:
            generic_error_type() = இயல்பு;
            generic_error_type(T value) : value{value} {}
            operator T() மாறிலி { திரும்பு value; }

        தனிப்பட்ட:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    வகுப்பு dim3 {
        பொது:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(மாறிலி sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() மாறிலி { திரும்பு sycl::range<3>(z, y, x); }
    }; // namespace dim3

    உள்வரி dim3 operator*(மாறிலி dim3 &a, மாறிலி dim3 &b) {
    திரும்பு dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    வகுப்பு pitched_data
    {
    பொது:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(வெறுமை *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        வெறுமை *get_data_ptr() { திரும்பு _data; }
        வெறுமை set_data_ptr(வெறுமை *data) { _data = data; }

        size_t get_pitch() { திரும்பு _pitch; }
        வெறுமை set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { திரும்பு _x; }
        வெறுமை set_x(size_t x) { _x = x; }

        size_t get_y() { திரும்பு _y; }
        வெறுமை set_y(size_t y) { _y = y; }

    தனிப்பட்ட:
        வெறுமை *_data;
        size_t _pitch, _x, _y;
    };

    வகுப்பு device_info
    {
    பொது:
        // get interface
        மாறிலி எழுத்து *get_name() மாறிலி { திரும்பு _name; }
        எழுத்து *get_name() { திரும்பு _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, முழு *>,
                                   முழு> = 0>
        auto get_max_work_item_sizes() மாறிலி
        {
            எனில் constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                திரும்பு sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            இல்லையெனில்
            {
                திரும்பு _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, முழு *>,
                                   முழு> = 0>
        auto get_max_work_item_sizes()
        {
            எனில் constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                திரும்பு sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            இல்லையெனில்
            {
                திரும்பு _max_work_item_sizes_i;
            }
        }
        தருக்க get_host_unified_memory() மாறிலி { திரும்பு _host_unified_memory; }
        முழு get_major_version() மாறிலி { திரும்பு _major; }
        முழு get_minor_version() மாறிலி { திரும்பு _minor; }
        முழு get_integrated() மாறிலி { திரும்பு _integrated; }
        முழு get_max_clock_frequency() மாறிலி { திரும்பு _frequency; }
        முழு get_max_compute_units() மாறிலி { திரும்பு _max_compute_units; }
        முழு get_max_work_group_size() மாறிலி { திரும்பு _max_work_group_size; }
        முழு get_max_sub_group_size() மாறிலி { திரும்பு _max_sub_group_size; }
        முழு get_max_work_பொருட்கள்_per_compute_unit() மாறிலி
        {
            திரும்பு max_work_பொருட்கள்_per_compute_unit;
        }
        முழு get_max_register_size_per_work_group() மாறிலி
        {
            திரும்பு _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, முழு *>,
                                   முழு> = 0>
        auto get_max_nd_range_size() மாறிலி
        {
            எனில் constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                திரும்பு _max_nd_range_size;
            இல்லையெனில்
                திரும்பு _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, முழு *>,
                                   முழு> = 0>
        auto get_max_nd_range_size()
        {
            எனில் constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                திரும்பு _max_nd_range_size;
            இல்லையெனில்
                திரும்பு _max_nd_range_size_i;
        }
        size_t get_global_mem_size() மாறிலி { திரும்பு _global_mem_size; }
        size_t get_local_mem_size() மாறிலி { திரும்பு _local_mem_size; }
        size_t get_max_mem_alloc_size() மாறிலி { திரும்பு _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned முழு get_memory_clock_rate() மாறிலி { திரும்பு _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned முழு get_memory_bus_width() மாறிலி { திரும்பு _memory_bus_width; }
        uint32_t get_device_id() மாறிலி { திரும்பு _device_id; }
        std::array<unsigned எழுத்து, 16> get_uuid() மாறிலி { திரும்பு _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned முழு get_global_mem_cache_size() மாறிலி
        {
            திரும்பு _global_mem_cache_size;
        }

        // set interface
        வெறுமை set_name(மாறிலி எழுத்து *name)
        {
            size_t length = strlen(name);
            எனில் (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            இல்லையெனில்
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        வெறுமை set_max_work_item_sizes(மாறிலி sycl::range<3> max_work_item_sizes)
        {
            க்கு (முழு i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] வெறுமை
        set_max_work_item_sizes(மாறிலி sycl::id<3> max_work_item_sizes)
        {
            க்கு (முழு i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        வெறுமை set_host_unified_memory(தருக்க host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        வெறுமை set_major_version(முழு major) { _major = major; }
        வெறுமை set_minor_version(முழு minor) { _minor = minor; }
        வெறுமை set_integrated(முழு integrated) { _integrated = integrated; }
        வெறுமை set_max_clock_frequency(முழு frequency) { _frequency = frequency; }
        வெறுமை set_max_compute_units(முழு max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        வெறுமை set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        வெறுமை set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        வெறுமை set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        வெறுமை set_max_work_group_size(முழு max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        வெறுமை set_max_sub_group_size(முழு max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        வெறுமை
        set_max_work_பொருட்கள்_per_compute_unit(முழு max_work_பொருட்கள்_per_compute_unit_2)
        {
            max_work_பொருட்கள்_per_compute_unit = max_work_பொருட்கள்_per_compute_unit_2;
        }
        வெறுமை set_max_nd_range_size(முழு max_nd_range_size[])
        {
            க்கு (முழு i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        வெறுமை set_memory_clock_rate(unsigned முழு memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        வெறுமை set_memory_bus_width(unsigned முழு memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        வெறுமை
        set_max_register_size_per_work_group(முழு max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        வெறுமை set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        வெறுமை set_uuid(std::array<unsigned எழுத்து, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        வெறுமை set_global_mem_cache_size(unsigned முழு global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    தனிப்பட்ட:
        எழுத்து _name[256];
        முழு _max_work_item_sizes_i[3];
        தருக்க _host_unified_memory = பொய்;
        முழு _major;
        முழு _minor;
        முழு _integrated = 0;
        முழு _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned முழு _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned முழு _memory_bus_width = 64;
        unsigned முழு _global_mem_cache_size;
        முழு _max_compute_units;
        முழு _max_work_group_size;
        முழு _max_sub_group_size;
        முழு max_work_பொருட்கள்_per_compute_unit;
        முழு _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        முழு _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned எழுத்து, 16> _uuid;
    };

    நிலையான முழு get_major_version(மாறிலி sycl::device &dev)
    {
        முழு major, minor;
        detail::get_version(dev, major, minor);
        திரும்பு major;
    }

    நிலையான முழு get_minor_version(மாறிலி sycl::device &dev)
    {
        முழு major, minor;
        detail::get_version(dev, major, minor);
        திரும்பு minor;
    }

    நிலையான வெறுமை get_device_info(device_info &out, மாறிலி sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        முழு major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#எனில் (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#இல்லையெனில்
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

#எனில் (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        எனில் (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned முழு tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            எனில் (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        எனில் (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        எனில் (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        எனில் (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#அல்லதெனில் defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate இயல்பு value. \
        Use 64 bits as memory_bus_width இயல்பு value.")
#இல்லையெனில்
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate இயல்பு value. \
        Use 64 bits as memory_bus_width இயல்பு value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        க்கு (மாறிலி auto &sub_group_size : sub_group_sizes)
        {
            எனில் (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_பொருட்கள்_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        முழு max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    வகுப்பு device_ext : பொது sycl::device {
      வகைவரை std::mutex mutex_type;

     பொது:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(மாறிலி sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      முழு is_native_atomic_supported() { திரும்பு 0; }
      முழு get_major_version() மாறிலி { திரும்பு dpct::get_major_version(*இது); }

      முழு get_minor_version() மாறிலி { திரும்பு dpct::get_minor_version(*இது); }

      முழு get_max_compute_units() மாறிலி {
        திரும்பு get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      முழு get_max_clock_frequency() மாறிலி {
        திரும்பு get_device_info().get_max_clock_frequency();
      }

      முழு get_integrated() மாறிலி { திரும்பு get_device_info().get_integrated(); }

      முழு get_max_sub_group_size() மாறிலி {
        திரும்பு get_device_info().get_max_sub_group_size();
      }

      முழு get_max_register_size_per_work_group() மாறிலி {
        திரும்பு get_device_info().get_max_register_size_per_work_group();
      }

      முழு get_max_work_group_size() மாறிலி {
        திரும்பு get_device_info().get_max_work_group_size();
      }

      முழு get_mem_base_addr_align() மாறிலி {
        திரும்பு get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() மாறிலி {
        திரும்பு get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() மாறிலி {
        திரும்பு get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      வெறுமை get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        மாறிலி எழுத்து *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#எனில் (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        எனில் (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } இல்லையெனில் {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#இல்லையெனில்
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#எனில் defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#இல்லையெனில்
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      வெறுமை get_device_info(device_info &out) மாறிலி {
        dpct::get_device_info(out, *இது);
      }

      device_info get_device_info() மாறிலி {
        device_info prop;
        dpct::get_device_info(prop, *இது);
        திரும்பு prop;
      }

      வெறுமை reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { திரும்பு _q_in_order; }

      sycl::queue &out_of_order_queue() { திரும்பு _q_out_of_order; }

      sycl::queue &default_queue() { திரும்பு in_order_queue(); }

      வெறுமை queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        க்கு (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(தருக்க enable_exception_handler = பொய்) {
        திரும்பு create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               தருக்க enable_exception_handler = பொய்) {
        திரும்பு create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(தருக்க enable_exception_handler = பொய்) {
        std::lock_guard<mutex_type> lock(m_mutex);
        திரும்பு create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        தருக்க enable_exception_handler = பொய்) {
        std::lock_guard<mutex_type> lock(m_mutex);
        திரும்பு create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          தருக்க enable_exception_handler = பொய்) {
        std::lock_guard<mutex_type> lock(m_mutex);
        திரும்பு create_queue_impl(enable_exception_handler);
      }

      வெறுமை destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](மாறிலி sycl::queue &q) -> தருக்க
                                    {
                                        திரும்பு q == queue;
                                    }),
                    _queues.end());
      }
      வெறுமை set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() மாறிலி {
        std::lock_guard<mutex_type> lock(m_mutex);
        திரும்பு _saved_queue;
      }

     தனிப்பட்ட:
      வெறுமை clear_queues() { _queues.clear(); }

      வெறுமை init_queues() {
        _q_in_order =
            create_queue_impl(உண்மை, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(உண்மை);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <வகுப்பு... Properties>
      sycl::queue create_queue_impl(தருக்க enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        எனில் (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *இது, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        திரும்பு _queues.back();
      }

      template <வகுப்பு... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    தருக்க enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        எனில் (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        திரும்பு _queues.back();
      }

      வெறுமை get_version(முழு &major, முழு &minor) மாறிலி {
        detail::get_version(*இது, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    வகுப்பு dev_mgr
    {
    பொது:
        device_ext &current_device()
        {
            unsigned முழு dev_id = current_device_id();
            check_id(dev_id);
            திரும்பு *_devs[dev_id];
        }
        device_ext &cpu_device() மாறிலி
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            எனில் (_cpu_device == -1)
            {
                எறி std::runtime_error("no valid cpu device");
            }
            இல்லையெனில்
            {
                திரும்பு *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned முழு id) மாறிலி
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            திரும்பு *_devs[id];
        }
        unsigned முழு current_device_id() மாறிலி
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            எனில் (it != _thread2dev_map.end())
                திரும்பு it->second;
            திரும்பு DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        வெறுமை select_device(unsigned முழு id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned முழு device_மொத்தம்() { திரும்பு _devs.size(); }

        unsigned முழு get_device_id(மாறிலி sycl::device &dev)
        {
            unsigned முழு id = 0;
            க்கு (auto &dev_item : _devs)
            {
                எனில் (*dev_item == dev)
                {
                    திரும்பு id;
                }
                id++;
            }
            திரும்பு -1;
        }

        உள்வரி std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            எழுத்து* env = getenv("ONEAPI_DEVICE_SELECTOR");
            எனில் (env) {
                எனில் (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                இல்லையெனில் எனில் (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                இல்லையெனில் எனில் (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                இல்லையெனில் எனில் (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                இல்லையெனில் {
                    எறி std::runtime_error("invalid device filter: " + std::string(env));
                }
            } இல்லையெனில் {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                எனில் (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                இல்லையெனில் எனில் (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                இல்லையெனில் எனில் (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            க்கு (மாறிலி auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](மாறிலி sycl::device& d) {
                    திரும்பு d.is_gpu();
                });

                எனில் (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    தொடர்;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_கீழ்_case;
                platform_name_கீழ்_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_கீழ்_case.begin(), ::tolower);

                எனில் (platform_name_கீழ்_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    தொடர்;
                }

                result = platform_name;
            }

            எனில் (result.empty())
                எறி std::runtime_error("can not find preferred GPU platform");

            திரும்பு result;
        }

        template <வகுப்பு DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<முழு, DeviceSelector, மாறிலி sycl::device &>>
        select_device(மாறிலி DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned முழு selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        நிலையான dev_mgr &instance()
        {
            நிலையான dev_mgr d_m;
            திரும்பு d_m;
        }
        dev_mgr(மாறிலி dev_mgr &) = delete;
        dev_mgr &operator=(மாறிலி dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    தனிப்பட்ட:
        mutable std::recursive_mutex m_mutex;
        நிலையான தருக்க compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            எனில்(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) திரும்பு உண்மை;
            எனில்(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) திரும்பு பொய்;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            திரும்பு prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        நிலையான முழு convert_backend_index(std::string & backend) {
            எனில் (backend == "ext_oneapi_level_zero:gpu") திரும்பு 0;
            எனில் (backend == "opencl:gpu") திரும்பு 1;
            எனில் (backend == "ext_oneapi_cuda:gpu") திரும்பு 2;
            எனில் (backend == "ext_oneapi_hip:gpu") திரும்பு 3;
            எனில் (backend == "opencl:cpu") திரும்பு 4;
            எனில் (backend == "opencl:acc") திரும்பு 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        நிலையான தருக்க compare_backend(std::string &backend1, std::string &backend2) {
            திரும்பு convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            எனில் (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            போது (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                எனில் (platform_name.compare(preferred_platform_name) != 0) {
                    தொடர்;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                க்கு (மாறிலி auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> சாவிகள்;
            க்கு(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                சாவிகள்.push_back(it->first);
            }
            std::sort(சாவிகள்.begin(), சாவிகள்.end(), compare_backend);

            க்கு (auto &சாவி : சாவிகள்) {
                std::vector<sycl::device> devs = backend_devices[சாவி];
                std::sort(devs.begin(), devs.end(), compare_dev);
                க்கு (மாறிலி auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            க்கு (auto &dev : sycl_all_devs)
            {
                எனில் (dev == default_device)
                {
                    தொடர்;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                எனில் (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        வெறுமை check_id(unsigned முழு id) மாறிலி
        {
            எனில் (id >= _devs.size())
            {
                எறி std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        மாறிலி unsigned முழு DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned முழு, unsigned முழு> _thread2dev_map;
        முழு _cpu_device = -1;
    };

    நிலையான உள்வரி sycl::queue &get_default_queue()
    {
        திரும்பு dev_mgr::instance().current_device().default_queue();
    }

    நாமாகாஶ detail
    {
        எண்ணிக்கை வகுப்பு pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        நிலையான pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              மாறிலி வெறுமை *ptr)
        {
            தேர்வு (sycl::get_pointer_type(ptr, q.get_context()))
            {
            நிலை sycl::usm::alloc::unknown:
                திரும்பு pointer_access_attribute::host_only;
            நிலை sycl::usm::alloc::device:
                திரும்பு pointer_access_attribute::device_only;
            நிலை sycl::usm::alloc::shared:
            நிலை sycl::usm::alloc::host:
                திரும்பு pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        உள்வரி constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned எழுத்து)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned எழுத்து>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            திரும்பு (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        உள்வரி constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned எழுத்து>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(அளவு...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            திரும்பு get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        வகுப்பு mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#எனில் defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#அல்லதெனில் defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    வெற்று,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#இல்லையெனில்
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        பொது:
            using buffer_id_t = முழு;

            கட்டமைப்பு allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#எனில் defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#அல்லதெனில் defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#இல்லையெனில்
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(மாறிலி mem_mgr &) = delete;
            mem_mgr &operator=(மாறிலி mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            வெறுமை *mem_alloc(size_t size)
            {
                எனில் (!size)
                    திரும்பு nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                எனில் (next_free + size > mapped_address_space + mapped_region_size)
                {
                    எறி std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                வெறுமை *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                திரும்பு result;
            }

            /// Deallocate
            வெறுமை mem_free(மாறிலி வெறுமை *ptr)
            {
                எனில் (!ptr)
                    திரும்பு;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(மாறிலி வெறுமை *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                திரும்பு it->second;
            }

            /// Check if the pointer represents device pointer or not.
            தருக்க is_device_ptr(மாறிலி வெறுமை *ptr) மாறிலி
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                திரும்பு (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            நிலையான mem_mgr &instance()
            {
                நிலையான mem_mgr m;
                திரும்பு m;
            }

        தனிப்பட்ட:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            மாறிலி size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            மாறிலி size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            மாறிலி size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(மாறிலி வெறுமை *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<மாறிலி byte_t *>(ptr)));
                எனில் (it == m_map.end())
                {
                    // Not a virtual pointer.
                    எறி std::runtime_error("can not get buffer from non-virtual pointer");
                }
                மாறிலி allocation &alloc = it->second;
                எனில் (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    எறி std::runtime_error("invalid virtual pointer");
                }
                திரும்பு it;
            }
        };

        template <வகுப்பு T, memory_region Memory, size_t Dimension>
        வகுப்பு accessor;
        template <memory_region Memory, வகுப்பு T = byte_t>
        வகுப்பு memory_traits
        {
        பொது:
            நிலையான constexpr sycl::access::target target =
                sycl::access::target::device;
            நிலையான constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::பட
                                     : sycl::access_mode::read_write;
            நிலையான constexpr size_t type_size = அளவு(T);
            using element_t =
                typename std::conditional<Memory == constant, மாறிலி T, T>::type;
            using மதிப்பு_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<மதிப்பு_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        நிலையான உள்வரி வெறுமை *dpct_malloc(size_t size, sycl::queue &q)
        {
            திரும்பு sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        நிலையான உள்வரி வெறுமை *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            திரும்பு dpct_malloc(pitch * y * z, q);
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
        template <typename மதிப்பு_T>
        நிலையான உள்வரி sycl::event dpct_memset(sycl::queue &q, வெறுமை *dev_ptr,
                                              மதிப்பு_T value, size_t size)
        {
            திரும்பு q.fill(dev_ptr, value, size);
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
        template <typename மதிப்பு_T>
        நிலையான உள்வரி std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, மதிப்பு_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned எழுத்து *data_surface = (unsigned எழுத்து *)data.get_data_ptr();
            க்கு (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned எழுத்து *data_ptr = data_surface;
                க்கு (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            திரும்பு event_list;
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
        template <typename மதிப்பு_T>
        நிலையான உள்வரி std::vector<sycl::event>
        dpct_memset(sycl::queue &q, வெறுமை *ptr, size_t pitch, மதிப்பு_T val, size_t x,
                    size_t y)
        {
            திரும்பு dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        நிலையான memcpy_direction deduce_memcpy_direction(sycl::queue &q, வெறுமை *to_ptr,
                                                        மாறிலி வெறுமை *from_ptr,
                                                        memcpy_direction dir)
        {
            தேர்வு (dir)
            {
            நிலை memcpy_direction::host_to_host:
            நிலை memcpy_direction::host_to_device:
            நிலை memcpy_direction::device_to_host:
            நிலை memcpy_direction::device_to_device:
                திரும்பு dir;
            நிலை memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                நிலையான மாறிலி memcpy_direction
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
                திரும்பு direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            இயல்பு:
                எறி std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        நிலையான sycl::event
        dpct_memcpy(sycl::queue &q, வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr, size_t size,
                    memcpy_direction direction,
                    மாறிலி std::vector<sycl::event> &dep_events = {})
        {
            எனில் (!size)
                திரும்பு sycl::event{};
            திரும்பு q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        நிலையான உள்வரி size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            திரும்பு slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        நிலையான உள்வரி size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            திரும்பு slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        நிலையான உள்வரி std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    மாறிலி std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            வகுப்பு host_buffer
            {
                வெறுமை *_buf;
                size_t _size;
                sycl::queue &_q;
                மாறிலி std::vector<sycl::event> &_deps; // free operation depends

            பொது:
                host_buffer(size_t size, sycl::queue &q,
                            மாறிலி std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                வெறுமை *get_ptr() மாறிலி { திரும்பு _buf; }
                size_t get_size() மாறிலி { திரும்பு _size; }
                ~host_buffer()
                {
                    எனில் (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::விடுவி(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned எழுத்து *to_surface =
                (unsigned எழுத்து *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            மாறிலி unsigned எழுத்து *from_surface =
                (மாறிலி unsigned எழுத்து *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            எனில் (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                திரும்பு {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            தேர்வு (direction)
            {
            நிலை host_to_host:
                க்கு (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned எழுத்து *to_ptr = to_surface;
                    மாறிலி unsigned எழுத்து *from_ptr = from_surface;
                    எனில் (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    இல்லையெனில்
                    {
                        க்கு (size_t y = 0; y < size.get(1); ++y)
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
                நிறுத்து;
            நிலை host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                எனில் (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                இல்லையெனில்
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
                நிறுத்து;
            }
            நிலை device_to_host:
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
                நிறுத்து;
            }
            நிலை device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<வகுப்பு dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                நிறுத்து;
            இயல்பு:
                எறி std::runtime_error("dpct_memcpy: invalid direction value");
            }
            திரும்பு event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        நிலையான உள்வரி std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            திரும்பு dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        நிலையான உள்வரி std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            திரும்பு dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        நாமாகாஶ deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            வகுப்பு usm_allocator
            {
            தனிப்பட்ட:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            பொது:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    மாறிலி typename std::allocator_traits<Alloc>::value_type &;
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
                கட்டமைப்பு rebind
                {
                    வகைவரை usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(மாறிலி usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { திரும்பு &r; }
                const_pointer address(const_reference r) { திரும்பு &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    திரும்பு std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                வெறுமை deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() மாறிலி
                {
                    திரும்பு std::allocator_traits<Alloc>::max_size(_impl);
                }
                தருக்க operator==(மாறிலி usm_allocator &other) மாறிலி { திரும்பு _impl == other._impl; }
                தருக்க operator!=(மாறிலி usm_allocator &other) மாறிலி { திரும்பு _impl != other._impl; }
            };

        } // namespace deprecated

        உள்வரி வெறுமை dpct_free(வெறுமை *ptr,
                              மாறிலி sycl::queue &q)
        {
            எனில் (ptr)
            {
                sycl::விடுவி(ptr, q.get_context());
            }
        }

        template <typename T>
        உள்வரி auto get_memory(மாறிலி வெறுமை *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<வெறுமை *>(x));
            திரும்பு new_x;
        }

        template <typename T>
        உள்வரி typename DataType<T>::T2 get_மதிப்பு(மாறிலி T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            எனில் (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (வெறுமை *)&s_h, (மாறிலி வெறுமை *)s, அளவு(T), device_to_host)
                    .wait();
            இல்லையெனில்
                s_h = *reinterpret_cast<மாறிலி Ty *>(s);
            திரும்பு s_h;
        }

    } // namespace detail

    template <typename T>
    உள்வரி auto get_மதிப்பு(மாறிலி T *s, sycl::queue &q)
    {
        திரும்பு detail::get_மதிப்பு(s, q);
    }

    நாமாகாஶ detail
    {
    template <வகுப்பு Ta, வகுப்பு Tb, வகுப்பு Tc, வகுப்பு Ts>
    உள்வரி வெறுமை gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, முழு m,
                          முழு n, முழு k, மாறிலி வெறுமை * alpha, மாறிலி வெறுமை * a, முழு lda, மாறிலி வெறுமை * b, முழு ldb,
                          மாறிலி வெறுமை * beta, வெறுமை * c, முழு ldc) {
        Ts   alpha_மதிப்பு_2 = dpct::get_மதிப்பு(reinterpret_cast<மாறிலி Ts *>(alpha), q);
        Ts   beta_மதிப்பு  = dpct::get_மதிப்பு(reinterpret_cast<மாறிலி Ts *>(beta), q);
        auto data_a      = get_memory<மாறிலி Ta>(a);
        auto data_b      = get_memory<மாறிலி Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_மதிப்பு_2, data_a,
                                               lda, data_b, ldb, beta_மதிப்பு, data_c, ldc);
    }

        template <typename VecT, வகுப்பு இருமம்_Operation, வகுப்பு = வெறுமை>
        வகுப்பு vectorized_இருமம்
        {
        பொது:
            உள்வரி VecT operator()(VecT a, VecT b, மாறிலி இருமம்_Operation இருமம்_op)
            {
                VecT v4;
                க்கு (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = இருமம்_op(a[i], b[i]);
                }
                திரும்பு v4;
            }
        };

        template <typename VecT, வகுப்பு இருமம்_Operation>
        வகுப்பு vectorized_இருமம்<
            VecT, இருமம்_Operation,
            std::void_t<std::invoke_result_t<இருமம்_Operation, VecT, VecT>>>
        {
        பொது:
            உள்வரி VecT operator()(VecT a, VecT b, மாறிலி இருமம்_Operation இருமம்_op)
            {
                திரும்பு இருமம்_op(a, b).template as<VecT>();
            }
        };

        template <வகுப்பு Ta, வகுப்பு Tb, வகுப்பு Tc, வகுப்பு Ts>
        உள்வரி வெறுமை gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    முழு m, முழு n, முழு k, மாறிலி வெறுமை * alpha, மாறிலி வெறுமை ** a, முழு lda, மாறிலி வெறுமை ** b,
                                    முழு ldb, மாறிலி வெறுமை * beta, வெறுமை ** c, முழு ldc, முழு batch_size,
                                    matrix_info_t<தசம> * matrix_info) {
            Ts alpha_மதிப்பு_2 = dpct::get_மதிப்பு(reinterpret_cast<மாறிலி Ts *>(alpha), q);
            Ts beta_மதிப்பு = dpct::get_மதிப்பு(reinterpret_cast<மாறிலி Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->மதிப்பு_info[0] = alpha_மதிப்பு_2;
            matrix_info->மதிப்பு_info[1] = beta_மதிப்பு;
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
                reinterpret_cast<Ts *>(matrix_info->மதிப்பு_info), reinterpret_cast<மாறிலி Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<மாறிலி Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->மதிப்பு_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <வகுப்பு Ta, வகுப்பு Tb, வகுப்பு Tc, வகுப்பு Ts>
        உள்வரி வெறுமை gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    முழு m, முழு n, முழு k, மாறிலி வெறுமை * alpha, மாறிலி வெறுமை * a, முழு lda,
                                    long long முழு stride_a, மாறிலி வெறுமை * b, முழு ldb, long long முழு stride_b,
                                    மாறிலி வெறுமை * beta, வெறுமை * c, முழு ldc, long long முழு stride_c, முழு batch_size) {
            Ts alpha_மதிப்பு_2 = dpct::get_மதிப்பு(reinterpret_cast<மாறிலி Ts *>(alpha), q);
            Ts beta_மதிப்பு = dpct::get_மதிப்பு(reinterpret_cast<மாறிலி Ts *>(beta), q);
            auto data_a = get_memory<மாறிலி Ta>(a);
            auto data_b = get_memory<மாறிலி Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_மதிப்பு_2,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_மதிப்பு,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, வகுப்பு இருமம்_Operation>
    உள்வரி unsigned vectorized_இருமம்(unsigned a, unsigned b,
                                      மாறிலி இருமம்_Operation இருமம்_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_இருமம்<VecT, இருமம்_Operation>()(v2, v3, இருமம்_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        திரும்பு v0;
    }

    நிலையான வெறுமை async_dpct_memcpy(வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    நிலையான உள்வரி unsigned முழு select_device(unsigned முழு id)
    {
        dev_mgr::instance().select_device(id);
        திரும்பு id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned முழு mask,
                               unsigned முழு logical_sub_group_size = 32)
    {
        unsigned முழு id = g.get_local_linear_id();
        unsigned முழு start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned முழு target_offset = (id % logical_sub_group_size) ^ mask;
        திரும்பு sycl::select_from_group(g, x,
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
      திரும்பு sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    உள்வரி auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      திரும்பு res;
    }

    கட்டமைப்பு sub_sat
    {
        template <typename T>
        auto operator()(மாறிலி T x, மாறிலி T y) மாறிலி
        {
            திரும்பு sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    உள்வரி T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        திரும்பு v0;
    }

    உள்வரி தசம pow(மாறிலி தசம a, மாறிலி முழு b) { திரும்பு sycl::pown(a, b); }
    உள்வரி இரட்டை pow(மாறிலி இரட்டை a, மாறிலி முழு b) { திரும்பு sycl::pown(a, b); }
    உள்வரி தசம pow(மாறிலி தசம a, மாறிலி தசம b) { திரும்பு sycl::pow(a, b); }
    உள்வரி இரட்டை pow(மாறிலி இரட்டை a, மாறிலி இரட்டை b) { திரும்பு sycl::pow(a, b); }
    template <typename T, typename U>
    உள்வரி typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(மாறிலி T a, மாறிலி U b)
    {
        திரும்பு sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    உள்வரி typename std::enable_if_t<!std::is_floating_point_v<T>, இரட்டை>
    pow(மாறிலி T a, மாறிலி U b)
    {
        திரும்பு sycl::pow(static_cast<இரட்டை>(a), static_cast<இரட்டை>(b));
    }

    உள்வரி இரட்டை min(மாறிலி இரட்டை a, மாறிலி தசம b)
    {
        திரும்பு sycl::fmin(a, static_cast<இரட்டை>(b));
    }
    உள்வரி இரட்டை min(மாறிலி தசம a, மாறிலி இரட்டை b)
    {
        திரும்பு sycl::fmin(static_cast<இரட்டை>(a), b);
    }
    உள்வரி தசம min(மாறிலி தசம a, மாறிலி தசம b) { திரும்பு sycl::fmin(a, b); }
    உள்வரி இரட்டை min(மாறிலி இரட்டை a, மாறிலி இரட்டை b) { திரும்பு sycl::fmin(a, b); }
    உள்வரி std::uint32_t min(மாறிலி std::uint32_t a, மாறிலி std::int32_t b)
    {
        திரும்பு sycl::min(a, static_cast<std::uint32_t>(b));
    }
    உள்வரி std::uint32_t min(மாறிலி std::int32_t a, மாறிலி std::uint32_t b)
    {
        திரும்பு sycl::min(static_cast<std::uint32_t>(a), b);
    }
    உள்வரி std::int32_t min(மாறிலி std::int32_t a, மாறிலி std::int32_t b)
    {
        திரும்பு sycl::min(a, b);
    }
    உள்வரி std::uint32_t min(மாறிலி std::uint32_t a, மாறிலி std::uint32_t b)
    {
        திரும்பு sycl::min(a, b);
    }
    உள்வரி std::uint64_t min(மாறிலி std::uint64_t a, மாறிலி std::int64_t b)
    {
        திரும்பு sycl::min(a, static_cast<std::uint64_t>(b));
    }
    உள்வரி std::uint64_t min(மாறிலி std::int64_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::min(static_cast<std::uint64_t>(a), b);
    }
    உள்வரி std::int64_t min(மாறிலி std::int64_t a, மாறிலி std::int64_t b)
    {
        திரும்பு sycl::min(a, b);
    }
    உள்வரி std::uint64_t min(மாறிலி std::uint64_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::min(a, b);
    }
    உள்வரி std::uint64_t min(மாறிலி std::uint64_t a, மாறிலி std::int32_t b)
    {
        திரும்பு sycl::min(a, static_cast<std::uint64_t>(b));
    }
    உள்வரி std::uint64_t min(மாறிலி std::int32_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::min(static_cast<std::uint64_t>(a), b);
    }
    உள்வரி std::uint64_t min(மாறிலி std::uint64_t a, மாறிலி std::uint32_t b)
    {
        திரும்பு sycl::min(a, static_cast<std::uint64_t>(b));
    }
    உள்வரி std::uint64_t min(மாறிலி std::uint32_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    உள்வரி இரட்டை max(மாறிலி இரட்டை a, மாறிலி தசம b)
    {
        திரும்பு sycl::fmax(a, static_cast<இரட்டை>(b));
    }
    உள்வரி இரட்டை max(மாறிலி தசம a, மாறிலி இரட்டை b)
    {
        திரும்பு sycl::fmax(static_cast<இரட்டை>(a), b);
    }
    உள்வரி தசம max(மாறிலி தசம a, மாறிலி தசம b) { திரும்பு sycl::fmax(a, b); }
    உள்வரி இரட்டை max(மாறிலி இரட்டை a, மாறிலி இரட்டை b) { திரும்பு sycl::fmax(a, b); }
    உள்வரி std::uint32_t max(மாறிலி std::uint32_t a, மாறிலி std::int32_t b)
    {
        திரும்பு sycl::max(a, static_cast<std::uint32_t>(b));
    }
    உள்வரி std::uint32_t max(மாறிலி std::int32_t a, மாறிலி std::uint32_t b)
    {
        திரும்பு sycl::max(static_cast<std::uint32_t>(a), b);
    }
    உள்வரி std::int32_t max(மாறிலி std::int32_t a, மாறிலி std::int32_t b)
    {
        திரும்பு sycl::max(a, b);
    }
    உள்வரி std::uint32_t max(மாறிலி std::uint32_t a, மாறிலி std::uint32_t b)
    {
        திரும்பு sycl::max(a, b);
    }
    உள்வரி std::uint64_t max(மாறிலி std::uint64_t a, மாறிலி std::int64_t b)
    {
        திரும்பு sycl::max(a, static_cast<std::uint64_t>(b));
    }
    உள்வரி std::uint64_t max(மாறிலி std::int64_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::max(static_cast<std::uint64_t>(a), b);
    }
    உள்வரி std::int64_t max(மாறிலி std::int64_t a, மாறிலி std::int64_t b)
    {
        திரும்பு sycl::max(a, b);
    }
    உள்வரி std::uint64_t max(மாறிலி std::uint64_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::max(a, b);
    }
    உள்வரி std::uint64_t max(மாறிலி std::uint64_t a, மாறிலி std::int32_t b)
    {
        திரும்பு sycl::max(a, static_cast<std::uint64_t>(b));
    }
    உள்வரி std::uint64_t max(மாறிலி std::int32_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::max(static_cast<std::uint64_t>(a), b);
    }
    உள்வரி std::uint64_t max(மாறிலி std::uint64_t a, மாறிலி std::uint32_t b)
    {
        திரும்பு sycl::max(a, static_cast<std::uint64_t>(b));
    }
    உள்வரி std::uint64_t max(மாறிலி std::uint32_t a, மாறிலி std::uint64_t b)
    {
        திரும்பு sycl::max(static_cast<std::uint64_t>(a), b);
    }

    உள்வரி வெறுமை
    has_capability_or_fail(மாறிலி sycl::device &dev,
                           மாறிலி std::initializer_list<sycl::aspect> &props)
    {
        க்கு (மாறிலி auto &it : props)
        {
            எனில் (dev.has(it))
                தொடர்;
            தேர்வு (it)
            {
            நிலை sycl::aspect::fp64:
                எறி std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                நிறுத்து;
            நிலை sycl::aspect::fp16:
                எறி std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                நிறுத்து;
            இயல்பு:
#define __SYCL_ASPECT(ASPECT, ID) \
    நிலை sycl::aspect::ASPECT:    \
        திரும்பு #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect Aspect_இலக்கம்) -> std::string
                {
                    தேர்வு (Aspect_இலக்கம்)
                    {
#அந்தர்பாவ <sycl/info/aspects.def>
#அந்தர்பாவ <sycl/info/aspects_deprecated.def>
                    இயல்பு:
                        திரும்பு "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                எறி std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            நிறுத்து;
        }
    }

    நிலையான உள்வரி unsigned முழு get_current_device_id()
    {
        திரும்பு dev_mgr::instance().current_device_id();
    }

    நிலையான உள்வரி device_ext &get_current_device()
    {
        திரும்பு dev_mgr::instance().current_device();
    }

    நிலையான உள்வரி device_ext &get_device(unsigned முழு id)
    {
        திரும்பு dev_mgr::instance().get_device(id);
    }

    நிலையான உள்வரி sycl::queue &get_in_order_queue()
    {
        திரும்பு dev_mgr::instance().current_device().in_order_queue();
    }

    நிலையான sycl::event
    dpct_memcpy(sycl::queue &q, வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr, size_t size,
                memcpy_direction direction,
                மாறிலி std::vector<sycl::event> &dep_events = {})
    {
        எனில் (!size)
            திரும்பு sycl::event{};
        திரும்பு q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    நிலையான உள்வரி size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        திரும்பு slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    நிலையான உள்வரி size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        திரும்பு slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    நிலையான உள்வரி std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                மாறிலி std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        வகுப்பு host_buffer
        {
            வெறுமை *_buf;
            size_t _size;
            sycl::queue &_q;
            மாறிலி std::vector<sycl::event> &_deps; // free operation depends

        பொது:
            host_buffer(size_t size, sycl::queue &q,
                        மாறிலி std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            வெறுமை *get_ptr() மாறிலி { திரும்பு _buf; }
            size_t get_size() மாறிலி { திரும்பு _size; }
            ~host_buffer()
            {
                எனில் (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::விடுவி(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned எழுத்து *to_surface =
            (unsigned எழுத்து *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        மாறிலி unsigned எழுத்து *from_surface =
            (மாறிலி unsigned எழுத்து *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        எனில் (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            திரும்பு {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        தேர்வு (direction)
        {
        நிலை host_to_host:
            க்கு (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned எழுத்து *to_ptr = to_surface;
                மாறிலி unsigned எழுத்து *from_ptr = from_surface;
                எனில் (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                இல்லையெனில்
                {
                    க்கு (size_t y = 0; y < size.get(1); ++y)
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
            நிறுத்து;
        நிலை host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            எனில் (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            இல்லையெனில்
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
            நிறுத்து;
        }
        நிலை device_to_host:
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
            நிறுத்து;
        }
        நிலை device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<வகுப்பு dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        நிறுத்து;
        இயல்பு:
            எறி std::runtime_error("dpct_memcpy: invalid direction value");
        }
        திரும்பு event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    நிலையான உள்வரி std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        திரும்பு dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    நிலையான உள்வரி std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, வெறுமை *to_ptr, மாறிலி வெறுமை *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        திரும்பு dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    உள்வரி வெறுமை gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, முழு m, முழு n,
                     முழு k, மாறிலி வெறுமை * alpha, மாறிலி வெறுமை * a, library_data_t a_type, முழு lda, மாறிலி வெறுமை * b,
                     library_data_t b_type, முழு ldb, மாறிலி வெறுமை * beta, வெறுமை * c, library_data_t c_type, முழு ldc,
                     library_data_t scaling_type) {
        எனில் (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        இல்லையெனில் எனில் (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t சாவி =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        தேர்வு (சாவி)
        {
        நிலை detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<தசம, தசம, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<இரட்டை, இரட்டை, இரட்டை, இரட்டை>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<தசம>, std::complex<தசம>,
                              std::complex<தசம>, std::complex<தசம>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<இரட்டை>, std::complex<இரட்டை>,
                              std::complex<இரட்டை>, std::complex<இரட்டை>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
#ifdef __INTEL_MKL__
        நிலை detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            தசம alpha_மதிப்பு_2 =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி தசம *>(alpha), q);
            தசம beta_மதிப்பு =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி தசம *>(beta), q);
            sycl::half alpha_half(alpha_மதிப்பு_2);
            sycl::half beta_half(beta_மதிப்பு);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            தசம alpha_float =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி std::int32_t *>(alpha), q);
            தசம beta_float =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, தசம>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            நிறுத்து;
        }
#endif // __INTEL_MKL__
        இயல்பு:
            எறி std::runtime_error("the combination of data type is unsupported");
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
    உள்வரி வெறுமை gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, முழு m,
                           முழு n, முழு k, மாறிலி வெறுமை * alpha, மாறிலி வெறுமை * a[], library_data_t a_type, முழு lda,
                           மாறிலி வெறுமை * b[], library_data_t b_type, முழு ldb, மாறிலி வெறுமை * beta, வெறுமை * c[],
                           library_data_t c_type, முழு ldc, முழு batch_size, library_data_t scaling_type,
                           matrix_info_t<தசம> * matrix_info) {
        std::uint64_t சாவி =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        தேர்வு (சாவி)
        {
        நிலை detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<தசம, தசம, தசம, தசம>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<இரட்டை, இரட்டை, இரட்டை, இரட்டை>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
#ifdef __INTEL_MKL__
        நிலை detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
#endif
        நிலை detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            தசம alpha_float =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி std::int32_t *>(alpha), q);
            தசம beta_float =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, தசம>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            தசம alpha_மதிப்பு_2 =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி தசம *>(alpha), q);
            தசம beta_மதிப்பு =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி தசம *>(beta), q);
            sycl::half alpha_half(alpha_மதிப்பு_2);
            sycl::half beta_half(beta_மதிப்பு);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            நிறுத்து;
        }
        இயல்பு:
            எறி std::runtime_error("the combination of data type is unsupported");
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
    உள்வரி வெறுமை gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, முழு m,
                           முழு n, முழு k, மாறிலி வெறுமை * alpha, மாறிலி வெறுமை * a, library_data_t a_type, முழு lda,
                           long long முழு stride_a, மாறிலி வெறுமை * b, library_data_t b_type, முழு ldb,
                           long long முழு stride_b, மாறிலி வெறுமை * beta, வெறுமை * c, library_data_t c_type, முழு ldc,
                           long long முழு stride_c, முழு batch_size, library_data_t scaling_type) {
        எனில் (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        இல்லையெனில் எனில் (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t சாவி =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        தேர்வு (சாவி)
        {
        நிலை detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<தசம, தசம, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<இரட்டை, இரட்டை, இரட்டை, இரட்டை>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<தசம>, std::complex<தசம>,
                                    std::complex<தசம>, std::complex<தசம>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<இரட்டை>, std::complex<இரட்டை>,
                                    std::complex<இரட்டை>, std::complex<இரட்டை>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
#ifdef __INTEL_MKL__
        நிலை detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            நிறுத்து;
        }
#endif
        நிலை detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, தசம, தசம>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        நிலை detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            தசம alpha_மதிப்பு_2 =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி தசம *>(alpha), q);
            தசம beta_மதிப்பு =
                dpct::get_மதிப்பு(reinterpret_cast<மாறிலி தசம *>(beta), q);
            sycl::half alpha_half(alpha_மதிப்பு_2);
            sycl::half beta_half(beta_மதிப்பு);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            நிறுத்து;
        }
        இயல்பு:
            எறி std::runtime_error("the combination of data type is unsupported");
        }
    }

    நிலையான உள்வரி வெறுமை
    async_dpct_memcpy(வெறுமை *to_ptr, size_t to_pitch, மாறிலி வெறுமை *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<கட்டமைப்பு err0_tag, முழு>;
    using err1 = detail::generic_error_type<கட்டமைப்பு err1_tag, முழு>;

    நிலையான உள்வரி வெறுமை dpct_free(வெறுமை *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <வகுப்பு T, memory_region Memory, size_t Dimension> வகுப்பு accessor;
    template <வகுப்பு T, memory_region Memory> வகுப்பு accessor<T, Memory, 3> {
    பொது:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, மாறிலி sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, மாறிலி accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(மாறிலி accessor_t &acc, மாறிலி sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) மாறிலி {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            திரும்பு accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() மாறிலி { திரும்பு _data; }

    தனிப்பட்ட:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <வகுப்பு T, memory_region Memory> வகுப்பு accessor<T, Memory, 2> {
    பொது:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, மாறிலி sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, மாறிலி accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(மாறிலி accessor_t &acc, மாறிலி sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) மாறிலி {
            திரும்பு _data + _range.get(1) * index;
        }

        pointer_t get_ptr() மாறிலி { திரும்பு _data; }

    தனிப்பட்ட:
        pointer_t _data;
        sycl::range<2> _range;
    };

    நாமாகாஶ detail {
        /// Device variable with address space of shared, global or constant.
        template <வகுப்பு T, memory_region Memory, size_t Dimension> வகுப்பு device_memory {
        பொது:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using மதிப்பு_t = typename detail::memory_traits<Memory, T>::மதிப்பு_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(மாறிலி sycl::range<Dimension> &in_range,
                        std::initializer_list<மதிப்பு_t> &&init_list)
                : device_memory(in_range) {
                ப்ரதிஜ்ஞா(init_list.size() <= in_range.size());
                _host_ptr = (மதிப்பு_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * அளவு(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                மாறிலி typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<மதிப்பு_t>> &&init_list)
                : device_memory(in_range) {
                ப்ரதிஜ்ஞா(init_list.size() <= in_range[0]);
                _host_ptr = (மதிப்பு_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                க்கு (auto sub_list : init_list) {
                    ப்ரதிஜ்ஞா(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * அளவு(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(மாறிலி sycl::range<Dimension> &range_in)
                : _size(range_in.size() * அளவு(T)), _range(range_in),
                _reference(பொய்), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <வகுப்பு... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                எனில் (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                எனில் (_host_ptr)
                    std::விடுவி(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            வெறுமை init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            வெறுமை init(sycl::queue &q) {
                எனில் (_device_ptr)
                    திரும்பு;
                எனில் (!_size)
                    திரும்பு;
                allocate_device(q);
                எனில் (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            வெறுமை assign(மதிப்பு_t *src, size_t size) {
                இது->~device_memory();
                புதிய (இது) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            மதிப்பு_t *get_ptr() { திரும்பு get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            மதிப்பு_t *get_ptr(sycl::queue &q) {
                init(q);
                திரும்பு _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { திரும்பு _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                திரும்பு _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                திரும்பு dpct_accessor_t((T *)_device_ptr, _range);
            }

        தனிப்பட்ட:
            device_memory(மதிப்பு_t *memory_ptr, size_t size)
                : _size(size), _range(size / அளவு(T)), _reference(உண்மை),
                _device_ptr(memory_ptr) {}

            வெறுமை allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                எனில் (Memory == shared) {
                    _device_ptr = (மதிப்பு_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    திரும்பு;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                எனில் (Memory == constant) {
                    _device_ptr = (மதிப்பு_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    திரும்பு;
                }
        #endif
        #endif
                _device_ptr = (மதிப்பு_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            தருக்க _reference;
            மதிப்பு_t *_host_ptr;
            மதிப்பு_t *_device_ptr;
        };
        template <வகுப்பு T, memory_region Memory>
        வகுப்பு device_memory<T, Memory, 0> : பொது device_memory<T, Memory, 1> {
        பொது:
            using base = device_memory<T, Memory, 1>;
            using மதிப்பு_t = typename base::மதிப்பு_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(மாறிலி மதிப்பு_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <வகுப்பு T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <வகுப்பு T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <வகுப்பு T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    உள்வரி T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    திரும்பு atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    உள்வரி T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    திரும்பு atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    உள்வரி T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    தேர்வு (memoryOrder) {
        நிலை sycl::memory_order::relaxed:
            திரும்பு atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        நிலை sycl::memory_order::acq_rel:
            திரும்பு atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        நிலை sycl::memory_order::seq_cst:
            திரும்பு atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        இயல்பு:
            ப்ரதிஜ்ஞா(பொய் && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    உள்வரி T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    உள்வரி unsigned முழு byte_level_permute(
        unsigned முழு a, unsigned முழு b, unsigned முழு s) {
      unsigned முழு ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      திரும்பு ret;
    }

    உள்வரி uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, முழு mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      எனில் (mode >= 1 && mode <= 6) {
        திரும்பு byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } இல்லையெனில் எனில் (!mode) {
        திரும்பு byte_level_permute(low32, high32, sel);
      }
      திரும்பு 0;
    }

    template <முழு n_nondefault_params, முழு n_default_params, typename T>
    வகுப்பு args_selector;

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
    template <முழு n_nondefault_params, முழு n_default_params, typename R,
              typename... Ts>
    வகுப்பு args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      தனிப்பட்ட:
        வெறுமை **kernel_params;
        எழுத்து *args_buffer;

        template <முழு i> நிலையான constexpr முழு account_for_default_params() {
            constexpr முழு n_total_params = அளவு...(Ts);
            எனில் constexpr (i >= n_nondefault_params) {
                திரும்பு n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } இல்லையெனில் {
                திரும்பு i;
            }
        }

      பொது:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <முழு i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        நிலையான constexpr முழு params_இலக்கம் = அளவு...(Ts);

      தனிப்பட்ட:
        template <முழு i> நிலையான constexpr முழு get_offset() {
            எனில் constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                திரும்பு 0;
            } இல்லையெனில் {
                constexpr முழு prev_off = get_offset<i - 1>();
                constexpr முழு prev_past_end =
                    prev_off + அளவு(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                எனில் constexpr (prev_past_end % alignof(T) == 0) {
                    திரும்பு prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                இல்லையெனில் {
                    திரும்பு prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        நிலையான எழுத்து *get_args_buffer(வெறுமை **extra) {
            எனில் (!extra)
                திரும்பு nullptr;
            க்கு (; (std::size_t)*extra != 0; ++extra) {
                எனில் ((std::size_t)*extra == 1) {
                    திரும்பு static_cast<எழுத்து *>(*(extra + 1));
                }
            }
            திரும்பு nullptr;
        }

      பொது:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(வெறுமை **kernel_params, வெறுமை **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <முழு i> arg_type<i> &get() {
            எனில் (kernel_params) {
                திரும்பு *static_cast<arg_type<i> *>(kernel_params[i]);
            } இல்லையெனில் {
                திரும்பு *reinterpret_cast<arg_type<i> *>(args_buffer +
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
    வகுப்பு kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        நிலையான வெறுமை launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        நிலையான வெறுமை set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned முழு local_mem_size,
                                         queue_ptr que) {
            எனில் (que) {
                _que = que;
            } இல்லையெனில் {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        நிலையான உள்வரி std::mutex kernel_function_ptr_map_mutex;

      பொது:
        /// Variables for storing execution configuration.
        நிலையான உள்வரி thread_local sycl::queue *_que = nullptr;
        நிலையான உள்வரி thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        நிலையான உள்வரி thread_local unsigned முழு _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        நிலையான உள்வரி std::map<
            மாறிலி வெறுமை *,
            std::function<வெறுமை(dim3, dim3, வெறுமை **, unsigned முழு, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        நிலையான வெறுமை register_kernel_ptr(
            மாறிலி வெறுமை *func,
            std::function<வெறுமை(dim3, dim3, வெறுமை **, unsigned முழு, queue_ptr)>
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
        நிலையான std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, வெறுமை>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned முழு local_mem_size, queue_ptr que, ArgsT... args) {
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
        நிலையான வெறுமை launch(மாறிலி வெறுமை *func, dim3 group_range, dim3 local_range,
                           வெறுமை **args, unsigned முழு local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            எனில் (Iter == kernel_function_ptr_map.end()) {
                எறி std::runtime_error("dpct::launch() : no registered "
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
        நிலையான std::enable_if_t<std::is_function_v<FuncT>, வெறுமை>
        launch(FuncT *func, dim3 group_range, dim3 local_range, வெறுமை **args,
               unsigned முழு local_mem_size, queue_ptr que) {
            constexpr size_t p_இலக்கம் = args_selector<0, 0, FuncT>::params_இலக்கம்;
            set_execution_config(group_range, local_range, local_mem_size, que);
            args_selector<p_இலக்கம், p_இலக்கம், FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_இலக்கம்>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    template <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        முழு remote_local_id,
        முழு logical_sub_group_size = 32) {
      unsigned முழு start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      திரும்பு sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    வெறுமை ldmatrix(uintptr_t addr, T* m, தருக்க trans = பொய், unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      முழு lane = sg.get_local_linear_id();

      முழு lane_group8_row = lane / 8;
      முழு lane_group8_col = lane % 8;

      எனில் (!trans) {
        // calculate the source lane
        முழு src_lane = 2 * lane_group8_row;
        எனில் (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } இல்லையெனில் {
        // calculate the source lane
        முழு src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        முழு index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    வெறுமை ldmatrix(uintptr_t addr, T* m1, T* m2, தருக்க trans = பொய்) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    வெறுமை ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, தருக்க trans = பொய்) {
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
    கட்டமைப்பு MMAType {
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
    template <முழு M, முழு N, முழு K, typename ABType, typename CDType>
    வெறுமை mma(
        நிலையற்ற வெறுமை** d_mat_frag,
        வெறுமை* a_mat_frag,
        வெறுமை* b_mat_frag,
        வெறுமை* c_mat_frag) {
      auto d = reinterpret_cast<நிலையற்ற CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      முழு lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      எனில் constexpr (M == 8 && N == 8 && K == 4) {
        எனில் constexpr (std::is_floating_point_v<CDType>) {
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
          முழு r_ind = (lane % 2) ? 1 : 0;
          முழு c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          க்கு (முழு i = 0; i < 4; i++) {
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
                static_cast<தசம>(ra[r_ind]) * static_cast<தசம>(rb[c_ind]);
            *d[1] += static_cast<தசம>(ra[r_ind]) *
                     static_cast<தசம>(rb[c_ind + 1]);
            *d[2] += static_cast<தசம>(ra[r_ind + 2]) *
                     static_cast<தசம>(rb[c_ind]);
            *d[3] += static_cast<தசம>(ra[r_ind + 2]) *
                     static_cast<தசம>(rb[c_ind + 1]);

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
                static_cast<தசம>(ra[r_ind]) * static_cast<தசம>(rb[c_ind]);
            *d[5] += static_cast<தசம>(ra[r_ind]) *
                     static_cast<தசம>(rb[c_ind + 1]);
            *d[6] += static_cast<தசம>(ra[r_ind + 2]) *
                     static_cast<தசம>(rb[c_ind]);
            *d[7] += static_cast<தசம>(ra[r_ind + 2]) *
                     static_cast<தசம>(rb[c_ind + 1]);
          }
        }
      } இல்லையெனில் எனில் constexpr (M == 8 && N == 8 && K == 16) {
        எனில் constexpr (std::is_integral_v<ABType>) {
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
          க்கு (முழு i = 0; i < 4; i++) {
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
            க்கு (முழு j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } இல்லையெனில் எனில் constexpr (M == 16 && N == 8 && K == 8) {
        எனில் constexpr (std::is_floating_point_v<CDType>) {
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
          க்கு (முழு i = 0; i < 4; i++) {
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
            க்கு (முழு j = 0; j < 2; j++) {
              *d[0] += static_cast<தசம>(ra[j]) * static_cast<தசம>(rb[j]);
              *d[1] +=
                  static_cast<தசம>(ra[j]) * static_cast<தசம>(rb[j + 2]);
              *d[2] +=
                  static_cast<தசம>(ra[j + 2]) * static_cast<தசம>(rb[j]);
              *d[3] +=
                  static_cast<தசம>(ra[j + 2]) * static_cast<தசம>(rb[j + 2]);
            }
          }
        }
      } இல்லையெனில் எனில் constexpr (M == 16 && N == 8 && K == 16) {
        எனில் constexpr (std::is_floating_point_v<CDType>) {
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
          க்கு (முழு i = 0; i < 4; i++) {
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
            க்கு (முழு j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } இல்லையெனில் எனில் constexpr (std::is_integral_v<ABType>) {
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
          க்கு (முழு i = 0; i < 4; i++) {
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
            க்கு (முழு i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } இல்லையெனில் எனில் constexpr (M == 16 && N == 8 && K == 32) {
        எனில் constexpr (std::is_integral_v<ABType>) {
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
          க்கு (முழு i = 0; i < 4; i++) {
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
            க்கு (முழு j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          க்கு (முழு i = 0; i < 4; i++) {
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
            க்கு (முழு j = 0; j < 4; j++) {
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

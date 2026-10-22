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

#അന്തര്ഭാവ <cstdlib>
#അന്തര്ഭാവ <iostream>
#അന്തര്ഭാവ <map>

#അന്തര്ഭാവ <sycl/sycl.hpp>
#അന്തര്ഭാവ <sycl/half_type.hpp>
#അന്തര്ഭാവ <oneapi/mkl.hpp>

#എങ്കിൽ defined(__linux__)
#അന്തര്ഭാവ <sys/mman.h>
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#അന്തര്ഭാവ <windows.h>
#അല്ലെങ്കിൽ
#error "Only support Windows and Linux."
#endif

#എങ്കിൽ defined(__linux__)
#അന്തര്ഭാവ <unistd.h>
#അന്തര്ഭാവ <sys/syscall.h>
#endif
#എങ്കിൽ defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#അന്തര്ഭാവ <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#എങ്കിൽ defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#അല്ലെങ്കിൽ
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#എങ്കിൽ defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#അല്ലെങ്കിൽ
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (ശൂന്യം)(x)

[[noreturn]] വരിയിൽ ശൂന്യം _abort(സ്ഥിരം അക്ഷരം * str) {
    std::cerr << str << std::endl;
    std::abort();
}

വരിയിൽ std::string get_device_type_name(സ്ഥിരം sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    തിരഞ്ഞെടുപ്പ് (DeviceType) {
    അവസ്ഥ sycl::info::device_type::cpu:
        തിരികെ "cpu";
    അവസ്ഥ sycl::info::device_type::gpu:
        തിരികെ "gpu";
    അവസ്ഥ sycl::info::device_type::host:
        തിരികെ "host";
    അവസ്ഥ sycl::info::device_type::accelerator:
        തിരികെ "acc";
    സ്ഥിരസ്ഥിതി:
        തിരികെ "unknown";
    }
}

വരിയിൽ std::string get_device_backend_and_type(സ്ഥിരം sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    തിരികെ device_type.str();
}

template <typename Ts> ഘടന matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     മൂല്യം_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

നാമാകാശ dpct
{
    തരനിർവചനം sycl::queue *queue_ptr;
    തരനിർവചനം sycl::event *event_ptr;
    തരനിർവചനം അക്ഷരം *device_ptr;
    തരനിർവചനം uint8_t byte_t;
    തരനിർവചനം sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    വരിയിൽ auto exception_handler = [](sycl::exception_list exceptions)
    {
        വേണ്ടി (std::exception_ptr സ്ഥിരം &e : exceptions)
        {
            ശ്രമിക്കുക
            {
                std::rethrow_exception(e);
            }
            പിടിക്കുക (sycl::exception സ്ഥിരം &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    എണ്ണൽ error_code
    {
        success = 0,
        default_error = 999
    };

    എണ്ണൽ memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    എണ്ണൽ memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    എണ്ണൽ വർഗ്ഗം library_data_t : unsigned അക്ഷരം
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
    ഘടന DataType
    {
        using T2 = T;
    };
    template <typename T>
    ഘടന DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    സ്ഥിതം ശൂന്യം destroy_event(event_ptr event)
    {
        delete event;
    }

    സ്ഥിതം വരിയിൽ unsigned പൂർണ്ണസംഖ്യ get_tid()
    {
#എങ്കിൽ defined(__linux__)
        തിരികെ syscall(SYS_gettid);
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN64)
        തിരികെ GetCurrentThreadId();
#അല്ലെങ്കിൽ
#error "Only support Windows and Linux."
#endif
    }

    നാമാകാശ detail
    {
        സ്ഥിതം ശൂന്യം get_version(സ്ഥിരം sycl::device &dev, പൂർണ്ണസംഖ്യ &major, പൂർണ്ണസംഖ്യ &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            ആകുമ്പോൾ (i < ver.size()) {
              എങ്കിൽ (isdigit(ver[i]))
                നിർത്തുക;
              i++;
            }
            major = std::stoi(&(ver[i]));
            ആകുമ്പോൾ (i < ver.size()) {
              എങ്കിൽ (ver[i] == '.')
                നിർത്തുക;
              i++;
            }
            എങ്കിൽ (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } അല്ലെങ്കിൽ {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        വർഗ്ഗം generic_error_type
        {
        പൊതു:
            generic_error_type() = സ്ഥിരസ്ഥിതി;
            generic_error_type(T value) : value{value} {}
            operator T() സ്ഥിരം { തിരികെ value; }

        സ്വകാര്യ:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    വർഗ്ഗം dim3 {
        പൊതു:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(സ്ഥിരം sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() സ്ഥിരം { തിരികെ sycl::range<3>(z, y, x); }
    }; // namespace dim3

    വരിയിൽ dim3 operator*(സ്ഥിരം dim3 &a, സ്ഥിരം dim3 &b) {
    തിരികെ dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    വർഗ്ഗം pitched_data
    {
    പൊതു:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(ശൂന്യം *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        ശൂന്യം *get_data_ptr() { തിരികെ _data; }
        ശൂന്യം set_data_ptr(ശൂന്യം *data) { _data = data; }

        size_t get_pitch() { തിരികെ _pitch; }
        ശൂന്യം set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { തിരികെ _x; }
        ശൂന്യം set_x(size_t x) { _x = x; }

        size_t get_y() { തിരികെ _y; }
        ശൂന്യം set_y(size_t y) { _y = y; }

    സ്വകാര്യ:
        ശൂന്യം *_data;
        size_t _pitch, _x, _y;
    };

    വർഗ്ഗം device_info
    {
    പൊതു:
        // get interface
        സ്ഥിരം അക്ഷരം *get_name() സ്ഥിരം { തിരികെ _name; }
        അക്ഷരം *get_name() { തിരികെ _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, പൂർണ്ണസംഖ്യ *>,
                                   പൂർണ്ണസംഖ്യ> = 0>
        auto get_max_work_item_sizes() സ്ഥിരം
        {
            എങ്കിൽ constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                തിരികെ sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            അല്ലെങ്കിൽ
            {
                തിരികെ _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, പൂർണ്ണസംഖ്യ *>,
                                   പൂർണ്ണസംഖ്യ> = 0>
        auto get_max_work_item_sizes()
        {
            എങ്കിൽ constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                തിരികെ sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            അല്ലെങ്കിൽ
            {
                തിരികെ _max_work_item_sizes_i;
            }
        }
        തർക്കം get_host_unified_memory() സ്ഥിരം { തിരികെ _host_unified_memory; }
        പൂർണ്ണസംഖ്യ get_major_version() സ്ഥിരം { തിരികെ _major; }
        പൂർണ്ണസംഖ്യ get_minor_version() സ്ഥിരം { തിരികെ _minor; }
        പൂർണ്ണസംഖ്യ get_integrated() സ്ഥിരം { തിരികെ _integrated; }
        പൂർണ്ണസംഖ്യ get_max_clock_frequency() സ്ഥിരം { തിരികെ _frequency; }
        പൂർണ്ണസംഖ്യ get_max_compute_units() സ്ഥിരം { തിരികെ _max_compute_units; }
        പൂർണ്ണസംഖ്യ get_max_work_group_size() സ്ഥിരം { തിരികെ _max_work_group_size; }
        പൂർണ്ണസംഖ്യ get_max_sub_group_size() സ്ഥിരം { തിരികെ _max_sub_group_size; }
        പൂർണ്ണസംഖ്യ get_max_work_ഇനങ്ങൾ_per_compute_unit() സ്ഥിരം
        {
            തിരികെ max_work_ഇനങ്ങൾ_per_compute_unit;
        }
        പൂർണ്ണസംഖ്യ get_max_register_size_per_work_group() സ്ഥിരം
        {
            തിരികെ _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, പൂർണ്ണസംഖ്യ *>,
                                   പൂർണ്ണസംഖ്യ> = 0>
        auto get_max_nd_range_size() സ്ഥിരം
        {
            എങ്കിൽ constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                തിരികെ _max_nd_range_size;
            അല്ലെങ്കിൽ
                തിരികെ _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, പൂർണ്ണസംഖ്യ *>,
                                   പൂർണ്ണസംഖ്യ> = 0>
        auto get_max_nd_range_size()
        {
            എങ്കിൽ constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                തിരികെ _max_nd_range_size;
            അല്ലെങ്കിൽ
                തിരികെ _max_nd_range_size_i;
        }
        size_t get_global_mem_size() സ്ഥിരം { തിരികെ _global_mem_size; }
        size_t get_local_mem_size() സ്ഥിരം { തിരികെ _local_mem_size; }
        size_t get_max_mem_alloc_size() സ്ഥിരം { തിരികെ _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned പൂർണ്ണസംഖ്യ get_memory_clock_rate() സ്ഥിരം { തിരികെ _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned പൂർണ്ണസംഖ്യ get_memory_bus_width() സ്ഥിരം { തിരികെ _memory_bus_width; }
        uint32_t get_device_id() സ്ഥിരം { തിരികെ _device_id; }
        std::array<unsigned അക്ഷരം, 16> get_uuid() സ്ഥിരം { തിരികെ _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned പൂർണ്ണസംഖ്യ get_global_mem_cache_size() സ്ഥിരം
        {
            തിരികെ _global_mem_cache_size;
        }

        // set interface
        ശൂന്യം set_name(സ്ഥിരം അക്ഷരം *name)
        {
            size_t length = strlen(name);
            എങ്കിൽ (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            അല്ലെങ്കിൽ
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        ശൂന്യം set_max_work_item_sizes(സ്ഥിരം sycl::range<3> max_work_item_sizes)
        {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] ശൂന്യം
        set_max_work_item_sizes(സ്ഥിരം sycl::id<3> max_work_item_sizes)
        {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        ശൂന്യം set_host_unified_memory(തർക്കം host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        ശൂന്യം set_major_version(പൂർണ്ണസംഖ്യ major) { _major = major; }
        ശൂന്യം set_minor_version(പൂർണ്ണസംഖ്യ minor) { _minor = minor; }
        ശൂന്യം set_integrated(പൂർണ്ണസംഖ്യ integrated) { _integrated = integrated; }
        ശൂന്യം set_max_clock_frequency(പൂർണ്ണസംഖ്യ frequency) { _frequency = frequency; }
        ശൂന്യം set_max_compute_units(പൂർണ്ണസംഖ്യ max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        ശൂന്യം set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        ശൂന്യം set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        ശൂന്യം set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        ശൂന്യം set_max_work_group_size(പൂർണ്ണസംഖ്യ max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        ശൂന്യം set_max_sub_group_size(പൂർണ്ണസംഖ്യ max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        ശൂന്യം
        set_max_work_ഇനങ്ങൾ_per_compute_unit(പൂർണ്ണസംഖ്യ max_work_ഇനങ്ങൾ_per_compute_unit_2)
        {
            max_work_ഇനങ്ങൾ_per_compute_unit = max_work_ഇനങ്ങൾ_per_compute_unit_2;
        }
        ശൂന്യം set_max_nd_range_size(പൂർണ്ണസംഖ്യ max_nd_range_size[])
        {
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        ശൂന്യം set_memory_clock_rate(unsigned പൂർണ്ണസംഖ്യ memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        ശൂന്യം set_memory_bus_width(unsigned പൂർണ്ണസംഖ്യ memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        ശൂന്യം
        set_max_register_size_per_work_group(പൂർണ്ണസംഖ്യ max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        ശൂന്യം set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        ശൂന്യം set_uuid(std::array<unsigned അക്ഷരം, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        ശൂന്യം set_global_mem_cache_size(unsigned പൂർണ്ണസംഖ്യ global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    സ്വകാര്യ:
        അക്ഷരം _name[256];
        പൂർണ്ണസംഖ്യ _max_work_item_sizes_i[3];
        തർക്കം _host_unified_memory = അസത്യം;
        പൂർണ്ണസംഖ്യ _major;
        പൂർണ്ണസംഖ്യ _minor;
        പൂർണ്ണസംഖ്യ _integrated = 0;
        പൂർണ്ണസംഖ്യ _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned പൂർണ്ണസംഖ്യ _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned പൂർണ്ണസംഖ്യ _memory_bus_width = 64;
        unsigned പൂർണ്ണസംഖ്യ _global_mem_cache_size;
        പൂർണ്ണസംഖ്യ _max_compute_units;
        പൂർണ്ണസംഖ്യ _max_work_group_size;
        പൂർണ്ണസംഖ്യ _max_sub_group_size;
        പൂർണ്ണസംഖ്യ max_work_ഇനങ്ങൾ_per_compute_unit;
        പൂർണ്ണസംഖ്യ _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        പൂർണ്ണസംഖ്യ _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned അക്ഷരം, 16> _uuid;
    };

    സ്ഥിതം പൂർണ്ണസംഖ്യ get_major_version(സ്ഥിരം sycl::device &dev)
    {
        പൂർണ്ണസംഖ്യ major, minor;
        detail::get_version(dev, major, minor);
        തിരികെ major;
    }

    സ്ഥിതം പൂർണ്ണസംഖ്യ get_minor_version(സ്ഥിരം sycl::device &dev)
    {
        പൂർണ്ണസംഖ്യ major, minor;
        detail::get_version(dev, major, minor);
        തിരികെ minor;
    }

    സ്ഥിതം ശൂന്യം get_device_info(device_info &out, സ്ഥിരം sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        പൂർണ്ണസംഖ്യ major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#എങ്കിൽ (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#അല്ലെങ്കിൽ
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

#എങ്കിൽ (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        എങ്കിൽ (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned പൂർണ്ണസംഖ്യ tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            എങ്കിൽ (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        എങ്കിൽ (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        എങ്കിൽ (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        എങ്കിൽ (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#അല്ലെങ്കിലെങ്കിൽ defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate സ്ഥിരസ്ഥിതി value. \
        Use 64 bits as memory_bus_width സ്ഥിരസ്ഥിതി value.")
#അല്ലെങ്കിൽ
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate സ്ഥിരസ്ഥിതി value. \
        Use 64 bits as memory_bus_width സ്ഥിരസ്ഥിതി value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        വേണ്ടി (സ്ഥിരം auto &sub_group_size : sub_group_sizes)
        {
            എങ്കിൽ (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_ഇനങ്ങൾ_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        പൂർണ്ണസംഖ്യ max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    വർഗ്ഗം device_ext : പൊതു sycl::device {
      തരനിർവചനം std::mutex mutex_type;

     പൊതു:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(സ്ഥിരം sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      പൂർണ്ണസംഖ്യ is_native_atomic_supported() { തിരികെ 0; }
      പൂർണ്ണസംഖ്യ get_major_version() സ്ഥിരം { തിരികെ dpct::get_major_version(*ഇത്); }

      പൂർണ്ണസംഖ്യ get_minor_version() സ്ഥിരം { തിരികെ dpct::get_minor_version(*ഇത്); }

      പൂർണ്ണസംഖ്യ get_max_compute_units() സ്ഥിരം {
        തിരികെ get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      പൂർണ്ണസംഖ്യ get_max_clock_frequency() സ്ഥിരം {
        തിരികെ get_device_info().get_max_clock_frequency();
      }

      പൂർണ്ണസംഖ്യ get_integrated() സ്ഥിരം { തിരികെ get_device_info().get_integrated(); }

      പൂർണ്ണസംഖ്യ get_max_sub_group_size() സ്ഥിരം {
        തിരികെ get_device_info().get_max_sub_group_size();
      }

      പൂർണ്ണസംഖ്യ get_max_register_size_per_work_group() സ്ഥിരം {
        തിരികെ get_device_info().get_max_register_size_per_work_group();
      }

      പൂർണ്ണസംഖ്യ get_max_work_group_size() സ്ഥിരം {
        തിരികെ get_device_info().get_max_work_group_size();
      }

      പൂർണ്ണസംഖ്യ get_mem_base_addr_align() സ്ഥിരം {
        തിരികെ get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() സ്ഥിരം {
        തിരികെ get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() സ്ഥിരം {
        തിരികെ get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      ശൂന്യം get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        സ്ഥിരം അക്ഷരം *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#എങ്കിൽ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        എങ്കിൽ (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } അല്ലെങ്കിൽ {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#അല്ലെങ്കിൽ
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#എങ്കിൽ defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#അല്ലെങ്കിൽ
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      ശൂന്യം get_device_info(device_info &out) സ്ഥിരം {
        dpct::get_device_info(out, *ഇത്);
      }

      device_info get_device_info() സ്ഥിരം {
        device_info prop;
        dpct::get_device_info(prop, *ഇത്);
        തിരികെ prop;
      }

      ശൂന്യം reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { തിരികെ _q_in_order; }

      sycl::queue &out_of_order_queue() { തിരികെ _q_out_of_order; }

      sycl::queue &default_queue() { തിരികെ in_order_queue(); }

      ശൂന്യം queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        വേണ്ടി (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(തർക്കം enable_exception_handler = അസത്യം) {
        തിരികെ create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               തർക്കം enable_exception_handler = അസത്യം) {
        തിരികെ create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(തർക്കം enable_exception_handler = അസത്യം) {
        std::lock_guard<mutex_type> lock(m_mutex);
        തിരികെ create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        തർക്കം enable_exception_handler = അസത്യം) {
        std::lock_guard<mutex_type> lock(m_mutex);
        തിരികെ create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          തർക്കം enable_exception_handler = അസത്യം) {
        std::lock_guard<mutex_type> lock(m_mutex);
        തിരികെ create_queue_impl(enable_exception_handler);
      }

      ശൂന്യം destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](സ്ഥിരം sycl::queue &q) -> തർക്കം
                                    {
                                        തിരികെ q == queue;
                                    }),
                    _queues.end());
      }
      ശൂന്യം set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() സ്ഥിരം {
        std::lock_guard<mutex_type> lock(m_mutex);
        തിരികെ _saved_queue;
      }

     സ്വകാര്യ:
      ശൂന്യം clear_queues() { _queues.clear(); }

      ശൂന്യം init_queues() {
        _q_in_order =
            create_queue_impl(സത്യം, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(സത്യം);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <വർഗ്ഗം... Properties>
      sycl::queue create_queue_impl(തർക്കം enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        എങ്കിൽ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *ഇത്, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        തിരികെ _queues.back();
      }

      template <വർഗ്ഗം... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    തർക്കം enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        എങ്കിൽ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        തിരികെ _queues.back();
      }

      ശൂന്യം get_version(പൂർണ്ണസംഖ്യ &major, പൂർണ്ണസംഖ്യ &minor) സ്ഥിരം {
        detail::get_version(*ഇത്, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    വർഗ്ഗം dev_mgr
    {
    പൊതു:
        device_ext &current_device()
        {
            unsigned പൂർണ്ണസംഖ്യ dev_id = current_device_id();
            check_id(dev_id);
            തിരികെ *_devs[dev_id];
        }
        device_ext &cpu_device() സ്ഥിരം
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            എങ്കിൽ (_cpu_device == -1)
            {
                എറിയുക std::runtime_error("no valid cpu device");
            }
            അല്ലെങ്കിൽ
            {
                തിരികെ *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned പൂർണ്ണസംഖ്യ id) സ്ഥിരം
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            തിരികെ *_devs[id];
        }
        unsigned പൂർണ്ണസംഖ്യ current_device_id() സ്ഥിരം
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            എങ്കിൽ (it != _thread2dev_map.end())
                തിരികെ it->second;
            തിരികെ DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        ശൂന്യം select_device(unsigned പൂർണ്ണസംഖ്യ id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned പൂർണ്ണസംഖ്യ device_എണ്ണം() { തിരികെ _devs.size(); }

        unsigned പൂർണ്ണസംഖ്യ get_device_id(സ്ഥിരം sycl::device &dev)
        {
            unsigned പൂർണ്ണസംഖ്യ id = 0;
            വേണ്ടി (auto &dev_item : _devs)
            {
                എങ്കിൽ (*dev_item == dev)
                {
                    തിരികെ id;
                }
                id++;
            }
            തിരികെ -1;
        }

        വരിയിൽ std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            അക്ഷരം* env = getenv("ONEAPI_DEVICE_SELECTOR");
            എങ്കിൽ (env) {
                എങ്കിൽ (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                അല്ലെങ്കിൽ എങ്കിൽ (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                അല്ലെങ്കിൽ എങ്കിൽ (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                അല്ലെങ്കിൽ എങ്കിൽ (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                അല്ലെങ്കിൽ {
                    എറിയുക std::runtime_error("invalid device filter: " + std::string(env));
                }
            } അല്ലെങ്കിൽ {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                എങ്കിൽ (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                അല്ലെങ്കിൽ എങ്കിൽ (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                അല്ലെങ്കിൽ എങ്കിൽ (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            വേണ്ടി (സ്ഥിരം auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](സ്ഥിരം sycl::device& d) {
                    തിരികെ d.is_gpu();
                });

                എങ്കിൽ (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    തുടരുക;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_താഴെയുള്ള_case;
                platform_name_താഴെയുള്ള_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_താഴെയുള്ള_case.begin(), ::tolower);

                എങ്കിൽ (platform_name_താഴെയുള്ള_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    തുടരുക;
                }

                result = platform_name;
            }

            എങ്കിൽ (result.empty())
                എറിയുക std::runtime_error("can not find preferred GPU platform");

            തിരികെ result;
        }

        template <വർഗ്ഗം DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<പൂർണ്ണസംഖ്യ, DeviceSelector, സ്ഥിരം sycl::device &>>
        select_device(സ്ഥിരം DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned പൂർണ്ണസംഖ്യ selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        സ്ഥിതം dev_mgr &instance()
        {
            സ്ഥിതം dev_mgr d_m;
            തിരികെ d_m;
        }
        dev_mgr(സ്ഥിരം dev_mgr &) = delete;
        dev_mgr &operator=(സ്ഥിരം dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    സ്വകാര്യ:
        mutable std::recursive_mutex m_mutex;
        സ്ഥിതം തർക്കം compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            എങ്കിൽ(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) തിരികെ സത്യം;
            എങ്കിൽ(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) തിരികെ അസത്യം;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            തിരികെ prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        സ്ഥിതം പൂർണ്ണസംഖ്യ convert_backend_index(std::string & backend) {
            എങ്കിൽ (backend == "ext_oneapi_level_zero:gpu") തിരികെ 0;
            എങ്കിൽ (backend == "opencl:gpu") തിരികെ 1;
            എങ്കിൽ (backend == "ext_oneapi_cuda:gpu") തിരികെ 2;
            എങ്കിൽ (backend == "ext_oneapi_hip:gpu") തിരികെ 3;
            എങ്കിൽ (backend == "opencl:cpu") തിരികെ 4;
            എങ്കിൽ (backend == "opencl:acc") തിരികെ 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        സ്ഥിതം തർക്കം compare_backend(std::string &backend1, std::string &backend2) {
            തിരികെ convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            എങ്കിൽ (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            ആകുമ്പോൾ (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                എങ്കിൽ (platform_name.compare(preferred_platform_name) != 0) {
                    തുടരുക;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                വേണ്ടി (സ്ഥിരം auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> താക്കോലുകൾ;
            വേണ്ടി(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                താക്കോലുകൾ.push_back(it->first);
            }
            std::sort(താക്കോലുകൾ.begin(), താക്കോലുകൾ.end(), compare_backend);

            വേണ്ടി (auto &താക്കോൽ : താക്കോലുകൾ) {
                std::vector<sycl::device> devs = backend_devices[താക്കോൽ];
                std::sort(devs.begin(), devs.end(), compare_dev);
                വേണ്ടി (സ്ഥിരം auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            വേണ്ടി (auto &dev : sycl_all_devs)
            {
                എങ്കിൽ (dev == default_device)
                {
                    തുടരുക;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                എങ്കിൽ (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        ശൂന്യം check_id(unsigned പൂർണ്ണസംഖ്യ id) സ്ഥിരം
        {
            എങ്കിൽ (id >= _devs.size())
            {
                എറിയുക std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        സ്ഥിരം unsigned പൂർണ്ണസംഖ്യ DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned പൂർണ്ണസംഖ്യ, unsigned പൂർണ്ണസംഖ്യ> _thread2dev_map;
        പൂർണ്ണസംഖ്യ _cpu_device = -1;
    };

    സ്ഥിതം വരിയിൽ sycl::queue &get_default_queue()
    {
        തിരികെ dev_mgr::instance().current_device().default_queue();
    }

    നാമാകാശ detail
    {
        എണ്ണൽ വർഗ്ഗം pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        സ്ഥിതം pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              സ്ഥിരം ശൂന്യം *ptr)
        {
            തിരഞ്ഞെടുപ്പ് (sycl::get_pointer_type(ptr, q.get_context()))
            {
            അവസ്ഥ sycl::usm::alloc::unknown:
                തിരികെ pointer_access_attribute::host_only;
            അവസ്ഥ sycl::usm::alloc::device:
                തിരികെ pointer_access_attribute::device_only;
            അവസ്ഥ sycl::usm::alloc::shared:
            അവസ്ഥ sycl::usm::alloc::host:
                തിരികെ pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        വരിയിൽ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned അക്ഷരം)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned അക്ഷരം>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            തിരികെ (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        വരിയിൽ constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned അക്ഷരം>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(വലുപ്പം...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            തിരികെ get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        വർഗ്ഗം mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#എങ്കിൽ defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    ശൂന്യ,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#അല്ലെങ്കിൽ
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        പൊതു:
            using buffer_id_t = പൂർണ്ണസംഖ്യ;

            ഘടന allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#എങ്കിൽ defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#അല്ലെങ്കിലെങ്കിൽ defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#അല്ലെങ്കിൽ
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(സ്ഥിരം mem_mgr &) = delete;
            mem_mgr &operator=(സ്ഥിരം mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            ശൂന്യം *mem_alloc(size_t size)
            {
                എങ്കിൽ (!size)
                    തിരികെ nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                എങ്കിൽ (next_free + size > mapped_address_space + mapped_region_size)
                {
                    എറിയുക std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                ശൂന്യം *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                തിരികെ result;
            }

            /// Deallocate
            ശൂന്യം mem_free(സ്ഥിരം ശൂന്യം *ptr)
            {
                എങ്കിൽ (!ptr)
                    തിരികെ;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(സ്ഥിരം ശൂന്യം *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                തിരികെ it->second;
            }

            /// Check if the pointer represents device pointer or not.
            തർക്കം is_device_ptr(സ്ഥിരം ശൂന്യം *ptr) സ്ഥിരം
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                തിരികെ (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            സ്ഥിതം mem_mgr &instance()
            {
                സ്ഥിതം mem_mgr m;
                തിരികെ m;
            }

        സ്വകാര്യ:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            സ്ഥിരം size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            സ്ഥിരം size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            സ്ഥിരം size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(സ്ഥിരം ശൂന്യം *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<സ്ഥിരം byte_t *>(ptr)));
                എങ്കിൽ (it == m_map.end())
                {
                    // Not a virtual pointer.
                    എറിയുക std::runtime_error("can not get buffer from non-virtual pointer");
                }
                സ്ഥിരം allocation &alloc = it->second;
                എങ്കിൽ (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    എറിയുക std::runtime_error("invalid virtual pointer");
                }
                തിരികെ it;
            }
        };

        template <വർഗ്ഗം T, memory_region Memory, size_t Dimension>
        വർഗ്ഗം accessor;
        template <memory_region Memory, വർഗ്ഗം T = byte_t>
        വർഗ്ഗം memory_traits
        {
        പൊതു:
            സ്ഥിതം constexpr sycl::access::target target =
                sycl::access::target::device;
            സ്ഥിതം constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::പഠ
                                     : sycl::access_mode::read_write;
            സ്ഥിതം constexpr size_t type_size = വലുപ്പം(T);
            using element_t =
                typename std::conditional<Memory == constant, സ്ഥിരം T, T>::type;
            using മൂല്യം_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<മൂല്യം_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        സ്ഥിതം വരിയിൽ ശൂന്യം *dpct_malloc(size_t size, sycl::queue &q)
        {
            തിരികെ sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        സ്ഥിതം വരിയിൽ ശൂന്യം *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            തിരികെ dpct_malloc(pitch * y * z, q);
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
        template <typename മൂല്യം_T>
        സ്ഥിതം വരിയിൽ sycl::event dpct_memset(sycl::queue &q, ശൂന്യം *dev_ptr,
                                              മൂല്യം_T value, size_t size)
        {
            തിരികെ q.fill(dev_ptr, value, size);
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
        template <typename മൂല്യം_T>
        സ്ഥിതം വരിയിൽ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, മൂല്യം_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned അക്ഷരം *data_surface = (unsigned അക്ഷരം *)data.get_data_ptr();
            വേണ്ടി (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned അക്ഷരം *data_ptr = data_surface;
                വേണ്ടി (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            തിരികെ event_list;
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
        template <typename മൂല്യം_T>
        സ്ഥിതം വരിയിൽ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, ശൂന്യം *ptr, size_t pitch, മൂല്യം_T val, size_t x,
                    size_t y)
        {
            തിരികെ dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        സ്ഥിതം memcpy_direction deduce_memcpy_direction(sycl::queue &q, ശൂന്യം *to_ptr,
                                                        സ്ഥിരം ശൂന്യം *from_ptr,
                                                        memcpy_direction dir)
        {
            തിരഞ്ഞെടുപ്പ് (dir)
            {
            അവസ്ഥ memcpy_direction::host_to_host:
            അവസ്ഥ memcpy_direction::host_to_device:
            അവസ്ഥ memcpy_direction::device_to_host:
            അവസ്ഥ memcpy_direction::device_to_device:
                തിരികെ dir;
            അവസ്ഥ memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                സ്ഥിതം സ്ഥിരം memcpy_direction
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
                തിരികെ direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            സ്ഥിരസ്ഥിതി:
                എറിയുക std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        സ്ഥിതം sycl::event
        dpct_memcpy(sycl::queue &q, ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr, size_t size,
                    memcpy_direction direction,
                    സ്ഥിരം std::vector<sycl::event> &dep_events = {})
        {
            എങ്കിൽ (!size)
                തിരികെ sycl::event{};
            തിരികെ q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        സ്ഥിതം വരിയിൽ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            തിരികെ slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        സ്ഥിതം വരിയിൽ size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            തിരികെ slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        സ്ഥിതം വരിയിൽ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    സ്ഥിരം std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            വർഗ്ഗം host_buffer
            {
                ശൂന്യം *_buf;
                size_t _size;
                sycl::queue &_q;
                സ്ഥിരം std::vector<sycl::event> &_deps; // free operation depends

            പൊതു:
                host_buffer(size_t size, sycl::queue &q,
                            സ്ഥിരം std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                ശൂന്യം *get_ptr() സ്ഥിരം { തിരികെ _buf; }
                size_t get_size() സ്ഥിരം { തിരികെ _size; }
                ~host_buffer()
                {
                    എങ്കിൽ (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::സ്വതന്ത്രം(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned അക്ഷരം *to_surface =
                (unsigned അക്ഷരം *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            സ്ഥിരം unsigned അക്ഷരം *from_surface =
                (സ്ഥിരം unsigned അക്ഷരം *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            എങ്കിൽ (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                തിരികെ {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            തിരഞ്ഞെടുപ്പ് (direction)
            {
            അവസ്ഥ host_to_host:
                വേണ്ടി (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned അക്ഷരം *to_ptr = to_surface;
                    സ്ഥിരം unsigned അക്ഷരം *from_ptr = from_surface;
                    എങ്കിൽ (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    അല്ലെങ്കിൽ
                    {
                        വേണ്ടി (size_t y = 0; y < size.get(1); ++y)
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
                നിർത്തുക;
            അവസ്ഥ host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                എങ്കിൽ (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                അല്ലെങ്കിൽ
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
                നിർത്തുക;
            }
            അവസ്ഥ device_to_host:
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
                നിർത്തുക;
            }
            അവസ്ഥ device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<വർഗ്ഗം dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
                എറിയുക std::runtime_error("dpct_memcpy: invalid direction value");
            }
            തിരികെ event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        സ്ഥിതം വരിയിൽ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            തിരികെ dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        സ്ഥിതം വരിയിൽ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            തിരികെ dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        നാമാകാശ deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            വർഗ്ഗം usm_allocator
            {
            സ്വകാര്യ:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            പൊതു:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    സ്ഥിരം typename std::allocator_traits<Alloc>::value_type &;
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
                ഘടന rebind
                {
                    തരനിർവചനം usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(സ്ഥിരം usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { തിരികെ &r; }
                const_pointer address(const_reference r) { തിരികെ &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    തിരികെ std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                ശൂന്യം deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() സ്ഥിരം
                {
                    തിരികെ std::allocator_traits<Alloc>::max_size(_impl);
                }
                തർക്കം operator==(സ്ഥിരം usm_allocator &other) സ്ഥിരം { തിരികെ _impl == other._impl; }
                തർക്കം operator!=(സ്ഥിരം usm_allocator &other) സ്ഥിരം { തിരികെ _impl != other._impl; }
            };

        } // namespace deprecated

        വരിയിൽ ശൂന്യം dpct_free(ശൂന്യം *ptr,
                              സ്ഥിരം sycl::queue &q)
        {
            എങ്കിൽ (ptr)
            {
                sycl::സ്വതന്ത്രം(ptr, q.get_context());
            }
        }

        template <typename T>
        വരിയിൽ auto get_memory(സ്ഥിരം ശൂന്യം *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<ശൂന്യം *>(x));
            തിരികെ new_x;
        }

        template <typename T>
        വരിയിൽ typename DataType<T>::T2 get_മൂല്യം(സ്ഥിരം T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            എങ്കിൽ (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (ശൂന്യം *)&s_h, (സ്ഥിരം ശൂന്യം *)s, വലുപ്പം(T), device_to_host)
                    .wait();
            അല്ലെങ്കിൽ
                s_h = *reinterpret_cast<സ്ഥിരം Ty *>(s);
            തിരികെ s_h;
        }

    } // namespace detail

    template <typename T>
    വരിയിൽ auto get_മൂല്യം(സ്ഥിരം T *s, sycl::queue &q)
    {
        തിരികെ detail::get_മൂല്യം(s, q);
    }

    നാമാകാശ detail
    {
    template <വർഗ്ഗം Ta, വർഗ്ഗം Tb, വർഗ്ഗം Tc, വർഗ്ഗം Ts>
    വരിയിൽ ശൂന്യം gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, പൂർണ്ണസംഖ്യ m,
                          പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ k, സ്ഥിരം ശൂന്യം * alpha, സ്ഥിരം ശൂന്യം * a, പൂർണ്ണസംഖ്യ lda, സ്ഥിരം ശൂന്യം * b, പൂർണ്ണസംഖ്യ ldb,
                          സ്ഥിരം ശൂന്യം * beta, ശൂന്യം * c, പൂർണ്ണസംഖ്യ ldc) {
        Ts   alpha_മൂല്യം_2 = dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം Ts *>(alpha), q);
        Ts   beta_മൂല്യം  = dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം Ts *>(beta), q);
        auto data_a      = get_memory<സ്ഥിരം Ta>(a);
        auto data_b      = get_memory<സ്ഥിരം Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_മൂല്യം_2, data_a,
                                               lda, data_b, ldb, beta_മൂല്യം, data_c, ldc);
    }

        template <typename VecT, വർഗ്ഗം ദ്വിമാന_Operation, വർഗ്ഗം = ശൂന്യം>
        വർഗ്ഗം vectorized_ദ്വിമാന
        {
        പൊതു:
            വരിയിൽ VecT operator()(VecT a, VecT b, സ്ഥിരം ദ്വിമാന_Operation ദ്വിമാന_op)
            {
                VecT v4;
                വേണ്ടി (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ദ്വിമാന_op(a[i], b[i]);
                }
                തിരികെ v4;
            }
        };

        template <typename VecT, വർഗ്ഗം ദ്വിമാന_Operation>
        വർഗ്ഗം vectorized_ദ്വിമാന<
            VecT, ദ്വിമാന_Operation,
            std::void_t<std::invoke_result_t<ദ്വിമാന_Operation, VecT, VecT>>>
        {
        പൊതു:
            വരിയിൽ VecT operator()(VecT a, VecT b, സ്ഥിരം ദ്വിമാന_Operation ദ്വിമാന_op)
            {
                തിരികെ ദ്വിമാന_op(a, b).template as<VecT>();
            }
        };

        template <വർഗ്ഗം Ta, വർഗ്ഗം Tb, വർഗ്ഗം Tc, വർഗ്ഗം Ts>
        വരിയിൽ ശൂന്യം gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    പൂർണ്ണസംഖ്യ m, പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ k, സ്ഥിരം ശൂന്യം * alpha, സ്ഥിരം ശൂന്യം ** a, പൂർണ്ണസംഖ്യ lda, സ്ഥിരം ശൂന്യം ** b,
                                    പൂർണ്ണസംഖ്യ ldb, സ്ഥിരം ശൂന്യം * beta, ശൂന്യം ** c, പൂർണ്ണസംഖ്യ ldc, പൂർണ്ണസംഖ്യ batch_size,
                                    matrix_info_t<ദശാംശം> * matrix_info) {
            Ts alpha_മൂല്യം_2 = dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം Ts *>(alpha), q);
            Ts beta_മൂല്യം = dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->മൂല്യം_info[0] = alpha_മൂല്യം_2;
            matrix_info->മൂല്യം_info[1] = beta_മൂല്യം;
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
                reinterpret_cast<Ts *>(matrix_info->മൂല്യം_info), reinterpret_cast<സ്ഥിരം Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<സ്ഥിരം Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->മൂല്യം_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <വർഗ്ഗം Ta, വർഗ്ഗം Tb, വർഗ്ഗം Tc, വർഗ്ഗം Ts>
        വരിയിൽ ശൂന്യം gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    പൂർണ്ണസംഖ്യ m, പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ k, സ്ഥിരം ശൂന്യം * alpha, സ്ഥിരം ശൂന്യം * a, പൂർണ്ണസംഖ്യ lda,
                                    long long പൂർണ്ണസംഖ്യ stride_a, സ്ഥിരം ശൂന്യം * b, പൂർണ്ണസംഖ്യ ldb, long long പൂർണ്ണസംഖ്യ stride_b,
                                    സ്ഥിരം ശൂന്യം * beta, ശൂന്യം * c, പൂർണ്ണസംഖ്യ ldc, long long പൂർണ്ണസംഖ്യ stride_c, പൂർണ്ണസംഖ്യ batch_size) {
            Ts alpha_മൂല്യം_2 = dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം Ts *>(alpha), q);
            Ts beta_മൂല്യം = dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം Ts *>(beta), q);
            auto data_a = get_memory<സ്ഥിരം Ta>(a);
            auto data_b = get_memory<സ്ഥിരം Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_മൂല്യം_2,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_മൂല്യം,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, വർഗ്ഗം ദ്വിമാന_Operation>
    വരിയിൽ unsigned vectorized_ദ്വിമാന(unsigned a, unsigned b,
                                      സ്ഥിരം ദ്വിമാന_Operation ദ്വിമാന_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_ദ്വിമാന<VecT, ദ്വിമാന_Operation>()(v2, v3, ദ്വിമാന_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        തിരികെ v0;
    }

    സ്ഥിതം ശൂന്യം async_dpct_memcpy(ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    സ്ഥിതം വരിയിൽ unsigned പൂർണ്ണസംഖ്യ select_device(unsigned പൂർണ്ണസംഖ്യ id)
    {
        dev_mgr::instance().select_device(id);
        തിരികെ id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned പൂർണ്ണസംഖ്യ mask,
                               unsigned പൂർണ്ണസംഖ്യ logical_sub_group_size = 32)
    {
        unsigned പൂർണ്ണസംഖ്യ id = g.get_local_linear_id();
        unsigned പൂർണ്ണസംഖ്യ start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned പൂർണ്ണസംഖ്യ target_offset = (id % logical_sub_group_size) ^ mask;
        തിരികെ sycl::select_from_group(g, x,
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
      തിരികെ sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    വരിയിൽ auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      തിരികെ res;
    }

    ഘടന sub_sat
    {
        template <typename T>
        auto operator()(സ്ഥിരം T x, സ്ഥിരം T y) സ്ഥിരം
        {
            തിരികെ sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    വരിയിൽ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        തിരികെ v0;
    }

    വരിയിൽ ദശാംശം pow(സ്ഥിരം ദശാംശം a, സ്ഥിരം പൂർണ്ണസംഖ്യ b) { തിരികെ sycl::pown(a, b); }
    വരിയിൽ ഇരട്ട pow(സ്ഥിരം ഇരട്ട a, സ്ഥിരം പൂർണ്ണസംഖ്യ b) { തിരികെ sycl::pown(a, b); }
    വരിയിൽ ദശാംശം pow(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) { തിരികെ sycl::pow(a, b); }
    വരിയിൽ ഇരട്ട pow(സ്ഥിരം ഇരട്ട a, സ്ഥിരം ഇരട്ട b) { തിരികെ sycl::pow(a, b); }
    template <typename T, typename U>
    വരിയിൽ typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(സ്ഥിരം T a, സ്ഥിരം U b)
    {
        തിരികെ sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    വരിയിൽ typename std::enable_if_t<!std::is_floating_point_v<T>, ഇരട്ട>
    pow(സ്ഥിരം T a, സ്ഥിരം U b)
    {
        തിരികെ sycl::pow(static_cast<ഇരട്ട>(a), static_cast<ഇരട്ട>(b));
    }

    വരിയിൽ ഇരട്ട min(സ്ഥിരം ഇരട്ട a, സ്ഥിരം ദശാംശം b)
    {
        തിരികെ sycl::fmin(a, static_cast<ഇരട്ട>(b));
    }
    വരിയിൽ ഇരട്ട min(സ്ഥിരം ദശാംശം a, സ്ഥിരം ഇരട്ട b)
    {
        തിരികെ sycl::fmin(static_cast<ഇരട്ട>(a), b);
    }
    വരിയിൽ ദശാംശം min(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) { തിരികെ sycl::fmin(a, b); }
    വരിയിൽ ഇരട്ട min(സ്ഥിരം ഇരട്ട a, സ്ഥിരം ഇരട്ട b) { തിരികെ sycl::fmin(a, b); }
    വരിയിൽ std::uint32_t min(സ്ഥിരം std::uint32_t a, സ്ഥിരം std::int32_t b)
    {
        തിരികെ sycl::min(a, static_cast<std::uint32_t>(b));
    }
    വരിയിൽ std::uint32_t min(സ്ഥിരം std::int32_t a, സ്ഥിരം std::uint32_t b)
    {
        തിരികെ sycl::min(static_cast<std::uint32_t>(a), b);
    }
    വരിയിൽ std::int32_t min(സ്ഥിരം std::int32_t a, സ്ഥിരം std::int32_t b)
    {
        തിരികെ sycl::min(a, b);
    }
    വരിയിൽ std::uint32_t min(സ്ഥിരം std::uint32_t a, സ്ഥിരം std::uint32_t b)
    {
        തിരികെ sycl::min(a, b);
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::int64_t b)
    {
        തിരികെ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::int64_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    വരിയിൽ std::int64_t min(സ്ഥിരം std::int64_t a, സ്ഥിരം std::int64_t b)
    {
        തിരികെ sycl::min(a, b);
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::min(a, b);
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::int32_t b)
    {
        തിരികെ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::int32_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::uint32_t b)
    {
        തിരികെ sycl::min(a, static_cast<std::uint64_t>(b));
    }
    വരിയിൽ std::uint64_t min(സ്ഥിരം std::uint32_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    വരിയിൽ ഇരട്ട max(സ്ഥിരം ഇരട്ട a, സ്ഥിരം ദശാംശം b)
    {
        തിരികെ sycl::fmax(a, static_cast<ഇരട്ട>(b));
    }
    വരിയിൽ ഇരട്ട max(സ്ഥിരം ദശാംശം a, സ്ഥിരം ഇരട്ട b)
    {
        തിരികെ sycl::fmax(static_cast<ഇരട്ട>(a), b);
    }
    വരിയിൽ ദശാംശം max(സ്ഥിരം ദശാംശം a, സ്ഥിരം ദശാംശം b) { തിരികെ sycl::fmax(a, b); }
    വരിയിൽ ഇരട്ട max(സ്ഥിരം ഇരട്ട a, സ്ഥിരം ഇരട്ട b) { തിരികെ sycl::fmax(a, b); }
    വരിയിൽ std::uint32_t max(സ്ഥിരം std::uint32_t a, സ്ഥിരം std::int32_t b)
    {
        തിരികെ sycl::max(a, static_cast<std::uint32_t>(b));
    }
    വരിയിൽ std::uint32_t max(സ്ഥിരം std::int32_t a, സ്ഥിരം std::uint32_t b)
    {
        തിരികെ sycl::max(static_cast<std::uint32_t>(a), b);
    }
    വരിയിൽ std::int32_t max(സ്ഥിരം std::int32_t a, സ്ഥിരം std::int32_t b)
    {
        തിരികെ sycl::max(a, b);
    }
    വരിയിൽ std::uint32_t max(സ്ഥിരം std::uint32_t a, സ്ഥിരം std::uint32_t b)
    {
        തിരികെ sycl::max(a, b);
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::int64_t b)
    {
        തിരികെ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::int64_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    വരിയിൽ std::int64_t max(സ്ഥിരം std::int64_t a, സ്ഥിരം std::int64_t b)
    {
        തിരികെ sycl::max(a, b);
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::max(a, b);
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::int32_t b)
    {
        തിരികെ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::int32_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::max(static_cast<std::uint64_t>(a), b);
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::uint64_t a, സ്ഥിരം std::uint32_t b)
    {
        തിരികെ sycl::max(a, static_cast<std::uint64_t>(b));
    }
    വരിയിൽ std::uint64_t max(സ്ഥിരം std::uint32_t a, സ്ഥിരം std::uint64_t b)
    {
        തിരികെ sycl::max(static_cast<std::uint64_t>(a), b);
    }

    വരിയിൽ ശൂന്യം
    has_capability_or_fail(സ്ഥിരം sycl::device &dev,
                           സ്ഥിരം std::initializer_list<sycl::aspect> &props)
    {
        വേണ്ടി (സ്ഥിരം auto &it : props)
        {
            എങ്കിൽ (dev.has(it))
                തുടരുക;
            തിരഞ്ഞെടുപ്പ് (it)
            {
            അവസ്ഥ sycl::aspect::fp64:
                എറിയുക std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                നിർത്തുക;
            അവസ്ഥ sycl::aspect::fp16:
                എറിയുക std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                നിർത്തുക;
            സ്ഥിരസ്ഥിതി:
#define __SYCL_ASPECT(ASPECT, ID) \
    അവസ്ഥ sycl::aspect::ASPECT:    \
        തിരികെ #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect Aspect_അക്കം) -> std::string
                {
                    തിരഞ്ഞെടുപ്പ് (Aspect_അക്കം)
                    {
#അന്തര്ഭാവ <sycl/info/aspects.def>
#അന്തര്ഭാവ <sycl/info/aspects_deprecated.def>
                    സ്ഥിരസ്ഥിതി:
                        തിരികെ "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                എറിയുക std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            നിർത്തുക;
        }
    }

    സ്ഥിതം വരിയിൽ unsigned പൂർണ്ണസംഖ്യ get_current_device_id()
    {
        തിരികെ dev_mgr::instance().current_device_id();
    }

    സ്ഥിതം വരിയിൽ device_ext &get_current_device()
    {
        തിരികെ dev_mgr::instance().current_device();
    }

    സ്ഥിതം വരിയിൽ device_ext &get_device(unsigned പൂർണ്ണസംഖ്യ id)
    {
        തിരികെ dev_mgr::instance().get_device(id);
    }

    സ്ഥിതം വരിയിൽ sycl::queue &get_in_order_queue()
    {
        തിരികെ dev_mgr::instance().current_device().in_order_queue();
    }

    സ്ഥിതം sycl::event
    dpct_memcpy(sycl::queue &q, ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr, size_t size,
                memcpy_direction direction,
                സ്ഥിരം std::vector<sycl::event> &dep_events = {})
    {
        എങ്കിൽ (!size)
            തിരികെ sycl::event{};
        തിരികെ q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    സ്ഥിതം വരിയിൽ size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        തിരികെ slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    സ്ഥിതം വരിയിൽ size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        തിരികെ slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    സ്ഥിതം വരിയിൽ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                സ്ഥിരം std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        വർഗ്ഗം host_buffer
        {
            ശൂന്യം *_buf;
            size_t _size;
            sycl::queue &_q;
            സ്ഥിരം std::vector<sycl::event> &_deps; // free operation depends

        പൊതു:
            host_buffer(size_t size, sycl::queue &q,
                        സ്ഥിരം std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            ശൂന്യം *get_ptr() സ്ഥിരം { തിരികെ _buf; }
            size_t get_size() സ്ഥിരം { തിരികെ _size; }
            ~host_buffer()
            {
                എങ്കിൽ (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::സ്വതന്ത്രം(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned അക്ഷരം *to_surface =
            (unsigned അക്ഷരം *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        സ്ഥിരം unsigned അക്ഷരം *from_surface =
            (സ്ഥിരം unsigned അക്ഷരം *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        എങ്കിൽ (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            തിരികെ {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        തിരഞ്ഞെടുപ്പ് (direction)
        {
        അവസ്ഥ host_to_host:
            വേണ്ടി (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned അക്ഷരം *to_ptr = to_surface;
                സ്ഥിരം unsigned അക്ഷരം *from_ptr = from_surface;
                എങ്കിൽ (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                അല്ലെങ്കിൽ
                {
                    വേണ്ടി (size_t y = 0; y < size.get(1); ++y)
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
            നിർത്തുക;
        അവസ്ഥ host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            എങ്കിൽ (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            അല്ലെങ്കിൽ
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
            നിർത്തുക;
        }
        അവസ്ഥ device_to_host:
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
            നിർത്തുക;
        }
        അവസ്ഥ device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<വർഗ്ഗം dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            എറിയുക std::runtime_error("dpct_memcpy: invalid direction value");
        }
        തിരികെ event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    സ്ഥിതം വരിയിൽ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        തിരികെ dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    സ്ഥിതം വരിയിൽ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, ശൂന്യം *to_ptr, സ്ഥിരം ശൂന്യം *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        തിരികെ dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    വരിയിൽ ശൂന്യം gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, പൂർണ്ണസംഖ്യ m, പൂർണ്ണസംഖ്യ n,
                     പൂർണ്ണസംഖ്യ k, സ്ഥിരം ശൂന്യം * alpha, സ്ഥിരം ശൂന്യം * a, library_data_t a_type, പൂർണ്ണസംഖ്യ lda, സ്ഥിരം ശൂന്യം * b,
                     library_data_t b_type, പൂർണ്ണസംഖ്യ ldb, സ്ഥിരം ശൂന്യം * beta, ശൂന്യം * c, library_data_t c_type, പൂർണ്ണസംഖ്യ ldc,
                     library_data_t scaling_type) {
        എങ്കിൽ (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        അല്ലെങ്കിൽ എങ്കിൽ (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t താക്കോൽ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        തിരഞ്ഞെടുപ്പ് (താക്കോൽ)
        {
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<ദശാംശം, ദശാംശം, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<ഇരട്ട, ഇരട്ട, ഇരട്ട, ഇരട്ട>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<ദശാംശം>, std::complex<ദശാംശം>,
                              std::complex<ദശാംശം>, std::complex<ദശാംശം>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<ഇരട്ട>, std::complex<ഇരട്ട>,
                              std::complex<ഇരട്ട>, std::complex<ഇരട്ട>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
#ifdef __INTEL_MKL__
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ദശാംശം alpha_മൂല്യം_2 =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം ദശാംശം *>(alpha), q);
            ദശാംശം beta_മൂല്യം =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം ദശാംശം *>(beta), q);
            sycl::half alpha_half(alpha_മൂല്യം_2);
            sycl::half beta_half(beta_മൂല്യം);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ദശാംശം alpha_float =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം std::int32_t *>(alpha), q);
            ദശാംശം beta_float =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            നിർത്തുക;
        }
#endif // __INTEL_MKL__
        സ്ഥിരസ്ഥിതി:
            എറിയുക std::runtime_error("the combination of data type is unsupported");
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
    വരിയിൽ ശൂന്യം gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, പൂർണ്ണസംഖ്യ m,
                           പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ k, സ്ഥിരം ശൂന്യം * alpha, സ്ഥിരം ശൂന്യം * a[], library_data_t a_type, പൂർണ്ണസംഖ്യ lda,
                           സ്ഥിരം ശൂന്യം * b[], library_data_t b_type, പൂർണ്ണസംഖ്യ ldb, സ്ഥിരം ശൂന്യം * beta, ശൂന്യം * c[],
                           library_data_t c_type, പൂർണ്ണസംഖ്യ ldc, പൂർണ്ണസംഖ്യ batch_size, library_data_t scaling_type,
                           matrix_info_t<ദശാംശം> * matrix_info) {
        std::uint64_t താക്കോൽ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        തിരഞ്ഞെടുപ്പ് (താക്കോൽ)
        {
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ദശാംശം, ദശാംശം, ദശാംശം, ദശാംശം>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<ഇരട്ട, ഇരട്ട, ഇരട്ട, ഇരട്ട>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
#ifdef __INTEL_MKL__
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
#endif
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            ദശാംശം alpha_float =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം std::int32_t *>(alpha), q);
            ദശാംശം beta_float =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ദശാംശം alpha_മൂല്യം_2 =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം ദശാംശം *>(alpha), q);
            ദശാംശം beta_മൂല്യം =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം ദശാംശം *>(beta), q);
            sycl::half alpha_half(alpha_മൂല്യം_2);
            sycl::half beta_half(beta_മൂല്യം);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            എറിയുക std::runtime_error("the combination of data type is unsupported");
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
    വരിയിൽ ശൂന്യം gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, പൂർണ്ണസംഖ്യ m,
                           പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ k, സ്ഥിരം ശൂന്യം * alpha, സ്ഥിരം ശൂന്യം * a, library_data_t a_type, പൂർണ്ണസംഖ്യ lda,
                           long long പൂർണ്ണസംഖ്യ stride_a, സ്ഥിരം ശൂന്യം * b, library_data_t b_type, പൂർണ്ണസംഖ്യ ldb,
                           long long പൂർണ്ണസംഖ്യ stride_b, സ്ഥിരം ശൂന്യം * beta, ശൂന്യം * c, library_data_t c_type, പൂർണ്ണസംഖ്യ ldc,
                           long long പൂർണ്ണസംഖ്യ stride_c, പൂർണ്ണസംഖ്യ batch_size, library_data_t scaling_type) {
        എങ്കിൽ (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        അല്ലെങ്കിൽ എങ്കിൽ (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t താക്കോൽ =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        തിരഞ്ഞെടുപ്പ് (താക്കോൽ)
        {
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<ദശാംശം, ദശാംശം, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<ഇരട്ട, ഇരട്ട, ഇരട്ട, ഇരട്ട>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<ദശാംശം>, std::complex<ദശാംശം>,
                                    std::complex<ദശാംശം>, std::complex<ദശാംശം>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<ഇരട്ട>, std::complex<ഇരട്ട>,
                                    std::complex<ഇരട്ട>, std::complex<ഇരട്ട>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
#ifdef __INTEL_MKL__
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            നിർത്തുക;
        }
#endif
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, ദശാംശം, ദശാംശം>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        അവസ്ഥ detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            ദശാംശം alpha_മൂല്യം_2 =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം ദശാംശം *>(alpha), q);
            ദശാംശം beta_മൂല്യം =
                dpct::get_മൂല്യം(reinterpret_cast<സ്ഥിരം ദശാംശം *>(beta), q);
            sycl::half alpha_half(alpha_മൂല്യം_2);
            sycl::half beta_half(beta_മൂല്യം);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            നിർത്തുക;
        }
        സ്ഥിരസ്ഥിതി:
            എറിയുക std::runtime_error("the combination of data type is unsupported");
        }
    }

    സ്ഥിതം വരിയിൽ ശൂന്യം
    async_dpct_memcpy(ശൂന്യം *to_ptr, size_t to_pitch, സ്ഥിരം ശൂന്യം *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<ഘടന err0_tag, പൂർണ്ണസംഖ്യ>;
    using err1 = detail::generic_error_type<ഘടന err1_tag, പൂർണ്ണസംഖ്യ>;

    സ്ഥിതം വരിയിൽ ശൂന്യം dpct_free(ശൂന്യം *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <വർഗ്ഗം T, memory_region Memory, size_t Dimension> വർഗ്ഗം accessor;
    template <വർഗ്ഗം T, memory_region Memory> വർഗ്ഗം accessor<T, Memory, 3> {
    പൊതു:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, സ്ഥിരം sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, സ്ഥിരം accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(സ്ഥിരം accessor_t &acc, സ്ഥിരം sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) സ്ഥിരം {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            തിരികെ accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() സ്ഥിരം { തിരികെ _data; }

    സ്വകാര്യ:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <വർഗ്ഗം T, memory_region Memory> വർഗ്ഗം accessor<T, Memory, 2> {
    പൊതു:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, സ്ഥിരം sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, സ്ഥിരം accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(സ്ഥിരം accessor_t &acc, സ്ഥിരം sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) സ്ഥിരം {
            തിരികെ _data + _range.get(1) * index;
        }

        pointer_t get_ptr() സ്ഥിരം { തിരികെ _data; }

    സ്വകാര്യ:
        pointer_t _data;
        sycl::range<2> _range;
    };

    നാമാകാശ detail {
        /// Device variable with address space of shared, global or constant.
        template <വർഗ്ഗം T, memory_region Memory, size_t Dimension> വർഗ്ഗം device_memory {
        പൊതു:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using മൂല്യം_t = typename detail::memory_traits<Memory, T>::മൂല്യം_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(സ്ഥിരം sycl::range<Dimension> &in_range,
                        std::initializer_list<മൂല്യം_t> &&init_list)
                : device_memory(in_range) {
                പ്രതിജ്ഞാ(init_list.size() <= in_range.size());
                _host_ptr = (മൂല്യം_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * വലുപ്പം(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                സ്ഥിരം typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<മൂല്യം_t>> &&init_list)
                : device_memory(in_range) {
                പ്രതിജ്ഞാ(init_list.size() <= in_range[0]);
                _host_ptr = (മൂല്യം_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                വേണ്ടി (auto sub_list : init_list) {
                    പ്രതിജ്ഞാ(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * വലുപ്പം(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(സ്ഥിരം sycl::range<Dimension> &range_in)
                : _size(range_in.size() * വലുപ്പം(T)), _range(range_in),
                _reference(അസത്യം), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <വർഗ്ഗം... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                എങ്കിൽ (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                എങ്കിൽ (_host_ptr)
                    std::സ്വതന്ത്രം(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            ശൂന്യം init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            ശൂന്യം init(sycl::queue &q) {
                എങ്കിൽ (_device_ptr)
                    തിരികെ;
                എങ്കിൽ (!_size)
                    തിരികെ;
                allocate_device(q);
                എങ്കിൽ (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            ശൂന്യം assign(മൂല്യം_t *src, size_t size) {
                ഇത്->~device_memory();
                പുതിയ (ഇത്) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            മൂല്യം_t *get_ptr() { തിരികെ get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            മൂല്യം_t *get_ptr(sycl::queue &q) {
                init(q);
                തിരികെ _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { തിരികെ _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                തിരികെ _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                തിരികെ dpct_accessor_t((T *)_device_ptr, _range);
            }

        സ്വകാര്യ:
            device_memory(മൂല്യം_t *memory_ptr, size_t size)
                : _size(size), _range(size / വലുപ്പം(T)), _reference(സത്യം),
                _device_ptr(memory_ptr) {}

            ശൂന്യം allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                എങ്കിൽ (Memory == shared) {
                    _device_ptr = (മൂല്യം_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    തിരികെ;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                എങ്കിൽ (Memory == constant) {
                    _device_ptr = (മൂല്യം_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    തിരികെ;
                }
        #endif
        #endif
                _device_ptr = (മൂല്യം_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            തർക്കം _reference;
            മൂല്യം_t *_host_ptr;
            മൂല്യം_t *_device_ptr;
        };
        template <വർഗ്ഗം T, memory_region Memory>
        വർഗ്ഗം device_memory<T, Memory, 0> : പൊതു device_memory<T, Memory, 1> {
        പൊതു:
            using base = device_memory<T, Memory, 1>;
            using മൂല്യം_t = typename base::മൂല്യം_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(സ്ഥിരം മൂല്യം_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <വർഗ്ഗം T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <വർഗ്ഗം T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <വർഗ്ഗം T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    വരിയിൽ T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    തിരികെ atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    വരിയിൽ T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    തിരികെ atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    വരിയിൽ T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    തിരഞ്ഞെടുപ്പ് (memoryOrder) {
        അവസ്ഥ sycl::memory_order::relaxed:
            തിരികെ atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        അവസ്ഥ sycl::memory_order::acq_rel:
            തിരികെ atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        അവസ്ഥ sycl::memory_order::seq_cst:
            തിരികെ atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        സ്ഥിരസ്ഥിതി:
            പ്രതിജ്ഞാ(അസത്യം && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    വരിയിൽ T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    വരിയിൽ unsigned പൂർണ്ണസംഖ്യ byte_level_permute(
        unsigned പൂർണ്ണസംഖ്യ a, unsigned പൂർണ്ണസംഖ്യ b, unsigned പൂർണ്ണസംഖ്യ s) {
      unsigned പൂർണ്ണസംഖ്യ ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      തിരികെ ret;
    }

    വരിയിൽ uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, പൂർണ്ണസംഖ്യ mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      എങ്കിൽ (mode >= 1 && mode <= 6) {
        തിരികെ byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } അല്ലെങ്കിൽ എങ്കിൽ (!mode) {
        തിരികെ byte_level_permute(low32, high32, sel);
      }
      തിരികെ 0;
    }

    template <പൂർണ്ണസംഖ്യ n_nondefault_params, പൂർണ്ണസംഖ്യ n_default_params, typename T>
    വർഗ്ഗം args_selector;

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
    template <പൂർണ്ണസംഖ്യ n_nondefault_params, പൂർണ്ണസംഖ്യ n_default_params, typename R,
              typename... Ts>
    വർഗ്ഗം args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      സ്വകാര്യ:
        ശൂന്യം **kernel_params;
        അക്ഷരം *args_buffer;

        template <പൂർണ്ണസംഖ്യ i> സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ account_for_default_params() {
            constexpr പൂർണ്ണസംഖ്യ n_total_params = വലുപ്പം...(Ts);
            എങ്കിൽ constexpr (i >= n_nondefault_params) {
                തിരികെ n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } അല്ലെങ്കിൽ {
                തിരികെ i;
            }
        }

      പൊതു:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <പൂർണ്ണസംഖ്യ i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ params_അക്കം = വലുപ്പം...(Ts);

      സ്വകാര്യ:
        template <പൂർണ്ണസംഖ്യ i> സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ get_offset() {
            എങ്കിൽ constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                തിരികെ 0;
            } അല്ലെങ്കിൽ {
                constexpr പൂർണ്ണസംഖ്യ prev_off = get_offset<i - 1>();
                constexpr പൂർണ്ണസംഖ്യ prev_past_end =
                    prev_off + വലുപ്പം(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                എങ്കിൽ constexpr (prev_past_end % alignof(T) == 0) {
                    തിരികെ prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                അല്ലെങ്കിൽ {
                    തിരികെ prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        സ്ഥിതം അക്ഷരം *get_args_buffer(ശൂന്യം **extra) {
            എങ്കിൽ (!extra)
                തിരികെ nullptr;
            വേണ്ടി (; (std::size_t)*extra != 0; ++extra) {
                എങ്കിൽ ((std::size_t)*extra == 1) {
                    തിരികെ static_cast<അക്ഷരം *>(*(extra + 1));
                }
            }
            തിരികെ nullptr;
        }

      പൊതു:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(ശൂന്യം **kernel_params, ശൂന്യം **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <പൂർണ്ണസംഖ്യ i> arg_type<i> &get() {
            എങ്കിൽ (kernel_params) {
                തിരികെ *static_cast<arg_type<i> *>(kernel_params[i]);
            } അല്ലെങ്കിൽ {
                തിരികെ *reinterpret_cast<arg_type<i> *>(args_buffer +
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
    വർഗ്ഗം kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        സ്ഥിതം ശൂന്യം launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        സ്ഥിതം ശൂന്യം set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned പൂർണ്ണസംഖ്യ local_mem_size,
                                         queue_ptr que) {
            എങ്കിൽ (que) {
                _que = que;
            } അല്ലെങ്കിൽ {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        സ്ഥിതം വരിയിൽ std::mutex kernel_function_ptr_map_mutex;

      പൊതു:
        /// Variables for storing execution configuration.
        സ്ഥിതം വരിയിൽ thread_local sycl::queue *_que = nullptr;
        സ്ഥിതം വരിയിൽ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        സ്ഥിതം വരിയിൽ thread_local unsigned പൂർണ്ണസംഖ്യ _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        സ്ഥിതം വരിയിൽ std::map<
            സ്ഥിരം ശൂന്യം *,
            std::function<ശൂന്യം(dim3, dim3, ശൂന്യം **, unsigned പൂർണ്ണസംഖ്യ, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        സ്ഥിതം ശൂന്യം register_kernel_ptr(
            സ്ഥിരം ശൂന്യം *func,
            std::function<ശൂന്യം(dim3, dim3, ശൂന്യം **, unsigned പൂർണ്ണസംഖ്യ, queue_ptr)>
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
        സ്ഥിതം std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, ശൂന്യം>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned പൂർണ്ണസംഖ്യ local_mem_size, queue_ptr que, ArgsT... args) {
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
        സ്ഥിതം ശൂന്യം launch(സ്ഥിരം ശൂന്യം *func, dim3 group_range, dim3 local_range,
                           ശൂന്യം **args, unsigned പൂർണ്ണസംഖ്യ local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            എങ്കിൽ (Iter == kernel_function_ptr_map.end()) {
                എറിയുക std::runtime_error("dpct::launch() : no registered "
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
        സ്ഥിതം std::enable_if_t<std::is_function_v<FuncT>, ശൂന്യം>
        launch(FuncT *func, dim3 group_range, dim3 local_range, ശൂന്യം **args,
               unsigned പൂർണ്ണസംഖ്യ local_mem_size, queue_ptr que) {
            constexpr size_t p_അക്കം = args_selector<0, 0, FuncT>::params_അക്കം;
            set_execution_config(group_range, local_range, local_mem_size, que);
            args_selector<p_അക്കം, p_അക്കം, FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_അക്കം>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    template <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        പൂർണ്ണസംഖ്യ remote_local_id,
        പൂർണ്ണസംഖ്യ logical_sub_group_size = 32) {
      unsigned പൂർണ്ണസംഖ്യ start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      തിരികെ sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    ശൂന്യം ldmatrix(uintptr_t addr, T* m, തർക്കം trans = അസത്യം, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      പൂർണ്ണസംഖ്യ lane = sg.get_local_linear_id();

      പൂർണ്ണസംഖ്യ lane_group8_row = lane / 8;
      പൂർണ്ണസംഖ്യ lane_group8_col = lane % 8;

      എങ്കിൽ (!trans) {
        // calculate the source lane
        പൂർണ്ണസംഖ്യ src_lane = 2 * lane_group8_row;
        എങ്കിൽ (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } അല്ലെങ്കിൽ {
        // calculate the source lane
        പൂർണ്ണസംഖ്യ src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        പൂർണ്ണസംഖ്യ index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    ശൂന്യം ldmatrix(uintptr_t addr, T* m1, T* m2, തർക്കം trans = അസത്യം) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    ശൂന്യം ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, തർക്കം trans = അസത്യം) {
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
    ഘടന MMAType {
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
    template <പൂർണ്ണസംഖ്യ M, പൂർണ്ണസംഖ്യ N, പൂർണ്ണസംഖ്യ K, typename ABType, typename CDType>
    ശൂന്യം mma(
        അസ്ഥിര ശൂന്യം** d_mat_frag,
        ശൂന്യം* a_mat_frag,
        ശൂന്യം* b_mat_frag,
        ശൂന്യം* c_mat_frag) {
      auto d = reinterpret_cast<അസ്ഥിര CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      പൂർണ്ണസംഖ്യ lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      എങ്കിൽ constexpr (M == 8 && N == 8 && K == 4) {
        എങ്കിൽ constexpr (std::is_floating_point_v<CDType>) {
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
          പൂർണ്ണസംഖ്യ r_ind = (lane % 2) ? 1 : 0;
          പൂർണ്ണസംഖ്യ c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
                static_cast<ദശാംശം>(ra[r_ind]) * static_cast<ദശാംശം>(rb[c_ind]);
            *d[1] += static_cast<ദശാംശം>(ra[r_ind]) *
                     static_cast<ദശാംശം>(rb[c_ind + 1]);
            *d[2] += static_cast<ദശാംശം>(ra[r_ind + 2]) *
                     static_cast<ദശാംശം>(rb[c_ind]);
            *d[3] += static_cast<ദശാംശം>(ra[r_ind + 2]) *
                     static_cast<ദശാംശം>(rb[c_ind + 1]);

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
                static_cast<ദശാംശം>(ra[r_ind]) * static_cast<ദശാംശം>(rb[c_ind]);
            *d[5] += static_cast<ദശാംശം>(ra[r_ind]) *
                     static_cast<ദശാംശം>(rb[c_ind + 1]);
            *d[6] += static_cast<ദശാംശം>(ra[r_ind + 2]) *
                     static_cast<ദശാംശം>(rb[c_ind]);
            *d[7] += static_cast<ദശാംശം>(ra[r_ind + 2]) *
                     static_cast<ദശാംശം>(rb[c_ind + 1]);
          }
        }
      } അല്ലെങ്കിൽ എങ്കിൽ constexpr (M == 8 && N == 8 && K == 16) {
        എങ്കിൽ constexpr (std::is_integral_v<ABType>) {
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
          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } അല്ലെങ്കിൽ എങ്കിൽ constexpr (M == 16 && N == 8 && K == 8) {
        എങ്കിൽ constexpr (std::is_floating_point_v<CDType>) {
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
          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 2; j++) {
              *d[0] += static_cast<ദശാംശം>(ra[j]) * static_cast<ദശാംശം>(rb[j]);
              *d[1] +=
                  static_cast<ദശാംശം>(ra[j]) * static_cast<ദശാംശം>(rb[j + 2]);
              *d[2] +=
                  static_cast<ദശാംശം>(ra[j + 2]) * static_cast<ദശാംശം>(rb[j]);
              *d[3] +=
                  static_cast<ദശാംശം>(ra[j + 2]) * static_cast<ദശാംശം>(rb[j + 2]);
            }
          }
        }
      } അല്ലെങ്കിൽ എങ്കിൽ constexpr (M == 16 && N == 8 && K == 16) {
        എങ്കിൽ constexpr (std::is_floating_point_v<CDType>) {
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
          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } അല്ലെങ്കിൽ എങ്കിൽ constexpr (std::is_integral_v<ABType>) {
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
          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } അല്ലെങ്കിൽ എങ്കിൽ constexpr (M == 16 && N == 8 && K == 32) {
        എങ്കിൽ constexpr (std::is_integral_v<ABType>) {
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
          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < 4; i++) {
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
            വേണ്ടി (പൂർണ്ണസംഖ്യ j = 0; j < 4; j++) {
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

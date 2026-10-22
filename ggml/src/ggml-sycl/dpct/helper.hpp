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

#ضمن <cstdlib>
#ضمن <iostream>
#ضمن <map>

#ضمن <sycl/sycl.hpp>
#ضمن <sycl/half_type.hpp>
#ضمن <oneapi/mkl.hpp>

#𐤀𐤌 defined(__linux__)
#ضمن <sys/mman.h>
#وإلاإذا defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ضمن <windows.h>
#وإلا
#error "Only support Windows and Linux."
#endif

#𐤀𐤌 defined(__linux__)
#ضمن <unistd.h>
#ضمن <sys/syscall.h>
#endif
#𐤀𐤌 defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ضمن <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#𐤀𐤌 defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#وإلا
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#𐤀𐤌 defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#وإلا
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (فراغ)(x)

[[noreturn]] مضمن فراغ _abort(ثابت حرف * str) {
    std::cerr << str << std::endl;
    std::abort();
}

مضمن std::string get_device_type_name(ثابت sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    بدل (DeviceType) {
    حالة sycl::info::device_type::cpu:
        أرجع "cpu";
    حالة sycl::info::device_type::gpu:
        أرجع "gpu";
    حالة sycl::info::device_type::host:
        أرجع "host";
    حالة sycl::info::device_type::accelerator:
        أرجع "acc";
    افتراضي:
        أرجع "unknown";
    }
}

مضمن std::string get_device_backend_and_type(ثابت sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    أرجع device_type.str();
}

template <typename Ts> بنية matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     value_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

نطاق dpct
{
    تعريفنوع sycl::queue *queue_ptr;
    تعريفنوع sycl::event *event_ptr;
    تعريفنوع حرف *device_ptr;
    تعريفنوع uint8_t byte_t;
    تعريفنوع sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    مضمن auto exception_handler = [](sycl::exception_list exceptions)
    {
        لكل (std::exception_ptr ثابت &e : exceptions)
        {
            حاول
            {
                std::rethrow_exception(e);
            }
            امسك (sycl::exception ثابت &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    تعداد error_code
    {
        success = 0,
        default_error = 999
    };

    تعداد memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    تعداد memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    تعداد صنف library_data_t : unsigned حرف
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
    بنية DataType
    {
        using T2 = T;
    };
    template <typename T>
    بنية DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    ساكن فراغ destroy_event(event_ptr event)
    {
        delete event;
    }

    ساكن مضمن unsigned صحيح get_tid()
    {
#𐤀𐤌 defined(__linux__)
        أرجع syscall(SYS_gettid);
#وإلاإذا defined(_WIN64)
        أرجع GetCurrentThreadId();
#وإلا
#error "Only support Windows and Linux."
#endif
    }

    نطاق detail
    {
        ساكن فراغ get_version(ثابت sycl::device &dev, صحيح &major, صحيح &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            طالما (i < ver.size()) {
              𐤀𐤌 (isdigit(ver[i]))
                اكسر;
              i++;
            }
            major = std::stoi(&(ver[i]));
            طالما (i < ver.size()) {
              𐤀𐤌 (ver[i] == '.')
                اكسر;
              i++;
            }
            𐤀𐤌 (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } وإلا {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        صنف generic_error_type
        {
        عام:
            generic_error_type() = افتراضي;
            generic_error_type(T value) : value{value} {}
            operator T() ثابت { أرجع value; }

        خاص:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    صنف dim3 {
        عام:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(ثابت sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() ثابت { أرجع sycl::range<3>(z, y, x); }
    }; // namespace dim3

    مضمن dim3 operator*(ثابت dim3 &a, ثابت dim3 &b) {
    أرجع dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    صنف pitched_data
    {
    عام:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(فراغ *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        فراغ *get_data_ptr() { أرجع _data; }
        فراغ set_data_ptr(فراغ *data) { _data = data; }

        size_t get_pitch() { أرجع _pitch; }
        فراغ set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { أرجع _x; }
        فراغ set_x(size_t x) { _x = x; }

        size_t get_y() { أرجع _y; }
        فراغ set_y(size_t y) { _y = y; }

    خاص:
        فراغ *_data;
        size_t _pitch, _x, _y;
    };

    صنف device_info
    {
    عام:
        // get interface
        ثابت حرف *get_name() ثابت { أرجع _name; }
        حرف *get_name() { أرجع _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, صحيح *>,
                                   صحيح> = 0>
        auto get_max_work_item_sizes() ثابت
        {
            𐤀𐤌 constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                أرجع sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            وإلا
            {
                أرجع _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, صحيح *>,
                                   صحيح> = 0>
        auto get_max_work_item_sizes()
        {
            𐤀𐤌 constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                أرجع sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            وإلا
            {
                أرجع _max_work_item_sizes_i;
            }
        }
        منطقي get_host_unified_memory() ثابت { أرجع _host_unified_memory; }
        صحيح get_major_version() ثابت { أرجع _major; }
        صحيح get_minor_version() ثابت { أرجع _minor; }
        صحيح get_integrated() ثابت { أرجع _integrated; }
        صحيح get_max_clock_frequency() ثابت { أرجع _frequency; }
        صحيح get_max_compute_units() ثابت { أرجع _max_compute_units; }
        صحيح get_max_work_group_size() ثابت { أرجع _max_work_group_size; }
        صحيح get_max_sub_group_size() ثابت { أرجع _max_sub_group_size; }
        صحيح get_max_work_items_per_compute_unit() ثابت
        {
            أرجع _max_work_items_per_compute_unit;
        }
        صحيح get_max_register_size_per_work_group() ثابت
        {
            أرجع _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, صحيح *>,
                                   صحيح> = 0>
        auto get_max_nd_range_size() ثابت
        {
            𐤀𐤌 constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                أرجع _max_nd_range_size;
            وإلا
                أرجع _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, صحيح *>,
                                   صحيح> = 0>
        auto get_max_nd_range_size()
        {
            𐤀𐤌 constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                أرجع _max_nd_range_size;
            وإلا
                أرجع _max_nd_range_size_i;
        }
        size_t get_global_mem_size() ثابت { أرجع _global_mem_size; }
        size_t get_local_mem_size() ثابت { أرجع _local_mem_size; }
        size_t get_max_mem_alloc_size() ثابت { أرجع _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned صحيح get_memory_clock_rate() ثابت { أرجع _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned صحيح get_memory_bus_width() ثابت { أرجع _memory_bus_width; }
        uint32_t get_device_id() ثابت { أرجع _device_id; }
        std::array<unsigned حرف, 16> get_uuid() ثابت { أرجع _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned صحيح get_global_mem_cache_size() ثابت
        {
            أرجع _global_mem_cache_size;
        }

        // set interface
        فراغ set_name(ثابت حرف *name)
        {
            size_t length = strlen(name);
            𐤀𐤌 (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            وإلا
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        فراغ set_max_work_item_sizes(ثابت sycl::range<3> max_work_item_sizes)
        {
            لكل (صحيح i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] فراغ
        set_max_work_item_sizes(ثابت sycl::id<3> max_work_item_sizes)
        {
            لكل (صحيح i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        فراغ set_host_unified_memory(منطقي host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        فراغ set_major_version(صحيح major) { _major = major; }
        فراغ set_minor_version(صحيح minor) { _minor = minor; }
        فراغ set_integrated(صحيح integrated) { _integrated = integrated; }
        فراغ set_max_clock_frequency(صحيح frequency) { _frequency = frequency; }
        فراغ set_max_compute_units(صحيح max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        فراغ set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        فراغ set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        فراغ set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        فراغ set_max_work_group_size(صحيح max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        فراغ set_max_sub_group_size(صحيح max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        فراغ
        set_max_work_items_per_compute_unit(صحيح max_work_items_per_compute_unit)
        {
            _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
        }
        فراغ set_max_nd_range_size(صحيح max_nd_range_size[])
        {
            لكل (صحيح i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        فراغ set_memory_clock_rate(unsigned صحيح memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        فراغ set_memory_bus_width(unsigned صحيح memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        فراغ
        set_max_register_size_per_work_group(صحيح max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        فراغ set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        فراغ set_uuid(std::array<unsigned حرف, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        فراغ set_global_mem_cache_size(unsigned صحيح global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    خاص:
        حرف _name[256];
        صحيح _max_work_item_sizes_i[3];
        منطقي _host_unified_memory = خطأ;
        صحيح _major;
        صحيح _minor;
        صحيح _integrated = 0;
        صحيح _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned صحيح _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned صحيح _memory_bus_width = 64;
        unsigned صحيح _global_mem_cache_size;
        صحيح _max_compute_units;
        صحيح _max_work_group_size;
        صحيح _max_sub_group_size;
        صحيح _max_work_items_per_compute_unit;
        صحيح _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        صحيح _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned حرف, 16> _uuid;
    };

    ساكن صحيح get_major_version(ثابت sycl::device &dev)
    {
        صحيح major, minor;
        detail::get_version(dev, major, minor);
        أرجع major;
    }

    ساكن صحيح get_minor_version(ثابت sycl::device &dev)
    {
        صحيح major, minor;
        detail::get_version(dev, major, minor);
        أرجع minor;
    }

    ساكن فراغ get_device_info(device_info &out, ثابت sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        صحيح major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#𐤀𐤌 (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#وإلا
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

#𐤀𐤌 (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        𐤀𐤌 (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned صحيح tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            𐤀𐤌 (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        𐤀𐤌 (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        𐤀𐤌 (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        𐤀𐤌 (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#وإلاإذا defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate افتراضي value. \
        Use 64 bits as memory_bus_width افتراضي value.")
#وإلا
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate افتراضي value. \
        Use 64 bits as memory_bus_width افتراضي value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        لكل (ثابت auto &sub_group_size : sub_group_sizes)
        {
            𐤀𐤌 (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_items_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        صحيح max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    صنف device_ext : عام sycl::device {
      تعريفنوع std::mutex mutex_type;

     عام:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(ثابت sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      صحيح is_native_atomic_supported() { أرجع 0; }
      صحيح get_major_version() ثابت { أرجع dpct::get_major_version(*هذا); }

      صحيح get_minor_version() ثابت { أرجع dpct::get_minor_version(*هذا); }

      صحيح get_max_compute_units() ثابت {
        أرجع get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      صحيح get_max_clock_frequency() ثابت {
        أرجع get_device_info().get_max_clock_frequency();
      }

      صحيح get_integrated() ثابت { أرجع get_device_info().get_integrated(); }

      صحيح get_max_sub_group_size() ثابت {
        أرجع get_device_info().get_max_sub_group_size();
      }

      صحيح get_max_register_size_per_work_group() ثابت {
        أرجع get_device_info().get_max_register_size_per_work_group();
      }

      صحيح get_max_work_group_size() ثابت {
        أرجع get_device_info().get_max_work_group_size();
      }

      صحيح get_mem_base_addr_align() ثابت {
        أرجع get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() ثابت {
        أرجع get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() ثابت {
        أرجع get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      فراغ get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        ثابت حرف *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#𐤀𐤌 (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        𐤀𐤌 (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } وإلا {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#وإلا
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#𐤀𐤌 defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#وإلا
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      فراغ get_device_info(device_info &out) ثابت {
        dpct::get_device_info(out, *هذا);
      }

      device_info get_device_info() ثابت {
        device_info prop;
        dpct::get_device_info(prop, *هذا);
        أرجع prop;
      }

      فراغ reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { أرجع _q_in_order; }

      sycl::queue &out_of_order_queue() { أرجع _q_out_of_order; }

      sycl::queue &default_queue() { أرجع in_order_queue(); }

      فراغ queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        لكل (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(منطقي enable_exception_handler = خطأ) {
        أرجع create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               منطقي enable_exception_handler = خطأ) {
        أرجع create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(منطقي enable_exception_handler = خطأ) {
        std::lock_guard<mutex_type> lock(m_mutex);
        أرجع create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        منطقي enable_exception_handler = خطأ) {
        std::lock_guard<mutex_type> lock(m_mutex);
        أرجع create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          منطقي enable_exception_handler = خطأ) {
        std::lock_guard<mutex_type> lock(m_mutex);
        أرجع create_queue_impl(enable_exception_handler);
      }

      فراغ destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ثابت sycl::queue &q) -> منطقي
                                    {
                                        أرجع q == queue;
                                    }),
                    _queues.end());
      }
      فراغ set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() ثابت {
        std::lock_guard<mutex_type> lock(m_mutex);
        أرجع _saved_queue;
      }

     خاص:
      فراغ clear_queues() { _queues.clear(); }

      فراغ init_queues() {
        _q_in_order =
            create_queue_impl(true, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(true);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <صنف... Properties>
      sycl::queue create_queue_impl(منطقي enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        𐤀𐤌 (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *هذا, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        أرجع _queues.back();
      }

      template <صنف... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    منطقي enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        𐤀𐤌 (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        أرجع _queues.back();
      }

      فراغ get_version(صحيح &major, صحيح &minor) ثابت {
        detail::get_version(*هذا, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    صنف dev_mgr
    {
    عام:
        device_ext &current_device()
        {
            unsigned صحيح dev_id = current_device_id();
            check_id(dev_id);
            أرجع *_devs[dev_id];
        }
        device_ext &cpu_device() ثابت
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            𐤀𐤌 (_cpu_device == -1)
            {
                ارم std::runtime_error("no valid cpu device");
            }
            وإلا
            {
                أرجع *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned صحيح id) ثابت
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            أرجع *_devs[id];
        }
        unsigned صحيح current_device_id() ثابت
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            𐤀𐤌 (it != _thread2dev_map.end())
                أرجع it->second;
            أرجع DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        فراغ select_device(unsigned صحيح id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned صحيح device_count() { أرجع _devs.size(); }

        unsigned صحيح get_device_id(ثابت sycl::device &dev)
        {
            unsigned صحيح id = 0;
            لكل (auto &dev_item : _devs)
            {
                𐤀𐤌 (*dev_item == dev)
                {
                    أرجع id;
                }
                id++;
            }
            أرجع -1;
        }

        مضمن std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            حرف* env = getenv("ONEAPI_DEVICE_SELECTOR");
            𐤀𐤌 (env) {
                𐤀𐤌 (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                وإلا 𐤀𐤌 (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                وإلا 𐤀𐤌 (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                وإلا 𐤀𐤌 (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                وإلا {
                    ارم std::runtime_error("invalid device filter: " + std::string(env));
                }
            } وإلا {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                𐤀𐤌 (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                وإلا 𐤀𐤌 (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                وإلا 𐤀𐤌 (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            لكل (ثابت auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](ثابت sycl::device& d) {
                    أرجع d.is_gpu();
                });

                𐤀𐤌 (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    استمر;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_low_case;
                platform_name_low_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_low_case.begin(), ::tolower);

                𐤀𐤌 (platform_name_low_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    استمر;
                }

                result = platform_name;
            }

            𐤀𐤌 (result.empty())
                ارم std::runtime_error("can not find preferred GPU platform");

            أرجع result;
        }

        template <صنف DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<صحيح, DeviceSelector, ثابت sycl::device &>>
        select_device(ثابت DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned صحيح selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ساكن dev_mgr &instance()
        {
            ساكن dev_mgr d_m;
            أرجع d_m;
        }
        dev_mgr(ثابت dev_mgr &) = delete;
        dev_mgr &operator=(ثابت dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    خاص:
        mutable std::recursive_mutex m_mutex;
        ساكن منطقي compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            𐤀𐤌(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) أرجع true;
            𐤀𐤌(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) أرجع خطأ;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            أرجع prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ساكن صحيح convert_backend_index(std::string & backend) {
            𐤀𐤌 (backend == "ext_oneapi_level_zero:gpu") أرجع 0;
            𐤀𐤌 (backend == "opencl:gpu") أرجع 1;
            𐤀𐤌 (backend == "ext_oneapi_cuda:gpu") أرجع 2;
            𐤀𐤌 (backend == "ext_oneapi_hip:gpu") أرجع 3;
            𐤀𐤌 (backend == "opencl:cpu") أرجع 4;
            𐤀𐤌 (backend == "opencl:acc") أرجع 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ساكن منطقي compare_backend(std::string &backend1, std::string &backend2) {
            أرجع convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            𐤀𐤌 (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            طالما (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                𐤀𐤌 (platform_name.compare(preferred_platform_name) != 0) {
                    استمر;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                لكل (ثابت auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> keys;
            لكل(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                keys.push_back(it->first);
            }
            std::sort(keys.begin(), keys.end(), compare_backend);

            لكل (auto &key : keys) {
                std::vector<sycl::device> devs = backend_devices[key];
                std::sort(devs.begin(), devs.end(), compare_dev);
                لكل (ثابت auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            لكل (auto &dev : sycl_all_devs)
            {
                𐤀𐤌 (dev == default_device)
                {
                    استمر;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                𐤀𐤌 (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        فراغ check_id(unsigned صحيح id) ثابت
        {
            𐤀𐤌 (id >= _devs.size())
            {
                ارم std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ثابت unsigned صحيح DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned صحيح, unsigned صحيح> _thread2dev_map;
        صحيح _cpu_device = -1;
    };

    ساكن مضمن sycl::queue &get_default_queue()
    {
        أرجع dev_mgr::instance().current_device().default_queue();
    }

    نطاق detail
    {
        تعداد صنف pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ساكن pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              ثابت فراغ *ptr)
        {
            بدل (sycl::get_pointer_type(ptr, q.get_context()))
            {
            حالة sycl::usm::alloc::unknown:
                أرجع pointer_access_attribute::host_only;
            حالة sycl::usm::alloc::device:
                أرجع pointer_access_attribute::device_only;
            حالة sycl::usm::alloc::shared:
            حالة sycl::usm::alloc::host:
                أرجع pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        مضمن constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned حرف)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned حرف>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            أرجع (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        مضمن constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned حرف>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(حجم...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            أرجع get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        صنف mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#𐤀𐤌 defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#وإلاإذا defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    عدم,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#وإلا
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        عام:
            using buffer_id_t = صحيح;

            بنية allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#𐤀𐤌 defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#وإلاإذا defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#وإلا
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(ثابت mem_mgr &) = delete;
            mem_mgr &operator=(ثابت mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            فراغ *mem_alloc(size_t size)
            {
                𐤀𐤌 (!size)
                    أرجع nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                𐤀𐤌 (next_free + size > mapped_address_space + mapped_region_size)
                {
                    ارم std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                فراغ *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                أرجع result;
            }

            /// Deallocate
            فراغ mem_free(ثابت فراغ *ptr)
            {
                𐤀𐤌 (!ptr)
                    أرجع;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(ثابت فراغ *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                أرجع it->second;
            }

            /// Check if the pointer represents device pointer or not.
            منطقي is_device_ptr(ثابت فراغ *ptr) ثابت
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                أرجع (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            ساكن mem_mgr &instance()
            {
                ساكن mem_mgr m;
                أرجع m;
            }

        خاص:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            ثابت size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            ثابت size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ثابت size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(ثابت فراغ *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ثابت byte_t *>(ptr)));
                𐤀𐤌 (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ارم std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ثابت allocation &alloc = it->second;
                𐤀𐤌 (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ارم std::runtime_error("invalid virtual pointer");
                }
                أرجع it;
            }
        };

        template <صنف T, memory_region Memory, size_t Dimension>
        صنف accessor;
        template <memory_region Memory, صنف T = byte_t>
        صنف memory_traits
        {
        عام:
            ساكن constexpr sycl::access::target target =
                sycl::access::target::device;
            ساكن constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::اقرأ
                                     : sycl::access_mode::read_write;
            ساكن constexpr size_t type_size = حجم(T);
            using element_t =
                typename std::conditional<Memory == constant, ثابت T, T>::type;
            using value_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<value_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        ساكن مضمن فراغ *dpct_malloc(size_t size, sycl::queue &q)
        {
            أرجع sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        ساكن مضمن فراغ *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            أرجع dpct_malloc(pitch * y * z, q);
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
        template <typename valueT>
        ساكن مضمن sycl::event dpct_memset(sycl::queue &q, فراغ *dev_ptr,
                                              valueT value, size_t size)
        {
            أرجع q.fill(dev_ptr, value, size);
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
        template <typename valueT>
        ساكن مضمن std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, valueT value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned حرف *data_surface = (unsigned حرف *)data.get_data_ptr();
            لكل (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned حرف *data_ptr = data_surface;
                لكل (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            أرجع event_list;
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
        template <typename valueT>
        ساكن مضمن std::vector<sycl::event>
        dpct_memset(sycl::queue &q, فراغ *ptr, size_t pitch, valueT val, size_t x,
                    size_t y)
        {
            أرجع dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ساكن memcpy_direction deduce_memcpy_direction(sycl::queue &q, فراغ *to_ptr,
                                                        ثابت فراغ *from_ptr,
                                                        memcpy_direction dir)
        {
            بدل (dir)
            {
            حالة memcpy_direction::host_to_host:
            حالة memcpy_direction::host_to_device:
            حالة memcpy_direction::device_to_host:
            حالة memcpy_direction::device_to_device:
                أرجع dir;
            حالة memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ساكن ثابت memcpy_direction
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
                أرجع direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            افتراضي:
                ارم std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ساكن sycl::event
        dpct_memcpy(sycl::queue &q, فراغ *to_ptr, ثابت فراغ *from_ptr, size_t size,
                    memcpy_direction direction,
                    ثابت std::vector<sycl::event> &dep_events = {})
        {
            𐤀𐤌 (!size)
                أرجع sycl::event{};
            أرجع q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ساكن مضمن size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            أرجع slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ساكن مضمن size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            أرجع slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ساكن مضمن std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, فراغ *to_ptr, ثابت فراغ *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ثابت std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            صنف host_buffer
            {
                فراغ *_buf;
                size_t _size;
                sycl::queue &_q;
                ثابت std::vector<sycl::event> &_deps; // free operation depends

            عام:
                host_buffer(size_t size, sycl::queue &q,
                            ثابت std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                فراغ *get_ptr() ثابت { أرجع _buf; }
                size_t get_size() ثابت { أرجع _size; }
                ~host_buffer()
                {
                    𐤀𐤌 (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::حرر(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned حرف *to_surface =
                (unsigned حرف *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            ثابت unsigned حرف *from_surface =
                (ثابت unsigned حرف *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            𐤀𐤌 (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                أرجع {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            بدل (direction)
            {
            حالة host_to_host:
                لكل (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned حرف *to_ptr = to_surface;
                    ثابت unsigned حرف *from_ptr = from_surface;
                    𐤀𐤌 (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    وإلا
                    {
                        لكل (size_t y = 0; y < size.get(1); ++y)
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
                اكسر;
            حالة host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                𐤀𐤌 (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                وإلا
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
                اكسر;
            }
            حالة device_to_host:
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
                اكسر;
            }
            حالة device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<صنف dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                اكسر;
            افتراضي:
                ارم std::runtime_error("dpct_memcpy: invalid direction value");
            }
            أرجع event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ساكن مضمن std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            أرجع dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ساكن مضمن std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, فراغ *to_ptr, ثابت فراغ *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            أرجع dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        نطاق deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            صنف usm_allocator
            {
            خاص:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            عام:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    ثابت typename std::allocator_traits<Alloc>::value_type &;
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
                بنية rebind
                {
                    تعريفنوع usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(ثابت usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { أرجع &r; }
                const_pointer address(const_reference r) { أرجع &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    أرجع std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                فراغ deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ثابت
                {
                    أرجع std::allocator_traits<Alloc>::max_size(_impl);
                }
                منطقي operator==(ثابت usm_allocator &other) ثابت { أرجع _impl == other._impl; }
                منطقي operator!=(ثابت usm_allocator &other) ثابت { أرجع _impl != other._impl; }
            };

        } // namespace deprecated

        مضمن فراغ dpct_free(فراغ *ptr,
                              ثابت sycl::queue &q)
        {
            𐤀𐤌 (ptr)
            {
                sycl::حرر(ptr, q.get_context());
            }
        }

        template <typename T>
        مضمن auto get_memory(ثابت فراغ *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<فراغ *>(x));
            أرجع new_x;
        }

        template <typename T>
        مضمن typename DataType<T>::T2 get_value(ثابت T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            𐤀𐤌 (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (فراغ *)&s_h, (ثابت فراغ *)s, حجم(T), device_to_host)
                    .wait();
            وإلا
                s_h = *reinterpret_cast<ثابت Ty *>(s);
            أرجع s_h;
        }

    } // namespace detail

    template <typename T>
    مضمن auto get_value(ثابت T *s, sycl::queue &q)
    {
        أرجع detail::get_value(s, q);
    }

    نطاق detail
    {
    template <صنف Ta, صنف Tb, صنف Tc, صنف Ts>
    مضمن فراغ gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحيح m,
                          صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, صحيح lda, ثابت فراغ * b, صحيح ldb,
                          ثابت فراغ * beta, فراغ * c, صحيح ldc) {
        Ts   alpha_value = dpct::get_value(reinterpret_cast<ثابت Ts *>(alpha), q);
        Ts   beta_value  = dpct::get_value(reinterpret_cast<ثابت Ts *>(beta), q);
        auto data_a      = get_memory<ثابت Ta>(a);
        auto data_b      = get_memory<ثابت Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_value, data_a,
                                               lda, data_b, ldb, beta_value, data_c, ldc);
    }

        template <typename VecT, صنف BinaryOperation, صنف = فراغ>
        صنف vectorized_binary
        {
        عام:
            مضمن VecT operator()(VecT a, VecT b, ثابت BinaryOperation binary_op)
            {
                VecT v4;
                لكل (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = binary_op(a[i], b[i]);
                }
                أرجع v4;
            }
        };

        template <typename VecT, صنف BinaryOperation>
        صنف vectorized_binary<
            VecT, BinaryOperation,
            std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>>
        {
        عام:
            مضمن VecT operator()(VecT a, VecT b, ثابت BinaryOperation binary_op)
            {
                أرجع binary_op(a, b).template as<VecT>();
            }
        };

        template <صنف Ta, صنف Tb, صنف Tc, صنف Ts>
        مضمن فراغ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    صحيح m, صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ ** a, صحيح lda, ثابت فراغ ** b,
                                    صحيح ldb, ثابت فراغ * beta, فراغ ** c, صحيح ldc, صحيح batch_size,
                                    matrix_info_t<عائم> * matrix_info) {
            Ts alpha_value = dpct::get_value(reinterpret_cast<ثابت Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<ثابت Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->value_info[0] = alpha_value;
            matrix_info->value_info[1] = beta_value;
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
                reinterpret_cast<Ts *>(matrix_info->value_info), reinterpret_cast<ثابت Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<ثابت Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->value_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <صنف Ta, صنف Tb, صنف Tc, صنف Ts>
        مضمن فراغ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    صحيح m, صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, صحيح lda,
                                    long long صحيح stride_a, ثابت فراغ * b, صحيح ldb, long long صحيح stride_b,
                                    ثابت فراغ * beta, فراغ * c, صحيح ldc, long long صحيح stride_c, صحيح batch_size) {
            Ts alpha_value = dpct::get_value(reinterpret_cast<ثابت Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<ثابت Ts *>(beta), q);
            auto data_a = get_memory<ثابت Ta>(a);
            auto data_b = get_memory<ثابت Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_value,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_value,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, صنف BinaryOperation>
    مضمن unsigned vectorized_binary(unsigned a, unsigned b,
                                      ثابت BinaryOperation binary_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        أرجع v0;
    }

    ساكن فراغ async_dpct_memcpy(فراغ *to_ptr, ثابت فراغ *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    ساكن مضمن unsigned صحيح select_device(unsigned صحيح id)
    {
        dev_mgr::instance().select_device(id);
        أرجع id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned صحيح mask,
                               unsigned صحيح logical_sub_group_size = 32)
    {
        unsigned صحيح id = g.get_local_linear_id();
        unsigned صحيح start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned صحيح target_offset = (id % logical_sub_group_size) ^ mask;
        أرجع sycl::select_from_group(g, x,
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
      أرجع sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    مضمن auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      أرجع res;
    }

    بنية sub_sat
    {
        template <typename T>
        auto operator()(ثابت T x, ثابت T y) ثابت
        {
            أرجع sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    مضمن T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        أرجع v0;
    }

    مضمن عائم pow(ثابت عائم a, ثابت صحيح b) { أرجع sycl::pown(a, b); }
    مضمن مزدوج pow(ثابت مزدوج a, ثابت صحيح b) { أرجع sycl::pown(a, b); }
    مضمن عائم pow(ثابت عائم a, ثابت عائم b) { أرجع sycl::pow(a, b); }
    مضمن مزدوج pow(ثابت مزدوج a, ثابت مزدوج b) { أرجع sycl::pow(a, b); }
    template <typename T, typename U>
    مضمن typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(ثابت T a, ثابت U b)
    {
        أرجع sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    مضمن typename std::enable_if_t<!std::is_floating_point_v<T>, مزدوج>
    pow(ثابت T a, ثابت U b)
    {
        أرجع sycl::pow(static_cast<مزدوج>(a), static_cast<مزدوج>(b));
    }

    مضمن مزدوج min(ثابت مزدوج a, ثابت عائم b)
    {
        أرجع sycl::fmin(a, static_cast<مزدوج>(b));
    }
    مضمن مزدوج min(ثابت عائم a, ثابت مزدوج b)
    {
        أرجع sycl::fmin(static_cast<مزدوج>(a), b);
    }
    مضمن عائم min(ثابت عائم a, ثابت عائم b) { أرجع sycl::fmin(a, b); }
    مضمن مزدوج min(ثابت مزدوج a, ثابت مزدوج b) { أرجع sycl::fmin(a, b); }
    مضمن std::uint32_t min(ثابت std::uint32_t a, ثابت std::int32_t b)
    {
        أرجع sycl::min(a, static_cast<std::uint32_t>(b));
    }
    مضمن std::uint32_t min(ثابت std::int32_t a, ثابت std::uint32_t b)
    {
        أرجع sycl::min(static_cast<std::uint32_t>(a), b);
    }
    مضمن std::int32_t min(ثابت std::int32_t a, ثابت std::int32_t b)
    {
        أرجع sycl::min(a, b);
    }
    مضمن std::uint32_t min(ثابت std::uint32_t a, ثابت std::uint32_t b)
    {
        أرجع sycl::min(a, b);
    }
    مضمن std::uint64_t min(ثابت std::uint64_t a, ثابت std::int64_t b)
    {
        أرجع sycl::min(a, static_cast<std::uint64_t>(b));
    }
    مضمن std::uint64_t min(ثابت std::int64_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::min(static_cast<std::uint64_t>(a), b);
    }
    مضمن std::int64_t min(ثابت std::int64_t a, ثابت std::int64_t b)
    {
        أرجع sycl::min(a, b);
    }
    مضمن std::uint64_t min(ثابت std::uint64_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::min(a, b);
    }
    مضمن std::uint64_t min(ثابت std::uint64_t a, ثابت std::int32_t b)
    {
        أرجع sycl::min(a, static_cast<std::uint64_t>(b));
    }
    مضمن std::uint64_t min(ثابت std::int32_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::min(static_cast<std::uint64_t>(a), b);
    }
    مضمن std::uint64_t min(ثابت std::uint64_t a, ثابت std::uint32_t b)
    {
        أرجع sycl::min(a, static_cast<std::uint64_t>(b));
    }
    مضمن std::uint64_t min(ثابت std::uint32_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    مضمن مزدوج max(ثابت مزدوج a, ثابت عائم b)
    {
        أرجع sycl::fmax(a, static_cast<مزدوج>(b));
    }
    مضمن مزدوج max(ثابت عائم a, ثابت مزدوج b)
    {
        أرجع sycl::fmax(static_cast<مزدوج>(a), b);
    }
    مضمن عائم max(ثابت عائم a, ثابت عائم b) { أرجع sycl::fmax(a, b); }
    مضمن مزدوج max(ثابت مزدوج a, ثابت مزدوج b) { أرجع sycl::fmax(a, b); }
    مضمن std::uint32_t max(ثابت std::uint32_t a, ثابت std::int32_t b)
    {
        أرجع sycl::max(a, static_cast<std::uint32_t>(b));
    }
    مضمن std::uint32_t max(ثابت std::int32_t a, ثابت std::uint32_t b)
    {
        أرجع sycl::max(static_cast<std::uint32_t>(a), b);
    }
    مضمن std::int32_t max(ثابت std::int32_t a, ثابت std::int32_t b)
    {
        أرجع sycl::max(a, b);
    }
    مضمن std::uint32_t max(ثابت std::uint32_t a, ثابت std::uint32_t b)
    {
        أرجع sycl::max(a, b);
    }
    مضمن std::uint64_t max(ثابت std::uint64_t a, ثابت std::int64_t b)
    {
        أرجع sycl::max(a, static_cast<std::uint64_t>(b));
    }
    مضمن std::uint64_t max(ثابت std::int64_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::max(static_cast<std::uint64_t>(a), b);
    }
    مضمن std::int64_t max(ثابت std::int64_t a, ثابت std::int64_t b)
    {
        أرجع sycl::max(a, b);
    }
    مضمن std::uint64_t max(ثابت std::uint64_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::max(a, b);
    }
    مضمن std::uint64_t max(ثابت std::uint64_t a, ثابت std::int32_t b)
    {
        أرجع sycl::max(a, static_cast<std::uint64_t>(b));
    }
    مضمن std::uint64_t max(ثابت std::int32_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::max(static_cast<std::uint64_t>(a), b);
    }
    مضمن std::uint64_t max(ثابت std::uint64_t a, ثابت std::uint32_t b)
    {
        أرجع sycl::max(a, static_cast<std::uint64_t>(b));
    }
    مضمن std::uint64_t max(ثابت std::uint32_t a, ثابت std::uint64_t b)
    {
        أرجع sycl::max(static_cast<std::uint64_t>(a), b);
    }

    مضمن فراغ
    has_capability_or_fail(ثابت sycl::device &dev,
                           ثابت std::initializer_list<sycl::aspect> &props)
    {
        لكل (ثابت auto &it : props)
        {
            𐤀𐤌 (dev.has(it))
                استمر;
            بدل (it)
            {
            حالة sycl::aspect::fp64:
                ارم std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                اكسر;
            حالة sycl::aspect::fp16:
                ارم std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                اكسر;
            افتراضي:
#define __SYCL_ASPECT(ASPECT, ID) \
    حالة sycl::aspect::ASPECT:    \
        أرجع #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect AspectNum) -> std::string
                {
                    بدل (AspectNum)
                    {
#ضمن <sycl/info/aspects.def>
#ضمن <sycl/info/aspects_deprecated.def>
                    افتراضي:
                        أرجع "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                ارم std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            اكسر;
        }
    }

    ساكن مضمن unsigned صحيح get_current_device_id()
    {
        أرجع dev_mgr::instance().current_device_id();
    }

    ساكن مضمن device_ext &get_current_device()
    {
        أرجع dev_mgr::instance().current_device();
    }

    ساكن مضمن device_ext &get_device(unsigned صحيح id)
    {
        أرجع dev_mgr::instance().get_device(id);
    }

    ساكن مضمن sycl::queue &get_in_order_queue()
    {
        أرجع dev_mgr::instance().current_device().in_order_queue();
    }

    ساكن sycl::event
    dpct_memcpy(sycl::queue &q, فراغ *to_ptr, ثابت فراغ *from_ptr, size_t size,
                memcpy_direction direction,
                ثابت std::vector<sycl::event> &dep_events = {})
    {
        𐤀𐤌 (!size)
            أرجع sycl::event{};
        أرجع q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ساكن مضمن size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        أرجع slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ساكن مضمن size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        أرجع slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ساكن مضمن std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, فراغ *to_ptr, ثابت فراغ *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ثابت std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        صنف host_buffer
        {
            فراغ *_buf;
            size_t _size;
            sycl::queue &_q;
            ثابت std::vector<sycl::event> &_deps; // free operation depends

        عام:
            host_buffer(size_t size, sycl::queue &q,
                        ثابت std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            فراغ *get_ptr() ثابت { أرجع _buf; }
            size_t get_size() ثابت { أرجع _size; }
            ~host_buffer()
            {
                𐤀𐤌 (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::حرر(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned حرف *to_surface =
            (unsigned حرف *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        ثابت unsigned حرف *from_surface =
            (ثابت unsigned حرف *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        𐤀𐤌 (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            أرجع {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        بدل (direction)
        {
        حالة host_to_host:
            لكل (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned حرف *to_ptr = to_surface;
                ثابت unsigned حرف *from_ptr = from_surface;
                𐤀𐤌 (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                وإلا
                {
                    لكل (size_t y = 0; y < size.get(1); ++y)
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
            اكسر;
        حالة host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            𐤀𐤌 (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            وإلا
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
            اكسر;
        }
        حالة device_to_host:
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
            اكسر;
        }
        حالة device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<صنف dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        اكسر;
        افتراضي:
            ارم std::runtime_error("dpct_memcpy: invalid direction value");
        }
        أرجع event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ساكن مضمن std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        أرجع dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ساكن مضمن std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, فراغ *to_ptr, ثابت فراغ *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        أرجع dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    مضمن فراغ gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحيح m, صحيح n,
                     صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, library_data_t a_type, صحيح lda, ثابت فراغ * b,
                     library_data_t b_type, صحيح ldb, ثابت فراغ * beta, فراغ * c, library_data_t c_type, صحيح ldc,
                     library_data_t scaling_type) {
        𐤀𐤌 (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        وإلا 𐤀𐤌 (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        بدل (key)
        {
        حالة detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<عائم, عائم, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<مزدوج, مزدوج, مزدوج, مزدوج>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<عائم>, std::complex<عائم>,
                              std::complex<عائم>, std::complex<عائم>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<مزدوج>, std::complex<مزدوج>,
                              std::complex<مزدوج>, std::complex<مزدوج>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            اكسر;
        }
#ifdef __INTEL_MKL__
        حالة detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            عائم alpha_value =
                dpct::get_value(reinterpret_cast<ثابت عائم *>(alpha), q);
            عائم beta_value =
                dpct::get_value(reinterpret_cast<ثابت عائم *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            عائم alpha_float =
                dpct::get_value(reinterpret_cast<ثابت std::int32_t *>(alpha), q);
            عائم beta_float =
                dpct::get_value(reinterpret_cast<ثابت std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, عائم>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            اكسر;
        }
#endif // __INTEL_MKL__
        افتراضي:
            ارم std::runtime_error("the combination of data type is unsupported");
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
    مضمن فراغ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحيح m,
                           صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a[], library_data_t a_type, صحيح lda,
                           ثابت فراغ * b[], library_data_t b_type, صحيح ldb, ثابت فراغ * beta, فراغ * c[],
                           library_data_t c_type, صحيح ldc, صحيح batch_size, library_data_t scaling_type,
                           matrix_info_t<عائم> * matrix_info) {
        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        بدل (key)
        {
        حالة detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<عائم, عائم, عائم, عائم>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<مزدوج, مزدوج, مزدوج, مزدوج>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
#ifdef __INTEL_MKL__
        حالة detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
#endif
        حالة detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            عائم alpha_float =
                dpct::get_value(reinterpret_cast<ثابت std::int32_t *>(alpha), q);
            عائم beta_float =
                dpct::get_value(reinterpret_cast<ثابت std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, عائم>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            عائم alpha_value =
                dpct::get_value(reinterpret_cast<ثابت عائم *>(alpha), q);
            عائم beta_value =
                dpct::get_value(reinterpret_cast<ثابت عائم *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            اكسر;
        }
        افتراضي:
            ارم std::runtime_error("the combination of data type is unsupported");
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
    مضمن فراغ gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحيح m,
                           صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, library_data_t a_type, صحيح lda,
                           long long صحيح stride_a, ثابت فراغ * b, library_data_t b_type, صحيح ldb,
                           long long صحيح stride_b, ثابت فراغ * beta, فراغ * c, library_data_t c_type, صحيح ldc,
                           long long صحيح stride_c, صحيح batch_size, library_data_t scaling_type) {
        𐤀𐤌 (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        وإلا 𐤀𐤌 (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        بدل (key)
        {
        حالة detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<عائم, عائم, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<مزدوج, مزدوج, مزدوج, مزدوج>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<عائم>, std::complex<عائم>,
                                    std::complex<عائم>, std::complex<عائم>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<مزدوج>, std::complex<مزدوج>,
                                    std::complex<مزدوج>, std::complex<مزدوج>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
#ifdef __INTEL_MKL__
        حالة detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            اكسر;
        }
#endif
        حالة detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            عائم alpha_value =
                dpct::get_value(reinterpret_cast<ثابت عائم *>(alpha), q);
            عائم beta_value =
                dpct::get_value(reinterpret_cast<ثابت عائم *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            اكسر;
        }
        افتراضي:
            ارم std::runtime_error("the combination of data type is unsupported");
        }
    }

    ساكن مضمن فراغ
    async_dpct_memcpy(فراغ *to_ptr, size_t to_pitch, ثابت فراغ *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<بنية err0_tag, صحيح>;
    using err1 = detail::generic_error_type<بنية err1_tag, صحيح>;

    ساكن مضمن فراغ dpct_free(فراغ *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <صنف T, memory_region Memory, size_t Dimension> صنف accessor;
    template <صنف T, memory_region Memory> صنف accessor<T, Memory, 3> {
    عام:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, ثابت sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ثابت accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ثابت accessor_t &acc, ثابت sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) ثابت {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            أرجع accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() ثابت { أرجع _data; }

    خاص:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <صنف T, memory_region Memory> صنف accessor<T, Memory, 2> {
    عام:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, ثابت sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, ثابت accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ثابت accessor_t &acc, ثابت sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) ثابت {
            أرجع _data + _range.get(1) * index;
        }

        pointer_t get_ptr() ثابت { أرجع _data; }

    خاص:
        pointer_t _data;
        sycl::range<2> _range;
    };

    نطاق detail {
        /// Device variable with address space of shared, global or constant.
        template <صنف T, memory_region Memory, size_t Dimension> صنف device_memory {
        عام:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using value_t = typename detail::memory_traits<Memory, T>::value_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(ثابت sycl::range<Dimension> &in_range,
                        std::initializer_list<value_t> &&init_list)
                : device_memory(in_range) {
                أكد(init_list.size() <= in_range.size());
                _host_ptr = (value_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * حجم(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                ثابت typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<value_t>> &&init_list)
                : device_memory(in_range) {
                أكد(init_list.size() <= in_range[0]);
                _host_ptr = (value_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                لكل (auto sub_list : init_list) {
                    أكد(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * حجم(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(ثابت sycl::range<Dimension> &range_in)
                : _size(range_in.size() * حجم(T)), _range(range_in),
                _reference(خطأ), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <صنف... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                𐤀𐤌 (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                𐤀𐤌 (_host_ptr)
                    std::حرر(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            فراغ init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            فراغ init(sycl::queue &q) {
                𐤀𐤌 (_device_ptr)
                    أرجع;
                𐤀𐤌 (!_size)
                    أرجع;
                allocate_device(q);
                𐤀𐤌 (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            فراغ assign(value_t *src, size_t size) {
                هذا->~device_memory();
                جديد (هذا) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            value_t *get_ptr() { أرجع get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            value_t *get_ptr(sycl::queue &q) {
                init(q);
                أرجع _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { أرجع _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                أرجع _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                أرجع dpct_accessor_t((T *)_device_ptr, _range);
            }

        خاص:
            device_memory(value_t *memory_ptr, size_t size)
                : _size(size), _range(size / حجم(T)), _reference(true),
                _device_ptr(memory_ptr) {}

            فراغ allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                𐤀𐤌 (Memory == shared) {
                    _device_ptr = (value_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    أرجع;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                𐤀𐤌 (Memory == constant) {
                    _device_ptr = (value_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    أرجع;
                }
        #endif
        #endif
                _device_ptr = (value_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            منطقي _reference;
            value_t *_host_ptr;
            value_t *_device_ptr;
        };
        template <صنف T, memory_region Memory>
        صنف device_memory<T, Memory, 0> : عام device_memory<T, Memory, 1> {
        عام:
            using base = device_memory<T, Memory, 1>;
            using value_t = typename base::value_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(ثابت value_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <صنف T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <صنف T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <صنف T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    مضمن T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    أرجع atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    مضمن T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    أرجع atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    مضمن T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    بدل (memoryOrder) {
        حالة sycl::memory_order::relaxed:
            أرجع atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        حالة sycl::memory_order::acq_rel:
            أرجع atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        حالة sycl::memory_order::seq_cst:
            أرجع atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        افتراضي:
            أكد(خطأ && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    مضمن T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    مضمن unsigned صحيح byte_level_permute(
        unsigned صحيح a, unsigned صحيح b, unsigned صحيح s) {
      unsigned صحيح ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      أرجع ret;
    }

    مضمن uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, صحيح mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      𐤀𐤌 (mode >= 1 && mode <= 6) {
        أرجع byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } وإلا 𐤀𐤌 (!mode) {
        أرجع byte_level_permute(low32, high32, sel);
      }
      أرجع 0;
    }

    template <صحيح n_nondefault_params, صحيح n_default_params, typename T>
    صنف args_selector;

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
    template <صحيح n_nondefault_params, صحيح n_default_params, typename R,
              typename... Ts>
    صنف args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      خاص:
        فراغ **kernel_params;
        حرف *args_buffer;

        template <صحيح i> ساكن constexpr صحيح account_for_default_params() {
            constexpr صحيح n_total_params = حجم...(Ts);
            𐤀𐤌 constexpr (i >= n_nondefault_params) {
                أرجع n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } وإلا {
                أرجع i;
            }
        }

      عام:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <صحيح i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ساكن constexpr صحيح params_num = حجم...(Ts);

      خاص:
        template <صحيح i> ساكن constexpr صحيح get_offset() {
            𐤀𐤌 constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                أرجع 0;
            } وإلا {
                constexpr صحيح prev_off = get_offset<i - 1>();
                constexpr صحيح prev_past_end =
                    prev_off + حجم(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                𐤀𐤌 constexpr (prev_past_end % alignof(T) == 0) {
                    أرجع prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                وإلا {
                    أرجع prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        ساكن حرف *get_args_buffer(فراغ **extra) {
            𐤀𐤌 (!extra)
                أرجع nullptr;
            لكل (; (std::size_t)*extra != 0; ++extra) {
                𐤀𐤌 ((std::size_t)*extra == 1) {
                    أرجع static_cast<حرف *>(*(extra + 1));
                }
            }
            أرجع nullptr;
        }

      عام:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(فراغ **kernel_params, فراغ **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <صحيح i> arg_type<i> &get() {
            𐤀𐤌 (kernel_params) {
                أرجع *static_cast<arg_type<i> *>(kernel_params[i]);
            } وإلا {
                أرجع *reinterpret_cast<arg_type<i> *>(args_buffer +
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
    صنف kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        ساكن فراغ launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        ساكن فراغ set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned صحيح local_mem_size,
                                         queue_ptr que) {
            𐤀𐤌 (que) {
                _que = que;
            } وإلا {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        ساكن مضمن std::mutex kernel_function_ptr_map_mutex;

      عام:
        /// Variables for storing execution configuration.
        ساكن مضمن thread_local sycl::queue *_que = nullptr;
        ساكن مضمن thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ساكن مضمن thread_local unsigned صحيح _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ساكن مضمن std::map<
            ثابت فراغ *,
            std::function<فراغ(dim3, dim3, فراغ **, unsigned صحيح, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ساكن فراغ register_kernel_ptr(
            ثابت فراغ *func,
            std::function<فراغ(dim3, dim3, فراغ **, unsigned صحيح, queue_ptr)>
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
        ساكن std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, فراغ>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned صحيح local_mem_size, queue_ptr que, ArgsT... args) {
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
        ساكن فراغ launch(ثابت فراغ *func, dim3 group_range, dim3 local_range,
                           فراغ **args, unsigned صحيح local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            𐤀𐤌 (Iter == kernel_function_ptr_map.end()) {
                ارم std::runtime_error("dpct::launch() : no registered "
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
        ساكن std::enable_if_t<std::is_function_v<FuncT>, فراغ>
        launch(FuncT *func, dim3 group_range, dim3 local_range, فراغ **args,
               unsigned صحيح local_mem_size, queue_ptr que) {
            constexpr size_t p_num = args_selector<0, 0, FuncT>::params_num;
            set_execution_config(group_range, local_range, local_mem_size, que);
            args_selector<p_num, p_num, FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_num>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    template <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        صحيح remote_local_id,
        صحيح logical_sub_group_size = 32) {
      unsigned صحيح start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      أرجع sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    فراغ ldmatrix(uintptr_t addr, T* m, منطقي trans = خطأ, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      صحيح lane = sg.get_local_linear_id();

      صحيح lane_group8_row = lane / 8;
      صحيح lane_group8_col = lane % 8;

      𐤀𐤌 (!trans) {
        // calculate the source lane
        صحيح src_lane = 2 * lane_group8_row;
        𐤀𐤌 (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } وإلا {
        // calculate the source lane
        صحيح src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        صحيح index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    فراغ ldmatrix(uintptr_t addr, T* m1, T* m2, منطقي trans = خطأ) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    فراغ ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, منطقي trans = خطأ) {
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
    بنية MMAType {
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
    template <صحيح M, صحيح N, صحيح K, typename ABType, typename CDType>
    فراغ mma(
        متطاير فراغ** d_mat_frag,
        فراغ* a_mat_frag,
        فراغ* b_mat_frag,
        فراغ* c_mat_frag) {
      auto d = reinterpret_cast<متطاير CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      صحيح lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      𐤀𐤌 constexpr (M == 8 && N == 8 && K == 4) {
        𐤀𐤌 constexpr (std::is_floating_point_v<CDType>) {
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
          صحيح r_ind = (lane % 2) ? 1 : 0;
          صحيح c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          لكل (صحيح i = 0; i < 4; i++) {
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
                static_cast<عائم>(ra[r_ind]) * static_cast<عائم>(rb[c_ind]);
            *d[1] += static_cast<عائم>(ra[r_ind]) *
                     static_cast<عائم>(rb[c_ind + 1]);
            *d[2] += static_cast<عائم>(ra[r_ind + 2]) *
                     static_cast<عائم>(rb[c_ind]);
            *d[3] += static_cast<عائم>(ra[r_ind + 2]) *
                     static_cast<عائم>(rb[c_ind + 1]);

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
                static_cast<عائم>(ra[r_ind]) * static_cast<عائم>(rb[c_ind]);
            *d[5] += static_cast<عائم>(ra[r_ind]) *
                     static_cast<عائم>(rb[c_ind + 1]);
            *d[6] += static_cast<عائم>(ra[r_ind + 2]) *
                     static_cast<عائم>(rb[c_ind]);
            *d[7] += static_cast<عائم>(ra[r_ind + 2]) *
                     static_cast<عائم>(rb[c_ind + 1]);
          }
        }
      } وإلا 𐤀𐤌 constexpr (M == 8 && N == 8 && K == 16) {
        𐤀𐤌 constexpr (std::is_integral_v<ABType>) {
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
          لكل (صحيح i = 0; i < 4; i++) {
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
            لكل (صحيح j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } وإلا 𐤀𐤌 constexpr (M == 16 && N == 8 && K == 8) {
        𐤀𐤌 constexpr (std::is_floating_point_v<CDType>) {
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
          لكل (صحيح i = 0; i < 4; i++) {
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
            لكل (صحيح j = 0; j < 2; j++) {
              *d[0] += static_cast<عائم>(ra[j]) * static_cast<عائم>(rb[j]);
              *d[1] +=
                  static_cast<عائم>(ra[j]) * static_cast<عائم>(rb[j + 2]);
              *d[2] +=
                  static_cast<عائم>(ra[j + 2]) * static_cast<عائم>(rb[j]);
              *d[3] +=
                  static_cast<عائم>(ra[j + 2]) * static_cast<عائم>(rb[j + 2]);
            }
          }
        }
      } وإلا 𐤀𐤌 constexpr (M == 16 && N == 8 && K == 16) {
        𐤀𐤌 constexpr (std::is_floating_point_v<CDType>) {
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
          لكل (صحيح i = 0; i < 4; i++) {
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
            لكل (صحيح j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } وإلا 𐤀𐤌 constexpr (std::is_integral_v<ABType>) {
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
          لكل (صحيح i = 0; i < 4; i++) {
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
            لكل (صحيح i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } وإلا 𐤀𐤌 constexpr (M == 16 && N == 8 && K == 32) {
        𐤀𐤌 constexpr (std::is_integral_v<ABType>) {
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
          لكل (صحيح i = 0; i < 4; i++) {
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
            لكل (صحيح j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          لكل (صحيح i = 0; i < 4; i++) {
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
            لكل (صحيح j = 0; j < 4; j++) {
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

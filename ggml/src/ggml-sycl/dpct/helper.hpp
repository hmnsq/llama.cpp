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

#include <cstdlib>
#include <iostream>
#include <map>

#include <sycl/sycl.hpp>
#include <sycl/half_type.hpp>
#include <oneapi/mkl.hpp>

#nếu defined(__linux__)
#include <sys/mman.h>
#elif defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#không_thì
#error "Only support Windows and Linux."
#endif

#nếu defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif
#nếu defined(_WIN64)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif


#define DPCT_COMPATIBILITY_TEMP (900)

#nếu defined(_MSC_VER)
#define __dpct_align__(n) __declspec(align(n))
#define __dpct_inline__ __forceinline
#không_thì
#define __dpct_align__(n) __attribute__((aligned(n)))
#define __dpct_inline__ __inline__ __attribute__((always_inline))
#endif

#nếu defined(_MSC_VER)
#define __dpct_noinline__ __declspec(noinline)
#không_thì
#define __dpct_noinline__ __attribute__((noinline))
#endif

#define DPCT_UNUSED(x) (void)(x)

[[noreturn]] inline void _abort(const char * str) {
    std::cerr << str << std::endl;
    std::abort();
}

inline std::string get_device_type_name(const sycl::device &Device) {
    auto DeviceType = Device.get_info<sycl::info::device::device_type>();
    switch (DeviceType) {
    case sycl::info::device_type::cpu:
        trả_về "cpu";
    case sycl::info::device_type::gpu:
        trả_về "gpu";
    case sycl::info::device_type::host:
        trả_về "host";
    case sycl::info::device_type::accelerator:
        trả_về "acc";
    default:
        trả_về "unknown";
    }
}

inline std::string get_device_backend_and_type(const sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    trả_về device_type.str();
}

template <typename Ts> struct matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     value_info[2];
    std::int64_t           size_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

namespace dpct
{
    typedef sycl::queue *queue_ptr;
    typedef sycl::event *event_ptr;
    typedef char *device_ptr;
    typedef uint8_t byte_t;
    typedef sycl::buffer<byte_t> buffer_t;

    /// SYCL default exception handler
    inline auto exception_handler = [](sycl::exception_list exceptions)
    {
        cho (std::exception_ptr const &e : exceptions)
        {
            try
            {
                std::rethrow_exception(e);
            }
            catch (sycl::exception const &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                          << e.what() << std::endl
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    enum error_code
    {
        success = 0,
        default_error = 999
    };

    enum memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    enum memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    enum class library_data_t : unsigned char
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
    struct DataType
    {
        using T2 = T;
    };
    template <typename T>
    struct DataType<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    static void destroy_event(event_ptr event)
    {
        delete event;
    }

    static inline unsigned số_nguyên get_tid()
    {
#nếu defined(__linux__)
        trả_về syscall(SYS_gettid);
#elif defined(_WIN64)
        trả_về GetCurrentThreadId();
#không_thì
#error "Only support Windows and Linux."
#endif
    }

    namespace detail
    {
        static void get_version(const sycl::device &dev, số_nguyên &major, số_nguyên &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            trong_khi (i < ver.size()) {
              nếu (isdigit(ver[i]))
                break;
              i++;
            }
            major = std::stoi(&(ver[i]));
            trong_khi (i < ver.size()) {
              nếu (ver[i] == '.')
                break;
              i++;
            }
            nếu (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } không_thì {
              // c.
              minor = 0;
            }
        }

        template <typename tag, typename T>
        class generic_error_type
        {
        public:
            generic_error_type() = default;
            generic_error_type(T value) : value{value} {}
            operator T() const { trả_về value; }

        private:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    class dim3 {
        public:
        unsigned x, y, z;

        constexpr dim3(unsigned x = 1, unsigned y = 1, unsigned z = 1)
            : x(x), y(y), z(z) {}

        dim3(const sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        operator sycl::range<3>() const { trả_về sycl::range<3>(z, y, x); }
    }; // namespace dim3

    inline dim3 operator*(const dim3 &a, const dim3 &b) {
    trả_về dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    class pitched_data
    {
    public:
        pitched_data() : pitched_data(nullptr, 0, 0, 0) {}
        pitched_data(void *data, size_t pitch, size_t x, size_t y)
            : _data(data), _pitch(pitch), _x(x), _y(y) {}

        void *get_data_ptr() { trả_về _data; }
        void set_data_ptr(void *data) { _data = data; }

        size_t get_pitch() { trả_về _pitch; }
        void set_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { trả_về _x; }
        void set_x(size_t x) { _x = x; }

        size_t get_y() { trả_về _y; }
        void set_y(size_t y) { _y = y; }

    private:
        void *_data;
        size_t _pitch, _x, _y;
    };

    class device_info
    {
    public:
        // get interface
        const char *get_name() const { trả_về _name; }
        char *get_name() { trả_về _name; }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, số_nguyên *>,
                                   số_nguyên> = 0>
        auto get_max_work_item_sizes() const
        {
            nếu constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                trả_về sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            không_thì
            {
                trả_về _max_work_item_sizes_i;
            }
        }
        template <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, số_nguyên *>,
                                   số_nguyên> = 0>
        auto get_max_work_item_sizes()
        {
            nếu constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                trả_về sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            không_thì
            {
                trả_về _max_work_item_sizes_i;
            }
        }
        bool get_host_unified_memory() const { trả_về _host_unified_memory; }
        số_nguyên get_major_version() const { trả_về _major; }
        số_nguyên get_minor_version() const { trả_về _minor; }
        số_nguyên get_integrated() const { trả_về _integrated; }
        số_nguyên get_max_clock_frequency() const { trả_về _frequency; }
        số_nguyên get_max_compute_units() const { trả_về _max_compute_units; }
        số_nguyên get_max_work_group_size() const { trả_về _max_work_group_size; }
        số_nguyên get_max_sub_group_size() const { trả_về _max_sub_group_size; }
        số_nguyên get_max_work_items_per_compute_unit() const
        {
            trả_về _max_work_items_per_compute_unit;
        }
        số_nguyên get_max_register_size_per_work_group() const
        {
            trả_về _max_register_size_per_work_group;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, số_nguyên *>,
                                   số_nguyên> = 0>
        auto get_max_nd_range_size() const
        {
            nếu constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                trả_về _max_nd_range_size;
            không_thì
                trả_về _max_nd_range_size_i;
        }
        template <typename NDRangeSizeTy = size_t *,
                  std::enable_if_t<std::is_same_v<NDRangeSizeTy, size_t *> ||
                                       std::is_same_v<NDRangeSizeTy, số_nguyên *>,
                                   số_nguyên> = 0>
        auto get_max_nd_range_size()
        {
            nếu constexpr (std::is_same_v<NDRangeSizeTy, size_t *>)
                trả_về _max_nd_range_size;
            không_thì
                trả_về _max_nd_range_size_i;
        }
        size_t get_global_mem_size() const { trả_về _global_mem_size; }
        size_t get_local_mem_size() const { trả_về _local_mem_size; }
        size_t get_max_mem_alloc_size() const { trả_về _max_mem_alloc_size; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        unsigned số_nguyên get_memory_clock_rate() const { trả_về _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        unsigned số_nguyên get_memory_bus_width() const { trả_về _memory_bus_width; }
        uint32_t get_device_id() const { trả_về _device_id; }
        std::array<unsigned char, 16> get_uuid() const { trả_về _uuid; }
        /// Returns global memory cache size in bytes.
        unsigned số_nguyên get_global_mem_cache_size() const
        {
            trả_về _global_mem_cache_size;
        }

        // set interface
        void set_name(const char *name)
        {
            size_t length = strlen(name);
            nếu (length < 256)
            {
                std::memcpy(_name, name, length + 1);
            }
            không_thì
            {
                std::memcpy(_name, name, 255);
                _name[255] = '\0';
            }
        }
        void set_max_work_item_sizes(const sycl::range<3> max_work_item_sizes)
        {
            cho (số_nguyên i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] void
        set_max_work_item_sizes(const sycl::id<3> max_work_item_sizes)
        {
            cho (số_nguyên i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        void set_host_unified_memory(bool host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        void set_major_version(số_nguyên major) { _major = major; }
        void set_minor_version(số_nguyên minor) { _minor = minor; }
        void set_integrated(số_nguyên integrated) { _integrated = integrated; }
        void set_max_clock_frequency(số_nguyên frequency) { _frequency = frequency; }
        void set_max_compute_units(số_nguyên max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        void set_global_mem_size(size_t global_mem_size)
        {
            _global_mem_size = global_mem_size;
        }
        void set_local_mem_size(size_t local_mem_size)
        {
            _local_mem_size = local_mem_size;
        }
        void set_max_mem_alloc_size(size_t max_mem_alloc_size)
        {
            _max_mem_alloc_size = max_mem_alloc_size;
        }
        void set_max_work_group_size(số_nguyên max_work_group_size)
        {
            _max_work_group_size = max_work_group_size;
        }
        void set_max_sub_group_size(số_nguyên max_sub_group_size)
        {
            _max_sub_group_size = max_sub_group_size;
        }
        void
        set_max_work_items_per_compute_unit(số_nguyên max_work_items_per_compute_unit)
        {
            _max_work_items_per_compute_unit = max_work_items_per_compute_unit;
        }
        void set_max_nd_range_size(số_nguyên max_nd_range_size[])
        {
            cho (số_nguyên i = 0; i < 3; i++)
            {
                _max_nd_range_size[i] = max_nd_range_size[i];
                _max_nd_range_size_i[i] = max_nd_range_size[i];
            }
        }
        void set_memory_clock_rate(unsigned số_nguyên memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        void set_memory_bus_width(unsigned số_nguyên memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        void
        set_max_register_size_per_work_group(số_nguyên max_register_size_per_work_group)
        {
            _max_register_size_per_work_group = max_register_size_per_work_group;
        }
        void set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        void set_uuid(std::array<unsigned char, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        void set_global_mem_cache_size(unsigned số_nguyên global_mem_cache_size)
        {
            _global_mem_cache_size = global_mem_cache_size;
        }

    private:
        char _name[256];
        số_nguyên _max_work_item_sizes_i[3];
        bool _host_unified_memory = false;
        số_nguyên _major;
        số_nguyên _minor;
        số_nguyên _integrated = 0;
        số_nguyên _frequency;
        // Set estimated value 3200000 kHz as default value.
        unsigned số_nguyên _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        unsigned số_nguyên _memory_bus_width = 64;
        unsigned số_nguyên _global_mem_cache_size;
        số_nguyên _max_compute_units;
        số_nguyên _max_work_group_size;
        số_nguyên _max_sub_group_size;
        số_nguyên _max_work_items_per_compute_unit;
        số_nguyên _max_register_size_per_work_group;
        size_t _global_mem_size;
        size_t _local_mem_size;
        size_t _max_mem_alloc_size;
        size_t _max_nd_range_size[3];
        số_nguyên _max_nd_range_size_i[3];
        uint32_t _device_id;
        std::array<unsigned char, 16> _uuid;
    };

    static số_nguyên get_major_version(const sycl::device &dev)
    {
        số_nguyên major, minor;
        detail::get_version(dev, major, minor);
        trả_về major;
    }

    static số_nguyên get_minor_version(const sycl::device &dev)
    {
        số_nguyên major, minor;
        detail::get_version(dev, major, minor);
        trả_về minor;
    }

    static void get_device_info(device_info &out, const sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        số_nguyên major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#nếu (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#không_thì
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

#nếu (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        nếu (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            unsigned số_nguyên tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            nếu (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        nếu (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        nếu (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        nếu (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#elif defined(_MSC_VER) && !defined(__clang__)
#pragma message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate default value. \
        Use 64 bits as memory_bus_width default value.")
#không_thì
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate default value. \
        Use 64 bits as memory_bus_width default value."
#endif

        size_t max_sub_group_size = 1;
        std::vector<size_t> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        cho (const auto &sub_group_size : sub_group_sizes)
        {
            nếu (max_sub_group_size < sub_group_size)
                max_sub_group_size = sub_group_size;
        }

        prop.set_max_sub_group_size(max_sub_group_size);

        prop.set_max_work_items_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_size>());
        số_nguyên max_nd_range_size[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_size(max_nd_range_size);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_size_per_work_group(65536);

        prop.set_global_mem_cache_size(
            dev.get_info<sycl::info::device::global_mem_cache_size>());
        out = prop;
    }

    /// dpct device extension
    class device_ext : public sycl::device {
      typedef std::mutex mutex_type;

     public:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(const sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      số_nguyên is_native_atomic_supported() { trả_về 0; }
      số_nguyên get_major_version() const { trả_về dpct::get_major_version(*this); }

      số_nguyên get_minor_version() const { trả_về dpct::get_minor_version(*this); }

      số_nguyên get_max_compute_units() const {
        trả_về get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      số_nguyên get_max_clock_frequency() const {
        trả_về get_device_info().get_max_clock_frequency();
      }

      số_nguyên get_integrated() const { trả_về get_device_info().get_integrated(); }

      số_nguyên get_max_sub_group_size() const {
        trả_về get_device_info().get_max_sub_group_size();
      }

      số_nguyên get_max_register_size_per_work_group() const {
        trả_về get_device_info().get_max_register_size_per_work_group();
      }

      số_nguyên get_max_work_group_size() const {
        trả_về get_device_info().get_max_work_group_size();
      }

      số_nguyên get_mem_base_addr_align() const {
        trả_về get_info<sycl::info::device::mem_base_addr_align>();
      }

      size_t get_global_mem_size() const {
        trả_về get_device_info().get_global_mem_size();
      }

      size_t get_max_mem_alloc_size() const {
        trả_về get_device_info().get_max_mem_alloc_size();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      void get_memory_info(size_t &free_memory, size_t &total_memory) {
        total_memory = get_device_info().get_global_mem_size();
        const char *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#nếu (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        nếu (!has(sycl::aspect::ext_intel_free_memory)) {
          std::cerr << warning_info << std::endl;
          free_memory = total_memory;
        } không_thì {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#không_thì
        std::cerr << warning_info << std::endl;
        free_memory = total_memory;
#nếu defined(_MSC_VER) && !defined(__clang__)
#pragma message("Querying the number of bytes of free memory is not supported")
#không_thì
#warning "Querying the number of bytes of free memory is not supported"
#endif
#endif
      }

      void get_device_info(device_info &out) const {
        dpct::get_device_info(out, *this);
      }

      device_info get_device_info() const {
        device_info prop;
        dpct::get_device_info(prop, *this);
        trả_về prop;
      }

      void reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { trả_về _q_in_order; }

      sycl::queue &out_of_order_queue() { trả_về _q_out_of_order; }

      sycl::queue &default_queue() { trả_về in_order_queue(); }

      void queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        cho (auto &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(bool enable_exception_handler = false) {
        trả_về create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               bool enable_exception_handler = false) {
        trả_về create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(bool enable_exception_handler = false) {
        std::lock_guard<mutex_type> lock(m_mutex);
        trả_về create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        bool enable_exception_handler = false) {
        std::lock_guard<mutex_type> lock(m_mutex);
        trả_về create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          bool enable_exception_handler = false) {
        std::lock_guard<mutex_type> lock(m_mutex);
        trả_về create_queue_impl(enable_exception_handler);
      }

      void destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](const sycl::queue &q) -> bool
                                    {
                                        trả_về q == queue;
                                    }),
                    _queues.end());
      }
      void set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() const {
        std::lock_guard<mutex_type> lock(m_mutex);
        trả_về _saved_queue;
      }

     private:
      void clear_queues() { _queues.clear(); }

      void init_queues() {
        _q_in_order =
            create_queue_impl(true, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(true);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      template <class... Properties>
      sycl::queue create_queue_impl(bool enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        nếu (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *this, eh,
            sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#endif
                properties...)));

        trả_về _queues.back();
      }

      template <class... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    bool enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        nếu (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#ifdef DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#endif
                            properties...)));

        trả_về _queues.back();
      }

      void get_version(số_nguyên &major, số_nguyên &minor) const {
        detail::get_version(*this, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    class dev_mgr
    {
    public:
        device_ext &current_device()
        {
            unsigned số_nguyên dev_id = current_device_id();
            check_id(dev_id);
            trả_về *_devs[dev_id];
        }
        device_ext &cpu_device() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            nếu (_cpu_device == -1)
            {
                throw std::runtime_error("no valid cpu device");
            }
            không_thì
            {
                trả_về *_devs[_cpu_device];
            }
        }
        device_ext &get_device(unsigned số_nguyên id) const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            trả_về *_devs[id];
        }
        unsigned số_nguyên current_device_id() const
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            auto it = _thread2dev_map.find(get_tid());
            nếu (it != _thread2dev_map.end())
                trả_về it->second;
            trả_về DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        void select_device(unsigned số_nguyên id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        unsigned số_nguyên device_count() { trả_về _devs.size(); }

        unsigned số_nguyên get_device_id(const sycl::device &dev)
        {
            unsigned số_nguyên id = 0;
            cho (auto &dev_item : _devs)
            {
                nếu (*dev_item == dev)
                {
                    trả_về id;
                }
                id++;
            }
            trả_về -1;
        }

        inline std::string get_preferred_gpu_platform_name() {
            std::string result;

            std::string filter = "";
            char* env = getenv("ONEAPI_DEVICE_SELECTOR");
            nếu (env) {
                nếu (std::strstr(env, "level_zero")) {
                    filter = "level-zero";
                }
                không_thì nếu (std::strstr(env, "opencl")) {
                    filter = "opencl";
                }
                không_thì nếu (std::strstr(env, "cuda")) {
                    filter = "cuda";
                }
                không_thì nếu (std::strstr(env, "hip")) {
                    filter = "hip";
                }
                không_thì {
                    throw std::runtime_error("invalid device filter: " + std::string(env));
                }
            } không_thì {
                auto default_device = sycl::device(sycl::default_selector_v);
                auto default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                nếu (std::strstr(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                không_thì nếu (std::strstr(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                không_thì nếu (std::strstr(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            auto platform_list = sycl::platform::get_platforms();

            cho (const auto& platform : platform_list) {
                auto devices = platform.get_devices();
                auto gpu_dev = std::find_if(devices.begin(), devices.end(), [](const sycl::device& d) {
                    trả_về d.is_gpu();
                });

                nếu (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    continue;
                }

                auto platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_low_case;
                platform_name_low_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_low_case.begin(), ::tolower);

                nếu (platform_name_low_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    continue;
                }

                result = platform_name;
            }

            nếu (result.empty())
                throw std::runtime_error("can not find preferred GPU platform");

            trả_về result;
        }

        template <class DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<số_nguyên, DeviceSelector, const sycl::device &>>
        select_device(const DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            unsigned số_nguyên selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        static dev_mgr &instance()
        {
            static dev_mgr d_m;
            trả_về d_m;
        }
        dev_mgr(const dev_mgr &) = delete;
        dev_mgr &operator=(const dev_mgr &) = delete;
        dev_mgr(dev_mgr &&) = delete;
        dev_mgr &operator=(dev_mgr &&) = delete;

    private:
        mutable std::recursive_mutex m_mutex;
        static bool compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            nếu(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) trả_về true;
            nếu(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) trả_về false;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            trả_về prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        static số_nguyên convert_backend_index(std::string & backend) {
            nếu (backend == "ext_oneapi_level_zero:gpu") trả_về 0;
            nếu (backend == "opencl:gpu") trả_về 1;
            nếu (backend == "ext_oneapi_cuda:gpu") trả_về 2;
            nếu (backend == "ext_oneapi_hip:gpu") trả_về 3;
            nếu (backend == "opencl:cpu") trả_về 4;
            nếu (backend == "opencl:acc") trả_về 5;
            printf("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        static bool compare_backend(std::string &backend1, std::string &backend2) {
            trả_về convert_backend_index(backend1) < convert_backend_index(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            nếu (default_device.is_cpu())
                _cpu_device = 0;

            auto Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            auto preferred_platform_name = get_preferred_gpu_platform_name();

            trong_khi (!Platforms.empty()) {
                auto Platform = Platforms.back();
                Platforms.pop_back();
                auto platform_name = Platform.get_info<sycl::info::platform::name>();
                nếu (platform_name.compare(preferred_platform_name) != 0) {
                    continue;
                }
                auto devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                cho (const auto &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> keys;
            cho(auto it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                keys.push_back(it->first);
            }
            std::sort(keys.begin(), keys.end(), compare_backend);

            cho (auto &key : keys) {
                std::vector<sycl::device> devs = backend_devices[key];
                std::sort(devs.begin(), devs.end(), compare_dev);
                cho (const auto &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            cho (auto &dev : sycl_all_devs)
            {
                nếu (dev == default_device)
                {
                    continue;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                nếu (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        void check_id(unsigned số_nguyên id) const
        {
            nếu (id >= _devs.size())
            {
                throw std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        const unsigned số_nguyên DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<unsigned số_nguyên, unsigned số_nguyên> _thread2dev_map;
        số_nguyên _cpu_device = -1;
    };

    static inline sycl::queue &get_default_queue()
    {
        trả_về dev_mgr::instance().current_device().default_queue();
    }

    namespace detail
    {
        enum class pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        static pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              const void *ptr)
        {
            switch (sycl::get_pointer_type(ptr, q.get_context()))
            {
            case sycl::usm::alloc::unknown:
                trả_về pointer_access_attribute::host_only;
            case sycl::usm::alloc::device:
                trả_về pointer_access_attribute::device_only;
            case sycl::usm::alloc::shared:
            case sycl::usm::alloc::host:
                trả_về pointer_access_attribute::host_device;
            }
        }

        template <typename ArgT>
        inline constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((unsigned char)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned char>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_data_t>, "Unsupported ArgT");
            trả_về (std::uint64_t)Val;
        }

        template <typename FirstT, typename... RestT>
        inline constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_data_t::library_data_t_size <=
                              std::numeric_limits<unsigned char>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(sizeof...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_data_t>, "Unsupported FirstT");
            trả_về get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        class mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#nếu defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_size, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#elif defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    NULL,               // NULL specified as the base address parameter
                    mapped_region_size, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#không_thì
#error "Only support Windows and Linux."
#endif
                next_free = mapped_address_space;
            }

        public:
            using buffer_id_t = số_nguyên;

            struct allocation
            {
                buffer_t buffer;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#nếu defined(__linux__)
                munmap(mapped_address_space, mapped_region_size);
#elif defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#không_thì
#error "Only support Windows and Linux."
#endif
            }

            mem_mgr(const mem_mgr &) = delete;
            mem_mgr &operator=(const mem_mgr &) = delete;
            mem_mgr(mem_mgr &&) = delete;
            mem_mgr &operator=(mem_mgr &&) = delete;

            /// Allocate
            void *mem_alloc(size_t size)
            {
                nếu (!size)
                    trả_về nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                nếu (next_free + size > mapped_address_space + mapped_region_size)
                {
                    throw std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                buffer_t buf(r);
                allocation A{buf, next_free, size};
                // Map allocation to device pointer
                void *result = next_free;
                m_map.emplace(next_free + size, A);
                // Update pointer to the next free space.
                next_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                trả_về result;
            }

            /// Deallocate
            void mem_free(const void *ptr)
            {
                nếu (!ptr)
                    trả_về;
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(const void *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto it = get_map_iterator(ptr);
                trả_về it->second;
            }

            /// Check if the pointer represents device pointer or not.
            bool is_device_ptr(const void *ptr) const
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                trả_về (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_size);
            }

            /// Returns the instance of memory manager singleton.
            static mem_mgr &instance()
            {
                static mem_mgr m;
                trả_về m;
            }

        private:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *next_free;
            const size_t mapped_region_size = 128ull * 1024 * 1024 * 1024;
            const size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            const size_t extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(const void *ptr)
            {
                auto it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<const byte_t *>(ptr)));
                nếu (it == m_map.end())
                {
                    // Not a virtual pointer.
                    throw std::runtime_error("can not get buffer from non-virtual pointer");
                }
                const allocation &alloc = it->second;
                nếu (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    throw std::runtime_error("invalid virtual pointer");
                }
                trả_về it;
            }
        };

        template <class T, memory_region Memory, size_t Dimension>
        class accessor;
        template <memory_region Memory, class T = byte_t>
        class memory_traits
        {
        public:
            static constexpr sycl::access::target target =
                sycl::access::target::device;
            static constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::read
                                     : sycl::access_mode::read_write;
            static constexpr size_t type_size = sizeof(T);
            using element_t =
                typename std::conditional<Memory == constant, const T, T>::type;
            using value_t = typename std::remove_cv<T>::type;
            template <size_t Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<value_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        static inline void *dpct_malloc(size_t size, sycl::queue &q)
        {
            trả_về sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#define PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        static inline void *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            trả_về dpct_malloc(pitch * y * z, q);
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
        static inline sycl::event dpct_memset(sycl::queue &q, void *dev_ptr,
                                              valueT value, size_t size)
        {
            trả_về q.fill(dev_ptr, value, size);
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
        static inline std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_data data, valueT value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t slice = data.get_pitch() * data.get_y();
            unsigned char *data_surface = (unsigned char *)data.get_data_ptr();
            cho (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned char *data_ptr = data_surface;
                cho (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, data_ptr, value, size.get(0)));
                    data_ptr += data.get_pitch();
                }
                data_surface += slice;
            }
            trả_về event_list;
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
        static inline std::vector<sycl::event>
        dpct_memset(sycl::queue &q, void *ptr, size_t pitch, valueT val, size_t x,
                    size_t y)
        {
            trả_về dpct_memset(q, pitched_data(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        static memcpy_direction deduce_memcpy_direction(sycl::queue &q, void *to_ptr,
                                                        const void *from_ptr,
                                                        memcpy_direction dir)
        {
            switch (dir)
            {
            case memcpy_direction::host_to_host:
            case memcpy_direction::host_to_device:
            case memcpy_direction::device_to_host:
            case memcpy_direction::device_to_device:
                trả_về dir;
            case memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                static const memcpy_direction
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
                trả_về direction_table[static_cast<unsigned>(get_pointer_attribute(
                    q, to_ptr))][static_cast<unsigned>(get_pointer_attribute(q, from_ptr))];
            }
            default:
                throw std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        static sycl::event
        dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
                    memcpy_direction direction,
                    const std::vector<sycl::event> &dep_events = {})
        {
            nếu (!size)
                trả_về sycl::event{};
            trả_về q.memcpy(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                            size_t pitch)
        {
            trả_về slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        static inline size_t get_offset(sycl::id<3> id, size_t slice,
                                        size_t pitch)
        {
            trả_về slice * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        static inline std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    const std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            class host_buffer
            {
                void *_buf;
                size_t _size;
                sycl::queue &_q;
                const std::vector<sycl::event> &_deps; // free operation depends

            public:
                host_buffer(size_t size, sycl::queue &q,
                            const std::vector<sycl::event> &deps)
                    : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
                void *get_ptr() const { trả_về _buf; }
                size_t get_size() const { trả_về _size; }
                ~host_buffer()
                {
                    nếu (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::free(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t to_slice = to_range.get(1) * to_range.get(0),
                   from_slice = from_range.get(1) * from_range.get(0);
            unsigned char *to_surface =
                (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
            const unsigned char *from_surface =
                (const unsigned char *)from_ptr +
                get_offset(from_id, from_slice, from_range.get(0));

            nếu (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
            {
                trả_về {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            size_t size_slice = size.get(1) * size.get(0);
            switch (direction)
            {
            case host_to_host:
                cho (size_t z = 0; z < size.get(2); ++z)
                {
                    unsigned char *to_ptr = to_surface;
                    const unsigned char *from_ptr = from_surface;
                    nếu (to_range.get(0) == from_range.get(0) &&
                        to_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                         direction, dep_events));
                    }
                    không_thì
                    {
                        cho (size_t y = 0; y < size.get(1); ++y)
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
                break;
            case host_to_device:
            {
                host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                nếu (to_slice == size_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                không_thì
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
                break;
            }
            case device_to_host:
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
                break;
            }
            case device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<class dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.get(0))] =
                            from_surface[get_offset(id, from_slice, from_range.get(0))];
                    }); }));
                break;
            default:
                throw std::runtime_error("dpct_memcpy: invalid direction value");
            }
            trả_về event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        static inline std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                    pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            trả_về dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        static inline std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                    size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            trả_về dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        namespace deprecated
        {

            template <typename T, sycl::usm::alloc AllocKind>
            class usm_allocator
            {
            private:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            public:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    const typename std::allocator_traits<Alloc>::value_type &;
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
                struct rebind
                {
                    typedef usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(const usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { trả_về &r; }
                const_pointer address(const_reference r) { trả_về &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    trả_về std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                void deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() const
                {
                    trả_về std::allocator_traits<Alloc>::max_size(_impl);
                }
                bool operator==(const usm_allocator &other) const { trả_về _impl == other._impl; }
                bool operator!=(const usm_allocator &other) const { trả_về _impl != other._impl; }
            };

        } // namespace deprecated

        inline void dpct_free(void *ptr,
                              const sycl::queue &q)
        {
            nếu (ptr)
            {
                sycl::free(ptr, q.get_context());
            }
        }

        template <typename T>
        inline auto get_memory(const void *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<void *>(x));
            trả_về new_x;
        }

        template <typename T>
        inline typename DataType<T>::T2 get_value(const T *s, sycl::queue &q)
        {
            using Ty = typename DataType<T>::T2;
            Ty s_h;
            nếu (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (void *)&s_h, (const void *)s, sizeof(T), device_to_host)
                    .wait();
            không_thì
                s_h = *reinterpret_cast<const Ty *>(s);
            trả_về s_h;
        }

    } // namespace detail

    template <typename T>
    inline auto get_value(const T *s, sycl::queue &q)
    {
        trả_về detail::get_value(s, q);
    }

    namespace detail
    {
    template <class Ta, class Tb, class Tc, class Ts>
    inline void gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, số_nguyên m,
                          số_nguyên n, số_nguyên k, const void * alpha, const void * a, số_nguyên lda, const void * b, số_nguyên ldb,
                          const void * beta, void * c, số_nguyên ldc) {
        Ts   alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
        Ts   beta_value  = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
        auto data_a      = get_memory<const Ta>(a);
        auto data_b      = get_memory<const Tb>(b);
        auto data_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_value, data_a,
                                               lda, data_b, ldb, beta_value, data_c, ldc);
    }

        template <typename VecT, class BinaryOperation, class = void>
        class vectorized_binary
        {
        public:
            inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op)
            {
                VecT v4;
                cho (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = binary_op(a[i], b[i]);
                }
                trả_về v4;
            }
        };

        template <typename VecT, class BinaryOperation>
        class vectorized_binary<
            VecT, BinaryOperation,
            std::void_t<std::invoke_result_t<BinaryOperation, VecT, VecT>>>
        {
        public:
            inline VecT operator()(VecT a, VecT b, const BinaryOperation binary_op)
            {
                trả_về binary_op(a, b).template as<VecT>();
            }
        };

        template <class Ta, class Tb, class Tc, class Ts>
        inline void gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    số_nguyên m, số_nguyên n, số_nguyên k, const void * alpha, const void ** a, số_nguyên lda, const void ** b,
                                    số_nguyên ldb, const void * beta, void ** c, số_nguyên ldc, số_nguyên batch_size,
                                    matrix_info_t<float> * matrix_info) {
            Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);

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
                reinterpret_cast<Ts *>(matrix_info->value_info), reinterpret_cast<const Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<const Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->value_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        template <class Ta, class Tb, class Tc, class Ts>
        inline void gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    số_nguyên m, số_nguyên n, số_nguyên k, const void * alpha, const void * a, số_nguyên lda,
                                    long long số_nguyên stride_a, const void * b, số_nguyên ldb, long long số_nguyên stride_b,
                                    const void * beta, void * c, số_nguyên ldc, long long số_nguyên stride_c, số_nguyên batch_size) {
            Ts alpha_value = dpct::get_value(reinterpret_cast<const Ts *>(alpha), q);
            Ts beta_value = dpct::get_value(reinterpret_cast<const Ts *>(beta), q);
            auto data_a = get_memory<const Ta>(a);
            auto data_b = get_memory<const Tb>(b);
            auto data_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_value,
                                                         data_a, lda, stride_a, data_b, ldb, stride_b, beta_value,
                                                         data_c, ldc, stride_c, batch_size);
        }

    } // namespace detail

    template <typename VecT, class BinaryOperation>
    inline unsigned vectorized_binary(unsigned a, unsigned b,
                                      const BinaryOperation binary_op)
    {
        sycl::vec<unsigned, 1> v0{a}, v1{b};
        auto v2 = v0.as<VecT>();
        auto v3 = v1.as<VecT>();
        auto v4 =
            detail::vectorized_binary<VecT, BinaryOperation>()(v2, v3, binary_op);
        v0 = v4.template as<sycl::vec<unsigned, 1>>();
        trả_về v0;
    }

    static void async_dpct_memcpy(void *to_ptr, const void *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    static inline unsigned số_nguyên select_device(unsigned số_nguyên id)
    {
        dev_mgr::instance().select_device(id);
        trả_về id;
    }

    template <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, unsigned số_nguyên mask,
                               unsigned số_nguyên logical_sub_group_size = 32)
    {
        unsigned số_nguyên id = g.get_local_linear_id();
        unsigned số_nguyên start_index =
            id / logical_sub_group_size * logical_sub_group_size;
        unsigned số_nguyên target_offset = (id % logical_sub_group_size) ^ mask;
        trả_về sycl::select_from_group(g, x,
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
      trả_về sycl::vec<T, 1>(val)
          .template as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .template convert<T>();
    }

    template <typename T1, typename T2, typename T3>
    inline auto dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      auto va = extract_and_sign_or_zero_extend4(a);
      auto vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      trả_về res;
    }

    struct sub_sat
    {
        template <typename T>
        auto operator()(const T x, const T y) const
        {
            trả_về sycl::sub_sat(x, y);
        }
    };

    template <typename S, typename T>
    inline T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        auto v2 = v0.template as<S>();
        auto v3 = v1.template as<S>();
        auto v4 = sycl::min(v2, v3);
        v0 = v4.template as<sycl::vec<T, 1>>();
        trả_về v0;
    }

    inline float pow(const float a, const số_nguyên b) { trả_về sycl::pown(a, b); }
    inline double pow(const double a, const số_nguyên b) { trả_về sycl::pown(a, b); }
    inline float pow(const float a, const float b) { trả_về sycl::pow(a, b); }
    inline double pow(const double a, const double b) { trả_về sycl::pow(a, b); }
    template <typename T, typename U>
    inline typename std::enable_if_t<std::is_floating_point_v<T>, T>
    pow(const T a, const U b)
    {
        trả_về sycl::pow(a, static_cast<T>(b));
    }
    template <typename T, typename U>
    inline typename std::enable_if_t<!std::is_floating_point_v<T>, double>
    pow(const T a, const U b)
    {
        trả_về sycl::pow(static_cast<double>(a), static_cast<double>(b));
    }

    inline double min(const double a, const float b)
    {
        trả_về sycl::fmin(a, static_cast<double>(b));
    }
    inline double min(const float a, const double b)
    {
        trả_về sycl::fmin(static_cast<double>(a), b);
    }
    inline float min(const float a, const float b) { trả_về sycl::fmin(a, b); }
    inline double min(const double a, const double b) { trả_về sycl::fmin(a, b); }
    inline std::uint32_t min(const std::uint32_t a, const std::int32_t b)
    {
        trả_về sycl::min(a, static_cast<std::uint32_t>(b));
    }
    inline std::uint32_t min(const std::int32_t a, const std::uint32_t b)
    {
        trả_về sycl::min(static_cast<std::uint32_t>(a), b);
    }
    inline std::int32_t min(const std::int32_t a, const std::int32_t b)
    {
        trả_về sycl::min(a, b);
    }
    inline std::uint32_t min(const std::uint32_t a, const std::uint32_t b)
    {
        trả_về sycl::min(a, b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::int64_t b)
    {
        trả_về sycl::min(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t min(const std::int64_t a, const std::uint64_t b)
    {
        trả_về sycl::min(static_cast<std::uint64_t>(a), b);
    }
    inline std::int64_t min(const std::int64_t a, const std::int64_t b)
    {
        trả_về sycl::min(a, b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::uint64_t b)
    {
        trả_về sycl::min(a, b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::int32_t b)
    {
        trả_về sycl::min(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t min(const std::int32_t a, const std::uint64_t b)
    {
        trả_về sycl::min(static_cast<std::uint64_t>(a), b);
    }
    inline std::uint64_t min(const std::uint64_t a, const std::uint32_t b)
    {
        trả_về sycl::min(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t min(const std::uint32_t a, const std::uint64_t b)
    {
        trả_về sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    inline double max(const double a, const float b)
    {
        trả_về sycl::fmax(a, static_cast<double>(b));
    }
    inline double max(const float a, const double b)
    {
        trả_về sycl::fmax(static_cast<double>(a), b);
    }
    inline float max(const float a, const float b) { trả_về sycl::fmax(a, b); }
    inline double max(const double a, const double b) { trả_về sycl::fmax(a, b); }
    inline std::uint32_t max(const std::uint32_t a, const std::int32_t b)
    {
        trả_về sycl::max(a, static_cast<std::uint32_t>(b));
    }
    inline std::uint32_t max(const std::int32_t a, const std::uint32_t b)
    {
        trả_về sycl::max(static_cast<std::uint32_t>(a), b);
    }
    inline std::int32_t max(const std::int32_t a, const std::int32_t b)
    {
        trả_về sycl::max(a, b);
    }
    inline std::uint32_t max(const std::uint32_t a, const std::uint32_t b)
    {
        trả_về sycl::max(a, b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::int64_t b)
    {
        trả_về sycl::max(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t max(const std::int64_t a, const std::uint64_t b)
    {
        trả_về sycl::max(static_cast<std::uint64_t>(a), b);
    }
    inline std::int64_t max(const std::int64_t a, const std::int64_t b)
    {
        trả_về sycl::max(a, b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::uint64_t b)
    {
        trả_về sycl::max(a, b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::int32_t b)
    {
        trả_về sycl::max(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t max(const std::int32_t a, const std::uint64_t b)
    {
        trả_về sycl::max(static_cast<std::uint64_t>(a), b);
    }
    inline std::uint64_t max(const std::uint64_t a, const std::uint32_t b)
    {
        trả_về sycl::max(a, static_cast<std::uint64_t>(b));
    }
    inline std::uint64_t max(const std::uint32_t a, const std::uint64_t b)
    {
        trả_về sycl::max(static_cast<std::uint64_t>(a), b);
    }

    inline void
    has_capability_or_fail(const sycl::device &dev,
                           const std::initializer_list<sycl::aspect> &props)
    {
        cho (const auto &it : props)
        {
            nếu (dev.has(it))
                continue;
            switch (it)
            {
            case sycl::aspect::fp64:
                throw std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                break;
            case sycl::aspect::fp16:
                throw std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                break;
            default:
#define __SYCL_ASPECT(ASPECT, ID) \
    case sycl::aspect::ASPECT:    \
        trả_về #ASPECT;
#define __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#define __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                auto getAspectNameStr = [](sycl::aspect AspectNum) -> std::string
                {
                    switch (AspectNum)
                    {
#include <sycl/info/aspects.def>
#include <sycl/info/aspects_deprecated.def>
                    default:
                        trả_về "unknown aspect";
                    }
                };
#undef __SYCL_ASPECT_DEPRECATED_ALIAS
#undef __SYCL_ASPECT_DEPRECATED
#undef __SYCL_ASPECT
                throw std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            break;
        }
    }

    static inline unsigned số_nguyên get_current_device_id()
    {
        trả_về dev_mgr::instance().current_device_id();
    }

    static inline device_ext &get_current_device()
    {
        trả_về dev_mgr::instance().current_device();
    }

    static inline device_ext &get_device(unsigned số_nguyên id)
    {
        trả_về dev_mgr::instance().get_device(id);
    }

    static inline sycl::queue &get_in_order_queue()
    {
        trả_về dev_mgr::instance().current_device().in_order_queue();
    }

    static sycl::event
    dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr, size_t size,
                memcpy_direction direction,
                const std::vector<sycl::event> &dep_events = {})
    {
        nếu (!size)
            trả_về sycl::event{};
        trả_về q.memcpy(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    static inline size_t get_copy_range(sycl::range<3> size, size_t slice,
                                        size_t pitch)
    {
        trả_về slice * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    static inline size_t get_offset(sycl::id<3> id, size_t slice,
                                    size_t pitch)
    {
        trả_về slice * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    static inline std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                const std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        class host_buffer
        {
            void *_buf;
            size_t _size;
            sycl::queue &_q;
            const std::vector<sycl::event> &_deps; // free operation depends

        public:
            host_buffer(size_t size, sycl::queue &q,
                        const std::vector<sycl::event> &deps)
                : _buf(std::malloc(size)), _size(size), _q(q), _deps(deps) {}
            void *get_ptr() const { trả_về _buf; }
            size_t get_size() const { trả_về _size; }
            ~host_buffer()
            {
                nếu (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::free(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t to_slice = to_range.get(1) * to_range.get(0),
               from_slice = from_range.get(1) * from_range.get(0);
        unsigned char *to_surface =
            (unsigned char *)to_ptr + get_offset(to_id, to_slice, to_range.get(0));
        const unsigned char *from_surface =
            (const unsigned char *)from_ptr +
            get_offset(from_id, from_slice, from_range.get(0));

        nếu (to_slice == from_slice && to_slice == size.get(1) * size.get(0))
        {
            trả_về {dpct_memcpy(q, to_surface, from_surface, to_slice * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        size_t size_slice = size.get(1) * size.get(0);
        switch (direction)
        {
        case host_to_host:
            cho (size_t z = 0; z < size.get(2); ++z)
            {
                unsigned char *to_ptr = to_surface;
                const unsigned char *from_ptr = from_surface;
                nếu (to_range.get(0) == from_range.get(0) &&
                    to_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size_slice,
                                                     direction, dep_events));
                }
                không_thì
                {
                    cho (size_t y = 0; y < size.get(1); ++y)
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
            break;
        case host_to_device:
        {
            host_buffer buf(get_copy_range(size, to_slice, to_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            nếu (to_slice == size_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            không_thì
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
            break;
        }
        case device_to_host:
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
            break;
        }
        case device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<class dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.get(0))] =
                    from_surface[get_offset(id, from_slice, from_range.get(0))];
            }); }));
        break;
        default:
            throw std::runtime_error("dpct_memcpy: invalid direction value");
        }
        trả_về event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    static inline std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_data to, sycl::id<3> to_id,
                pitched_data from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        trả_về dpct_memcpy(q, to.get_data_ptr(), from.get_data_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    static inline std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, void *to_ptr, const void *from_ptr,
                size_t to_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        trả_về dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    inline void gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, số_nguyên m, số_nguyên n,
                     số_nguyên k, const void * alpha, const void * a, library_data_t a_type, số_nguyên lda, const void * b,
                     library_data_t b_type, số_nguyên ldb, const void * beta, void * c, library_data_t c_type, số_nguyên ldc,
                     library_data_t scaling_type) {
        nếu (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        không_thì nếu (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        switch (key)
        {
        case detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<float, float, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_impl<double, double, double, double>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_impl<std::complex<float>, std::complex<float>,
                              std::complex<float>, std::complex<float>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_impl<std::complex<double>, std::complex<double>,
                              std::complex<double>, std::complex<double>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            break;
        }
#ifdef __INTEL_MKL__
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            float alpha_value =
                dpct::get_value(reinterpret_cast<const float *>(alpha), q);
            float beta_value =
                dpct::get_value(reinterpret_cast<const float *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            float alpha_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
            float beta_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, float>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            break;
        }
#endif // __INTEL_MKL__
        default:
            throw std::runtime_error("the combination of data type is unsupported");
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
    inline void gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, số_nguyên m,
                           số_nguyên n, số_nguyên k, const void * alpha, const void * a[], library_data_t a_type, số_nguyên lda,
                           const void * b[], library_data_t b_type, số_nguyên ldb, const void * beta, void * c[],
                           library_data_t c_type, số_nguyên ldc, số_nguyên batch_size, library_data_t scaling_type,
                           matrix_info_t<float> * matrix_info) {
        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        switch (key)
        {
        case detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<float, float, float, float>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_size, matrix_info);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<double, double, double, double>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_size, matrix_info);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            break;
        }
#ifdef __INTEL_MKL__
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            break;
        }
#endif
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            float alpha_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(alpha), q);
            float beta_float =
                dpct::get_value(reinterpret_cast<const std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, float>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_size,
                matrix_info);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_size, matrix_info);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            float alpha_value =
                dpct::get_value(reinterpret_cast<const float *>(alpha), q);
            float beta_value =
                dpct::get_value(reinterpret_cast<const float *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_size, matrix_info);
            break;
        }
        default:
            throw std::runtime_error("the combination of data type is unsupported");
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
    inline void gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, số_nguyên m,
                           số_nguyên n, số_nguyên k, const void * alpha, const void * a, library_data_t a_type, số_nguyên lda,
                           long long số_nguyên stride_a, const void * b, library_data_t b_type, số_nguyên ldb,
                           long long số_nguyên stride_b, const void * beta, void * c, library_data_t c_type, số_nguyên ldc,
                           long long số_nguyên stride_c, số_nguyên batch_size, library_data_t scaling_type) {
        nếu (scaling_type == library_data_t::real_float &&
            c_type == library_data_t::complex_float)
        {
            scaling_type = library_data_t::complex_float;
        }
        không_thì nếu (scaling_type == library_data_t::real_double &&
                 c_type == library_data_t::complex_double)
        {
            scaling_type = library_data_t::complex_double;
        }

        std::uint64_t key =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        switch (key)
        {
        case detail::get_type_combination_id(
            library_data_t::real_float, library_data_t::real_float,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<float, float, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_double, library_data_t::real_double,
            library_data_t::real_double, library_data_t::real_double):
        {
            detail::gemm_batch_impl<double, double, double, double>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_float, library_data_t::complex_float,
            library_data_t::complex_float, library_data_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<float>, std::complex<float>,
                                    std::complex<float>, std::complex<float>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::complex_double, library_data_t::complex_double,
            library_data_t::complex_double, library_data_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<double>, std::complex<double>,
                                    std::complex<double>, std::complex<double>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_size);
            break;
        }
#ifdef __INTEL_MKL__
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_bfloat16, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_bfloat16, library_data_t::real_bfloat16,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_size);
            break;
        }
#endif
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_int32, library_data_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_int8, library_data_t::real_int8,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_float, library_data_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, float, float>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_size);
            break;
        }
        case detail::get_type_combination_id(
            library_data_t::real_half, library_data_t::real_half,
            library_data_t::real_half, library_data_t::real_float):
        {
            float alpha_value =
                dpct::get_value(reinterpret_cast<const float *>(alpha), q);
            float beta_value =
                dpct::get_value(reinterpret_cast<const float *>(beta), q);
            sycl::half alpha_half(alpha_value);
            sycl::half beta_half(beta_value);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_size);
            break;
        }
        default:
            throw std::runtime_error("the combination of data type is unsupported");
        }
    }

    static inline void
    async_dpct_memcpy(void *to_ptr, size_t to_pitch, const void *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<struct err0_tag, số_nguyên>;
    using err1 = detail::generic_error_type<struct err1_tag, số_nguyên>;

    static inline void dpct_free(void *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    template <class T, memory_region Memory, size_t Dimension> class accessor;
    template <class T, memory_region Memory> class accessor<T, Memory, 3> {
    public:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<3>;
        accessor(pointer_t data, const sycl::range<3> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(const accessor_t &acc, const sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> operator[](size_t index) const {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            trả_về accessor<T, Memory, 2>(_data + index * sub.size(), sub);
        }

        pointer_t get_ptr() const { trả_về _data; }

    private:
        pointer_t _data;
        sycl::range<3> _range;
    };
    template <class T, memory_region Memory> class accessor<T, Memory, 2> {
    public:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::template accessor_t<2>;
        accessor(pointer_t data, const sycl::range<2> &in_range)
            : _data(data), _range(in_range) {}
        template <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, const accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(const accessor_t &acc, const sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t operator[](size_t index) const {
            trả_về _data + _range.get(1) * index;
        }

        pointer_t get_ptr() const { trả_về _data; }

    private:
        pointer_t _data;
        sycl::range<2> _range;
    };

    namespace detail {
        /// Device variable with address space of shared, global or constant.
        template <class T, memory_region Memory, size_t Dimension> class device_memory {
        public:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::template accessor_t<Dimension>;
            using value_t = typename detail::memory_traits<Memory, T>::value_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(const sycl::range<Dimension> &in_range,
                        std::initializer_list<value_t> &&init_list)
                : device_memory(in_range) {
                assert(init_list.size() <= in_range.size());
                _host_ptr = (value_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                std::memcpy(_host_ptr, init_list.begin(), init_list.size() * sizeof(T));
            }

            /// Constructor of 2-D array with initializer list
            template <size_t D = Dimension>
            device_memory(
                const typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<value_t>> &&init_list)
                : device_memory(in_range) {
                assert(init_list.size() <= in_range[0]);
                _host_ptr = (value_t *)std::malloc(_size);
                std::memset(_host_ptr, 0, _size);
                auto tmp_data = _host_ptr;
                cho (auto sub_list : init_list) {
                    assert(sub_list.size() <= in_range[1]);
                    std::memcpy(tmp_data, sub_list.begin(),
                                sub_list.size() * sizeof(T));
                    tmp_data += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(const sycl::range<Dimension> &range_in)
                : _size(range_in.size() * sizeof(T)), _range(range_in),
                _reference(false), _host_ptr(nullptr), _device_ptr(nullptr) {
                static_assert(
                    (Memory == global) || (Memory == constant) || (Memory == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            template <class... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                nếu (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                nếu (_host_ptr)
                    std::free(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            void init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            void init(sycl::queue &q) {
                nếu (_device_ptr)
                    trả_về;
                nếu (!_size)
                    trả_về;
                allocate_device(q);
                nếu (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, _size,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            void assign(value_t *src, size_t size) {
                this->~device_memory();
                new (this) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            value_t *get_ptr() { trả_về get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            value_t *get_ptr(sycl::queue &q) {
                init(q);
                trả_về _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_size() { trả_về _size; }

            template <size_t D = Dimension>
            typename std::enable_if<D == 1, T>::type &operator[](size_t index) {
                init();
                trả_về _device_ptr[index];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            template <size_t D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                trả_về dpct_accessor_t((T *)_device_ptr, _range);
            }

        private:
            device_memory(value_t *memory_ptr, size_t size)
                : _size(size), _range(size / sizeof(T)), _reference(true),
                _device_ptr(memory_ptr) {}

            void allocate_device(sycl::queue &q) {
        #ifndef DPCT_USM_LEVEL_NONE
                nếu (Memory == shared) {
                    _device_ptr = (value_t *)sycl::malloc_shared(_size, q.get_device(),
                                                                q.get_context());
                    trả_về;
                }
        #ifdef SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                nếu (Memory == constant) {
                    _device_ptr = (value_t *)sycl::malloc_device(
                        _size, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    trả_về;
                }
        #endif
        #endif
                _device_ptr = (value_t *)detail::dpct_malloc(_size, q);
            }

            size_t _size;
            sycl::range<Dimension> _range;
            bool _reference;
            value_t *_host_ptr;
            value_t *_device_ptr;
        };
        template <class T, memory_region Memory>
        class device_memory<T, Memory, 0> : public device_memory<T, Memory, 1> {
        public:
            using base = device_memory<T, Memory, 1>;
            using value_t = typename base::value_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::template accessor_t<0>;

            /// Constructor with initial value.
            device_memory(const value_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    template <class T, size_t Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    template <class T, size_t Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    template <class T, size_t Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    template <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    inline T atomic_fetch_add(T *addr, T operand) {
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    trả_về atm.fetch_add(operand);
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    inline T1 atomic_fetch_add(T1 *addr, T2 operand) {
    auto atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    trả_về atm.fetch_add(operand);
    }

    template <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    inline T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    switch (memoryOrder) {
        case sycl::memory_order::relaxed:
            trả_về atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        case sycl::memory_order::acq_rel:
            trả_về atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        case sycl::memory_order::seq_cst:
            trả_về atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        default:
            assert(false && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    template <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    inline T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    inline unsigned số_nguyên byte_level_permute(
        unsigned số_nguyên a, unsigned số_nguyên b, unsigned số_nguyên s) {
      unsigned số_nguyên ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      trả_về ret;
    }

    inline uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, số_nguyên mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      nếu (mode >= 1 && mode <= 6) {
        trả_về byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } không_thì nếu (!mode) {
        trả_về byte_level_permute(low32, high32, sel);
      }
      trả_về 0;
    }

    template <số_nguyên n_nondefault_params, số_nguyên n_default_params, typename T>
    class args_selector;

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
    template <số_nguyên n_nondefault_params, số_nguyên n_default_params, typename R,
              typename... Ts>
    class args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      private:
        void **kernel_params;
        char *args_buffer;

        template <số_nguyên i> static constexpr số_nguyên account_for_default_params() {
            constexpr số_nguyên n_total_params = sizeof...(Ts);
            nếu constexpr (i >= n_nondefault_params) {
                trả_về n_total_params - n_default_params +
                       (i - n_nondefault_params);
            } không_thì {
                trả_về i;
            }
        }

      public:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        template <số_nguyên i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        static constexpr số_nguyên params_num = sizeof...(Ts);

      private:
        template <số_nguyên i> static constexpr số_nguyên get_offset() {
            nếu constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                trả_về 0;
            } không_thì {
                constexpr số_nguyên prev_off = get_offset<i - 1>();
                constexpr số_nguyên prev_past_end =
                    prev_off + sizeof(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                nếu constexpr (prev_past_end % alignof(T) == 0) {
                    trả_về prev_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                không_thì {
                    trả_về prev_past_end +
                           (alignof(T) - (prev_past_end % alignof(T)));
                }
            }
        }

        static char *get_args_buffer(void **extra) {
            nếu (!extra)
                trả_về nullptr;
            cho (; (std::size_t)*extra != 0; ++extra) {
                nếu ((std::size_t)*extra == 1) {
                    trả_về static_cast<char *>(*(extra + 1));
                }
            }
            trả_về nullptr;
        }

      public:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(void **kernel_params, void **extra)
            : kernel_params(kernel_params),
              args_buffer(get_args_buffer(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        template <số_nguyên i> arg_type<i> &get() {
            nếu (kernel_params) {
                trả_về *static_cast<arg_type<i> *>(kernel_params[i]);
            } không_thì {
                trả_về *reinterpret_cast<arg_type<i> *>(args_buffer +
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
    class kernel_launcher {
        template <typename FuncT, typename ArgSelector, std::size_t... Index>
        static void launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<Index...>) {
            func(selector.template get<Index>()...);
        }
        static void set_execution_config(dim3 group_range, dim3 local_range,
                                         unsigned số_nguyên local_mem_size,
                                         queue_ptr que) {
            nếu (que) {
                _que = que;
            } không_thì {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            _local_mem_size = local_mem_size;


        }
        static inline std::mutex kernel_function_ptr_map_mutex;

      public:
        /// Variables for storing execution configuration.
        static inline thread_local sycl::queue *_que = nullptr;
        static inline thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        static inline thread_local unsigned số_nguyên _local_mem_size = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        static inline std::map<
            const void *,
            std::function<void(dim3, dim3, void **, unsigned số_nguyên, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        static void register_kernel_ptr(
            const void *func,
            std::function<void(dim3, dim3, void **, unsigned số_nguyên, queue_ptr)>
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
        static std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, void>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               unsigned số_nguyên local_mem_size, queue_ptr que, ArgsT... args) {
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
        static void launch(const void *func, dim3 group_range, dim3 local_range,
                           void **args, unsigned số_nguyên local_mem_size,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            auto Iter = kernel_function_ptr_map.find(func);
            nếu (Iter == kernel_function_ptr_map.end()) {
                throw std::runtime_error("dpct::launch() : no registered "
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
        static std::enable_if_t<std::is_function_v<FuncT>, void>
        launch(FuncT *func, dim3 group_range, dim3 local_range, void **args,
               unsigned số_nguyên local_mem_size, queue_ptr que) {
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
        số_nguyên remote_local_id,
        số_nguyên logical_sub_group_size = 32) {
      unsigned số_nguyên start_index = g.get_local_linear_id() /
                                 logical_sub_group_size *
                                 logical_sub_group_size;
      trả_về sycl::select_from_group(
          g, x, start_index + remote_local_id % logical_sub_group_size);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    template <typename T>
    void ldmatrix(uintptr_t addr, T* m, bool trans = false, unsigned mat = 0) {
      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      số_nguyên lane = sg.get_local_linear_id();

      số_nguyên lane_group8_row = lane / 8;
      số_nguyên lane_group8_col = lane % 8;

      nếu (!trans) {
        // calculate the source lane
        số_nguyên src_lane = 2 * lane_group8_row;
        nếu (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        auto recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        auto recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } không_thì {
        // calculate the source lane
        số_nguyên src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        auto recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        auto recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        auto recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        auto recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        số_nguyên index = lane / 4;
        sycl::half val0 = recv_addr_1[index];
        sycl::half val1 = recv_addr_2[index];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    template <typename T>
    void ldmatrix(uintptr_t addr, T* m1, T* m2, bool trans = false) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    template <typename T>
    void ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, bool trans = false) {
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
    struct MMAType {
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
    template <số_nguyên M, số_nguyên N, số_nguyên K, typename ABType, typename CDType>
    void mma(
        volatile void** d_mat_frag,
        void* a_mat_frag,
        void* b_mat_frag,
        void* c_mat_frag) {
      auto d = reinterpret_cast<volatile CDType**>(d_mat_frag);
      auto a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      auto b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      auto c = reinterpret_cast<CDType*>(c_mat_frag);

      auto sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      số_nguyên lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      short row_load_offset = 4 * (lane >> 2);
      short col_load_offset = 8 * (lane % 4);

      nếu constexpr (M == 8 && N == 8 && K == 4) {
        nếu constexpr (std::is_floating_point_v<CDType>) {
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
          số_nguyên r_ind = (lane % 2) ? 1 : 0;
          số_nguyên c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          cho (số_nguyên i = 0; i < 4; i++) {
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
                static_cast<float>(ra[r_ind]) * static_cast<float>(rb[c_ind]);
            *d[1] += static_cast<float>(ra[r_ind]) *
                     static_cast<float>(rb[c_ind + 1]);
            *d[2] += static_cast<float>(ra[r_ind + 2]) *
                     static_cast<float>(rb[c_ind]);
            *d[3] += static_cast<float>(ra[r_ind + 2]) *
                     static_cast<float>(rb[c_ind + 1]);

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
                static_cast<float>(ra[r_ind]) * static_cast<float>(rb[c_ind]);
            *d[5] += static_cast<float>(ra[r_ind]) *
                     static_cast<float>(rb[c_ind + 1]);
            *d[6] += static_cast<float>(ra[r_ind + 2]) *
                     static_cast<float>(rb[c_ind]);
            *d[7] += static_cast<float>(ra[r_ind + 2]) *
                     static_cast<float>(rb[c_ind + 1]);
          }
        }
      } không_thì nếu constexpr (M == 8 && N == 8 && K == 16) {
        nếu constexpr (std::is_integral_v<ABType>) {
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
          cho (số_nguyên i = 0; i < 4; i++) {
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
            cho (số_nguyên j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } không_thì nếu constexpr (M == 16 && N == 8 && K == 8) {
        nếu constexpr (std::is_floating_point_v<CDType>) {
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
          cho (số_nguyên i = 0; i < 4; i++) {
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
            cho (số_nguyên j = 0; j < 2; j++) {
              *d[0] += static_cast<float>(ra[j]) * static_cast<float>(rb[j]);
              *d[1] +=
                  static_cast<float>(ra[j]) * static_cast<float>(rb[j + 2]);
              *d[2] +=
                  static_cast<float>(ra[j + 2]) * static_cast<float>(rb[j]);
              *d[3] +=
                  static_cast<float>(ra[j + 2]) * static_cast<float>(rb[j + 2]);
            }
          }
        }
      } không_thì nếu constexpr (M == 16 && N == 8 && K == 16) {
        nếu constexpr (std::is_floating_point_v<CDType>) {
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
          cho (số_nguyên i = 0; i < 4; i++) {
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
            cho (số_nguyên j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } không_thì nếu constexpr (std::is_integral_v<ABType>) {
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
          cho (số_nguyên i = 0; i < 4; i++) {
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
            cho (số_nguyên i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } không_thì nếu constexpr (M == 16 && N == 8 && K == 32) {
        nếu constexpr (std::is_integral_v<ABType>) {
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
          cho (số_nguyên i = 0; i < 4; i++) {
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
            cho (số_nguyên j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          cho (số_nguyên i = 0; i < 4; i++) {
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
            cho (số_nguyên j = 0; j < 4; j++) {
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

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

#अगर_अघोषित GGML_SYCL_DPCT_HELPER_HPP
#घोषणा GGML_SYCL_DPCT_HELPER_HPP

#समावेश <cstdlib>
#समावेश <iostream>
#समावेश <map>

#समावेश <sycl/sycl.hpp>
#समावेश <sycl/half_type.hpp>
#समावेश <oneapi/mkl.hpp>

#अगर defined(__linux__)
#समावेश <sys/mman.h>
#या_अगर defined(_WIN64)
#अगर_अघोषित NOMINMAX
#घोषणा NOMINMAX
#पूर्ण_अगर
#समावेश <windows.h>
#अन्यथा
#त्रुटि "Only support Windows and Linux."
#पूर्ण_अगर

#अगर defined(__linux__)
#समावेश <unistd.h>
#समावेश <sys/syscall.h>
#पूर्ण_अगर
#अगर defined(_WIN64)
#अगर_अघोषित NOMINMAX
#घोषणा NOMINMAX
#पूर्ण_अगर
#समावेश <windows.h>
#पूर्ण_अगर


#घोषणा DPCT_COMPATIBILITY_अस्थायी (900)

#अगर defined(_MSC_VER)
#घोषणा __dpct_align__(n) __declspec(align(n))
#घोषणा __dpct_inline__ __forceinline
#अन्यथा
#घोषणा __dpct_align__(n) __attribute__((aligned(n)))
#घोषणा __dpct_inline__ __inline__ __attribute__((always_inline))
#पूर्ण_अगर

#अगर defined(_MSC_VER)
#घोषणा __dpct_noinline__ __declspec(noinline)
#अन्यथा
#घोषणा __dpct_noinline__ __attribute__((noinline))
#पूर्ण_अगर

#घोषणा DPCT_UNUSED(x) (व्योम)(x)

[[noreturn]] अंतरभूत व्योम _abort(स्थिर अक्षर * str) {
    std::अ_त्रुटि << str << std::प_पूर्ण;
    std::पात();
}

अंतरभूत std::string get_device_type_name(स्थिर sycl::device &Device) {
    स्वतः DeviceType = Device.get_info<sycl::info::device::device_type>();
    चयन (DeviceType) {
    हाल sycl::info::device_type::cpu:
        वापस "cpu";
    हाल sycl::info::device_type::gpu:
        वापस "gpu";
    हाल sycl::info::device_type::host:
        वापस "host";
    हाल sycl::info::device_type::accelerator:
        वापस "acc";
    शेष:
        वापस "unknown";
    }
}

अंतरभूत std::string get_device_backend_and_type(स्थिर sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    वापस device_type.str();
}

ढाँचा <typename Ts> काष्ठा matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     मूल्य_info[2];
    std::int64_t           आकार_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

namespace dpct
{
    प्रकार sycl::queue *queue_ptr;
    प्रकार sycl::event *event_ptr;
    प्रकार अक्षर *device_ptr;
    प्रकार uint8_t byte_t;
    प्रकार sycl::बफर<byte_t> बफर_t;

    /// SYCL default exception handler
    अंतरभूत स्वतः exception_handler = [](sycl::exception_list exceptions)
    {
        क्रम (std::exception_ptr स्थिर &e : exceptions)
        {
            प्रयास
            {
                std::rethrow_exception(e);
            }
            पकड़ो (sycl::exception स्थिर &e)
            {
                std::अ_त्रुटि << "Caught asynchronous SYCL exception:" << std::प_पूर्ण
                          << e.what() << std::प_पूर्ण
                          << "Exception caught at file:" << __खाता__
                          << ", line:" << __LINE__ << std::प_पूर्ण;
            }
        }
    };

    क्रमागत error_code
    {
        success = 0,
        default_error = 999
    };

    क्रमागत memcpy_direction
    {
        host_to_host,
        host_to_device,
        device_to_host,
        device_to_device,
        automatic
    };

    क्रमागत memory_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    क्रमागत श्रेणी library_तथ्य_t : अचिन्हित अक्षर
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
        library_तथ्य_t_आकार
    };

    ढाँचा <typename T>
    काष्ठा तथ्य_Type_2
    {
        using T2 = T;
    };
    ढाँचा <typename T>
    काष्ठा तथ्य_Type_2<sycl::vec<T, 2>>
    {
        using T2 = std::complex<T>;
    };

    अटल व्योम destroy_event(event_ptr event)
    {
        मिटाओ event;
    }

    अटल अंतरभूत अचिन्हित पूर्णांक get_tid()
    {
#अगर defined(__linux__)
        वापस syscall(SYS_gettid);
#या_अगर defined(_WIN64)
        वापस GetCurrentThreadId();
#अन्यथा
#त्रुटि "Only support Windows and Linux."
#पूर्ण_अगर
    }

    namespace detail
    {
        अटल व्योम get_version(स्थिर sycl::device &dev, पूर्णांक &major, पूर्णांक &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::version>();
            std::string::size_type i = 0;
            जबतक (i < ver.size()) {
              अगर (है_अंक(ver[i]))
                अवरोध;
              i++;
            }
            major = std::stoi(&(ver[i]));
            जबतक (i < ver.size()) {
              अगर (ver[i] == '.')
                अवरोध;
              i++;
            }
            अगर (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } अन्यथा {
              // c.
              minor = 0;
            }
        }

        ढाँचा <typename tag, typename T>
        श्रेणी generic_error_type
        {
        खुला:
            generic_error_type() = शेष;
            generic_error_type(T value) : value{value} {}
            चालक T() स्थिर { वापस value; }

        निजी:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    श्रेणी dim3 {
        खुला:
        अचिन्हित x, y, z;

        constexpr dim3(अचिन्हित x = 1, अचिन्हित y = 1, अचिन्हित z = 1)
            : x(x), y(y), z(z) {}

        dim3(स्थिर sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        चालक sycl::range<3>() स्थिर { वापस sycl::range<3>(z, y, x); }
    }; // namespace dim3

    अंतरभूत dim3 चालक*(स्थिर dim3 &a, स्थिर dim3 &b) {
    वापस dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    श्रेणी pitched_तथ्य
    {
    खुला:
        pitched_तथ्य() : pitched_तथ्य(nullptr, 0, 0, 0) {}
        pitched_तथ्य(व्योम *तथ्य, माप_प्रकार pitch, माप_प्रकार x, माप_प्रकार y)
            : तथ्य_2(तथ्य), _pitch(pitch), _x(x), _y(y) {}

        व्योम *get_तथ्य_ptr() { वापस तथ्य_2; }
        व्योम set_तथ्य_ptr(व्योम *तथ्य) { तथ्य_2 = तथ्य; }

        माप_प्रकार get_pitch() { वापस _pitch; }
        व्योम set_pitch(माप_प्रकार pitch) { _pitch = pitch; }

        माप_प्रकार get_x() { वापस _x; }
        व्योम set_x(माप_प्रकार x) { _x = x; }

        माप_प्रकार get_y() { वापस _y; }
        व्योम set_y(माप_प्रकार y) { _y = y; }

    निजी:
        व्योम *तथ्य_2;
        माप_प्रकार _pitch, _x, _y;
    };

    श्रेणी device_info
    {
    खुला:
        // get interface
        स्थिर अक्षर *get_name() स्थिर { वापस _name; }
        अक्षर *get_name() { वापस _name; }
        ढाँचा <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, पूर्णांक *>,
                                   पूर्णांक> = 0>
        स्वतः get_max_work_item_sizes() स्थिर
        {
            अगर constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                वापस sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            अन्यथा
            {
                वापस _max_work_item_sizes_i;
            }
        }
        ढाँचा <typename WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, पूर्णांक *>,
                                   पूर्णांक> = 0>
        स्वतः get_max_work_item_sizes()
        {
            अगर constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                वापस sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            अन्यथा
            {
                वापस _max_work_item_sizes_i;
            }
        }
        bool get_host_unified_memory() स्थिर { वापस _host_unified_memory; }
        पूर्णांक get_major_version() स्थिर { वापस _major; }
        पूर्णांक get_minor_version() स्थिर { वापस _minor; }
        पूर्णांक get_integrated() स्थिर { वापस _integrated; }
        पूर्णांक get_max_clock_frequency() स्थिर { वापस _frequency; }
        पूर्णांक get_max_compute_units() स्थिर { वापस _max_compute_units; }
        पूर्णांक get_max_work_group_आकार() स्थिर { वापस max_work_group_आकार_2; }
        पूर्णांक get_max_sub_group_आकार() स्थिर { वापस max_sub_group_आकार; }
        पूर्णांक get_max_work_वस्तुएँ_per_compute_unit() स्थिर
        {
            वापस max_work_वस्तुएँ_per_compute_unit;
        }
        पूर्णांक get_max_register_आकार_per_work_group() स्थिर
        {
            वापस max_register_आकार_per_work_group;
        }
        ढाँचा <typename NDRange_आकार_Ty = माप_प्रकार *,
                  std::enable_if_t<std::is_same_v<NDRange_आकार_Ty, माप_प्रकार *> ||
                                       std::is_same_v<NDRange_आकार_Ty, पूर्णांक *>,
                                   पूर्णांक> = 0>
        स्वतः get_max_nd_range_आकार() स्थिर
        {
            अगर constexpr (std::is_same_v<NDRange_आकार_Ty, माप_प्रकार *>)
                वापस max_nd_range_आकार;
            अन्यथा
                वापस max_nd_range_आकार_i;
        }
        ढाँचा <typename NDRange_आकार_Ty = माप_प्रकार *,
                  std::enable_if_t<std::is_same_v<NDRange_आकार_Ty, माप_प्रकार *> ||
                                       std::is_same_v<NDRange_आकार_Ty, पूर्णांक *>,
                                   पूर्णांक> = 0>
        स्वतः get_max_nd_range_आकार()
        {
            अगर constexpr (std::is_same_v<NDRange_आकार_Ty, माप_प्रकार *>)
                वापस max_nd_range_आकार;
            अन्यथा
                वापस max_nd_range_आकार_i;
        }
        माप_प्रकार get_global_mem_आकार() स्थिर { वापस global_mem_आकार_2; }
        माप_प्रकार get_local_mem_आकार() स्थिर { वापस local_mem_आकार; }
        माप_प्रकार get_max_mem_alloc_आकार() स्थिर { वापस max_mem_alloc_आकार; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        अचिन्हित पूर्णांक get_memory_clock_rate() स्थिर { वापस _memory_clock_rate; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        अचिन्हित पूर्णांक get_memory_bus_width() स्थिर { वापस _memory_bus_width; }
        uint32_t get_device_id() स्थिर { वापस _device_id; }
        std::array<अचिन्हित अक्षर, 16> get_uuid() स्थिर { वापस _uuid; }
        /// Returns global memory cache size in bytes.
        अचिन्हित पूर्णांक get_global_mem_cache_आकार() स्थिर
        {
            वापस global_mem_cache_आकार;
        }

        // set interface
        व्योम set_name(स्थिर अक्षर *name)
        {
            माप_प्रकार लंबाई = म_माप(name);
            अगर (लंबाई < 256)
            {
                std::स_नकल(_name, name, लंबाई + 1);
            }
            अन्यथा
            {
                std::स_नकल(_name, name, 255);
                _name[255] = '\0';
            }
        }
        व्योम set_max_work_item_sizes(स्थिर sycl::range<3> max_work_item_sizes)
        {
            क्रम (पूर्णांक i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] व्योम
        set_max_work_item_sizes(स्थिर sycl::id<3> max_work_item_sizes)
        {
            क्रम (पूर्णांक i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        व्योम set_host_unified_memory(bool host_unified_memory)
        {
            _host_unified_memory = host_unified_memory;
        }
        व्योम set_major_version(पूर्णांक major) { _major = major; }
        व्योम set_minor_version(पूर्णांक minor) { _minor = minor; }
        व्योम set_integrated(पूर्णांक integrated) { _integrated = integrated; }
        व्योम set_max_clock_frequency(पूर्णांक frequency) { _frequency = frequency; }
        व्योम set_max_compute_units(पूर्णांक max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        व्योम set_global_mem_आकार(माप_प्रकार global_mem_आकार)
        {
            global_mem_आकार_2 = global_mem_आकार;
        }
        व्योम set_local_mem_आकार(माप_प्रकार local_mem_आकार_2)
        {
            local_mem_आकार = local_mem_आकार_2;
        }
        व्योम set_max_mem_alloc_आकार(माप_प्रकार max_mem_alloc_आकार_2)
        {
            max_mem_alloc_आकार = max_mem_alloc_आकार_2;
        }
        व्योम set_max_work_group_आकार(पूर्णांक max_work_group_आकार)
        {
            max_work_group_आकार_2 = max_work_group_आकार;
        }
        व्योम set_max_sub_group_आकार(पूर्णांक max_sub_group_आकार_2)
        {
            max_sub_group_आकार = max_sub_group_आकार_2;
        }
        व्योम
        set_max_work_वस्तुएँ_per_compute_unit(पूर्णांक max_work_वस्तुएँ_per_compute_unit_2)
        {
            max_work_वस्तुएँ_per_compute_unit = max_work_वस्तुएँ_per_compute_unit_2;
        }
        व्योम set_max_nd_range_आकार(पूर्णांक max_nd_range_आकार_2[])
        {
            क्रम (पूर्णांक i = 0; i < 3; i++)
            {
                max_nd_range_आकार[i] = max_nd_range_आकार_2[i];
                max_nd_range_आकार_i[i] = max_nd_range_आकार_2[i];
            }
        }
        व्योम set_memory_clock_rate(अचिन्हित पूर्णांक memory_clock_rate)
        {
            _memory_clock_rate = memory_clock_rate;
        }
        व्योम set_memory_bus_width(अचिन्हित पूर्णांक memory_bus_width)
        {
            _memory_bus_width = memory_bus_width;
        }
        व्योम
        set_max_register_आकार_per_work_group(पूर्णांक max_register_आकार_per_work_group_2)
        {
            max_register_आकार_per_work_group = max_register_आकार_per_work_group_2;
        }
        व्योम set_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        व्योम set_uuid(std::array<अचिन्हित अक्षर, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        व्योम set_global_mem_cache_आकार(अचिन्हित पूर्णांक global_mem_cache_आकार_2)
        {
            global_mem_cache_आकार = global_mem_cache_आकार_2;
        }

    निजी:
        अक्षर _name[256];
        पूर्णांक _max_work_item_sizes_i[3];
        bool _host_unified_memory = false;
        पूर्णांक _major;
        पूर्णांक _minor;
        पूर्णांक _integrated = 0;
        पूर्णांक _frequency;
        // Set estimated value 3200000 kHz as default value.
        अचिन्हित पूर्णांक _memory_clock_rate = 3200000;
        // Set estimated value 64 bits as default value.
        अचिन्हित पूर्णांक _memory_bus_width = 64;
        अचिन्हित पूर्णांक global_mem_cache_आकार;
        पूर्णांक _max_compute_units;
        पूर्णांक max_work_group_आकार_2;
        पूर्णांक max_sub_group_आकार;
        पूर्णांक max_work_वस्तुएँ_per_compute_unit;
        पूर्णांक max_register_आकार_per_work_group;
        माप_प्रकार global_mem_आकार_2;
        माप_प्रकार local_mem_आकार;
        माप_प्रकार max_mem_alloc_आकार;
        माप_प्रकार max_nd_range_आकार[3];
        पूर्णांक max_nd_range_आकार_i[3];
        uint32_t _device_id;
        std::array<अचिन्हित अक्षर, 16> _uuid;
    };

    अटल पूर्णांक get_major_version(स्थिर sycl::device &dev)
    {
        पूर्णांक major, minor;
        detail::get_version(dev, major, minor);
        वापस major;
    }

    अटल पूर्णांक get_minor_version(स्थिर sycl::device &dev)
    {
        पूर्णांक major, minor;
        detail::get_version(dev, major, minor);
        वापस minor;
    }

    अटल व्योम get_device_info(device_info &out, स्थिर sycl::device &dev)
    {
        device_info prop;
        prop.set_name(dev.get_info<sycl::info::device::name>().c_str());

        पूर्णांक major, minor;
        detail::get_version(dev, major, minor);
        prop.set_major_version(major);
        prop.set_minor_version(minor);

        prop.set_max_work_item_sizes(
#अगर (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#अन्यथा
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_item_sizes<3>>());
#पूर्ण_अगर
        prop.set_host_unified_memory(dev.has(sycl::aspect::usm_host_allocations));

        prop.set_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.set_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.set_max_work_group_आकार(
            dev.get_info<sycl::info::device::max_work_group_आकार>());
        prop.set_global_mem_आकार(dev.get_info<sycl::info::device::global_mem_आकार>());
        prop.set_local_mem_आकार(dev.get_info<sycl::info::device::local_mem_आकार_2>());
        prop.set_max_mem_alloc_आकार(dev.get_info<sycl::info::device::max_mem_alloc_आकार_2>());

#अगर (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        अगर (dev.has(sycl::aspect::ext_intel_memory_clock_rate))
        {
            अचिन्हित पूर्णांक tmp =
                dev.get_info<sycl::ext::intel::info::device::memory_clock_rate>();
            अगर (tmp != 0)
                prop.set_memory_clock_rate(1000 * tmp);
        }
        अगर (dev.has(sycl::aspect::ext_intel_memory_bus_width))
        {
            prop.set_memory_bus_width(
                dev.get_info<sycl::ext::intel::info::device::memory_bus_width>());
        }
        अगर (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.set_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        अगर (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.set_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#या_अगर defined(_MSC_VER) && !defined(__clang__)
#आशय message("get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate शेष value. \
        Use 64 bits as memory_bus_width शेष value.")
#अन्यथा
#warning "get_device_info: querying memory_clock_rate and \
        memory_bus_width are not supported by the compiler used. \
        Use 3200000 kHz as memory_clock_rate शेष value. \
        Use 64 bits as memory_bus_width शेष value."
#पूर्ण_अगर

        माप_प्रकार max_sub_group_आकार_2 = 1;
        std::vector<माप_प्रकार> sub_group_sizes =
            dev.get_info<sycl::info::device::sub_group_sizes>();

        क्रम (स्थिर स्वतः &sub_group_आकार : sub_group_sizes)
        {
            अगर (max_sub_group_आकार_2 < sub_group_आकार)
                max_sub_group_आकार_2 = sub_group_आकार;
        }

        prop.set_max_sub_group_आकार(max_sub_group_आकार_2);

        prop.set_max_work_वस्तुएँ_per_compute_unit(
            dev.get_info<sycl::info::device::max_work_group_आकार>());
        पूर्णांक max_nd_range_आकार_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.set_max_nd_range_आकार(max_nd_range_आकार_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.set_max_register_आकार_per_work_group(65536);

        prop.set_global_mem_cache_आकार(
            dev.get_info<sycl::info::device::global_mem_cache_आकार_2>());
        out = prop;
    }

    /// dpct device extension
    श्रेणी device_ext : खुला sycl::device {
      प्रकार std::mutex mutex_type;

     खुला:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
      }
      device_ext(स्थिर sycl::device &base) : sycl::device(base) {
        std::lock_guard<mutex_type> lock(m_mutex);
        init_queues();
      }

      पूर्णांक is_native_atomic_supported() { वापस 0; }
      पूर्णांक get_major_version() स्थिर { वापस dpct::get_major_version(*यह); }

      पूर्णांक get_minor_version() स्थिर { वापस dpct::get_minor_version(*यह); }

      पूर्णांक get_max_compute_units() स्थिर {
        वापस get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      पूर्णांक get_max_clock_frequency() स्थिर {
        वापस get_device_info().get_max_clock_frequency();
      }

      पूर्णांक get_integrated() स्थिर { वापस get_device_info().get_integrated(); }

      पूर्णांक get_max_sub_group_आकार() स्थिर {
        वापस get_device_info().get_max_sub_group_आकार();
      }

      पूर्णांक get_max_register_आकार_per_work_group() स्थिर {
        वापस get_device_info().get_max_register_आकार_per_work_group();
      }

      पूर्णांक get_max_work_group_आकार() स्थिर {
        वापस get_device_info().get_max_work_group_आकार();
      }

      पूर्णांक get_mem_base_addr_align() स्थिर {
        वापस get_info<sycl::info::device::mem_base_addr_align>();
      }

      माप_प्रकार get_global_mem_आकार() स्थिर {
        वापस get_device_info().get_global_mem_आकार();
      }

      माप_प्रकार get_max_mem_alloc_आकार() स्थिर {
        वापस get_device_info().get_max_mem_alloc_आकार();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      व्योम get_memory_info(माप_प्रकार &free_memory, माप_प्रकार &योग_memory) {
        योग_memory = get_device_info().get_global_mem_आकार();
        स्थिर अक्षर *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#अगर (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        अगर (!has(sycl::aspect::ext_intel_free_memory)) {
          std::अ_त्रुटि << warning_info << std::प_पूर्ण;
          free_memory = योग_memory;
        } अन्यथा {
          free_memory = get_info<sycl::ext::intel::info::device::free_memory>();
        }
#अन्यथा
        std::अ_त्रुटि << warning_info << std::प_पूर्ण;
        free_memory = योग_memory;
#अगर defined(_MSC_VER) && !defined(__clang__)
#आशय message("Querying the number of bytes of free memory is not supported")
#अन्यथा
#warning "Querying the number of bytes of free memory is not supported"
#पूर्ण_अगर
#पूर्ण_अगर
      }

      व्योम get_device_info(device_info &out) स्थिर {
        dpct::get_device_info(out, *यह);
      }

      device_info get_device_info() स्थिर {
        device_info prop;
        dpct::get_device_info(prop, *यह);
        वापस prop;
      }

      व्योम reset() {
        std::lock_guard<mutex_type> lock(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &in_order_queue() { वापस _q_in_order; }

      sycl::queue &out_of_order_queue() { वापस _q_out_of_order; }

      sycl::queue &default_queue() { वापस in_order_queue(); }

      व्योम queues_wait_and_throw() {
        std::unique_lock<mutex_type> lock(m_mutex);
        lock.unlock();
        क्रम (स्वतः &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        lock.lock();
      }

      sycl::queue create_queue(bool enable_exception_handler = false) {
        वापस create_in_order_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               bool enable_exception_handler = false) {
        वापस create_in_order_queue(device, enable_exception_handler);
      }

      sycl::queue create_in_order_queue(bool enable_exception_handler = false) {
        std::lock_guard<mutex_type> lock(m_mutex);
        वापस create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_in_order_queue(sycl::device device,
                                        bool enable_exception_handler = false) {
        std::lock_guard<mutex_type> lock(m_mutex);
        वापस create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::in_order());
      }

      sycl::queue create_out_of_order_queue(
          bool enable_exception_handler = false) {
        std::lock_guard<mutex_type> lock(m_mutex);
        वापस create_queue_impl(enable_exception_handler);
      }

      व्योम destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](स्थिर sycl::queue &q) -> bool
                                    {
                                        वापस q == queue;
                                    }),
                    _queues.end());
      }
      व्योम set_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> lock(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() स्थिर {
        std::lock_guard<mutex_type> lock(m_mutex);
        वापस _saved_queue;
      }

     निजी:
      व्योम clear_queues() { _queues.सफा(); }

      व्योम init_queues() {
        _q_in_order =
            create_queue_impl(true, sycl::property::queue::in_order());
        _q_out_of_order = create_queue_impl(true);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      ढाँचा <श्रेणी... Properties>
      sycl::queue create_queue_impl(bool enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        अगर (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *यह, eh,
            sycl::property_list(
#अगर_घोषित DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#पूर्ण_अगर
                properties...)));

        वापस _queues.back();
      }

      ढाँचा <श्रेणी... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    bool enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        अगर (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#अगर_घोषित DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#पूर्ण_अगर
                            properties...)));

        वापस _queues.back();
      }

      व्योम get_version(पूर्णांक &major, पूर्णांक &minor) स्थिर {
        detail::get_version(*यह, major, minor);
      }
      sycl::queue _q_in_order, _q_out_of_order;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      mutable mutex_type m_mutex;
    };


    /// device manager
    श्रेणी dev_mgr
    {
    खुला:
        device_ext &current_device()
        {
            अचिन्हित पूर्णांक dev_id = current_device_id();
            check_id(dev_id);
            वापस *_devs[dev_id];
        }
        device_ext &cpu_device() स्थिर
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            अगर (_cpu_device == -1)
            {
                फेंको std::runtime_error("no valid cpu device");
            }
            अन्यथा
            {
                वापस *_devs[_cpu_device];
            }
        }
        device_ext &get_device(अचिन्हित पूर्णांक id) स्थिर
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            वापस *_devs[id];
        }
        अचिन्हित पूर्णांक current_device_id() स्थिर
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            स्वतः it = _thread2dev_map.find(get_tid());
            अगर (it != _thread2dev_map.end())
                वापस it->second;
            वापस DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        व्योम select_device(अचिन्हित पूर्णांक id)
        {
            std::lock_guard<std::recursive_mutex> lock(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        अचिन्हित पूर्णांक device_गिनती() { वापस _devs.size(); }

        अचिन्हित पूर्णांक get_device_id(स्थिर sycl::device &dev)
        {
            अचिन्हित पूर्णांक id = 0;
            क्रम (स्वतः &dev_item : _devs)
            {
                अगर (*dev_item == dev)
                {
                    वापस id;
                }
                id++;
            }
            वापस -1;
        }

        अंतरभूत std::string get_preferred_gpu_platform_name() {
            std::string परिणाम;

            std::string filter = "";
            अक्षर* env = दो_पर्या("ONEAPI_DEVICE_SELECTOR");
            अगर (env) {
                अगर (std::म_माला(env, "level_zero")) {
                    filter = "level-zero";
                }
                अन्यथा अगर (std::म_माला(env, "opencl")) {
                    filter = "opencl";
                }
                अन्यथा अगर (std::म_माला(env, "cuda")) {
                    filter = "cuda";
                }
                अन्यथा अगर (std::म_माला(env, "hip")) {
                    filter = "hip";
                }
                अन्यथा {
                    फेंको std::runtime_error("invalid device filter: " + std::string(env));
                }
            } अन्यथा {
                स्वतः default_device = sycl::device(sycl::default_selector_v);
                स्वतः default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                अगर (std::म_माला(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                अन्यथा अगर (std::म_माला(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                अन्यथा अगर (std::म_माला(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            स्वतः platform_list = sycl::platform::get_platforms();

            क्रम (स्थिर स्वतः& platform : platform_list) {
                स्वतः devices = platform.get_devices();
                स्वतः gpu_dev = std::find_if(devices.begin(), devices.end(), [](स्थिर sycl::device& d) {
                    वापस d.is_gpu();
                });

                अगर (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    जारी;
                }

                स्वतः platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_निचला_case;
                platform_name_निचला_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_निचला_case.begin(), ::छोटे);

                अगर (platform_name_निचला_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    जारी;
                }

                परिणाम = platform_name;
            }

            अगर (परिणाम.empty())
                फेंको std::runtime_error("can not find preferred GPU platform");

            वापस परिणाम;
        }

        ढाँचा <श्रेणी DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<पूर्णांक, DeviceSelector, स्थिर sycl::device &>>
        select_device(स्थिर DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            अचिन्हित पूर्णांक selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        अटल dev_mgr &instance()
        {
            अटल dev_mgr d_m;
            वापस d_m;
        }
        dev_mgr(स्थिर dev_mgr &) = मिटाओ;
        dev_mgr &चालक=(स्थिर dev_mgr &) = मिटाओ;
        dev_mgr(dev_mgr &&) = मिटाओ;
        dev_mgr &चालक=(dev_mgr &&) = मिटाओ;

    निजी:
        mutable std::recursive_mutex m_mutex;
        अटल bool compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            अगर(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) वापस true;
            अगर(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) वापस false;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            वापस prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        अटल पूर्णांक convert_backend_सूचकांक(std::string & backend) {
            अगर (backend == "ext_oneapi_level_zero:gpu") वापस 0;
            अगर (backend == "opencl:gpu") वापस 1;
            अगर (backend == "ext_oneapi_cuda:gpu") वापस 2;
            अगर (backend == "ext_oneapi_hip:gpu") वापस 3;
            अगर (backend == "opencl:cpu") वापस 4;
            अगर (backend == "opencl:acc") वापस 5;
            म_लिखो("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        अटल bool compare_backend(std::string &backend1, std::string &backend2) {
            वापस convert_backend_सूचकांक(backend1) < convert_backend_सूचकांक(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            अगर (default_device.is_cpu())
                _cpu_device = 0;

            स्वतः Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, माप_प्रकार> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            स्वतः preferred_platform_name = get_preferred_gpu_platform_name();

            जबतक (!Platforms.empty()) {
                स्वतः Platform = Platforms.back();
                Platforms.pop_back();
                स्वतः platform_name = Platform.get_info<sycl::info::platform::name>();
                अगर (platform_name.compare(preferred_platform_name) != 0) {
                    जारी;
                }
                स्वतः devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                क्रम (स्थिर स्वतः &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> कुंजियाँ;
            क्रम(स्वतः it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                कुंजियाँ.push_back(it->first);
            }
            std::sort(कुंजियाँ.begin(), कुंजियाँ.end(), compare_backend);

            क्रम (स्वतः &कुंजी : कुंजियाँ) {
                std::vector<sycl::device> devs = backend_devices[कुंजी];
                std::sort(devs.begin(), devs.end(), compare_dev);
                क्रम (स्थिर स्वतः &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            क्रम (स्वतः &dev : sycl_all_devs)
            {
                अगर (dev == default_device)
                {
                    जारी;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                अगर (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        व्योम check_id(अचिन्हित पूर्णांक id) स्थिर
        {
            अगर (id >= _devs.size())
            {
                फेंको std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        स्थिर अचिन्हित पूर्णांक DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<अचिन्हित पूर्णांक, अचिन्हित पूर्णांक> _thread2dev_map;
        पूर्णांक _cpu_device = -1;
    };

    अटल अंतरभूत sycl::queue &get_default_queue()
    {
        वापस dev_mgr::instance().current_device().default_queue();
    }

    namespace detail
    {
        क्रमागत श्रेणी pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        अटल pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              स्थिर व्योम *ptr)
        {
            चयन (sycl::get_pointer_type(ptr, q.get_context()))
            {
            हाल sycl::usm::alloc::unknown:
                वापस pointer_access_attribute::host_only;
            हाल sycl::usm::alloc::device:
                वापस pointer_access_attribute::device_only;
            हाल sycl::usm::alloc::shared:
            हाल sycl::usm::alloc::host:
                वापस pointer_access_attribute::host_device;
            }
        }

        ढाँचा <typename ArgT>
        अंतरभूत constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((अचिन्हित अक्षर)library_तथ्य_t::library_तथ्य_t_आकार <=
                              std::numeric_limits<अचिन्हित अक्षर>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_तथ्य_t>, "Unsupported ArgT");
            वापस (std::uint64_t)Val;
        }

        ढाँचा <typename FirstT, typename... RestT>
        अंतरभूत constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_तथ्य_t::library_तथ्य_t_आकार <=
                              std::numeric_limits<अचिन्हित अक्षर>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(माप...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_तथ्य_t>, "Unsupported FirstT");
            वापस get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        श्रेणी mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#अगर defined(__linux__)
                mapped_address_space =
                    (byte_t *)mmap(nullptr, mapped_region_आकार, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#या_अगर defined(_WIN64)
                mapped_address_space = (byte_t *)VirtualAlloc(
                    शून्य,               // NULL specified as the base address parameter
                    mapped_region_आकार, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#अन्यथा
#त्रुटि "Only support Windows and Linux."
#पूर्ण_अगर
                अग्र_free = mapped_address_space;
            }

        खुला:
            using बफर_id_t = पूर्णांक;

            काष्ठा allocation
            {
                बफर_t बफर;
                byte_t *alloc_ptr;
                माप_प्रकार size;
            };

            ~mem_mgr()
            {
#अगर defined(__linux__)
                munmap(mapped_address_space, mapped_region_आकार);
#या_अगर defined(_WIN64)
                VirtualFree(mapped_address_space, 0, MEM_RELEASE);
#अन्यथा
#त्रुटि "Only support Windows and Linux."
#पूर्ण_अगर
            }

            mem_mgr(स्थिर mem_mgr &) = मिटाओ;
            mem_mgr &चालक=(स्थिर mem_mgr &) = मिटाओ;
            mem_mgr(mem_mgr &&) = मिटाओ;
            mem_mgr &चालक=(mem_mgr &&) = मिटाओ;

            /// Allocate
            व्योम *mem_alloc(माप_प्रकार size)
            {
                अगर (!size)
                    वापस nullptr;
                std::lock_guard<std::mutex> lock(m_mutex);
                अगर (अग्र_free + size > mapped_address_space + mapped_region_आकार)
                {
                    फेंको std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                // Allocation
                sycl::range<1> r(size);
                बफर_t buf(r);
                allocation A{buf, अग्र_free, size};
                // Map allocation to device pointer
                व्योम *परिणाम = अग्र_free;
                m_map.emplace(अग्र_free + size, A);
                // Update pointer to the next free space.
                अग्र_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                वापस परिणाम;
            }

            /// Deallocate
            व्योम mem_free(स्थिर व्योम *ptr)
            {
                अगर (!ptr)
                    वापस;
                std::lock_guard<std::mutex> lock(m_mutex);
                स्वतः it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            allocation translate_ptr(स्थिर व्योम *ptr)
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                स्वतः it = get_map_iterator(ptr);
                वापस it->second;
            }

            /// Check if the pointer represents device pointer or not.
            bool is_device_ptr(स्थिर व्योम *ptr) स्थिर
            {
                std::lock_guard<std::mutex> lock(m_mutex);
                वापस (mapped_address_space <= ptr) &&
                       (ptr < mapped_address_space + mapped_region_आकार);
            }

            /// Returns the instance of memory manager singleton.
            अटल mem_mgr &instance()
            {
                अटल mem_mgr m;
                वापस m;
            }

        निजी:
            std::map<byte_t *, allocation> m_map;
            mutable std::mutex m_mutex;
            byte_t *mapped_address_space;
            byte_t *अग्र_free;
            स्थिर माप_प्रकार mapped_region_आकार = 128ull * 1024 * 1024 * 1024;
            स्थिर माप_प्रकार alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            स्थिर माप_प्रकार extra_padding = 0;

            std::map<byte_t *, allocation>::iterator get_map_iterator(स्थिर व्योम *ptr)
            {
                स्वतः it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<स्थिर byte_t *>(ptr)));
                अगर (it == m_map.end())
                {
                    // Not a virtual pointer.
                    फेंको std::runtime_error("can not get buffer from non-virtual pointer");
                }
                स्थिर allocation &alloc = it->second;
                अगर (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    फेंको std::runtime_error("invalid virtual pointer");
                }
                वापस it;
            }
        };

        ढाँचा <श्रेणी T, memory_region Memory, माप_प्रकार Dimension>
        श्रेणी accessor;
        ढाँचा <memory_region Memory, श्रेणी T = byte_t>
        श्रेणी memory_traits
        {
        खुला:
            अटल constexpr sycl::access::target target =
                sycl::access::target::device;
            अटल constexpr sycl::access_mode mode =
                (Memory == constant) ? sycl::access_mode::पढ़ो
                                     : sycl::access_mode::read_write;
            अटल constexpr माप_प्रकार type_आकार = माप(T);
            using element_t =
                typename std::conditional<Memory == constant, स्थिर T, T>::type;
            using मूल्य_t = typename std::remove_cv<T>::type;
            ढाँचा <माप_प्रकार Dimension = 1>
            using accessor_t = typename std::conditional<
                Memory == local, sycl::local_accessor<मूल्य_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            using pointer_t = T *;
        };

        अटल अंतरभूत व्योम *dpct_malloc(माप_प्रकार size, sycl::queue &q)
        {
            वापस sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#घोषणा PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        अटल अंतरभूत व्योम *dpct_malloc(माप_प्रकार &pitch, माप_प्रकार x, माप_प्रकार y, माप_प्रकार z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            वापस dpct_malloc(pitch * y * z, q);
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
        ढाँचा <typename मूल्य_T>
        अटल अंतरभूत sycl::event dpct_memset(sycl::queue &q, व्योम *dev_ptr,
                                              मूल्य_T value, माप_प्रकार size)
        {
            वापस q.fill(dev_ptr, value, size);
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
        ढाँचा <typename मूल्य_T>
        अटल अंतरभूत std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_तथ्य तथ्य, मूल्य_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            माप_प्रकार slice = तथ्य.get_pitch() * तथ्य.get_y();
            अचिन्हित अक्षर *तथ्य_surface = (अचिन्हित अक्षर *)तथ्य.get_तथ्य_ptr();
            क्रम (माप_प्रकार z = 0; z < size.लो(2); ++z)
            {
                अचिन्हित अक्षर *तथ्य_ptr = तथ्य_surface;
                क्रम (माप_प्रकार y = 0; y < size.लो(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, तथ्य_ptr, value, size.लो(0)));
                    तथ्य_ptr += तथ्य.get_pitch();
                }
                तथ्य_surface += slice;
            }
            वापस event_list;
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
        ढाँचा <typename मूल्य_T>
        अटल अंतरभूत std::vector<sycl::event>
        dpct_memset(sycl::queue &q, व्योम *ptr, माप_प्रकार pitch, मूल्य_T val, माप_प्रकार x,
                    माप_प्रकार y)
        {
            वापस dpct_memset(q, pitched_तथ्य(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        अटल memcpy_direction deduce_memcpy_direction(sycl::queue &q, व्योम *to_ptr,
                                                        स्थिर व्योम *from_ptr,
                                                        memcpy_direction dir)
        {
            चयन (dir)
            {
            हाल memcpy_direction::host_to_host:
            हाल memcpy_direction::host_to_device:
            हाल memcpy_direction::device_to_host:
            हाल memcpy_direction::device_to_device:
                वापस dir;
            हाल memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                अटल स्थिर memcpy_direction
                    direction_table[static_cast<अचिन्हित>(pointer_access_attribute::end)]
                                   [static_cast<अचिन्हित>(pointer_access_attribute::end)] =
                                       {{memcpy_direction::host_to_host,
                                         memcpy_direction::device_to_host,
                                         memcpy_direction::host_to_host},
                                        {memcpy_direction::host_to_device,
                                         memcpy_direction::device_to_device,
                                         memcpy_direction::device_to_device},
                                        {memcpy_direction::host_to_host,
                                         memcpy_direction::device_to_device,
                                         memcpy_direction::device_to_device}};
                वापस direction_table[static_cast<अचिन्हित>(get_pointer_attribute(
                    q, to_ptr))][static_cast<अचिन्हित>(get_pointer_attribute(q, from_ptr))];
            }
            शेष:
                फेंको std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        अटल sycl::event
        dpct_memcpy(sycl::queue &q, व्योम *to_ptr, स्थिर व्योम *from_ptr, माप_प्रकार size,
                    memcpy_direction direction,
                    स्थिर std::vector<sycl::event> &dep_events = {})
        {
            अगर (!size)
                वापस sycl::event{};
            वापस q.स_नकल(to_ptr, from_ptr, size, dep_events);
            DPCT_UNUSED(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        अटल अंतरभूत माप_प्रकार get_copy_range(sycl::range<3> size, माप_प्रकार slice,
                                            माप_प्रकार pitch)
        {
            वापस slice * (size.लो(2) - 1) + pitch * (size.लो(1) - 1) + size.लो(0);
        }

        अटल अंतरभूत माप_प्रकार get_offset(sycl::id<3> id, माप_प्रकार slice,
                                        माप_प्रकार pitch)
        {
            वापस slice * id.लो(2) + pitch * id.लो(1) + id.लो(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        अटल अंतरभूत std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, व्योम *to_ptr, स्थिर व्योम *from_ptr,
                    sycl::range<3> to_range, sycl::range<3> from_range,
                    sycl::id<3> to_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    स्थिर std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            श्रेणी host_बफर
            {
                व्योम *_buf;
                माप_प्रकार आकार_2;
                sycl::queue &_q;
                स्थिर std::vector<sycl::event> &_deps; // free operation depends

            खुला:
                host_बफर(माप_प्रकार size, sycl::queue &q,
                            स्थिर std::vector<sycl::event> &deps)
                    : _buf(std::दो_स्मृति(size)), आकार_2(size), _q(q), _deps(deps) {}
                व्योम *get_ptr() स्थिर { वापस _buf; }
                माप_प्रकार get_आकार() स्थिर { वापस आकार_2; }
                ~host_बफर()
                {
                    अगर (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::मुक्त(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            माप_प्रकार to_slice = to_range.लो(1) * to_range.लो(0),
                   from_slice = from_range.लो(1) * from_range.लो(0);
            अचिन्हित अक्षर *to_surface =
                (अचिन्हित अक्षर *)to_ptr + get_offset(to_id, to_slice, to_range.लो(0));
            स्थिर अचिन्हित अक्षर *from_surface =
                (स्थिर अचिन्हित अक्षर *)from_ptr +
                get_offset(from_id, from_slice, from_range.लो(0));

            अगर (to_slice == from_slice && to_slice == size.लो(1) * size.लो(0))
            {
                वापस {dpct_memcpy(q, to_surface, from_surface, to_slice * size.लो(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
            माप_प्रकार आकार_slice = size.लो(1) * size.लो(0);
            चयन (direction)
            {
            हाल host_to_host:
                क्रम (माप_प्रकार z = 0; z < size.लो(2); ++z)
                {
                    अचिन्हित अक्षर *to_ptr = to_surface;
                    स्थिर अचिन्हित अक्षर *from_ptr = from_surface;
                    अगर (to_range.लो(0) == from_range.लो(0) &&
                        to_range.लो(0) == size.लो(0))
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, आकार_slice,
                                                         direction, dep_events));
                    }
                    अन्यथा
                    {
                        क्रम (माप_प्रकार y = 0; y < size.लो(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.लो(0),
                                                             direction, dep_events));
                            to_ptr += to_range.लो(0);
                            from_ptr += from_range.लो(0);
                        }
                    }
                    to_surface += to_slice;
                    from_surface += from_slice;
                }
                अवरोध;
            हाल host_to_device:
            {
                host_बफर buf(get_copy_range(size, to_slice, to_range.लो(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                अगर (to_slice == आकार_slice)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_to_host, dep_events);
                }
                अन्यथा
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, to_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_आकार(),
                                        device_to_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                                 buf.get_आकार(), host_to_device,
                                                 host_events));
                अवरोध;
            }
            हाल device_to_host:
            {
                host_बफर buf(get_copy_range(size, from_slice, from_range.लो(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_to_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_आकार(),
                                                         device_to_host, dep_events)});
                अवरोध;
            }
            हाल device_to_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<श्रेणी dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        to_surface[get_offset(id, to_slice, to_range.लो(0))] =
                            from_surface[get_offset(id, from_slice, from_range.लो(0))];
                    }); }));
                अवरोध;
            शेष:
                फेंको std::runtime_error("dpct_memcpy: invalid direction value");
            }
            वापस event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        अटल अंतरभूत std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_तथ्य to, sycl::id<3> to_id,
                    pitched_तथ्य from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            वापस dpct_memcpy(q, to.get_तथ्य_ptr(), from.get_तथ्य_ptr(),
                               sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        अटल अंतरभूत std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, व्योम *to_ptr, स्थिर व्योम *from_ptr,
                    माप_प्रकार to_pitch, माप_प्रकार from_pitch, माप_प्रकार x, माप_प्रकार y,
                    memcpy_direction direction = automatic)
        {
            वापस dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        namespace deprecated
        {

            ढाँचा <typename T, sycl::usm::alloc AllocKind>
            श्रेणी usm_allocator
            {
            निजी:
                using Alloc = sycl::usm_allocator<T, AllocKind>;
                Alloc _impl;

            खुला:
                using value_type = typename std::allocator_traits<Alloc>::value_type;
                using pointer = typename std::allocator_traits<Alloc>::pointer;
                using const_pointer = typename std::allocator_traits<Alloc>::const_pointer;
                using void_pointer = typename std::allocator_traits<Alloc>::void_pointer;
                using const_void_pointer =
                    typename std::allocator_traits<Alloc>::const_void_pointer;
                using reference = typename std::allocator_traits<Alloc>::value_type &;
                using const_reference =
                    स्थिर typename std::allocator_traits<Alloc>::value_type &;
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

                ढाँचा <typename U>
                काष्ठा rebind
                {
                    प्रकार usm_allocator<U, AllocKind> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(स्थिर usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { वापस &r; }
                const_pointer address(const_reference r) { वापस &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = nullptr)
                {
                    वापस std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                व्योम deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() स्थिर
                {
                    वापस std::allocator_traits<Alloc>::max_size(_impl);
                }
                bool चालक==(स्थिर usm_allocator &other) स्थिर { वापस _impl == other._impl; }
                bool चालक!=(स्थिर usm_allocator &other) स्थिर { वापस _impl != other._impl; }
            };

        } // namespace deprecated

        अंतरभूत व्योम dpct_free(व्योम *ptr,
                              स्थिर sycl::queue &q)
        {
            अगर (ptr)
            {
                sycl::मुक्त(ptr, q.get_context());
            }
        }

        ढाँचा <typename T>
        अंतरभूत स्वतः get_memory(स्थिर व्योम *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<व्योम *>(x));
            वापस new_x;
        }

        ढाँचा <typename T>
        अंतरभूत typename तथ्य_Type_2<T>::T2 get_मूल्य(स्थिर T *s, sycl::queue &q)
        {
            using Ty = typename तथ्य_Type_2<T>::T2;
            Ty s_h;
            अगर (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (व्योम *)&s_h, (स्थिर व्योम *)s, माप(T), device_to_host)
                    .रुको();
            अन्यथा
                s_h = *reinterpret_cast<स्थिर Ty *>(s);
            वापस s_h;
        }

    } // namespace detail

    ढाँचा <typename T>
    अंतरभूत स्वतः get_मूल्य(स्थिर T *s, sycl::queue &q)
    {
        वापस detail::get_मूल्य(s, q);
    }

    namespace detail
    {
    ढाँचा <श्रेणी Ta, श्रेणी Tb, श्रेणी Tc, श्रेणी Ts>
    अंतरभूत व्योम gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णांक m,
                          पूर्णांक n, पूर्णांक k, स्थिर व्योम * alpha, स्थिर व्योम * a, पूर्णांक lda, स्थिर व्योम * b, पूर्णांक ldb,
                          स्थिर व्योम * beta, व्योम * c, पूर्णांक ldc) {
        Ts   alpha_मूल्य_2 = dpct::get_मूल्य(reinterpret_cast<स्थिर Ts *>(alpha), q);
        Ts   beta_मूल्य  = dpct::get_मूल्य(reinterpret_cast<स्थिर Ts *>(beta), q);
        स्वतः तथ्य_a      = get_memory<स्थिर Ta>(a);
        स्वतः तथ्य_b      = get_memory<स्थिर Tb>(b);
        स्वतः तथ्य_c      = get_memory<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_मूल्य_2, तथ्य_a,
                                               lda, तथ्य_b, ldb, beta_मूल्य, तथ्य_c, ldc);
    }

        ढाँचा <typename VecT, श्रेणी द्विआधारी_Operation, श्रेणी = व्योम>
        श्रेणी vectorized_द्विआधारी
        {
        खुला:
            अंतरभूत VecT चालक()(VecT a, VecT b, स्थिर द्विआधारी_Operation द्विआधारी_op)
            {
                VecT v4;
                क्रम (माप_प्रकार i = 0; i < v4.size(); ++i)
                {
                    v4[i] = द्विआधारी_op(a[i], b[i]);
                }
                वापस v4;
            }
        };

        ढाँचा <typename VecT, श्रेणी द्विआधारी_Operation>
        श्रेणी vectorized_द्विआधारी<
            VecT, द्विआधारी_Operation,
            std::void_t<std::invoke_result_t<द्विआधारी_Operation, VecT, VecT>>>
        {
        खुला:
            अंतरभूत VecT चालक()(VecT a, VecT b, स्थिर द्विआधारी_Operation द्विआधारी_op)
            {
                वापस द्विआधारी_op(a, b).ढाँचा as<VecT>();
            }
        };

        ढाँचा <श्रेणी Ta, श्रेणी Tb, श्रेणी Tc, श्रेणी Ts>
        अंतरभूत व्योम gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    पूर्णांक m, पूर्णांक n, पूर्णांक k, स्थिर व्योम * alpha, स्थिर व्योम ** a, पूर्णांक lda, स्थिर व्योम ** b,
                                    पूर्णांक ldb, स्थिर व्योम * beta, व्योम ** c, पूर्णांक ldc, पूर्णांक batch_आकार,
                                    matrix_info_t<भग्न> * matrix_info) {
            Ts alpha_मूल्य_2 = dpct::get_मूल्य(reinterpret_cast<स्थिर Ts *>(alpha), q);
            Ts beta_मूल्य = dpct::get_मूल्य(reinterpret_cast<स्थिर Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->मूल्य_info[0] = alpha_मूल्य_2;
            matrix_info->मूल्य_info[1] = beta_मूल्य;
            matrix_info->आकार_info[0] = m;
            matrix_info->आकार_info[1] = n;
            matrix_info->आकार_info[2] = k;
            matrix_info->ld_info[0] = lda;
            matrix_info->ld_info[1] = ldb;
            matrix_info->ld_info[2] = ldc;
            matrix_info->groupsize_info = batch_आकार;

            sycl::event e = oneapi::mkl::blas::column_major::gemm_batch(
                q, matrix_info->transpose_info, matrix_info->transpose_info + 1,
                matrix_info->आकार_info, matrix_info->आकार_info + 1, matrix_info->आकार_info + 2,
                reinterpret_cast<Ts *>(matrix_info->मूल्य_info), reinterpret_cast<स्थिर Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<स्थिर Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->मूल्य_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        ढाँचा <श्रेणी Ta, श्रेणी Tb, श्रेणी Tc, श्रेणी Ts>
        अंतरभूत व्योम gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    पूर्णांक m, पूर्णांक n, पूर्णांक k, स्थिर व्योम * alpha, स्थिर व्योम * a, पूर्णांक lda,
                                    दीर्घ दीर्घ पूर्णांक stride_a, स्थिर व्योम * b, पूर्णांक ldb, दीर्घ दीर्घ पूर्णांक stride_b,
                                    स्थिर व्योम * beta, व्योम * c, पूर्णांक ldc, दीर्घ दीर्घ पूर्णांक stride_c, पूर्णांक batch_आकार) {
            Ts alpha_मूल्य_2 = dpct::get_मूल्य(reinterpret_cast<स्थिर Ts *>(alpha), q);
            Ts beta_मूल्य = dpct::get_मूल्य(reinterpret_cast<स्थिर Ts *>(beta), q);
            स्वतः तथ्य_a = get_memory<स्थिर Ta>(a);
            स्वतः तथ्य_b = get_memory<स्थिर Tb>(b);
            स्वतः तथ्य_c = get_memory<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_मूल्य_2,
                                                         तथ्य_a, lda, stride_a, तथ्य_b, ldb, stride_b, beta_मूल्य,
                                                         तथ्य_c, ldc, stride_c, batch_आकार);
        }

    } // namespace detail

    ढाँचा <typename VecT, श्रेणी द्विआधारी_Operation>
    अंतरभूत अचिन्हित vectorized_द्विआधारी(अचिन्हित a, अचिन्हित b,
                                      स्थिर द्विआधारी_Operation द्विआधारी_op)
    {
        sycl::vec<अचिन्हित, 1> v0{a}, v1{b};
        स्वतः v2 = v0.as<VecT>();
        स्वतः v3 = v1.as<VecT>();
        स्वतः v4 =
            detail::vectorized_द्विआधारी<VecT, द्विआधारी_Operation>()(v2, v3, द्विआधारी_op);
        v0 = v4.ढाँचा as<sycl::vec<अचिन्हित, 1>>();
        वापस v0;
    }

    अटल व्योम async_dpct_memcpy(व्योम *to_ptr, स्थिर व्योम *from_ptr, माप_प्रकार size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, size, direction);
    }

    अटल अंतरभूत अचिन्हित पूर्णांक select_device(अचिन्हित पूर्णांक id)
    {
        dev_mgr::instance().select_device(id);
        वापस id;
    }

    ढाँचा <typename T>
    T permute_sub_group_by_xor(sycl::sub_group g, T x, अचिन्हित पूर्णांक mask,
                               अचिन्हित पूर्णांक logical_sub_group_आकार = 32)
    {
        अचिन्हित पूर्णांक id = g.get_local_linear_id();
        अचिन्हित पूर्णांक start_सूचकांक =
            id / logical_sub_group_आकार * logical_sub_group_आकार;
        अचिन्हित पूर्णांक target_offset = (id % logical_sub_group_आकार) ^ mask;
        वापस sycl::select_from_group(g, x,
                                       target_offset < logical_sub_group_आकार
                                           ? start_सूचकांक + target_offset
                                           : id);
    }

    ढाँचा <typename T1, typename T2>
    using dot_product_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    ढाँचा <typename T>
    sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
      वापस sycl::vec<T, 1>(val)
          .ढाँचा as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .ढाँचा convert<T>();
    }

    ढाँचा <typename T1, typename T2, typename T3>
    अंतरभूत स्वतः dp4a(T1 a, T2 b, T3 c) {
      dot_product_acc_t<T1, T2> res = c;
      स्वतः va = extract_and_sign_or_zero_extend4(a);
      स्वतः vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      वापस res;
    }

    काष्ठा sub_sat
    {
        ढाँचा <typename T>
        स्वतः चालक()(स्थिर T x, स्थिर T y) स्थिर
        {
            वापस sycl::sub_sat(x, y);
        }
    };

    ढाँचा <typename S, typename T>
    अंतरभूत T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        स्वतः v2 = v0.ढाँचा as<S>();
        स्वतः v3 = v1.ढाँचा as<S>();
        स्वतः v4 = sycl::min(v2, v3);
        v0 = v4.ढाँचा as<sycl::vec<T, 1>>();
        वापस v0;
    }

    अंतरभूत भग्न घात(स्थिर भग्न a, स्थिर पूर्णांक b) { वापस sycl::pown(a, b); }
    अंतरभूत द्विगुन घात(स्थिर द्विगुन a, स्थिर पूर्णांक b) { वापस sycl::pown(a, b); }
    अंतरभूत भग्न घात(स्थिर भग्न a, स्थिर भग्न b) { वापस sycl::घात(a, b); }
    अंतरभूत द्विगुन घात(स्थिर द्विगुन a, स्थिर द्विगुन b) { वापस sycl::घात(a, b); }
    ढाँचा <typename T, typename U>
    अंतरभूत typename std::enable_if_t<std::is_floating_point_v<T>, T>
    घात(स्थिर T a, स्थिर U b)
    {
        वापस sycl::घात(a, static_cast<T>(b));
    }
    ढाँचा <typename T, typename U>
    अंतरभूत typename std::enable_if_t<!std::is_floating_point_v<T>, द्विगुन>
    घात(स्थिर T a, स्थिर U b)
    {
        वापस sycl::घात(static_cast<द्विगुन>(a), static_cast<द्विगुन>(b));
    }

    अंतरभूत द्विगुन min(स्थिर द्विगुन a, स्थिर भग्न b)
    {
        वापस sycl::fmin(a, static_cast<द्विगुन>(b));
    }
    अंतरभूत द्विगुन min(स्थिर भग्न a, स्थिर द्विगुन b)
    {
        वापस sycl::fmin(static_cast<द्विगुन>(a), b);
    }
    अंतरभूत भग्न min(स्थिर भग्न a, स्थिर भग्न b) { वापस sycl::fmin(a, b); }
    अंतरभूत द्विगुन min(स्थिर द्विगुन a, स्थिर द्विगुन b) { वापस sycl::fmin(a, b); }
    अंतरभूत std::uint32_t min(स्थिर std::uint32_t a, स्थिर std::int32_t b)
    {
        वापस sycl::min(a, static_cast<std::uint32_t>(b));
    }
    अंतरभूत std::uint32_t min(स्थिर std::int32_t a, स्थिर std::uint32_t b)
    {
        वापस sycl::min(static_cast<std::uint32_t>(a), b);
    }
    अंतरभूत std::int32_t min(स्थिर std::int32_t a, स्थिर std::int32_t b)
    {
        वापस sycl::min(a, b);
    }
    अंतरभूत std::uint32_t min(स्थिर std::uint32_t a, स्थिर std::uint32_t b)
    {
        वापस sycl::min(a, b);
    }
    अंतरभूत std::uint64_t min(स्थिर std::uint64_t a, स्थिर std::int64_t b)
    {
        वापस sycl::min(a, static_cast<std::uint64_t>(b));
    }
    अंतरभूत std::uint64_t min(स्थिर std::int64_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::min(static_cast<std::uint64_t>(a), b);
    }
    अंतरभूत std::int64_t min(स्थिर std::int64_t a, स्थिर std::int64_t b)
    {
        वापस sycl::min(a, b);
    }
    अंतरभूत std::uint64_t min(स्थिर std::uint64_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::min(a, b);
    }
    अंतरभूत std::uint64_t min(स्थिर std::uint64_t a, स्थिर std::int32_t b)
    {
        वापस sycl::min(a, static_cast<std::uint64_t>(b));
    }
    अंतरभूत std::uint64_t min(स्थिर std::int32_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::min(static_cast<std::uint64_t>(a), b);
    }
    अंतरभूत std::uint64_t min(स्थिर std::uint64_t a, स्थिर std::uint32_t b)
    {
        वापस sycl::min(a, static_cast<std::uint64_t>(b));
    }
    अंतरभूत std::uint64_t min(स्थिर std::uint32_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    अंतरभूत द्विगुन max(स्थिर द्विगुन a, स्थिर भग्न b)
    {
        वापस sycl::fmax(a, static_cast<द्विगुन>(b));
    }
    अंतरभूत द्विगुन max(स्थिर भग्न a, स्थिर द्विगुन b)
    {
        वापस sycl::fmax(static_cast<द्विगुन>(a), b);
    }
    अंतरभूत भग्न max(स्थिर भग्न a, स्थिर भग्न b) { वापस sycl::fmax(a, b); }
    अंतरभूत द्विगुन max(स्थिर द्विगुन a, स्थिर द्विगुन b) { वापस sycl::fmax(a, b); }
    अंतरभूत std::uint32_t max(स्थिर std::uint32_t a, स्थिर std::int32_t b)
    {
        वापस sycl::max(a, static_cast<std::uint32_t>(b));
    }
    अंतरभूत std::uint32_t max(स्थिर std::int32_t a, स्थिर std::uint32_t b)
    {
        वापस sycl::max(static_cast<std::uint32_t>(a), b);
    }
    अंतरभूत std::int32_t max(स्थिर std::int32_t a, स्थिर std::int32_t b)
    {
        वापस sycl::max(a, b);
    }
    अंतरभूत std::uint32_t max(स्थिर std::uint32_t a, स्थिर std::uint32_t b)
    {
        वापस sycl::max(a, b);
    }
    अंतरभूत std::uint64_t max(स्थिर std::uint64_t a, स्थिर std::int64_t b)
    {
        वापस sycl::max(a, static_cast<std::uint64_t>(b));
    }
    अंतरभूत std::uint64_t max(स्थिर std::int64_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::max(static_cast<std::uint64_t>(a), b);
    }
    अंतरभूत std::int64_t max(स्थिर std::int64_t a, स्थिर std::int64_t b)
    {
        वापस sycl::max(a, b);
    }
    अंतरभूत std::uint64_t max(स्थिर std::uint64_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::max(a, b);
    }
    अंतरभूत std::uint64_t max(स्थिर std::uint64_t a, स्थिर std::int32_t b)
    {
        वापस sycl::max(a, static_cast<std::uint64_t>(b));
    }
    अंतरभूत std::uint64_t max(स्थिर std::int32_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::max(static_cast<std::uint64_t>(a), b);
    }
    अंतरभूत std::uint64_t max(स्थिर std::uint64_t a, स्थिर std::uint32_t b)
    {
        वापस sycl::max(a, static_cast<std::uint64_t>(b));
    }
    अंतरभूत std::uint64_t max(स्थिर std::uint32_t a, स्थिर std::uint64_t b)
    {
        वापस sycl::max(static_cast<std::uint64_t>(a), b);
    }

    अंतरभूत व्योम
    has_capability_or_fail(स्थिर sycl::device &dev,
                           स्थिर std::initializer_list<sycl::aspect> &props)
    {
        क्रम (स्थिर स्वतः &it : props)
        {
            अगर (dev.has(it))
                जारी;
            चयन (it)
            {
            हाल sycl::aspect::fp64:
                फेंको std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                अवरोध;
            हाल sycl::aspect::fp16:
                फेंको std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                अवरोध;
            शेष:
#घोषणा __SYCL_ASPECT(ASPECT, ID) \
    हाल sycl::aspect::ASPECT:    \
        वापस #ASPECT;
#घोषणा __SYCL_ASPECT_DEPRECATED(ASPECT, ID, MESSAGE) __SYCL_ASPECT(ASPECT, ID)
#घोषणा __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, MESSAGE)
                स्वतः getAspectNameStr = [](sycl::aspect Aspect_अंक) -> std::string
                {
                    चयन (Aspect_अंक)
                    {
#समावेश <sycl/info/aspects.def>
#समावेश <sycl/info/aspects_deprecated.def>
                    शेष:
                        वापस "unknown aspect";
                    }
                };
#अघोषित __SYCL_ASPECT_DEPRECATED_ALIAS
#अघोषित __SYCL_ASPECT_DEPRECATED
#अघोषित __SYCL_ASPECT
                फेंको std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            अवरोध;
        }
    }

    अटल अंतरभूत अचिन्हित पूर्णांक get_current_device_id()
    {
        वापस dev_mgr::instance().current_device_id();
    }

    अटल अंतरभूत device_ext &get_current_device()
    {
        वापस dev_mgr::instance().current_device();
    }

    अटल अंतरभूत device_ext &get_device(अचिन्हित पूर्णांक id)
    {
        वापस dev_mgr::instance().get_device(id);
    }

    अटल अंतरभूत sycl::queue &get_in_order_queue()
    {
        वापस dev_mgr::instance().current_device().in_order_queue();
    }

    अटल sycl::event
    dpct_memcpy(sycl::queue &q, व्योम *to_ptr, स्थिर व्योम *from_ptr, माप_प्रकार size,
                memcpy_direction direction,
                स्थिर std::vector<sycl::event> &dep_events = {})
    {
        अगर (!size)
            वापस sycl::event{};
        वापस q.स_नकल(to_ptr, from_ptr, size, dep_events);
        DPCT_UNUSED(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    अटल अंतरभूत माप_प्रकार get_copy_range(sycl::range<3> size, माप_प्रकार slice,
                                        माप_प्रकार pitch)
    {
        वापस slice * (size.लो(2) - 1) + pitch * (size.लो(1) - 1) + size.लो(0);
    }

    अटल अंतरभूत माप_प्रकार get_offset(sycl::id<3> id, माप_प्रकार slice,
                                    माप_प्रकार pitch)
    {
        वापस slice * id.लो(2) + pitch * id.लो(1) + id.लो(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    अटल अंतरभूत std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, व्योम *to_ptr, स्थिर व्योम *from_ptr,
                sycl::range<3> to_range, sycl::range<3> from_range,
                sycl::id<3> to_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                स्थिर std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        श्रेणी host_बफर
        {
            व्योम *_buf;
            माप_प्रकार आकार_2;
            sycl::queue &_q;
            स्थिर std::vector<sycl::event> &_deps; // free operation depends

        खुला:
            host_बफर(माप_प्रकार size, sycl::queue &q,
                        स्थिर std::vector<sycl::event> &deps)
                : _buf(std::दो_स्मृति(size)), आकार_2(size), _q(q), _deps(deps) {}
            व्योम *get_ptr() स्थिर { वापस _buf; }
            माप_प्रकार get_आकार() स्थिर { वापस आकार_2; }
            ~host_बफर()
            {
                अगर (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::मुक्त(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        माप_प्रकार to_slice = to_range.लो(1) * to_range.लो(0),
               from_slice = from_range.लो(1) * from_range.लो(0);
        अचिन्हित अक्षर *to_surface =
            (अचिन्हित अक्षर *)to_ptr + get_offset(to_id, to_slice, to_range.लो(0));
        स्थिर अचिन्हित अक्षर *from_surface =
            (स्थिर अचिन्हित अक्षर *)from_ptr +
            get_offset(from_id, from_slice, from_range.लो(0));

        अगर (to_slice == from_slice && to_slice == size.लो(1) * size.लो(0))
        {
            वापस {dpct_memcpy(q, to_surface, from_surface, to_slice * size.लो(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, to_ptr, from_ptr, direction);
        माप_प्रकार आकार_slice = size.लो(1) * size.लो(0);
        चयन (direction)
        {
        हाल host_to_host:
            क्रम (माप_प्रकार z = 0; z < size.लो(2); ++z)
            {
                अचिन्हित अक्षर *to_ptr = to_surface;
                स्थिर अचिन्हित अक्षर *from_ptr = from_surface;
                अगर (to_range.लो(0) == from_range.लो(0) &&
                    to_range.लो(0) == size.लो(0))
                {
                    event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, आकार_slice,
                                                     direction, dep_events));
                }
                अन्यथा
                {
                    क्रम (माप_प्रकार y = 0; y < size.लो(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, to_ptr, from_ptr, size.लो(0),
                                                         direction, dep_events));
                        to_ptr += to_range.लो(0);
                        from_ptr += from_range.लो(0);
                    }
                }
                to_surface += to_slice;
                from_surface += from_slice;
            }
            अवरोध;
        हाल host_to_device:
        {
            host_बफर buf(get_copy_range(size, to_slice, to_range.लो(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            अगर (to_slice == आकार_slice)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, to_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_to_host, dep_events);
            }
            अन्यथा
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, to_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_to_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), to_surface, buf.get_आकार(),
                                    device_to_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, to_surface, buf.get_ptr(),
                                             buf.get_आकार(), host_to_device,
                                             host_events));
            अवरोध;
        }
        हाल device_to_host:
        {
            host_बफर buf(get_copy_range(size, from_slice, from_range.लो(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, to_surface, buf.get_ptr(), to_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_to_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_आकार(),
                                                     device_to_host, dep_events)});
            अवरोध;
        }
        हाल device_to_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<श्रेणी dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                to_surface[get_offset(id, to_slice, to_range.लो(0))] =
                    from_surface[get_offset(id, from_slice, from_range.लो(0))];
            }); }));
        अवरोध;
        शेष:
            फेंको std::runtime_error("dpct_memcpy: invalid direction value");
        }
        वापस event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    अटल अंतरभूत std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_तथ्य to, sycl::id<3> to_id,
                pitched_तथ्य from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        वापस dpct_memcpy(q, to.get_तथ्य_ptr(), from.get_तथ्य_ptr(),
                           sycl::range<3>(to.get_pitch(), to.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), to_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    अटल अंतरभूत std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, व्योम *to_ptr, स्थिर व्योम *from_ptr,
                माप_प्रकार to_pitch, माप_प्रकार from_pitch, माप_प्रकार x, माप_प्रकार y,
                memcpy_direction direction = automatic)
    {
        वापस dpct_memcpy(q, to_ptr, from_ptr, sycl::range<3>(to_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    अंतरभूत व्योम gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णांक m, पूर्णांक n,
                     पूर्णांक k, स्थिर व्योम * alpha, स्थिर व्योम * a, library_तथ्य_t a_type, पूर्णांक lda, स्थिर व्योम * b,
                     library_तथ्य_t b_type, पूर्णांक ldb, स्थिर व्योम * beta, व्योम * c, library_तथ्य_t c_type, पूर्णांक ldc,
                     library_तथ्य_t scaling_type) {
        अगर (scaling_type == library_तथ्य_t::real_float &&
            c_type == library_तथ्य_t::complex_float)
        {
            scaling_type = library_तथ्य_t::complex_float;
        }
        अन्यथा अगर (scaling_type == library_तथ्य_t::real_double &&
                 c_type == library_तथ्य_t::complex_double)
        {
            scaling_type = library_तथ्य_t::complex_double;
        }

        std::uint64_t कुंजी =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        चयन (कुंजी)
        {
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_float, library_तथ्य_t::real_float,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_impl<भग्न, भग्न, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_double, library_तथ्य_t::real_double,
            library_तथ्य_t::real_double, library_तथ्य_t::real_double):
        {
            detail::gemm_impl<द्विगुन, द्विगुन, द्विगुन, द्विगुन>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::complex_float, library_तथ्य_t::complex_float,
            library_तथ्य_t::complex_float, library_तथ्य_t::complex_float):
        {
            detail::gemm_impl<std::complex<भग्न>, std::complex<भग्न>,
                              std::complex<भग्न>, std::complex<भग्न>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::complex_double, library_तथ्य_t::complex_double,
            library_तथ्य_t::complex_double, library_तथ्य_t::complex_double):
        {
            detail::gemm_impl<std::complex<द्विगुन>, std::complex<द्विगुन>,
                              std::complex<द्विगुन>, std::complex<द्विगुन>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_half, library_तथ्य_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
#अगर_घोषित __INTEL_MKL__
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_bfloat16,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_half, library_तथ्य_t::real_float):
        {
            भग्न alpha_मूल्य_2 =
                dpct::get_मूल्य(reinterpret_cast<स्थिर भग्न *>(alpha), q);
            भग्न beta_मूल्य =
                dpct::get_मूल्य(reinterpret_cast<स्थिर भग्न *>(beta), q);
            sycl::half alpha_half(alpha_मूल्य_2);
            sycl::half beta_half(beta_मूल्य);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_int8, library_तथ्य_t::real_int8,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_bfloat16,
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_int8, library_तथ्य_t::real_int8,
            library_तथ्य_t::real_int32, library_तथ्य_t::real_int32):
        {
            भग्न alpha_float =
                dpct::get_मूल्य(reinterpret_cast<स्थिर std::int32_t *>(alpha), q);
            भग्न beta_float =
                dpct::get_मूल्य(reinterpret_cast<स्थिर std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, भग्न>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            अवरोध;
        }
#पूर्ण_अगर // __INTEL_MKL__
        शेष:
            फेंको std::runtime_error("the combination of data type is unsupported");
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
    अंतरभूत व्योम gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णांक m,
                           पूर्णांक n, पूर्णांक k, स्थिर व्योम * alpha, स्थिर व्योम * a[], library_तथ्य_t a_type, पूर्णांक lda,
                           स्थिर व्योम * b[], library_तथ्य_t b_type, पूर्णांक ldb, स्थिर व्योम * beta, व्योम * c[],
                           library_तथ्य_t c_type, पूर्णांक ldc, पूर्णांक batch_आकार, library_तथ्य_t scaling_type,
                           matrix_info_t<भग्न> * matrix_info) {
        std::uint64_t कुंजी =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        चयन (कुंजी)
        {
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_float, library_तथ्य_t::real_float,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<भग्न, भग्न, भग्न, भग्न>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_double, library_तथ्य_t::real_double,
            library_तथ्य_t::real_double, library_तथ्य_t::real_double):
        {
            detail::gemm_batch_impl<द्विगुन, द्विगुन, द्विगुन, द्विगुन>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_half, library_तथ्य_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
#अगर_घोषित __INTEL_MKL__
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_bfloat16,
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_bfloat16,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
#पूर्ण_अगर
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_int8, library_तथ्य_t::real_int8,
            library_तथ्य_t::real_int32, library_तथ्य_t::real_int32):
        {
            भग्न alpha_float =
                dpct::get_मूल्य(reinterpret_cast<स्थिर std::int32_t *>(alpha), q);
            भग्न beta_float =
                dpct::get_मूल्य(reinterpret_cast<स्थिर std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, भग्न>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_आकार,
                matrix_info);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_int8, library_तथ्य_t::real_int8,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_half, library_तथ्य_t::real_float):
        {
            भग्न alpha_मूल्य_2 =
                dpct::get_मूल्य(reinterpret_cast<स्थिर भग्न *>(alpha), q);
            भग्न beta_मूल्य =
                dpct::get_मूल्य(reinterpret_cast<स्थिर भग्न *>(beta), q);
            sycl::half alpha_half(alpha_मूल्य_2);
            sycl::half beta_half(beta_मूल्य);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_आकार, matrix_info);
            अवरोध;
        }
        शेष:
            फेंको std::runtime_error("the combination of data type is unsupported");
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
    अंतरभूत व्योम gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णांक m,
                           पूर्णांक n, पूर्णांक k, स्थिर व्योम * alpha, स्थिर व्योम * a, library_तथ्य_t a_type, पूर्णांक lda,
                           दीर्घ दीर्घ पूर्णांक stride_a, स्थिर व्योम * b, library_तथ्य_t b_type, पूर्णांक ldb,
                           दीर्घ दीर्घ पूर्णांक stride_b, स्थिर व्योम * beta, व्योम * c, library_तथ्य_t c_type, पूर्णांक ldc,
                           दीर्घ दीर्घ पूर्णांक stride_c, पूर्णांक batch_आकार, library_तथ्य_t scaling_type) {
        अगर (scaling_type == library_तथ्य_t::real_float &&
            c_type == library_तथ्य_t::complex_float)
        {
            scaling_type = library_तथ्य_t::complex_float;
        }
        अन्यथा अगर (scaling_type == library_तथ्य_t::real_double &&
                 c_type == library_तथ्य_t::complex_double)
        {
            scaling_type = library_तथ्य_t::complex_double;
        }

        std::uint64_t कुंजी =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        चयन (कुंजी)
        {
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_float, library_तथ्य_t::real_float,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<भग्न, भग्न, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_double, library_तथ्य_t::real_double,
            library_तथ्य_t::real_double, library_तथ्य_t::real_double):
        {
            detail::gemm_batch_impl<द्विगुन, द्विगुन, द्विगुन, द्विगुन>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::complex_float, library_तथ्य_t::complex_float,
            library_तथ्य_t::complex_float, library_तथ्य_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<भग्न>, std::complex<भग्न>,
                                    std::complex<भग्न>, std::complex<भग्न>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::complex_double, library_तथ्य_t::complex_double,
            library_तथ्य_t::complex_double, library_तथ्य_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<द्विगुन>, std::complex<द्विगुन>,
                                    std::complex<द्विगुन>, std::complex<द्विगुन>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_half, library_तथ्य_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
#अगर_घोषित __INTEL_MKL__
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_bfloat16,
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_bfloat16, library_तथ्य_t::real_bfloat16,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_आकार);
            अवरोध;
        }
#पूर्ण_अगर
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_int8, library_तथ्य_t::real_int8,
            library_तथ्य_t::real_int32, library_तथ्य_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_int8, library_तथ्य_t::real_int8,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_float, library_तथ्य_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, भग्न, भग्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        हाल detail::get_type_combination_id(
            library_तथ्य_t::real_half, library_तथ्य_t::real_half,
            library_तथ्य_t::real_half, library_तथ्य_t::real_float):
        {
            भग्न alpha_मूल्य_2 =
                dpct::get_मूल्य(reinterpret_cast<स्थिर भग्न *>(alpha), q);
            भग्न beta_मूल्य =
                dpct::get_मूल्य(reinterpret_cast<स्थिर भग्न *>(beta), q);
            sycl::half alpha_half(alpha_मूल्य_2);
            sycl::half beta_half(beta_मूल्य);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_आकार);
            अवरोध;
        }
        शेष:
            फेंको std::runtime_error("the combination of data type is unsupported");
        }
    }

    अटल अंतरभूत व्योम
    async_dpct_memcpy(व्योम *to_ptr, माप_प्रकार to_pitch, स्थिर व्योम *from_ptr,
                      माप_प्रकार from_pitch, माप_प्रकार x, माप_प्रकार y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, to_ptr, from_ptr, to_pitch, from_pitch, x, y,
                            direction);
    }

    using err0 = detail::generic_error_type<काष्ठा err0_tag, पूर्णांक>;
    using err1 = detail::generic_error_type<काष्ठा err1_tag, पूर्णांक>;

    अटल अंतरभूत व्योम dpct_free(व्योम *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    ढाँचा <श्रेणी T, memory_region Memory, माप_प्रकार Dimension> श्रेणी accessor;
    ढाँचा <श्रेणी T, memory_region Memory> श्रेणी accessor<T, Memory, 3> {
    खुला:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::ढाँचा accessor_t<3>;
        accessor(pointer_t तथ्य, स्थिर sycl::range<3> &in_range)
            : तथ्य_2(तथ्य), _range(in_range) {}
        ढाँचा <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, स्थिर accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(स्थिर accessor_t &acc, स्थिर sycl::range<3> &in_range)
            : accessor(acc.get_pointer(), in_range) {}
        accessor<T, Memory, 2> चालक[](माप_प्रकार सूचकांक) स्थिर {
            sycl::range<2> sub(_range.लो(1), _range.लो(2));
            वापस accessor<T, Memory, 2>(तथ्य_2 + सूचकांक * sub.size(), sub);
        }

        pointer_t get_ptr() स्थिर { वापस तथ्य_2; }

    निजी:
        pointer_t तथ्य_2;
        sycl::range<3> _range;
    };
    ढाँचा <श्रेणी T, memory_region Memory> श्रेणी accessor<T, Memory, 2> {
    खुला:
        using memory_t = detail::memory_traits<Memory, T>;
        using element_t = typename memory_t::element_t;
        using pointer_t = typename memory_t::pointer_t;
        using accessor_t = typename memory_t::ढाँचा accessor_t<2>;
        accessor(pointer_t तथ्य, स्थिर sycl::range<2> &in_range)
            : तथ्य_2(तथ्य), _range(in_range) {}
        ढाँचा <memory_region M = Memory>
        accessor(typename std::enable_if<M != local, स्थिर accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(स्थिर accessor_t &acc, स्थिर sycl::range<2> &in_range)
            : accessor(acc.get_pointer(), in_range) {}

        pointer_t चालक[](माप_प्रकार सूचकांक) स्थिर {
            वापस तथ्य_2 + _range.लो(1) * सूचकांक;
        }

        pointer_t get_ptr() स्थिर { वापस तथ्य_2; }

    निजी:
        pointer_t तथ्य_2;
        sycl::range<2> _range;
    };

    namespace detail {
        /// Device variable with address space of shared, global or constant.
        ढाँचा <श्रेणी T, memory_region Memory, माप_प्रकार Dimension> श्रेणी device_memory {
        खुला:
            using accessor_t =
                typename detail::memory_traits<Memory,
                                            T>::ढाँचा accessor_t<Dimension>;
            using मूल्य_t = typename detail::memory_traits<Memory, T>::मूल्य_t;
            using dpct_accessor_t = dpct::accessor<T, Memory, Dimension>;

            device_memory() : device_memory(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_memory(स्थिर sycl::range<Dimension> &in_range,
                        std::initializer_list<मूल्य_t> &&init_list)
                : device_memory(in_range) {
                निश्चित(init_list.size() <= in_range.size());
                _host_ptr = (मूल्य_t *)std::दो_स्मृति(आकार_2);
                std::स_रखो(_host_ptr, 0, आकार_2);
                std::स_नकल(_host_ptr, init_list.begin(), init_list.size() * माप(T));
            }

            /// Constructor of 2-D array with initializer list
            ढाँचा <माप_प्रकार D = Dimension>
            device_memory(
                स्थिर typename std::enable_if<D == 2, sycl::range<2>>::type &in_range,
                std::initializer_list<std::initializer_list<मूल्य_t>> &&init_list)
                : device_memory(in_range) {
                निश्चित(init_list.size() <= in_range[0]);
                _host_ptr = (मूल्य_t *)std::दो_स्मृति(आकार_2);
                std::स_रखो(_host_ptr, 0, आकार_2);
                स्वतः tmp_तथ्य = _host_ptr;
                क्रम (स्वतः sub_list : init_list) {
                    निश्चित(sub_list.size() <= in_range[1]);
                    std::स_नकल(tmp_तथ्य, sub_list.begin(),
                                sub_list.size() * माप(T));
                    tmp_तथ्य += in_range[1];
                }
            }

            /// Constructor with range
            device_memory(स्थिर sycl::range<Dimension> &range_in)
                : आकार_2(range_in.size() * माप(T)), _range(range_in),
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
            ढाँचा <श्रेणी... Args>
            device_memory(Args... Arguments)
                : device_memory(sycl::range<Dimension>(Arguments...)) {}

            ~device_memory() {
                अगर (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                अगर (_host_ptr)
                    std::मुक्त(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            व्योम init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            व्योम init(sycl::queue &q) {
                अगर (_device_ptr)
                    वापस;
                अगर (!आकार_2)
                    वापस;
                allocate_device(q);
                अगर (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, आकार_2,
                                        host_to_device);
            }

            /// The variable is assigned to a device pointer.
            व्योम assign(मूल्य_t *src, माप_प्रकार size) {
                यह->~device_memory();
                नया (यह) device_memory(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            मूल्य_t *get_ptr() { वापस get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            मूल्य_t *get_ptr(sycl::queue &q) {
                init(q);
                वापस _device_ptr;
            }

            /// Get the device memory object size in bytes.
            माप_प्रकार get_आकार() { वापस आकार_2; }

            ढाँचा <माप_प्रकार D = Dimension>
            typename std::enable_if<D == 1, T>::type &चालक[](माप_प्रकार सूचकांक) {
                init();
                वापस _device_ptr[सूचकांक];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            ढाँचा <माप_प्रकार D = Dimension>
            typename std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                वापस dpct_accessor_t((T *)_device_ptr, _range);
            }

        निजी:
            device_memory(मूल्य_t *memory_ptr, माप_प्रकार size)
                : आकार_2(size), _range(size / माप(T)), _reference(true),
                _device_ptr(memory_ptr) {}

            व्योम allocate_device(sycl::queue &q) {
        #अगर_अघोषित DPCT_USM_LEVEL_NONE
                अगर (Memory == shared) {
                    _device_ptr = (मूल्य_t *)sycl::malloc_shared(आकार_2, q.get_device(),
                                                                q.get_context());
                    वापस;
                }
        #अगर_घोषित SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                अगर (Memory == constant) {
                    _device_ptr = (मूल्य_t *)sycl::malloc_device(
                        आकार_2, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    वापस;
                }
        #पूर्ण_अगर
        #पूर्ण_अगर
                _device_ptr = (मूल्य_t *)detail::dpct_malloc(आकार_2, q);
            }

            माप_प्रकार आकार_2;
            sycl::range<Dimension> _range;
            bool _reference;
            मूल्य_t *_host_ptr;
            मूल्य_t *_device_ptr;
        };
        ढाँचा <श्रेणी T, memory_region Memory>
        श्रेणी device_memory<T, Memory, 0> : खुला device_memory<T, Memory, 1> {
        खुला:
            using base = device_memory<T, Memory, 1>;
            using मूल्य_t = typename base::मूल्य_t;
            using accessor_t =
                typename detail::memory_traits<Memory, T>::ढाँचा accessor_t<0>;

            /// Constructor with initial value.
            device_memory(स्थिर मूल्य_t &val) : base(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_memory() : base(1) {}
        };
        } // namespace detail

    ढाँचा <श्रेणी T, माप_प्रकार Dimension>
    using global_memory = detail::device_memory<T, global, Dimension>;
    ढाँचा <श्रेणी T, माप_प्रकार Dimension>
    using constant_memory = detail::device_memory<T, constant, Dimension>;
    ढाँचा <श्रेणी T, माप_प्रकार Dimension>
    using shared_memory = detail::device_memory<T, shared, Dimension>;


    ढाँचा <typename T,
            sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device>
    अंतरभूत T atomic_fetch_add(T *addr, T operand) {
    स्वतः atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    वापस atm.fetch_add(operand);
    }

    ढाँचा <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
            sycl::memory_scope memoryScope = sycl::memory_scope::device,
            typename T1, typename T2>
    अंतरभूत T1 atomic_fetch_add(T1 *addr, T2 operand) {
    स्वतः atm =
        sycl::atomic_ref<T1, memoryOrder, memoryScope, addressSpace>(addr[0]);
    वापस atm.fetch_add(operand);
    }

    ढाँचा <typename T, sycl::access::address_space addressSpace =
                            sycl::access::address_space::global_space>
    अंतरभूत T atomic_fetch_add(T *addr, T operand,
                            sycl::memory_order memoryOrder) {
    चयन (memoryOrder) {
        हाल sycl::memory_order::relaxed:
            वापस atomic_fetch_add<T, addressSpace, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device>(addr, operand);
        हाल sycl::memory_order::acq_rel:
            वापस atomic_fetch_add<T, addressSpace, sycl::memory_order::acq_rel,
                                    sycl::memory_scope::device>(addr, operand);
        हाल sycl::memory_order::seq_cst:
            वापस atomic_fetch_add<T, addressSpace, sycl::memory_order::seq_cst,
                                    sycl::memory_scope::device>(addr, operand);
        शेष:
            निश्चित(false && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    ढाँचा <sycl::access::address_space addressSpace =
                sycl::access::address_space::global_space,
            typename T1, typename T2>
    अंतरभूत T1 atomic_fetch_add(T1 *addr, T2 operand,
                            sycl::memory_order memoryOrder) {
    atomic_fetch_add<T1, addressSpace>(addr, operand, memoryOrder);
    }

    अंतरभूत अचिन्हित पूर्णांक byte_level_permute(
        अचिन्हित पूर्णांक a, अचिन्हित पूर्णांक b, अचिन्हित पूर्णांक s) {
      अचिन्हित पूर्णांक ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      वापस ret;
    }

    अंतरभूत uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, पूर्णांक mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      अगर (mode >= 1 && mode <= 6) {
        वापस byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } अन्यथा अगर (!mode) {
        वापस byte_level_permute(low32, high32, sel);
      }
      वापस 0;
    }

    ढाँचा <पूर्णांक n_nondefault_params, पूर्णांक n_default_params, typename T>
    श्रेणी args_selector;

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
    ढाँचा <पूर्णांक n_nondefault_params, पूर्णांक n_default_params, typename R,
              typename... Ts>
    श्रेणी args_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      निजी:
        व्योम **kernel_params;
        अक्षर *args_बफर;

        ढाँचा <पूर्णांक i> अटल constexpr पूर्णांक account_for_default_params() {
            constexpr पूर्णांक n_योग_params = माप...(Ts);
            अगर constexpr (i >= n_nondefault_params) {
                वापस n_योग_params - n_default_params +
                       (i - n_nondefault_params);
            } अन्यथा {
                वापस i;
            }
        }

      खुला:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        ढाँचा <पूर्णांक i>
        using arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        अटल constexpr पूर्णांक params_अंक = माप...(Ts);

      निजी:
        ढाँचा <पूर्णांक i> अटल constexpr पूर्णांक get_offset() {
            अगर constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                वापस 0;
            } अन्यथा {
                constexpr पूर्णांक पूर्व_off = get_offset<i - 1>();
                constexpr पूर्णांक पूर्व_past_end =
                    पूर्व_off + माप(arg_type<i - 1>);
                using T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                अगर constexpr (पूर्व_past_end % alignof(T) == 0) {
                    वापस पूर्व_past_end;
                }
                // otherwise bump prev_past_end to match alignment
                अन्यथा {
                    वापस पूर्व_past_end +
                           (alignof(T) - (पूर्व_past_end % alignof(T)));
                }
            }
        }

        अटल अक्षर *get_args_बफर(व्योम **extra) {
            अगर (!extra)
                वापस nullptr;
            क्रम (; (std::माप_प्रकार)*extra != 0; ++extra) {
                अगर ((std::माप_प्रकार)*extra == 1) {
                    वापस static_cast<अक्षर *>(*(extra + 1));
                }
            }
            वापस nullptr;
        }

      खुला:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        args_selector(व्योम **kernel_params, व्योम **extra)
            : kernel_params(kernel_params),
              args_बफर(get_args_बफर(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        ढाँचा <पूर्णांक i> arg_type<i> &लो() {
            अगर (kernel_params) {
                वापस *static_cast<arg_type<i> *>(kernel_params[i]);
            } अन्यथा {
                वापस *reinterpret_cast<arg_type<i> *>(args_बफर +
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
    श्रेणी kernel_launcher {
        ढाँचा <typename FuncT, typename ArgSelector, std::माप_प्रकार... सूचकांक_2>
        अटल व्योम launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<सूचकांक_2...>) {
            func(selector.ढाँचा लो<सूचकांक_2>()...);
        }
        अटल व्योम set_execution_config(dim3 group_range, dim3 local_range,
                                         अचिन्हित पूर्णांक local_mem_आकार_2,
                                         queue_ptr que) {
            अगर (que) {
                _que = que;
            } अन्यथा {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(group_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_आकार = local_mem_आकार_2;


        }
        अटल अंतरभूत std::mutex kernel_function_ptr_map_mutex;

      खुला:
        /// Variables for storing execution configuration.
        अटल अंतरभूत thread_local sycl::queue *_que = nullptr;
        अटल अंतरभूत thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        अटल अंतरभूत thread_local अचिन्हित पूर्णांक local_mem_आकार = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        अटल अंतरभूत std::map<
            स्थिर व्योम *,
            std::function<व्योम(dim3, dim3, व्योम **, अचिन्हित पूर्णांक, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        अटल व्योम register_kernel_ptr(
            स्थिर व्योम *func,
            std::function<व्योम(dim3, dim3, व्योम **, अचिन्हित पूर्णांक, queue_ptr)>
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
        ढाँचा <typename FuncT, typename... ArgsT>
        अटल std::enable_if_t<std::is_invocable_v<FuncT *, ArgsT...>, व्योम>
        launch(FuncT *func, dim3 group_range, dim3 local_range,
               अचिन्हित पूर्णांक local_mem_आकार_2, queue_ptr que, ArgsT... args) {
            set_execution_config(group_range, local_range, local_mem_आकार_2, que);
            func(args...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        अटल व्योम launch(स्थिर व्योम *func, dim3 group_range, dim3 local_range,
                           व्योम **args, अचिन्हित पूर्णांक local_mem_आकार_2,
                           queue_ptr que) {
            std::lock_guard<std::mutex> lock(kernel_function_ptr_map_mutex);
            स्वतः Iter = kernel_function_ptr_map.find(func);
            अगर (Iter == kernel_function_ptr_map.end()) {
                फेंको std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(group_range, local_range, args, local_mem_आकार_2, que);
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
        ढाँचा <typename FuncT>
        अटल std::enable_if_t<std::is_function_v<FuncT>, व्योम>
        launch(FuncT *func, dim3 group_range, dim3 local_range, व्योम **args,
               अचिन्हित पूर्णांक local_mem_आकार_2, queue_ptr que) {
            constexpr माप_प्रकार p_अंक = args_selector<0, 0, FuncT>::params_अंक;
            set_execution_config(group_range, local_range, local_mem_आकार_2, que);
            args_selector<p_अंक, p_अंक, FuncT> selector(args, nullptr);
            launch_helper(func, selector, std::make_index_sequence<p_अंक>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    ढाँचा <typename T>
    T select_from_sub_group(
        sycl::sub_group g,
        T x,
        पूर्णांक remote_local_id,
        पूर्णांक logical_sub_group_आकार = 32) {
      अचिन्हित पूर्णांक start_सूचकांक = g.get_local_linear_id() /
                                 logical_sub_group_आकार *
                                 logical_sub_group_आकार;
      वापस sycl::select_from_group(
          g, x, start_सूचकांक + remote_local_id % logical_sub_group_आकार);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    ढाँचा <typename T>
    व्योम ldmatrix(uintptr_t addr, T* m, bool trans = false, अचिन्हित mat = 0) {
      स्वतः sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      पूर्णांक lane = sg.get_local_linear_id();

      पूर्णांक lane_group8_row = lane / 8;
      पूर्णांक lane_group8_col = lane % 8;

      अगर (!trans) {
        // calculate the source lane
        पूर्णांक src_lane = 2 * lane_group8_row;
        अगर (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        स्वतः recv_addr_uintp =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        स्वतः recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } अन्यथा {
        // calculate the source lane
        पूर्णांक src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        स्वतः recv_addr_uintp_1 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane);
        स्वतः recv_addr_uintp_2 =
            dpct::select_from_sub_group(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        स्वतः recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        स्वतः recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        पूर्णांक सूचकांक = lane / 4;
        sycl::half val0 = recv_addr_1[सूचकांक];
        sycl::half val1 = recv_addr_2[सूचकांक];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    ढाँचा <typename T>
    व्योम ldmatrix(uintptr_t addr, T* m1, T* m2, bool trans = false) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    ढाँचा <typename T>
    व्योम ldmatrix(
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
    ढाँचा <typename T>
    काष्ठा MMAType {
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
    ढाँचा <पूर्णांक M, पूर्णांक N, पूर्णांक K, typename ABType, typename CDType>
    व्योम mma(
        अस्थिर व्योम** d_mat_frag,
        व्योम* a_mat_frag,
        व्योम* b_mat_frag,
        व्योम* c_mat_frag) {
      स्वतः d = reinterpret_cast<अस्थिर CDType**>(d_mat_frag);
      स्वतः a =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(a_mat_frag);
      स्वतः b =
          reinterpret_cast<typename MMAType<ABType>::PackType*>(b_mat_frag);
      स्वतः c = reinterpret_cast<CDType*>(c_mat_frag);

      स्वतः sg = sycl::ext::oneapi::this_work_item::get_sub_group();
      पूर्णांक lane = sg.get_local_linear_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      लघु row_load_offset = 4 * (lane >> 2);
      लघु col_load_offset = 8 * (lane % 4);

      अगर constexpr (M == 8 && N == 8 && K == 4) {
        अगर constexpr (std::is_floating_point_v<CDType>) {
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
          पूर्णांक r_ind = (lane % 2) ? 1 : 0;
          पूर्णांक c_ind = ((lane % 4) / 2) ? 2 : 0;

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

          क्रम (पूर्णांक i = 0; i < 4; i++) {
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

            स्वतः ra = reinterpret_cast<ABType*>(recv_a);
            स्वतः rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment (for
            // even work item indices) d0 += col0{ a0 } * row0{ b0 } d1 += col0{
            // a0 } * row0{ b1 } d2 += col1{ a2 } * row0{ b0 } d3 += col1{ a2 }
            // * row0{ b1 } (for odd work item indices) d0 += col0{ a1 } * row0{
            // b2 } d1 += col0{ a1 } * row0{ b3 } d2 += col1{ a3 } * row0{ b2 }
            // d3 += col1{ a3 } * row0{ b3 }
            *d[0] +=
                static_cast<भग्न>(ra[r_ind]) * static_cast<भग्न>(rb[c_ind]);
            *d[1] += static_cast<भग्न>(ra[r_ind]) *
                     static_cast<भग्न>(rb[c_ind + 1]);
            *d[2] += static_cast<भग्न>(ra[r_ind + 2]) *
                     static_cast<भग्न>(rb[c_ind]);
            *d[3] += static_cast<भग्न>(ra[r_ind + 2]) *
                     static_cast<भग्न>(rb[c_ind + 1]);

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
                static_cast<भग्न>(ra[r_ind]) * static_cast<भग्न>(rb[c_ind]);
            *d[5] += static_cast<भग्न>(ra[r_ind]) *
                     static_cast<भग्न>(rb[c_ind + 1]);
            *d[6] += static_cast<भग्न>(ra[r_ind + 2]) *
                     static_cast<भग्न>(rb[c_ind]);
            *d[7] += static_cast<भग्न>(ra[r_ind + 2]) *
                     static_cast<भग्न>(rb[c_ind + 1]);
          }
        }
      } अन्यथा अगर constexpr (M == 8 && N == 8 && K == 16) {
        अगर constexpr (std::is_integral_v<ABType>) {
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
          क्रम (पूर्णांक i = 0; i < 4; i++) {
            typename MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_group(sg, a[0], row_load_offset + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_group(sg, b[0], col_load_offset + i + 4);

            स्वतः a = reinterpret_cast<ABType*>(&recv_a);
            स्वतः b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            क्रम (पूर्णांक j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } अन्यथा अगर constexpr (M == 16 && N == 8 && K == 8) {
        अगर constexpr (std::is_floating_point_v<CDType>) {
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
          क्रम (पूर्णांक i = 0; i < 4; i++) {
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

            स्वतः ra = reinterpret_cast<ABType*>(recv_a);
            स्वतः rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            क्रम (पूर्णांक j = 0; j < 2; j++) {
              *d[0] += static_cast<भग्न>(ra[j]) * static_cast<भग्न>(rb[j]);
              *d[1] +=
                  static_cast<भग्न>(ra[j]) * static_cast<भग्न>(rb[j + 2]);
              *d[2] +=
                  static_cast<भग्न>(ra[j + 2]) * static_cast<भग्न>(rb[j]);
              *d[3] +=
                  static_cast<भग्न>(ra[j + 2]) * static_cast<भग्न>(rb[j + 2]);
            }
          }
        }
      } अन्यथा अगर constexpr (M == 16 && N == 8 && K == 16) {
        अगर constexpr (std::is_floating_point_v<CDType>) {
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
          क्रम (पूर्णांक i = 0; i < 4; i++) {
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

            स्वतः ra = reinterpret_cast<ABType*>(recv_a);
            स्वतः rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            क्रम (पूर्णांक j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } अन्यथा अगर constexpr (std::is_integral_v<ABType>) {
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
          क्रम (पूर्णांक i = 0; i < 4; i++) {
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

            स्वतः ra = reinterpret_cast<ABType*>(recv_a);
            स्वतः rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            क्रम (पूर्णांक i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } अन्यथा अगर constexpr (M == 16 && N == 8 && K == 32) {
        अगर constexpr (std::is_integral_v<ABType>) {
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
          क्रम (पूर्णांक i = 0; i < 4; i++) {
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

            स्वतः a = reinterpret_cast<ABType*>(recv_a);
            स्वतः b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            क्रम (पूर्णांक j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          क्रम (पूर्णांक i = 0; i < 4; i++) {
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

            स्वतः a = reinterpret_cast<ABType*>(recv_a);
            स्वतः b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            क्रम (पूर्णांक j = 0; j < 4; j++) {
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

#पूर्ण_अगर // GGML_SYCL_DPCT_HELPER_HPP

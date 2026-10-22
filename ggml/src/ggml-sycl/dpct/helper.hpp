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

#यदि_अपरिभाषितम् GGML_SYCL_DPCT_HELPER_HPP
#परिभाषय GGML_SYCL_DPCT_HELPER_HPP

#अन्तर्भाव <cstdlib>
#अन्तर्भाव <iostream>
#अन्तर्भाव <map>

#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव <sycl/half_type.hpp>
#अन्तर्भाव <oneapi/mkl.hpp>

#यदि defined(__linux__)
#अन्तर्भाव <sys/mman.h>
#नोचेद्यदि defined(_WIN64)
#यदि_अपरिभाषितम् NOMINMAX
#परिभाषय NOMINMAX
#यदि_अन्तः
#अन्तर्भाव <windows.h>
#अन्यथा
#दोषः "Only support Windows and Linux."
#यदि_अन्तः

#यदि defined(__linux__)
#अन्तर्भाव <unistd.h>
#अन्तर्भाव <sys/syscall.h>
#यदि_अन्तः
#यदि defined(_WIN64)
#यदि_अपरिभाषितम् NOMINMAX
#परिभाषय NOMINMAX
#यदि_अन्तः
#अन्तर्भाव <windows.h>
#यदि_अन्तः


#परिभाषय DPCT_COMPATIBILITY_अस्थायी (900)

#यदि defined(_MSC_VER)
#परिभाषय __dpct_align__(n) __declspec(align(n))
#परिभाषय __dpct_inline__ __forceinline
#अन्यथा
#परिभाषय __dpct_align__(n) __attribute__((aligned(n)))
#परिभाषय __dpct_inline__ __inline__ __attribute__((always_inline))
#यदि_अन्तः

#यदि defined(_MSC_VER)
#परिभाषय __dpct_noinline__ __declspec(noinline)
#अन्यथा
#परिभाषय __dpct_noinline__ __attribute__((noinline))
#यदि_अन्तः

#परिभाषय DPCT_अप्रयुक्तम्(x) (शून्य)(x)

[[noreturn]] अन्तःस्थ शून्य _abort(ध्रुव वर्ण * str) {
    std::cerr << str << std::पङ्क्त्यन्तः;
    std::abort();
}

अन्तःस्थ std::string get_device_type_name(ध्रुव sycl::device &Device) {
    स्वचालित DeviceType = Device.get_info<sycl::info::device::device_type>();
    विकल्प (DeviceType) {
    पक्ष sycl::info::device_type::cpu:
        प्रत्यागम "cpu";
    पक्ष sycl::info::device_type::gpu:
        प्रत्यागम "gpu";
    पक्ष sycl::info::device_type::host:
        प्रत्यागम "host";
    पक्ष sycl::info::device_type::accelerator:
        प्रत्यागम "acc";
    स्वभाव:
        प्रत्यागम "unknown";
    }
}

अन्तःस्थ std::string get_device_backend_and_type(ध्रुव sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    प्रत्यागम device_type.str();
}

प्रतिरूपम् <प्रकारनाम Ts> संरचना matrix_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     मानम्_info[2];
    std::int64_t           आकार_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

नामाकाश dpct
{
    प्रकारनिर्देश sycl::queue *queue_ptr;
    प्रकारनिर्देश sycl::event *event_ptr;
    प्रकारनिर्देश वर्ण *device_ptr;
    प्रकारनिर्देश uint8_t byte_t;
    प्रकारनिर्देश sycl::बफर<byte_t> बफर_t;

    /// SYCL default exception handler
    अन्तःस्थ स्वचालित exception_handler = [](sycl::exception_list exceptions)
    {
        प्रति (std::exception_ptr ध्रुव &e : exceptions)
        {
            यत्न
            {
                std::rethrow_exception(e);
            }
            गृहाण (sycl::exception ध्रुव &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::पङ्क्त्यन्तः
                          << e.what() << std::पङ्क्त्यन्तः
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::पङ्क्त्यन्तः;
            }
        }
    };

    गणना error_code
    {
        success = 0,
        default_error = 999
    };

    गणना memcpy_direction
    {
        host_गन्तव्यम्_host,
        host_गन्तव्यम्_device,
        device_गन्तव्यम्_host,
        device_गन्तव्यम्_device,
        automatic
    };

    गणना स्मृतिः_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    गणना वर्ग library_दत्त_t : अचिह्नित वर्ण
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
        library_दत्त_t_आकार
    };

    प्रतिरूपम् <प्रकारनाम T>
    संरचना दत्त_Type_2
    {
        उपयुज्य T2 = T;
    };
    प्रतिरूपम् <प्रकारनाम T>
    संरचना दत्त_Type_2<sycl::vec<T, 2>>
    {
        उपयुज्य T2 = std::complex<T>;
    };

    स्थिर शून्य destroy_event(event_ptr event)
    {
        विलोपय event;
    }

    स्थिर अन्तःस्थ अचिह्नित पूर्णाङ्क get_tid()
    {
#यदि defined(__linux__)
        प्रत्यागम syscall(SYS_gettid);
#नोचेद्यदि defined(_WIN64)
        प्रत्यागम GetCurrentThreadId();
#अन्यथा
#दोषः "Only support Windows and Linux."
#यदि_अन्तः
    }

    नामाकाश detail
    {
        स्थिर शून्य get_संस्करणम्(ध्रुव sycl::device &dev, पूर्णाङ्क &major, पूर्णाङ्क &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::संस्करणम्_2>();
            std::string::size_type i = 0;
            यावत् (i < ver.size()) {
              यदि (अङ्कः_वा(ver[i]))
                भङ्ग;
              i++;
            }
            major = std::stoi(&(ver[i]));
            यावत् (i < ver.size()) {
              यदि (ver[i] == '.')
                भङ्ग;
              i++;
            }
            यदि (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } अन्यथा {
              // c.
              minor = 0;
            }
        }

        प्रतिरूपम् <प्रकारनाम चिह्नकम्, प्रकारनाम T>
        वर्ग generic_error_type
        {
        सार्वजनिक:
            generic_error_type() = स्वभाव;
            generic_error_type(T value) : value{value} {}
            संक्रियकः T() ध्रुव { प्रत्यागम value; }

        गुप्त:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    वर्ग dim3 {
        सार्वजनिक:
        अचिह्नित x, y, z;

        constexpr dim3(अचिह्नित x = 1, अचिह्नित y = 1, अचिह्नित z = 1)
            : x(x), y(y), z(z) {}

        dim3(ध्रुव sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        संक्रियकः sycl::range<3>() ध्रुव { प्रत्यागम sycl::range<3>(z, y, x); }
    }; // namespace dim3

    अन्तःस्थ dim3 संक्रियकः*(ध्रुव dim3 &a, ध्रुव dim3 &b) {
    प्रत्यागम dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    वर्ग pitched_दत्त
    {
    सार्वजनिक:
        pitched_दत्त() : pitched_दत्त(शून्यसूचकः, 0, 0, 0) {}
        pitched_दत्त(शून्य *दत्त, size_t pitch, size_t x, size_t y)
            : दत्त_2(दत्त), _pitch(pitch), _x(x), _y(y) {}

        शून्य *get_दत्त_ptr() { प्रत्यागम दत्त_2; }
        शून्य समुच्चयः_दत्त_ptr(शून्य *दत्त) { दत्त_2 = दत्त; }

        size_t get_pitch() { प्रत्यागम _pitch; }
        शून्य समुच्चयः_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { प्रत्यागम _x; }
        शून्य समुच्चयः_x(size_t x) { _x = x; }

        size_t get_y() { प्रत्यागम _y; }
        शून्य समुच्चयः_y(size_t y) { _y = y; }

    गुप्त:
        शून्य *दत्त_2;
        size_t _pitch, _x, _y;
    };

    वर्ग device_info
    {
    सार्वजनिक:
        // get interface
        ध्रुव वर्ण *get_name() ध्रुव { प्रत्यागम _name; }
        वर्ण *get_name() { प्रत्यागम _name; }
        प्रतिरूपम् <प्रकारनाम WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, पूर्णाङ्क *>,
                                   पूर्णाङ्क> = 0>
        स्वचालित get_max_work_item_sizes() ध्रुव
        {
            यदि constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                प्रत्यागम sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            अन्यथा
            {
                प्रत्यागम _max_work_item_sizes_i;
            }
        }
        प्रतिरूपम् <प्रकारनाम WorkItemSizesTy = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<WorkItemSizesTy, sycl::range<3>> ||
                                       std::is_same_v<WorkItemSizesTy, पूर्णाङ्क *>,
                                   पूर्णाङ्क> = 0>
        स्वचालित get_max_work_item_sizes()
        {
            यदि constexpr (std::is_same_v<WorkItemSizesTy, sycl::range<3>>)
                प्रत्यागम sycl::range<3>(_max_work_item_sizes_i[0],
                                      _max_work_item_sizes_i[1],
                                      _max_work_item_sizes_i[2]);
            अन्यथा
            {
                प्रत्यागम _max_work_item_sizes_i;
            }
        }
        तार्किक get_host_unified_स्मृतिः() ध्रुव { प्रत्यागम host_unified_स्मृतिः; }
        पूर्णाङ्क get_major_संस्करणम्() ध्रुव { प्रत्यागम _major; }
        पूर्णाङ्क get_minor_संस्करणम्() ध्रुव { प्रत्यागम _minor; }
        पूर्णाङ्क get_integrated() ध्रुव { प्रत्यागम _integrated; }
        पूर्णाङ्क get_max_clock_frequency() ध्रुव { प्रत्यागम _frequency; }
        पूर्णाङ्क get_max_compute_units() ध्रुव { प्रत्यागम _max_compute_units; }
        पूर्णाङ्क get_max_work_गणः_आकार() ध्रुव { प्रत्यागम max_work_गणः_आकार_2; }
        पूर्णाङ्क get_max_sub_गणः_आकार() ध्रुव { प्रत्यागम max_sub_गणः_आकार; }
        पूर्णाङ्क get_max_work_वस्तूनि_प्रत्येकम्_compute_unit() ध्रुव
        {
            प्रत्यागम max_work_वस्तूनि_प्रत्येकम्_compute_unit;
        }
        पूर्णाङ्क get_max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः() ध्रुव
        {
            प्रत्यागम max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः;
        }
        प्रतिरूपम् <प्रकारनाम NDRange_आकार_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_आकार_Ty, size_t *> ||
                                       std::is_same_v<NDRange_आकार_Ty, पूर्णाङ्क *>,
                                   पूर्णाङ्क> = 0>
        स्वचालित get_max_nd_range_आकार() ध्रुव
        {
            यदि constexpr (std::is_same_v<NDRange_आकार_Ty, size_t *>)
                प्रत्यागम max_nd_range_आकार;
            अन्यथा
                प्रत्यागम max_nd_range_आकार_i;
        }
        प्रतिरूपम् <प्रकारनाम NDRange_आकार_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_आकार_Ty, size_t *> ||
                                       std::is_same_v<NDRange_आकार_Ty, पूर्णाङ्क *>,
                                   पूर्णाङ्क> = 0>
        स्वचालित get_max_nd_range_आकार()
        {
            यदि constexpr (std::is_same_v<NDRange_आकार_Ty, size_t *>)
                प्रत्यागम max_nd_range_आकार;
            अन्यथा
                प्रत्यागम max_nd_range_आकार_i;
        }
        size_t get_global_mem_आकार() ध्रुव { प्रत्यागम global_mem_आकार_2; }
        size_t get_local_mem_आकार() ध्रुव { प्रत्यागम local_mem_आकार; }
        size_t get_max_mem_alloc_आकार() ध्रुव { प्रत्यागम max_mem_alloc_आकार; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        अचिह्नित पूर्णाङ्क get_स्मृतिः_clock_दरः() ध्रुव { प्रत्यागम स्मृतिः_clock_दरः; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        अचिह्नित पूर्णाङ्क get_स्मृतिः_bus_विस्तारः() ध्रुव { प्रत्यागम स्मृतिः_bus_विस्तारः; }
        uint32_t get_device_id() ध्रुव { प्रत्यागम _device_id; }
        std::array<अचिह्नित वर्ण, 16> get_uuid() ध्रुव { प्रत्यागम _uuid; }
        /// Returns global memory cache size in bytes.
        अचिह्नित पूर्णाङ्क get_global_mem_cache_आकार() ध्रुव
        {
            प्रत्यागम global_mem_cache_आकार;
        }

        // set interface
        शून्य समुच्चयः_name(ध्रुव वर्ण *name)
        {
            size_t दीर्घता = वाक्यदैर्घ्यम्(name);
            यदि (दीर्घता < 256)
            {
                std::स्मृतिं_प्रतिलिख(_name, name, दीर्घता + 1);
            }
            अन्यथा
            {
                std::स्मृतिं_प्रतिलिख(_name, name, 255);
                _name[255] = '\0';
            }
        }
        शून्य समुच्चयः_max_work_item_sizes(ध्रुव sycl::range<3> max_work_item_sizes)
        {
            प्रति (पूर्णाङ्क i = 0; i < 3; ++i)
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
        }
        [[deprecated]] शून्य
        समुच्चयः_max_work_item_sizes(ध्रुव sycl::id<3> max_work_item_sizes)
        {
            प्रति (पूर्णाङ्क i = 0; i < 3; ++i)
            {
                _max_work_item_sizes_i[i] = max_work_item_sizes[i];
            }
        }
        शून्य समुच्चयः_host_unified_स्मृतिः(तार्किक host_unified_स्मृतिः_2)
        {
            host_unified_स्मृतिः = host_unified_स्मृतिः_2;
        }
        शून्य समुच्चयः_major_संस्करणम्(पूर्णाङ्क major) { _major = major; }
        शून्य समुच्चयः_minor_संस्करणम्(पूर्णाङ्क minor) { _minor = minor; }
        शून्य समुच्चयः_integrated(पूर्णाङ्क integrated) { _integrated = integrated; }
        शून्य समुच्चयः_max_clock_frequency(पूर्णाङ्क frequency) { _frequency = frequency; }
        शून्य समुच्चयः_max_compute_units(पूर्णाङ्क max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        शून्य समुच्चयः_global_mem_आकार(size_t global_mem_आकार)
        {
            global_mem_आकार_2 = global_mem_आकार;
        }
        शून्य समुच्चयः_local_mem_आकार(size_t local_mem_आकार_2)
        {
            local_mem_आकार = local_mem_आकार_2;
        }
        शून्य समुच्चयः_max_mem_alloc_आकार(size_t max_mem_alloc_आकार_2)
        {
            max_mem_alloc_आकार = max_mem_alloc_आकार_2;
        }
        शून्य समुच्चयः_max_work_गणः_आकार(पूर्णाङ्क max_work_गणः_आकार)
        {
            max_work_गणः_आकार_2 = max_work_गणः_आकार;
        }
        शून्य समुच्चयः_max_sub_गणः_आकार(पूर्णाङ्क max_sub_गणः_आकार_2)
        {
            max_sub_गणः_आकार = max_sub_गणः_आकार_2;
        }
        शून्य
        समुच्चयः_max_work_वस्तूनि_प्रत्येकम्_compute_unit(पूर्णाङ्क max_work_वस्तूनि_प्रत्येकम्_compute_unit_2)
        {
            max_work_वस्तूनि_प्रत्येकम्_compute_unit = max_work_वस्तूनि_प्रत्येकम्_compute_unit_2;
        }
        शून्य समुच्चयः_max_nd_range_आकार(पूर्णाङ्क max_nd_range_आकार_2[])
        {
            प्रति (पूर्णाङ्क i = 0; i < 3; i++)
            {
                max_nd_range_आकार[i] = max_nd_range_आकार_2[i];
                max_nd_range_आकार_i[i] = max_nd_range_आकार_2[i];
            }
        }
        शून्य समुच्चयः_स्मृतिः_clock_दरः(अचिह्नित पूर्णाङ्क स्मृतिः_clock_दरः_2)
        {
            स्मृतिः_clock_दरः = स्मृतिः_clock_दरः_2;
        }
        शून्य समुच्चयः_स्मृतिः_bus_विस्तारः(अचिह्नित पूर्णाङ्क स्मृतिः_bus_विस्तारः_2)
        {
            स्मृतिः_bus_विस्तारः = स्मृतिः_bus_विस्तारः_2;
        }
        शून्य
        समुच्चयः_max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः(पूर्णाङ्क max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः_2)
        {
            max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः = max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः_2;
        }
        शून्य समुच्चयः_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        शून्य समुच्चयः_uuid(std::array<अचिह्नित वर्ण, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        शून्य समुच्चयः_global_mem_cache_आकार(अचिह्नित पूर्णाङ्क global_mem_cache_आकार_2)
        {
            global_mem_cache_आकार = global_mem_cache_आकार_2;
        }

    गुप्त:
        वर्ण _name[256];
        पूर्णाङ्क _max_work_item_sizes_i[3];
        तार्किक host_unified_स्मृतिः = असत्य;
        पूर्णाङ्क _major;
        पूर्णाङ्क _minor;
        पूर्णाङ्क _integrated = 0;
        पूर्णाङ्क _frequency;
        // Set estimated value 3200000 kHz as default value.
        अचिह्नित पूर्णाङ्क स्मृतिः_clock_दरः = 3200000;
        // Set estimated value 64 bits as default value.
        अचिह्नित पूर्णाङ्क स्मृतिः_bus_विस्तारः = 64;
        अचिह्नित पूर्णाङ्क global_mem_cache_आकार;
        पूर्णाङ्क _max_compute_units;
        पूर्णाङ्क max_work_गणः_आकार_2;
        पूर्णाङ्क max_sub_गणः_आकार;
        पूर्णाङ्क max_work_वस्तूनि_प्रत्येकम्_compute_unit;
        पूर्णाङ्क max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः;
        size_t global_mem_आकार_2;
        size_t local_mem_आकार;
        size_t max_mem_alloc_आकार;
        size_t max_nd_range_आकार[3];
        पूर्णाङ्क max_nd_range_आकार_i[3];
        uint32_t _device_id;
        std::array<अचिह्नित वर्ण, 16> _uuid;
    };

    स्थिर पूर्णाङ्क get_major_संस्करणम्(ध्रुव sycl::device &dev)
    {
        पूर्णाङ्क major, minor;
        detail::get_संस्करणम्(dev, major, minor);
        प्रत्यागम major;
    }

    स्थिर पूर्णाङ्क get_minor_संस्करणम्(ध्रुव sycl::device &dev)
    {
        पूर्णाङ्क major, minor;
        detail::get_संस्करणम्(dev, major, minor);
        प्रत्यागम minor;
    }

    स्थिर शून्य get_device_info(device_info &बहिर्गमनम्, ध्रुव sycl::device &dev)
    {
        device_info prop;
        prop.समुच्चयः_name(dev.get_info<sycl::info::device::name>().c_str());

        पूर्णाङ्क major, minor;
        detail::get_संस्करणम्(dev, major, minor);
        prop.समुच्चयः_major_संस्करणम्(major);
        prop.समुच्चयः_minor_संस्करणम्(minor);

        prop.समुच्चयः_max_work_item_sizes(
#यदि (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_item_sizes>());
#अन्यथा
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_item_sizes<3>>());
#यदि_अन्तः
        prop.समुच्चयः_host_unified_स्मृतिः(dev.has(sycl::aspect::usm_host_allocations));

        prop.समुच्चयः_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.समुच्चयः_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.समुच्चयः_max_work_गणः_आकार(
            dev.get_info<sycl::info::device::max_work_गणः_आकार>());
        prop.समुच्चयः_global_mem_आकार(dev.get_info<sycl::info::device::global_mem_आकार>());
        prop.समुच्चयः_local_mem_आकार(dev.get_info<sycl::info::device::local_mem_आकार_2>());
        prop.समुच्चयः_max_mem_alloc_आकार(dev.get_info<sycl::info::device::max_mem_alloc_आकार_2>());

#यदि (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        यदि (dev.has(sycl::aspect::ext_intel_स्मृतिः_clock_दरः))
        {
            अचिह्नित पूर्णाङ्क tmp =
                dev.get_info<sycl::ext::intel::info::device::स्मृतिः_clock_दरः_2>();
            यदि (tmp != 0)
                prop.समुच्चयः_स्मृतिः_clock_दरः(1000 * tmp);
        }
        यदि (dev.has(sycl::aspect::ext_intel_स्मृतिः_bus_विस्तारः))
        {
            prop.समुच्चयः_स्मृतिः_bus_विस्तारः(
                dev.get_info<sycl::ext::intel::info::device::स्मृतिः_bus_विस्तारः_2>());
        }
        यदि (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.समुच्चयः_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        यदि (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.समुच्चयः_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#नोचेद्यदि defined(_MSC_VER) && !defined(__clang__)
#निर्देशः सन्देशः("get_device_info: querying स्मृतिः_clock_दरः_2 and \
        स्मृतिः_bus_विस्तारः_2 are not supported by the compiler प्रयुक्तम्. \
        Use 3200000 kHz as स्मृतिः_clock_दरः_2 स्वभाव value. \
        Use 64 द्वयङ्काः as स्मृतिः_bus_विस्तारः_2 स्वभाव value.")
#अन्यथा
#warning "get_device_info: querying स्मृतिः_clock_दरः_2 and \
        स्मृतिः_bus_विस्तारः_2 are not supported by the compiler प्रयुक्तम्. \
        Use 3200000 kHz as स्मृतिः_clock_दरः_2 स्वभाव value. \
        Use 64 द्वयङ्काः as स्मृतिः_bus_विस्तारः_2 स्वभाव value."
#यदि_अन्तः

        size_t max_sub_गणः_आकार_2 = 1;
        std::vector<size_t> sub_गणः_sizes =
            dev.get_info<sycl::info::device::sub_गणः_sizes>();

        प्रति (ध्रुव स्वचालित &sub_गणः_आकार : sub_गणः_sizes)
        {
            यदि (max_sub_गणः_आकार_2 < sub_गणः_आकार)
                max_sub_गणः_आकार_2 = sub_गणः_आकार;
        }

        prop.समुच्चयः_max_sub_गणः_आकार(max_sub_गणः_आकार_2);

        prop.समुच्चयः_max_work_वस्तूनि_प्रत्येकम्_compute_unit(
            dev.get_info<sycl::info::device::max_work_गणः_आकार>());
        पूर्णाङ्क max_nd_range_आकार_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.समुच्चयः_max_nd_range_आकार(max_nd_range_आकार_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.समुच्चयः_max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः(65536);

        prop.समुच्चयः_global_mem_cache_आकार(
            dev.get_info<sycl::info::device::global_mem_cache_आकार_2>());
        बहिर्गमनम् = prop;
    }

    /// dpct device extension
    वर्ग device_ext : सार्वजनिक sycl::device {
      प्रकारनिर्देश std::mutex mutex_type;

     सार्वजनिक:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        clear_queues();
      }
      device_ext(ध्रुव sycl::device &आधारः) : sycl::device(आधारः) {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        init_queues();
      }

      पूर्णाङ्क is_native_atomic_supported() { प्रत्यागम 0; }
      पूर्णाङ्क get_major_संस्करणम्() ध्रुव { प्रत्यागम dpct::get_major_संस्करणम्(*एतत्); }

      पूर्णाङ्क get_minor_संस्करणम्() ध्रुव { प्रत्यागम dpct::get_minor_संस्करणम्(*एतत्); }

      पूर्णाङ्क get_max_compute_units() ध्रुव {
        प्रत्यागम get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      पूर्णाङ्क get_max_clock_frequency() ध्रुव {
        प्रत्यागम get_device_info().get_max_clock_frequency();
      }

      पूर्णाङ्क get_integrated() ध्रुव { प्रत्यागम get_device_info().get_integrated(); }

      पूर्णाङ्क get_max_sub_गणः_आकार() ध्रुव {
        प्रत्यागम get_device_info().get_max_sub_गणः_आकार();
      }

      पूर्णाङ्क get_max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः() ध्रुव {
        प्रत्यागम get_device_info().get_max_पञ्जिकाकोशः_आकार_प्रत्येकम्_work_गणः();
      }

      पूर्णाङ्क get_max_work_गणः_आकार() ध्रुव {
        प्रत्यागम get_device_info().get_max_work_गणः_आकार();
      }

      पूर्णाङ्क get_mem_आधारः_addr_align() ध्रुव {
        प्रत्यागम get_info<sycl::info::device::mem_आधारः_addr_align>();
      }

      size_t get_global_mem_आकार() ध्रुव {
        प्रत्यागम get_device_info().get_global_mem_आकार();
      }

      size_t get_max_mem_alloc_आकार() ध्रुव {
        प्रत्यागम get_device_info().get_max_mem_alloc_आकार();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      शून्य get_स्मृतिः_info(size_t &free_स्मृतिः, size_t &योग_स्मृतिः) {
        योग_स्मृतिः = get_device_info().get_global_mem_आकार();
        ध्रुव वर्ण *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#यदि (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        यदि (!has(sycl::aspect::ext_intel_free_स्मृतिः)) {
          std::cerr << warning_info << std::पङ्क्त्यन्तः;
          free_स्मृतिः = योग_स्मृतिः;
        } अन्यथा {
          free_स्मृतिः = get_info<sycl::ext::intel::info::device::free_स्मृतिः>();
        }
#अन्यथा
        std::cerr << warning_info << std::पङ्क्त्यन्तः;
        free_स्मृतिः = योग_स्मृतिः;
#यदि defined(_MSC_VER) && !defined(__clang__)
#निर्देशः सन्देशः("Querying the number of bytes of free memory is not supported")
#अन्यथा
#warning "Querying the number of bytes of free memory is not supported"
#यदि_अन्तः
#यदि_अन्तः
      }

      शून्य get_device_info(device_info &बहिर्गमनम्) ध्रुव {
        dpct::get_device_info(बहिर्गमनम्, *एतत्);
      }

      device_info get_device_info() ध्रुव {
        device_info prop;
        dpct::get_device_info(prop, *एतत्);
        प्रत्यागम prop;
      }

      शून्य reset() {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &अन्तर्_क्रमः_queue() { प्रत्यागम q_अन्तर्_क्रमः; }

      sycl::queue &बहिर्गमनम्_सम्बन्धि_क्रमः_queue() { प्रत्यागम q_बहिर्गमनम्_सम्बन्धि_क्रमः; }

      sycl::queue &default_queue() { प्रत्यागम अन्तर्_क्रमः_queue(); }

      शून्य queues_wait_and_throw() {
        std::unique_lock<mutex_type> तालकम्(m_mutex);
        तालकम्.unlock();
        प्रति (स्वचालित &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        तालकम्.तालकम्();
      }

      sycl::queue create_queue(तार्किक enable_exception_handler = असत्य) {
        प्रत्यागम create_अन्तर्_क्रमः_queue(enable_exception_handler);
      }

      sycl::queue create_queue(sycl::device device,
                               तार्किक enable_exception_handler = असत्य) {
        प्रत्यागम create_अन्तर्_क्रमः_queue(device, enable_exception_handler);
      }

      sycl::queue create_अन्तर्_क्रमः_queue(तार्किक enable_exception_handler = असत्य) {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        प्रत्यागम create_queue_impl(enable_exception_handler,
                                 sycl::property::queue::अन्तर्_क्रमः());
      }

      sycl::queue create_अन्तर्_क्रमः_queue(sycl::device device,
                                        तार्किक enable_exception_handler = असत्य) {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        प्रत्यागम create_queue_impl(device, enable_exception_handler,
                                 sycl::property::queue::अन्तर्_क्रमः());
      }

      sycl::queue create_बहिर्गमनम्_सम्बन्धि_क्रमः_queue(
          तार्किक enable_exception_handler = असत्य) {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        प्रत्यागम create_queue_impl(enable_exception_handler);
      }

      शून्य destroy_queue(sycl::queue queue) {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ध्रुव sycl::queue &q) -> तार्किक
                                    {
                                        प्रत्यागम q == queue;
                                    }),
                    _queues.end());
      }
      शून्य समुच्चयः_saved_queue(sycl::queue q) {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        _saved_queue = q;
      }
      sycl::queue get_saved_queue() ध्रुव {
        std::lock_guard<mutex_type> तालकम्(m_mutex);
        प्रत्यागम _saved_queue;
      }

     गुप्त:
      शून्य clear_queues() { _queues.clear(); }

      शून्य init_queues() {
        q_अन्तर्_क्रमः =
            create_queue_impl(सत्य, sycl::property::queue::अन्तर्_क्रमः());
        q_बहिर्गमनम्_सम्बन्धि_क्रमः = create_queue_impl(सत्य);
        _saved_queue = default_queue();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      प्रतिरूपम् <वर्ग... Properties>
      sycl::queue create_queue_impl(तार्किक enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        यदि (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *एतत्, eh,
            sycl::property_list(
#यदि_परिभाषितम् DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#यदि_अन्तः
                properties...)));

        प्रत्यागम _queues.back();
      }

      प्रतिरूपम् <वर्ग... Properties>
      sycl::queue create_queue_impl(sycl::device device,
                                    तार्किक enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        यदि (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#यदि_परिभाषितम् DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#यदि_अन्तः
                            properties...)));

        प्रत्यागम _queues.back();
      }

      शून्य get_संस्करणम्(पूर्णाङ्क &major, पूर्णाङ्क &minor) ध्रुव {
        detail::get_संस्करणम्(*एतत्, major, minor);
      }
      sycl::queue q_अन्तर्_क्रमः, q_बहिर्गमनम्_सम्बन्धि_क्रमः;
      sycl::queue _saved_queue;
      std::vector<sycl::queue> _queues;
      परिवर्तनीय mutex_type m_mutex;
    };


    /// device manager
    वर्ग dev_mgr
    {
    सार्वजनिक:
        device_ext &current_device()
        {
            अचिह्नित पूर्णाङ्क dev_id = current_device_id();
            check_id(dev_id);
            प्रत्यागम *_devs[dev_id];
        }
        device_ext &cpu_device() ध्रुव
        {
            std::lock_guard<std::recursive_mutex> तालकम्(m_mutex);
            यदि (_cpu_device == -1)
            {
                क्षिप std::runtime_error("no valid cpu device");
            }
            अन्यथा
            {
                प्रत्यागम *_devs[_cpu_device];
            }
        }
        device_ext &get_device(अचिह्नित पूर्णाङ्क id) ध्रुव
        {
            std::lock_guard<std::recursive_mutex> तालकम्(m_mutex);
            check_id(id);
            प्रत्यागम *_devs[id];
        }
        अचिह्नित पूर्णाङ्क current_device_id() ध्रुव
        {
            std::lock_guard<std::recursive_mutex> तालकम्(m_mutex);
            स्वचालित it = _thread2dev_map.find(get_tid());
            यदि (it != _thread2dev_map.end())
                प्रत्यागम it->second;
            प्रत्यागम DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        शून्य select_device(अचिह्नित पूर्णाङ्क id)
        {
            std::lock_guard<std::recursive_mutex> तालकम्(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        अचिह्नित पूर्णाङ्क device_गणनम्() { प्रत्यागम _devs.size(); }

        अचिह्नित पूर्णाङ्क get_device_id(ध्रुव sycl::device &dev)
        {
            अचिह्नित पूर्णाङ्क id = 0;
            प्रति (स्वचालित &dev_item : _devs)
            {
                यदि (*dev_item == dev)
                {
                    प्रत्यागम id;
                }
                id++;
            }
            प्रत्यागम -1;
        }

        अन्तःस्थ std::string get_preferred_gpu_platform_name() {
            std::string परिणाम;

            std::string filter = "";
            वर्ण* env = getenv("ONEAPI_DEVICE_SELECTOR");
            यदि (env) {
                यदि (std::वाक्ये_अन्विष(env, "level_zero")) {
                    filter = "level-zero";
                }
                अन्यथा यदि (std::वाक्ये_अन्विष(env, "opencl")) {
                    filter = "opencl";
                }
                अन्यथा यदि (std::वाक्ये_अन्विष(env, "cuda")) {
                    filter = "cuda";
                }
                अन्यथा यदि (std::वाक्ये_अन्विष(env, "hip")) {
                    filter = "hip";
                }
                अन्यथा {
                    क्षिप std::runtime_error("invalid device filter: " + std::string(env));
                }
            } अन्यथा {
                स्वचालित default_device = sycl::device(sycl::default_selector_v);
                स्वचालित default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                यदि (std::वाक्ये_अन्विष(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                अन्यथा यदि (std::वाक्ये_अन्विष(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                अन्यथा यदि (std::वाक्ये_अन्विष(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            स्वचालित platform_list = sycl::platform::get_platforms();

            प्रति (ध्रुव स्वचालित& platform : platform_list) {
                स्वचालित devices = platform.get_devices();
                स्वचालित gpu_dev = std::find_if(devices.begin(), devices.end(), [](ध्रुव sycl::device& d) {
                    प्रत्यागम d.is_gpu();
                });

                यदि (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    अनुवर्त;
                }

                स्वचालित platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_नीचः_case;
                platform_name_नीचः_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_नीचः_case.begin(), ::tolower);

                यदि (platform_name_नीचः_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    अनुवर्त;
                }

                परिणाम = platform_name;
            }

            यदि (परिणाम.रिक्तानि())
                क्षिप std::runtime_error("can not find preferred GPU platform");

            प्रत्यागम परिणाम;
        }

        प्रतिरूपम् <वर्ग DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<पूर्णाङ्क, DeviceSelector, ध्रुव sycl::device &>>
        select_device(ध्रुव DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            अचिह्नित पूर्णाङ्क selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        स्थिर dev_mgr &instance()
        {
            स्थिर dev_mgr d_m;
            प्रत्यागम d_m;
        }
        dev_mgr(ध्रुव dev_mgr &) = विलोपय;
        dev_mgr &संक्रियकः=(ध्रुव dev_mgr &) = विलोपय;
        dev_mgr(dev_mgr &&) = विलोपय;
        dev_mgr &संक्रियकः=(dev_mgr &&) = विलोपय;

    गुप्त:
        परिवर्तनीय std::recursive_mutex m_mutex;
        स्थिर तार्किक compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            यदि(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) प्रत्यागम सत्य;
            यदि(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) प्रत्यागम असत्य;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            प्रत्यागम prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        स्थिर पूर्णाङ्क convert_backend_सूचकांक(std::string & backend) {
            यदि (backend == "ext_oneapi_level_zero:gpu") प्रत्यागम 0;
            यदि (backend == "opencl:gpu") प्रत्यागम 1;
            यदि (backend == "ext_oneapi_cuda:gpu") प्रत्यागम 2;
            यदि (backend == "ext_oneapi_hip:gpu") प्रत्यागम 3;
            यदि (backend == "opencl:cpu") प्रत्यागम 4;
            यदि (backend == "opencl:acc") प्रत्यागम 5;
            प्रारूप_लिख("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        स्थिर तार्किक compare_backend(std::string &backend1, std::string &backend2) {
            प्रत्यागम convert_backend_सूचकांक(backend1) < convert_backend_सूचकांक(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            यदि (default_device.is_cpu())
                _cpu_device = 0;

            स्वचालित Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            स्वचालित preferred_platform_name = get_preferred_gpu_platform_name();

            यावत् (!Platforms.रिक्तानि()) {
                स्वचालित Platform = Platforms.back();
                Platforms.pop_back();
                स्वचालित platform_name = Platform.get_info<sycl::info::platform::name>();
                यदि (platform_name.compare(preferred_platform_name) != 0) {
                    अनुवर्त;
                }
                स्वचालित devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                प्रति (ध्रुव स्वचालित &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> कुञ्चिकाः;
            प्रति(स्वचालित it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                कुञ्चिकाः.push_back(it->first);
            }
            std::sort(कुञ्चिकाः.begin(), कुञ्चिकाः.end(), compare_backend);

            प्रति (स्वचालित &कुञ्चिका : कुञ्चिकाः) {
                std::vector<sycl::device> devs = backend_devices[कुञ्चिका];
                std::sort(devs.begin(), devs.end(), compare_dev);
                प्रति (ध्रुव स्वचालित &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            प्रति (स्वचालित &dev : sycl_all_devs)
            {
                यदि (dev == default_device)
                {
                    अनुवर्त;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                यदि (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        शून्य check_id(अचिह्नित पूर्णाङ्क id) ध्रुव
        {
            यदि (id >= _devs.size())
            {
                क्षिप std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ध्रुव अचिह्नित पूर्णाङ्क DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<अचिह्नित पूर्णाङ्क, अचिह्नित पूर्णाङ्क> _thread2dev_map;
        पूर्णाङ्क _cpu_device = -1;
    };

    स्थिर अन्तःस्थ sycl::queue &get_default_queue()
    {
        प्रत्यागम dev_mgr::instance().current_device().default_queue();
    }

    नामाकाश detail
    {
        गणना वर्ग pointer_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        स्थिर pointer_access_attribute get_pointer_attribute(sycl::queue &q,
                                                              ध्रुव शून्य *ptr)
        {
            विकल्प (sycl::get_pointer_type(ptr, q.get_context()))
            {
            पक्ष sycl::usm::alloc::unknown:
                प्रत्यागम pointer_access_attribute::host_only;
            पक्ष sycl::usm::alloc::device:
                प्रत्यागम pointer_access_attribute::device_only;
            पक्ष sycl::usm::alloc::shared:
            पक्ष sycl::usm::alloc::host:
                प्रत्यागम pointer_access_attribute::host_device;
            }
        }

        प्रतिरूपम् <प्रकारनाम ArgT>
        अन्तःस्थ constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((अचिह्नित वर्ण)library_दत्त_t::library_दत्त_t_आकार <=
                              std::numeric_limits<अचिह्नित वर्ण>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_दत्त_t>, "Unsupported ArgT");
            प्रत्यागम (std::uint64_t)Val;
        }

        प्रतिरूपम् <प्रकारनाम FirstT, प्रकारनाम... RestT>
        अन्तःस्थ constexpr std::uint64_t get_type_combination_id(FirstT FirstVal,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_दत्त_t::library_दत्त_t_आकार <=
                              std::numeric_limits<अचिह्नित वर्ण>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(परिमाण...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<FirstT, library_दत्त_t>, "Unsupported FirstT");
            प्रत्यागम get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)FirstVal);
        }

        वर्ग mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#यदि defined(__linux__)
                mapped_स्थानाङ्कः_space =
                    (byte_t *)mmap(शून्यसूचकः, mapped_region_आकार, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#नोचेद्यदि defined(_WIN64)
                mapped_स्थानाङ्कः_space = (byte_t *)VirtualAlloc(
                    रिक्त,               // NULL specified as the base address parameter
                    mapped_region_आकार, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    PAGE_NOACCESS);     // Protection = no access
#अन्यथा
#दोषः "Only support Windows and Linux."
#यदि_अन्तः
                अग्र_free = mapped_स्थानाङ्कः_space;
            }

        सार्वजनिक:
            उपयुज्य बफर_id_t = पूर्णाङ्क;

            संरचना आवण्टनम्
            {
                बफर_t बफर;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#यदि defined(__linux__)
                munmap(mapped_स्थानाङ्कः_space, mapped_region_आकार);
#नोचेद्यदि defined(_WIN64)
                VirtualFree(mapped_स्थानाङ्कः_space, 0, MEM_RELEASE);
#अन्यथा
#दोषः "Only support Windows and Linux."
#यदि_अन्तः
            }

            mem_mgr(ध्रुव mem_mgr &) = विलोपय;
            mem_mgr &संक्रियकः=(ध्रुव mem_mgr &) = विलोपय;
            mem_mgr(mem_mgr &&) = विलोपय;
            mem_mgr &संक्रियकः=(mem_mgr &&) = विलोपय;

            /// Allocate
            शून्य *mem_alloc(size_t size)
            {
                यदि (!size)
                    प्रत्यागम शून्यसूचकः;
                std::lock_guard<std::mutex> तालकम्(m_mutex);
                यदि (अग्र_free + size > mapped_स्थानाङ्कः_space + mapped_region_आकार)
                {
                    क्षिप std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                //आवण्टनम्
                sycl::range<1> r(size);
                बफर_t buf(r);
                आवण्टनम् A{buf, अग्र_free, size};
                // Map allocation to device pointer
                शून्य *परिणाम = अग्र_free;
                m_map.emplace(अग्र_free + size, A);
                // Update pointer to the next free space.
                अग्र_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                प्रत्यागम परिणाम;
            }

            /// Deallocate
            शून्य mem_free(ध्रुव शून्य *ptr)
            {
                यदि (!ptr)
                    प्रत्यागम;
                std::lock_guard<std::mutex> तालकम्(m_mutex);
                स्वचालित it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            आवण्टनम् translate_ptr(ध्रुव शून्य *ptr)
            {
                std::lock_guard<std::mutex> तालकम्(m_mutex);
                स्वचालित it = get_map_iterator(ptr);
                प्रत्यागम it->second;
            }

            /// Check if the pointer represents device pointer or not.
            तार्किक is_device_ptr(ध्रुव शून्य *ptr) ध्रुव
            {
                std::lock_guard<std::mutex> तालकम्(m_mutex);
                प्रत्यागम (mapped_स्थानाङ्कः_space <= ptr) &&
                       (ptr < mapped_स्थानाङ्कः_space + mapped_region_आकार);
            }

            /// Returns the instance of memory manager singleton.
            स्थिर mem_mgr &instance()
            {
                स्थिर mem_mgr m;
                प्रत्यागम m;
            }

        गुप्त:
            std::map<byte_t *, आवण्टनम्> m_map;
            परिवर्तनीय std::mutex m_mutex;
            byte_t *mapped_स्थानाङ्कः_space;
            byte_t *अग्र_free;
            ध्रुव size_t mapped_region_आकार = 128ull * 1024 * 1024 * 1024;
            ध्रुव size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ध्रुव size_t extra_padding = 0;

            std::map<byte_t *, आवण्टनम्>::iterator get_map_iterator(ध्रुव शून्य *ptr)
            {
                स्वचालित it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ध्रुव byte_t *>(ptr)));
                यदि (it == m_map.end())
                {
                    // Not a virtual pointer.
                    क्षिप std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ध्रुव आवण्टनम् &alloc = it->second;
                यदि (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    क्षिप std::runtime_error("invalid virtual pointer");
                }
                प्रत्यागम it;
            }
        };

        प्रतिरूपम् <वर्ग T, स्मृतिः_region स्मृतिः_2, size_t Dimension>
        वर्ग accessor;
        प्रतिरूपम् <स्मृतिः_region स्मृतिः_2, वर्ग T = byte_t>
        वर्ग स्मृतिः_traits
        {
        सार्वजनिक:
            स्थिर constexpr sycl::access::target target =
                sycl::access::target::device;
            स्थिर constexpr sycl::access_बहुलकम् mode =
                (स्मृतिः_2 == constant) ? sycl::access_बहुलकम्::पठ
                                     : sycl::access_बहुलकम्::read_write;
            स्थिर constexpr size_t type_आकार = परिमाण(T);
            उपयुज्य element_t =
                प्रकारनाम std::conditional<स्मृतिः_2 == constant, ध्रुव T, T>::type;
            उपयुज्य मानम्_t = प्रकारनाम std::remove_cv<T>::type;
            प्रतिरूपम् <size_t Dimension = 1>
            उपयुज्य accessor_t = प्रकारनाम std::conditional<
                स्मृतिः_2 == local, sycl::local_accessor<मानम्_t, Dimension>,
                sycl::accessor<T, Dimension, mode, target>>::type;
            उपयुज्य pointer_t = T *;
        };

        स्थिर अन्तःस्थ शून्य *dpct_malloc(size_t size, sycl::queue &q)
        {
            प्रत्यागम sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#परिभाषय PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        स्थिर अन्तःस्थ शून्य *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            प्रत्यागम dpct_malloc(pitch * y * z, q);
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
        प्रतिरूपम् <प्रकारनाम मानम्_T>
        स्थिर अन्तःस्थ sycl::event dpct_memset(sycl::queue &q, शून्य *dev_ptr,
                                              मानम्_T value, size_t size)
        {
            प्रत्यागम q.fill(dev_ptr, value, size);
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
        प्रतिरूपम् <प्रकारनाम मानम्_T>
        स्थिर अन्तःस्थ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_दत्त दत्त, मानम्_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t खण्डिका_2 = दत्त.get_pitch() * दत्त.get_y();
            अचिह्नित वर्ण *दत्त_surface = (अचिह्नित वर्ण *)दत्त.get_दत्त_ptr();
            प्रति (size_t z = 0; z < size.get(2); ++z)
            {
                अचिह्नित वर्ण *दत्त_ptr = दत्त_surface;
                प्रति (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, दत्त_ptr, value, size.get(0)));
                    दत्त_ptr += दत्त.get_pitch();
                }
                दत्त_surface += खण्डिका_2;
            }
            प्रत्यागम event_list;
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
        प्रतिरूपम् <प्रकारनाम मानम्_T>
        स्थिर अन्तःस्थ std::vector<sycl::event>
        dpct_memset(sycl::queue &q, शून्य *ptr, size_t pitch, मानम्_T val, size_t x,
                    size_t y)
        {
            प्रत्यागम dpct_memset(q, pitched_दत्त(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        स्थिर memcpy_direction deduce_memcpy_direction(sycl::queue &q, शून्य *गन्तव्यम्_ptr,
                                                        ध्रुव शून्य *from_ptr,
                                                        memcpy_direction dir)
        {
            विकल्प (dir)
            {
            पक्ष memcpy_direction::host_गन्तव्यम्_host:
            पक्ष memcpy_direction::host_गन्तव्यम्_device:
            पक्ष memcpy_direction::device_गन्तव्यम्_host:
            पक्ष memcpy_direction::device_गन्तव्यम्_device:
                प्रत्यागम dir;
            पक्ष memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                स्थिर ध्रुव memcpy_direction
                    direction_सारणी[static_cast<अचिह्नित>(pointer_access_attribute::end)]
                                   [static_cast<अचिह्नित>(pointer_access_attribute::end)] =
                                       {{memcpy_direction::host_गन्तव्यम्_host,
                                         memcpy_direction::device_गन्तव्यम्_host,
                                         memcpy_direction::host_गन्तव्यम्_host},
                                        {memcpy_direction::host_गन्तव्यम्_device,
                                         memcpy_direction::device_गन्तव्यम्_device,
                                         memcpy_direction::device_गन्तव्यम्_device},
                                        {memcpy_direction::host_गन्तव्यम्_host,
                                         memcpy_direction::device_गन्तव्यम्_device,
                                         memcpy_direction::device_गन्तव्यम्_device}};
                प्रत्यागम direction_सारणी[static_cast<अचिह्नित>(get_pointer_attribute(
                    q, गन्तव्यम्_ptr))][static_cast<अचिह्नित>(get_pointer_attribute(q, from_ptr))];
            }
            स्वभाव:
                क्षिप std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        स्थिर sycl::event
        dpct_memcpy(sycl::queue &q, शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr, size_t size,
                    memcpy_direction direction,
                    ध्रुव std::vector<sycl::event> &dep_events = {})
        {
            यदि (!size)
                प्रत्यागम sycl::event{};
            प्रत्यागम q.स्मृतिं_प्रतिलिख(गन्तव्यम्_ptr, from_ptr, size, dep_events);
            DPCT_अप्रयुक्तम्(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        स्थिर अन्तःस्थ size_t get_copy_range(sycl::range<3> size, size_t खण्डिका_2,
                                            size_t pitch)
        {
            प्रत्यागम खण्डिका_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        स्थिर अन्तःस्थ size_t get_विस्थापनम्(sycl::id<3> id, size_t खण्डिका_2,
                                        size_t pitch)
        {
            प्रत्यागम खण्डिका_2 * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        स्थिर अन्तःस्थ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr,
                    sycl::range<3> गन्तव्यम्_range, sycl::range<3> from_range,
                    sycl::id<3> गन्तव्यम्_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ध्रुव std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            वर्ग host_बफर
            {
                शून्य *_buf;
                size_t आकार_2;
                sycl::queue &_q;
                ध्रुव std::vector<sycl::event> &_deps; // free operation depends

            सार्वजनिक:
                host_बफर(size_t size, sycl::queue &q,
                            ध्रुव std::vector<sycl::event> &deps)
                    : _buf(std::स्मृतिं_देहि(size)), आकार_2(size), _q(q), _deps(deps) {}
                शून्य *get_ptr() ध्रुव { प्रत्यागम _buf; }
                size_t get_आकार() ध्रुव { प्रत्यागम आकार_2; }
                ~host_बफर()
                {
                    यदि (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::मुक्त(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t गन्तव्यम्_खण्डिका = गन्तव्यम्_range.get(1) * गन्तव्यम्_range.get(0),
                   from_खण्डिका = from_range.get(1) * from_range.get(0);
            अचिह्नित वर्ण *गन्तव्यम्_surface =
                (अचिह्नित वर्ण *)गन्तव्यम्_ptr + get_विस्थापनम्(गन्तव्यम्_id, गन्तव्यम्_खण्डिका, गन्तव्यम्_range.get(0));
            ध्रुव अचिह्नित वर्ण *from_surface =
                (ध्रुव अचिह्नित वर्ण *)from_ptr +
                get_विस्थापनम्(from_id, from_खण्डिका, from_range.get(0));

            यदि (गन्तव्यम्_खण्डिका == from_खण्डिका && गन्तव्यम्_खण्डिका == size.get(1) * size.get(0))
            {
                प्रत्यागम {dpct_memcpy(q, गन्तव्यम्_surface, from_surface, गन्तव्यम्_खण्डिका * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, गन्तव्यम्_ptr, from_ptr, direction);
            size_t आकार_खण्डिका = size.get(1) * size.get(0);
            विकल्प (direction)
            {
            पक्ष host_गन्तव्यम्_host:
                प्रति (size_t z = 0; z < size.get(2); ++z)
                {
                    अचिह्नित वर्ण *गन्तव्यम्_ptr = गन्तव्यम्_surface;
                    ध्रुव अचिह्नित वर्ण *from_ptr = from_surface;
                    यदि (गन्तव्यम्_range.get(0) == from_range.get(0) &&
                        गन्तव्यम्_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, आकार_खण्डिका,
                                                         direction, dep_events));
                    }
                    अन्यथा
                    {
                        प्रति (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            गन्तव्यम्_ptr += गन्तव्यम्_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    गन्तव्यम्_surface += गन्तव्यम्_खण्डिका;
                    from_surface += from_खण्डिका;
                }
                भङ्ग;
            पक्ष host_गन्तव्यम्_device:
            {
                host_बफर buf(get_copy_range(size, गन्तव्यम्_खण्डिका, गन्तव्यम्_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                यदि (गन्तव्यम्_खण्डिका == आकार_खण्डिका)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, गन्तव्यम्_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_गन्तव्यम्_host, dep_events);
                }
                अन्यथा
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, गन्तव्यम्_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_गन्तव्यम्_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), गन्तव्यम्_surface, buf.get_आकार(),
                                        device_गन्तव्यम्_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, गन्तव्यम्_surface, buf.get_ptr(),
                                                 buf.get_आकार(), host_गन्तव्यम्_device,
                                                 host_events));
                भङ्ग;
            }
            पक्ष device_गन्तव्यम्_host:
            {
                host_बफर buf(get_copy_range(size, from_खण्डिका, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, गन्तव्यम्_surface, buf.get_ptr(), गन्तव्यम्_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_गन्तव्यम्_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_आकार(),
                                                         device_गन्तव्यम्_host, dep_events)});
                भङ्ग;
            }
            पक्ष device_गन्तव्यम्_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<वर्ग dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        गन्तव्यम्_surface[get_विस्थापनम्(id, गन्तव्यम्_खण्डिका, गन्तव्यम्_range.get(0))] =
                            from_surface[get_विस्थापनम्(id, from_खण्डिका, from_range.get(0))];
                    }); }));
                भङ्ग;
            स्वभाव:
                क्षिप std::runtime_error("dpct_memcpy: invalid direction value");
            }
            प्रत्यागम event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        स्थिर अन्तःस्थ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_दत्त गन्तव्यम्, sycl::id<3> गन्तव्यम्_id,
                    pitched_दत्त from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            प्रत्यागम dpct_memcpy(q, गन्तव्यम्.get_दत्त_ptr(), from.get_दत्त_ptr(),
                               sycl::range<3>(गन्तव्यम्.get_pitch(), गन्तव्यम्.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), गन्तव्यम्_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        स्थिर अन्तःस्थ std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr,
                    size_t गन्तव्यम्_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            प्रत्यागम dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, sycl::range<3>(गन्तव्यम्_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        नामाकाश deprecated
        {

            प्रतिरूपम् <प्रकारनाम T, sycl::usm::alloc Alloc_विधा>
            वर्ग usm_allocator
            {
            गुप्त:
                उपयुज्य Alloc = sycl::usm_allocator<T, Alloc_विधा>;
                Alloc _impl;

            सार्वजनिक:
                उपयुज्य value_type = प्रकारनाम std::allocator_traits<Alloc>::value_type;
                उपयुज्य pointer = प्रकारनाम std::allocator_traits<Alloc>::pointer;
                उपयुज्य const_pointer = प्रकारनाम std::allocator_traits<Alloc>::const_pointer;
                उपयुज्य void_pointer = प्रकारनाम std::allocator_traits<Alloc>::void_pointer;
                उपयुज्य const_void_pointer =
                    प्रकारनाम std::allocator_traits<Alloc>::const_void_pointer;
                उपयुज्य reference = प्रकारनाम std::allocator_traits<Alloc>::value_type &;
                उपयुज्य const_reference =
                    ध्रुव प्रकारनाम std::allocator_traits<Alloc>::value_type &;
                उपयुज्य difference_type =
                    प्रकारनाम std::allocator_traits<Alloc>::difference_type;
                उपयुज्य size_type = प्रकारनाम std::allocator_traits<Alloc>::size_type;
                उपयुज्य propagate_on_container_copy_assignment = प्रकारनाम std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                उपयुज्य propagate_on_container_move_assignment = प्रकारनाम std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                उपयुज्य propagate_on_container_swap =
                    प्रकारनाम std::allocator_traits<Alloc>::propagate_on_container_swap;
                उपयुज्य is_always_equal =
                    प्रकारनाम std::allocator_traits<Alloc>::is_always_equal;

                प्रतिरूपम् <प्रकारनाम U>
                संरचना rebind
                {
                    प्रकारनिर्देश usm_allocator<U, Alloc_विधा> other;
                };

                usm_allocator() : _impl(dpct::get_default_queue()) {}
                ~usm_allocator() {}
                usm_allocator(ध्रुव usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { प्रत्यागम &r; }
                const_pointer address(const_reference r) { प्रत्यागम &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = शून्यसूचकः)
                {
                    प्रत्यागम std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                शून्य deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ध्रुव
                {
                    प्रत्यागम std::allocator_traits<Alloc>::max_size(_impl);
                }
                तार्किक संक्रियकः==(ध्रुव usm_allocator &other) ध्रुव { प्रत्यागम _impl == other._impl; }
                तार्किक संक्रियकः!=(ध्रुव usm_allocator &other) ध्रुव { प्रत्यागम _impl != other._impl; }
            };

        } // namespace deprecated

        अन्तःस्थ शून्य dpct_free(शून्य *ptr,
                              ध्रुव sycl::queue &q)
        {
            यदि (ptr)
            {
                sycl::मुक्त(ptr, q.get_context());
            }
        }

        प्रतिरूपम् <प्रकारनाम T>
        अन्तःस्थ स्वचालित get_स्मृतिः(ध्रुव शून्य *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<शून्य *>(x));
            प्रत्यागम new_x;
        }

        प्रतिरूपम् <प्रकारनाम T>
        अन्तःस्थ प्रकारनाम दत्त_Type_2<T>::T2 get_मानम्(ध्रुव T *s, sycl::queue &q)
        {
            उपयुज्य Ty = प्रकारनाम दत्त_Type_2<T>::T2;
            Ty s_h;
            यदि (get_pointer_attribute(q, s) == pointer_access_attribute::device_only)
                detail::dpct_memcpy(q, (शून्य *)&s_h, (ध्रुव शून्य *)s, परिमाण(T), device_गन्तव्यम्_host)
                    .प्रतीक्षस्व();
            अन्यथा
                s_h = *reinterpret_cast<ध्रुव Ty *>(s);
            प्रत्यागम s_h;
        }

    } // namespace detail

    प्रतिरूपम् <प्रकारनाम T>
    अन्तःस्थ स्वचालित get_मानम्(ध्रुव T *s, sycl::queue &q)
    {
        प्रत्यागम detail::get_मानम्(s, q);
    }

    नामाकाश detail
    {
    प्रतिरूपम् <वर्ग Ta, वर्ग Tb, वर्ग Tc, वर्ग Ts>
    अन्तःस्थ शून्य gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णाङ्क m,
                          पूर्णाङ्क n, पूर्णाङ्क k, ध्रुव शून्य * alpha, ध्रुव शून्य * a, पूर्णाङ्क lda, ध्रुव शून्य * b, पूर्णाङ्क ldb,
                          ध्रुव शून्य * beta, शून्य * c, पूर्णाङ्क ldc) {
        Ts   alpha_मानम्_2 = dpct::get_मानम्(reinterpret_cast<ध्रुव Ts *>(alpha), q);
        Ts   beta_मानम्  = dpct::get_मानम्(reinterpret_cast<ध्रुव Ts *>(beta), q);
        स्वचालित दत्त_a      = get_स्मृतिः<ध्रुव Ta>(a);
        स्वचालित दत्त_b      = get_स्मृतिः<ध्रुव Tb>(b);
        स्वचालित दत्त_c      = get_स्मृतिः<Tc>(c);
        oneapi::mkl::blas::column_major::gemm(q, a_trans, b_trans, m, n, k, alpha_मानम्_2, दत्त_a,
                                               lda, दत्त_b, ldb, beta_मानम्, दत्त_c, ldc);
    }

        प्रतिरूपम् <प्रकारनाम VecT, वर्ग द्विमान_Operation, वर्ग = शून्य>
        वर्ग vectorized_द्विमान
        {
        सार्वजनिक:
            अन्तःस्थ VecT संक्रियकः()(VecT a, VecT b, ध्रुव द्विमान_Operation द्विमान_op)
            {
                VecT v4;
                प्रति (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = द्विमान_op(a[i], b[i]);
                }
                प्रत्यागम v4;
            }
        };

        प्रतिरूपम् <प्रकारनाम VecT, वर्ग द्विमान_Operation>
        वर्ग vectorized_द्विमान<
            VecT, द्विमान_Operation,
            std::void_t<std::invoke_result_t<द्विमान_Operation, VecT, VecT>>>
        {
        सार्वजनिक:
            अन्तःस्थ VecT संक्रियकः()(VecT a, VecT b, ध्रुव द्विमान_Operation द्विमान_op)
            {
                प्रत्यागम द्विमान_op(a, b).प्रतिरूपम् as<VecT>();
            }
        };

        प्रतिरूपम् <वर्ग Ta, वर्ग Tb, वर्ग Tc, वर्ग Ts>
        अन्तःस्थ शून्य gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    पूर्णाङ्क m, पूर्णाङ्क n, पूर्णाङ्क k, ध्रुव शून्य * alpha, ध्रुव शून्य ** a, पूर्णाङ्क lda, ध्रुव शून्य ** b,
                                    पूर्णाङ्क ldb, ध्रुव शून्य * beta, शून्य ** c, पूर्णाङ्क ldc, पूर्णाङ्क batch_आकार,
                                    matrix_info_t<भिन्न> * matrix_info) {
            Ts alpha_मानम्_2 = dpct::get_मानम्(reinterpret_cast<ध्रुव Ts *>(alpha), q);
            Ts beta_मानम् = dpct::get_मानम्(reinterpret_cast<ध्रुव Ts *>(beta), q);

            matrix_info->transpose_info[0] = a_trans;
            matrix_info->transpose_info[1] = b_trans;
            matrix_info->मानम्_info[0] = alpha_मानम्_2;
            matrix_info->मानम्_info[1] = beta_मानम्;
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
                reinterpret_cast<Ts *>(matrix_info->मानम्_info), reinterpret_cast<ध्रुव Ta **>(a), matrix_info->ld_info,
                reinterpret_cast<ध्रुव Tb **>(b), matrix_info->ld_info + 1,
                reinterpret_cast<Ts *>(matrix_info->मानम्_info + 1), reinterpret_cast<Tc **>(c),
                matrix_info->ld_info + 2, 1, &(matrix_info->groupsize_info));
        }

        प्रतिरूपम् <वर्ग Ta, वर्ग Tb, वर्ग Tc, वर्ग Ts>
        अन्तःस्थ शून्य gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    पूर्णाङ्क m, पूर्णाङ्क n, पूर्णाङ्क k, ध्रुव शून्य * alpha, ध्रुव शून्य * a, पूर्णाङ्क lda,
                                    दीर्घ दीर्घ पूर्णाङ्क stride_a, ध्रुव शून्य * b, पूर्णाङ्क ldb, दीर्घ दीर्घ पूर्णाङ्क stride_b,
                                    ध्रुव शून्य * beta, शून्य * c, पूर्णाङ्क ldc, दीर्घ दीर्घ पूर्णाङ्क stride_c, पूर्णाङ्क batch_आकार) {
            Ts alpha_मानम्_2 = dpct::get_मानम्(reinterpret_cast<ध्रुव Ts *>(alpha), q);
            Ts beta_मानम् = dpct::get_मानम्(reinterpret_cast<ध्रुव Ts *>(beta), q);
            स्वचालित दत्त_a = get_स्मृतिः<ध्रुव Ta>(a);
            स्वचालित दत्त_b = get_स्मृतिः<ध्रुव Tb>(b);
            स्वचालित दत्त_c = get_स्मृतिः<Tc>(c);
            oneapi::mkl::blas::column_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_मानम्_2,
                                                         दत्त_a, lda, stride_a, दत्त_b, ldb, stride_b, beta_मानम्,
                                                         दत्त_c, ldc, stride_c, batch_आकार);
        }

    } // namespace detail

    प्रतिरूपम् <प्रकारनाम VecT, वर्ग द्विमान_Operation>
    अन्तःस्थ अचिह्नित vectorized_द्विमान(अचिह्नित a, अचिह्नित b,
                                      ध्रुव द्विमान_Operation द्विमान_op)
    {
        sycl::vec<अचिह्नित, 1> v0{a}, v1{b};
        स्वचालित v2 = v0.as<VecT>();
        स्वचालित v3 = v1.as<VecT>();
        स्वचालित v4 =
            detail::vectorized_द्विमान<VecT, द्विमान_Operation>()(v2, v3, द्विमान_op);
        v0 = v4.प्रतिरूपम् as<sycl::vec<अचिह्नित, 1>>();
        प्रत्यागम v0;
    }

    स्थिर शून्य async_dpct_memcpy(शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_queue())
    {
        detail::dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, size, direction);
    }

    स्थिर अन्तःस्थ अचिह्नित पूर्णाङ्क select_device(अचिह्नित पूर्णाङ्क id)
    {
        dev_mgr::instance().select_device(id);
        प्रत्यागम id;
    }

    प्रतिरूपम् <प्रकारनाम T>
    T permute_sub_गणः_by_xor(sycl::sub_गणः g, T x, अचिह्नित पूर्णाङ्क आवरणम्,
                               अचिह्नित पूर्णाङ्क logical_sub_गणः_आकार = 32)
    {
        अचिह्नित पूर्णाङ्क id = g.get_local_रैखिकम्_id();
        अचिह्नित पूर्णाङ्क start_सूचकांक =
            id / logical_sub_गणः_आकार * logical_sub_गणः_आकार;
        अचिह्नित पूर्णाङ्क target_विस्थापनम् = (id % logical_sub_गणः_आकार) ^ आवरणम्;
        प्रत्यागम sycl::select_from_गणः(g, x,
                                       target_विस्थापनम् < logical_sub_गणः_आकार
                                           ? start_सूचकांक + target_विस्थापनम्
                                           : id);
    }

    प्रतिरूपम् <प्रकारनाम T1, प्रकारनाम T2>
    उपयुज्य dot_गुणनफलम्_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    प्रतिरूपम् <प्रकारनाम T>
    sycl::vec<T, 4> extract_and_sign_or_zero_extend4(T val) {
      प्रत्यागम sycl::vec<T, 1>(val)
          .प्रतिरूपम् as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .प्रतिरूपम् convert<T>();
    }

    प्रतिरूपम् <प्रकारनाम T1, प्रकारनाम T2, प्रकारनाम T3>
    अन्तःस्थ स्वचालित dp4a(T1 a, T2 b, T3 c) {
      dot_गुणनफलम्_acc_t<T1, T2> res = c;
      स्वचालित va = extract_and_sign_or_zero_extend4(a);
      स्वचालित vb = extract_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      प्रत्यागम res;
    }

    संरचना sub_sat
    {
        प्रतिरूपम् <प्रकारनाम T>
        स्वचालित संक्रियकः()(ध्रुव T x, ध्रुव T y) ध्रुव
        {
            प्रत्यागम sycl::sub_sat(x, y);
        }
    };

    प्रतिरूपम् <प्रकारनाम S, प्रकारनाम T>
    अन्तःस्थ T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        स्वचालित v2 = v0.प्रतिरूपम् as<S>();
        स्वचालित v3 = v1.प्रतिरूपम् as<S>();
        स्वचालित v4 = sycl::min(v2, v3);
        v0 = v4.प्रतिरूपम् as<sycl::vec<T, 1>>();
        प्रत्यागम v0;
    }

    अन्तःस्थ भिन्न घातः(ध्रुव भिन्न a, ध्रुव पूर्णाङ्क b) { प्रत्यागम sycl::pown(a, b); }
    अन्तःस्थ द्विगुण घातः(ध्रुव द्विगुण a, ध्रुव पूर्णाङ्क b) { प्रत्यागम sycl::pown(a, b); }
    अन्तःस्थ भिन्न घातः(ध्रुव भिन्न a, ध्रुव भिन्न b) { प्रत्यागम sycl::घातः(a, b); }
    अन्तःस्थ द्विगुण घातः(ध्रुव द्विगुण a, ध्रुव द्विगुण b) { प्रत्यागम sycl::घातः(a, b); }
    प्रतिरूपम् <प्रकारनाम T, प्रकारनाम U>
    अन्तःस्थ प्रकारनाम std::enable_if_t<std::is_floating_point_v<T>, T>
    घातः(ध्रुव T a, ध्रुव U b)
    {
        प्रत्यागम sycl::घातः(a, static_cast<T>(b));
    }
    प्रतिरूपम् <प्रकारनाम T, प्रकारनाम U>
    अन्तःस्थ प्रकारनाम std::enable_if_t<!std::is_floating_point_v<T>, द्विगुण>
    घातः(ध्रुव T a, ध्रुव U b)
    {
        प्रत्यागम sycl::घातः(static_cast<द्विगुण>(a), static_cast<द्विगुण>(b));
    }

    अन्तःस्थ द्विगुण min(ध्रुव द्विगुण a, ध्रुव भिन्न b)
    {
        प्रत्यागम sycl::fmin(a, static_cast<द्विगुण>(b));
    }
    अन्तःस्थ द्विगुण min(ध्रुव भिन्न a, ध्रुव द्विगुण b)
    {
        प्रत्यागम sycl::fmin(static_cast<द्विगुण>(a), b);
    }
    अन्तःस्थ भिन्न min(ध्रुव भिन्न a, ध्रुव भिन्न b) { प्रत्यागम sycl::fmin(a, b); }
    अन्तःस्थ द्विगुण min(ध्रुव द्विगुण a, ध्रुव द्विगुण b) { प्रत्यागम sycl::fmin(a, b); }
    अन्तःस्थ std::uint32_t min(ध्रुव std::uint32_t a, ध्रुव std::int32_t b)
    {
        प्रत्यागम sycl::min(a, static_cast<std::uint32_t>(b));
    }
    अन्तःस्थ std::uint32_t min(ध्रुव std::int32_t a, ध्रुव std::uint32_t b)
    {
        प्रत्यागम sycl::min(static_cast<std::uint32_t>(a), b);
    }
    अन्तःस्थ std::int32_t min(ध्रुव std::int32_t a, ध्रुव std::int32_t b)
    {
        प्रत्यागम sycl::min(a, b);
    }
    अन्तःस्थ std::uint32_t min(ध्रुव std::uint32_t a, ध्रुव std::uint32_t b)
    {
        प्रत्यागम sycl::min(a, b);
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::uint64_t a, ध्रुव std::int64_t b)
    {
        प्रत्यागम sycl::min(a, static_cast<std::uint64_t>(b));
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::int64_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::min(static_cast<std::uint64_t>(a), b);
    }
    अन्तःस्थ std::int64_t min(ध्रुव std::int64_t a, ध्रुव std::int64_t b)
    {
        प्रत्यागम sycl::min(a, b);
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::uint64_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::min(a, b);
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::uint64_t a, ध्रुव std::int32_t b)
    {
        प्रत्यागम sycl::min(a, static_cast<std::uint64_t>(b));
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::int32_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::min(static_cast<std::uint64_t>(a), b);
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::uint64_t a, ध्रुव std::uint32_t b)
    {
        प्रत्यागम sycl::min(a, static_cast<std::uint64_t>(b));
    }
    अन्तःस्थ std::uint64_t min(ध्रुव std::uint32_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    अन्तःस्थ द्विगुण max(ध्रुव द्विगुण a, ध्रुव भिन्न b)
    {
        प्रत्यागम sycl::fmax(a, static_cast<द्विगुण>(b));
    }
    अन्तःस्थ द्विगुण max(ध्रुव भिन्न a, ध्रुव द्विगुण b)
    {
        प्रत्यागम sycl::fmax(static_cast<द्विगुण>(a), b);
    }
    अन्तःस्थ भिन्न max(ध्रुव भिन्न a, ध्रुव भिन्न b) { प्रत्यागम sycl::fmax(a, b); }
    अन्तःस्थ द्विगुण max(ध्रुव द्विगुण a, ध्रुव द्विगुण b) { प्रत्यागम sycl::fmax(a, b); }
    अन्तःस्थ std::uint32_t max(ध्रुव std::uint32_t a, ध्रुव std::int32_t b)
    {
        प्रत्यागम sycl::max(a, static_cast<std::uint32_t>(b));
    }
    अन्तःस्थ std::uint32_t max(ध्रुव std::int32_t a, ध्रुव std::uint32_t b)
    {
        प्रत्यागम sycl::max(static_cast<std::uint32_t>(a), b);
    }
    अन्तःस्थ std::int32_t max(ध्रुव std::int32_t a, ध्रुव std::int32_t b)
    {
        प्रत्यागम sycl::max(a, b);
    }
    अन्तःस्थ std::uint32_t max(ध्रुव std::uint32_t a, ध्रुव std::uint32_t b)
    {
        प्रत्यागम sycl::max(a, b);
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::uint64_t a, ध्रुव std::int64_t b)
    {
        प्रत्यागम sycl::max(a, static_cast<std::uint64_t>(b));
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::int64_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::max(static_cast<std::uint64_t>(a), b);
    }
    अन्तःस्थ std::int64_t max(ध्रुव std::int64_t a, ध्रुव std::int64_t b)
    {
        प्रत्यागम sycl::max(a, b);
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::uint64_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::max(a, b);
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::uint64_t a, ध्रुव std::int32_t b)
    {
        प्रत्यागम sycl::max(a, static_cast<std::uint64_t>(b));
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::int32_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::max(static_cast<std::uint64_t>(a), b);
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::uint64_t a, ध्रुव std::uint32_t b)
    {
        प्रत्यागम sycl::max(a, static_cast<std::uint64_t>(b));
    }
    अन्तःस्थ std::uint64_t max(ध्रुव std::uint32_t a, ध्रुव std::uint64_t b)
    {
        प्रत्यागम sycl::max(static_cast<std::uint64_t>(a), b);
    }

    अन्तःस्थ शून्य
    has_capability_or_fail(ध्रुव sycl::device &dev,
                           ध्रुव std::initializer_list<sycl::aspect> &props)
    {
        प्रति (ध्रुव स्वचालित &it : props)
        {
            यदि (dev.has(it))
                अनुवर्त;
            विकल्प (it)
            {
            पक्ष sycl::aspect::fp64:
                क्षिप std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                भङ्ग;
            पक्ष sycl::aspect::fp16:
                क्षिप std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                भङ्ग;
            स्वभाव:
#परिभाषय __SYCL_ASPECT(ASPECT, ID) \
    पक्ष sycl::aspect::ASPECT:    \
        प्रत्यागम #ASPECT;
#परिभाषय __SYCL_ASPECT_DEPRECATED(ASPECT, ID, सन्देशः_3) __SYCL_ASPECT(ASPECT, ID)
#परिभाषय __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, सन्देशः_3)
                स्वचालित getAspectNameStr = [](sycl::aspect Aspect_अङ्कः) -> std::string
                {
                    विकल्प (Aspect_अङ्कः)
                    {
#अन्तर्भाव <sycl/info/aspects.def>
#अन्तर्भाव <sycl/info/aspects_deprecated.def>
                    स्वभाव:
                        प्रत्यागम "unknown aspect";
                    }
                };
#अपरिभाषय __SYCL_ASPECT_DEPRECATED_ALIAS
#अपरिभाषय __SYCL_ASPECT_DEPRECATED
#अपरिभाषय __SYCL_ASPECT
                क्षिप std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            भङ्ग;
        }
    }

    स्थिर अन्तःस्थ अचिह्नित पूर्णाङ्क get_current_device_id()
    {
        प्रत्यागम dev_mgr::instance().current_device_id();
    }

    स्थिर अन्तःस्थ device_ext &get_current_device()
    {
        प्रत्यागम dev_mgr::instance().current_device();
    }

    स्थिर अन्तःस्थ device_ext &get_device(अचिह्नित पूर्णाङ्क id)
    {
        प्रत्यागम dev_mgr::instance().get_device(id);
    }

    स्थिर अन्तःस्थ sycl::queue &get_अन्तर्_क्रमः_queue()
    {
        प्रत्यागम dev_mgr::instance().current_device().अन्तर्_क्रमः_queue();
    }

    स्थिर sycl::event
    dpct_memcpy(sycl::queue &q, शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr, size_t size,
                memcpy_direction direction,
                ध्रुव std::vector<sycl::event> &dep_events = {})
    {
        यदि (!size)
            प्रत्यागम sycl::event{};
        प्रत्यागम q.स्मृतिं_प्रतिलिख(गन्तव्यम्_ptr, from_ptr, size, dep_events);
        DPCT_अप्रयुक्तम्(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    स्थिर अन्तःस्थ size_t get_copy_range(sycl::range<3> size, size_t खण्डिका_2,
                                        size_t pitch)
    {
        प्रत्यागम खण्डिका_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    स्थिर अन्तःस्थ size_t get_विस्थापनम्(sycl::id<3> id, size_t खण्डिका_2,
                                    size_t pitch)
    {
        प्रत्यागम खण्डिका_2 * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    स्थिर अन्तःस्थ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr,
                sycl::range<3> गन्तव्यम्_range, sycl::range<3> from_range,
                sycl::id<3> गन्तव्यम्_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ध्रुव std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        वर्ग host_बफर
        {
            शून्य *_buf;
            size_t आकार_2;
            sycl::queue &_q;
            ध्रुव std::vector<sycl::event> &_deps; // free operation depends

        सार्वजनिक:
            host_बफर(size_t size, sycl::queue &q,
                        ध्रुव std::vector<sycl::event> &deps)
                : _buf(std::स्मृतिं_देहि(size)), आकार_2(size), _q(q), _deps(deps) {}
            शून्य *get_ptr() ध्रुव { प्रत्यागम _buf; }
            size_t get_आकार() ध्रुव { प्रत्यागम आकार_2; }
            ~host_बफर()
            {
                यदि (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::मुक्त(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t गन्तव्यम्_खण्डिका = गन्तव्यम्_range.get(1) * गन्तव्यम्_range.get(0),
               from_खण्डिका = from_range.get(1) * from_range.get(0);
        अचिह्नित वर्ण *गन्तव्यम्_surface =
            (अचिह्नित वर्ण *)गन्तव्यम्_ptr + get_विस्थापनम्(गन्तव्यम्_id, गन्तव्यम्_खण्डिका, गन्तव्यम्_range.get(0));
        ध्रुव अचिह्नित वर्ण *from_surface =
            (ध्रुव अचिह्नित वर्ण *)from_ptr +
            get_विस्थापनम्(from_id, from_खण्डिका, from_range.get(0));

        यदि (गन्तव्यम्_खण्डिका == from_खण्डिका && गन्तव्यम्_खण्डिका == size.get(1) * size.get(0))
        {
            प्रत्यागम {dpct_memcpy(q, गन्तव्यम्_surface, from_surface, गन्तव्यम्_खण्डिका * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, गन्तव्यम्_ptr, from_ptr, direction);
        size_t आकार_खण्डिका = size.get(1) * size.get(0);
        विकल्प (direction)
        {
        पक्ष host_गन्तव्यम्_host:
            प्रति (size_t z = 0; z < size.get(2); ++z)
            {
                अचिह्नित वर्ण *गन्तव्यम्_ptr = गन्तव्यम्_surface;
                ध्रुव अचिह्नित वर्ण *from_ptr = from_surface;
                यदि (गन्तव्यम्_range.get(0) == from_range.get(0) &&
                    गन्तव्यम्_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, आकार_खण्डिका,
                                                     direction, dep_events));
                }
                अन्यथा
                {
                    प्रति (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        गन्तव्यम्_ptr += गन्तव्यम्_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                गन्तव्यम्_surface += गन्तव्यम्_खण्डिका;
                from_surface += from_खण्डिका;
            }
            भङ्ग;
        पक्ष host_गन्तव्यम्_device:
        {
            host_बफर buf(get_copy_range(size, गन्तव्यम्_खण्डिका, गन्तव्यम्_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            यदि (गन्तव्यम्_खण्डिका == आकार_खण्डिका)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, गन्तव्यम्_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_गन्तव्यम्_host, dep_events);
            }
            अन्यथा
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, गन्तव्यम्_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_गन्तव्यम्_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), गन्तव्यम्_surface, buf.get_आकार(),
                                    device_गन्तव्यम्_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, गन्तव्यम्_surface, buf.get_ptr(),
                                             buf.get_आकार(), host_गन्तव्यम्_device,
                                             host_events));
            भङ्ग;
        }
        पक्ष device_गन्तव्यम्_host:
        {
            host_बफर buf(get_copy_range(size, from_खण्डिका, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, गन्तव्यम्_surface, buf.get_ptr(), गन्तव्यम्_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_गन्तव्यम्_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_आकार(),
                                                     device_गन्तव्यम्_host, dep_events)});
            भङ्ग;
        }
        पक्ष device_गन्तव्यम्_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<वर्ग dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                गन्तव्यम्_surface[get_विस्थापनम्(id, गन्तव्यम्_खण्डिका, गन्तव्यम्_range.get(0))] =
                    from_surface[get_विस्थापनम्(id, from_खण्डिका, from_range.get(0))];
            }); }));
        भङ्ग;
        स्वभाव:
            क्षिप std::runtime_error("dpct_memcpy: invalid direction value");
        }
        प्रत्यागम event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    स्थिर अन्तःस्थ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_दत्त गन्तव्यम्, sycl::id<3> गन्तव्यम्_id,
                pitched_दत्त from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        प्रत्यागम dpct_memcpy(q, गन्तव्यम्.get_दत्त_ptr(), from.get_दत्त_ptr(),
                           sycl::range<3>(गन्तव्यम्.get_pitch(), गन्तव्यम्.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), गन्तव्यम्_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    स्थिर अन्तःस्थ std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, शून्य *गन्तव्यम्_ptr, ध्रुव शून्य *from_ptr,
                size_t गन्तव्यम्_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        प्रत्यागम dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, sycl::range<3>(गन्तव्यम्_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    अन्तःस्थ शून्य gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णाङ्क m, पूर्णाङ्क n,
                     पूर्णाङ्क k, ध्रुव शून्य * alpha, ध्रुव शून्य * a, library_दत्त_t a_type, पूर्णाङ्क lda, ध्रुव शून्य * b,
                     library_दत्त_t b_type, पूर्णाङ्क ldb, ध्रुव शून्य * beta, शून्य * c, library_दत्त_t c_type, पूर्णाङ्क ldc,
                     library_दत्त_t scaling_type) {
        यदि (scaling_type == library_दत्त_t::real_float &&
            c_type == library_दत्त_t::complex_float)
        {
            scaling_type = library_दत्त_t::complex_float;
        }
        अन्यथा यदि (scaling_type == library_दत्त_t::real_double &&
                 c_type == library_दत्त_t::complex_double)
        {
            scaling_type = library_दत्त_t::complex_double;
        }

        std::uint64_t कुञ्चिका =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        विकल्प (कुञ्चिका)
        {
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_float, library_दत्त_t::real_float,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_impl<भिन्न, भिन्न, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_double, library_दत्त_t::real_double,
            library_दत्त_t::real_double, library_दत्त_t::real_double):
        {
            detail::gemm_impl<द्विगुण, द्विगुण, द्विगुण, द्विगुण>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::complex_float, library_दत्त_t::complex_float,
            library_दत्त_t::complex_float, library_दत्त_t::complex_float):
        {
            detail::gemm_impl<std::complex<भिन्न>, std::complex<भिन्न>,
                              std::complex<भिन्न>, std::complex<भिन्न>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::complex_double, library_दत्त_t::complex_double,
            library_दत्त_t::complex_double, library_दत्त_t::complex_double):
        {
            detail::gemm_impl<std::complex<द्विगुण>, std::complex<द्विगुण>,
                              std::complex<द्विगुण>, std::complex<द्विगुण>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_half, library_दत्त_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
#यदि_परिभाषितम् __INTEL_MKL__
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_bfloat16,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_half, library_दत्त_t::real_float):
        {
            भिन्न alpha_मानम्_2 =
                dpct::get_मानम्(reinterpret_cast<ध्रुव भिन्न *>(alpha), q);
            भिन्न beta_मानम् =
                dpct::get_मानम्(reinterpret_cast<ध्रुव भिन्न *>(beta), q);
            sycl::half alpha_half(alpha_मानम्_2);
            sycl::half beta_half(beta_मानम्);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_int8, library_दत्त_t::real_int8,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_bfloat16,
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_int8, library_दत्त_t::real_int8,
            library_दत्त_t::real_int32, library_दत्त_t::real_int32):
        {
            भिन्न alpha_float =
                dpct::get_मानम्(reinterpret_cast<ध्रुव std::int32_t *>(alpha), q);
            भिन्न beta_float =
                dpct::get_मानम्(reinterpret_cast<ध्रुव std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, भिन्न>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            भङ्ग;
        }
#यदि_अन्तः // __INTEL_MKL__
        स्वभाव:
            क्षिप std::runtime_error("the combination of data type is unsupported");
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
    अन्तःस्थ शून्य gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णाङ्क m,
                           पूर्णाङ्क n, पूर्णाङ्क k, ध्रुव शून्य * alpha, ध्रुव शून्य * a[], library_दत्त_t a_type, पूर्णाङ्क lda,
                           ध्रुव शून्य * b[], library_दत्त_t b_type, पूर्णाङ्क ldb, ध्रुव शून्य * beta, शून्य * c[],
                           library_दत्त_t c_type, पूर्णाङ्क ldc, पूर्णाङ्क batch_आकार, library_दत्त_t scaling_type,
                           matrix_info_t<भिन्न> * matrix_info) {
        std::uint64_t कुञ्चिका =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        विकल्प (कुञ्चिका)
        {
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_float, library_दत्त_t::real_float,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<भिन्न, भिन्न, भिन्न, भिन्न>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_double, library_दत्त_t::real_double,
            library_दत्त_t::real_double, library_दत्त_t::real_double):
        {
            detail::gemm_batch_impl<द्विगुण, द्विगुण, द्विगुण, द्विगुण>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_half, library_दत्त_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
#यदि_परिभाषितम् __INTEL_MKL__
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_bfloat16,
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_bfloat16,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
#यदि_अन्तः
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_int8, library_दत्त_t::real_int8,
            library_दत्त_t::real_int32, library_दत्त_t::real_int32):
        {
            भिन्न alpha_float =
                dpct::get_मानम्(reinterpret_cast<ध्रुव std::int32_t *>(alpha), q);
            भिन्न beta_float =
                dpct::get_मानम्(reinterpret_cast<ध्रुव std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, भिन्न>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_आकार,
                matrix_info);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_int8, library_दत्त_t::real_int8,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_half, library_दत्त_t::real_float):
        {
            भिन्न alpha_मानम्_2 =
                dpct::get_मानम्(reinterpret_cast<ध्रुव भिन्न *>(alpha), q);
            भिन्न beta_मानम् =
                dpct::get_मानम्(reinterpret_cast<ध्रुव भिन्न *>(beta), q);
            sycl::half alpha_half(alpha_मानम्_2);
            sycl::half beta_half(beta_मानम्);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_आकार, matrix_info);
            भङ्ग;
        }
        स्वभाव:
            क्षिप std::runtime_error("the combination of data type is unsupported");
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
    अन्तःस्थ शून्य gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, पूर्णाङ्क m,
                           पूर्णाङ्क n, पूर्णाङ्क k, ध्रुव शून्य * alpha, ध्रुव शून्य * a, library_दत्त_t a_type, पूर्णाङ्क lda,
                           दीर्घ दीर्घ पूर्णाङ्क stride_a, ध्रुव शून्य * b, library_दत्त_t b_type, पूर्णाङ्क ldb,
                           दीर्घ दीर्घ पूर्णाङ्क stride_b, ध्रुव शून्य * beta, शून्य * c, library_दत्त_t c_type, पूर्णाङ्क ldc,
                           दीर्घ दीर्घ पूर्णाङ्क stride_c, पूर्णाङ्क batch_आकार, library_दत्त_t scaling_type) {
        यदि (scaling_type == library_दत्त_t::real_float &&
            c_type == library_दत्त_t::complex_float)
        {
            scaling_type = library_दत्त_t::complex_float;
        }
        अन्यथा यदि (scaling_type == library_दत्त_t::real_double &&
                 c_type == library_दत्त_t::complex_double)
        {
            scaling_type = library_दत्त_t::complex_double;
        }

        std::uint64_t कुञ्चिका =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        विकल्प (कुञ्चिका)
        {
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_float, library_दत्त_t::real_float,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<भिन्न, भिन्न, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_double, library_दत्त_t::real_double,
            library_दत्त_t::real_double, library_दत्त_t::real_double):
        {
            detail::gemm_batch_impl<द्विगुण, द्विगुण, द्विगुण, द्विगुण>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::complex_float, library_दत्त_t::complex_float,
            library_दत्त_t::complex_float, library_दत्त_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<भिन्न>, std::complex<भिन्न>,
                                    std::complex<भिन्न>, std::complex<भिन्न>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::complex_double, library_दत्त_t::complex_double,
            library_दत्त_t::complex_double, library_दत्त_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<द्विगुण>, std::complex<द्विगुण>,
                                    std::complex<द्विगुण>, std::complex<द्विगुण>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_half, library_दत्त_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
#यदि_परिभाषितम् __INTEL_MKL__
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_bfloat16,
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_bfloat16, library_दत्त_t::real_bfloat16,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_आकार);
            भङ्ग;
        }
#यदि_अन्तः
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_int8, library_दत्त_t::real_int8,
            library_दत्त_t::real_int32, library_दत्त_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_int8, library_दत्त_t::real_int8,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_float, library_दत्त_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, भिन्न, भिन्न>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        पक्ष detail::get_type_combination_id(
            library_दत्त_t::real_half, library_दत्त_t::real_half,
            library_दत्त_t::real_half, library_दत्त_t::real_float):
        {
            भिन्न alpha_मानम्_2 =
                dpct::get_मानम्(reinterpret_cast<ध्रुव भिन्न *>(alpha), q);
            भिन्न beta_मानम् =
                dpct::get_मानम्(reinterpret_cast<ध्रुव भिन्न *>(beta), q);
            sycl::half alpha_half(alpha_मानम्_2);
            sycl::half beta_half(beta_मानम्);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_आकार);
            भङ्ग;
        }
        स्वभाव:
            क्षिप std::runtime_error("the combination of data type is unsupported");
        }
    }

    स्थिर अन्तःस्थ शून्य
    async_dpct_memcpy(शून्य *गन्तव्यम्_ptr, size_t गन्तव्यम्_pitch, ध्रुव शून्य *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_queue())
    {
        detail::dpct_memcpy(q, गन्तव्यम्_ptr, from_ptr, गन्तव्यम्_pitch, from_pitch, x, y,
                            direction);
    }

    उपयुज्य err0 = detail::generic_error_type<संरचना err0_चिह्नकम्, पूर्णाङ्क>;
    उपयुज्य err1 = detail::generic_error_type<संरचना err1_चिह्नकम्, पूर्णाङ्क>;

    स्थिर अन्तःस्थ शून्य dpct_free(शून्य *ptr, sycl::queue &q = get_default_queue()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    प्रतिरूपम् <वर्ग T, स्मृतिः_region स्मृतिः_2, size_t Dimension> वर्ग accessor;
    प्रतिरूपम् <वर्ग T, स्मृतिः_region स्मृतिः_2> वर्ग accessor<T, स्मृतिः_2, 3> {
    सार्वजनिक:
        उपयुज्य स्मृतिः_t = detail::स्मृतिः_traits<स्मृतिः_2, T>;
        उपयुज्य element_t = प्रकारनाम स्मृतिः_t::element_t;
        उपयुज्य pointer_t = प्रकारनाम स्मृतिः_t::pointer_t;
        उपयुज्य accessor_t = प्रकारनाम स्मृतिः_t::प्रतिरूपम् accessor_t<3>;
        accessor(pointer_t दत्त, ध्रुव sycl::range<3> &अन्तर्_range)
            : दत्त_2(दत्त), _range(अन्तर्_range) {}
        प्रतिरूपम् <स्मृतिः_region M = स्मृतिः_2>
        accessor(प्रकारनाम std::enable_if<M != local, ध्रुव accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ध्रुव accessor_t &acc, ध्रुव sycl::range<3> &अन्तर्_range)
            : accessor(acc.get_pointer(), अन्तर्_range) {}
        accessor<T, स्मृतिः_2, 2> संक्रियकः[](size_t सूचकांक) ध्रुव {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            प्रत्यागम accessor<T, स्मृतिः_2, 2>(दत्त_2 + सूचकांक * sub.size(), sub);
        }

        pointer_t get_ptr() ध्रुव { प्रत्यागम दत्त_2; }

    गुप्त:
        pointer_t दत्त_2;
        sycl::range<3> _range;
    };
    प्रतिरूपम् <वर्ग T, स्मृतिः_region स्मृतिः_2> वर्ग accessor<T, स्मृतिः_2, 2> {
    सार्वजनिक:
        उपयुज्य स्मृतिः_t = detail::स्मृतिः_traits<स्मृतिः_2, T>;
        उपयुज्य element_t = प्रकारनाम स्मृतिः_t::element_t;
        उपयुज्य pointer_t = प्रकारनाम स्मृतिः_t::pointer_t;
        उपयुज्य accessor_t = प्रकारनाम स्मृतिः_t::प्रतिरूपम् accessor_t<2>;
        accessor(pointer_t दत्त, ध्रुव sycl::range<2> &अन्तर्_range)
            : दत्त_2(दत्त), _range(अन्तर्_range) {}
        प्रतिरूपम् <स्मृतिः_region M = स्मृतिः_2>
        accessor(प्रकारनाम std::enable_if<M != local, ध्रुव accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ध्रुव accessor_t &acc, ध्रुव sycl::range<2> &अन्तर्_range)
            : accessor(acc.get_pointer(), अन्तर्_range) {}

        pointer_t संक्रियकः[](size_t सूचकांक) ध्रुव {
            प्रत्यागम दत्त_2 + _range.get(1) * सूचकांक;
        }

        pointer_t get_ptr() ध्रुव { प्रत्यागम दत्त_2; }

    गुप्त:
        pointer_t दत्त_2;
        sycl::range<2> _range;
    };

    नामाकाश detail {
        /// Device variable with address space of shared, global or constant.
        प्रतिरूपम् <वर्ग T, स्मृतिः_region स्मृतिः_2, size_t Dimension> वर्ग device_स्मृतिः {
        सार्वजनिक:
            उपयुज्य accessor_t =
                प्रकारनाम detail::स्मृतिः_traits<स्मृतिः_2,
                                            T>::प्रतिरूपम् accessor_t<Dimension>;
            उपयुज्य मानम्_t = प्रकारनाम detail::स्मृतिः_traits<स्मृतिः_2, T>::मानम्_t;
            उपयुज्य dpct_accessor_t = dpct::accessor<T, स्मृतिः_2, Dimension>;

            device_स्मृतिः() : device_स्मृतिः(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_स्मृतिः(ध्रुव sycl::range<Dimension> &अन्तर्_range,
                        std::initializer_list<मानम्_t> &&init_list)
                : device_स्मृतिः(अन्तर्_range) {
                प्रतिज्ञा(init_list.size() <= अन्तर्_range.size());
                _host_ptr = (मानम्_t *)std::स्मृतिं_देहि(आकार_2);
                std::स्मृतिं_पूरय(_host_ptr, 0, आकार_2);
                std::स्मृतिं_प्रतिलिख(_host_ptr, init_list.begin(), init_list.size() * परिमाण(T));
            }

            /// Constructor of 2-D array with initializer list
            प्रतिरूपम् <size_t D = Dimension>
            device_स्मृतिः(
                ध्रुव प्रकारनाम std::enable_if<D == 2, sycl::range<2>>::type &अन्तर्_range,
                std::initializer_list<std::initializer_list<मानम्_t>> &&init_list)
                : device_स्मृतिः(अन्तर्_range) {
                प्रतिज्ञा(init_list.size() <= अन्तर्_range[0]);
                _host_ptr = (मानम्_t *)std::स्मृतिं_देहि(आकार_2);
                std::स्मृतिं_पूरय(_host_ptr, 0, आकार_2);
                स्वचालित tmp_दत्त = _host_ptr;
                प्रति (स्वचालित sub_list : init_list) {
                    प्रतिज्ञा(sub_list.size() <= अन्तर्_range[1]);
                    std::स्मृतिं_प्रतिलिख(tmp_दत्त, sub_list.begin(),
                                sub_list.size() * परिमाण(T));
                    tmp_दत्त += अन्तर्_range[1];
                }
            }

            /// Constructor with range
            device_स्मृतिः(ध्रुव sycl::range<Dimension> &range_अन्तर्)
                : आकार_2(range_अन्तर्.size() * परिमाण(T)), _range(range_अन्तर्),
                _reference(असत्य), _host_ptr(शून्यसूचकः), _device_ptr(शून्यसूचकः) {
                static_assert(
                    (स्मृतिः_2 == global) || (स्मृतिः_2 == constant) || (स्मृतिः_2 == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            प्रतिरूपम् <वर्ग... तर्काः>
            device_स्मृतिः(तर्काः... Arguments)
                : device_स्मृतिः(sycl::range<Dimension>(Arguments...)) {}

            ~device_स्मृतिः() {
                यदि (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                यदि (_host_ptr)
                    std::मुक्त(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            शून्य init() { init(dpct::get_default_queue()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            शून्य init(sycl::queue &q) {
                यदि (_device_ptr)
                    प्रत्यागम;
                यदि (!आकार_2)
                    प्रत्यागम;
                allocate_device(q);
                यदि (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, आकार_2,
                                        host_गन्तव्यम्_device);
            }

            /// The variable is assigned to a device pointer.
            शून्य नियोजय(मानम्_t *src, size_t size) {
                एतत्->~device_स्मृतिः();
                नूतन (एतत्) device_स्मृतिः(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            मानम्_t *get_ptr() { प्रत्यागम get_ptr(get_default_queue()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            मानम्_t *get_ptr(sycl::queue &q) {
                init(q);
                प्रत्यागम _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_आकार() { प्रत्यागम आकार_2; }

            प्रतिरूपम् <size_t D = Dimension>
            प्रकारनाम std::enable_if<D == 1, T>::type &संक्रियकः[](size_t सूचकांक) {
                init();
                प्रत्यागम _device_ptr[सूचकांक];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            प्रतिरूपम् <size_t D = Dimension>
            प्रकारनाम std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                प्रत्यागम dpct_accessor_t((T *)_device_ptr, _range);
            }

        गुप्त:
            device_स्मृतिः(मानम्_t *स्मृतिः_ptr, size_t size)
                : आकार_2(size), _range(size / परिमाण(T)), _reference(सत्य),
                _device_ptr(स्मृतिः_ptr) {}

            शून्य allocate_device(sycl::queue &q) {
        #यदि_अपरिभाषितम् DPCT_USM_LEVEL_NONE
                यदि (स्मृतिः_2 == shared) {
                    _device_ptr = (मानम्_t *)sycl::malloc_shared(आकार_2, q.get_device(),
                                                                q.get_context());
                    प्रत्यागम;
                }
        #यदि_परिभाषितम् SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                यदि (स्मृतिः_2 == constant) {
                    _device_ptr = (मानम्_t *)sycl::malloc_device(
                        आकार_2, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    प्रत्यागम;
                }
        #यदि_अन्तः
        #यदि_अन्तः
                _device_ptr = (मानम्_t *)detail::dpct_malloc(आकार_2, q);
            }

            size_t आकार_2;
            sycl::range<Dimension> _range;
            तार्किक _reference;
            मानम्_t *_host_ptr;
            मानम्_t *_device_ptr;
        };
        प्रतिरूपम् <वर्ग T, स्मृतिः_region स्मृतिः_2>
        वर्ग device_स्मृतिः<T, स्मृतिः_2, 0> : सार्वजनिक device_स्मृतिः<T, स्मृतिः_2, 1> {
        सार्वजनिक:
            उपयुज्य आधारः = device_स्मृतिः<T, स्मृतिः_2, 1>;
            उपयुज्य मानम्_t = प्रकारनाम आधारः::मानम्_t;
            उपयुज्य accessor_t =
                प्रकारनाम detail::स्मृतिः_traits<स्मृतिः_2, T>::प्रतिरूपम् accessor_t<0>;

            /// Constructor with initial value.
            device_स्मृतिः(ध्रुव मानम्_t &val) : आधारः(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_स्मृतिः() : आधारः(1) {}
        };
        } // namespace detail

    प्रतिरूपम् <वर्ग T, size_t Dimension>
    उपयुज्य global_स्मृतिः = detail::device_स्मृतिः<T, global, Dimension>;
    प्रतिरूपम् <वर्ग T, size_t Dimension>
    उपयुज्य constant_स्मृतिः = detail::device_स्मृतिः<T, constant, Dimension>;
    प्रतिरूपम् <वर्ग T, size_t Dimension>
    उपयुज्य shared_स्मृतिः = detail::device_स्मृतिः<T, shared, Dimension>;


    प्रतिरूपम् <प्रकारनाम T,
            sycl::access::स्थानाङ्कः_space स्थानाङ्कः_Space =
                sycl::access::स्थानाङ्कः_space::global_space,
            sycl::memory_order स्मृतिः_क्रमः = sycl::memory_order::relaxed,
            sycl::स्मृतिः_scope स्मृतिः_Scope = sycl::स्मृतिः_scope::device>
    अन्तःस्थ T atomic_fetch_add(T *addr, T सङ्कार्यम्) {
    स्वचालित atm =
        sycl::atomic_ref<T, स्मृतिः_क्रमः, स्मृतिः_Scope, स्थानाङ्कः_Space>(addr[0]);
    प्रत्यागम atm.fetch_add(सङ्कार्यम्);
    }

    प्रतिरूपम् <sycl::access::स्थानाङ्कः_space स्थानाङ्कः_Space =
                sycl::access::स्थानाङ्कः_space::global_space,
            sycl::memory_order स्मृतिः_क्रमः = sycl::memory_order::relaxed,
            sycl::स्मृतिः_scope स्मृतिः_Scope = sycl::स्मृतिः_scope::device,
            प्रकारनाम T1, प्रकारनाम T2>
    अन्तःस्थ T1 atomic_fetch_add(T1 *addr, T2 सङ्कार्यम्) {
    स्वचालित atm =
        sycl::atomic_ref<T1, स्मृतिः_क्रमः, स्मृतिः_Scope, स्थानाङ्कः_Space>(addr[0]);
    प्रत्यागम atm.fetch_add(सङ्कार्यम्);
    }

    प्रतिरूपम् <प्रकारनाम T, sycl::access::स्थानाङ्कः_space स्थानाङ्कः_Space =
                            sycl::access::स्थानाङ्कः_space::global_space>
    अन्तःस्थ T atomic_fetch_add(T *addr, T सङ्कार्यम्,
                            sycl::memory_order स्मृतिः_क्रमः) {
    विकल्प (स्मृतिः_क्रमः) {
        पक्ष sycl::memory_order::relaxed:
            प्रत्यागम atomic_fetch_add<T, स्थानाङ्कः_Space, sycl::memory_order::relaxed,
                                    sycl::स्मृतिः_scope::device>(addr, सङ्कार्यम्);
        पक्ष sycl::memory_order::acq_rel:
            प्रत्यागम atomic_fetch_add<T, स्थानाङ्कः_Space, sycl::memory_order::acq_rel,
                                    sycl::स्मृतिः_scope::device>(addr, सङ्कार्यम्);
        पक्ष sycl::memory_order::seq_cst:
            प्रत्यागम atomic_fetch_add<T, स्थानाङ्कः_Space, sycl::memory_order::seq_cst,
                                    sycl::स्मृतिः_scope::device>(addr, सङ्कार्यम्);
        स्वभाव:
            प्रतिज्ञा(असत्य && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    प्रतिरूपम् <sycl::access::स्थानाङ्कः_space स्थानाङ्कः_Space =
                sycl::access::स्थानाङ्कः_space::global_space,
            प्रकारनाम T1, प्रकारनाम T2>
    अन्तःस्थ T1 atomic_fetch_add(T1 *addr, T2 सङ्कार्यम्,
                            sycl::memory_order स्मृतिः_क्रमः) {
    atomic_fetch_add<T1, स्थानाङ्कः_Space>(addr, सङ्कार्यम्, स्मृतिः_क्रमः);
    }

    अन्तःस्थ अचिह्नित पूर्णाङ्क byte_level_permute(
        अचिह्नित पूर्णाङ्क a, अचिह्नित पूर्णाङ्क b, अचिह्नित पूर्णाङ्क s) {
      अचिह्नित पूर्णाङ्क ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      प्रत्यागम ret;
    }

    अन्तःस्थ uint32_t byte_level_permute_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, पूर्णाङ्क mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      यदि (mode >= 1 && mode <= 6) {
        प्रत्यागम byte_level_permute(low32, high32, lookup[mode - 1][sel & 0x3]);
      } अन्यथा यदि (!mode) {
        प्रत्यागम byte_level_permute(low32, high32, sel);
      }
      प्रत्यागम 0;
    }

    प्रतिरूपम् <पूर्णाङ्क n_nondefault_params, पूर्णाङ्क n_default_params, प्रकारनाम T>
    वर्ग तर्काः_selector;

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
    प्रतिरूपम् <पूर्णाङ्क n_nondefault_params, पूर्णाङ्क n_default_params, प्रकारनाम R,
              प्रकारनाम... Ts>
    वर्ग तर्काः_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      गुप्त:
        शून्य **kernel_params;
        वर्ण *तर्काः_बफर;

        प्रतिरूपम् <पूर्णाङ्क i> स्थिर constexpr पूर्णाङ्क account_for_default_params() {
            constexpr पूर्णाङ्क n_योग_params = परिमाण...(Ts);
            यदि constexpr (i >= n_nondefault_params) {
                प्रत्यागम n_योग_params - n_default_params +
                       (i - n_nondefault_params);
            } अन्यथा {
                प्रत्यागम i;
            }
        }

      सार्वजनिक:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        प्रतिरूपम् <पूर्णाङ्क i>
        उपयुज्य arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        स्थिर constexpr पूर्णाङ्क params_अङ्कः = परिमाण...(Ts);

      गुप्त:
        प्रतिरूपम् <पूर्णाङ्क i> स्थिर constexpr पूर्णाङ्क get_विस्थापनम्() {
            यदि constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                प्रत्यागम 0;
            } अन्यथा {
                constexpr पूर्णाङ्क पूर्व_off = get_विस्थापनम्<i - 1>();
                constexpr पूर्णाङ्क पूर्व_past_अन्तः =
                    पूर्व_off + परिमाण(arg_type<i - 1>);
                उपयुज्य T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                यदि constexpr (पूर्व_past_अन्तः % alignof(T) == 0) {
                    प्रत्यागम पूर्व_past_अन्तः;
                }
                // otherwise bump prev_past_end to match alignment
                अन्यथा {
                    प्रत्यागम पूर्व_past_अन्तः +
                           (alignof(T) - (पूर्व_past_अन्तः % alignof(T)));
                }
            }
        }

        स्थिर वर्ण *get_तर्काः_बफर(शून्य **extra) {
            यदि (!extra)
                प्रत्यागम शून्यसूचकः;
            प्रति (; (std::size_t)*extra != 0; ++extra) {
                यदि ((std::size_t)*extra == 1) {
                    प्रत्यागम static_cast<वर्ण *>(*(extra + 1));
                }
            }
            प्रत्यागम शून्यसूचकः;
        }

      सार्वजनिक:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        तर्काः_selector(शून्य **kernel_params, शून्य **extra)
            : kernel_params(kernel_params),
              तर्काः_बफर(get_तर्काः_बफर(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        प्रतिरूपम् <पूर्णाङ्क i> arg_type<i> &get() {
            यदि (kernel_params) {
                प्रत्यागम *static_cast<arg_type<i> *>(kernel_params[i]);
            } अन्यथा {
                प्रत्यागम *reinterpret_cast<arg_type<i> *>(तर्काः_बफर +
                                                        get_विस्थापनम्<i>());
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
    वर्ग kernel_launcher {
        प्रतिरूपम् <प्रकारनाम FuncT, प्रकारनाम ArgSelector, std::size_t... सूचकांक_2>
        स्थिर शून्य launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<सूचकांक_2...>) {
            func(selector.प्रतिरूपम् get<सूचकांक_2>()...);
        }
        स्थिर शून्य समुच्चयः_execution_config(dim3 गणः_range, dim3 local_range,
                                         अचिह्नित पूर्णाङ्क local_mem_आकार_2,
                                         queue_ptr que) {
            यदि (que) {
                _que = que;
            } अन्यथा {
                _que = &get_default_queue();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(गणः_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_आकार = local_mem_आकार_2;


        }
        स्थिर अन्तःस्थ std::mutex kernel_function_ptr_map_mutex;

      सार्वजनिक:
        /// Variables for storing execution configuration.
        स्थिर अन्तःस्थ thread_local sycl::queue *_que = शून्यसूचकः;
        स्थिर अन्तःस्थ thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        स्थिर अन्तःस्थ thread_local अचिह्नित पूर्णाङ्क local_mem_आकार = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        स्थिर अन्तःस्थ std::map<
            ध्रुव शून्य *,
            std::function<शून्य(dim3, dim3, शून्य **, अचिह्नित पूर्णाङ्क, queue_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        स्थिर शून्य पञ्जिकाकोशः_kernel_ptr(
            ध्रुव शून्य *func,
            std::function<शून्य(dim3, dim3, शून्य **, अचिह्नित पूर्णाङ्क, queue_ptr)>
                launcher) {
            std::lock_guard<std::mutex> तालकम्(kernel_function_ptr_map_mutex);
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
        प्रतिरूपम् <प्रकारनाम FuncT, प्रकारनाम... तर्काः_T>
        स्थिर std::enable_if_t<std::is_invocable_v<FuncT *, तर्काः_T...>, शून्य>
        launch(FuncT *func, dim3 गणः_range, dim3 local_range,
               अचिह्नित पूर्णाङ्क local_mem_आकार_2, queue_ptr que, तर्काः_T... तर्काः_2) {
            समुच्चयः_execution_config(गणः_range, local_range, local_mem_आकार_2, que);
            func(तर्काः_2...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        स्थिर शून्य launch(ध्रुव शून्य *func, dim3 गणः_range, dim3 local_range,
                           शून्य **तर्काः_2, अचिह्नित पूर्णाङ्क local_mem_आकार_2,
                           queue_ptr que) {
            std::lock_guard<std::mutex> तालकम्(kernel_function_ptr_map_mutex);
            स्वचालित Iter = kernel_function_ptr_map.find(func);
            यदि (Iter == kernel_function_ptr_map.end()) {
                क्षिप std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(गणः_range, local_range, तर्काः_2, local_mem_आकार_2, que);
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
        प्रतिरूपम् <प्रकारनाम FuncT>
        स्थिर std::enable_if_t<std::is_function_v<FuncT>, शून्य>
        launch(FuncT *func, dim3 गणः_range, dim3 local_range, शून्य **तर्काः_2,
               अचिह्नित पूर्णाङ्क local_mem_आकार_2, queue_ptr que) {
            constexpr size_t p_अङ्कः = तर्काः_selector<0, 0, FuncT>::params_अङ्कः;
            समुच्चयः_execution_config(गणः_range, local_range, local_mem_आकार_2, que);
            तर्काः_selector<p_अङ्कः, p_अङ्कः, FuncT> selector(तर्काः_2, शून्यसूचकः);
            launch_helper(func, selector, std::make_index_sequence<p_अङ्कः>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    प्रतिरूपम् <प्रकारनाम T>
    T select_from_sub_गणः(
        sycl::sub_गणः g,
        T x,
        पूर्णाङ्क remote_local_id,
        पूर्णाङ्क logical_sub_गणः_आकार = 32) {
      अचिह्नित पूर्णाङ्क start_सूचकांक = g.get_local_रैखिकम्_id() /
                                 logical_sub_गणः_आकार *
                                 logical_sub_गणः_आकार;
      प्रत्यागम sycl::select_from_गणः(
          g, x, start_सूचकांक + remote_local_id % logical_sub_गणः_आकार);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    प्रतिरूपम् <प्रकारनाम T>
    शून्य ldmatrix(uintptr_t addr, T* m, तार्किक trans = असत्य, अचिह्नित mat = 0) {
      स्वचालित sg = sycl::ext::oneapi::this_work_item::get_sub_गणः();
      पूर्णाङ्क lane = sg.get_local_रैखिकम्_id();

      पूर्णाङ्क lane_group8_row = lane / 8;
      पूर्णाङ्क lane_group8_col = lane % 8;

      यदि (!trans) {
        // calculate the source lane
        पूर्णाङ्क src_lane = 2 * lane_group8_row;
        यदि (lane_group8_col >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        स्वचालित recv_addr_uintp =
            dpct::select_from_sub_गणः(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        स्वचालित recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_col % 4];
      } अन्यथा {
        // calculate the source lane
        पूर्णाङ्क src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        स्वचालित recv_addr_uintp_1 =
            dpct::select_from_sub_गणः(sg, addr, mat * 8 + src_lane);
        स्वचालित recv_addr_uintp_2 =
            dpct::select_from_sub_गणः(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        स्वचालित recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        स्वचालित recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        पूर्णाङ्क सूचकांक = lane / 4;
        sycl::half val0 = recv_addr_1[सूचकांक];
        sycl::half val1 = recv_addr_2[सूचकांक];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    प्रतिरूपम् <प्रकारनाम T>
    शून्य ldmatrix(uintptr_t addr, T* m1, T* m2, तार्किक trans = असत्य) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    प्रतिरूपम् <प्रकारनाम T>
    शून्य ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, तार्किक trans = असत्य) {
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
    प्रतिरूपम् <प्रकारनाम T>
    संरचना MMAType {
      उपयुज्य PackType = uint32_t;
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
    प्रतिरूपम् <पूर्णाङ्क M, पूर्णाङ्क N, पूर्णाङ्क K, प्रकारनाम ABType, प्रकारनाम CDType>
    शून्य mma(
        चञ्चल शून्य** d_mat_frag,
        शून्य* a_mat_frag,
        शून्य* b_mat_frag,
        शून्य* c_mat_frag) {
      स्वचालित d = reinterpret_cast<चञ्चल CDType**>(d_mat_frag);
      स्वचालित a =
          reinterpret_cast<प्रकारनाम MMAType<ABType>::PackType*>(a_mat_frag);
      स्वचालित b =
          reinterpret_cast<प्रकारनाम MMAType<ABType>::PackType*>(b_mat_frag);
      स्वचालित c = reinterpret_cast<CDType*>(c_mat_frag);

      स्वचालित sg = sycl::ext::oneapi::this_work_item::get_sub_गणः();
      पूर्णाङ्क lane = sg.get_local_रैखिकम्_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      ह्रस्व row_आरोपय_विस्थापनम् = 4 * (lane >> 2);
      ह्रस्व col_आरोपय_विस्थापनम् = 8 * (lane % 4);

      यदि constexpr (M == 8 && N == 8 && K == 4) {
        यदि constexpr (std::is_floating_point_v<CDType>) {
          col_आरोपय_विस्थापनम् = row_आरोपय_विस्थापनम् % 16;

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
          पूर्णाङ्क r_ind = (lane % 2) ? 1 : 0;
          पूर्णाङ्क c_ind = ((lane % 4) / 2) ? 2 : 0;

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
          प्रकारनाम MMAType<ABType>::PackType recv_a[2], recv_b[2];

          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_गणः(sg, a[0], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_गणः(sg, a[1], row_आरोपय_विस्थापनम् + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[1], col_आरोपय_विस्थापनम् + i);

            स्वचालित ra = reinterpret_cast<ABType*>(recv_a);
            स्वचालित rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment (for
            // even work item indices) d0 += col0{ a0 } * row0{ b0 } d1 += col0{
            // a0 } * row0{ b1 } d2 += col1{ a2 } * row0{ b0 } d3 += col1{ a2 }
            // * row0{ b1 } (for odd work item indices) d0 += col0{ a1 } * row0{
            // b2 } d1 += col0{ a1 } * row0{ b3 } d2 += col1{ a3 } * row0{ b2 }
            // d3 += col1{ a3 } * row0{ b3 }
            *d[0] +=
                static_cast<भिन्न>(ra[r_ind]) * static_cast<भिन्न>(rb[c_ind]);
            *d[1] += static_cast<भिन्न>(ra[r_ind]) *
                     static_cast<भिन्न>(rb[c_ind + 1]);
            *d[2] += static_cast<भिन्न>(ra[r_ind + 2]) *
                     static_cast<भिन्न>(rb[c_ind]);
            *d[3] += static_cast<भिन्न>(ra[r_ind + 2]) *
                     static_cast<भिन्न>(rb[c_ind + 1]);

            // Load partial fragment from row1 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[1], col_आरोपय_विस्थापनम् + i + 16);

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
                static_cast<भिन्न>(ra[r_ind]) * static_cast<भिन्न>(rb[c_ind]);
            *d[5] += static_cast<भिन्न>(ra[r_ind]) *
                     static_cast<भिन्न>(rb[c_ind + 1]);
            *d[6] += static_cast<भिन्न>(ra[r_ind + 2]) *
                     static_cast<भिन्न>(rb[c_ind]);
            *d[7] += static_cast<भिन्न>(ra[r_ind + 2]) *
                     static_cast<भिन्न>(rb[c_ind + 1]);
          }
        }
      } अन्यथा यदि constexpr (M == 8 && N == 8 && K == 16) {
        यदि constexpr (std::is_integral_v<ABType>) {
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
          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            प्रकारनाम MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_गणः(sg, a[0], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i + 4);

            स्वचालित a = reinterpret_cast<ABType*>(&recv_a);
            स्वचालित b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            प्रति (पूर्णाङ्क j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } अन्यथा यदि constexpr (M == 16 && N == 8 && K == 8) {
        यदि constexpr (std::is_floating_point_v<CDType>) {
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
          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            प्रकारनाम MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_गणः(sg, a[0], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_गणः(sg, a[1], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i + 4);

            स्वचालित ra = reinterpret_cast<ABType*>(recv_a);
            स्वचालित rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            प्रति (पूर्णाङ्क j = 0; j < 2; j++) {
              *d[0] += static_cast<भिन्न>(ra[j]) * static_cast<भिन्न>(rb[j]);
              *d[1] +=
                  static_cast<भिन्न>(ra[j]) * static_cast<भिन्न>(rb[j + 2]);
              *d[2] +=
                  static_cast<भिन्न>(ra[j + 2]) * static_cast<भिन्न>(rb[j]);
              *d[3] +=
                  static_cast<भिन्न>(ra[j + 2]) * static_cast<भिन्न>(rb[j + 2]);
            }
          }
        }
      } अन्यथा यदि constexpr (M == 16 && N == 8 && K == 16) {
        यदि constexpr (std::is_floating_point_v<CDType>) {
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
          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            प्रकारनाम MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_गणः(sg, a[0], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_गणः(sg, a[2], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_गणः(sg, a[1], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_गणः(sg, a[3], row_आरोपय_विस्थापनम् + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[1], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_गणः(sg, b[1], col_आरोपय_विस्थापनम् + 4 + i);

            स्वचालित ra = reinterpret_cast<ABType*>(recv_a);
            स्वचालित rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            प्रति (पूर्णाङ्क j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } अन्यथा यदि constexpr (std::is_integral_v<ABType>) {
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
          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            प्रकारनाम MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_गणः(sg, a[0], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_गणः(sg, a[1], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i + 4);

            स्वचालित ra = reinterpret_cast<ABType*>(recv_a);
            स्वचालित rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } अन्यथा यदि constexpr (M == 16 && N == 8 && K == 32) {
        यदि constexpr (std::is_integral_v<ABType>) {
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
          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            प्रकारनाम MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_गणः(sg, a[0], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_गणः(sg, a[1], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[0], col_आरोपय_विस्थापनम् + i + 4);

            स्वचालित a = reinterpret_cast<ABType*>(recv_a);
            स्वचालित b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            प्रति (पूर्णाङ्क j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            प्रकारनाम MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_गणः(sg, a[2], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_गणः(sg, a[3], row_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_गणः(sg, b[1], col_आरोपय_विस्थापनम् + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_गणः(sg, b[1], col_आरोपय_विस्थापनम् + i + 4);

            स्वचालित a = reinterpret_cast<ABType*>(recv_a);
            स्वचालित b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            प्रति (पूर्णाङ्क j = 0; j < 4; j++) {
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

#यदि_अन्तः // GGML_SYCL_DPCT_HELPER_HPP

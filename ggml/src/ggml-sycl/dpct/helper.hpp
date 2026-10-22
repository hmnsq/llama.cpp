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

#جے_بنا_تعریف GGML_SYCL_DPCT_HELPER_HPP
#تعریف_کرو GGML_SYCL_DPCT_HELPER_HPP

#ضمن <cstdlib>
#ضمن <iostream>
#ضمن <map>

#ضمن <sycl/sycl.hpp>
#ضمن <sycl/half_type.hpp>
#ضمن <oneapi/mkl.hpp>

#إذا defined(__linux__)
#ضمن <sys/mman.h>
#وإلاإذا defined(_WIN64)
#جے_بنا_تعریف NOMINMAX
#تعریف_کرو NOMINMAX
#جے_انت
#ضمن <windows.h>
#وإلا
#غلطی "Only support Windows and Linux."
#جے_انت

#إذا defined(__linux__)
#ضمن <unistd.h>
#ضمن <sys/syscall.h>
#جے_انت
#إذا defined(_WIN64)
#جے_بنا_تعریف NOMINMAX
#تعریف_کرو NOMINMAX
#جے_انت
#ضمن <windows.h>
#جے_انت


#تعریف_کرو DPCT_COMPATIBILITY_عارضی (900)

#إذا defined(_MSC_VER)
#تعریف_کرو __dpct_align__(n) __declspec(align(n))
#تعریف_کرو __dpct_inline__ __forceinline
#وإلا
#تعریف_کرو __dpct_align__(n) __attribute__((aligned(n)))
#تعریف_کرو __dpct_inline__ __inline__ __attribute__((always_inline))
#جے_انت

#إذا defined(_MSC_VER)
#تعریف_کرو __dpct_noinline__ __declspec(noinline)
#وإلا
#تعریف_کرو __dpct_noinline__ __attribute__((noinline))
#جے_انت

#تعریف_کرو DPCT_بن_ورتیا(x) (فراغ)(x)

[[noreturn]] مضمن فراغ _abort(ثابت حرف * str) {
    std::cerr << str << std::سطر_انت;
    std::abort();
}

مضمن std::string get_device_type_name(ثابت sycl::device &Device) {
    خودکار DeviceType = Device.get_info<sycl::info::device::device_type>();
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

سانچا <قسم_ناں Ts> بنية میٹرکس_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     قدر_info[2];
    std::int64_t           ناپ_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

نطاق dpct
{
    تعريفنوع sycl::queue *قطار_ptr;
    تعريفنوع sycl::event *event_ptr;
    تعريفنوع حرف *device_ptr;
    تعريفنوع uint8_t byte_t;
    تعريفنوع sycl::بفر<byte_t> بفر_t;

    /// SYCL default exception handler
    مضمن خودکار exception_handler = [](sycl::exception_list exceptions)
    {
        لكل (std::exception_ptr ثابت &e : exceptions)
        {
            حاول
            {
                std::rethrow_exception(e);
            }
            امسك (sycl::exception ثابت &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::سطر_انت
                          << e.what() << std::سطر_انت
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::سطر_انت;
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
        host_منزل_host,
        host_منزل_device,
        device_منزل_host,
        device_منزل_device,
        automatic
    };

    تعداد حافظہ_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    تعداد صنف library_انکڑے_t : بنانشانی حرف
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
        library_انکڑے_t_ناپ
    };

    سانچا <قسم_ناں T>
    بنية انکڑے_Type_2
    {
        ورتوں T2 = T;
    };
    سانچا <قسم_ناں T>
    بنية انکڑے_Type_2<sycl::vec<T, 2>>
    {
        ورتوں T2 = std::complex<T>;
    };

    ساكن فراغ destroy_event(event_ptr event)
    {
        مٹاؤ event;
    }

    ساكن مضمن بنانشانی صحيح get_tid()
    {
#إذا defined(__linux__)
        أرجع syscall(SYS_gettid);
#وإلاإذا defined(_WIN64)
        أرجع GetCurrentThreadId();
#وإلا
#غلطی "Only support Windows and Linux."
#جے_انت
    }

    نطاق detail
    {
        ساكن فراغ get_نسخہ(ثابت sycl::device &dev, صحيح &major, صحيح &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::نسخہ_2>();
            std::string::size_type i = 0;
            طالما (i < ver.size()) {
              إذا (ہندسہ_اے(ver[i]))
                اكسر;
              i++;
            }
            major = std::stoi(&(ver[i]));
            طالما (i < ver.size()) {
              إذا (ver[i] == '.')
                اكسر;
              i++;
            }
            إذا (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } وإلا {
              // c.
              minor = 0;
            }
        }

        سانچا <قسم_ناں ٹیگ, قسم_ناں T>
        صنف generic_error_type
        {
        عام:
            generic_error_type() = افتراضي;
            generic_error_type(T value) : value{value} {}
            عامل T() ثابت { أرجع value; }

        خاص:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    صنف dim3 {
        عام:
        بنانشانی x, y, z;

        constexpr dim3(بنانشانی x = 1, بنانشانی y = 1, بنانشانی z = 1)
            : x(x), y(y), z(z) {}

        dim3(ثابت sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        عامل sycl::range<3>() ثابت { أرجع sycl::range<3>(z, y, x); }
    }; // namespace dim3

    مضمن dim3 عامل*(ثابت dim3 &a, ثابت dim3 &b) {
    أرجع dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    صنف pitched_انکڑے
    {
    عام:
        pitched_انکڑے() : pitched_انکڑے(خالی_سوچک, 0, 0, 0) {}
        pitched_انکڑے(فراغ *انکڑے, size_t pitch, size_t x, size_t y)
            : انکڑے_2(انکڑے), _pitch(pitch), _x(x), _y(y) {}

        فراغ *get_انکڑے_ptr() { أرجع انکڑے_2; }
        فراغ سیٹ_انکڑے_ptr(فراغ *انکڑے) { انکڑے_2 = انکڑے; }

        size_t get_pitch() { أرجع _pitch; }
        فراغ سیٹ_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { أرجع _x; }
        فراغ سیٹ_x(size_t x) { _x = x; }

        size_t get_y() { أرجع _y; }
        فراغ سیٹ_y(size_t y) { _y = y; }

    خاص:
        فراغ *انکڑے_2;
        size_t _pitch, _x, _y;
    };

    صنف device_info
    {
    عام:
        // get interface
        ثابت حرف *get_name() ثابت { أرجع _name; }
        حرف *get_name() { أرجع _name; }
        سانچا <قسم_ناں Work_شے_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_شے_Sizes_Ty, صحيح *>,
                                   صحيح> = 0>
        خودکار get_max_work_شے_sizes() ثابت
        {
            إذا constexpr (std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>>)
                أرجع sycl::range<3>(max_work_شے_sizes_i[0],
                                      max_work_شے_sizes_i[1],
                                      max_work_شے_sizes_i[2]);
            وإلا
            {
                أرجع max_work_شے_sizes_i;
            }
        }
        سانچا <قسم_ناں Work_شے_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_شے_Sizes_Ty, صحيح *>,
                                   صحيح> = 0>
        خودکار get_max_work_شے_sizes()
        {
            إذا constexpr (std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>>)
                أرجع sycl::range<3>(max_work_شے_sizes_i[0],
                                      max_work_شے_sizes_i[1],
                                      max_work_شے_sizes_i[2]);
            وإلا
            {
                أرجع max_work_شے_sizes_i;
            }
        }
        منطقي get_host_unified_حافظہ() ثابت { أرجع host_unified_حافظہ; }
        صحيح get_major_نسخہ() ثابت { أرجع _major; }
        صحيح get_minor_نسخہ() ثابت { أرجع _minor; }
        صحيح get_integrated() ثابت { أرجع _integrated; }
        صحيح get_max_clock_frequency() ثابت { أرجع _frequency; }
        صحيح get_max_compute_units() ثابت { أرجع _max_compute_units; }
        صحيح get_max_work_گروہ_ناپ() ثابت { أرجع max_work_گروہ_ناپ_2; }
        صحيح get_max_sub_گروہ_ناپ() ثابت { أرجع max_sub_گروہ_ناپ; }
        صحيح get_max_work_اشیاء_فی_compute_unit() ثابت
        {
            أرجع max_work_اشیاء_فی_compute_unit;
        }
        صحيح get_max_رجسٹر_ناپ_فی_work_گروہ() ثابت
        {
            أرجع max_رجسٹر_ناپ_فی_work_گروہ;
        }
        سانچا <قسم_ناں NDRange_ناپ_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_ناپ_Ty, size_t *> ||
                                       std::is_same_v<NDRange_ناپ_Ty, صحيح *>,
                                   صحيح> = 0>
        خودکار get_max_nd_range_ناپ() ثابت
        {
            إذا constexpr (std::is_same_v<NDRange_ناپ_Ty, size_t *>)
                أرجع max_nd_range_ناپ;
            وإلا
                أرجع max_nd_range_ناپ_i;
        }
        سانچا <قسم_ناں NDRange_ناپ_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_ناپ_Ty, size_t *> ||
                                       std::is_same_v<NDRange_ناپ_Ty, صحيح *>,
                                   صحيح> = 0>
        خودکار get_max_nd_range_ناپ()
        {
            إذا constexpr (std::is_same_v<NDRange_ناپ_Ty, size_t *>)
                أرجع max_nd_range_ناپ;
            وإلا
                أرجع max_nd_range_ناپ_i;
        }
        size_t get_global_mem_ناپ() ثابت { أرجع global_mem_ناپ_2; }
        size_t get_local_mem_ناپ() ثابت { أرجع local_mem_ناپ; }
        size_t get_max_mem_alloc_ناپ() ثابت { أرجع max_mem_alloc_ناپ; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        بنانشانی صحيح get_حافظہ_clock_شرح() ثابت { أرجع حافظہ_clock_شرح; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        بنانشانی صحيح get_حافظہ_bus_چوڑائی() ثابت { أرجع حافظہ_bus_چوڑائی; }
        uint32_t get_device_id() ثابت { أرجع _device_id; }
        std::array<بنانشانی حرف, 16> get_uuid() ثابت { أرجع _uuid; }
        /// Returns global memory cache size in bytes.
        بنانشانی صحيح get_global_mem_cache_ناپ() ثابت
        {
            أرجع global_mem_cache_ناپ;
        }

        // set interface
        فراغ سیٹ_name(ثابت حرف *name)
        {
            size_t لمائی = لڑی_لمبائی(name);
            إذا (لمائی < 256)
            {
                std::حافظہ_نقل(_name, name, لمائی + 1);
            }
            وإلا
            {
                std::حافظہ_نقل(_name, name, 255);
                _name[255] = '\0';
            }
        }
        فراغ سیٹ_max_work_شے_sizes(ثابت sycl::range<3> max_work_شے_sizes)
        {
            لكل (صحيح i = 0; i < 3; ++i)
                max_work_شے_sizes_i[i] = max_work_شے_sizes[i];
        }
        [[deprecated]] فراغ
        سیٹ_max_work_شے_sizes(ثابت sycl::id<3> max_work_شے_sizes)
        {
            لكل (صحيح i = 0; i < 3; ++i)
            {
                max_work_شے_sizes_i[i] = max_work_شے_sizes[i];
            }
        }
        فراغ سیٹ_host_unified_حافظہ(منطقي host_unified_حافظہ_2)
        {
            host_unified_حافظہ = host_unified_حافظہ_2;
        }
        فراغ سیٹ_major_نسخہ(صحيح major) { _major = major; }
        فراغ سیٹ_minor_نسخہ(صحيح minor) { _minor = minor; }
        فراغ سیٹ_integrated(صحيح integrated) { _integrated = integrated; }
        فراغ سیٹ_max_clock_frequency(صحيح frequency) { _frequency = frequency; }
        فراغ سیٹ_max_compute_units(صحيح max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        فراغ سیٹ_global_mem_ناپ(size_t global_mem_ناپ)
        {
            global_mem_ناپ_2 = global_mem_ناپ;
        }
        فراغ سیٹ_local_mem_ناپ(size_t local_mem_ناپ_2)
        {
            local_mem_ناپ = local_mem_ناپ_2;
        }
        فراغ سیٹ_max_mem_alloc_ناپ(size_t max_mem_alloc_ناپ_2)
        {
            max_mem_alloc_ناپ = max_mem_alloc_ناپ_2;
        }
        فراغ سیٹ_max_work_گروہ_ناپ(صحيح max_work_گروہ_ناپ)
        {
            max_work_گروہ_ناپ_2 = max_work_گروہ_ناپ;
        }
        فراغ سیٹ_max_sub_گروہ_ناپ(صحيح max_sub_گروہ_ناپ_2)
        {
            max_sub_گروہ_ناپ = max_sub_گروہ_ناپ_2;
        }
        فراغ
        سیٹ_max_work_اشیاء_فی_compute_unit(صحيح max_work_اشیاء_فی_compute_unit_2)
        {
            max_work_اشیاء_فی_compute_unit = max_work_اشیاء_فی_compute_unit_2;
        }
        فراغ سیٹ_max_nd_range_ناپ(صحيح max_nd_range_ناپ_2[])
        {
            لكل (صحيح i = 0; i < 3; i++)
            {
                max_nd_range_ناپ[i] = max_nd_range_ناپ_2[i];
                max_nd_range_ناپ_i[i] = max_nd_range_ناپ_2[i];
            }
        }
        فراغ سیٹ_حافظہ_clock_شرح(بنانشانی صحيح حافظہ_clock_شرح_2)
        {
            حافظہ_clock_شرح = حافظہ_clock_شرح_2;
        }
        فراغ سیٹ_حافظہ_bus_چوڑائی(بنانشانی صحيح حافظہ_bus_چوڑائی_2)
        {
            حافظہ_bus_چوڑائی = حافظہ_bus_چوڑائی_2;
        }
        فراغ
        سیٹ_max_رجسٹر_ناپ_فی_work_گروہ(صحيح max_رجسٹر_ناپ_فی_work_گروہ_2)
        {
            max_رجسٹر_ناپ_فی_work_گروہ = max_رجسٹر_ناپ_فی_work_گروہ_2;
        }
        فراغ سیٹ_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        فراغ سیٹ_uuid(std::array<بنانشانی حرف, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        فراغ سیٹ_global_mem_cache_ناپ(بنانشانی صحيح global_mem_cache_ناپ_2)
        {
            global_mem_cache_ناپ = global_mem_cache_ناپ_2;
        }

    خاص:
        حرف _name[256];
        صحيح max_work_شے_sizes_i[3];
        منطقي host_unified_حافظہ = خطأ;
        صحيح _major;
        صحيح _minor;
        صحيح _integrated = 0;
        صحيح _frequency;
        // Set estimated value 3200000 kHz as default value.
        بنانشانی صحيح حافظہ_clock_شرح = 3200000;
        // Set estimated value 64 bits as default value.
        بنانشانی صحيح حافظہ_bus_چوڑائی = 64;
        بنانشانی صحيح global_mem_cache_ناپ;
        صحيح _max_compute_units;
        صحيح max_work_گروہ_ناپ_2;
        صحيح max_sub_گروہ_ناپ;
        صحيح max_work_اشیاء_فی_compute_unit;
        صحيح max_رجسٹر_ناپ_فی_work_گروہ;
        size_t global_mem_ناپ_2;
        size_t local_mem_ناپ;
        size_t max_mem_alloc_ناپ;
        size_t max_nd_range_ناپ[3];
        صحيح max_nd_range_ناپ_i[3];
        uint32_t _device_id;
        std::array<بنانشانی حرف, 16> _uuid;
    };

    ساكن صحيح get_major_نسخہ(ثابت sycl::device &dev)
    {
        صحيح major, minor;
        detail::get_نسخہ(dev, major, minor);
        أرجع major;
    }

    ساكن صحيح get_minor_نسخہ(ثابت sycl::device &dev)
    {
        صحيح major, minor;
        detail::get_نسخہ(dev, major, minor);
        أرجع minor;
    }

    ساكن فراغ get_device_info(device_info &باہرلا, ثابت sycl::device &dev)
    {
        device_info prop;
        prop.سیٹ_name(dev.get_info<sycl::info::device::name>().c_str());

        صحيح major, minor;
        detail::get_نسخہ(dev, major, minor);
        prop.سیٹ_major_نسخہ(major);
        prop.سیٹ_minor_نسخہ(minor);

        prop.سیٹ_max_work_شے_sizes(
#إذا (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_شے_sizes>());
#وإلا
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_شے_sizes<3>>());
#جے_انت
        prop.سیٹ_host_unified_حافظہ(dev.has(sycl::aspect::usm_host_allocations));

        prop.سیٹ_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.سیٹ_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.سیٹ_max_work_گروہ_ناپ(
            dev.get_info<sycl::info::device::max_work_گروہ_ناپ>());
        prop.سیٹ_global_mem_ناپ(dev.get_info<sycl::info::device::global_mem_ناپ>());
        prop.سیٹ_local_mem_ناپ(dev.get_info<sycl::info::device::local_mem_ناپ_2>());
        prop.سیٹ_max_mem_alloc_ناپ(dev.get_info<sycl::info::device::max_mem_alloc_ناپ_2>());

#إذا (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        إذا (dev.has(sycl::aspect::ext_intel_حافظہ_clock_شرح))
        {
            بنانشانی صحيح tmp =
                dev.get_info<sycl::ext::intel::info::device::حافظہ_clock_شرح_2>();
            إذا (tmp != 0)
                prop.سیٹ_حافظہ_clock_شرح(1000 * tmp);
        }
        إذا (dev.has(sycl::aspect::ext_intel_حافظہ_bus_چوڑائی))
        {
            prop.سیٹ_حافظہ_bus_چوڑائی(
                dev.get_info<sycl::ext::intel::info::device::حافظہ_bus_چوڑائی_2>());
        }
        إذا (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.سیٹ_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        إذا (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.سیٹ_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#وإلاإذا defined(_MSC_VER) && !defined(__clang__)
#ہدایت سنیہا("get_device_info: querying حافظہ_clock_شرح_2 and \
        حافظہ_bus_چوڑائی_2 are not supported by the compiler ورتیا. \
        Use 3200000 kHz as حافظہ_clock_شرح_2 افتراضي value. \
        Use 64 بٹ as حافظہ_bus_چوڑائی_2 افتراضي value.")
#وإلا
#warning "get_device_info: querying حافظہ_clock_شرح_2 and \
        حافظہ_bus_چوڑائی_2 are not supported by the compiler ورتیا. \
        Use 3200000 kHz as حافظہ_clock_شرح_2 افتراضي value. \
        Use 64 بٹ as حافظہ_bus_چوڑائی_2 افتراضي value."
#جے_انت

        size_t max_sub_گروہ_ناپ_2 = 1;
        std::vector<size_t> sub_گروہ_sizes =
            dev.get_info<sycl::info::device::sub_گروہ_sizes>();

        لكل (ثابت خودکار &sub_گروہ_ناپ : sub_گروہ_sizes)
        {
            إذا (max_sub_گروہ_ناپ_2 < sub_گروہ_ناپ)
                max_sub_گروہ_ناپ_2 = sub_گروہ_ناپ;
        }

        prop.سیٹ_max_sub_گروہ_ناپ(max_sub_گروہ_ناپ_2);

        prop.سیٹ_max_work_اشیاء_فی_compute_unit(
            dev.get_info<sycl::info::device::max_work_گروہ_ناپ>());
        صحيح max_nd_range_ناپ_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.سیٹ_max_nd_range_ناپ(max_nd_range_ناپ_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.سیٹ_max_رجسٹر_ناپ_فی_work_گروہ(65536);

        prop.سیٹ_global_mem_cache_ناپ(
            dev.get_info<sycl::info::device::global_mem_cache_ناپ_2>());
        باہرلا = prop;
    }

    /// dpct device extension
    صنف device_ext : عام sycl::device {
      تعريفنوع std::mutex mutex_type;

     عام:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        clear_queues();
      }
      device_ext(ثابت sycl::device &ادھار) : sycl::device(ادھار) {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        init_queues();
      }

      صحيح is_native_atomic_supported() { أرجع 0; }
      صحيح get_major_نسخہ() ثابت { أرجع dpct::get_major_نسخہ(*هذا); }

      صحيح get_minor_نسخہ() ثابت { أرجع dpct::get_minor_نسخہ(*هذا); }

      صحيح get_max_compute_units() ثابت {
        أرجع get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      صحيح get_max_clock_frequency() ثابت {
        أرجع get_device_info().get_max_clock_frequency();
      }

      صحيح get_integrated() ثابت { أرجع get_device_info().get_integrated(); }

      صحيح get_max_sub_گروہ_ناپ() ثابت {
        أرجع get_device_info().get_max_sub_گروہ_ناپ();
      }

      صحيح get_max_رجسٹر_ناپ_فی_work_گروہ() ثابت {
        أرجع get_device_info().get_max_رجسٹر_ناپ_فی_work_گروہ();
      }

      صحيح get_max_work_گروہ_ناپ() ثابت {
        أرجع get_device_info().get_max_work_گروہ_ناپ();
      }

      صحيح get_mem_ادھار_addr_align() ثابت {
        أرجع get_info<sycl::info::device::mem_ادھار_addr_align>();
      }

      size_t get_global_mem_ناپ() ثابت {
        أرجع get_device_info().get_global_mem_ناپ();
      }

      size_t get_max_mem_alloc_ناپ() ثابت {
        أرجع get_device_info().get_max_mem_alloc_ناپ();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      فراغ get_حافظہ_info(size_t &free_حافظہ, size_t &کل_حافظہ) {
        کل_حافظہ = get_device_info().get_global_mem_ناپ();
        ثابت حرف *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#إذا (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        إذا (!has(sycl::aspect::ext_intel_free_حافظہ)) {
          std::cerr << warning_info << std::سطر_انت;
          free_حافظہ = کل_حافظہ;
        } وإلا {
          free_حافظہ = get_info<sycl::ext::intel::info::device::free_حافظہ>();
        }
#وإلا
        std::cerr << warning_info << std::سطر_انت;
        free_حافظہ = کل_حافظہ;
#إذا defined(_MSC_VER) && !defined(__clang__)
#ہدایت سنیہا("Querying the number of bytes of free memory is not supported")
#وإلا
#warning "Querying the number of bytes of free memory is not supported"
#جے_انت
#جے_انت
      }

      فراغ get_device_info(device_info &باہرلا) ثابت {
        dpct::get_device_info(باہرلا, *هذا);
      }

      device_info get_device_info() ثابت {
        device_info prop;
        dpct::get_device_info(prop, *هذا);
        أرجع prop;
      }

      فراغ reset() {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &وچ_ترتیب_قطار() { أرجع q_وچ_ترتیب; }

      sycl::queue &باہرلا_دا_ترتیب_قطار() { أرجع q_باہرلا_دا_ترتیب; }

      sycl::queue &default_قطار() { أرجع وچ_ترتیب_قطار(); }

      فراغ queues_wait_and_throw() {
        std::unique_lock<mutex_type> جندرا(m_mutex);
        جندرا.unlock();
        لكل (خودکار &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        جندرا.جندرا();
      }

      sycl::queue create_قطار(منطقي enable_exception_handler = خطأ) {
        أرجع create_وچ_ترتیب_قطار(enable_exception_handler);
      }

      sycl::queue create_قطار(sycl::device device,
                               منطقي enable_exception_handler = خطأ) {
        أرجع create_وچ_ترتیب_قطار(device, enable_exception_handler);
      }

      sycl::queue create_وچ_ترتیب_قطار(منطقي enable_exception_handler = خطأ) {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        أرجع create_قطار_impl(enable_exception_handler,
                                 sycl::property::queue::وچ_ترتیب());
      }

      sycl::queue create_وچ_ترتیب_قطار(sycl::device device,
                                        منطقي enable_exception_handler = خطأ) {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        أرجع create_قطار_impl(device, enable_exception_handler,
                                 sycl::property::queue::وچ_ترتیب());
      }

      sycl::queue create_باہرلا_دا_ترتیب_قطار(
          منطقي enable_exception_handler = خطأ) {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        أرجع create_قطار_impl(enable_exception_handler);
      }

      فراغ destroy_قطار(sycl::queue queue) {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ثابت sycl::queue &q) -> منطقي
                                    {
                                        أرجع q == queue;
                                    }),
                    _queues.end());
      }
      فراغ سیٹ_saved_قطار(sycl::queue q) {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        saved_قطار = q;
      }
      sycl::queue get_saved_قطار() ثابت {
        std::lock_guard<mutex_type> جندرا(m_mutex);
        أرجع saved_قطار;
      }

     خاص:
      فراغ clear_queues() { _queues.clear(); }

      فراغ init_queues() {
        q_وچ_ترتیب =
            create_قطار_impl(سچ, sycl::property::queue::وچ_ترتیب());
        q_باہرلا_دا_ترتیب = create_قطار_impl(سچ);
        saved_قطار = default_قطار();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      سانچا <صنف... Properties>
      sycl::queue create_قطار_impl(منطقي enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        إذا (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *هذا, eh,
            sycl::property_list(
#جے_تعریف_شدہ DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#جے_انت
                properties...)));

        أرجع _queues.back();
      }

      سانچا <صنف... Properties>
      sycl::queue create_قطار_impl(sycl::device device,
                                    منطقي enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        إذا (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#جے_تعریف_شدہ DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#جے_انت
                            properties...)));

        أرجع _queues.back();
      }

      فراغ get_نسخہ(صحيح &major, صحيح &minor) ثابت {
        detail::get_نسخہ(*هذا, major, minor);
      }
      sycl::queue q_وچ_ترتیب, q_باہرلا_دا_ترتیب;
      sycl::queue saved_قطار;
      std::vector<sycl::queue> _queues;
      بدلنجوگ mutex_type m_mutex;
    };


    /// device manager
    صنف dev_mgr
    {
    عام:
        device_ext &current_device()
        {
            بنانشانی صحيح dev_id = current_device_id();
            check_id(dev_id);
            أرجع *_devs[dev_id];
        }
        device_ext &cpu_device() ثابت
        {
            std::lock_guard<std::recursive_mutex> جندرا(m_mutex);
            إذا (_cpu_device == -1)
            {
                ارم std::runtime_error("no valid cpu device");
            }
            وإلا
            {
                أرجع *_devs[_cpu_device];
            }
        }
        device_ext &get_device(بنانشانی صحيح id) ثابت
        {
            std::lock_guard<std::recursive_mutex> جندرا(m_mutex);
            check_id(id);
            أرجع *_devs[id];
        }
        بنانشانی صحيح current_device_id() ثابت
        {
            std::lock_guard<std::recursive_mutex> جندرا(m_mutex);
            خودکار it = _thread2dev_map.find(get_tid());
            إذا (it != _thread2dev_map.end())
                أرجع it->second;
            أرجع DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        فراغ select_device(بنانشانی صحيح id)
        {
            std::lock_guard<std::recursive_mutex> جندرا(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        بنانشانی صحيح device_گنتی() { أرجع _devs.size(); }

        بنانشانی صحيح get_device_id(ثابت sycl::device &dev)
        {
            بنانشانی صحيح id = 0;
            لكل (خودکار &dev_شے : _devs)
            {
                إذا (*dev_شے == dev)
                {
                    أرجع id;
                }
                id++;
            }
            أرجع -1;
        }

        مضمن std::string get_preferred_gpu_platform_name() {
            std::string نتیجہ;

            std::string filter = "";
            حرف* env = getenv("ONEAPI_DEVICE_SELECTOR");
            إذا (env) {
                إذا (std::لڑی_کھوج(env, "level_zero")) {
                    filter = "level-zero";
                }
                وإلا إذا (std::لڑی_کھوج(env, "opencl")) {
                    filter = "opencl";
                }
                وإلا إذا (std::لڑی_کھوج(env, "cuda")) {
                    filter = "cuda";
                }
                وإلا إذا (std::لڑی_کھوج(env, "hip")) {
                    filter = "hip";
                }
                وإلا {
                    ارم std::runtime_error("invalid device filter: " + std::string(env));
                }
            } وإلا {
                خودکار default_device = sycl::device(sycl::default_selector_v);
                خودکار default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                إذا (std::لڑی_کھوج(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                وإلا إذا (std::لڑی_کھوج(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                وإلا إذا (std::لڑی_کھوج(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            خودکار platform_list = sycl::platform::get_platforms();

            لكل (ثابت خودکار& platform : platform_list) {
                خودکار devices = platform.get_devices();
                خودکار gpu_dev = std::find_if(devices.begin(), devices.end(), [](ثابت sycl::device& d) {
                    أرجع d.is_gpu();
                });

                إذا (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    استمر;
                }

                خودکار platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_کم_case;
                platform_name_کم_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_کم_case.begin(), ::tolower);

                إذا (platform_name_کم_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    استمر;
                }

                نتیجہ = platform_name;
            }

            إذا (نتیجہ.خالی_تھاواں())
                ارم std::runtime_error("can not find preferred GPU platform");

            أرجع نتیجہ;
        }

        سانچا <صنف DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<صحيح, DeviceSelector, ثابت sycl::device &>>
        select_device(ثابت DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            بنانشانی صحيح selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ساكن dev_mgr &instance()
        {
            ساكن dev_mgr d_m;
            أرجع d_m;
        }
        dev_mgr(ثابت dev_mgr &) = مٹاؤ;
        dev_mgr &عامل=(ثابت dev_mgr &) = مٹاؤ;
        dev_mgr(dev_mgr &&) = مٹاؤ;
        dev_mgr &عامل=(dev_mgr &&) = مٹاؤ;

    خاص:
        بدلنجوگ std::recursive_mutex m_mutex;
        ساكن منطقي compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            إذا(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) أرجع سچ;
            إذا(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) أرجع خطأ;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            أرجع prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ساكن صحيح convert_backend_اشاریہ(std::string & backend) {
            إذا (backend == "ext_oneapi_level_zero:gpu") أرجع 0;
            إذا (backend == "opencl:gpu") أرجع 1;
            إذا (backend == "ext_oneapi_cuda:gpu") أرجع 2;
            إذا (backend == "ext_oneapi_hip:gpu") أرجع 3;
            إذا (backend == "opencl:cpu") أرجع 4;
            إذا (backend == "opencl:acc") أرجع 5;
            لکھو("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ساكن منطقي compare_backend(std::string &backend1, std::string &backend2) {
            أرجع convert_backend_اشاریہ(backend1) < convert_backend_اشاریہ(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            إذا (default_device.is_cpu())
                _cpu_device = 0;

            خودکار Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            خودکار preferred_platform_name = get_preferred_gpu_platform_name();

            طالما (!Platforms.خالی_تھاواں()) {
                خودکار Platform = Platforms.back();
                Platforms.pop_back();
                خودکار platform_name = Platform.get_info<sycl::info::platform::name>();
                إذا (platform_name.compare(preferred_platform_name) != 0) {
                    استمر;
                }
                خودکار devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                لكل (ثابت خودکار &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> کلیدیں;
            لكل(خودکار it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                کلیدیں.push_back(it->پہلا);
            }
            std::sort(کلیدیں.begin(), کلیدیں.end(), compare_backend);

            لكل (خودکار &کلید : کلیدیں) {
                std::vector<sycl::device> devs = backend_devices[کلید];
                std::sort(devs.begin(), devs.end(), compare_dev);
                لكل (ثابت خودکار &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            لكل (خودکار &dev : sycl_all_devs)
            {
                إذا (dev == default_device)
                {
                    استمر;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                إذا (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        فراغ check_id(بنانشانی صحيح id) ثابت
        {
            إذا (id >= _devs.size())
            {
                ارم std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ثابت بنانشانی صحيح DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<بنانشانی صحيح, بنانشانی صحيح> _thread2dev_map;
        صحيح _cpu_device = -1;
    };

    ساكن مضمن sycl::queue &get_default_قطار()
    {
        أرجع dev_mgr::instance().current_device().default_قطار();
    }

    نطاق detail
    {
        تعداد صنف اشارہ_گر_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ساكن اشارہ_گر_access_attribute get_اشارہ_گر_attribute(sycl::queue &q,
                                                              ثابت فراغ *ptr)
        {
            بدل (sycl::get_اشارہ_گر_type(ptr, q.get_context()))
            {
            حالة sycl::usm::alloc::unknown:
                أرجع اشارہ_گر_access_attribute::host_only;
            حالة sycl::usm::alloc::device:
                أرجع اشارہ_گر_access_attribute::device_only;
            حالة sycl::usm::alloc::shared:
            حالة sycl::usm::alloc::host:
                أرجع اشارہ_گر_access_attribute::host_device;
            }
        }

        سانچا <قسم_ناں ArgT>
        مضمن constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((بنانشانی حرف)library_انکڑے_t::library_انکڑے_t_ناپ <=
                              std::numeric_limits<بنانشانی حرف>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_انکڑے_t>, "Unsupported ArgT");
            أرجع (std::uint64_t)Val;
        }

        سانچا <قسم_ناں پہلا_T, قسم_ناں... RestT>
        مضمن constexpr std::uint64_t get_type_combination_id(پہلا_T پہلا_Val,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_انکڑے_t::library_انکڑے_t_ناپ <=
                              std::numeric_limits<بنانشانی حرف>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(حجم...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<پہلا_T, library_انکڑے_t>, "Unsupported FirstT");
            أرجع get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)پہلا_Val);
        }

        صنف mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#إذا defined(__linux__)
                mapped_پتہ_space =
                    (byte_t *)mmap(خالی_سوچک, mapped_region_ناپ, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#وإلاإذا defined(_WIN64)
                mapped_پتہ_space = (byte_t *)VirtualAlloc(
                    عدم,               // NULL specified as the base address parameter
                    mapped_region_ناپ, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    ورقہ_NOACCESS);     // Protection = no access
#وإلا
#غلطی "Only support Windows and Linux."
#جے_انت
                اگلا_free = mapped_پتہ_space;
            }

        عام:
            ورتوں بفر_id_t = صحيح;

            بنية ونڈ
            {
                بفر_t بفر;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#إذا defined(__linux__)
                munmap(mapped_پتہ_space, mapped_region_ناپ);
#وإلاإذا defined(_WIN64)
                VirtualFree(mapped_پتہ_space, 0, MEM_RELEASE);
#وإلا
#غلطی "Only support Windows and Linux."
#جے_انت
            }

            mem_mgr(ثابت mem_mgr &) = مٹاؤ;
            mem_mgr &عامل=(ثابت mem_mgr &) = مٹاؤ;
            mem_mgr(mem_mgr &&) = مٹاؤ;
            mem_mgr &عامل=(mem_mgr &&) = مٹاؤ;

            /// Allocate
            فراغ *mem_alloc(size_t size)
            {
                إذا (!size)
                    أرجع خالی_سوچک;
                std::lock_guard<std::mutex> جندرا(m_mutex);
                إذا (اگلا_free + size > mapped_پتہ_space + mapped_region_ناپ)
                {
                    ارم std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                //ونڈ
                sycl::range<1> r(size);
                بفر_t buf(r);
                ونڈ A{buf, اگلا_free, size};
                // Map allocation to device pointer
                فراغ *نتیجہ = اگلا_free;
                m_map.emplace(اگلا_free + size, A);
                // Update pointer to the next free space.
                اگلا_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                أرجع نتیجہ;
            }

            /// Deallocate
            فراغ mem_free(ثابت فراغ *ptr)
            {
                إذا (!ptr)
                    أرجع;
                std::lock_guard<std::mutex> جندرا(m_mutex);
                خودکار it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            ونڈ translate_ptr(ثابت فراغ *ptr)
            {
                std::lock_guard<std::mutex> جندرا(m_mutex);
                خودکار it = get_map_iterator(ptr);
                أرجع it->second;
            }

            /// Check if the pointer represents device pointer or not.
            منطقي is_device_ptr(ثابت فراغ *ptr) ثابت
            {
                std::lock_guard<std::mutex> جندرا(m_mutex);
                أرجع (mapped_پتہ_space <= ptr) &&
                       (ptr < mapped_پتہ_space + mapped_region_ناپ);
            }

            /// Returns the instance of memory manager singleton.
            ساكن mem_mgr &instance()
            {
                ساكن mem_mgr m;
                أرجع m;
            }

        خاص:
            std::map<byte_t *, ونڈ> m_map;
            بدلنجوگ std::mutex m_mutex;
            byte_t *mapped_پتہ_space;
            byte_t *اگلا_free;
            ثابت size_t mapped_region_ناپ = 128ull * 1024 * 1024 * 1024;
            ثابت size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ثابت size_t extra_padding = 0;

            std::map<byte_t *, ونڈ>::iterator get_map_iterator(ثابت فراغ *ptr)
            {
                خودکار it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ثابت byte_t *>(ptr)));
                إذا (it == m_map.end())
                {
                    // Not a virtual pointer.
                    ارم std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ثابت ونڈ &alloc = it->second;
                إذا (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    ارم std::runtime_error("invalid virtual pointer");
                }
                أرجع it;
            }
        };

        سانچا <صنف T, حافظہ_region حافظہ_2, size_t Dimension>
        صنف accessor;
        سانچا <حافظہ_region حافظہ_2, صنف T = byte_t>
        صنف حافظہ_traits
        {
        عام:
            ساكن constexpr sycl::access::نشانہ نشانہ =
                sycl::access::نشانہ::device;
            ساكن constexpr sycl::access_منوال mode =
                (حافظہ_2 == constant) ? sycl::access_منوال::اقرأ
                                     : sycl::access_منوال::read_write;
            ساكن constexpr size_t type_ناپ = حجم(T);
            ورتوں element_t =
                قسم_ناں std::conditional<حافظہ_2 == constant, ثابت T, T>::type;
            ورتوں قدر_t = قسم_ناں std::remove_cv<T>::type;
            سانچا <size_t Dimension = 1>
            ورتوں accessor_t = قسم_ناں std::conditional<
                حافظہ_2 == local, sycl::local_accessor<قدر_t, Dimension>,
                sycl::accessor<T, Dimension, mode, نشانہ>>::type;
            ورتوں اشارہ_گر_t = T *;
        };

        ساكن مضمن فراغ *dpct_malloc(size_t size, sycl::queue &q)
        {
            أرجع sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#تعریف_کرو PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
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
        سانچا <قسم_ناں قدر_T>
        ساكن مضمن sycl::event dpct_memset(sycl::queue &q, فراغ *dev_ptr,
                                              قدر_T value, size_t size)
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
        سانچا <قسم_ناں قدر_T>
        ساكن مضمن std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_انکڑے انکڑے, قدر_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t ٹکڑا_2 = انکڑے.get_pitch() * انکڑے.get_y();
            بنانشانی حرف *انکڑے_surface = (بنانشانی حرف *)انکڑے.get_انکڑے_ptr();
            لكل (size_t z = 0; z < size.get(2); ++z)
            {
                بنانشانی حرف *انکڑے_ptr = انکڑے_surface;
                لكل (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, انکڑے_ptr, value, size.get(0)));
                    انکڑے_ptr += انکڑے.get_pitch();
                }
                انکڑے_surface += ٹکڑا_2;
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
        سانچا <قسم_ناں قدر_T>
        ساكن مضمن std::vector<sycl::event>
        dpct_memset(sycl::queue &q, فراغ *ptr, size_t pitch, قدر_T val, size_t x,
                    size_t y)
        {
            أرجع dpct_memset(q, pitched_انکڑے(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ساكن memcpy_direction deduce_memcpy_direction(sycl::queue &q, فراغ *منزل_ptr,
                                                        ثابت فراغ *from_ptr,
                                                        memcpy_direction dir)
        {
            بدل (dir)
            {
            حالة memcpy_direction::host_منزل_host:
            حالة memcpy_direction::host_منزل_device:
            حالة memcpy_direction::device_منزل_host:
            حالة memcpy_direction::device_منزل_device:
                أرجع dir;
            حالة memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ساكن ثابت memcpy_direction
                    direction_جدول[static_cast<بنانشانی>(اشارہ_گر_access_attribute::end)]
                                   [static_cast<بنانشانی>(اشارہ_گر_access_attribute::end)] =
                                       {{memcpy_direction::host_منزل_host,
                                         memcpy_direction::device_منزل_host,
                                         memcpy_direction::host_منزل_host},
                                        {memcpy_direction::host_منزل_device,
                                         memcpy_direction::device_منزل_device,
                                         memcpy_direction::device_منزل_device},
                                        {memcpy_direction::host_منزل_host,
                                         memcpy_direction::device_منزل_device,
                                         memcpy_direction::device_منزل_device}};
                أرجع direction_جدول[static_cast<بنانشانی>(get_اشارہ_گر_attribute(
                    q, منزل_ptr))][static_cast<بنانشانی>(get_اشارہ_گر_attribute(q, from_ptr))];
            }
            افتراضي:
                ارم std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ساكن sycl::event
        dpct_memcpy(sycl::queue &q, فراغ *منزل_ptr, ثابت فراغ *from_ptr, size_t size,
                    memcpy_direction direction,
                    ثابت std::vector<sycl::event> &dep_events = {})
        {
            إذا (!size)
                أرجع sycl::event{};
            أرجع q.حافظہ_نقل(منزل_ptr, from_ptr, size, dep_events);
            DPCT_بن_ورتیا(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ساكن مضمن size_t get_copy_range(sycl::range<3> size, size_t ٹکڑا_2,
                                            size_t pitch)
        {
            أرجع ٹکڑا_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ساكن مضمن size_t get_وتھ(sycl::id<3> id, size_t ٹکڑا_2,
                                        size_t pitch)
        {
            أرجع ٹکڑا_2 * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ساكن مضمن std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, فراغ *منزل_ptr, ثابت فراغ *from_ptr,
                    sycl::range<3> منزل_range, sycl::range<3> from_range,
                    sycl::id<3> منزل_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ثابت std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            صنف host_بفر
            {
                فراغ *_buf;
                size_t ناپ_2;
                sycl::queue &_q;
                ثابت std::vector<sycl::event> &_deps; // free operation depends

            عام:
                host_بفر(size_t size, sycl::queue &q,
                            ثابت std::vector<sycl::event> &deps)
                    : _buf(std::حافظہ_دیو(size)), ناپ_2(size), _q(q), _deps(deps) {}
                فراغ *get_ptr() ثابت { أرجع _buf; }
                size_t get_ناپ() ثابت { أرجع ناپ_2; }
                ~host_بفر()
                {
                    إذا (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::حرر(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t منزل_ٹکڑا = منزل_range.get(1) * منزل_range.get(0),
                   from_ٹکڑا = from_range.get(1) * from_range.get(0);
            بنانشانی حرف *منزل_surface =
                (بنانشانی حرف *)منزل_ptr + get_وتھ(منزل_id, منزل_ٹکڑا, منزل_range.get(0));
            ثابت بنانشانی حرف *from_surface =
                (ثابت بنانشانی حرف *)from_ptr +
                get_وتھ(from_id, from_ٹکڑا, from_range.get(0));

            إذا (منزل_ٹکڑا == from_ٹکڑا && منزل_ٹکڑا == size.get(1) * size.get(0))
            {
                أرجع {dpct_memcpy(q, منزل_surface, from_surface, منزل_ٹکڑا * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, منزل_ptr, from_ptr, direction);
            size_t ناپ_ٹکڑا = size.get(1) * size.get(0);
            بدل (direction)
            {
            حالة host_منزل_host:
                لكل (size_t z = 0; z < size.get(2); ++z)
                {
                    بنانشانی حرف *منزل_ptr = منزل_surface;
                    ثابت بنانشانی حرف *from_ptr = from_surface;
                    إذا (منزل_range.get(0) == from_range.get(0) &&
                        منزل_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, منزل_ptr, from_ptr, ناپ_ٹکڑا,
                                                         direction, dep_events));
                    }
                    وإلا
                    {
                        لكل (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, منزل_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            منزل_ptr += منزل_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    منزل_surface += منزل_ٹکڑا;
                    from_surface += from_ٹکڑا;
                }
                اكسر;
            حالة host_منزل_device:
            {
                host_بفر buf(get_copy_range(size, منزل_ٹکڑا, منزل_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                إذا (منزل_ٹکڑا == ناپ_ٹکڑا)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, منزل_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_منزل_host, dep_events);
                }
                وإلا
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, منزل_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_منزل_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), منزل_surface, buf.get_ناپ(),
                                        device_منزل_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, منزل_surface, buf.get_ptr(),
                                                 buf.get_ناپ(), host_منزل_device,
                                                 host_events));
                اكسر;
            }
            حالة device_منزل_host:
            {
                host_بفر buf(get_copy_range(size, from_ٹکڑا, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, منزل_surface, buf.get_ptr(), منزل_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_منزل_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_ناپ(),
                                                         device_منزل_host, dep_events)});
                اكسر;
            }
            حالة device_منزل_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<صنف dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        منزل_surface[get_وتھ(id, منزل_ٹکڑا, منزل_range.get(0))] =
                            from_surface[get_وتھ(id, from_ٹکڑا, from_range.get(0))];
                    }); }));
                اكسر;
            افتراضي:
                ارم std::runtime_error("dpct_memcpy: invalid direction value");
            }
            أرجع event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ساكن مضمن std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_انکڑے منزل, sycl::id<3> منزل_id,
                    pitched_انکڑے from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            أرجع dpct_memcpy(q, منزل.get_انکڑے_ptr(), from.get_انکڑے_ptr(),
                               sycl::range<3>(منزل.get_pitch(), منزل.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), منزل_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ساكن مضمن std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, فراغ *منزل_ptr, ثابت فراغ *from_ptr,
                    size_t منزل_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            أرجع dpct_memcpy(q, منزل_ptr, from_ptr, sycl::range<3>(منزل_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        نطاق deprecated
        {

            سانچا <قسم_ناں T, sycl::usm::alloc Alloc_قسم>
            صنف usm_allocator
            {
            خاص:
                ورتوں Alloc = sycl::usm_allocator<T, Alloc_قسم>;
                Alloc _impl;

            عام:
                ورتوں value_type = قسم_ناں std::allocator_traits<Alloc>::value_type;
                ورتوں pointer = قسم_ناں std::allocator_traits<Alloc>::pointer;
                ورتوں const_pointer = قسم_ناں std::allocator_traits<Alloc>::const_pointer;
                ورتوں void_pointer = قسم_ناں std::allocator_traits<Alloc>::void_pointer;
                ورتوں const_void_pointer =
                    قسم_ناں std::allocator_traits<Alloc>::const_void_pointer;
                ورتوں reference = قسم_ناں std::allocator_traits<Alloc>::value_type &;
                ورتوں const_reference =
                    ثابت قسم_ناں std::allocator_traits<Alloc>::value_type &;
                ورتوں difference_type =
                    قسم_ناں std::allocator_traits<Alloc>::difference_type;
                ورتوں size_type = قسم_ناں std::allocator_traits<Alloc>::size_type;
                ورتوں propagate_on_container_copy_assignment = قسم_ناں std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                ورتوں propagate_on_container_move_assignment = قسم_ناں std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                ورتوں propagate_on_container_swap =
                    قسم_ناں std::allocator_traits<Alloc>::propagate_on_container_swap;
                ورتوں is_always_equal =
                    قسم_ناں std::allocator_traits<Alloc>::is_always_equal;

                سانچا <قسم_ناں U>
                بنية rebind
                {
                    تعريفنوع usm_allocator<U, Alloc_قسم> other;
                };

                usm_allocator() : _impl(dpct::get_default_قطار()) {}
                ~usm_allocator() {}
                usm_allocator(ثابت usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { أرجع &r; }
                const_pointer address(const_reference r) { أرجع &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = خالی_سوچک)
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
                منطقي عامل==(ثابت usm_allocator &other) ثابت { أرجع _impl == other._impl; }
                منطقي عامل!=(ثابت usm_allocator &other) ثابت { أرجع _impl != other._impl; }
            };

        } // namespace deprecated

        مضمن فراغ dpct_free(فراغ *ptr,
                              ثابت sycl::queue &q)
        {
            إذا (ptr)
            {
                sycl::حرر(ptr, q.get_context());
            }
        }

        سانچا <قسم_ناں T>
        مضمن خودکار get_حافظہ(ثابت فراغ *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<فراغ *>(x));
            أرجع new_x;
        }

        سانچا <قسم_ناں T>
        مضمن قسم_ناں انکڑے_Type_2<T>::T2 get_قدر(ثابت T *s, sycl::queue &q)
        {
            ورتوں Ty = قسم_ناں انکڑے_Type_2<T>::T2;
            Ty s_h;
            إذا (get_اشارہ_گر_attribute(q, s) == اشارہ_گر_access_attribute::device_only)
                detail::dpct_memcpy(q, (فراغ *)&s_h, (ثابت فراغ *)s, حجم(T), device_منزل_host)
                    .wait();
            وإلا
                s_h = *reinterpret_cast<ثابت Ty *>(s);
            أرجع s_h;
        }

    } // namespace detail

    سانچا <قسم_ناں T>
    مضمن خودکار get_قدر(ثابت T *s, sycl::queue &q)
    {
        أرجع detail::get_قدر(s, q);
    }

    نطاق detail
    {
    سانچا <صنف Ta, صنف Tb, صنف Tc, صنف Ts>
    مضمن فراغ gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحيح m,
                          صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, صحيح lda, ثابت فراغ * b, صحيح ldb,
                          ثابت فراغ * beta, فراغ * c, صحيح ldc) {
        Ts   alpha_قدر_2 = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(alpha), q);
        Ts   beta_قدر  = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(beta), q);
        خودکار انکڑے_a      = get_حافظہ<ثابت Ta>(a);
        خودکار انکڑے_b      = get_حافظہ<ثابت Tb>(b);
        خودکار انکڑے_c      = get_حافظہ<Tc>(c);
        oneapi::mkl::blas::کالم_major::gemm(q, a_trans, b_trans, m, n, k, alpha_قدر_2, انکڑے_a,
                                               lda, انکڑے_b, ldb, beta_قدر, انکڑے_c, ldc);
    }

        سانچا <قسم_ناں VecT, صنف ثنائی_Operation, صنف = فراغ>
        صنف vectorized_ثنائی
        {
        عام:
            مضمن VecT عامل()(VecT a, VecT b, ثابت ثنائی_Operation ثنائی_op)
            {
                VecT v4;
                لكل (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ثنائی_op(a[i], b[i]);
                }
                أرجع v4;
            }
        };

        سانچا <قسم_ناں VecT, صنف ثنائی_Operation>
        صنف vectorized_ثنائی<
            VecT, ثنائی_Operation,
            std::void_t<std::invoke_result_t<ثنائی_Operation, VecT, VecT>>>
        {
        عام:
            مضمن VecT عامل()(VecT a, VecT b, ثابت ثنائی_Operation ثنائی_op)
            {
                أرجع ثنائی_op(a, b).سانچا as<VecT>();
            }
        };

        سانچا <صنف Ta, صنف Tb, صنف Tc, صنف Ts>
        مضمن فراغ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    صحيح m, صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ ** a, صحيح lda, ثابت فراغ ** b,
                                    صحيح ldb, ثابت فراغ * beta, فراغ ** c, صحيح ldc, صحيح batch_ناپ,
                                    میٹرکس_info_t<عائم> * میٹرکس_info) {
            Ts alpha_قدر_2 = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(alpha), q);
            Ts beta_قدر = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(beta), q);

            میٹرکس_info->transpose_info[0] = a_trans;
            میٹرکس_info->transpose_info[1] = b_trans;
            میٹرکس_info->قدر_info[0] = alpha_قدر_2;
            میٹرکس_info->قدر_info[1] = beta_قدر;
            میٹرکس_info->ناپ_info[0] = m;
            میٹرکس_info->ناپ_info[1] = n;
            میٹرکس_info->ناپ_info[2] = k;
            میٹرکس_info->ld_info[0] = lda;
            میٹرکس_info->ld_info[1] = ldb;
            میٹرکس_info->ld_info[2] = ldc;
            میٹرکس_info->groupsize_info = batch_ناپ;

            sycl::event e = oneapi::mkl::blas::کالم_major::gemm_batch(
                q, میٹرکس_info->transpose_info, میٹرکس_info->transpose_info + 1,
                میٹرکس_info->ناپ_info, میٹرکس_info->ناپ_info + 1, میٹرکس_info->ناپ_info + 2,
                reinterpret_cast<Ts *>(میٹرکس_info->قدر_info), reinterpret_cast<ثابت Ta **>(a), میٹرکس_info->ld_info,
                reinterpret_cast<ثابت Tb **>(b), میٹرکس_info->ld_info + 1,
                reinterpret_cast<Ts *>(میٹرکس_info->قدر_info + 1), reinterpret_cast<Tc **>(c),
                میٹرکس_info->ld_info + 2, 1, &(میٹرکس_info->groupsize_info));
        }

        سانچا <صنف Ta, صنف Tb, صنف Tc, صنف Ts>
        مضمن فراغ gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    صحيح m, صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, صحيح lda,
                                    لما لما صحيح stride_a, ثابت فراغ * b, صحيح ldb, لما لما صحيح stride_b,
                                    ثابت فراغ * beta, فراغ * c, صحيح ldc, لما لما صحيح stride_c, صحيح batch_ناپ) {
            Ts alpha_قدر_2 = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(alpha), q);
            Ts beta_قدر = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(beta), q);
            خودکار انکڑے_a = get_حافظہ<ثابت Ta>(a);
            خودکار انکڑے_b = get_حافظہ<ثابت Tb>(b);
            خودکار انکڑے_c = get_حافظہ<Tc>(c);
            oneapi::mkl::blas::کالم_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_قدر_2,
                                                         انکڑے_a, lda, stride_a, انکڑے_b, ldb, stride_b, beta_قدر,
                                                         انکڑے_c, ldc, stride_c, batch_ناپ);
        }

    } // namespace detail

    سانچا <قسم_ناں VecT, صنف ثنائی_Operation>
    مضمن بنانشانی vectorized_ثنائی(بنانشانی a, بنانشانی b,
                                      ثابت ثنائی_Operation ثنائی_op)
    {
        sycl::vec<بنانشانی, 1> v0{a}, v1{b};
        خودکار v2 = v0.as<VecT>();
        خودکار v3 = v1.as<VecT>();
        خودکار v4 =
            detail::vectorized_ثنائی<VecT, ثنائی_Operation>()(v2, v3, ثنائی_op);
        v0 = v4.سانچا as<sycl::vec<بنانشانی, 1>>();
        أرجع v0;
    }

    ساكن فراغ async_dpct_memcpy(فراغ *منزل_ptr, ثابت فراغ *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_قطار())
    {
        detail::dpct_memcpy(q, منزل_ptr, from_ptr, size, direction);
    }

    ساكن مضمن بنانشانی صحيح select_device(بنانشانی صحيح id)
    {
        dev_mgr::instance().select_device(id);
        أرجع id;
    }

    سانچا <قسم_ناں T>
    T ترتیب_بدلو_sub_گروہ_by_xor(sycl::sub_گروہ g, T x, بنانشانی صحيح نقاب,
                               بنانشانی صحيح logical_sub_گروہ_ناپ = 32)
    {
        بنانشانی صحيح id = g.get_local_خطی_id();
        بنانشانی صحيح شروع_اشاریہ =
            id / logical_sub_گروہ_ناپ * logical_sub_گروہ_ناپ;
        بنانشانی صحيح نشانہ_وتھ = (id % logical_sub_گروہ_ناپ) ^ نقاب;
        أرجع sycl::select_from_گروہ(g, x,
                                       نشانہ_وتھ < logical_sub_گروہ_ناپ
                                           ? شروع_اشاریہ + نشانہ_وتھ
                                           : id);
    }

    سانچا <قسم_ناں T1, قسم_ناں T2>
    ورتوں dot_ضرب_حاصل_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    سانچا <قسم_ناں T>
    sycl::vec<T, 4> کڈھو_and_sign_or_zero_extend4(T val) {
      أرجع sycl::vec<T, 1>(val)
          .سانچا as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .سانچا convert<T>();
    }

    سانچا <قسم_ناں T1, قسم_ناں T2, قسم_ناں T3>
    مضمن خودکار dp4a(T1 a, T2 b, T3 c) {
      dot_ضرب_حاصل_acc_t<T1, T2> res = c;
      خودکار va = کڈھو_and_sign_or_zero_extend4(a);
      خودکار vb = کڈھو_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      أرجع res;
    }

    بنية sub_sat
    {
        سانچا <قسم_ناں T>
        خودکار عامل()(ثابت T x, ثابت T y) ثابت
        {
            أرجع sycl::sub_sat(x, y);
        }
    };

    سانچا <قسم_ناں S, قسم_ناں T>
    مضمن T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        خودکار v2 = v0.سانچا as<S>();
        خودکار v3 = v1.سانچا as<S>();
        خودکار v4 = sycl::min(v2, v3);
        v0 = v4.سانچا as<sycl::vec<T, 1>>();
        أرجع v0;
    }

    مضمن عائم قوت(ثابت عائم a, ثابت صحيح b) { أرجع sycl::pown(a, b); }
    مضمن مزدوج قوت(ثابت مزدوج a, ثابت صحيح b) { أرجع sycl::pown(a, b); }
    مضمن عائم قوت(ثابت عائم a, ثابت عائم b) { أرجع sycl::قوت(a, b); }
    مضمن مزدوج قوت(ثابت مزدوج a, ثابت مزدوج b) { أرجع sycl::قوت(a, b); }
    سانچا <قسم_ناں T, قسم_ناں U>
    مضمن قسم_ناں std::enable_if_t<std::is_floating_point_v<T>, T>
    قوت(ثابت T a, ثابت U b)
    {
        أرجع sycl::قوت(a, static_cast<T>(b));
    }
    سانچا <قسم_ناں T, قسم_ناں U>
    مضمن قسم_ناں std::enable_if_t<!std::is_floating_point_v<T>, مزدوج>
    قوت(ثابت T a, ثابت U b)
    {
        أرجع sycl::قوت(static_cast<مزدوج>(a), static_cast<مزدوج>(b));
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
        لكل (ثابت خودکار &it : props)
        {
            إذا (dev.has(it))
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
#تعریف_کرو __SYCL_ASPECT(ASPECT, ID) \
    حالة sycl::aspect::ASPECT:    \
        أرجع #ASPECT;
#تعریف_کرو __SYCL_ASPECT_DEPRECATED(ASPECT, ID, سنیہا_2) __SYCL_ASPECT(ASPECT, ID)
#تعریف_کرو __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, سنیہا_2)
                خودکار getAspectNameStr = [](sycl::aspect Aspect_عدد) -> std::string
                {
                    بدل (Aspect_عدد)
                    {
#ضمن <sycl/info/aspects.def>
#ضمن <sycl/info/aspects_deprecated.def>
                    افتراضي:
                        أرجع "unknown aspect";
                    }
                };
#تعریف_رد __SYCL_ASPECT_DEPRECATED_ALIAS
#تعریف_رد __SYCL_ASPECT_DEPRECATED
#تعریف_رد __SYCL_ASPECT
                ارم std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            اكسر;
        }
    }

    ساكن مضمن بنانشانی صحيح get_current_device_id()
    {
        أرجع dev_mgr::instance().current_device_id();
    }

    ساكن مضمن device_ext &get_current_device()
    {
        أرجع dev_mgr::instance().current_device();
    }

    ساكن مضمن device_ext &get_device(بنانشانی صحيح id)
    {
        أرجع dev_mgr::instance().get_device(id);
    }

    ساكن مضمن sycl::queue &get_وچ_ترتیب_قطار()
    {
        أرجع dev_mgr::instance().current_device().وچ_ترتیب_قطار();
    }

    ساكن sycl::event
    dpct_memcpy(sycl::queue &q, فراغ *منزل_ptr, ثابت فراغ *from_ptr, size_t size,
                memcpy_direction direction,
                ثابت std::vector<sycl::event> &dep_events = {})
    {
        إذا (!size)
            أرجع sycl::event{};
        أرجع q.حافظہ_نقل(منزل_ptr, from_ptr, size, dep_events);
        DPCT_بن_ورتیا(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ساكن مضمن size_t get_copy_range(sycl::range<3> size, size_t ٹکڑا_2,
                                        size_t pitch)
    {
        أرجع ٹکڑا_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ساكن مضمن size_t get_وتھ(sycl::id<3> id, size_t ٹکڑا_2,
                                    size_t pitch)
    {
        أرجع ٹکڑا_2 * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ساكن مضمن std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, فراغ *منزل_ptr, ثابت فراغ *from_ptr,
                sycl::range<3> منزل_range, sycl::range<3> from_range,
                sycl::id<3> منزل_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ثابت std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        صنف host_بفر
        {
            فراغ *_buf;
            size_t ناپ_2;
            sycl::queue &_q;
            ثابت std::vector<sycl::event> &_deps; // free operation depends

        عام:
            host_بفر(size_t size, sycl::queue &q,
                        ثابت std::vector<sycl::event> &deps)
                : _buf(std::حافظہ_دیو(size)), ناپ_2(size), _q(q), _deps(deps) {}
            فراغ *get_ptr() ثابت { أرجع _buf; }
            size_t get_ناپ() ثابت { أرجع ناپ_2; }
            ~host_بفر()
            {
                إذا (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::حرر(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t منزل_ٹکڑا = منزل_range.get(1) * منزل_range.get(0),
               from_ٹکڑا = from_range.get(1) * from_range.get(0);
        بنانشانی حرف *منزل_surface =
            (بنانشانی حرف *)منزل_ptr + get_وتھ(منزل_id, منزل_ٹکڑا, منزل_range.get(0));
        ثابت بنانشانی حرف *from_surface =
            (ثابت بنانشانی حرف *)from_ptr +
            get_وتھ(from_id, from_ٹکڑا, from_range.get(0));

        إذا (منزل_ٹکڑا == from_ٹکڑا && منزل_ٹکڑا == size.get(1) * size.get(0))
        {
            أرجع {dpct_memcpy(q, منزل_surface, from_surface, منزل_ٹکڑا * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, منزل_ptr, from_ptr, direction);
        size_t ناپ_ٹکڑا = size.get(1) * size.get(0);
        بدل (direction)
        {
        حالة host_منزل_host:
            لكل (size_t z = 0; z < size.get(2); ++z)
            {
                بنانشانی حرف *منزل_ptr = منزل_surface;
                ثابت بنانشانی حرف *from_ptr = from_surface;
                إذا (منزل_range.get(0) == from_range.get(0) &&
                    منزل_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, منزل_ptr, from_ptr, ناپ_ٹکڑا,
                                                     direction, dep_events));
                }
                وإلا
                {
                    لكل (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, منزل_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        منزل_ptr += منزل_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                منزل_surface += منزل_ٹکڑا;
                from_surface += from_ٹکڑا;
            }
            اكسر;
        حالة host_منزل_device:
        {
            host_بفر buf(get_copy_range(size, منزل_ٹکڑا, منزل_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            إذا (منزل_ٹکڑا == ناپ_ٹکڑا)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, منزل_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_منزل_host, dep_events);
            }
            وإلا
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, منزل_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_منزل_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), منزل_surface, buf.get_ناپ(),
                                    device_منزل_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, منزل_surface, buf.get_ptr(),
                                             buf.get_ناپ(), host_منزل_device,
                                             host_events));
            اكسر;
        }
        حالة device_منزل_host:
        {
            host_بفر buf(get_copy_range(size, from_ٹکڑا, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, منزل_surface, buf.get_ptr(), منزل_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_منزل_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_ناپ(),
                                                     device_منزل_host, dep_events)});
            اكسر;
        }
        حالة device_منزل_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<صنف dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                منزل_surface[get_وتھ(id, منزل_ٹکڑا, منزل_range.get(0))] =
                    from_surface[get_وتھ(id, from_ٹکڑا, from_range.get(0))];
            }); }));
        اكسر;
        افتراضي:
            ارم std::runtime_error("dpct_memcpy: invalid direction value");
        }
        أرجع event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ساكن مضمن std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_انکڑے منزل, sycl::id<3> منزل_id,
                pitched_انکڑے from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        أرجع dpct_memcpy(q, منزل.get_انکڑے_ptr(), from.get_انکڑے_ptr(),
                           sycl::range<3>(منزل.get_pitch(), منزل.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), منزل_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ساكن مضمن std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, فراغ *منزل_ptr, ثابت فراغ *from_ptr,
                size_t منزل_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        أرجع dpct_memcpy(q, منزل_ptr, from_ptr, sycl::range<3>(منزل_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    مضمن فراغ gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحيح m, صحيح n,
                     صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, library_انکڑے_t a_type, صحيح lda, ثابت فراغ * b,
                     library_انکڑے_t b_type, صحيح ldb, ثابت فراغ * beta, فراغ * c, library_انکڑے_t c_type, صحيح ldc,
                     library_انکڑے_t scaling_type) {
        إذا (scaling_type == library_انکڑے_t::real_float &&
            c_type == library_انکڑے_t::complex_float)
        {
            scaling_type = library_انکڑے_t::complex_float;
        }
        وإلا إذا (scaling_type == library_انکڑے_t::real_double &&
                 c_type == library_انکڑے_t::complex_double)
        {
            scaling_type = library_انکڑے_t::complex_double;
        }

        std::uint64_t کلید =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        بدل (کلید)
        {
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_float, library_انکڑے_t::real_float,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_impl<عائم, عائم, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_double, library_انکڑے_t::real_double,
            library_انکڑے_t::real_double, library_انکڑے_t::real_double):
        {
            detail::gemm_impl<مزدوج, مزدوج, مزدوج, مزدوج>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::complex_float, library_انکڑے_t::complex_float,
            library_انکڑے_t::complex_float, library_انکڑے_t::complex_float):
        {
            detail::gemm_impl<std::complex<عائم>, std::complex<عائم>,
                              std::complex<عائم>, std::complex<عائم>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::complex_double, library_انکڑے_t::complex_double,
            library_انکڑے_t::complex_double, library_انکڑے_t::complex_double):
        {
            detail::gemm_impl<std::complex<مزدوج>, std::complex<مزدوج>,
                              std::complex<مزدوج>, std::complex<مزدوج>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_half, library_انکڑے_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            اكسر;
        }
#جے_تعریف_شدہ __INTEL_MKL__
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_bfloat16,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_half, library_انکڑے_t::real_float):
        {
            عائم alpha_قدر_2 =
                dpct::get_قدر(reinterpret_cast<ثابت عائم *>(alpha), q);
            عائم beta_قدر =
                dpct::get_قدر(reinterpret_cast<ثابت عائم *>(beta), q);
            sycl::half alpha_half(alpha_قدر_2);
            sycl::half beta_half(beta_قدر);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_int8, library_انکڑے_t::real_int8,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_bfloat16,
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_int8, library_انکڑے_t::real_int8,
            library_انکڑے_t::real_int32, library_انکڑے_t::real_int32):
        {
            عائم alpha_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(alpha), q);
            عائم beta_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, عائم>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            اكسر;
        }
#جے_انت // __INTEL_MKL__
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
                           صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a[], library_انکڑے_t a_type, صحيح lda,
                           ثابت فراغ * b[], library_انکڑے_t b_type, صحيح ldb, ثابت فراغ * beta, فراغ * c[],
                           library_انکڑے_t c_type, صحيح ldc, صحيح batch_ناپ, library_انکڑے_t scaling_type,
                           میٹرکس_info_t<عائم> * میٹرکس_info) {
        std::uint64_t کلید =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        بدل (کلید)
        {
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_float, library_انکڑے_t::real_float,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<عائم, عائم, عائم, عائم>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_double, library_انکڑے_t::real_double,
            library_انکڑے_t::real_double, library_انکڑے_t::real_double):
        {
            detail::gemm_batch_impl<مزدوج, مزدوج, مزدوج, مزدوج>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_half, library_انکڑے_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
#جے_تعریف_شدہ __INTEL_MKL__
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_bfloat16,
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_bfloat16,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
#جے_انت
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_int8, library_انکڑے_t::real_int8,
            library_انکڑے_t::real_int32, library_انکڑے_t::real_int32):
        {
            عائم alpha_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(alpha), q);
            عائم beta_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, عائم>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_ناپ,
                میٹرکس_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_int8, library_انکڑے_t::real_int8,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_ناپ, میٹرکس_info);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_half, library_انکڑے_t::real_float):
        {
            عائم alpha_قدر_2 =
                dpct::get_قدر(reinterpret_cast<ثابت عائم *>(alpha), q);
            عائم beta_قدر =
                dpct::get_قدر(reinterpret_cast<ثابت عائم *>(beta), q);
            sycl::half alpha_half(alpha_قدر_2);
            sycl::half beta_half(beta_قدر);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_ناپ, میٹرکس_info);
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
                           صحيح n, صحيح k, ثابت فراغ * alpha, ثابت فراغ * a, library_انکڑے_t a_type, صحيح lda,
                           لما لما صحيح stride_a, ثابت فراغ * b, library_انکڑے_t b_type, صحيح ldb,
                           لما لما صحيح stride_b, ثابت فراغ * beta, فراغ * c, library_انکڑے_t c_type, صحيح ldc,
                           لما لما صحيح stride_c, صحيح batch_ناپ, library_انکڑے_t scaling_type) {
        إذا (scaling_type == library_انکڑے_t::real_float &&
            c_type == library_انکڑے_t::complex_float)
        {
            scaling_type = library_انکڑے_t::complex_float;
        }
        وإلا إذا (scaling_type == library_انکڑے_t::real_double &&
                 c_type == library_انکڑے_t::complex_double)
        {
            scaling_type = library_انکڑے_t::complex_double;
        }

        std::uint64_t کلید =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        بدل (کلید)
        {
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_float, library_انکڑے_t::real_float,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<عائم, عائم, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_double, library_انکڑے_t::real_double,
            library_انکڑے_t::real_double, library_انکڑے_t::real_double):
        {
            detail::gemm_batch_impl<مزدوج, مزدوج, مزدوج, مزدوج>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::complex_float, library_انکڑے_t::complex_float,
            library_انکڑے_t::complex_float, library_انکڑے_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<عائم>, std::complex<عائم>,
                                    std::complex<عائم>, std::complex<عائم>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::complex_double, library_انکڑے_t::complex_double,
            library_انکڑے_t::complex_double, library_انکڑے_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<مزدوج>, std::complex<مزدوج>,
                                    std::complex<مزدوج>, std::complex<مزدوج>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_half, library_انکڑے_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
#جے_تعریف_شدہ __INTEL_MKL__
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_bfloat16,
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_bfloat16, library_انکڑے_t::real_bfloat16,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_ناپ);
            اكسر;
        }
#جے_انت
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_int8, library_انکڑے_t::real_int8,
            library_انکڑے_t::real_int32, library_انکڑے_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_int8, library_انکڑے_t::real_int8,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_float, library_انکڑے_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, عائم, عائم>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        حالة detail::get_type_combination_id(
            library_انکڑے_t::real_half, library_انکڑے_t::real_half,
            library_انکڑے_t::real_half, library_انکڑے_t::real_float):
        {
            عائم alpha_قدر_2 =
                dpct::get_قدر(reinterpret_cast<ثابت عائم *>(alpha), q);
            عائم beta_قدر =
                dpct::get_قدر(reinterpret_cast<ثابت عائم *>(beta), q);
            sycl::half alpha_half(alpha_قدر_2);
            sycl::half beta_half(beta_قدر);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_ناپ);
            اكسر;
        }
        افتراضي:
            ارم std::runtime_error("the combination of data type is unsupported");
        }
    }

    ساكن مضمن فراغ
    async_dpct_memcpy(فراغ *منزل_ptr, size_t منزل_pitch, ثابت فراغ *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_قطار())
    {
        detail::dpct_memcpy(q, منزل_ptr, from_ptr, منزل_pitch, from_pitch, x, y,
                            direction);
    }

    ورتوں err0 = detail::generic_error_type<بنية err0_ٹیگ, صحيح>;
    ورتوں err1 = detail::generic_error_type<بنية err1_ٹیگ, صحيح>;

    ساكن مضمن فراغ dpct_free(فراغ *ptr, sycl::queue &q = get_default_قطار()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    سانچا <صنف T, حافظہ_region حافظہ_2, size_t Dimension> صنف accessor;
    سانچا <صنف T, حافظہ_region حافظہ_2> صنف accessor<T, حافظہ_2, 3> {
    عام:
        ورتوں حافظہ_t = detail::حافظہ_traits<حافظہ_2, T>;
        ورتوں element_t = قسم_ناں حافظہ_t::element_t;
        ورتوں اشارہ_گر_t = قسم_ناں حافظہ_t::اشارہ_گر_t;
        ورتوں accessor_t = قسم_ناں حافظہ_t::سانچا accessor_t<3>;
        accessor(اشارہ_گر_t انکڑے, ثابت sycl::range<3> &وچ_range)
            : انکڑے_2(انکڑے), _range(وچ_range) {}
        سانچا <حافظہ_region M = حافظہ_2>
        accessor(قسم_ناں std::enable_if<M != local, ثابت accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ثابت accessor_t &acc, ثابت sycl::range<3> &وچ_range)
            : accessor(acc.get_اشارہ_گر(), وچ_range) {}
        accessor<T, حافظہ_2, 2> عامل[](size_t اشاریہ) ثابت {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            أرجع accessor<T, حافظہ_2, 2>(انکڑے_2 + اشاریہ * sub.size(), sub);
        }

        اشارہ_گر_t get_ptr() ثابت { أرجع انکڑے_2; }

    خاص:
        اشارہ_گر_t انکڑے_2;
        sycl::range<3> _range;
    };
    سانچا <صنف T, حافظہ_region حافظہ_2> صنف accessor<T, حافظہ_2, 2> {
    عام:
        ورتوں حافظہ_t = detail::حافظہ_traits<حافظہ_2, T>;
        ورتوں element_t = قسم_ناں حافظہ_t::element_t;
        ورتوں اشارہ_گر_t = قسم_ناں حافظہ_t::اشارہ_گر_t;
        ورتوں accessor_t = قسم_ناں حافظہ_t::سانچا accessor_t<2>;
        accessor(اشارہ_گر_t انکڑے, ثابت sycl::range<2> &وچ_range)
            : انکڑے_2(انکڑے), _range(وچ_range) {}
        سانچا <حافظہ_region M = حافظہ_2>
        accessor(قسم_ناں std::enable_if<M != local, ثابت accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ثابت accessor_t &acc, ثابت sycl::range<2> &وچ_range)
            : accessor(acc.get_اشارہ_گر(), وچ_range) {}

        اشارہ_گر_t عامل[](size_t اشاریہ) ثابت {
            أرجع انکڑے_2 + _range.get(1) * اشاریہ;
        }

        اشارہ_گر_t get_ptr() ثابت { أرجع انکڑے_2; }

    خاص:
        اشارہ_گر_t انکڑے_2;
        sycl::range<2> _range;
    };

    نطاق detail {
        /// Device variable with address space of shared, global or constant.
        سانچا <صنف T, حافظہ_region حافظہ_2, size_t Dimension> صنف device_حافظہ {
        عام:
            ورتوں accessor_t =
                قسم_ناں detail::حافظہ_traits<حافظہ_2,
                                            T>::سانچا accessor_t<Dimension>;
            ورتوں قدر_t = قسم_ناں detail::حافظہ_traits<حافظہ_2, T>::قدر_t;
            ورتوں dpct_accessor_t = dpct::accessor<T, حافظہ_2, Dimension>;

            device_حافظہ() : device_حافظہ(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_حافظہ(ثابت sycl::range<Dimension> &وچ_range,
                        std::initializer_list<قدر_t> &&init_list)
                : device_حافظہ(وچ_range) {
                أكد(init_list.size() <= وچ_range.size());
                _host_ptr = (قدر_t *)std::حافظہ_دیو(ناپ_2);
                std::memset(_host_ptr, 0, ناپ_2);
                std::حافظہ_نقل(_host_ptr, init_list.begin(), init_list.size() * حجم(T));
            }

            /// Constructor of 2-D array with initializer list
            سانچا <size_t D = Dimension>
            device_حافظہ(
                ثابت قسم_ناں std::enable_if<D == 2, sycl::range<2>>::type &وچ_range,
                std::initializer_list<std::initializer_list<قدر_t>> &&init_list)
                : device_حافظہ(وچ_range) {
                أكد(init_list.size() <= وچ_range[0]);
                _host_ptr = (قدر_t *)std::حافظہ_دیو(ناپ_2);
                std::memset(_host_ptr, 0, ناپ_2);
                خودکار tmp_انکڑے = _host_ptr;
                لكل (خودکار sub_list : init_list) {
                    أكد(sub_list.size() <= وچ_range[1]);
                    std::حافظہ_نقل(tmp_انکڑے, sub_list.begin(),
                                sub_list.size() * حجم(T));
                    tmp_انکڑے += وچ_range[1];
                }
            }

            /// Constructor with range
            device_حافظہ(ثابت sycl::range<Dimension> &range_وچ)
                : ناپ_2(range_وچ.size() * حجم(T)), _range(range_وچ),
                _reference(خطأ), _host_ptr(خالی_سوچک), _device_ptr(خالی_سوچک) {
                static_assert(
                    (حافظہ_2 == global) || (حافظہ_2 == constant) || (حافظہ_2 == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            سانچا <صنف... دلیلاں>
            device_حافظہ(دلیلاں... Arguments)
                : device_حافظہ(sycl::range<Dimension>(Arguments...)) {}

            ~device_حافظہ() {
                إذا (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                إذا (_host_ptr)
                    std::حرر(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            فراغ init() { init(dpct::get_default_قطار()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            فراغ init(sycl::queue &q) {
                إذا (_device_ptr)
                    أرجع;
                إذا (!ناپ_2)
                    أرجع;
                allocate_device(q);
                إذا (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, ناپ_2,
                                        host_منزل_device);
            }

            /// The variable is assigned to a device pointer.
            فراغ سونپو(قدر_t *src, size_t size) {
                هذا->~device_حافظہ();
                جديد (هذا) device_حافظہ(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            قدر_t *get_ptr() { أرجع get_ptr(get_default_قطار()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            قدر_t *get_ptr(sycl::queue &q) {
                init(q);
                أرجع _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_ناپ() { أرجع ناپ_2; }

            سانچا <size_t D = Dimension>
            قسم_ناں std::enable_if<D == 1, T>::type &عامل[](size_t اشاریہ) {
                init();
                أرجع _device_ptr[اشاریہ];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            سانچا <size_t D = Dimension>
            قسم_ناں std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                أرجع dpct_accessor_t((T *)_device_ptr, _range);
            }

        خاص:
            device_حافظہ(قدر_t *حافظہ_ptr, size_t size)
                : ناپ_2(size), _range(size / حجم(T)), _reference(سچ),
                _device_ptr(حافظہ_ptr) {}

            فراغ allocate_device(sycl::queue &q) {
        #جے_بنا_تعریف DPCT_USM_LEVEL_NONE
                إذا (حافظہ_2 == shared) {
                    _device_ptr = (قدر_t *)sycl::malloc_shared(ناپ_2, q.get_device(),
                                                                q.get_context());
                    أرجع;
                }
        #جے_تعریف_شدہ SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                إذا (حافظہ_2 == constant) {
                    _device_ptr = (قدر_t *)sycl::malloc_device(
                        ناپ_2, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    أرجع;
                }
        #جے_انت
        #جے_انت
                _device_ptr = (قدر_t *)detail::dpct_malloc(ناپ_2, q);
            }

            size_t ناپ_2;
            sycl::range<Dimension> _range;
            منطقي _reference;
            قدر_t *_host_ptr;
            قدر_t *_device_ptr;
        };
        سانچا <صنف T, حافظہ_region حافظہ_2>
        صنف device_حافظہ<T, حافظہ_2, 0> : عام device_حافظہ<T, حافظہ_2, 1> {
        عام:
            ورتوں ادھار = device_حافظہ<T, حافظہ_2, 1>;
            ورتوں قدر_t = قسم_ناں ادھار::قدر_t;
            ورتوں accessor_t =
                قسم_ناں detail::حافظہ_traits<حافظہ_2, T>::سانچا accessor_t<0>;

            /// Constructor with initial value.
            device_حافظہ(ثابت قدر_t &val) : ادھار(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_حافظہ() : ادھار(1) {}
        };
        } // namespace detail

    سانچا <صنف T, size_t Dimension>
    ورتوں global_حافظہ = detail::device_حافظہ<T, global, Dimension>;
    سانچا <صنف T, size_t Dimension>
    ورتوں constant_حافظہ = detail::device_حافظہ<T, constant, Dimension>;
    سانچا <صنف T, size_t Dimension>
    ورتوں shared_حافظہ = detail::device_حافظہ<T, shared, Dimension>;


    سانچا <قسم_ناں T,
            sycl::access::پتہ_space پتہ_Space =
                sycl::access::پتہ_space::global_space,
            sycl::memory_order حافظہ_ترتیب = sycl::memory_order::relaxed,
            sycl::حافظہ_scope حافظہ_Scope = sycl::حافظہ_scope::device>
    مضمن T atomic_fetch_add(T *addr, T عملیہ) {
    خودکار atm =
        sycl::atomic_ref<T, حافظہ_ترتیب, حافظہ_Scope, پتہ_Space>(addr[0]);
    أرجع atm.fetch_add(عملیہ);
    }

    سانچا <sycl::access::پتہ_space پتہ_Space =
                sycl::access::پتہ_space::global_space,
            sycl::memory_order حافظہ_ترتیب = sycl::memory_order::relaxed,
            sycl::حافظہ_scope حافظہ_Scope = sycl::حافظہ_scope::device,
            قسم_ناں T1, قسم_ناں T2>
    مضمن T1 atomic_fetch_add(T1 *addr, T2 عملیہ) {
    خودکار atm =
        sycl::atomic_ref<T1, حافظہ_ترتیب, حافظہ_Scope, پتہ_Space>(addr[0]);
    أرجع atm.fetch_add(عملیہ);
    }

    سانچا <قسم_ناں T, sycl::access::پتہ_space پتہ_Space =
                            sycl::access::پتہ_space::global_space>
    مضمن T atomic_fetch_add(T *addr, T عملیہ,
                            sycl::memory_order حافظہ_ترتیب) {
    بدل (حافظہ_ترتیب) {
        حالة sycl::memory_order::relaxed:
            أرجع atomic_fetch_add<T, پتہ_Space, sycl::memory_order::relaxed,
                                    sycl::حافظہ_scope::device>(addr, عملیہ);
        حالة sycl::memory_order::acq_rel:
            أرجع atomic_fetch_add<T, پتہ_Space, sycl::memory_order::acq_rel,
                                    sycl::حافظہ_scope::device>(addr, عملیہ);
        حالة sycl::memory_order::seq_cst:
            أرجع atomic_fetch_add<T, پتہ_Space, sycl::memory_order::seq_cst,
                                    sycl::حافظہ_scope::device>(addr, عملیہ);
        افتراضي:
            أكد(خطأ && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    سانچا <sycl::access::پتہ_space پتہ_Space =
                sycl::access::پتہ_space::global_space,
            قسم_ناں T1, قسم_ناں T2>
    مضمن T1 atomic_fetch_add(T1 *addr, T2 عملیہ,
                            sycl::memory_order حافظہ_ترتیب) {
    atomic_fetch_add<T1, پتہ_Space>(addr, عملیہ, حافظہ_ترتیب);
    }

    مضمن بنانشانی صحيح byte_level_ترتیب_بدلو(
        بنانشانی صحيح a, بنانشانی صحيح b, بنانشانی صحيح s) {
      بنانشانی صحيح ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      أرجع ret;
    }

    مضمن uint32_t byte_level_ترتیب_بدلو_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, صحيح mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      إذا (mode >= 1 && mode <= 6) {
        أرجع byte_level_ترتیب_بدلو(low32, high32, lookup[mode - 1][sel & 0x3]);
      } وإلا إذا (!mode) {
        أرجع byte_level_ترتیب_بدلو(low32, high32, sel);
      }
      أرجع 0;
    }

    سانچا <صحيح n_nondefault_params, صحيح n_default_params, قسم_ناں T>
    صنف دلیلاں_selector;

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
    سانچا <صحيح n_nondefault_params, صحيح n_default_params, قسم_ناں R,
              قسم_ناں... Ts>
    صنف دلیلاں_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      خاص:
        فراغ **kernel_params;
        حرف *دلیلاں_بفر;

        سانچا <صحيح i> ساكن constexpr صحيح account_for_default_params() {
            constexpr صحيح n_کل_params = حجم...(Ts);
            إذا constexpr (i >= n_nondefault_params) {
                أرجع n_کل_params - n_default_params +
                       (i - n_nondefault_params);
            } وإلا {
                أرجع i;
            }
        }

      عام:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        سانچا <صحيح i>
        ورتوں arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ساكن constexpr صحيح params_عدد = حجم...(Ts);

      خاص:
        سانچا <صحيح i> ساكن constexpr صحيح get_وتھ() {
            إذا constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                أرجع 0;
            } وإلا {
                constexpr صحيح prev_off = get_وتھ<i - 1>();
                constexpr صحيح prev_past_سرا =
                    prev_off + حجم(arg_type<i - 1>);
                ورتوں T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                إذا constexpr (prev_past_سرا % alignof(T) == 0) {
                    أرجع prev_past_سرا;
                }
                // otherwise bump prev_past_end to match alignment
                وإلا {
                    أرجع prev_past_سرا +
                           (alignof(T) - (prev_past_سرا % alignof(T)));
                }
            }
        }

        ساكن حرف *get_دلیلاں_بفر(فراغ **extra) {
            إذا (!extra)
                أرجع خالی_سوچک;
            لكل (; (std::size_t)*extra != 0; ++extra) {
                إذا ((std::size_t)*extra == 1) {
                    أرجع static_cast<حرف *>(*(extra + 1));
                }
            }
            أرجع خالی_سوچک;
        }

      عام:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        دلیلاں_selector(فراغ **kernel_params, فراغ **extra)
            : kernel_params(kernel_params),
              دلیلاں_بفر(get_دلیلاں_بفر(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        سانچا <صحيح i> arg_type<i> &get() {
            إذا (kernel_params) {
                أرجع *static_cast<arg_type<i> *>(kernel_params[i]);
            } وإلا {
                أرجع *reinterpret_cast<arg_type<i> *>(دلیلاں_بفر +
                                                        get_وتھ<i>());
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
        سانچا <قسم_ناں FuncT, قسم_ناں ArgSelector, std::size_t... اشاریہ_2>
        ساكن فراغ launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<اشاریہ_2...>) {
            func(selector.سانچا get<اشاریہ_2>()...);
        }
        ساكن فراغ سیٹ_execution_config(dim3 گروہ_range, dim3 local_range,
                                         بنانشانی صحيح local_mem_ناپ_2,
                                         قطار_ptr que) {
            إذا (que) {
                _que = que;
            } وإلا {
                _que = &get_default_قطار();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(گروہ_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_ناپ = local_mem_ناپ_2;


        }
        ساكن مضمن std::mutex kernel_function_ptr_map_mutex;

      عام:
        /// Variables for storing execution configuration.
        ساكن مضمن thread_local sycl::queue *_que = خالی_سوچک;
        ساكن مضمن thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ساكن مضمن thread_local بنانشانی صحيح local_mem_ناپ = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ساكن مضمن std::map<
            ثابت فراغ *,
            std::function<فراغ(dim3, dim3, فراغ **, بنانشانی صحيح, قطار_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ساكن فراغ رجسٹر_kernel_ptr(
            ثابت فراغ *func,
            std::function<فراغ(dim3, dim3, فراغ **, بنانشانی صحيح, قطار_ptr)>
                launcher) {
            std::lock_guard<std::mutex> جندرا(kernel_function_ptr_map_mutex);
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
        سانچا <قسم_ناں FuncT, قسم_ناں... دلیلاں_T>
        ساكن std::enable_if_t<std::is_invocable_v<FuncT *, دلیلاں_T...>, فراغ>
        launch(FuncT *func, dim3 گروہ_range, dim3 local_range,
               بنانشانی صحيح local_mem_ناپ_2, قطار_ptr que, دلیلاں_T... دلیلاں_2) {
            سیٹ_execution_config(گروہ_range, local_range, local_mem_ناپ_2, que);
            func(دلیلاں_2...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        ساكن فراغ launch(ثابت فراغ *func, dim3 گروہ_range, dim3 local_range,
                           فراغ **دلیلاں_2, بنانشانی صحيح local_mem_ناپ_2,
                           قطار_ptr que) {
            std::lock_guard<std::mutex> جندرا(kernel_function_ptr_map_mutex);
            خودکار Iter = kernel_function_ptr_map.find(func);
            إذا (Iter == kernel_function_ptr_map.end()) {
                ارم std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(گروہ_range, local_range, دلیلاں_2, local_mem_ناپ_2, que);
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
        سانچا <قسم_ناں FuncT>
        ساكن std::enable_if_t<std::is_function_v<FuncT>, فراغ>
        launch(FuncT *func, dim3 گروہ_range, dim3 local_range, فراغ **دلیلاں_2,
               بنانشانی صحيح local_mem_ناپ_2, قطار_ptr que) {
            constexpr size_t p_عدد = دلیلاں_selector<0, 0, FuncT>::params_عدد;
            سیٹ_execution_config(گروہ_range, local_range, local_mem_ناپ_2, que);
            دلیلاں_selector<p_عدد, p_عدد, FuncT> selector(دلیلاں_2, خالی_سوچک);
            launch_helper(func, selector, std::make_index_sequence<p_عدد>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    سانچا <قسم_ناں T>
    T select_from_sub_گروہ(
        sycl::sub_گروہ g,
        T x,
        صحيح remote_local_id,
        صحيح logical_sub_گروہ_ناپ = 32) {
      بنانشانی صحيح شروع_اشاریہ = g.get_local_خطی_id() /
                                 logical_sub_گروہ_ناپ *
                                 logical_sub_گروہ_ناپ;
      أرجع sycl::select_from_گروہ(
          g, x, شروع_اشاریہ + remote_local_id % logical_sub_گروہ_ناپ);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    سانچا <قسم_ناں T>
    فراغ ldmatrix(uintptr_t addr, T* m, منطقي trans = خطأ, بنانشانی mat = 0) {
      خودکار sg = sycl::ext::oneapi::this_work_شے::get_sub_گروہ();
      صحيح lane = sg.get_local_خطی_id();

      صحيح lane_group8_صف = lane / 8;
      صحيح lane_group8_خانہ = lane % 8;

      إذا (!trans) {
        // calculate the source lane
        صحيح src_lane = 2 * lane_group8_صف;
        إذا (lane_group8_خانہ >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        خودکار recv_addr_uintp =
            dpct::select_from_sub_گروہ(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        خودکار recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_خانہ % 4];
      } وإلا {
        // calculate the source lane
        صحيح src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        خودکار recv_addr_uintp_1 =
            dpct::select_from_sub_گروہ(sg, addr, mat * 8 + src_lane);
        خودکار recv_addr_uintp_2 =
            dpct::select_from_sub_گروہ(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        خودکار recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        خودکار recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        صحيح اشاریہ = lane / 4;
        sycl::half val0 = recv_addr_1[اشاریہ];
        sycl::half val1 = recv_addr_2[اشاریہ];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    سانچا <قسم_ناں T>
    فراغ ldmatrix(uintptr_t addr, T* m1, T* m2, منطقي trans = خطأ) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    سانچا <قسم_ناں T>
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
    سانچا <قسم_ناں T>
    بنية MMAType {
      ورتوں PackType = uint32_t;
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
    سانچا <صحيح M, صحيح N, صحيح K, قسم_ناں ABType, قسم_ناں CDType>
    فراغ mma(
        متطاير فراغ** d_mat_frag,
        فراغ* a_mat_frag,
        فراغ* b_mat_frag,
        فراغ* c_mat_frag) {
      خودکار d = reinterpret_cast<متطاير CDType**>(d_mat_frag);
      خودکار a =
          reinterpret_cast<قسم_ناں MMAType<ABType>::PackType*>(a_mat_frag);
      خودکار b =
          reinterpret_cast<قسم_ناں MMAType<ABType>::PackType*>(b_mat_frag);
      خودکار c = reinterpret_cast<CDType*>(c_mat_frag);

      خودکار sg = sycl::ext::oneapi::this_work_شے::get_sub_گروہ();
      صحيح lane = sg.get_local_خطی_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      چھوٹا صف_لدو_وتھ = 4 * (lane >> 2);
      چھوٹا خانہ_لدو_وتھ = 8 * (lane % 4);

      إذا constexpr (M == 8 && N == 8 && K == 4) {
        إذا constexpr (std::is_floating_point_v<CDType>) {
          خانہ_لدو_وتھ = صف_لدو_وتھ % 16;

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
          قسم_ناں MMAType<ABType>::PackType recv_a[2], recv_b[2];

          لكل (صحيح i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_لدو_وتھ + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], خانہ_لدو_وتھ + i);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

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
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], خانہ_لدو_وتھ + i + 16);

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
      } وإلا إذا constexpr (M == 8 && N == 8 && K == 16) {
        إذا constexpr (std::is_integral_v<ABType>) {
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
            قسم_ناں MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_گروہ(sg, a[0], صف_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i + 4);

            خودکار a = reinterpret_cast<ABType*>(&recv_a);
            خودکار b = reinterpret_cast<ABType*>(recv_b);

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
      } وإلا إذا constexpr (M == 16 && N == 8 && K == 8) {
        إذا constexpr (std::is_floating_point_v<CDType>) {
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
            قسم_ناں MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_لدو_وتھ + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i + 4);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

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
      } وإلا إذا constexpr (M == 16 && N == 8 && K == 16) {
        إذا constexpr (std::is_floating_point_v<CDType>) {
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
            قسم_ناں MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_لدو_وتھ + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[2], صف_لدو_وتھ + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_لدو_وتھ + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_گروہ(sg, a[3], صف_لدو_وتھ + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], خانہ_لدو_وتھ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_گروہ(sg, b[1], خانہ_لدو_وتھ + 4 + i);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

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
        } وإلا إذا constexpr (std::is_integral_v<ABType>) {
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
            قسم_ناں MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_لدو_وتھ + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i + 4);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

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
      } وإلا إذا constexpr (M == 16 && N == 8 && K == 32) {
        إذا constexpr (std::is_integral_v<ABType>) {
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
            قسم_ناں MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_لدو_وتھ + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], خانہ_لدو_وتھ + i + 4);

            خودکار a = reinterpret_cast<ABType*>(recv_a);
            خودکار b = reinterpret_cast<ABType*>(recv_b);

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
            قسم_ناں MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[2], صف_لدو_وتھ + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[3], صف_لدو_وتھ + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[1], خانہ_لدو_وتھ + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], خانہ_لدو_وتھ + i + 4);

            خودکار a = reinterpret_cast<ABType*>(recv_a);
            خودکار b = reinterpret_cast<ABType*>(recv_b);

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

#جے_انت // GGML_SYCL_DPCT_HELPER_HPP

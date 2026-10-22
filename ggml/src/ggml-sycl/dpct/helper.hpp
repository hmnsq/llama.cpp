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

#اگر_غیر_تعریف_شدہ GGML_SYCL_DPCT_HELPER_HPP
#تعریف_کرو GGML_SYCL_DPCT_HELPER_HPP

#شامل <cstdlib>
#شامل <iostream>
#شامل <map>

#شامل <sycl/sycl.hpp>
#شامل <sycl/half_type.hpp>
#شامل <oneapi/mkl.hpp>

#اگر defined(__linux__)
#شامل <sys/mman.h>
#وگرنہ_اگر defined(_WIN64)
#اگر_غیر_تعریف_شدہ NOMINMAX
#تعریف_کرو NOMINMAX
#اگر_ختم
#شامل <windows.h>
#ورنہ
#غلطی "Only support Windows and Linux."
#اگر_ختم

#اگر defined(__linux__)
#شامل <unistd.h>
#شامل <sys/syscall.h>
#اگر_ختم
#اگر defined(_WIN64)
#اگر_غیر_تعریف_شدہ NOMINMAX
#تعریف_کرو NOMINMAX
#اگر_ختم
#شامل <windows.h>
#اگر_ختم


#تعریف_کرو DPCT_COMPATIBILITY_عارضی (900)

#اگر defined(_MSC_VER)
#تعریف_کرو __dpct_align__(n) __declspec(align(n))
#تعریف_کرو __dpct_inline__ __forceinline
#ورنہ
#تعریف_کرو __dpct_align__(n) __attribute__((aligned(n)))
#تعریف_کرو __dpct_inline__ __inline__ __attribute__((always_inline))
#اگر_ختم

#اگر defined(_MSC_VER)
#تعریف_کرو __dpct_noinline__ __declspec(noinline)
#ورنہ
#تعریف_کرو __dpct_noinline__ __attribute__((noinline))
#اگر_ختم

#تعریف_کرو DPCT_غیر_مستعمل(x) (خالی)(x)

[[noreturn]] اندرونی خالی _abort(ثابت حرف * str) {
    std::cerr << str << std::سطر_ختم;
    std::abort();
}

اندرونی std::string get_device_type_name(ثابت sycl::device &Device) {
    خودکار DeviceType = Device.get_info<sycl::info::device::device_type>();
    انتخاب (DeviceType) {
    حالت sycl::info::device_type::cpu:
        واپس "cpu";
    حالت sycl::info::device_type::gpu:
        واپس "gpu";
    حالت sycl::info::device_type::host:
        واپس "host";
    حالت sycl::info::device_type::accelerator:
        واپس "acc";
    طےشدہ:
        واپس "unknown";
    }
}

اندرونی std::string get_device_backend_and_type(ثابت sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    واپس device_type.str();
}

سانچہ <قسم_نام Ts> ساخت میٹرکس_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     قدر_info[2];
    std::int64_t           سائز_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

نامخلا dpct
{
    قسمتعریف sycl::queue *قطار_ptr;
    قسمتعریف sycl::event *event_ptr;
    قسمتعریف حرف *device_ptr;
    قسمتعریف uint8_t byte_t;
    قسمتعریف sycl::بفر<byte_t> بفر_t;

    /// SYCL default exception handler
    اندرونی خودکار exception_handler = [](sycl::exception_list exceptions)
    {
        برائے (std::exception_ptr ثابت &e : exceptions)
        {
            کوشش
            {
                std::rethrow_exception(e);
            }
            پکڑو (sycl::exception ثابت &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::سطر_ختم
                          << e.what() << std::سطر_ختم
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::سطر_ختم;
            }
        }
    };

    شمار error_code
    {
        success = 0,
        default_error = 999
    };

    شمار memcpy_direction
    {
        host_مقصد_host,
        host_مقصد_device,
        device_مقصد_host,
        device_مقصد_device,
        automatic
    };

    شمار حافظہ_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    شمار جماعت library_ڈیٹا_t : غیرعلامتی حرف
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
        library_ڈیٹا_t_سائز
    };

    سانچہ <قسم_نام T>
    ساخت ڈیٹا_Type_2
    {
        استعمال T2 = T;
    };
    سانچہ <قسم_نام T>
    ساخت ڈیٹا_Type_2<sycl::vec<T, 2>>
    {
        استعمال T2 = std::complex<T>;
    };

    ساکن خالی destroy_event(event_ptr event)
    {
        حذف event;
    }

    ساکن اندرونی غیرعلامتی صحیح get_tid()
    {
#اگر defined(__linux__)
        واپس syscall(SYS_gettid);
#وگرنہ_اگر defined(_WIN64)
        واپس GetCurrentThreadId();
#ورنہ
#غلطی "Only support Windows and Linux."
#اگر_ختم
    }

    نامخلا detail
    {
        ساکن خالی get_نسخہ(ثابت sycl::device &dev, صحیح &major, صحیح &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::نسخہ_2>();
            std::string::size_type i = 0;
            جب (i < ver.size()) {
              اگر (ہندسہ_ہے(ver[i]))
                توڑو;
              i++;
            }
            major = std::stoi(&(ver[i]));
            جب (i < ver.size()) {
              اگر (ver[i] == '.')
                توڑو;
              i++;
            }
            اگر (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } ورنہ {
              // c.
              minor = 0;
            }
        }

        سانچہ <قسم_نام لیبل, قسم_نام T>
        جماعت generic_error_type
        {
        عوامی:
            generic_error_type() = طےشدہ;
            generic_error_type(T value) : value{value} {}
            عامل T() ثابت { واپس value; }

        نجی:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    جماعت dim3 {
        عوامی:
        غیرعلامتی x, y, z;

        constexpr dim3(غیرعلامتی x = 1, غیرعلامتی y = 1, غیرعلامتی z = 1)
            : x(x), y(y), z(z) {}

        dim3(ثابت sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        عامل sycl::range<3>() ثابت { واپس sycl::range<3>(z, y, x); }
    }; // namespace dim3

    اندرونی dim3 عامل*(ثابت dim3 &a, ثابت dim3 &b) {
    واپس dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    جماعت pitched_ڈیٹا
    {
    عوامی:
        pitched_ڈیٹا() : pitched_ڈیٹا(خالی_اشاریہ, 0, 0, 0) {}
        pitched_ڈیٹا(خالی *ڈیٹا, size_t pitch, size_t x, size_t y)
            : ڈیٹا_2(ڈیٹا), _pitch(pitch), _x(x), _y(y) {}

        خالی *get_ڈیٹا_ptr() { واپس ڈیٹا_2; }
        خالی سیٹ_ڈیٹا_ptr(خالی *ڈیٹا) { ڈیٹا_2 = ڈیٹا; }

        size_t get_pitch() { واپس _pitch; }
        خالی سیٹ_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { واپس _x; }
        خالی سیٹ_x(size_t x) { _x = x; }

        size_t get_y() { واپس _y; }
        خالی سیٹ_y(size_t y) { _y = y; }

    نجی:
        خالی *ڈیٹا_2;
        size_t _pitch, _x, _y;
    };

    جماعت device_info
    {
    عوامی:
        // get interface
        ثابت حرف *get_name() ثابت { واپس _name; }
        حرف *get_name() { واپس _name; }
        سانچہ <قسم_نام Work_شے_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_شے_Sizes_Ty, صحیح *>,
                                   صحیح> = 0>
        خودکار get_max_work_شے_sizes() ثابت
        {
            اگر constexpr (std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>>)
                واپس sycl::range<3>(max_work_شے_sizes_i[0],
                                      max_work_شے_sizes_i[1],
                                      max_work_شے_sizes_i[2]);
            ورنہ
            {
                واپس max_work_شے_sizes_i;
            }
        }
        سانچہ <قسم_نام Work_شے_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_شے_Sizes_Ty, صحیح *>,
                                   صحیح> = 0>
        خودکار get_max_work_شے_sizes()
        {
            اگر constexpr (std::is_same_v<Work_شے_Sizes_Ty, sycl::range<3>>)
                واپس sycl::range<3>(max_work_شے_sizes_i[0],
                                      max_work_شے_sizes_i[1],
                                      max_work_شے_sizes_i[2]);
            ورنہ
            {
                واپس max_work_شے_sizes_i;
            }
        }
        منطقی get_host_unified_حافظہ() ثابت { واپس host_unified_حافظہ; }
        صحیح get_major_نسخہ() ثابت { واپس _major; }
        صحیح get_minor_نسخہ() ثابت { واپس _minor; }
        صحیح get_integrated() ثابت { واپس _integrated; }
        صحیح get_max_clock_frequency() ثابت { واپس _frequency; }
        صحیح get_max_compute_units() ثابت { واپس _max_compute_units; }
        صحیح get_max_work_گروہ_سائز() ثابت { واپس max_work_گروہ_سائز_2; }
        صحیح get_max_sub_گروہ_سائز() ثابت { واپس max_sub_گروہ_سائز; }
        صحیح get_max_work_اشیاء_فی_compute_unit() ثابت
        {
            واپس max_work_اشیاء_فی_compute_unit;
        }
        صحیح get_max_رجسٹر_سائز_فی_work_گروہ() ثابت
        {
            واپس max_رجسٹر_سائز_فی_work_گروہ;
        }
        سانچہ <قسم_نام NDRange_سائز_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_سائز_Ty, size_t *> ||
                                       std::is_same_v<NDRange_سائز_Ty, صحیح *>,
                                   صحیح> = 0>
        خودکار get_max_nd_range_سائز() ثابت
        {
            اگر constexpr (std::is_same_v<NDRange_سائز_Ty, size_t *>)
                واپس max_nd_range_سائز;
            ورنہ
                واپس max_nd_range_سائز_i;
        }
        سانچہ <قسم_نام NDRange_سائز_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_سائز_Ty, size_t *> ||
                                       std::is_same_v<NDRange_سائز_Ty, صحیح *>,
                                   صحیح> = 0>
        خودکار get_max_nd_range_سائز()
        {
            اگر constexpr (std::is_same_v<NDRange_سائز_Ty, size_t *>)
                واپس max_nd_range_سائز;
            ورنہ
                واپس max_nd_range_سائز_i;
        }
        size_t get_global_mem_سائز() ثابت { واپس global_mem_سائز_2; }
        size_t get_local_mem_سائز() ثابت { واپس local_mem_سائز; }
        size_t get_max_mem_alloc_سائز() ثابت { واپس max_mem_alloc_سائز; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        غیرعلامتی صحیح get_حافظہ_clock_شرح() ثابت { واپس حافظہ_clock_شرح; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        غیرعلامتی صحیح get_حافظہ_bus_چوڑائی() ثابت { واپس حافظہ_bus_چوڑائی; }
        uint32_t get_device_id() ثابت { واپس _device_id; }
        std::array<غیرعلامتی حرف, 16> get_uuid() ثابت { واپس _uuid; }
        /// Returns global memory cache size in bytes.
        غیرعلامتی صحیح get_global_mem_cache_سائز() ثابت
        {
            واپس global_mem_cache_سائز;
        }

        // set interface
        خالی سیٹ_name(ثابت حرف *name)
        {
            size_t لمبائی = لڑی_لمبائی(name);
            اگر (لمبائی < 256)
            {
                std::حافظہ_نقل(_name, name, لمبائی + 1);
            }
            ورنہ
            {
                std::حافظہ_نقل(_name, name, 255);
                _name[255] = '\0';
            }
        }
        خالی سیٹ_max_work_شے_sizes(ثابت sycl::range<3> max_work_شے_sizes)
        {
            برائے (صحیح i = 0; i < 3; ++i)
                max_work_شے_sizes_i[i] = max_work_شے_sizes[i];
        }
        [[deprecated]] خالی
        سیٹ_max_work_شے_sizes(ثابت sycl::id<3> max_work_شے_sizes)
        {
            برائے (صحیح i = 0; i < 3; ++i)
            {
                max_work_شے_sizes_i[i] = max_work_شے_sizes[i];
            }
        }
        خالی سیٹ_host_unified_حافظہ(منطقی host_unified_حافظہ_2)
        {
            host_unified_حافظہ = host_unified_حافظہ_2;
        }
        خالی سیٹ_major_نسخہ(صحیح major) { _major = major; }
        خالی سیٹ_minor_نسخہ(صحیح minor) { _minor = minor; }
        خالی سیٹ_integrated(صحیح integrated) { _integrated = integrated; }
        خالی سیٹ_max_clock_frequency(صحیح frequency) { _frequency = frequency; }
        خالی سیٹ_max_compute_units(صحیح max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        خالی سیٹ_global_mem_سائز(size_t global_mem_سائز)
        {
            global_mem_سائز_2 = global_mem_سائز;
        }
        خالی سیٹ_local_mem_سائز(size_t local_mem_سائز_2)
        {
            local_mem_سائز = local_mem_سائز_2;
        }
        خالی سیٹ_max_mem_alloc_سائز(size_t max_mem_alloc_سائز_2)
        {
            max_mem_alloc_سائز = max_mem_alloc_سائز_2;
        }
        خالی سیٹ_max_work_گروہ_سائز(صحیح max_work_گروہ_سائز)
        {
            max_work_گروہ_سائز_2 = max_work_گروہ_سائز;
        }
        خالی سیٹ_max_sub_گروہ_سائز(صحیح max_sub_گروہ_سائز_2)
        {
            max_sub_گروہ_سائز = max_sub_گروہ_سائز_2;
        }
        خالی
        سیٹ_max_work_اشیاء_فی_compute_unit(صحیح max_work_اشیاء_فی_compute_unit_2)
        {
            max_work_اشیاء_فی_compute_unit = max_work_اشیاء_فی_compute_unit_2;
        }
        خالی سیٹ_max_nd_range_سائز(صحیح max_nd_range_سائز_2[])
        {
            برائے (صحیح i = 0; i < 3; i++)
            {
                max_nd_range_سائز[i] = max_nd_range_سائز_2[i];
                max_nd_range_سائز_i[i] = max_nd_range_سائز_2[i];
            }
        }
        خالی سیٹ_حافظہ_clock_شرح(غیرعلامتی صحیح حافظہ_clock_شرح_2)
        {
            حافظہ_clock_شرح = حافظہ_clock_شرح_2;
        }
        خالی سیٹ_حافظہ_bus_چوڑائی(غیرعلامتی صحیح حافظہ_bus_چوڑائی_2)
        {
            حافظہ_bus_چوڑائی = حافظہ_bus_چوڑائی_2;
        }
        خالی
        سیٹ_max_رجسٹر_سائز_فی_work_گروہ(صحیح max_رجسٹر_سائز_فی_work_گروہ_2)
        {
            max_رجسٹر_سائز_فی_work_گروہ = max_رجسٹر_سائز_فی_work_گروہ_2;
        }
        خالی سیٹ_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        خالی سیٹ_uuid(std::array<غیرعلامتی حرف, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        خالی سیٹ_global_mem_cache_سائز(غیرعلامتی صحیح global_mem_cache_سائز_2)
        {
            global_mem_cache_سائز = global_mem_cache_سائز_2;
        }

    نجی:
        حرف _name[256];
        صحیح max_work_شے_sizes_i[3];
        منطقی host_unified_حافظہ = جھوٹ;
        صحیح _major;
        صحیح _minor;
        صحیح _integrated = 0;
        صحیح _frequency;
        // Set estimated value 3200000 kHz as default value.
        غیرعلامتی صحیح حافظہ_clock_شرح = 3200000;
        // Set estimated value 64 bits as default value.
        غیرعلامتی صحیح حافظہ_bus_چوڑائی = 64;
        غیرعلامتی صحیح global_mem_cache_سائز;
        صحیح _max_compute_units;
        صحیح max_work_گروہ_سائز_2;
        صحیح max_sub_گروہ_سائز;
        صحیح max_work_اشیاء_فی_compute_unit;
        صحیح max_رجسٹر_سائز_فی_work_گروہ;
        size_t global_mem_سائز_2;
        size_t local_mem_سائز;
        size_t max_mem_alloc_سائز;
        size_t max_nd_range_سائز[3];
        صحیح max_nd_range_سائز_i[3];
        uint32_t _device_id;
        std::array<غیرعلامتی حرف, 16> _uuid;
    };

    ساکن صحیح get_major_نسخہ(ثابت sycl::device &dev)
    {
        صحیح major, minor;
        detail::get_نسخہ(dev, major, minor);
        واپس major;
    }

    ساکن صحیح get_minor_نسخہ(ثابت sycl::device &dev)
    {
        صحیح major, minor;
        detail::get_نسخہ(dev, major, minor);
        واپس minor;
    }

    ساکن خالی get_device_info(device_info &برآمد, ثابت sycl::device &dev)
    {
        device_info prop;
        prop.سیٹ_name(dev.get_info<sycl::info::device::name>().c_str());

        صحیح major, minor;
        detail::get_نسخہ(dev, major, minor);
        prop.سیٹ_major_نسخہ(major);
        prop.سیٹ_minor_نسخہ(minor);

        prop.سیٹ_max_work_شے_sizes(
#اگر (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_شے_sizes>());
#ورنہ
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_شے_sizes<3>>());
#اگر_ختم
        prop.سیٹ_host_unified_حافظہ(dev.has(sycl::aspect::usm_host_allocations));

        prop.سیٹ_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.سیٹ_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.سیٹ_max_work_گروہ_سائز(
            dev.get_info<sycl::info::device::max_work_گروہ_سائز>());
        prop.سیٹ_global_mem_سائز(dev.get_info<sycl::info::device::global_mem_سائز>());
        prop.سیٹ_local_mem_سائز(dev.get_info<sycl::info::device::local_mem_سائز_2>());
        prop.سیٹ_max_mem_alloc_سائز(dev.get_info<sycl::info::device::max_mem_alloc_سائز_2>());

#اگر (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        اگر (dev.has(sycl::aspect::ext_intel_حافظہ_clock_شرح))
        {
            غیرعلامتی صحیح tmp =
                dev.get_info<sycl::ext::intel::info::device::حافظہ_clock_شرح_2>();
            اگر (tmp != 0)
                prop.سیٹ_حافظہ_clock_شرح(1000 * tmp);
        }
        اگر (dev.has(sycl::aspect::ext_intel_حافظہ_bus_چوڑائی))
        {
            prop.سیٹ_حافظہ_bus_چوڑائی(
                dev.get_info<sycl::ext::intel::info::device::حافظہ_bus_چوڑائی_2>());
        }
        اگر (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.سیٹ_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        اگر (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.سیٹ_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#وگرنہ_اگر defined(_MSC_VER) && !defined(__clang__)
#ہدایت پیغام("get_device_info: querying حافظہ_clock_شرح_2 and \
        حافظہ_bus_چوڑائی_2 are not supported by the compiler مستعمل. \
        Use 3200000 kHz as حافظہ_clock_شرح_2 طےشدہ value. \
        Use 64 بٹ as حافظہ_bus_چوڑائی_2 طےشدہ value.")
#ورنہ
#warning "get_device_info: querying حافظہ_clock_شرح_2 and \
        حافظہ_bus_چوڑائی_2 are not supported by the compiler مستعمل. \
        Use 3200000 kHz as حافظہ_clock_شرح_2 طےشدہ value. \
        Use 64 بٹ as حافظہ_bus_چوڑائی_2 طےشدہ value."
#اگر_ختم

        size_t max_sub_گروہ_سائز_2 = 1;
        std::vector<size_t> sub_گروہ_sizes =
            dev.get_info<sycl::info::device::sub_گروہ_sizes>();

        برائے (ثابت خودکار &sub_گروہ_سائز : sub_گروہ_sizes)
        {
            اگر (max_sub_گروہ_سائز_2 < sub_گروہ_سائز)
                max_sub_گروہ_سائز_2 = sub_گروہ_سائز;
        }

        prop.سیٹ_max_sub_گروہ_سائز(max_sub_گروہ_سائز_2);

        prop.سیٹ_max_work_اشیاء_فی_compute_unit(
            dev.get_info<sycl::info::device::max_work_گروہ_سائز>());
        صحیح max_nd_range_سائز_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.سیٹ_max_nd_range_سائز(max_nd_range_سائز_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.سیٹ_max_رجسٹر_سائز_فی_work_گروہ(65536);

        prop.سیٹ_global_mem_cache_سائز(
            dev.get_info<sycl::info::device::global_mem_cache_سائز_2>());
        برآمد = prop;
    }

    /// dpct device extension
    جماعت device_ext : عوامی sycl::device {
      قسمتعریف std::mutex mutex_type;

     عوامی:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> تالا(m_mutex);
        clear_queues();
      }
      device_ext(ثابت sycl::device &بنیاد) : sycl::device(بنیاد) {
        std::lock_guard<mutex_type> تالا(m_mutex);
        init_queues();
      }

      صحیح is_native_atomic_supported() { واپس 0; }
      صحیح get_major_نسخہ() ثابت { واپس dpct::get_major_نسخہ(*یہ); }

      صحیح get_minor_نسخہ() ثابت { واپس dpct::get_minor_نسخہ(*یہ); }

      صحیح get_max_compute_units() ثابت {
        واپس get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      صحیح get_max_clock_frequency() ثابت {
        واپس get_device_info().get_max_clock_frequency();
      }

      صحیح get_integrated() ثابت { واپس get_device_info().get_integrated(); }

      صحیح get_max_sub_گروہ_سائز() ثابت {
        واپس get_device_info().get_max_sub_گروہ_سائز();
      }

      صحیح get_max_رجسٹر_سائز_فی_work_گروہ() ثابت {
        واپس get_device_info().get_max_رجسٹر_سائز_فی_work_گروہ();
      }

      صحیح get_max_work_گروہ_سائز() ثابت {
        واپس get_device_info().get_max_work_گروہ_سائز();
      }

      صحیح get_mem_بنیاد_addr_align() ثابت {
        واپس get_info<sycl::info::device::mem_بنیاد_addr_align>();
      }

      size_t get_global_mem_سائز() ثابت {
        واپس get_device_info().get_global_mem_سائز();
      }

      size_t get_max_mem_alloc_سائز() ثابت {
        واپس get_device_info().get_max_mem_alloc_سائز();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      خالی get_حافظہ_info(size_t &free_حافظہ, size_t &کل_حافظہ) {
        کل_حافظہ = get_device_info().get_global_mem_سائز();
        ثابت حرف *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#اگر (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        اگر (!has(sycl::aspect::ext_intel_free_حافظہ)) {
          std::cerr << warning_info << std::سطر_ختم;
          free_حافظہ = کل_حافظہ;
        } ورنہ {
          free_حافظہ = get_info<sycl::ext::intel::info::device::free_حافظہ>();
        }
#ورنہ
        std::cerr << warning_info << std::سطر_ختم;
        free_حافظہ = کل_حافظہ;
#اگر defined(_MSC_VER) && !defined(__clang__)
#ہدایت پیغام("Querying the number of bytes of free memory is not supported")
#ورنہ
#warning "Querying the number of bytes of free memory is not supported"
#اگر_ختم
#اگر_ختم
      }

      خالی get_device_info(device_info &برآمد) ثابت {
        dpct::get_device_info(برآمد, *یہ);
      }

      device_info get_device_info() ثابت {
        device_info prop;
        dpct::get_device_info(prop, *یہ);
        واپس prop;
      }

      خالی reset() {
        std::lock_guard<mutex_type> تالا(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &اندر_ترتیب_قطار() { واپس q_اندر_ترتیب; }

      sycl::queue &برآمد_کا_ترتیب_قطار() { واپس q_برآمد_کا_ترتیب; }

      sycl::queue &default_قطار() { واپس اندر_ترتیب_قطار(); }

      خالی queues_wait_and_throw() {
        std::unique_lock<mutex_type> تالا(m_mutex);
        تالا.unlock();
        برائے (خودکار &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        تالا.تالا();
      }

      sycl::queue create_قطار(منطقی enable_exception_handler = جھوٹ) {
        واپس create_اندر_ترتیب_قطار(enable_exception_handler);
      }

      sycl::queue create_قطار(sycl::device device,
                               منطقی enable_exception_handler = جھوٹ) {
        واپس create_اندر_ترتیب_قطار(device, enable_exception_handler);
      }

      sycl::queue create_اندر_ترتیب_قطار(منطقی enable_exception_handler = جھوٹ) {
        std::lock_guard<mutex_type> تالا(m_mutex);
        واپس create_قطار_impl(enable_exception_handler,
                                 sycl::property::queue::اندر_ترتیب());
      }

      sycl::queue create_اندر_ترتیب_قطار(sycl::device device,
                                        منطقی enable_exception_handler = جھوٹ) {
        std::lock_guard<mutex_type> تالا(m_mutex);
        واپس create_قطار_impl(device, enable_exception_handler,
                                 sycl::property::queue::اندر_ترتیب());
      }

      sycl::queue create_برآمد_کا_ترتیب_قطار(
          منطقی enable_exception_handler = جھوٹ) {
        std::lock_guard<mutex_type> تالا(m_mutex);
        واپس create_قطار_impl(enable_exception_handler);
      }

      خالی destroy_قطار(sycl::queue queue) {
        std::lock_guard<mutex_type> تالا(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](ثابت sycl::queue &q) -> منطقی
                                    {
                                        واپس q == queue;
                                    }),
                    _queues.end());
      }
      خالی سیٹ_saved_قطار(sycl::queue q) {
        std::lock_guard<mutex_type> تالا(m_mutex);
        saved_قطار = q;
      }
      sycl::queue get_saved_قطار() ثابت {
        std::lock_guard<mutex_type> تالا(m_mutex);
        واپس saved_قطار;
      }

     نجی:
      خالی clear_queues() { _queues.clear(); }

      خالی init_queues() {
        q_اندر_ترتیب =
            create_قطار_impl(سچ, sycl::property::queue::اندر_ترتیب());
        q_برآمد_کا_ترتیب = create_قطار_impl(سچ);
        saved_قطار = default_قطار();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      سانچہ <جماعت... Properties>
      sycl::queue create_قطار_impl(منطقی enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        اگر (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *یہ, eh,
            sycl::property_list(
#اگر_تعریف_شدہ DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#اگر_ختم
                properties...)));

        واپس _queues.back();
      }

      سانچہ <جماعت... Properties>
      sycl::queue create_قطار_impl(sycl::device device,
                                    منطقی enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        اگر (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#اگر_تعریف_شدہ DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#اگر_ختم
                            properties...)));

        واپس _queues.back();
      }

      خالی get_نسخہ(صحیح &major, صحیح &minor) ثابت {
        detail::get_نسخہ(*یہ, major, minor);
      }
      sycl::queue q_اندر_ترتیب, q_برآمد_کا_ترتیب;
      sycl::queue saved_قطار;
      std::vector<sycl::queue> _queues;
      تغیرپذیر mutex_type m_mutex;
    };


    /// device manager
    جماعت dev_mgr
    {
    عوامی:
        device_ext &current_device()
        {
            غیرعلامتی صحیح dev_id = current_device_id();
            check_id(dev_id);
            واپس *_devs[dev_id];
        }
        device_ext &cpu_device() ثابت
        {
            std::lock_guard<std::recursive_mutex> تالا(m_mutex);
            اگر (_cpu_device == -1)
            {
                پھینکو std::runtime_error("no valid cpu device");
            }
            ورنہ
            {
                واپس *_devs[_cpu_device];
            }
        }
        device_ext &get_device(غیرعلامتی صحیح id) ثابت
        {
            std::lock_guard<std::recursive_mutex> تالا(m_mutex);
            check_id(id);
            واپس *_devs[id];
        }
        غیرعلامتی صحیح current_device_id() ثابت
        {
            std::lock_guard<std::recursive_mutex> تالا(m_mutex);
            خودکار it = _thread2dev_map.find(get_tid());
            اگر (it != _thread2dev_map.end())
                واپس it->second;
            واپس DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        خالی select_device(غیرعلامتی صحیح id)
        {
            std::lock_guard<std::recursive_mutex> تالا(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        غیرعلامتی صحیح device_گنتی() { واپس _devs.size(); }

        غیرعلامتی صحیح get_device_id(ثابت sycl::device &dev)
        {
            غیرعلامتی صحیح id = 0;
            برائے (خودکار &dev_شے : _devs)
            {
                اگر (*dev_شے == dev)
                {
                    واپس id;
                }
                id++;
            }
            واپس -1;
        }

        اندرونی std::string get_preferred_gpu_platform_name() {
            std::string نتیجہ;

            std::string filter = "";
            حرف* env = getenv("ONEAPI_DEVICE_SELECTOR");
            اگر (env) {
                اگر (std::لڑی_تلاش(env, "level_zero")) {
                    filter = "level-zero";
                }
                ورنہ اگر (std::لڑی_تلاش(env, "opencl")) {
                    filter = "opencl";
                }
                ورنہ اگر (std::لڑی_تلاش(env, "cuda")) {
                    filter = "cuda";
                }
                ورنہ اگر (std::لڑی_تلاش(env, "hip")) {
                    filter = "hip";
                }
                ورنہ {
                    پھینکو std::runtime_error("invalid device filter: " + std::string(env));
                }
            } ورنہ {
                خودکار default_device = sycl::device(sycl::default_selector_v);
                خودکار default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                اگر (std::لڑی_تلاش(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                ورنہ اگر (std::لڑی_تلاش(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                ورنہ اگر (std::لڑی_تلاش(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            خودکار platform_list = sycl::platform::get_platforms();

            برائے (ثابت خودکار& platform : platform_list) {
                خودکار devices = platform.get_devices();
                خودکار gpu_dev = std::find_if(devices.begin(), devices.end(), [](ثابت sycl::device& d) {
                    واپس d.is_gpu();
                });

                اگر (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    جاری;
                }

                خودکار platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_کم_case;
                platform_name_کم_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_کم_case.begin(), ::tolower);

                اگر (platform_name_کم_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    جاری;
                }

                نتیجہ = platform_name;
            }

            اگر (نتیجہ.خالی_خانے())
                پھینکو std::runtime_error("can not find preferred GPU platform");

            واپس نتیجہ;
        }

        سانچہ <جماعت DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<صحیح, DeviceSelector, ثابت sycl::device &>>
        select_device(ثابت DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            غیرعلامتی صحیح selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        ساکن dev_mgr &instance()
        {
            ساکن dev_mgr d_m;
            واپس d_m;
        }
        dev_mgr(ثابت dev_mgr &) = حذف;
        dev_mgr &عامل=(ثابت dev_mgr &) = حذف;
        dev_mgr(dev_mgr &&) = حذف;
        dev_mgr &عامل=(dev_mgr &&) = حذف;

    نجی:
        تغیرپذیر std::recursive_mutex m_mutex;
        ساکن منطقی compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            اگر(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) واپس سچ;
            اگر(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) واپس جھوٹ;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            واپس prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        ساکن صحیح convert_backend_اشاریہ(std::string & backend) {
            اگر (backend == "ext_oneapi_level_zero:gpu") واپس 0;
            اگر (backend == "opencl:gpu") واپس 1;
            اگر (backend == "ext_oneapi_cuda:gpu") واپس 2;
            اگر (backend == "ext_oneapi_hip:gpu") واپس 3;
            اگر (backend == "opencl:cpu") واپس 4;
            اگر (backend == "opencl:acc") واپس 5;
            لکھو("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        ساکن منطقی compare_backend(std::string &backend1, std::string &backend2) {
            واپس convert_backend_اشاریہ(backend1) < convert_backend_اشاریہ(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            اگر (default_device.is_cpu())
                _cpu_device = 0;

            خودکار Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            خودکار preferred_platform_name = get_preferred_gpu_platform_name();

            جب (!Platforms.خالی_خانے()) {
                خودکار Platform = Platforms.back();
                Platforms.pop_back();
                خودکار platform_name = Platform.get_info<sycl::info::platform::name>();
                اگر (platform_name.compare(preferred_platform_name) != 0) {
                    جاری;
                }
                خودکار devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                برائے (ثابت خودکار &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> کلیدیں;
            برائے(خودکار it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                کلیدیں.push_back(it->پہلا);
            }
            std::sort(کلیدیں.begin(), کلیدیں.end(), compare_backend);

            برائے (خودکار &کلید : کلیدیں) {
                std::vector<sycl::device> devs = backend_devices[کلید];
                std::sort(devs.begin(), devs.end(), compare_dev);
                برائے (ثابت خودکار &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            برائے (خودکار &dev : sycl_all_devs)
            {
                اگر (dev == default_device)
                {
                    جاری;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                اگر (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        خالی check_id(غیرعلامتی صحیح id) ثابت
        {
            اگر (id >= _devs.size())
            {
                پھینکو std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        ثابت غیرعلامتی صحیح DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<غیرعلامتی صحیح, غیرعلامتی صحیح> _thread2dev_map;
        صحیح _cpu_device = -1;
    };

    ساکن اندرونی sycl::queue &get_default_قطار()
    {
        واپس dev_mgr::instance().current_device().default_قطار();
    }

    نامخلا detail
    {
        شمار جماعت اشارہ_گر_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        ساکن اشارہ_گر_access_attribute get_اشارہ_گر_attribute(sycl::queue &q,
                                                              ثابت خالی *ptr)
        {
            انتخاب (sycl::get_اشارہ_گر_type(ptr, q.get_context()))
            {
            حالت sycl::usm::alloc::unknown:
                واپس اشارہ_گر_access_attribute::host_only;
            حالت sycl::usm::alloc::device:
                واپس اشارہ_گر_access_attribute::device_only;
            حالت sycl::usm::alloc::shared:
            حالت sycl::usm::alloc::host:
                واپس اشارہ_گر_access_attribute::host_device;
            }
        }

        سانچہ <قسم_نام ArgT>
        اندرونی constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((غیرعلامتی حرف)library_ڈیٹا_t::library_ڈیٹا_t_سائز <=
                              std::numeric_limits<غیرعلامتی حرف>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_ڈیٹا_t>, "Unsupported ArgT");
            واپس (std::uint64_t)Val;
        }

        سانچہ <قسم_نام پہلا_T, قسم_نام... RestT>
        اندرونی constexpr std::uint64_t get_type_combination_id(پہلا_T پہلا_Val,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_ڈیٹا_t::library_ڈیٹا_t_سائز <=
                              std::numeric_limits<غیرعلامتی حرف>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(حجم...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<پہلا_T, library_ڈیٹا_t>, "Unsupported FirstT");
            واپس get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)پہلا_Val);
        }

        جماعت mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#اگر defined(__linux__)
                mapped_پتہ_space =
                    (byte_t *)mmap(خالی_اشاریہ, mapped_region_سائز, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#وگرنہ_اگر defined(_WIN64)
                mapped_پتہ_space = (byte_t *)VirtualAlloc(
                    تہی,               // NULL specified as the base address parameter
                    mapped_region_سائز, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    صفحہ_NOACCESS);     // Protection = no access
#ورنہ
#غلطی "Only support Windows and Linux."
#اگر_ختم
                اگلا_free = mapped_پتہ_space;
            }

        عوامی:
            استعمال بفر_id_t = صحیح;

            ساخت تخصیص
            {
                بفر_t بفر;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#اگر defined(__linux__)
                munmap(mapped_پتہ_space, mapped_region_سائز);
#وگرنہ_اگر defined(_WIN64)
                VirtualFree(mapped_پتہ_space, 0, MEM_RELEASE);
#ورنہ
#غلطی "Only support Windows and Linux."
#اگر_ختم
            }

            mem_mgr(ثابت mem_mgr &) = حذف;
            mem_mgr &عامل=(ثابت mem_mgr &) = حذف;
            mem_mgr(mem_mgr &&) = حذف;
            mem_mgr &عامل=(mem_mgr &&) = حذف;

            /// Allocate
            خالی *mem_alloc(size_t size)
            {
                اگر (!size)
                    واپس خالی_اشاریہ;
                std::lock_guard<std::mutex> تالا(m_mutex);
                اگر (اگلا_free + size > mapped_پتہ_space + mapped_region_سائز)
                {
                    پھینکو std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                //تخصیص
                sycl::range<1> r(size);
                بفر_t buf(r);
                تخصیص A{buf, اگلا_free, size};
                // Map allocation to device pointer
                خالی *نتیجہ = اگلا_free;
                m_map.emplace(اگلا_free + size, A);
                // Update pointer to the next free space.
                اگلا_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                واپس نتیجہ;
            }

            /// Deallocate
            خالی mem_free(ثابت خالی *ptr)
            {
                اگر (!ptr)
                    واپس;
                std::lock_guard<std::mutex> تالا(m_mutex);
                خودکار it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            تخصیص translate_ptr(ثابت خالی *ptr)
            {
                std::lock_guard<std::mutex> تالا(m_mutex);
                خودکار it = get_map_iterator(ptr);
                واپس it->second;
            }

            /// Check if the pointer represents device pointer or not.
            منطقی is_device_ptr(ثابت خالی *ptr) ثابت
            {
                std::lock_guard<std::mutex> تالا(m_mutex);
                واپس (mapped_پتہ_space <= ptr) &&
                       (ptr < mapped_پتہ_space + mapped_region_سائز);
            }

            /// Returns the instance of memory manager singleton.
            ساکن mem_mgr &instance()
            {
                ساکن mem_mgr m;
                واپس m;
            }

        نجی:
            std::map<byte_t *, تخصیص> m_map;
            تغیرپذیر std::mutex m_mutex;
            byte_t *mapped_پتہ_space;
            byte_t *اگلا_free;
            ثابت size_t mapped_region_سائز = 128ull * 1024 * 1024 * 1024;
            ثابت size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            ثابت size_t extra_padding = 0;

            std::map<byte_t *, تخصیص>::iterator get_map_iterator(ثابت خالی *ptr)
            {
                خودکار it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<ثابت byte_t *>(ptr)));
                اگر (it == m_map.end())
                {
                    // Not a virtual pointer.
                    پھینکو std::runtime_error("can not get buffer from non-virtual pointer");
                }
                ثابت تخصیص &alloc = it->second;
                اگر (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    پھینکو std::runtime_error("invalid virtual pointer");
                }
                واپس it;
            }
        };

        سانچہ <جماعت T, حافظہ_region حافظہ_2, size_t Dimension>
        جماعت accessor;
        سانچہ <حافظہ_region حافظہ_2, جماعت T = byte_t>
        جماعت حافظہ_traits
        {
        عوامی:
            ساکن constexpr sycl::access::ہدف ہدف =
                sycl::access::ہدف::device;
            ساکن constexpr sycl::access_منوال mode =
                (حافظہ_2 == constant) ? sycl::access_منوال::اقرأ
                                     : sycl::access_منوال::read_write;
            ساکن constexpr size_t type_سائز = حجم(T);
            استعمال element_t =
                قسم_نام std::conditional<حافظہ_2 == constant, ثابت T, T>::type;
            استعمال قدر_t = قسم_نام std::remove_cv<T>::type;
            سانچہ <size_t Dimension = 1>
            استعمال accessor_t = قسم_نام std::conditional<
                حافظہ_2 == local, sycl::local_accessor<قدر_t, Dimension>,
                sycl::accessor<T, Dimension, mode, ہدف>>::type;
            استعمال اشارہ_گر_t = T *;
        };

        ساکن اندرونی خالی *dpct_malloc(size_t size, sycl::queue &q)
        {
            واپس sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#تعریف_کرو PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        ساکن اندرونی خالی *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            واپس dpct_malloc(pitch * y * z, q);
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
        سانچہ <قسم_نام قدر_T>
        ساکن اندرونی sycl::event dpct_memset(sycl::queue &q, خالی *dev_ptr,
                                              قدر_T value, size_t size)
        {
            واپس q.fill(dev_ptr, value, size);
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
        سانچہ <قسم_نام قدر_T>
        ساکن اندرونی std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_ڈیٹا ڈیٹا, قدر_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t قاش_2 = ڈیٹا.get_pitch() * ڈیٹا.get_y();
            غیرعلامتی حرف *ڈیٹا_surface = (غیرعلامتی حرف *)ڈیٹا.get_ڈیٹا_ptr();
            برائے (size_t z = 0; z < size.get(2); ++z)
            {
                غیرعلامتی حرف *ڈیٹا_ptr = ڈیٹا_surface;
                برائے (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, ڈیٹا_ptr, value, size.get(0)));
                    ڈیٹا_ptr += ڈیٹا.get_pitch();
                }
                ڈیٹا_surface += قاش_2;
            }
            واپس event_list;
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
        سانچہ <قسم_نام قدر_T>
        ساکن اندرونی std::vector<sycl::event>
        dpct_memset(sycl::queue &q, خالی *ptr, size_t pitch, قدر_T val, size_t x,
                    size_t y)
        {
            واپس dpct_memset(q, pitched_ڈیٹا(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        ساکن memcpy_direction deduce_memcpy_direction(sycl::queue &q, خالی *مقصد_ptr,
                                                        ثابت خالی *from_ptr,
                                                        memcpy_direction dir)
        {
            انتخاب (dir)
            {
            حالت memcpy_direction::host_مقصد_host:
            حالت memcpy_direction::host_مقصد_device:
            حالت memcpy_direction::device_مقصد_host:
            حالت memcpy_direction::device_مقصد_device:
                واپس dir;
            حالت memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                ساکن ثابت memcpy_direction
                    direction_جدول[static_cast<غیرعلامتی>(اشارہ_گر_access_attribute::end)]
                                   [static_cast<غیرعلامتی>(اشارہ_گر_access_attribute::end)] =
                                       {{memcpy_direction::host_مقصد_host,
                                         memcpy_direction::device_مقصد_host,
                                         memcpy_direction::host_مقصد_host},
                                        {memcpy_direction::host_مقصد_device,
                                         memcpy_direction::device_مقصد_device,
                                         memcpy_direction::device_مقصد_device},
                                        {memcpy_direction::host_مقصد_host,
                                         memcpy_direction::device_مقصد_device,
                                         memcpy_direction::device_مقصد_device}};
                واپس direction_جدول[static_cast<غیرعلامتی>(get_اشارہ_گر_attribute(
                    q, مقصد_ptr))][static_cast<غیرعلامتی>(get_اشارہ_گر_attribute(q, from_ptr))];
            }
            طےشدہ:
                پھینکو std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        ساکن sycl::event
        dpct_memcpy(sycl::queue &q, خالی *مقصد_ptr, ثابت خالی *from_ptr, size_t size,
                    memcpy_direction direction,
                    ثابت std::vector<sycl::event> &dep_events = {})
        {
            اگر (!size)
                واپس sycl::event{};
            واپس q.حافظہ_نقل(مقصد_ptr, from_ptr, size, dep_events);
            DPCT_غیر_مستعمل(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        ساکن اندرونی size_t get_copy_range(sycl::range<3> size, size_t قاش_2,
                                            size_t pitch)
        {
            واپس قاش_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        ساکن اندرونی size_t get_انحراف(sycl::id<3> id, size_t قاش_2,
                                        size_t pitch)
        {
            واپس قاش_2 * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        ساکن اندرونی std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, خالی *مقصد_ptr, ثابت خالی *from_ptr,
                    sycl::range<3> مقصد_range, sycl::range<3> from_range,
                    sycl::id<3> مقصد_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    ثابت std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            جماعت host_بفر
            {
                خالی *_buf;
                size_t سائز_2;
                sycl::queue &_q;
                ثابت std::vector<sycl::event> &_deps; // free operation depends

            عوامی:
                host_بفر(size_t size, sycl::queue &q,
                            ثابت std::vector<sycl::event> &deps)
                    : _buf(std::حافظہ_دو(size)), سائز_2(size), _q(q), _deps(deps) {}
                خالی *get_ptr() ثابت { واپس _buf; }
                size_t get_سائز() ثابت { واپس سائز_2; }
                ~host_بفر()
                {
                    اگر (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::آزاد(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t مقصد_قاش = مقصد_range.get(1) * مقصد_range.get(0),
                   from_قاش = from_range.get(1) * from_range.get(0);
            غیرعلامتی حرف *مقصد_surface =
                (غیرعلامتی حرف *)مقصد_ptr + get_انحراف(مقصد_id, مقصد_قاش, مقصد_range.get(0));
            ثابت غیرعلامتی حرف *from_surface =
                (ثابت غیرعلامتی حرف *)from_ptr +
                get_انحراف(from_id, from_قاش, from_range.get(0));

            اگر (مقصد_قاش == from_قاش && مقصد_قاش == size.get(1) * size.get(0))
            {
                واپس {dpct_memcpy(q, مقصد_surface, from_surface, مقصد_قاش * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, مقصد_ptr, from_ptr, direction);
            size_t سائز_قاش = size.get(1) * size.get(0);
            انتخاب (direction)
            {
            حالت host_مقصد_host:
                برائے (size_t z = 0; z < size.get(2); ++z)
                {
                    غیرعلامتی حرف *مقصد_ptr = مقصد_surface;
                    ثابت غیرعلامتی حرف *from_ptr = from_surface;
                    اگر (مقصد_range.get(0) == from_range.get(0) &&
                        مقصد_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, مقصد_ptr, from_ptr, سائز_قاش,
                                                         direction, dep_events));
                    }
                    ورنہ
                    {
                        برائے (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, مقصد_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            مقصد_ptr += مقصد_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    مقصد_surface += مقصد_قاش;
                    from_surface += from_قاش;
                }
                توڑو;
            حالت host_مقصد_device:
            {
                host_بفر buf(get_copy_range(size, مقصد_قاش, مقصد_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                اگر (مقصد_قاش == سائز_قاش)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, مقصد_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_مقصد_host, dep_events);
                }
                ورنہ
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, مقصد_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_مقصد_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), مقصد_surface, buf.get_سائز(),
                                        device_مقصد_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, مقصد_surface, buf.get_ptr(),
                                                 buf.get_سائز(), host_مقصد_device,
                                                 host_events));
                توڑو;
            }
            حالت device_مقصد_host:
            {
                host_بفر buf(get_copy_range(size, from_قاش, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, مقصد_surface, buf.get_ptr(), مقصد_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_مقصد_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_سائز(),
                                                         device_مقصد_host, dep_events)});
                توڑو;
            }
            حالت device_مقصد_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<جماعت dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        مقصد_surface[get_انحراف(id, مقصد_قاش, مقصد_range.get(0))] =
                            from_surface[get_انحراف(id, from_قاش, from_range.get(0))];
                    }); }));
                توڑو;
            طےشدہ:
                پھینکو std::runtime_error("dpct_memcpy: invalid direction value");
            }
            واپس event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        ساکن اندرونی std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_ڈیٹا مقصد, sycl::id<3> مقصد_id,
                    pitched_ڈیٹا from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            واپس dpct_memcpy(q, مقصد.get_ڈیٹا_ptr(), from.get_ڈیٹا_ptr(),
                               sycl::range<3>(مقصد.get_pitch(), مقصد.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), مقصد_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        ساکن اندرونی std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, خالی *مقصد_ptr, ثابت خالی *from_ptr,
                    size_t مقصد_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            واپس dpct_memcpy(q, مقصد_ptr, from_ptr, sycl::range<3>(مقصد_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        نامخلا deprecated
        {

            سانچہ <قسم_نام T, sycl::usm::alloc Alloc_قسم>
            جماعت usm_allocator
            {
            نجی:
                استعمال Alloc = sycl::usm_allocator<T, Alloc_قسم>;
                Alloc _impl;

            عوامی:
                استعمال value_type = قسم_نام std::allocator_traits<Alloc>::value_type;
                استعمال pointer = قسم_نام std::allocator_traits<Alloc>::pointer;
                استعمال const_pointer = قسم_نام std::allocator_traits<Alloc>::const_pointer;
                استعمال void_pointer = قسم_نام std::allocator_traits<Alloc>::void_pointer;
                استعمال const_void_pointer =
                    قسم_نام std::allocator_traits<Alloc>::const_void_pointer;
                استعمال reference = قسم_نام std::allocator_traits<Alloc>::value_type &;
                استعمال const_reference =
                    ثابت قسم_نام std::allocator_traits<Alloc>::value_type &;
                استعمال difference_type =
                    قسم_نام std::allocator_traits<Alloc>::difference_type;
                استعمال size_type = قسم_نام std::allocator_traits<Alloc>::size_type;
                استعمال propagate_on_container_copy_assignment = قسم_نام std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                استعمال propagate_on_container_move_assignment = قسم_نام std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                استعمال propagate_on_container_swap =
                    قسم_نام std::allocator_traits<Alloc>::propagate_on_container_swap;
                استعمال is_always_equal =
                    قسم_نام std::allocator_traits<Alloc>::is_always_equal;

                سانچہ <قسم_نام U>
                ساخت rebind
                {
                    قسمتعریف usm_allocator<U, Alloc_قسم> other;
                };

                usm_allocator() : _impl(dpct::get_default_قطار()) {}
                ~usm_allocator() {}
                usm_allocator(ثابت usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { واپس &r; }
                const_pointer address(const_reference r) { واپس &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = خالی_اشاریہ)
                {
                    واپس std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                خالی deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() ثابت
                {
                    واپس std::allocator_traits<Alloc>::max_size(_impl);
                }
                منطقی عامل==(ثابت usm_allocator &other) ثابت { واپس _impl == other._impl; }
                منطقی عامل!=(ثابت usm_allocator &other) ثابت { واپس _impl != other._impl; }
            };

        } // namespace deprecated

        اندرونی خالی dpct_free(خالی *ptr,
                              ثابت sycl::queue &q)
        {
            اگر (ptr)
            {
                sycl::آزاد(ptr, q.get_context());
            }
        }

        سانچہ <قسم_نام T>
        اندرونی خودکار get_حافظہ(ثابت خالی *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<خالی *>(x));
            واپس new_x;
        }

        سانچہ <قسم_نام T>
        اندرونی قسم_نام ڈیٹا_Type_2<T>::T2 get_قدر(ثابت T *s, sycl::queue &q)
        {
            استعمال Ty = قسم_نام ڈیٹا_Type_2<T>::T2;
            Ty s_h;
            اگر (get_اشارہ_گر_attribute(q, s) == اشارہ_گر_access_attribute::device_only)
                detail::dpct_memcpy(q, (خالی *)&s_h, (ثابت خالی *)s, حجم(T), device_مقصد_host)
                    .wait();
            ورنہ
                s_h = *reinterpret_cast<ثابت Ty *>(s);
            واپس s_h;
        }

    } // namespace detail

    سانچہ <قسم_نام T>
    اندرونی خودکار get_قدر(ثابت T *s, sycl::queue &q)
    {
        واپس detail::get_قدر(s, q);
    }

    نامخلا detail
    {
    سانچہ <جماعت Ta, جماعت Tb, جماعت Tc, جماعت Ts>
    اندرونی خالی gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحیح m,
                          صحیح n, صحیح k, ثابت خالی * alpha, ثابت خالی * a, صحیح lda, ثابت خالی * b, صحیح ldb,
                          ثابت خالی * beta, خالی * c, صحیح ldc) {
        Ts   alpha_قدر_2 = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(alpha), q);
        Ts   beta_قدر  = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(beta), q);
        خودکار ڈیٹا_a      = get_حافظہ<ثابت Ta>(a);
        خودکار ڈیٹا_b      = get_حافظہ<ثابت Tb>(b);
        خودکار ڈیٹا_c      = get_حافظہ<Tc>(c);
        oneapi::mkl::blas::عمود_major::gemm(q, a_trans, b_trans, m, n, k, alpha_قدر_2, ڈیٹا_a,
                                               lda, ڈیٹا_b, ldb, beta_قدر, ڈیٹا_c, ldc);
    }

        سانچہ <قسم_نام VecT, جماعت ثنائی_Operation, جماعت = خالی>
        جماعت vectorized_ثنائی
        {
        عوامی:
            اندرونی VecT عامل()(VecT a, VecT b, ثابت ثنائی_Operation ثنائی_op)
            {
                VecT v4;
                برائے (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ثنائی_op(a[i], b[i]);
                }
                واپس v4;
            }
        };

        سانچہ <قسم_نام VecT, جماعت ثنائی_Operation>
        جماعت vectorized_ثنائی<
            VecT, ثنائی_Operation,
            std::void_t<std::invoke_result_t<ثنائی_Operation, VecT, VecT>>>
        {
        عوامی:
            اندرونی VecT عامل()(VecT a, VecT b, ثابت ثنائی_Operation ثنائی_op)
            {
                واپس ثنائی_op(a, b).سانچہ as<VecT>();
            }
        };

        سانچہ <جماعت Ta, جماعت Tb, جماعت Tc, جماعت Ts>
        اندرونی خالی gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    صحیح m, صحیح n, صحیح k, ثابت خالی * alpha, ثابت خالی ** a, صحیح lda, ثابت خالی ** b,
                                    صحیح ldb, ثابت خالی * beta, خالی ** c, صحیح ldc, صحیح batch_سائز,
                                    میٹرکس_info_t<اعشاری> * میٹرکس_info) {
            Ts alpha_قدر_2 = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(alpha), q);
            Ts beta_قدر = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(beta), q);

            میٹرکس_info->transpose_info[0] = a_trans;
            میٹرکس_info->transpose_info[1] = b_trans;
            میٹرکس_info->قدر_info[0] = alpha_قدر_2;
            میٹرکس_info->قدر_info[1] = beta_قدر;
            میٹرکس_info->سائز_info[0] = m;
            میٹرکس_info->سائز_info[1] = n;
            میٹرکس_info->سائز_info[2] = k;
            میٹرکس_info->ld_info[0] = lda;
            میٹرکس_info->ld_info[1] = ldb;
            میٹرکس_info->ld_info[2] = ldc;
            میٹرکس_info->groupsize_info = batch_سائز;

            sycl::event e = oneapi::mkl::blas::عمود_major::gemm_batch(
                q, میٹرکس_info->transpose_info, میٹرکس_info->transpose_info + 1,
                میٹرکس_info->سائز_info, میٹرکس_info->سائز_info + 1, میٹرکس_info->سائز_info + 2,
                reinterpret_cast<Ts *>(میٹرکس_info->قدر_info), reinterpret_cast<ثابت Ta **>(a), میٹرکس_info->ld_info,
                reinterpret_cast<ثابت Tb **>(b), میٹرکس_info->ld_info + 1,
                reinterpret_cast<Ts *>(میٹرکس_info->قدر_info + 1), reinterpret_cast<Tc **>(c),
                میٹرکس_info->ld_info + 2, 1, &(میٹرکس_info->groupsize_info));
        }

        سانچہ <جماعت Ta, جماعت Tb, جماعت Tc, جماعت Ts>
        اندرونی خالی gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    صحیح m, صحیح n, صحیح k, ثابت خالی * alpha, ثابت خالی * a, صحیح lda,
                                    طویل طویل صحیح stride_a, ثابت خالی * b, صحیح ldb, طویل طویل صحیح stride_b,
                                    ثابت خالی * beta, خالی * c, صحیح ldc, طویل طویل صحیح stride_c, صحیح batch_سائز) {
            Ts alpha_قدر_2 = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(alpha), q);
            Ts beta_قدر = dpct::get_قدر(reinterpret_cast<ثابت Ts *>(beta), q);
            خودکار ڈیٹا_a = get_حافظہ<ثابت Ta>(a);
            خودکار ڈیٹا_b = get_حافظہ<ثابت Tb>(b);
            خودکار ڈیٹا_c = get_حافظہ<Tc>(c);
            oneapi::mkl::blas::عمود_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_قدر_2,
                                                         ڈیٹا_a, lda, stride_a, ڈیٹا_b, ldb, stride_b, beta_قدر,
                                                         ڈیٹا_c, ldc, stride_c, batch_سائز);
        }

    } // namespace detail

    سانچہ <قسم_نام VecT, جماعت ثنائی_Operation>
    اندرونی غیرعلامتی vectorized_ثنائی(غیرعلامتی a, غیرعلامتی b,
                                      ثابت ثنائی_Operation ثنائی_op)
    {
        sycl::vec<غیرعلامتی, 1> v0{a}, v1{b};
        خودکار v2 = v0.as<VecT>();
        خودکار v3 = v1.as<VecT>();
        خودکار v4 =
            detail::vectorized_ثنائی<VecT, ثنائی_Operation>()(v2, v3, ثنائی_op);
        v0 = v4.سانچہ as<sycl::vec<غیرعلامتی, 1>>();
        واپس v0;
    }

    ساکن خالی async_dpct_memcpy(خالی *مقصد_ptr, ثابت خالی *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_قطار())
    {
        detail::dpct_memcpy(q, مقصد_ptr, from_ptr, size, direction);
    }

    ساکن اندرونی غیرعلامتی صحیح select_device(غیرعلامتی صحیح id)
    {
        dev_mgr::instance().select_device(id);
        واپس id;
    }

    سانچہ <قسم_نام T>
    T ترتیب_بدلو_sub_گروہ_by_xor(sycl::sub_گروہ g, T x, غیرعلامتی صحیح نقاب,
                               غیرعلامتی صحیح logical_sub_گروہ_سائز = 32)
    {
        غیرعلامتی صحیح id = g.get_local_خطی_id();
        غیرعلامتی صحیح آغاز_اشاریہ =
            id / logical_sub_گروہ_سائز * logical_sub_گروہ_سائز;
        غیرعلامتی صحیح ہدف_انحراف = (id % logical_sub_گروہ_سائز) ^ نقاب;
        واپس sycl::select_from_گروہ(g, x,
                                       ہدف_انحراف < logical_sub_گروہ_سائز
                                           ? آغاز_اشاریہ + ہدف_انحراف
                                           : id);
    }

    سانچہ <قسم_نام T1, قسم_نام T2>
    استعمال dot_حاصل_ضرب_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    سانچہ <قسم_نام T>
    sycl::vec<T, 4> نکالو_and_sign_or_zero_extend4(T val) {
      واپس sycl::vec<T, 1>(val)
          .سانچہ as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .سانچہ convert<T>();
    }

    سانچہ <قسم_نام T1, قسم_نام T2, قسم_نام T3>
    اندرونی خودکار dp4a(T1 a, T2 b, T3 c) {
      dot_حاصل_ضرب_acc_t<T1, T2> res = c;
      خودکار va = نکالو_and_sign_or_zero_extend4(a);
      خودکار vb = نکالو_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      واپس res;
    }

    ساخت sub_sat
    {
        سانچہ <قسم_نام T>
        خودکار عامل()(ثابت T x, ثابت T y) ثابت
        {
            واپس sycl::sub_sat(x, y);
        }
    };

    سانچہ <قسم_نام S, قسم_نام T>
    اندرونی T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        خودکار v2 = v0.سانچہ as<S>();
        خودکار v3 = v1.سانچہ as<S>();
        خودکار v4 = sycl::min(v2, v3);
        v0 = v4.سانچہ as<sycl::vec<T, 1>>();
        واپس v0;
    }

    اندرونی اعشاری قوت(ثابت اعشاری a, ثابت صحیح b) { واپس sycl::pown(a, b); }
    اندرونی دوہرا قوت(ثابت دوہرا a, ثابت صحیح b) { واپس sycl::pown(a, b); }
    اندرونی اعشاری قوت(ثابت اعشاری a, ثابت اعشاری b) { واپس sycl::قوت(a, b); }
    اندرونی دوہرا قوت(ثابت دوہرا a, ثابت دوہرا b) { واپس sycl::قوت(a, b); }
    سانچہ <قسم_نام T, قسم_نام U>
    اندرونی قسم_نام std::enable_if_t<std::is_floating_point_v<T>, T>
    قوت(ثابت T a, ثابت U b)
    {
        واپس sycl::قوت(a, static_cast<T>(b));
    }
    سانچہ <قسم_نام T, قسم_نام U>
    اندرونی قسم_نام std::enable_if_t<!std::is_floating_point_v<T>, دوہرا>
    قوت(ثابت T a, ثابت U b)
    {
        واپس sycl::قوت(static_cast<دوہرا>(a), static_cast<دوہرا>(b));
    }

    اندرونی دوہرا min(ثابت دوہرا a, ثابت اعشاری b)
    {
        واپس sycl::fmin(a, static_cast<دوہرا>(b));
    }
    اندرونی دوہرا min(ثابت اعشاری a, ثابت دوہرا b)
    {
        واپس sycl::fmin(static_cast<دوہرا>(a), b);
    }
    اندرونی اعشاری min(ثابت اعشاری a, ثابت اعشاری b) { واپس sycl::fmin(a, b); }
    اندرونی دوہرا min(ثابت دوہرا a, ثابت دوہرا b) { واپس sycl::fmin(a, b); }
    اندرونی std::uint32_t min(ثابت std::uint32_t a, ثابت std::int32_t b)
    {
        واپس sycl::min(a, static_cast<std::uint32_t>(b));
    }
    اندرونی std::uint32_t min(ثابت std::int32_t a, ثابت std::uint32_t b)
    {
        واپس sycl::min(static_cast<std::uint32_t>(a), b);
    }
    اندرونی std::int32_t min(ثابت std::int32_t a, ثابت std::int32_t b)
    {
        واپس sycl::min(a, b);
    }
    اندرونی std::uint32_t min(ثابت std::uint32_t a, ثابت std::uint32_t b)
    {
        واپس sycl::min(a, b);
    }
    اندرونی std::uint64_t min(ثابت std::uint64_t a, ثابت std::int64_t b)
    {
        واپس sycl::min(a, static_cast<std::uint64_t>(b));
    }
    اندرونی std::uint64_t min(ثابت std::int64_t a, ثابت std::uint64_t b)
    {
        واپس sycl::min(static_cast<std::uint64_t>(a), b);
    }
    اندرونی std::int64_t min(ثابت std::int64_t a, ثابت std::int64_t b)
    {
        واپس sycl::min(a, b);
    }
    اندرونی std::uint64_t min(ثابت std::uint64_t a, ثابت std::uint64_t b)
    {
        واپس sycl::min(a, b);
    }
    اندرونی std::uint64_t min(ثابت std::uint64_t a, ثابت std::int32_t b)
    {
        واپس sycl::min(a, static_cast<std::uint64_t>(b));
    }
    اندرونی std::uint64_t min(ثابت std::int32_t a, ثابت std::uint64_t b)
    {
        واپس sycl::min(static_cast<std::uint64_t>(a), b);
    }
    اندرونی std::uint64_t min(ثابت std::uint64_t a, ثابت std::uint32_t b)
    {
        واپس sycl::min(a, static_cast<std::uint64_t>(b));
    }
    اندرونی std::uint64_t min(ثابت std::uint32_t a, ثابت std::uint64_t b)
    {
        واپس sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    اندرونی دوہرا max(ثابت دوہرا a, ثابت اعشاری b)
    {
        واپس sycl::fmax(a, static_cast<دوہرا>(b));
    }
    اندرونی دوہرا max(ثابت اعشاری a, ثابت دوہرا b)
    {
        واپس sycl::fmax(static_cast<دوہرا>(a), b);
    }
    اندرونی اعشاری max(ثابت اعشاری a, ثابت اعشاری b) { واپس sycl::fmax(a, b); }
    اندرونی دوہرا max(ثابت دوہرا a, ثابت دوہرا b) { واپس sycl::fmax(a, b); }
    اندرونی std::uint32_t max(ثابت std::uint32_t a, ثابت std::int32_t b)
    {
        واپس sycl::max(a, static_cast<std::uint32_t>(b));
    }
    اندرونی std::uint32_t max(ثابت std::int32_t a, ثابت std::uint32_t b)
    {
        واپس sycl::max(static_cast<std::uint32_t>(a), b);
    }
    اندرونی std::int32_t max(ثابت std::int32_t a, ثابت std::int32_t b)
    {
        واپس sycl::max(a, b);
    }
    اندرونی std::uint32_t max(ثابت std::uint32_t a, ثابت std::uint32_t b)
    {
        واپس sycl::max(a, b);
    }
    اندرونی std::uint64_t max(ثابت std::uint64_t a, ثابت std::int64_t b)
    {
        واپس sycl::max(a, static_cast<std::uint64_t>(b));
    }
    اندرونی std::uint64_t max(ثابت std::int64_t a, ثابت std::uint64_t b)
    {
        واپس sycl::max(static_cast<std::uint64_t>(a), b);
    }
    اندرونی std::int64_t max(ثابت std::int64_t a, ثابت std::int64_t b)
    {
        واپس sycl::max(a, b);
    }
    اندرونی std::uint64_t max(ثابت std::uint64_t a, ثابت std::uint64_t b)
    {
        واپس sycl::max(a, b);
    }
    اندرونی std::uint64_t max(ثابت std::uint64_t a, ثابت std::int32_t b)
    {
        واپس sycl::max(a, static_cast<std::uint64_t>(b));
    }
    اندرونی std::uint64_t max(ثابت std::int32_t a, ثابت std::uint64_t b)
    {
        واپس sycl::max(static_cast<std::uint64_t>(a), b);
    }
    اندرونی std::uint64_t max(ثابت std::uint64_t a, ثابت std::uint32_t b)
    {
        واپس sycl::max(a, static_cast<std::uint64_t>(b));
    }
    اندرونی std::uint64_t max(ثابت std::uint32_t a, ثابت std::uint64_t b)
    {
        واپس sycl::max(static_cast<std::uint64_t>(a), b);
    }

    اندرونی خالی
    has_capability_or_fail(ثابت sycl::device &dev,
                           ثابت std::initializer_list<sycl::aspect> &props)
    {
        برائے (ثابت خودکار &it : props)
        {
            اگر (dev.has(it))
                جاری;
            انتخاب (it)
            {
            حالت sycl::aspect::fp64:
                پھینکو std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                توڑو;
            حالت sycl::aspect::fp16:
                پھینکو std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                توڑو;
            طےشدہ:
#تعریف_کرو __SYCL_ASPECT(ASPECT, ID) \
    حالت sycl::aspect::ASPECT:    \
        واپس #ASPECT;
#تعریف_کرو __SYCL_ASPECT_DEPRECATED(ASPECT, ID, پیغام_3) __SYCL_ASPECT(ASPECT, ID)
#تعریف_کرو __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, پیغام_3)
                خودکار getAspectNameStr = [](sycl::aspect Aspect_عدد) -> std::string
                {
                    انتخاب (Aspect_عدد)
                    {
#شامل <sycl/info/aspects.def>
#شامل <sycl/info/aspects_deprecated.def>
                    طےشدہ:
                        واپس "unknown aspect";
                    }
                };
#تعریف_منسوخ __SYCL_ASPECT_DEPRECATED_ALIAS
#تعریف_منسوخ __SYCL_ASPECT_DEPRECATED
#تعریف_منسوخ __SYCL_ASPECT
                پھینکو std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            توڑو;
        }
    }

    ساکن اندرونی غیرعلامتی صحیح get_current_device_id()
    {
        واپس dev_mgr::instance().current_device_id();
    }

    ساکن اندرونی device_ext &get_current_device()
    {
        واپس dev_mgr::instance().current_device();
    }

    ساکن اندرونی device_ext &get_device(غیرعلامتی صحیح id)
    {
        واپس dev_mgr::instance().get_device(id);
    }

    ساکن اندرونی sycl::queue &get_اندر_ترتیب_قطار()
    {
        واپس dev_mgr::instance().current_device().اندر_ترتیب_قطار();
    }

    ساکن sycl::event
    dpct_memcpy(sycl::queue &q, خالی *مقصد_ptr, ثابت خالی *from_ptr, size_t size,
                memcpy_direction direction,
                ثابت std::vector<sycl::event> &dep_events = {})
    {
        اگر (!size)
            واپس sycl::event{};
        واپس q.حافظہ_نقل(مقصد_ptr, from_ptr, size, dep_events);
        DPCT_غیر_مستعمل(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    ساکن اندرونی size_t get_copy_range(sycl::range<3> size, size_t قاش_2,
                                        size_t pitch)
    {
        واپس قاش_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    ساکن اندرونی size_t get_انحراف(sycl::id<3> id, size_t قاش_2,
                                    size_t pitch)
    {
        واپس قاش_2 * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    ساکن اندرونی std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, خالی *مقصد_ptr, ثابت خالی *from_ptr,
                sycl::range<3> مقصد_range, sycl::range<3> from_range,
                sycl::id<3> مقصد_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                ثابت std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        جماعت host_بفر
        {
            خالی *_buf;
            size_t سائز_2;
            sycl::queue &_q;
            ثابت std::vector<sycl::event> &_deps; // free operation depends

        عوامی:
            host_بفر(size_t size, sycl::queue &q,
                        ثابت std::vector<sycl::event> &deps)
                : _buf(std::حافظہ_دو(size)), سائز_2(size), _q(q), _deps(deps) {}
            خالی *get_ptr() ثابت { واپس _buf; }
            size_t get_سائز() ثابت { واپس سائز_2; }
            ~host_بفر()
            {
                اگر (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::آزاد(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t مقصد_قاش = مقصد_range.get(1) * مقصد_range.get(0),
               from_قاش = from_range.get(1) * from_range.get(0);
        غیرعلامتی حرف *مقصد_surface =
            (غیرعلامتی حرف *)مقصد_ptr + get_انحراف(مقصد_id, مقصد_قاش, مقصد_range.get(0));
        ثابت غیرعلامتی حرف *from_surface =
            (ثابت غیرعلامتی حرف *)from_ptr +
            get_انحراف(from_id, from_قاش, from_range.get(0));

        اگر (مقصد_قاش == from_قاش && مقصد_قاش == size.get(1) * size.get(0))
        {
            واپس {dpct_memcpy(q, مقصد_surface, from_surface, مقصد_قاش * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, مقصد_ptr, from_ptr, direction);
        size_t سائز_قاش = size.get(1) * size.get(0);
        انتخاب (direction)
        {
        حالت host_مقصد_host:
            برائے (size_t z = 0; z < size.get(2); ++z)
            {
                غیرعلامتی حرف *مقصد_ptr = مقصد_surface;
                ثابت غیرعلامتی حرف *from_ptr = from_surface;
                اگر (مقصد_range.get(0) == from_range.get(0) &&
                    مقصد_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, مقصد_ptr, from_ptr, سائز_قاش,
                                                     direction, dep_events));
                }
                ورنہ
                {
                    برائے (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, مقصد_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        مقصد_ptr += مقصد_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                مقصد_surface += مقصد_قاش;
                from_surface += from_قاش;
            }
            توڑو;
        حالت host_مقصد_device:
        {
            host_بفر buf(get_copy_range(size, مقصد_قاش, مقصد_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            اگر (مقصد_قاش == سائز_قاش)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, مقصد_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_مقصد_host, dep_events);
            }
            ورنہ
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, مقصد_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_مقصد_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), مقصد_surface, buf.get_سائز(),
                                    device_مقصد_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, مقصد_surface, buf.get_ptr(),
                                             buf.get_سائز(), host_مقصد_device,
                                             host_events));
            توڑو;
        }
        حالت device_مقصد_host:
        {
            host_بفر buf(get_copy_range(size, from_قاش, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, مقصد_surface, buf.get_ptr(), مقصد_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_مقصد_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_سائز(),
                                                     device_مقصد_host, dep_events)});
            توڑو;
        }
        حالت device_مقصد_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<جماعت dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                مقصد_surface[get_انحراف(id, مقصد_قاش, مقصد_range.get(0))] =
                    from_surface[get_انحراف(id, from_قاش, from_range.get(0))];
            }); }));
        توڑو;
        طےشدہ:
            پھینکو std::runtime_error("dpct_memcpy: invalid direction value");
        }
        واپس event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    ساکن اندرونی std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_ڈیٹا مقصد, sycl::id<3> مقصد_id,
                pitched_ڈیٹا from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        واپس dpct_memcpy(q, مقصد.get_ڈیٹا_ptr(), from.get_ڈیٹا_ptr(),
                           sycl::range<3>(مقصد.get_pitch(), مقصد.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), مقصد_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    ساکن اندرونی std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, خالی *مقصد_ptr, ثابت خالی *from_ptr,
                size_t مقصد_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        واپس dpct_memcpy(q, مقصد_ptr, from_ptr, sycl::range<3>(مقصد_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    اندرونی خالی gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحیح m, صحیح n,
                     صحیح k, ثابت خالی * alpha, ثابت خالی * a, library_ڈیٹا_t a_type, صحیح lda, ثابت خالی * b,
                     library_ڈیٹا_t b_type, صحیح ldb, ثابت خالی * beta, خالی * c, library_ڈیٹا_t c_type, صحیح ldc,
                     library_ڈیٹا_t scaling_type) {
        اگر (scaling_type == library_ڈیٹا_t::real_float &&
            c_type == library_ڈیٹا_t::complex_float)
        {
            scaling_type = library_ڈیٹا_t::complex_float;
        }
        ورنہ اگر (scaling_type == library_ڈیٹا_t::real_double &&
                 c_type == library_ڈیٹا_t::complex_double)
        {
            scaling_type = library_ڈیٹا_t::complex_double;
        }

        std::uint64_t کلید =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        انتخاب (کلید)
        {
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_impl<اعشاری, اعشاری, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_double, library_ڈیٹا_t::real_double,
            library_ڈیٹا_t::real_double, library_ڈیٹا_t::real_double):
        {
            detail::gemm_impl<دوہرا, دوہرا, دوہرا, دوہرا>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::complex_float, library_ڈیٹا_t::complex_float,
            library_ڈیٹا_t::complex_float, library_ڈیٹا_t::complex_float):
        {
            detail::gemm_impl<std::complex<اعشاری>, std::complex<اعشاری>,
                              std::complex<اعشاری>, std::complex<اعشاری>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::complex_double, library_ڈیٹا_t::complex_double,
            library_ڈیٹا_t::complex_double, library_ڈیٹا_t::complex_double):
        {
            detail::gemm_impl<std::complex<دوہرا>, std::complex<دوہرا>,
                              std::complex<دوہرا>, std::complex<دوہرا>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            توڑو;
        }
#اگر_تعریف_شدہ __INTEL_MKL__
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_bfloat16,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_float):
        {
            اعشاری alpha_قدر_2 =
                dpct::get_قدر(reinterpret_cast<ثابت اعشاری *>(alpha), q);
            اعشاری beta_قدر =
                dpct::get_قدر(reinterpret_cast<ثابت اعشاری *>(beta), q);
            sycl::half alpha_half(alpha_قدر_2);
            sycl::half beta_half(beta_قدر);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_int8, library_ڈیٹا_t::real_int8,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_bfloat16,
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_int8, library_ڈیٹا_t::real_int8,
            library_ڈیٹا_t::real_int32, library_ڈیٹا_t::real_int32):
        {
            اعشاری alpha_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(alpha), q);
            اعشاری beta_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, اعشاری>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            توڑو;
        }
#اگر_ختم // __INTEL_MKL__
        طےشدہ:
            پھینکو std::runtime_error("the combination of data type is unsupported");
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
    اندرونی خالی gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحیح m,
                           صحیح n, صحیح k, ثابت خالی * alpha, ثابت خالی * a[], library_ڈیٹا_t a_type, صحیح lda,
                           ثابت خالی * b[], library_ڈیٹا_t b_type, صحیح ldb, ثابت خالی * beta, خالی * c[],
                           library_ڈیٹا_t c_type, صحیح ldc, صحیح batch_سائز, library_ڈیٹا_t scaling_type,
                           میٹرکس_info_t<اعشاری> * میٹرکس_info) {
        std::uint64_t کلید =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        انتخاب (کلید)
        {
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<اعشاری, اعشاری, اعشاری, اعشاری>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_double, library_ڈیٹا_t::real_double,
            library_ڈیٹا_t::real_double, library_ڈیٹا_t::real_double):
        {
            detail::gemm_batch_impl<دوہرا, دوہرا, دوہرا, دوہرا>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
#اگر_تعریف_شدہ __INTEL_MKL__
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_bfloat16,
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_bfloat16,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
#اگر_ختم
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_int8, library_ڈیٹا_t::real_int8,
            library_ڈیٹا_t::real_int32, library_ڈیٹا_t::real_int32):
        {
            اعشاری alpha_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(alpha), q);
            اعشاری beta_float =
                dpct::get_قدر(reinterpret_cast<ثابت std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, اعشاری>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_سائز,
                میٹرکس_info);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_int8, library_ڈیٹا_t::real_int8,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_float):
        {
            اعشاری alpha_قدر_2 =
                dpct::get_قدر(reinterpret_cast<ثابت اعشاری *>(alpha), q);
            اعشاری beta_قدر =
                dpct::get_قدر(reinterpret_cast<ثابت اعشاری *>(beta), q);
            sycl::half alpha_half(alpha_قدر_2);
            sycl::half beta_half(beta_قدر);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_سائز, میٹرکس_info);
            توڑو;
        }
        طےشدہ:
            پھینکو std::runtime_error("the combination of data type is unsupported");
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
    اندرونی خالی gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, صحیح m,
                           صحیح n, صحیح k, ثابت خالی * alpha, ثابت خالی * a, library_ڈیٹا_t a_type, صحیح lda,
                           طویل طویل صحیح stride_a, ثابت خالی * b, library_ڈیٹا_t b_type, صحیح ldb,
                           طویل طویل صحیح stride_b, ثابت خالی * beta, خالی * c, library_ڈیٹا_t c_type, صحیح ldc,
                           طویل طویل صحیح stride_c, صحیح batch_سائز, library_ڈیٹا_t scaling_type) {
        اگر (scaling_type == library_ڈیٹا_t::real_float &&
            c_type == library_ڈیٹا_t::complex_float)
        {
            scaling_type = library_ڈیٹا_t::complex_float;
        }
        ورنہ اگر (scaling_type == library_ڈیٹا_t::real_double &&
                 c_type == library_ڈیٹا_t::complex_double)
        {
            scaling_type = library_ڈیٹا_t::complex_double;
        }

        std::uint64_t کلید =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        انتخاب (کلید)
        {
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<اعشاری, اعشاری, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_double, library_ڈیٹا_t::real_double,
            library_ڈیٹا_t::real_double, library_ڈیٹا_t::real_double):
        {
            detail::gemm_batch_impl<دوہرا, دوہرا, دوہرا, دوہرا>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::complex_float, library_ڈیٹا_t::complex_float,
            library_ڈیٹا_t::complex_float, library_ڈیٹا_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<اعشاری>, std::complex<اعشاری>,
                                    std::complex<اعشاری>, std::complex<اعشاری>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::complex_double, library_ڈیٹا_t::complex_double,
            library_ڈیٹا_t::complex_double, library_ڈیٹا_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<دوہرا>, std::complex<دوہرا>,
                                    std::complex<دوہرا>, std::complex<دوہرا>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
#اگر_تعریف_شدہ __INTEL_MKL__
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_bfloat16,
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_bfloat16, library_ڈیٹا_t::real_bfloat16,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_سائز);
            توڑو;
        }
#اگر_ختم
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_int8, library_ڈیٹا_t::real_int8,
            library_ڈیٹا_t::real_int32, library_ڈیٹا_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_int8, library_ڈیٹا_t::real_int8,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_float, library_ڈیٹا_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, اعشاری, اعشاری>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        حالت detail::get_type_combination_id(
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_half,
            library_ڈیٹا_t::real_half, library_ڈیٹا_t::real_float):
        {
            اعشاری alpha_قدر_2 =
                dpct::get_قدر(reinterpret_cast<ثابت اعشاری *>(alpha), q);
            اعشاری beta_قدر =
                dpct::get_قدر(reinterpret_cast<ثابت اعشاری *>(beta), q);
            sycl::half alpha_half(alpha_قدر_2);
            sycl::half beta_half(beta_قدر);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_سائز);
            توڑو;
        }
        طےشدہ:
            پھینکو std::runtime_error("the combination of data type is unsupported");
        }
    }

    ساکن اندرونی خالی
    async_dpct_memcpy(خالی *مقصد_ptr, size_t مقصد_pitch, ثابت خالی *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_قطار())
    {
        detail::dpct_memcpy(q, مقصد_ptr, from_ptr, مقصد_pitch, from_pitch, x, y,
                            direction);
    }

    استعمال err0 = detail::generic_error_type<ساخت err0_لیبل, صحیح>;
    استعمال err1 = detail::generic_error_type<ساخت err1_لیبل, صحیح>;

    ساکن اندرونی خالی dpct_free(خالی *ptr, sycl::queue &q = get_default_قطار()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    سانچہ <جماعت T, حافظہ_region حافظہ_2, size_t Dimension> جماعت accessor;
    سانچہ <جماعت T, حافظہ_region حافظہ_2> جماعت accessor<T, حافظہ_2, 3> {
    عوامی:
        استعمال حافظہ_t = detail::حافظہ_traits<حافظہ_2, T>;
        استعمال element_t = قسم_نام حافظہ_t::element_t;
        استعمال اشارہ_گر_t = قسم_نام حافظہ_t::اشارہ_گر_t;
        استعمال accessor_t = قسم_نام حافظہ_t::سانچہ accessor_t<3>;
        accessor(اشارہ_گر_t ڈیٹا, ثابت sycl::range<3> &اندر_range)
            : ڈیٹا_2(ڈیٹا), _range(اندر_range) {}
        سانچہ <حافظہ_region M = حافظہ_2>
        accessor(قسم_نام std::enable_if<M != local, ثابت accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ثابت accessor_t &acc, ثابت sycl::range<3> &اندر_range)
            : accessor(acc.get_اشارہ_گر(), اندر_range) {}
        accessor<T, حافظہ_2, 2> عامل[](size_t اشاریہ) ثابت {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            واپس accessor<T, حافظہ_2, 2>(ڈیٹا_2 + اشاریہ * sub.size(), sub);
        }

        اشارہ_گر_t get_ptr() ثابت { واپس ڈیٹا_2; }

    نجی:
        اشارہ_گر_t ڈیٹا_2;
        sycl::range<3> _range;
    };
    سانچہ <جماعت T, حافظہ_region حافظہ_2> جماعت accessor<T, حافظہ_2, 2> {
    عوامی:
        استعمال حافظہ_t = detail::حافظہ_traits<حافظہ_2, T>;
        استعمال element_t = قسم_نام حافظہ_t::element_t;
        استعمال اشارہ_گر_t = قسم_نام حافظہ_t::اشارہ_گر_t;
        استعمال accessor_t = قسم_نام حافظہ_t::سانچہ accessor_t<2>;
        accessor(اشارہ_گر_t ڈیٹا, ثابت sycl::range<2> &اندر_range)
            : ڈیٹا_2(ڈیٹا), _range(اندر_range) {}
        سانچہ <حافظہ_region M = حافظہ_2>
        accessor(قسم_نام std::enable_if<M != local, ثابت accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(ثابت accessor_t &acc, ثابت sycl::range<2> &اندر_range)
            : accessor(acc.get_اشارہ_گر(), اندر_range) {}

        اشارہ_گر_t عامل[](size_t اشاریہ) ثابت {
            واپس ڈیٹا_2 + _range.get(1) * اشاریہ;
        }

        اشارہ_گر_t get_ptr() ثابت { واپس ڈیٹا_2; }

    نجی:
        اشارہ_گر_t ڈیٹا_2;
        sycl::range<2> _range;
    };

    نامخلا detail {
        /// Device variable with address space of shared, global or constant.
        سانچہ <جماعت T, حافظہ_region حافظہ_2, size_t Dimension> جماعت device_حافظہ {
        عوامی:
            استعمال accessor_t =
                قسم_نام detail::حافظہ_traits<حافظہ_2,
                                            T>::سانچہ accessor_t<Dimension>;
            استعمال قدر_t = قسم_نام detail::حافظہ_traits<حافظہ_2, T>::قدر_t;
            استعمال dpct_accessor_t = dpct::accessor<T, حافظہ_2, Dimension>;

            device_حافظہ() : device_حافظہ(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_حافظہ(ثابت sycl::range<Dimension> &اندر_range,
                        std::initializer_list<قدر_t> &&init_list)
                : device_حافظہ(اندر_range) {
                أكد(init_list.size() <= اندر_range.size());
                _host_ptr = (قدر_t *)std::حافظہ_دو(سائز_2);
                std::memset(_host_ptr, 0, سائز_2);
                std::حافظہ_نقل(_host_ptr, init_list.begin(), init_list.size() * حجم(T));
            }

            /// Constructor of 2-D array with initializer list
            سانچہ <size_t D = Dimension>
            device_حافظہ(
                ثابت قسم_نام std::enable_if<D == 2, sycl::range<2>>::type &اندر_range,
                std::initializer_list<std::initializer_list<قدر_t>> &&init_list)
                : device_حافظہ(اندر_range) {
                أكد(init_list.size() <= اندر_range[0]);
                _host_ptr = (قدر_t *)std::حافظہ_دو(سائز_2);
                std::memset(_host_ptr, 0, سائز_2);
                خودکار tmp_ڈیٹا = _host_ptr;
                برائے (خودکار sub_list : init_list) {
                    أكد(sub_list.size() <= اندر_range[1]);
                    std::حافظہ_نقل(tmp_ڈیٹا, sub_list.begin(),
                                sub_list.size() * حجم(T));
                    tmp_ڈیٹا += اندر_range[1];
                }
            }

            /// Constructor with range
            device_حافظہ(ثابت sycl::range<Dimension> &range_اندر)
                : سائز_2(range_اندر.size() * حجم(T)), _range(range_اندر),
                _reference(جھوٹ), _host_ptr(خالی_اشاریہ), _device_ptr(خالی_اشاریہ) {
                static_assert(
                    (حافظہ_2 == global) || (حافظہ_2 == constant) || (حافظہ_2 == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            سانچہ <جماعت... دلائل>
            device_حافظہ(دلائل... Arguments)
                : device_حافظہ(sycl::range<Dimension>(Arguments...)) {}

            ~device_حافظہ() {
                اگر (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                اگر (_host_ptr)
                    std::آزاد(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            خالی init() { init(dpct::get_default_قطار()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            خالی init(sycl::queue &q) {
                اگر (_device_ptr)
                    واپس;
                اگر (!سائز_2)
                    واپس;
                allocate_device(q);
                اگر (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, سائز_2,
                                        host_مقصد_device);
            }

            /// The variable is assigned to a device pointer.
            خالی تفویض(قدر_t *src, size_t size) {
                یہ->~device_حافظہ();
                نیا (یہ) device_حافظہ(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            قدر_t *get_ptr() { واپس get_ptr(get_default_قطار()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            قدر_t *get_ptr(sycl::queue &q) {
                init(q);
                واپس _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_سائز() { واپس سائز_2; }

            سانچہ <size_t D = Dimension>
            قسم_نام std::enable_if<D == 1, T>::type &عامل[](size_t اشاریہ) {
                init();
                واپس _device_ptr[اشاریہ];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            سانچہ <size_t D = Dimension>
            قسم_نام std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                واپس dpct_accessor_t((T *)_device_ptr, _range);
            }

        نجی:
            device_حافظہ(قدر_t *حافظہ_ptr, size_t size)
                : سائز_2(size), _range(size / حجم(T)), _reference(سچ),
                _device_ptr(حافظہ_ptr) {}

            خالی allocate_device(sycl::queue &q) {
        #اگر_غیر_تعریف_شدہ DPCT_USM_LEVEL_NONE
                اگر (حافظہ_2 == shared) {
                    _device_ptr = (قدر_t *)sycl::malloc_shared(سائز_2, q.get_device(),
                                                                q.get_context());
                    واپس;
                }
        #اگر_تعریف_شدہ SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                اگر (حافظہ_2 == constant) {
                    _device_ptr = (قدر_t *)sycl::malloc_device(
                        سائز_2, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    واپس;
                }
        #اگر_ختم
        #اگر_ختم
                _device_ptr = (قدر_t *)detail::dpct_malloc(سائز_2, q);
            }

            size_t سائز_2;
            sycl::range<Dimension> _range;
            منطقی _reference;
            قدر_t *_host_ptr;
            قدر_t *_device_ptr;
        };
        سانچہ <جماعت T, حافظہ_region حافظہ_2>
        جماعت device_حافظہ<T, حافظہ_2, 0> : عوامی device_حافظہ<T, حافظہ_2, 1> {
        عوامی:
            استعمال بنیاد = device_حافظہ<T, حافظہ_2, 1>;
            استعمال قدر_t = قسم_نام بنیاد::قدر_t;
            استعمال accessor_t =
                قسم_نام detail::حافظہ_traits<حافظہ_2, T>::سانچہ accessor_t<0>;

            /// Constructor with initial value.
            device_حافظہ(ثابت قدر_t &val) : بنیاد(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_حافظہ() : بنیاد(1) {}
        };
        } // namespace detail

    سانچہ <جماعت T, size_t Dimension>
    استعمال global_حافظہ = detail::device_حافظہ<T, global, Dimension>;
    سانچہ <جماعت T, size_t Dimension>
    استعمال constant_حافظہ = detail::device_حافظہ<T, constant, Dimension>;
    سانچہ <جماعت T, size_t Dimension>
    استعمال shared_حافظہ = detail::device_حافظہ<T, shared, Dimension>;


    سانچہ <قسم_نام T,
            sycl::access::پتہ_space پتہ_Space =
                sycl::access::پتہ_space::global_space,
            sycl::memory_order حافظہ_ترتیب = sycl::memory_order::relaxed,
            sycl::حافظہ_scope حافظہ_Scope = sycl::حافظہ_scope::device>
    اندرونی T atomic_fetch_add(T *addr, T عملیہ) {
    خودکار atm =
        sycl::atomic_ref<T, حافظہ_ترتیب, حافظہ_Scope, پتہ_Space>(addr[0]);
    واپس atm.fetch_add(عملیہ);
    }

    سانچہ <sycl::access::پتہ_space پتہ_Space =
                sycl::access::پتہ_space::global_space,
            sycl::memory_order حافظہ_ترتیب = sycl::memory_order::relaxed,
            sycl::حافظہ_scope حافظہ_Scope = sycl::حافظہ_scope::device,
            قسم_نام T1, قسم_نام T2>
    اندرونی T1 atomic_fetch_add(T1 *addr, T2 عملیہ) {
    خودکار atm =
        sycl::atomic_ref<T1, حافظہ_ترتیب, حافظہ_Scope, پتہ_Space>(addr[0]);
    واپس atm.fetch_add(عملیہ);
    }

    سانچہ <قسم_نام T, sycl::access::پتہ_space پتہ_Space =
                            sycl::access::پتہ_space::global_space>
    اندرونی T atomic_fetch_add(T *addr, T عملیہ,
                            sycl::memory_order حافظہ_ترتیب) {
    انتخاب (حافظہ_ترتیب) {
        حالت sycl::memory_order::relaxed:
            واپس atomic_fetch_add<T, پتہ_Space, sycl::memory_order::relaxed,
                                    sycl::حافظہ_scope::device>(addr, عملیہ);
        حالت sycl::memory_order::acq_rel:
            واپس atomic_fetch_add<T, پتہ_Space, sycl::memory_order::acq_rel,
                                    sycl::حافظہ_scope::device>(addr, عملیہ);
        حالت sycl::memory_order::seq_cst:
            واپس atomic_fetch_add<T, پتہ_Space, sycl::memory_order::seq_cst,
                                    sycl::حافظہ_scope::device>(addr, عملیہ);
        طےشدہ:
            أكد(جھوٹ && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    سانچہ <sycl::access::پتہ_space پتہ_Space =
                sycl::access::پتہ_space::global_space,
            قسم_نام T1, قسم_نام T2>
    اندرونی T1 atomic_fetch_add(T1 *addr, T2 عملیہ,
                            sycl::memory_order حافظہ_ترتیب) {
    atomic_fetch_add<T1, پتہ_Space>(addr, عملیہ, حافظہ_ترتیب);
    }

    اندرونی غیرعلامتی صحیح byte_level_ترتیب_بدلو(
        غیرعلامتی صحیح a, غیرعلامتی صحیح b, غیرعلامتی صحیح s) {
      غیرعلامتی صحیح ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      واپس ret;
    }

    اندرونی uint32_t byte_level_ترتیب_بدلو_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, صحیح mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      اگر (mode >= 1 && mode <= 6) {
        واپس byte_level_ترتیب_بدلو(low32, high32, lookup[mode - 1][sel & 0x3]);
      } ورنہ اگر (!mode) {
        واپس byte_level_ترتیب_بدلو(low32, high32, sel);
      }
      واپس 0;
    }

    سانچہ <صحیح n_nondefault_params, صحیح n_default_params, قسم_نام T>
    جماعت دلائل_selector;

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
    سانچہ <صحیح n_nondefault_params, صحیح n_default_params, قسم_نام R,
              قسم_نام... Ts>
    جماعت دلائل_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      نجی:
        خالی **kernel_params;
        حرف *دلائل_بفر;

        سانچہ <صحیح i> ساکن constexpr صحیح account_for_default_params() {
            constexpr صحیح n_کل_params = حجم...(Ts);
            اگر constexpr (i >= n_nondefault_params) {
                واپس n_کل_params - n_default_params +
                       (i - n_nondefault_params);
            } ورنہ {
                واپس i;
            }
        }

      عوامی:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        سانچہ <صحیح i>
        استعمال arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        ساکن constexpr صحیح params_عدد = حجم...(Ts);

      نجی:
        سانچہ <صحیح i> ساکن constexpr صحیح get_انحراف() {
            اگر constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                واپس 0;
            } ورنہ {
                constexpr صحیح پچھلا_off = get_انحراف<i - 1>();
                constexpr صحیح پچھلا_past_انتہا =
                    پچھلا_off + حجم(arg_type<i - 1>);
                استعمال T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                اگر constexpr (پچھلا_past_انتہا % alignof(T) == 0) {
                    واپس پچھلا_past_انتہا;
                }
                // otherwise bump prev_past_end to match alignment
                ورنہ {
                    واپس پچھلا_past_انتہا +
                           (alignof(T) - (پچھلا_past_انتہا % alignof(T)));
                }
            }
        }

        ساکن حرف *get_دلائل_بفر(خالی **extra) {
            اگر (!extra)
                واپس خالی_اشاریہ;
            برائے (; (std::size_t)*extra != 0; ++extra) {
                اگر ((std::size_t)*extra == 1) {
                    واپس static_cast<حرف *>(*(extra + 1));
                }
            }
            واپس خالی_اشاریہ;
        }

      عوامی:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        دلائل_selector(خالی **kernel_params, خالی **extra)
            : kernel_params(kernel_params),
              دلائل_بفر(get_دلائل_بفر(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        سانچہ <صحیح i> arg_type<i> &get() {
            اگر (kernel_params) {
                واپس *static_cast<arg_type<i> *>(kernel_params[i]);
            } ورنہ {
                واپس *reinterpret_cast<arg_type<i> *>(دلائل_بفر +
                                                        get_انحراف<i>());
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
    جماعت kernel_launcher {
        سانچہ <قسم_نام FuncT, قسم_نام ArgSelector, std::size_t... اشاریہ_2>
        ساکن خالی launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<اشاریہ_2...>) {
            func(selector.سانچہ get<اشاریہ_2>()...);
        }
        ساکن خالی سیٹ_execution_config(dim3 گروہ_range, dim3 local_range,
                                         غیرعلامتی صحیح local_mem_سائز_2,
                                         قطار_ptr que) {
            اگر (que) {
                _que = que;
            } ورنہ {
                _que = &get_default_قطار();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(گروہ_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_سائز = local_mem_سائز_2;


        }
        ساکن اندرونی std::mutex kernel_function_ptr_map_mutex;

      عوامی:
        /// Variables for storing execution configuration.
        ساکن اندرونی thread_local sycl::queue *_que = خالی_اشاریہ;
        ساکن اندرونی thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        ساکن اندرونی thread_local غیرعلامتی صحیح local_mem_سائز = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        ساکن اندرونی std::map<
            ثابت خالی *,
            std::function<خالی(dim3, dim3, خالی **, غیرعلامتی صحیح, قطار_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        ساکن خالی رجسٹر_kernel_ptr(
            ثابت خالی *func,
            std::function<خالی(dim3, dim3, خالی **, غیرعلامتی صحیح, قطار_ptr)>
                launcher) {
            std::lock_guard<std::mutex> تالا(kernel_function_ptr_map_mutex);
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
        سانچہ <قسم_نام FuncT, قسم_نام... دلائل_T>
        ساکن std::enable_if_t<std::is_invocable_v<FuncT *, دلائل_T...>, خالی>
        launch(FuncT *func, dim3 گروہ_range, dim3 local_range,
               غیرعلامتی صحیح local_mem_سائز_2, قطار_ptr que, دلائل_T... دلائل_2) {
            سیٹ_execution_config(گروہ_range, local_range, local_mem_سائز_2, que);
            func(دلائل_2...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        ساکن خالی launch(ثابت خالی *func, dim3 گروہ_range, dim3 local_range,
                           خالی **دلائل_2, غیرعلامتی صحیح local_mem_سائز_2,
                           قطار_ptr que) {
            std::lock_guard<std::mutex> تالا(kernel_function_ptr_map_mutex);
            خودکار Iter = kernel_function_ptr_map.find(func);
            اگر (Iter == kernel_function_ptr_map.end()) {
                پھینکو std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(گروہ_range, local_range, دلائل_2, local_mem_سائز_2, que);
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
        سانچہ <قسم_نام FuncT>
        ساکن std::enable_if_t<std::is_function_v<FuncT>, خالی>
        launch(FuncT *func, dim3 گروہ_range, dim3 local_range, خالی **دلائل_2,
               غیرعلامتی صحیح local_mem_سائز_2, قطار_ptr que) {
            constexpr size_t p_عدد = دلائل_selector<0, 0, FuncT>::params_عدد;
            سیٹ_execution_config(گروہ_range, local_range, local_mem_سائز_2, que);
            دلائل_selector<p_عدد, p_عدد, FuncT> selector(دلائل_2, خالی_اشاریہ);
            launch_helper(func, selector, std::make_index_sequence<p_عدد>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    سانچہ <قسم_نام T>
    T select_from_sub_گروہ(
        sycl::sub_گروہ g,
        T x,
        صحیح remote_local_id,
        صحیح logical_sub_گروہ_سائز = 32) {
      غیرعلامتی صحیح آغاز_اشاریہ = g.get_local_خطی_id() /
                                 logical_sub_گروہ_سائز *
                                 logical_sub_گروہ_سائز;
      واپس sycl::select_from_گروہ(
          g, x, آغاز_اشاریہ + remote_local_id % logical_sub_گروہ_سائز);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    سانچہ <قسم_نام T>
    خالی ldmatrix(uintptr_t addr, T* m, منطقی trans = جھوٹ, غیرعلامتی mat = 0) {
      خودکار sg = sycl::ext::oneapi::this_work_شے::get_sub_گروہ();
      صحیح lane = sg.get_local_خطی_id();

      صحیح lane_group8_صف = lane / 8;
      صحیح lane_group8_ستون = lane % 8;

      اگر (!trans) {
        // calculate the source lane
        صحیح src_lane = 2 * lane_group8_صف;
        اگر (lane_group8_ستون >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        خودکار recv_addr_uintp =
            dpct::select_from_sub_گروہ(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        خودکار recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_ستون % 4];
      } ورنہ {
        // calculate the source lane
        صحیح src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        خودکار recv_addr_uintp_1 =
            dpct::select_from_sub_گروہ(sg, addr, mat * 8 + src_lane);
        خودکار recv_addr_uintp_2 =
            dpct::select_from_sub_گروہ(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        خودکار recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        خودکار recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        صحیح اشاریہ = lane / 4;
        sycl::half val0 = recv_addr_1[اشاریہ];
        sycl::half val1 = recv_addr_2[اشاریہ];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    سانچہ <قسم_نام T>
    خالی ldmatrix(uintptr_t addr, T* m1, T* m2, منطقی trans = جھوٹ) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    سانچہ <قسم_نام T>
    خالی ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, منطقی trans = جھوٹ) {
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
    سانچہ <قسم_نام T>
    ساخت MMAType {
      استعمال PackType = uint32_t;
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
    سانچہ <صحیح M, صحیح N, صحیح K, قسم_نام ABType, قسم_نام CDType>
    خالی mma(
        متلون خالی** d_mat_frag,
        خالی* a_mat_frag,
        خالی* b_mat_frag,
        خالی* c_mat_frag) {
      خودکار d = reinterpret_cast<متلون CDType**>(d_mat_frag);
      خودکار a =
          reinterpret_cast<قسم_نام MMAType<ABType>::PackType*>(a_mat_frag);
      خودکار b =
          reinterpret_cast<قسم_نام MMAType<ABType>::PackType*>(b_mat_frag);
      خودکار c = reinterpret_cast<CDType*>(c_mat_frag);

      خودکار sg = sycl::ext::oneapi::this_work_شے::get_sub_گروہ();
      صحیح lane = sg.get_local_خطی_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      مختصر صف_بھرو_انحراف = 4 * (lane >> 2);
      مختصر ستون_بھرو_انحراف = 8 * (lane % 4);

      اگر constexpr (M == 8 && N == 8 && K == 4) {
        اگر constexpr (std::is_floating_point_v<CDType>) {
          ستون_بھرو_انحراف = صف_بھرو_انحراف % 16;

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
          صحیح r_ind = (lane % 2) ? 1 : 0;
          صحیح c_ind = ((lane % 4) / 2) ? 2 : 0;

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
          قسم_نام MMAType<ABType>::PackType recv_a[2], recv_b[2];

          برائے (صحیح i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_بھرو_انحراف + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], ستون_بھرو_انحراف + i);

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
                static_cast<اعشاری>(ra[r_ind]) * static_cast<اعشاری>(rb[c_ind]);
            *d[1] += static_cast<اعشاری>(ra[r_ind]) *
                     static_cast<اعشاری>(rb[c_ind + 1]);
            *d[2] += static_cast<اعشاری>(ra[r_ind + 2]) *
                     static_cast<اعشاری>(rb[c_ind]);
            *d[3] += static_cast<اعشاری>(ra[r_ind + 2]) *
                     static_cast<اعشاری>(rb[c_ind + 1]);

            // Load partial fragment from row1 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], ستون_بھرو_انحراف + i + 16);

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
                static_cast<اعشاری>(ra[r_ind]) * static_cast<اعشاری>(rb[c_ind]);
            *d[5] += static_cast<اعشاری>(ra[r_ind]) *
                     static_cast<اعشاری>(rb[c_ind + 1]);
            *d[6] += static_cast<اعشاری>(ra[r_ind + 2]) *
                     static_cast<اعشاری>(rb[c_ind]);
            *d[7] += static_cast<اعشاری>(ra[r_ind + 2]) *
                     static_cast<اعشاری>(rb[c_ind + 1]);
          }
        }
      } ورنہ اگر constexpr (M == 8 && N == 8 && K == 16) {
        اگر constexpr (std::is_integral_v<ABType>) {
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
          برائے (صحیح i = 0; i < 4; i++) {
            قسم_نام MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_گروہ(sg, a[0], صف_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i + 4);

            خودکار a = reinterpret_cast<ABType*>(&recv_a);
            خودکار b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            برائے (صحیح j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } ورنہ اگر constexpr (M == 16 && N == 8 && K == 8) {
        اگر constexpr (std::is_floating_point_v<CDType>) {
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
          برائے (صحیح i = 0; i < 4; i++) {
            قسم_نام MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_بھرو_انحراف + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i + 4);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            برائے (صحیح j = 0; j < 2; j++) {
              *d[0] += static_cast<اعشاری>(ra[j]) * static_cast<اعشاری>(rb[j]);
              *d[1] +=
                  static_cast<اعشاری>(ra[j]) * static_cast<اعشاری>(rb[j + 2]);
              *d[2] +=
                  static_cast<اعشاری>(ra[j + 2]) * static_cast<اعشاری>(rb[j]);
              *d[3] +=
                  static_cast<اعشاری>(ra[j + 2]) * static_cast<اعشاری>(rb[j + 2]);
            }
          }
        }
      } ورنہ اگر constexpr (M == 16 && N == 8 && K == 16) {
        اگر constexpr (std::is_floating_point_v<CDType>) {
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
          برائے (صحیح i = 0; i < 4; i++) {
            قسم_نام MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_بھرو_انحراف + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[2], صف_بھرو_انحراف + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_بھرو_انحراف + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_گروہ(sg, a[3], صف_بھرو_انحراف + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], ستون_بھرو_انحراف + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_گروہ(sg, b[1], ستون_بھرو_انحراف + 4 + i);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            برائے (صحیح j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } ورنہ اگر constexpr (std::is_integral_v<ABType>) {
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
          برائے (صحیح i = 0; i < 4; i++) {
            قسم_نام MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_بھرو_انحراف + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i + 4);

            خودکار ra = reinterpret_cast<ABType*>(recv_a);
            خودکار rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            برائے (صحیح i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } ورنہ اگر constexpr (M == 16 && N == 8 && K == 32) {
        اگر constexpr (std::is_integral_v<ABType>) {
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
          برائے (صحیح i = 0; i < 4; i++) {
            قسم_نام MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[0], صف_بھرو_انحراف + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[1], صف_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[0], ستون_بھرو_انحراف + i + 4);

            خودکار a = reinterpret_cast<ABType*>(recv_a);
            خودکار b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            برائے (صحیح j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          برائے (صحیح i = 0; i < 4; i++) {
            قسم_نام MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_گروہ(sg, a[2], صف_بھرو_انحراف + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_گروہ(sg, a[3], صف_بھرو_انحراف + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_گروہ(sg, b[1], ستون_بھرو_انحراف + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_گروہ(sg, b[1], ستون_بھرو_انحراف + i + 4);

            خودکار a = reinterpret_cast<ABType*>(recv_a);
            خودکار b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            برائے (صحیح j = 0; j < 4; j++) {
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

#اگر_ختم // GGML_SYCL_DPCT_HELPER_HPP

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

#నిర్వచితం_కాకపోతే GGML_SYCL_DPCT_HELPER_HPP
#నిర్వచించు GGML_SYCL_DPCT_HELPER_HPP

#అన్తర్భావ <cstdlib>
#అన్తర్భావ <iostream>
#అన్తర్భావ <map>

#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <sycl/half_type.hpp>
#అన్తర్భావ <oneapi/mkl.hpp>

#ఒకవేళ defined(__linux__)
#అన్తర్భావ <sys/mman.h>
#లేదావేళ defined(_WIN64)
#నిర్వచితం_కాకపోతే NOMINMAX
#నిర్వచించు NOMINMAX
#అయితే_ముగింపు
#అన్తర్భావ <windows.h>
#లేకపోతే
#దోషం "Only support Windows and Linux."
#అయితే_ముగింపు

#ఒకవేళ defined(__linux__)
#అన్తర్భావ <unistd.h>
#అన్తర్భావ <sys/syscall.h>
#అయితే_ముగింపు
#ఒకవేళ defined(_WIN64)
#నిర్వచితం_కాకపోతే NOMINMAX
#నిర్వచించు NOMINMAX
#అయితే_ముగింపు
#అన్తర్భావ <windows.h>
#అయితే_ముగింపు


#నిర్వచించు DPCT_COMPATIBILITY_తాత్కాలిక (900)

#ఒకవేళ defined(_MSC_VER)
#నిర్వచించు __dpct_align__(n) __declspec(align(n))
#నిర్వచించు __dpct_inline__ __forceinline
#లేకపోతే
#నిర్వచించు __dpct_align__(n) __attribute__((aligned(n)))
#నిర్వచించు __dpct_inline__ __inline__ __attribute__((always_inline))
#అయితే_ముగింపు

#ఒకవేళ defined(_MSC_VER)
#నిర్వచించు __dpct_noinline__ __declspec(noinline)
#లేకపోతే
#నిర్వచించు __dpct_noinline__ __attribute__((noinline))
#అయితే_ముగింపు

#నిర్వచించు DPCT_వాడని(x) (శూన్య)(x)

[[noreturn]] అంతఃవరుస శూన్య _abort(స్థిరరాశి అక్షరం * str) {
    std::cerr << str << std::పంక్తి_ముగింపు;
    std::abort();
}

అంతఃవరుస std::string get_device_type_name(స్థిరరాశి sycl::device &Device) {
    స్వయంచాలక DeviceType = Device.get_info<sycl::info::device::device_type>();
    ఎంపిక (DeviceType) {
    స్థితి sycl::info::device_type::cpu:
        తిరిగి "cpu";
    స్థితి sycl::info::device_type::gpu:
        తిరిగి "gpu";
    స్థితి sycl::info::device_type::host:
        తిరిగి "host";
    స్థితి sycl::info::device_type::accelerator:
        తిరిగి "acc";
    అప్రమేయ:
        తిరిగి "unknown";
    }
}

అంతఃవరుస std::string get_device_backend_and_type(స్థిరరాశి sycl::device &device) {
    std::stringstream device_type;
    sycl::backend backend = device.get_backend();
    device_type <<  backend << ":" << get_device_type_name(device);
    తిరిగి device_type.str();
}

మూస <రకం_పేరు Ts> నిర్మాణం మాతృక_info_t {
    oneapi::mkl::transpose transpose_info[2];
    Ts                     విలువ_info[2];
    std::int64_t           కొలత_info[3];
    std::int64_t           ld_info[3];
    std::int64_t           groupsize_info;
};

నామాకాశ dpct
{
    రకనిర్వచనం sycl::queue *వరుస_ptr;
    రకనిర్వచనం sycl::event *event_ptr;
    రకనిర్వచనం అక్షరం *device_ptr;
    రకనిర్వచనం uint8_t byte_t;
    రకనిర్వచనం sycl::బఫరు<byte_t> బఫరు_t;

    /// SYCL default exception handler
    అంతఃవరుస స్వయంచాలక exception_handler = [](sycl::exception_list exceptions)
    {
        కొరకు (std::exception_ptr స్థిరరాశి &e : exceptions)
        {
            ప్రయత్నించు
            {
                std::rethrow_exception(e);
            }
            పట్టుకో (sycl::exception స్థిరరాశి &e)
            {
                std::cerr << "Caught asynchronous SYCL exception:" << std::పంక్తి_ముగింపు
                          << e.what() << std::పంక్తి_ముగింపు
                          << "Exception caught at file:" << __FILE__
                          << ", line:" << __LINE__ << std::పంక్తి_ముగింపు;
            }
        }
    };

    గణన error_code
    {
        success = 0,
        default_error = 999
    };

    గణన memcpy_direction
    {
        host_గమ్యం_host,
        host_గమ్యం_device,
        device_గమ్యం_host,
        device_గమ్యం_device,
        automatic
    };

    గణన జ్ఞాపకం_region
    {
        global = 0, // device global memory
        constant,   // device constant memory
        local,      // device local memory
        shared,     // memory which can be accessed by host and device
    };

    గణన తరగతి library_దత్తాంశం_t : గుర్తులేని అక్షరం
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
        library_దత్తాంశం_t_కొలత
    };

    మూస <రకం_పేరు T>
    నిర్మాణం దత్తాంశం_Type_2
    {
        ఉపయోగించి T2 = T;
    };
    మూస <రకం_పేరు T>
    నిర్మాణం దత్తాంశం_Type_2<sycl::vec<T, 2>>
    {
        ఉపయోగించి T2 = std::complex<T>;
    };

    స్థిర శూన్య destroy_event(event_ptr event)
    {
        తొలగించు event;
    }

    స్థిర అంతఃవరుస గుర్తులేని పూర్ణాంకం get_tid()
    {
#ఒకవేళ defined(__linux__)
        తిరిగి syscall(SYS_gettid);
#లేదావేళ defined(_WIN64)
        తిరిగి GetCurrentThreadId();
#లేకపోతే
#దోషం "Only support Windows and Linux."
#అయితే_ముగింపు
    }

    నామాకాశ detail
    {
        స్థిర శూన్య get_సంస్కరణ(స్థిరరాశి sycl::device &dev, పూర్ణాంకం &major, పూర్ణాంకం &minor)
        {
            // Version string has the following format:
            // a. OpenCL<space><major.minor><space><vendor-specific-information>
            // b. <major.minor>
            // c. <AmdGcnArchName> e.g gfx1030
            std::string ver;
            ver = dev.get_info<sycl::info::device::సంస్కరణ_2>();
            std::string::size_type i = 0;
            అయితే (i < ver.size()) {
              ఒకవేళ (అంకెనా(ver[i]))
                ఆపు;
              i++;
            }
            major = std::stoi(&(ver[i]));
            అయితే (i < ver.size()) {
              ఒకవేళ (ver[i] == '.')
                ఆపు;
              i++;
            }
            ఒకవేళ (i < ver.size()) {
              // a. and b.
              i++;
              minor = std::stoi(&(ver[i]));
            } లేకపోతే {
              // c.
              minor = 0;
            }
        }

        మూస <రకం_పేరు ట్యాగ్, రకం_పేరు T>
        తరగతి generic_error_type
        {
        బహిరంగ:
            generic_error_type() = అప్రమేయ;
            generic_error_type(T value) : value{value} {}
            క్రియాకారకం T() స్థిరరాశి { తిరిగి value; }

        అంతరంగ:
            T value;
        };

    } // namespace detail

    // COPY from DPCT head files
    /// dim3 is used to store 3 component dimensions.
    తరగతి dim3 {
        బహిరంగ:
        గుర్తులేని x, y, z;

        constexpr dim3(గుర్తులేని x = 1, గుర్తులేని y = 1, గుర్తులేని z = 1)
            : x(x), y(y), z(z) {}

        dim3(స్థిరరాశి sycl::id<3> &r) : dim3(r[2], r[1], r[0]) {}

        క్రియాకారకం sycl::range<3>() స్థిరరాశి { తిరిగి sycl::range<3>(z, y, x); }
    }; // namespace dim3

    అంతఃవరుస dim3 క్రియాకారకం*(స్థిరరాశి dim3 &a, స్థిరరాశి dim3 &b) {
    తిరిగి dim3{a.x * b.x, a.y * b.y, a.z * b.z};
    }
    // COPY from DPCT head files


    /// Pitched 2D/3D memory data.
    తరగతి pitched_దత్తాంశం
    {
    బహిరంగ:
        pitched_దత్తాంశం() : pitched_దత్తాంశం(శూన్యసూచిక, 0, 0, 0) {}
        pitched_దత్తాంశం(శూన్య *దత్తాంశం, size_t pitch, size_t x, size_t y)
            : దత్తాంశం_2(దత్తాంశం), _pitch(pitch), _x(x), _y(y) {}

        శూన్య *get_దత్తాంశం_ptr() { తిరిగి దత్తాంశం_2; }
        శూన్య సమితి_దత్తాంశం_ptr(శూన్య *దత్తాంశం) { దత్తాంశం_2 = దత్తాంశం; }

        size_t get_pitch() { తిరిగి _pitch; }
        శూన్య సమితి_pitch(size_t pitch) { _pitch = pitch; }

        size_t get_x() { తిరిగి _x; }
        శూన్య సమితి_x(size_t x) { _x = x; }

        size_t get_y() { తిరిగి _y; }
        శూన్య సమితి_y(size_t y) { _y = y; }

    అంతరంగ:
        శూన్య *దత్తాంశం_2;
        size_t _pitch, _x, _y;
    };

    తరగతి device_info
    {
    బహిరంగ:
        // get interface
        స్థిరరాశి అక్షరం *get_name() స్థిరరాశి { తిరిగి _name; }
        అక్షరం *get_name() { తిరిగి _name; }
        మూస <రకం_పేరు Work_అంశం_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_అంశం_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_అంశం_Sizes_Ty, పూర్ణాంకం *>,
                                   పూర్ణాంకం> = 0>
        స్వయంచాలక get_max_work_అంశం_sizes() స్థిరరాశి
        {
            ఒకవేళ constexpr (std::is_same_v<Work_అంశం_Sizes_Ty, sycl::range<3>>)
                తిరిగి sycl::range<3>(max_work_అంశం_sizes_i[0],
                                      max_work_అంశం_sizes_i[1],
                                      max_work_అంశం_sizes_i[2]);
            లేకపోతే
            {
                తిరిగి max_work_అంశం_sizes_i;
            }
        }
        మూస <రకం_పేరు Work_అంశం_Sizes_Ty = sycl::range<3>,
                  std::enable_if_t<std::is_same_v<Work_అంశం_Sizes_Ty, sycl::range<3>> ||
                                       std::is_same_v<Work_అంశం_Sizes_Ty, పూర్ణాంకం *>,
                                   పూర్ణాంకం> = 0>
        స్వయంచాలక get_max_work_అంశం_sizes()
        {
            ఒకవేళ constexpr (std::is_same_v<Work_అంశం_Sizes_Ty, sycl::range<3>>)
                తిరిగి sycl::range<3>(max_work_అంశం_sizes_i[0],
                                      max_work_అంశం_sizes_i[1],
                                      max_work_అంశం_sizes_i[2]);
            లేకపోతే
            {
                తిరిగి max_work_అంశం_sizes_i;
            }
        }
        తార్కిక get_host_unified_జ్ఞాపకం() స్థిరరాశి { తిరిగి host_unified_జ్ఞాపకం; }
        పూర్ణాంకం get_major_సంస్కరణ() స్థిరరాశి { తిరిగి _major; }
        పూర్ణాంకం get_minor_సంస్కరణ() స్థిరరాశి { తిరిగి _minor; }
        పూర్ణాంకం get_integrated() స్థిరరాశి { తిరిగి _integrated; }
        పూర్ణాంకం get_max_clock_frequency() స్థిరరాశి { తిరిగి _frequency; }
        పూర్ణాంకం get_max_compute_units() స్థిరరాశి { తిరిగి _max_compute_units; }
        పూర్ణాంకం get_max_work_సమూహం_కొలత() స్థిరరాశి { తిరిగి max_work_సమూహం_కొలత_2; }
        పూర్ణాంకం get_max_sub_సమూహం_కొలత() స్థిరరాశి { తిరిగి max_sub_సమూహం_కొలత; }
        పూర్ణాంకం get_max_work_వస్తువులు_చొప్పున_compute_unit() స్థిరరాశి
        {
            తిరిగి max_work_వస్తువులు_చొప్పున_compute_unit;
        }
        పూర్ణాంకం get_max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం() స్థిరరాశి
        {
            తిరిగి max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం;
        }
        మూస <రకం_పేరు NDRange_కొలత_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_కొలత_Ty, size_t *> ||
                                       std::is_same_v<NDRange_కొలత_Ty, పూర్ణాంకం *>,
                                   పూర్ణాంకం> = 0>
        స్వయంచాలక get_max_nd_range_కొలత() స్థిరరాశి
        {
            ఒకవేళ constexpr (std::is_same_v<NDRange_కొలత_Ty, size_t *>)
                తిరిగి max_nd_range_కొలత;
            లేకపోతే
                తిరిగి max_nd_range_కొలత_i;
        }
        మూస <రకం_పేరు NDRange_కొలత_Ty = size_t *,
                  std::enable_if_t<std::is_same_v<NDRange_కొలత_Ty, size_t *> ||
                                       std::is_same_v<NDRange_కొలత_Ty, పూర్ణాంకం *>,
                                   పూర్ణాంకం> = 0>
        స్వయంచాలక get_max_nd_range_కొలత()
        {
            ఒకవేళ constexpr (std::is_same_v<NDRange_కొలత_Ty, size_t *>)
                తిరిగి max_nd_range_కొలత;
            లేకపోతే
                తిరిగి max_nd_range_కొలత_i;
        }
        size_t get_global_mem_కొలత() స్థిరరాశి { తిరిగి global_mem_కొలత_2; }
        size_t get_local_mem_కొలత() స్థిరరాశి { తిరిగి local_mem_కొలత; }
        size_t get_max_mem_alloc_కొలత() స్థిరరాశి { తిరిగి max_mem_alloc_కొలత; }
        /// Returns the maximum clock rate of device's global memory in kHz. If
        /// compiler does not support this API then returns default value 3200000 kHz.
        గుర్తులేని పూర్ణాంకం get_జ్ఞాపకం_clock_రేటు() స్థిరరాశి { తిరిగి జ్ఞాపకం_clock_రేటు; }
        /// Returns the maximum bus width between device and memory in bits. If
        /// compiler does not support this API then returns default value 64 bits.
        గుర్తులేని పూర్ణాంకం get_జ్ఞాపకం_bus_వెడల్పు() స్థిరరాశి { తిరిగి జ్ఞాపకం_bus_వెడల్పు; }
        uint32_t get_device_id() స్థిరరాశి { తిరిగి _device_id; }
        std::array<గుర్తులేని అక్షరం, 16> get_uuid() స్థిరరాశి { తిరిగి _uuid; }
        /// Returns global memory cache size in bytes.
        గుర్తులేని పూర్ణాంకం get_global_mem_cache_కొలత() స్థిరరాశి
        {
            తిరిగి global_mem_cache_కొలత;
        }

        // set interface
        శూన్య సమితి_name(స్థిరరాశి అక్షరం *name)
        {
            size_t నిడివి = అక్షరశ్రేణి_పొడవు(name);
            ఒకవేళ (నిడివి < 256)
            {
                std::జ్ఞాపకం_నకలు(_name, name, నిడివి + 1);
            }
            లేకపోతే
            {
                std::జ్ఞాపకం_నకలు(_name, name, 255);
                _name[255] = '\0';
            }
        }
        శూన్య సమితి_max_work_అంశం_sizes(స్థిరరాశి sycl::range<3> max_work_అంశం_sizes)
        {
            కొరకు (పూర్ణాంకం i = 0; i < 3; ++i)
                max_work_అంశం_sizes_i[i] = max_work_అంశం_sizes[i];
        }
        [[deprecated]] శూన్య
        సమితి_max_work_అంశం_sizes(స్థిరరాశి sycl::id<3> max_work_అంశం_sizes)
        {
            కొరకు (పూర్ణాంకం i = 0; i < 3; ++i)
            {
                max_work_అంశం_sizes_i[i] = max_work_అంశం_sizes[i];
            }
        }
        శూన్య సమితి_host_unified_జ్ఞాపకం(తార్కిక host_unified_జ్ఞాపకం_2)
        {
            host_unified_జ్ఞాపకం = host_unified_జ్ఞాపకం_2;
        }
        శూన్య సమితి_major_సంస్కరణ(పూర్ణాంకం major) { _major = major; }
        శూన్య సమితి_minor_సంస్కరణ(పూర్ణాంకం minor) { _minor = minor; }
        శూన్య సమితి_integrated(పూర్ణాంకం integrated) { _integrated = integrated; }
        శూన్య సమితి_max_clock_frequency(పూర్ణాంకం frequency) { _frequency = frequency; }
        శూన్య సమితి_max_compute_units(పూర్ణాంకం max_compute_units)
        {
            _max_compute_units = max_compute_units;
        }
        శూన్య సమితి_global_mem_కొలత(size_t global_mem_కొలత)
        {
            global_mem_కొలత_2 = global_mem_కొలత;
        }
        శూన్య సమితి_local_mem_కొలత(size_t local_mem_కొలత_2)
        {
            local_mem_కొలత = local_mem_కొలత_2;
        }
        శూన్య సమితి_max_mem_alloc_కొలత(size_t max_mem_alloc_కొలత_2)
        {
            max_mem_alloc_కొలత = max_mem_alloc_కొలత_2;
        }
        శూన్య సమితి_max_work_సమూహం_కొలత(పూర్ణాంకం max_work_సమూహం_కొలత)
        {
            max_work_సమూహం_కొలత_2 = max_work_సమూహం_కొలత;
        }
        శూన్య సమితి_max_sub_సమూహం_కొలత(పూర్ణాంకం max_sub_సమూహం_కొలత_2)
        {
            max_sub_సమూహం_కొలత = max_sub_సమూహం_కొలత_2;
        }
        శూన్య
        సమితి_max_work_వస్తువులు_చొప్పున_compute_unit(పూర్ణాంకం max_work_వస్తువులు_చొప్పున_compute_unit_2)
        {
            max_work_వస్తువులు_చొప్పున_compute_unit = max_work_వస్తువులు_చొప్పున_compute_unit_2;
        }
        శూన్య సమితి_max_nd_range_కొలత(పూర్ణాంకం max_nd_range_కొలత_2[])
        {
            కొరకు (పూర్ణాంకం i = 0; i < 3; i++)
            {
                max_nd_range_కొలత[i] = max_nd_range_కొలత_2[i];
                max_nd_range_కొలత_i[i] = max_nd_range_కొలత_2[i];
            }
        }
        శూన్య సమితి_జ్ఞాపకం_clock_రేటు(గుర్తులేని పూర్ణాంకం జ్ఞాపకం_clock_రేటు_2)
        {
            జ్ఞాపకం_clock_రేటు = జ్ఞాపకం_clock_రేటు_2;
        }
        శూన్య సమితి_జ్ఞాపకం_bus_వెడల్పు(గుర్తులేని పూర్ణాంకం జ్ఞాపకం_bus_వెడల్పు_2)
        {
            జ్ఞాపకం_bus_వెడల్పు = జ్ఞాపకం_bus_వెడల్పు_2;
        }
        శూన్య
        సమితి_max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం(పూర్ణాంకం max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం_2)
        {
            max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం = max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం_2;
        }
        శూన్య సమితి_device_id(uint32_t device_id)
        {
            _device_id = device_id;
        }
        శూన్య సమితి_uuid(std::array<గుర్తులేని అక్షరం, 16> uuid)
        {
            _uuid = std::move(uuid);
        }
        శూన్య సమితి_global_mem_cache_కొలత(గుర్తులేని పూర్ణాంకం global_mem_cache_కొలత_2)
        {
            global_mem_cache_కొలత = global_mem_cache_కొలత_2;
        }

    అంతరంగ:
        అక్షరం _name[256];
        పూర్ణాంకం max_work_అంశం_sizes_i[3];
        తార్కిక host_unified_జ్ఞాపకం = అబద్ధం;
        పూర్ణాంకం _major;
        పూర్ణాంకం _minor;
        పూర్ణాంకం _integrated = 0;
        పూర్ణాంకం _frequency;
        // Set estimated value 3200000 kHz as default value.
        గుర్తులేని పూర్ణాంకం జ్ఞాపకం_clock_రేటు = 3200000;
        // Set estimated value 64 bits as default value.
        గుర్తులేని పూర్ణాంకం జ్ఞాపకం_bus_వెడల్పు = 64;
        గుర్తులేని పూర్ణాంకం global_mem_cache_కొలత;
        పూర్ణాంకం _max_compute_units;
        పూర్ణాంకం max_work_సమూహం_కొలత_2;
        పూర్ణాంకం max_sub_సమూహం_కొలత;
        పూర్ణాంకం max_work_వస్తువులు_చొప్పున_compute_unit;
        పూర్ణాంకం max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం;
        size_t global_mem_కొలత_2;
        size_t local_mem_కొలత;
        size_t max_mem_alloc_కొలత;
        size_t max_nd_range_కొలత[3];
        పూర్ణాంకం max_nd_range_కొలత_i[3];
        uint32_t _device_id;
        std::array<గుర్తులేని అక్షరం, 16> _uuid;
    };

    స్థిర పూర్ణాంకం get_major_సంస్కరణ(స్థిరరాశి sycl::device &dev)
    {
        పూర్ణాంకం major, minor;
        detail::get_సంస్కరణ(dev, major, minor);
        తిరిగి major;
    }

    స్థిర పూర్ణాంకం get_minor_సంస్కరణ(స్థిరరాశి sycl::device &dev)
    {
        పూర్ణాంకం major, minor;
        detail::get_సంస్కరణ(dev, major, minor);
        తిరిగి minor;
    }

    స్థిర శూన్య get_device_info(device_info &బహిర్గతం, స్థిరరాశి sycl::device &dev)
    {
        device_info prop;
        prop.సమితి_name(dev.get_info<sycl::info::device::name>().c_str());

        పూర్ణాంకం major, minor;
        detail::get_సంస్కరణ(dev, major, minor);
        prop.సమితి_major_సంస్కరణ(major);
        prop.సమితి_minor_సంస్కరణ(minor);

        prop.సమితి_max_work_అంశం_sizes(
#ఒకవేళ (__SYCL_COMPILER_VERSION && __SYCL_COMPILER_VERSION < 20220902)
            // oneAPI DPC++ compiler older than 2022/09/02, where max_work_item_sizes
            // is an enum class element
            dev.get_info<sycl::info::device::max_work_అంశం_sizes>());
#లేకపోతే
            // SYCL 2020-conformant code, max_work_item_sizes is a struct templated by
            // an int
            dev.get_info<sycl::info::device::max_work_అంశం_sizes<3>>());
#అయితే_ముగింపు
        prop.సమితి_host_unified_జ్ఞాపకం(dev.has(sycl::aspect::usm_host_allocations));

        prop.సమితి_max_clock_frequency(
            dev.get_info<sycl::info::device::max_clock_frequency>() * 1000);

        prop.సమితి_max_compute_units(
            dev.get_info<sycl::info::device::max_compute_units>());
        prop.సమితి_max_work_సమూహం_కొలత(
            dev.get_info<sycl::info::device::max_work_సమూహం_కొలత>());
        prop.సమితి_global_mem_కొలత(dev.get_info<sycl::info::device::global_mem_కొలత>());
        prop.సమితి_local_mem_కొలత(dev.get_info<sycl::info::device::local_mem_కొలత_2>());
        prop.సమితి_max_mem_alloc_కొలత(dev.get_info<sycl::info::device::max_mem_alloc_కొలత_2>());

#ఒకవేళ (defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 6)
        ఒకవేళ (dev.has(sycl::aspect::ext_intel_జ్ఞాపకం_clock_రేటు))
        {
            గుర్తులేని పూర్ణాంకం tmp =
                dev.get_info<sycl::ext::intel::info::device::జ్ఞాపకం_clock_రేటు_2>();
            ఒకవేళ (tmp != 0)
                prop.సమితి_జ్ఞాపకం_clock_రేటు(1000 * tmp);
        }
        ఒకవేళ (dev.has(sycl::aspect::ext_intel_జ్ఞాపకం_bus_వెడల్పు))
        {
            prop.సమితి_జ్ఞాపకం_bus_వెడల్పు(
                dev.get_info<sycl::ext::intel::info::device::జ్ఞాపకం_bus_వెడల్పు_2>());
        }
        ఒకవేళ (dev.has(sycl::aspect::ext_intel_device_id))
        {
            prop.సమితి_device_id(
                dev.get_info<sycl::ext::intel::info::device::device_id>());
        }
        ఒకవేళ (dev.has(sycl::aspect::ext_intel_device_info_uuid))
        {
            prop.సమితి_uuid(dev.get_info<sycl::ext::intel::info::device::uuid>());
        }
#లేదావేళ defined(_MSC_VER) && !defined(__clang__)
#సూచన సందేశం("get_device_info: querying జ్ఞాపకం_clock_రేటు_2 and \
        జ్ఞాపకం_bus_వెడల్పు_2 are not supported by the compiler వాడిన. \
        Use 3200000 kHz as జ్ఞాపకం_clock_రేటు_2 అప్రమేయ value. \
        Use 64 బిట్లు as జ్ఞాపకం_bus_వెడల్పు_2 అప్రమేయ value.")
#లేకపోతే
#warning "get_device_info: querying జ్ఞాపకం_clock_రేటు_2 and \
        జ్ఞాపకం_bus_వెడల్పు_2 are not supported by the compiler వాడిన. \
        Use 3200000 kHz as జ్ఞాపకం_clock_రేటు_2 అప్రమేయ value. \
        Use 64 బిట్లు as జ్ఞాపకం_bus_వెడల్పు_2 అప్రమేయ value."
#అయితే_ముగింపు

        size_t max_sub_సమూహం_కొలత_2 = 1;
        std::vector<size_t> sub_సమూహం_sizes =
            dev.get_info<sycl::info::device::sub_సమూహం_sizes>();

        కొరకు (స్థిరరాశి స్వయంచాలక &sub_సమూహం_కొలత : sub_సమూహం_sizes)
        {
            ఒకవేళ (max_sub_సమూహం_కొలత_2 < sub_సమూహం_కొలత)
                max_sub_సమూహం_కొలత_2 = sub_సమూహం_కొలత;
        }

        prop.సమితి_max_sub_సమూహం_కొలత(max_sub_సమూహం_కొలత_2);

        prop.సమితి_max_work_వస్తువులు_చొప్పున_compute_unit(
            dev.get_info<sycl::info::device::max_work_సమూహం_కొలత>());
        పూర్ణాంకం max_nd_range_కొలత_2[] = {0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF};
        prop.సమితి_max_nd_range_కొలత(max_nd_range_కొలత_2);

        // Estimates max register size per work group, feel free to update the value
        // according to device properties.
        prop.సమితి_max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం(65536);

        prop.సమితి_global_mem_cache_కొలత(
            dev.get_info<sycl::info::device::global_mem_cache_కొలత_2>());
        బహిర్గతం = prop;
    }

    /// dpct device extension
    తరగతి device_ext : బహిరంగ sycl::device {
      రకనిర్వచనం std::mutex mutex_type;

     బహిరంగ:
      device_ext() : sycl::device() {}
      ~device_ext() {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        clear_queues();
      }
      device_ext(స్థిరరాశి sycl::device &పునాది) : sycl::device(పునాది) {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        init_queues();
      }

      పూర్ణాంకం is_native_atomic_supported() { తిరిగి 0; }
      పూర్ణాంకం get_major_సంస్కరణ() స్థిరరాశి { తిరిగి dpct::get_major_సంస్కరణ(*ఇది); }

      పూర్ణాంకం get_minor_సంస్కరణ() స్థిరరాశి { తిరిగి dpct::get_minor_సంస్కరణ(*ఇది); }

      పూర్ణాంకం get_max_compute_units() స్థిరరాశి {
        తిరిగి get_device_info().get_max_compute_units();
      }

      /// Return the maximum clock frequency of this device in KHz.
      పూర్ణాంకం get_max_clock_frequency() స్థిరరాశి {
        తిరిగి get_device_info().get_max_clock_frequency();
      }

      పూర్ణాంకం get_integrated() స్థిరరాశి { తిరిగి get_device_info().get_integrated(); }

      పూర్ణాంకం get_max_sub_సమూహం_కొలత() స్థిరరాశి {
        తిరిగి get_device_info().get_max_sub_సమూహం_కొలత();
      }

      పూర్ణాంకం get_max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం() స్థిరరాశి {
        తిరిగి get_device_info().get_max_రిజిస్టరు_కొలత_చొప్పున_work_సమూహం();
      }

      పూర్ణాంకం get_max_work_సమూహం_కొలత() స్థిరరాశి {
        తిరిగి get_device_info().get_max_work_సమూహం_కొలత();
      }

      పూర్ణాంకం get_mem_పునాది_addr_align() స్థిరరాశి {
        తిరిగి get_info<sycl::info::device::mem_పునాది_addr_align>();
      }

      size_t get_global_mem_కొలత() స్థిరరాశి {
        తిరిగి get_device_info().get_global_mem_కొలత();
      }

      size_t get_max_mem_alloc_కొలత() స్థిరరాశి {
        తిరిగి get_device_info().get_max_mem_alloc_కొలత();
      }

      /// Get the number of bytes of free and total memory on the SYCL device.
      /// \param [out] free_memory The number of bytes of free memory on the
      /// SYCL device. \param [out] total_memory The number of bytes of total
      /// memory on the SYCL device.
      శూన్య get_జ్ఞాపకం_info(size_t &free_జ్ఞాపకం, size_t &మొత్తం_జ్ఞాపకం) {
        మొత్తం_జ్ఞాపకం = get_device_info().get_global_mem_కొలత();
        స్థిరరాశి అక్షరం *warning_info =
            "get_memory_info: [warning] ext_intel_free_memory is not "
            "supported (export/set ZES_ENABLE_SYSMAN=1 to support), "
            "use total memory as free memory";
#ఒకవేళ (defined(__SYCL_COMPILER_VERSION) && __SYCL_COMPILER_VERSION >= 20221105)
        ఒకవేళ (!has(sycl::aspect::ext_intel_free_జ్ఞాపకం)) {
          std::cerr << warning_info << std::పంక్తి_ముగింపు;
          free_జ్ఞాపకం = మొత్తం_జ్ఞాపకం;
        } లేకపోతే {
          free_జ్ఞాపకం = get_info<sycl::ext::intel::info::device::free_జ్ఞాపకం>();
        }
#లేకపోతే
        std::cerr << warning_info << std::పంక్తి_ముగింపు;
        free_జ్ఞాపకం = మొత్తం_జ్ఞాపకం;
#ఒకవేళ defined(_MSC_VER) && !defined(__clang__)
#సూచన సందేశం("Querying the number of bytes of free memory is not supported")
#లేకపోతే
#warning "Querying the number of bytes of free memory is not supported"
#అయితే_ముగింపు
#అయితే_ముగింపు
      }

      శూన్య get_device_info(device_info &బహిర్గతం) స్థిరరాశి {
        dpct::get_device_info(బహిర్గతం, *ఇది);
      }

      device_info get_device_info() స్థిరరాశి {
        device_info prop;
        dpct::get_device_info(prop, *ఇది);
        తిరిగి prop;
      }

      శూన్య reset() {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        clear_queues();
        init_queues();
      }

      sycl::queue &లోపల_క్రమం_వరుస() { తిరిగి q_లోపల_క్రమం; }

      sycl::queue &బహిర్గతం_యొక్క_క్రమం_వరుస() { తిరిగి q_బహిర్గతం_యొక్క_క్రమం; }

      sycl::queue &default_వరుస() { తిరిగి లోపల_క్రమం_వరుస(); }

      శూన్య queues_wait_and_throw() {
        std::unique_lock<mutex_type> తాళం(m_mutex);
        తాళం.unlock();
        కొరకు (స్వయంచాలక &q : _queues) {
            q.wait_and_throw();
        }
        // Guard the destruct of current_queues to make sure the ref count is
        // safe.
        తాళం.తాళం();
      }

      sycl::queue create_వరుస(తార్కిక enable_exception_handler = అబద్ధం) {
        తిరిగి create_లోపల_క్రమం_వరుస(enable_exception_handler);
      }

      sycl::queue create_వరుస(sycl::device device,
                               తార్కిక enable_exception_handler = అబద్ధం) {
        తిరిగి create_లోపల_క్రమం_వరుస(device, enable_exception_handler);
      }

      sycl::queue create_లోపల_క్రమం_వరుస(తార్కిక enable_exception_handler = అబద్ధం) {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        తిరిగి create_వరుస_impl(enable_exception_handler,
                                 sycl::property::queue::లోపల_క్రమం());
      }

      sycl::queue create_లోపల_క్రమం_వరుస(sycl::device device,
                                        తార్కిక enable_exception_handler = అబద్ధం) {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        తిరిగి create_వరుస_impl(device, enable_exception_handler,
                                 sycl::property::queue::లోపల_క్రమం());
      }

      sycl::queue create_బహిర్గతం_యొక్క_క్రమం_వరుస(
          తార్కిక enable_exception_handler = అబద్ధం) {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        తిరిగి create_వరుస_impl(enable_exception_handler);
      }

      శూన్య destroy_వరుస(sycl::queue queue) {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        _queues.erase(std::remove_if(_queues.begin(), _queues.end(),
                                    [=](స్థిరరాశి sycl::queue &q) -> తార్కిక
                                    {
                                        తిరిగి q == queue;
                                    }),
                    _queues.end());
      }
      శూన్య సమితి_saved_వరుస(sycl::queue q) {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        saved_వరుస = q;
      }
      sycl::queue get_saved_వరుస() స్థిరరాశి {
        std::lock_guard<mutex_type> తాళం(m_mutex);
        తిరిగి saved_వరుస;
      }

     అంతరంగ:
      శూన్య clear_queues() { _queues.clear(); }

      శూన్య init_queues() {
        q_లోపల_క్రమం =
            create_వరుస_impl(నిజం, sycl::property::queue::లోపల_క్రమం());
        q_బహిర్గతం_యొక్క_క్రమం = create_వరుస_impl(నిజం);
        saved_వరుస = default_వరుస();
      }

      /// Caller should acquire resource \p m_mutex before calling this
      /// function.
      మూస <తరగతి... Properties>
      sycl::queue create_వరుస_impl(తార్కిక enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ఒకవేళ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            *ఇది, eh,
            sycl::property_list(
#నిర్వచితమైతే DPCT_PROFILING_ENABLED
                sycl::property::queue::enable_profiling(),
#అయితే_ముగింపు
                properties...)));

        తిరిగి _queues.back();
      }

      మూస <తరగతి... Properties>
      sycl::queue create_వరుస_impl(sycl::device device,
                                    తార్కిక enable_exception_handler,
                                    Properties... properties) {
        sycl::async_handler eh = {};
        ఒకవేళ (enable_exception_handler) {
          eh = exception_handler;
        }
        _queues.push_back(sycl::queue(
            device, eh,
                        sycl::property_list(
#నిర్వచితమైతే DPCT_PROFILING_ENABLED
                            sycl::property::queue::enable_profiling(),
#అయితే_ముగింపు
                            properties...)));

        తిరిగి _queues.back();
      }

      శూన్య get_సంస్కరణ(పూర్ణాంకం &major, పూర్ణాంకం &minor) స్థిరరాశి {
        detail::get_సంస్కరణ(*ఇది, major, minor);
      }
      sycl::queue q_లోపల_క్రమం, q_బహిర్గతం_యొక్క_క్రమం;
      sycl::queue saved_వరుస;
      std::vector<sycl::queue> _queues;
      మార్చదగిన mutex_type m_mutex;
    };


    /// device manager
    తరగతి dev_mgr
    {
    బహిరంగ:
        device_ext &current_device()
        {
            గుర్తులేని పూర్ణాంకం dev_id = current_device_id();
            check_id(dev_id);
            తిరిగి *_devs[dev_id];
        }
        device_ext &cpu_device() స్థిరరాశి
        {
            std::lock_guard<std::recursive_mutex> తాళం(m_mutex);
            ఒకవేళ (_cpu_device == -1)
            {
                విసురు std::runtime_error("no valid cpu device");
            }
            లేకపోతే
            {
                తిరిగి *_devs[_cpu_device];
            }
        }
        device_ext &get_device(గుర్తులేని పూర్ణాంకం id) స్థిరరాశి
        {
            std::lock_guard<std::recursive_mutex> తాళం(m_mutex);
            check_id(id);
            తిరిగి *_devs[id];
        }
        గుర్తులేని పూర్ణాంకం current_device_id() స్థిరరాశి
        {
            std::lock_guard<std::recursive_mutex> తాళం(m_mutex);
            స్వయంచాలక it = _thread2dev_map.find(get_tid());
            ఒకవేళ (it != _thread2dev_map.end())
                తిరిగి it->second;
            తిరిగి DEFAULT_DEVICE_ID;
        }

        /// Select device with a device ID.
        /// \param [in] id The id of the device which can
        /// be obtained through get_device_id(const sycl::device).
        శూన్య select_device(గుర్తులేని పూర్ణాంకం id)
        {
            std::lock_guard<std::recursive_mutex> తాళం(m_mutex);
            check_id(id);
            _thread2dev_map[get_tid()] = id;
        }
        గుర్తులేని పూర్ణాంకం device_లెక్క() { తిరిగి _devs.size(); }

        గుర్తులేని పూర్ణాంకం get_device_id(స్థిరరాశి sycl::device &dev)
        {
            గుర్తులేని పూర్ణాంకం id = 0;
            కొరకు (స్వయంచాలక &dev_అంశం : _devs)
            {
                ఒకవేళ (*dev_అంశం == dev)
                {
                    తిరిగి id;
                }
                id++;
            }
            తిరిగి -1;
        }

        అంతఃవరుస std::string get_preferred_gpu_platform_name() {
            std::string ఫలితం;

            std::string filter = "";
            అక్షరం* env = getenv("ONEAPI_DEVICE_SELECTOR");
            ఒకవేళ (env) {
                ఒకవేళ (std::అక్షరశ్రేణి_శోధన(env, "level_zero")) {
                    filter = "level-zero";
                }
                లేకపోతే ఒకవేళ (std::అక్షరశ్రేణి_శోధన(env, "opencl")) {
                    filter = "opencl";
                }
                లేకపోతే ఒకవేళ (std::అక్షరశ్రేణి_శోధన(env, "cuda")) {
                    filter = "cuda";
                }
                లేకపోతే ఒకవేళ (std::అక్షరశ్రేణి_శోధన(env, "hip")) {
                    filter = "hip";
                }
                లేకపోతే {
                    విసురు std::runtime_error("invalid device filter: " + std::string(env));
                }
            } లేకపోతే {
                స్వయంచాలక default_device = sycl::device(sycl::default_selector_v);
                స్వయంచాలక default_platform_name = default_device.get_platform().get_info<sycl::info::platform::name>();

                ఒకవేళ (std::అక్షరశ్రేణి_శోధన(default_platform_name.c_str(), "Level-Zero") || default_device.is_cpu()) {
                    filter = "level-zero";
                }
                లేకపోతే ఒకవేళ (std::అక్షరశ్రేణి_శోధన(default_platform_name.c_str(), "CUDA")) {
                    filter = "cuda";
                }
                లేకపోతే ఒకవేళ (std::అక్షరశ్రేణి_శోధన(default_platform_name.c_str(), "HIP")) {
                    filter = "hip";
                }
            }

            స్వయంచాలక platform_list = sycl::platform::get_platforms();

            కొరకు (స్థిరరాశి స్వయంచాలక& platform : platform_list) {
                స్వయంచాలక devices = platform.get_devices();
                స్వయంచాలక gpu_dev = std::find_if(devices.begin(), devices.end(), [](స్థిరరాశి sycl::device& d) {
                    తిరిగి d.is_gpu();
                });

                ఒకవేళ (gpu_dev == devices.end()) {
                    // cout << "platform [" << platform_name
                    //      << "] does not contain GPU devices, skipping\n";
                    కొనసాగు;
                }

                స్వయంచాలక platform_name = platform.get_info<sycl::info::platform::name>();
                std::string platform_name_దిగువ_case;
                platform_name_దిగువ_case.resize(platform_name.size());

                std::transform(
                    platform_name.begin(), platform_name.end(), platform_name_దిగువ_case.begin(), ::tolower);

                ఒకవేళ (platform_name_దిగువ_case.find(filter) == std::string::npos) {
                    // cout << "platform [" << platform_name
                    //      << "] does not match with requested "
                    //      << filter << ", skipping\n";
                    కొనసాగు;
                }

                ఫలితం = platform_name;
            }

            ఒకవేళ (ఫలితం.ఖాళీలు())
                విసురు std::runtime_error("can not find preferred GPU platform");

            తిరిగి ఫలితం;
        }

        మూస <తరగతి DeviceSelector>
        std::enable_if_t<
            std::is_invocable_r_v<పూర్ణాంకం, DeviceSelector, స్థిరరాశి sycl::device &>>
        select_device(స్థిరరాశి DeviceSelector &selector = sycl::gpu_selector_v)
        {
            sycl::device selected_device = sycl::device(selector);
            గుర్తులేని పూర్ణాంకం selected_device_id = get_device_id(selected_device);
            select_device(selected_device_id);
        }

        /// Returns the instance of device manager singleton.
        స్థిర dev_mgr &instance()
        {
            స్థిర dev_mgr d_m;
            తిరిగి d_m;
        }
        dev_mgr(స్థిరరాశి dev_mgr &) = తొలగించు;
        dev_mgr &క్రియాకారకం=(స్థిరరాశి dev_mgr &) = తొలగించు;
        dev_mgr(dev_mgr &&) = తొలగించు;
        dev_mgr &క్రియాకారకం=(dev_mgr &&) = తొలగించు;

    అంతరంగ:
        మార్చదగిన std::recursive_mutex m_mutex;
        స్థిర తార్కిక compare_dev(sycl::device &device1, sycl::device &device2)
        {
            sycl::backend backend1 = device1.get_backend();
            sycl::backend backend2 = device2.get_backend();
            // levelzero backends always come first
            ఒకవేళ(backend1 == sycl::backend::ext_oneapi_level_zero && backend2 != sycl::backend::ext_oneapi_level_zero) తిరిగి నిజం;
            ఒకవేళ(backend1 != sycl::backend::ext_oneapi_level_zero && backend2 == sycl::backend::ext_oneapi_level_zero) తిరిగి అబద్ధం;
            dpct::device_info prop1;
            dpct::get_device_info(prop1, device1);
            dpct::device_info prop2;
            dpct::get_device_info(prop2, device2);
            తిరిగి prop1.get_max_compute_units() > prop2.get_max_compute_units();
        }
        స్థిర పూర్ణాంకం convert_backend_సూచీ(std::string & backend) {
            ఒకవేళ (backend == "ext_oneapi_level_zero:gpu") తిరిగి 0;
            ఒకవేళ (backend == "opencl:gpu") తిరిగి 1;
            ఒకవేళ (backend == "ext_oneapi_cuda:gpu") తిరిగి 2;
            ఒకవేళ (backend == "ext_oneapi_hip:gpu") తిరిగి 3;
            ఒకవేళ (backend == "opencl:cpu") తిరిగి 4;
            ఒకవేళ (backend == "opencl:acc") తిరిగి 5;
            రాయి("convert_backend_index: can't handle backend=%s\n", backend.c_str());
            _abort("fatal error");
        }
        స్థిర తార్కిక compare_backend(std::string &backend1, std::string &backend2) {
            తిరిగి convert_backend_సూచీ(backend1) < convert_backend_సూచీ(backend2);
        }
        dev_mgr()
        {
            sycl::device default_device =
                sycl::device(sycl::default_selector_v);
            _devs.push_back(std::make_shared<device_ext>(default_device));

            std::vector<sycl::device> sycl_all_devs;
            // Collect other devices except for the default device.
            ఒకవేళ (default_device.is_cpu())
                _cpu_device = 0;

            స్వయంచాలక Platforms = sycl::platform::get_platforms();
            // Keep track of the number of devices per backend
            std::map<sycl::backend, size_t> DeviceNums;
            std::map<std::string, std::vector<sycl::device>> backend_devices;
            స్వయంచాలక preferred_platform_name = get_preferred_gpu_platform_name();

            అయితే (!Platforms.ఖాళీలు()) {
                స్వయంచాలక Platform = Platforms.back();
                Platforms.pop_back();
                స్వయంచాలక platform_name = Platform.get_info<sycl::info::platform::name>();
                ఒకవేళ (platform_name.compare(preferred_platform_name) != 0) {
                    కొనసాగు;
                }
                స్వయంచాలక devices = Platform.get_devices();
                std::string backend_type = get_device_backend_and_type(devices[0]);
                కొరకు (స్థిరరాశి స్వయంచాలక &device : devices) {
                    backend_devices[backend_type].push_back(device);
                }
            }

            std::vector<std::string> కీలకాలు;
            కొరకు(స్వయంచాలక it = backend_devices.begin(); it != backend_devices.end(); ++it) {
                కీలకాలు.push_back(it->మొదటి);
            }
            std::sort(కీలకాలు.begin(), కీలకాలు.end(), compare_backend);

            కొరకు (స్వయంచాలక &కీలకం : కీలకాలు) {
                std::vector<sycl::device> devs = backend_devices[కీలకం];
                std::sort(devs.begin(), devs.end(), compare_dev);
                కొరకు (స్థిరరాశి స్వయంచాలక &dev : devs) {
                    sycl_all_devs.push_back(dev);
                }
            }

            కొరకు (స్వయంచాలక &dev : sycl_all_devs)
            {
                ఒకవేళ (dev == default_device)
                {
                    కొనసాగు;
                }
                _devs.push_back(std::make_shared<device_ext>(dev));
                ఒకవేళ (_cpu_device == -1 && dev.is_cpu())
                {
                    _cpu_device = _devs.size() - 1;
                }
            }
        }
        శూన్య check_id(గుర్తులేని పూర్ణాంకం id) స్థిరరాశి
        {
            ఒకవేళ (id >= _devs.size())
            {
                విసురు std::runtime_error("invalid device id");
            }
        }
        std::vector<std::shared_ptr<device_ext>> _devs;
        /// DEFAULT_DEVICE_ID is used, if current_device_id() can not find current
        /// thread id in _thread2dev_map, which means default device should be used
        /// for the current thread.
        స్థిరరాశి గుర్తులేని పూర్ణాంకం DEFAULT_DEVICE_ID = 0;
        /// thread-id to device-id map.
        std::map<గుర్తులేని పూర్ణాంకం, గుర్తులేని పూర్ణాంకం> _thread2dev_map;
        పూర్ణాంకం _cpu_device = -1;
    };

    స్థిర అంతఃవరుస sycl::queue &get_default_వరుస()
    {
        తిరిగి dev_mgr::instance().current_device().default_వరుస();
    }

    నామాకాశ detail
    {
        గణన తరగతి సూచిక_access_attribute
        {
            host_only = 0,
            device_only,
            host_device,
            end
        };

        స్థిర సూచిక_access_attribute get_సూచిక_attribute(sycl::queue &q,
                                                              స్థిరరాశి శూన్య *ptr)
        {
            ఎంపిక (sycl::get_సూచిక_type(ptr, q.get_context()))
            {
            స్థితి sycl::usm::alloc::unknown:
                తిరిగి సూచిక_access_attribute::host_only;
            స్థితి sycl::usm::alloc::device:
                తిరిగి సూచిక_access_attribute::device_only;
            స్థితి sycl::usm::alloc::shared:
            స్థితి sycl::usm::alloc::host:
                తిరిగి సూచిక_access_attribute::host_device;
            }
        }

        మూస <రకం_పేరు ArgT>
        అంతఃవరుస constexpr std::uint64_t get_type_combination_id(ArgT Val)
        {
            static_assert((గుర్తులేని అక్షరం)library_దత్తాంశం_t::library_దత్తాంశం_t_కొలత <=
                              std::numeric_limits<గుర్తులేని అక్షరం>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(std::is_same_v<ArgT, library_దత్తాంశం_t>, "Unsupported ArgT");
            తిరిగి (std::uint64_t)Val;
        }

        మూస <రకం_పేరు మొదటి_T, రకం_పేరు... RestT>
        అంతఃవరుస constexpr std::uint64_t get_type_combination_id(మొదటి_T మొదటి_Val,
                                                               RestT... RestVal)
        {
            static_assert((std::uint8_t)library_దత్తాంశం_t::library_దత్తాంశం_t_కొలత <=
                              std::numeric_limits<గుర్తులేని అక్షరం>::max() &&
                          "library_data_t size exceeds limit.");
            static_assert(పరిమాణం...(RestT) <= 8 && "Too many parameters");
            static_assert(std::is_same_v<మొదటి_T, library_దత్తాంశం_t>, "Unsupported FirstT");
            తిరిగి get_type_combination_id(RestVal...) << 8 | ((std::uint64_t)మొదటి_Val);
        }

        తరగతి mem_mgr
        {
            mem_mgr()
            {
                // Reserved address space, no real memory allocation happens here.
#ఒకవేళ defined(__linux__)
                mapped_చిరునామా_space =
                    (byte_t *)mmap(శూన్యసూచిక, mapped_region_కొలత, PROT_NONE,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#లేదావేళ defined(_WIN64)
                mapped_చిరునామా_space = (byte_t *)VirtualAlloc(
                    ఖాళీ,               // NULL specified as the base address parameter
                    mapped_region_కొలత, // Size of allocation
                    MEM_RESERVE,        // Allocate reserved pages
                    పుట_NOACCESS);     // Protection = no access
#లేకపోతే
#దోషం "Only support Windows and Linux."
#అయితే_ముగింపు
                తదుపరి_free = mapped_చిరునామా_space;
            }

        బహిరంగ:
            ఉపయోగించి బఫరు_id_t = పూర్ణాంకం;

            నిర్మాణం కేటాయింపు
            {
                బఫరు_t బఫరు;
                byte_t *alloc_ptr;
                size_t size;
            };

            ~mem_mgr()
            {
#ఒకవేళ defined(__linux__)
                munmap(mapped_చిరునామా_space, mapped_region_కొలత);
#లేదావేళ defined(_WIN64)
                VirtualFree(mapped_చిరునామా_space, 0, MEM_RELEASE);
#లేకపోతే
#దోషం "Only support Windows and Linux."
#అయితే_ముగింపు
            }

            mem_mgr(స్థిరరాశి mem_mgr &) = తొలగించు;
            mem_mgr &క్రియాకారకం=(స్థిరరాశి mem_mgr &) = తొలగించు;
            mem_mgr(mem_mgr &&) = తొలగించు;
            mem_mgr &క్రియాకారకం=(mem_mgr &&) = తొలగించు;

            /// Allocate
            శూన్య *mem_alloc(size_t size)
            {
                ఒకవేళ (!size)
                    తిరిగి శూన్యసూచిక;
                std::lock_guard<std::mutex> తాళం(m_mutex);
                ఒకవేళ (తదుపరి_free + size > mapped_చిరునామా_space + mapped_region_కొలత)
                {
                    విసురు std::runtime_error("dpct_malloc: out of memory for virtual memory pool");
                }
                //కేటాయింపు
                sycl::range<1> r(size);
                బఫరు_t buf(r);
                కేటాయింపు A{buf, తదుపరి_free, size};
                // Map allocation to device pointer
                శూన్య *ఫలితం = తదుపరి_free;
                m_map.emplace(తదుపరి_free + size, A);
                // Update pointer to the next free space.
                తదుపరి_free += (size + extra_padding + alignment - 1) & ~(alignment - 1);

                తిరిగి ఫలితం;
            }

            /// Deallocate
            శూన్య mem_free(స్థిరరాశి శూన్య *ptr)
            {
                ఒకవేళ (!ptr)
                    తిరిగి;
                std::lock_guard<std::mutex> తాళం(m_mutex);
                స్వయంచాలక it = get_map_iterator(ptr);
                m_map.erase(it);
            }

            /// map: device pointer -> allocation(buffer, alloc_ptr, size)
            కేటాయింపు translate_ptr(స్థిరరాశి శూన్య *ptr)
            {
                std::lock_guard<std::mutex> తాళం(m_mutex);
                స్వయంచాలక it = get_map_iterator(ptr);
                తిరిగి it->second;
            }

            /// Check if the pointer represents device pointer or not.
            తార్కిక is_device_ptr(స్థిరరాశి శూన్య *ptr) స్థిరరాశి
            {
                std::lock_guard<std::mutex> తాళం(m_mutex);
                తిరిగి (mapped_చిరునామా_space <= ptr) &&
                       (ptr < mapped_చిరునామా_space + mapped_region_కొలత);
            }

            /// Returns the instance of memory manager singleton.
            స్థిర mem_mgr &instance()
            {
                స్థిర mem_mgr m;
                తిరిగి m;
            }

        అంతరంగ:
            std::map<byte_t *, కేటాయింపు> m_map;
            మార్చదగిన std::mutex m_mutex;
            byte_t *mapped_చిరునామా_space;
            byte_t *తదుపరి_free;
            స్థిరరాశి size_t mapped_region_కొలత = 128ull * 1024 * 1024 * 1024;
            స్థిరరాశి size_t alignment = 256;
            /// This padding may be defined to some positive value to debug
            /// out of bound accesses.
            స్థిరరాశి size_t extra_padding = 0;

            std::map<byte_t *, కేటాయింపు>::iterator get_map_iterator(స్థిరరాశి శూన్య *ptr)
            {
                స్వయంచాలక it = m_map.upper_bound(const_cast<byte_t *>(reinterpret_cast<స్థిరరాశి byte_t *>(ptr)));
                ఒకవేళ (it == m_map.end())
                {
                    // Not a virtual pointer.
                    విసురు std::runtime_error("can not get buffer from non-virtual pointer");
                }
                స్థిరరాశి కేటాయింపు &alloc = it->second;
                ఒకవేళ (ptr < alloc.alloc_ptr)
                {
                    // Out of bound.
                    // This may happen if there's a gap between allocations due to alignment
                    // or extra padding and pointer points to this gap.
                    విసురు std::runtime_error("invalid virtual pointer");
                }
                తిరిగి it;
            }
        };

        మూస <తరగతి T, జ్ఞాపకం_region జ్ఞాపకం_2, size_t Dimension>
        తరగతి accessor;
        మూస <జ్ఞాపకం_region జ్ఞాపకం_2, తరగతి T = byte_t>
        తరగతి జ్ఞాపకం_traits
        {
        బహిరంగ:
            స్థిర constexpr sycl::access::లక్ష్యం లక్ష్యం =
                sycl::access::లక్ష్యం::device;
            స్థిర constexpr sycl::access_బహుళకం mode =
                (జ్ఞాపకం_2 == constant) ? sycl::access_బహుళకం::పఠ
                                     : sycl::access_బహుళకం::read_write;
            స్థిర constexpr size_t type_కొలత = పరిమాణం(T);
            ఉపయోగించి element_t =
                రకం_పేరు std::conditional<జ్ఞాపకం_2 == constant, స్థిరరాశి T, T>::type;
            ఉపయోగించి విలువ_t = రకం_పేరు std::remove_cv<T>::type;
            మూస <size_t Dimension = 1>
            ఉపయోగించి accessor_t = రకం_పేరు std::conditional<
                జ్ఞాపకం_2 == local, sycl::local_accessor<విలువ_t, Dimension>,
                sycl::accessor<T, Dimension, mode, లక్ష్యం>>::type;
            ఉపయోగించి సూచిక_t = T *;
        };

        స్థిర అంతఃవరుస శూన్య *dpct_malloc(size_t size, sycl::queue &q)
        {
            తిరిగి sycl::malloc_device(size, q.get_device(), q.get_context());
        }

#నిర్వచించు PITCH_DEFAULT_ALIGN(x) (((x) + 31) & ~(0x1F))
        స్థిర అంతఃవరుస శూన్య *dpct_malloc(size_t &pitch, size_t x, size_t y, size_t z,
                                        sycl::queue &q)
        {
            pitch = PITCH_DEFAULT_ALIGN(x);
            తిరిగి dpct_malloc(pitch * y * z, q);
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
        మూస <రకం_పేరు విలువ_T>
        స్థిర అంతఃవరుస sycl::event dpct_memset(sycl::queue &q, శూన్య *dev_ptr,
                                              విలువ_T value, size_t size)
        {
            తిరిగి q.fill(dev_ptr, value, size);
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
        మూస <రకం_పేరు విలువ_T>
        స్థిర అంతఃవరుస std::vector<sycl::event>
        dpct_memset(sycl::queue &q, pitched_దత్తాంశం దత్తాంశం, విలువ_T value,
                    sycl::range<3> size)
        {
            std::vector<sycl::event> event_list;
            size_t ముక్క_2 = దత్తాంశం.get_pitch() * దత్తాంశం.get_y();
            గుర్తులేని అక్షరం *దత్తాంశం_surface = (గుర్తులేని అక్షరం *)దత్తాంశం.get_దత్తాంశం_ptr();
            కొరకు (size_t z = 0; z < size.get(2); ++z)
            {
                గుర్తులేని అక్షరం *దత్తాంశం_ptr = దత్తాంశం_surface;
                కొరకు (size_t y = 0; y < size.get(1); ++y)
                {
                    event_list.push_back(dpct_memset(q, దత్తాంశం_ptr, value, size.get(0)));
                    దత్తాంశం_ptr += దత్తాంశం.get_pitch();
                }
                దత్తాంశం_surface += ముక్క_2;
            }
            తిరిగి event_list;
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
        మూస <రకం_పేరు విలువ_T>
        స్థిర అంతఃవరుస std::vector<sycl::event>
        dpct_memset(sycl::queue &q, శూన్య *ptr, size_t pitch, విలువ_T val, size_t x,
                    size_t y)
        {
            తిరిగి dpct_memset(q, pitched_దత్తాంశం(ptr, pitch, x, 1), val,
                               sycl::range<3>(x, y, 1));
        }

        స్థిర memcpy_direction deduce_memcpy_direction(sycl::queue &q, శూన్య *గమ్యం_ptr,
                                                        స్థిరరాశి శూన్య *from_ptr,
                                                        memcpy_direction dir)
        {
            ఎంపిక (dir)
            {
            స్థితి memcpy_direction::host_గమ్యం_host:
            స్థితి memcpy_direction::host_గమ్యం_device:
            స్థితి memcpy_direction::device_గమ్యం_host:
            స్థితి memcpy_direction::device_గమ్యం_device:
                తిరిగి dir;
            స్థితి memcpy_direction::automatic:
            {
                // table[to_attribute][from_attribute]
                స్థిర స్థిరరాశి memcpy_direction
                    direction_పట్టిక[static_cast<గుర్తులేని>(సూచిక_access_attribute::end)]
                                   [static_cast<గుర్తులేని>(సూచిక_access_attribute::end)] =
                                       {{memcpy_direction::host_గమ్యం_host,
                                         memcpy_direction::device_గమ్యం_host,
                                         memcpy_direction::host_గమ్యం_host},
                                        {memcpy_direction::host_గమ్యం_device,
                                         memcpy_direction::device_గమ్యం_device,
                                         memcpy_direction::device_గమ్యం_device},
                                        {memcpy_direction::host_గమ్యం_host,
                                         memcpy_direction::device_గమ్యం_device,
                                         memcpy_direction::device_గమ్యం_device}};
                తిరిగి direction_పట్టిక[static_cast<గుర్తులేని>(get_సూచిక_attribute(
                    q, గమ్యం_ptr))][static_cast<గుర్తులేని>(get_సూచిక_attribute(q, from_ptr))];
            }
            అప్రమేయ:
                విసురు std::runtime_error("dpct_memcpy: invalid direction value");
            }
        }

        స్థిర sycl::event
        dpct_memcpy(sycl::queue &q, శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr, size_t size,
                    memcpy_direction direction,
                    స్థిరరాశి std::vector<sycl::event> &dep_events = {})
        {
            ఒకవేళ (!size)
                తిరిగి sycl::event{};
            తిరిగి q.జ్ఞాపకం_నకలు(గమ్యం_ptr, from_ptr, size, dep_events);
            DPCT_వాడని(direction);
        }

        // Get actual copy range and make sure it will not exceed range.
        స్థిర అంతఃవరుస size_t get_copy_range(sycl::range<3> size, size_t ముక్క_2,
                                            size_t pitch)
        {
            తిరిగి ముక్క_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
        }

        స్థిర అంతఃవరుస size_t get_స్థానభ్రంశం(sycl::id<3> id, size_t ముక్క_2,
                                        size_t pitch)
        {
            తిరిగి ముక్క_2 * id.get(2) + pitch * id.get(1) + id.get(0);
        }

        /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
        /// and \p from_range to another specified by \p to_ptr and \p to_range.
        స్థిర అంతఃవరుస std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr,
                    sycl::range<3> గమ్యం_range, sycl::range<3> from_range,
                    sycl::id<3> గమ్యం_id, sycl::id<3> from_id,
                    sycl::range<3> size, memcpy_direction direction,
                    స్థిరరాశి std::vector<sycl::event> &dep_events = {})
        {
            // RAII for host pointer
            తరగతి host_బఫరు
            {
                శూన్య *_buf;
                size_t కొలత_2;
                sycl::queue &_q;
                స్థిరరాశి std::vector<sycl::event> &_deps; // free operation depends

            బహిరంగ:
                host_బఫరు(size_t size, sycl::queue &q,
                            స్థిరరాశి std::vector<sycl::event> &deps)
                    : _buf(std::జ్ఞాపకం_ఇవ్వు(size)), కొలత_2(size), _q(q), _deps(deps) {}
                శూన్య *get_ptr() స్థిరరాశి { తిరిగి _buf; }
                size_t get_కొలత() స్థిరరాశి { తిరిగి కొలత_2; }
                ~host_బఫరు()
                {
                    ఒకవేళ (_buf)
                    {
                        _q.submit([&](sycl::handler &cgh)
                                  {
        cgh.depends_on(_deps);
        cgh.host_task([buf = _buf] { std::విడుదల(buf); }); });
                    }
                }
            };
            std::vector<sycl::event> event_list;

            size_t గమ్యం_ముక్క = గమ్యం_range.get(1) * గమ్యం_range.get(0),
                   from_ముక్క = from_range.get(1) * from_range.get(0);
            గుర్తులేని అక్షరం *గమ్యం_surface =
                (గుర్తులేని అక్షరం *)గమ్యం_ptr + get_స్థానభ్రంశం(గమ్యం_id, గమ్యం_ముక్క, గమ్యం_range.get(0));
            స్థిరరాశి గుర్తులేని అక్షరం *from_surface =
                (స్థిరరాశి గుర్తులేని అక్షరం *)from_ptr +
                get_స్థానభ్రంశం(from_id, from_ముక్క, from_range.get(0));

            ఒకవేళ (గమ్యం_ముక్క == from_ముక్క && గమ్యం_ముక్క == size.get(1) * size.get(0))
            {
                తిరిగి {dpct_memcpy(q, గమ్యం_surface, from_surface, గమ్యం_ముక్క * size.get(2),
                                    direction, dep_events)};
            }
            direction = deduce_memcpy_direction(q, గమ్యం_ptr, from_ptr, direction);
            size_t కొలత_ముక్క = size.get(1) * size.get(0);
            ఎంపిక (direction)
            {
            స్థితి host_గమ్యం_host:
                కొరకు (size_t z = 0; z < size.get(2); ++z)
                {
                    గుర్తులేని అక్షరం *గమ్యం_ptr = గమ్యం_surface;
                    స్థిరరాశి గుర్తులేని అక్షరం *from_ptr = from_surface;
                    ఒకవేళ (గమ్యం_range.get(0) == from_range.get(0) &&
                        గమ్యం_range.get(0) == size.get(0))
                    {
                        event_list.push_back(dpct_memcpy(q, గమ్యం_ptr, from_ptr, కొలత_ముక్క,
                                                         direction, dep_events));
                    }
                    లేకపోతే
                    {
                        కొరకు (size_t y = 0; y < size.get(1); ++y)
                        {
                            event_list.push_back(dpct_memcpy(q, గమ్యం_ptr, from_ptr, size.get(0),
                                                             direction, dep_events));
                            గమ్యం_ptr += గమ్యం_range.get(0);
                            from_ptr += from_range.get(0);
                        }
                    }
                    గమ్యం_surface += గమ్యం_ముక్క;
                    from_surface += from_ముక్క;
                }
                ఆపు;
            స్థితి host_గమ్యం_device:
            {
                host_బఫరు buf(get_copy_range(size, గమ్యం_ముక్క, గమ్యం_range.get(0)), q,
                                event_list);
                std::vector<sycl::event> host_events;
                ఒకవేళ (గమ్యం_ముక్క == కొలత_ముక్క)
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events =
                        dpct_memcpy(q, buf.get_ptr(), from_surface, గమ్యం_range, from_range,
                                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                    host_గమ్యం_host, dep_events);
                }
                లేకపోతే
                {
                    // Copy host data to a temp host buffer with the shape of target.
                    host_events = dpct_memcpy(
                        q, buf.get_ptr(), from_surface, గమ్యం_range, from_range,
                        sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_గమ్యం_host,
                        // If has padding data, not sure whether it is useless. So fill temp
                        // buffer with it.
                        std::vector<sycl::event>{
                            dpct_memcpy(q, buf.get_ptr(), గమ్యం_surface, buf.get_కొలత(),
                                        device_గమ్యం_host, dep_events)});
                }
                // Copy from temp host buffer to device with only one submit.
                event_list.push_back(dpct_memcpy(q, గమ్యం_surface, buf.get_ptr(),
                                                 buf.get_కొలత(), host_గమ్యం_device,
                                                 host_events));
                ఆపు;
            }
            స్థితి device_గమ్యం_host:
            {
                host_బఫరు buf(get_copy_range(size, from_ముక్క, from_range.get(0)), q,
                                event_list);
                // Copy from host temp buffer to host target with reshaping.
                event_list = dpct_memcpy(
                    q, గమ్యం_surface, buf.get_ptr(), గమ్యం_range, from_range, sycl::id<3>(0, 0, 0),
                    sycl::id<3>(0, 0, 0), size, host_గమ్యం_host,
                    // Copy from device to temp host buffer with only one submit.
                    std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                         buf.get_కొలత(),
                                                         device_గమ్యం_host, dep_events)});
                ఆపు;
            }
            స్థితి device_గమ్యం_device:
                event_list.push_back(q.submit([&](sycl::handler &cgh){
                cgh.depends_on(dep_events);
                cgh.parallel_for<తరగతి dpct_memcpy_3d_detail>(
                    size,
                    [=](sycl::id<3> id) {
                        గమ్యం_surface[get_స్థానభ్రంశం(id, గమ్యం_ముక్క, గమ్యం_range.get(0))] =
                            from_surface[get_స్థానభ్రంశం(id, from_ముక్క, from_range.get(0))];
                    }); }));
                ఆపు;
            అప్రమేయ:
                విసురు std::runtime_error("dpct_memcpy: invalid direction value");
            }
            తిరిగి event_list;
        }

        /// memcpy 2D/3D matrix specified by pitched_data.
        స్థిర అంతఃవరుస std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, pitched_దత్తాంశం గమ్యం, sycl::id<3> గమ్యం_id,
                    pitched_దత్తాంశం from, sycl::id<3> from_id, sycl::range<3> size,
                    memcpy_direction direction = automatic)
        {
            తిరిగి dpct_memcpy(q, గమ్యం.get_దత్తాంశం_ptr(), from.get_దత్తాంశం_ptr(),
                               sycl::range<3>(గమ్యం.get_pitch(), గమ్యం.get_y(), 1),
                               sycl::range<3>(from.get_pitch(), from.get_y(), 1), గమ్యం_id, from_id,
                               size, direction);
        }

        /// memcpy 2D matrix with pitch.
        స్థిర అంతఃవరుస std::vector<sycl::event>
        dpct_memcpy(sycl::queue &q, శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr,
                    size_t గమ్యం_pitch, size_t from_pitch, size_t x, size_t y,
                    memcpy_direction direction = automatic)
        {
            తిరిగి dpct_memcpy(q, గమ్యం_ptr, from_ptr, sycl::range<3>(గమ్యం_pitch, y, 1),
                               sycl::range<3>(from_pitch, y, 1),
                               sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                               sycl::range<3>(x, y, 1), direction);
        }

        నామాకాశ deprecated
        {

            మూస <రకం_పేరు T, sycl::usm::alloc Alloc_రకం>
            తరగతి usm_allocator
            {
            అంతరంగ:
                ఉపయోగించి Alloc = sycl::usm_allocator<T, Alloc_రకం>;
                Alloc _impl;

            బహిరంగ:
                ఉపయోగించి value_type = రకం_పేరు std::allocator_traits<Alloc>::value_type;
                ఉపయోగించి pointer = రకం_పేరు std::allocator_traits<Alloc>::pointer;
                ఉపయోగించి const_pointer = రకం_పేరు std::allocator_traits<Alloc>::const_pointer;
                ఉపయోగించి void_pointer = రకం_పేరు std::allocator_traits<Alloc>::void_pointer;
                ఉపయోగించి const_void_pointer =
                    రకం_పేరు std::allocator_traits<Alloc>::const_void_pointer;
                ఉపయోగించి reference = రకం_పేరు std::allocator_traits<Alloc>::value_type &;
                ఉపయోగించి const_reference =
                    స్థిరరాశి రకం_పేరు std::allocator_traits<Alloc>::value_type &;
                ఉపయోగించి difference_type =
                    రకం_పేరు std::allocator_traits<Alloc>::difference_type;
                ఉపయోగించి size_type = రకం_పేరు std::allocator_traits<Alloc>::size_type;
                ఉపయోగించి propagate_on_container_copy_assignment = రకం_పేరు std::allocator_traits<
                    Alloc>::propagate_on_container_copy_assignment;
                ఉపయోగించి propagate_on_container_move_assignment = రకం_పేరు std::allocator_traits<
                    Alloc>::propagate_on_container_move_assignment;
                ఉపయోగించి propagate_on_container_swap =
                    రకం_పేరు std::allocator_traits<Alloc>::propagate_on_container_swap;
                ఉపయోగించి is_always_equal =
                    రకం_పేరు std::allocator_traits<Alloc>::is_always_equal;

                మూస <రకం_పేరు U>
                నిర్మాణం rebind
                {
                    రకనిర్వచనం usm_allocator<U, Alloc_రకం> other;
                };

                usm_allocator() : _impl(dpct::get_default_వరుస()) {}
                ~usm_allocator() {}
                usm_allocator(స్థిరరాశి usm_allocator &other) : _impl(other._impl) {}
                usm_allocator(usm_allocator &&other) : _impl(std::move(other._impl)) {}
                pointer address(reference r) { తిరిగి &r; }
                const_pointer address(const_reference r) { తిరిగి &r; }
                pointer allocate(size_type cnt, const_void_pointer hint = శూన్యసూచిక)
                {
                    తిరిగి std::allocator_traits<Alloc>::allocate(_impl, cnt, hint);
                }
                శూన్య deallocate(pointer p, size_type cnt)
                {
                    std::allocator_traits<Alloc>::deallocate(_impl, p, cnt);
                }
                size_type max_size() స్థిరరాశి
                {
                    తిరిగి std::allocator_traits<Alloc>::max_size(_impl);
                }
                తార్కిక క్రియాకారకం==(స్థిరరాశి usm_allocator &other) స్థిరరాశి { తిరిగి _impl == other._impl; }
                తార్కిక క్రియాకారకం!=(స్థిరరాశి usm_allocator &other) స్థిరరాశి { తిరిగి _impl != other._impl; }
            };

        } // namespace deprecated

        అంతఃవరుస శూన్య dpct_free(శూన్య *ptr,
                              స్థిరరాశి sycl::queue &q)
        {
            ఒకవేళ (ptr)
            {
                sycl::విడుదల(ptr, q.get_context());
            }
        }

        మూస <రకం_పేరు T>
        అంతఃవరుస స్వయంచాలక get_జ్ఞాపకం(స్థిరరాశి శూన్య *x)
        {
            T *new_x = reinterpret_cast<T *>(const_cast<శూన్య *>(x));
            తిరిగి new_x;
        }

        మూస <రకం_పేరు T>
        అంతఃవరుస రకం_పేరు దత్తాంశం_Type_2<T>::T2 get_విలువ(స్థిరరాశి T *s, sycl::queue &q)
        {
            ఉపయోగించి Ty = రకం_పేరు దత్తాంశం_Type_2<T>::T2;
            Ty s_h;
            ఒకవేళ (get_సూచిక_attribute(q, s) == సూచిక_access_attribute::device_only)
                detail::dpct_memcpy(q, (శూన్య *)&s_h, (స్థిరరాశి శూన్య *)s, పరిమాణం(T), device_గమ్యం_host)
                    .wait();
            లేకపోతే
                s_h = *reinterpret_cast<స్థిరరాశి Ty *>(s);
            తిరిగి s_h;
        }

    } // namespace detail

    మూస <రకం_పేరు T>
    అంతఃవరుస స్వయంచాలక get_విలువ(స్థిరరాశి T *s, sycl::queue &q)
    {
        తిరిగి detail::get_విలువ(s, q);
    }

    నామాకాశ detail
    {
    మూస <తరగతి Ta, తరగతి Tb, తరగతి Tc, తరగతి Ts>
    అంతఃవరుస శూన్య gemm_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, పూర్ణాంకం m,
                          పూర్ణాంకం n, పూర్ణాంకం k, స్థిరరాశి శూన్య * alpha, స్థిరరాశి శూన్య * a, పూర్ణాంకం lda, స్థిరరాశి శూన్య * b, పూర్ణాంకం ldb,
                          స్థిరరాశి శూన్య * beta, శూన్య * c, పూర్ణాంకం ldc) {
        Ts   alpha_విలువ_2 = dpct::get_విలువ(reinterpret_cast<స్థిరరాశి Ts *>(alpha), q);
        Ts   beta_విలువ  = dpct::get_విలువ(reinterpret_cast<స్థిరరాశి Ts *>(beta), q);
        స్వయంచాలక దత్తాంశం_a      = get_జ్ఞాపకం<స్థిరరాశి Ta>(a);
        స్వయంచాలక దత్తాంశం_b      = get_జ్ఞాపకం<స్థిరరాశి Tb>(b);
        స్వయంచాలక దత్తాంశం_c      = get_జ్ఞాపకం<Tc>(c);
        oneapi::mkl::blas::స్తంభం_major::gemm(q, a_trans, b_trans, m, n, k, alpha_విలువ_2, దత్తాంశం_a,
                                               lda, దత్తాంశం_b, ldb, beta_విలువ, దత్తాంశం_c, ldc);
    }

        మూస <రకం_పేరు VecT, తరగతి ద్విమాన_Operation, తరగతి = శూన్య>
        తరగతి vectorized_ద్విమాన
        {
        బహిరంగ:
            అంతఃవరుస VecT క్రియాకారకం()(VecT a, VecT b, స్థిరరాశి ద్విమాన_Operation ద్విమాన_op)
            {
                VecT v4;
                కొరకు (size_t i = 0; i < v4.size(); ++i)
                {
                    v4[i] = ద్విమాన_op(a[i], b[i]);
                }
                తిరిగి v4;
            }
        };

        మూస <రకం_పేరు VecT, తరగతి ద్విమాన_Operation>
        తరగతి vectorized_ద్విమాన<
            VecT, ద్విమాన_Operation,
            std::void_t<std::invoke_result_t<ద్విమాన_Operation, VecT, VecT>>>
        {
        బహిరంగ:
            అంతఃవరుస VecT క్రియాకారకం()(VecT a, VecT b, స్థిరరాశి ద్విమాన_Operation ద్విమాన_op)
            {
                తిరిగి ద్విమాన_op(a, b).మూస as<VecT>();
            }
        };

        మూస <తరగతి Ta, తరగతి Tb, తరగతి Tc, తరగతి Ts>
        అంతఃవరుస శూన్య gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    పూర్ణాంకం m, పూర్ణాంకం n, పూర్ణాంకం k, స్థిరరాశి శూన్య * alpha, స్థిరరాశి శూన్య ** a, పూర్ణాంకం lda, స్థిరరాశి శూన్య ** b,
                                    పూర్ణాంకం ldb, స్థిరరాశి శూన్య * beta, శూన్య ** c, పూర్ణాంకం ldc, పూర్ణాంకం batch_కొలత,
                                    మాతృక_info_t<దశాంశం> * మాతృక_info) {
            Ts alpha_విలువ_2 = dpct::get_విలువ(reinterpret_cast<స్థిరరాశి Ts *>(alpha), q);
            Ts beta_విలువ = dpct::get_విలువ(reinterpret_cast<స్థిరరాశి Ts *>(beta), q);

            మాతృక_info->transpose_info[0] = a_trans;
            మాతృక_info->transpose_info[1] = b_trans;
            మాతృక_info->విలువ_info[0] = alpha_విలువ_2;
            మాతృక_info->విలువ_info[1] = beta_విలువ;
            మాతృక_info->కొలత_info[0] = m;
            మాతృక_info->కొలత_info[1] = n;
            మాతృక_info->కొలత_info[2] = k;
            మాతృక_info->ld_info[0] = lda;
            మాతృక_info->ld_info[1] = ldb;
            మాతృక_info->ld_info[2] = ldc;
            మాతృక_info->groupsize_info = batch_కొలత;

            sycl::event e = oneapi::mkl::blas::స్తంభం_major::gemm_batch(
                q, మాతృక_info->transpose_info, మాతృక_info->transpose_info + 1,
                మాతృక_info->కొలత_info, మాతృక_info->కొలత_info + 1, మాతృక_info->కొలత_info + 2,
                reinterpret_cast<Ts *>(మాతృక_info->విలువ_info), reinterpret_cast<స్థిరరాశి Ta **>(a), మాతృక_info->ld_info,
                reinterpret_cast<స్థిరరాశి Tb **>(b), మాతృక_info->ld_info + 1,
                reinterpret_cast<Ts *>(మాతృక_info->విలువ_info + 1), reinterpret_cast<Tc **>(c),
                మాతృక_info->ld_info + 2, 1, &(మాతృక_info->groupsize_info));
        }

        మూస <తరగతి Ta, తరగతి Tb, తరగతి Tc, తరగతి Ts>
        అంతఃవరుస శూన్య gemm_batch_impl(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans,
                                    పూర్ణాంకం m, పూర్ణాంకం n, పూర్ణాంకం k, స్థిరరాశి శూన్య * alpha, స్థిరరాశి శూన్య * a, పూర్ణాంకం lda,
                                    దీర్ఘ దీర్ఘ పూర్ణాంకం stride_a, స్థిరరాశి శూన్య * b, పూర్ణాంకం ldb, దీర్ఘ దీర్ఘ పూర్ణాంకం stride_b,
                                    స్థిరరాశి శూన్య * beta, శూన్య * c, పూర్ణాంకం ldc, దీర్ఘ దీర్ఘ పూర్ణాంకం stride_c, పూర్ణాంకం batch_కొలత) {
            Ts alpha_విలువ_2 = dpct::get_విలువ(reinterpret_cast<స్థిరరాశి Ts *>(alpha), q);
            Ts beta_విలువ = dpct::get_విలువ(reinterpret_cast<స్థిరరాశి Ts *>(beta), q);
            స్వయంచాలక దత్తాంశం_a = get_జ్ఞాపకం<స్థిరరాశి Ta>(a);
            స్వయంచాలక దత్తాంశం_b = get_జ్ఞాపకం<స్థిరరాశి Tb>(b);
            స్వయంచాలక దత్తాంశం_c = get_జ్ఞాపకం<Tc>(c);
            oneapi::mkl::blas::స్తంభం_major::gemm_batch(q, a_trans, b_trans, m, n, k, alpha_విలువ_2,
                                                         దత్తాంశం_a, lda, stride_a, దత్తాంశం_b, ldb, stride_b, beta_విలువ,
                                                         దత్తాంశం_c, ldc, stride_c, batch_కొలత);
        }

    } // namespace detail

    మూస <రకం_పేరు VecT, తరగతి ద్విమాన_Operation>
    అంతఃవరుస గుర్తులేని vectorized_ద్విమాన(గుర్తులేని a, గుర్తులేని b,
                                      స్థిరరాశి ద్విమాన_Operation ద్విమాన_op)
    {
        sycl::vec<గుర్తులేని, 1> v0{a}, v1{b};
        స్వయంచాలక v2 = v0.as<VecT>();
        స్వయంచాలక v3 = v1.as<VecT>();
        స్వయంచాలక v4 =
            detail::vectorized_ద్విమాన<VecT, ద్విమాన_Operation>()(v2, v3, ద్విమాన_op);
        v0 = v4.మూస as<sycl::vec<గుర్తులేని, 1>>();
        తిరిగి v0;
    }

    స్థిర శూన్య async_dpct_memcpy(శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr, size_t size,
                                  memcpy_direction direction = automatic,
                                  sycl::queue &q = dpct::get_default_వరుస())
    {
        detail::dpct_memcpy(q, గమ్యం_ptr, from_ptr, size, direction);
    }

    స్థిర అంతఃవరుస గుర్తులేని పూర్ణాంకం select_device(గుర్తులేని పూర్ణాంకం id)
    {
        dev_mgr::instance().select_device(id);
        తిరిగి id;
    }

    మూస <రకం_పేరు T>
    T క్రమమార్పు_sub_సమూహం_by_xor(sycl::sub_సమూహం g, T x, గుర్తులేని పూర్ణాంకం ముసుగు,
                               గుర్తులేని పూర్ణాంకం logical_sub_సమూహం_కొలత = 32)
    {
        గుర్తులేని పూర్ణాంకం id = g.get_local_రేఖీయ_id();
        గుర్తులేని పూర్ణాంకం ప్రారంభం_సూచీ =
            id / logical_sub_సమూహం_కొలత * logical_sub_సమూహం_కొలత;
        గుర్తులేని పూర్ణాంకం లక్ష్యం_స్థానభ్రంశం = (id % logical_sub_సమూహం_కొలత) ^ ముసుగు;
        తిరిగి sycl::select_from_సమూహం(g, x,
                                       లక్ష్యం_స్థానభ్రంశం < logical_sub_సమూహం_కొలత
                                           ? ప్రారంభం_సూచీ + లక్ష్యం_స్థానభ్రంశం
                                           : id);
    }

    మూస <రకం_పేరు T1, రకం_పేరు T2>
    ఉపయోగించి dot_లబ్ధం_acc_t = std::conditional_t<
        std::is_unsigned_v<T1> && std::is_unsigned_v<T2>,
        uint32_t,
        int32_t>;

    మూస <రకం_పేరు T>
    sycl::vec<T, 4> వెలికితీయి_and_sign_or_zero_extend4(T val) {
      తిరిగి sycl::vec<T, 1>(val)
          .మూస as<sycl::vec<
              std::conditional_t<std::is_signed_v<T>, int8_t, uint8_t>,
              4>>()
          .మూస convert<T>();
    }

    మూస <రకం_పేరు T1, రకం_పేరు T2, రకం_పేరు T3>
    అంతఃవరుస స్వయంచాలక dp4a(T1 a, T2 b, T3 c) {
      dot_లబ్ధం_acc_t<T1, T2> res = c;
      స్వయంచాలక va = వెలికితీయి_and_sign_or_zero_extend4(a);
      స్వయంచాలక vb = వెలికితీయి_and_sign_or_zero_extend4(b);
      res += va[0] * vb[0];
      res += va[1] * vb[1];
      res += va[2] * vb[2];
      res += va[3] * vb[3];
      తిరిగి res;
    }

    నిర్మాణం sub_sat
    {
        మూస <రకం_పేరు T>
        స్వయంచాలక క్రియాకారకం()(స్థిరరాశి T x, స్థిరరాశి T y) స్థిరరాశి
        {
            తిరిగి sycl::sub_sat(x, y);
        }
    };

    మూస <రకం_పేరు S, రకం_పేరు T>
    అంతఃవరుస T vectorized_min(T a, T b)
    {
        sycl::vec<T, 1> v0{a}, v1{b};
        స్వయంచాలక v2 = v0.మూస as<S>();
        స్వయంచాలక v3 = v1.మూస as<S>();
        స్వయంచాలక v4 = sycl::min(v2, v3);
        v0 = v4.మూస as<sycl::vec<T, 1>>();
        తిరిగి v0;
    }

    అంతఃవరుస దశాంశం ఘాతం(స్థిరరాశి దశాంశం a, స్థిరరాశి పూర్ణాంకం b) { తిరిగి sycl::pown(a, b); }
    అంతఃవరుస ద్విగుణ ఘాతం(స్థిరరాశి ద్విగుణ a, స్థిరరాశి పూర్ణాంకం b) { తిరిగి sycl::pown(a, b); }
    అంతఃవరుస దశాంశం ఘాతం(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) { తిరిగి sycl::ఘాతం(a, b); }
    అంతఃవరుస ద్విగుణ ఘాతం(స్థిరరాశి ద్విగుణ a, స్థిరరాశి ద్విగుణ b) { తిరిగి sycl::ఘాతం(a, b); }
    మూస <రకం_పేరు T, రకం_పేరు U>
    అంతఃవరుస రకం_పేరు std::enable_if_t<std::is_floating_point_v<T>, T>
    ఘాతం(స్థిరరాశి T a, స్థిరరాశి U b)
    {
        తిరిగి sycl::ఘాతం(a, static_cast<T>(b));
    }
    మూస <రకం_పేరు T, రకం_పేరు U>
    అంతఃవరుస రకం_పేరు std::enable_if_t<!std::is_floating_point_v<T>, ద్విగుణ>
    ఘాతం(స్థిరరాశి T a, స్థిరరాశి U b)
    {
        తిరిగి sycl::ఘాతం(static_cast<ద్విగుణ>(a), static_cast<ద్విగుణ>(b));
    }

    అంతఃవరుస ద్విగుణ min(స్థిరరాశి ద్విగుణ a, స్థిరరాశి దశాంశం b)
    {
        తిరిగి sycl::fmin(a, static_cast<ద్విగుణ>(b));
    }
    అంతఃవరుస ద్విగుణ min(స్థిరరాశి దశాంశం a, స్థిరరాశి ద్విగుణ b)
    {
        తిరిగి sycl::fmin(static_cast<ద్విగుణ>(a), b);
    }
    అంతఃవరుస దశాంశం min(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) { తిరిగి sycl::fmin(a, b); }
    అంతఃవరుస ద్విగుణ min(స్థిరరాశి ద్విగుణ a, స్థిరరాశి ద్విగుణ b) { తిరిగి sycl::fmin(a, b); }
    అంతఃవరుస std::uint32_t min(స్థిరరాశి std::uint32_t a, స్థిరరాశి std::int32_t b)
    {
        తిరిగి sycl::min(a, static_cast<std::uint32_t>(b));
    }
    అంతఃవరుస std::uint32_t min(స్థిరరాశి std::int32_t a, స్థిరరాశి std::uint32_t b)
    {
        తిరిగి sycl::min(static_cast<std::uint32_t>(a), b);
    }
    అంతఃవరుస std::int32_t min(స్థిరరాశి std::int32_t a, స్థిరరాశి std::int32_t b)
    {
        తిరిగి sycl::min(a, b);
    }
    అంతఃవరుస std::uint32_t min(స్థిరరాశి std::uint32_t a, స్థిరరాశి std::uint32_t b)
    {
        తిరిగి sycl::min(a, b);
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::int64_t b)
    {
        తిరిగి sycl::min(a, static_cast<std::uint64_t>(b));
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::int64_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::min(static_cast<std::uint64_t>(a), b);
    }
    అంతఃవరుస std::int64_t min(స్థిరరాశి std::int64_t a, స్థిరరాశి std::int64_t b)
    {
        తిరిగి sycl::min(a, b);
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::min(a, b);
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::int32_t b)
    {
        తిరిగి sycl::min(a, static_cast<std::uint64_t>(b));
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::int32_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::min(static_cast<std::uint64_t>(a), b);
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::uint32_t b)
    {
        తిరిగి sycl::min(a, static_cast<std::uint64_t>(b));
    }
    అంతఃవరుస std::uint64_t min(స్థిరరాశి std::uint32_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::min(static_cast<std::uint64_t>(a), b);
    }
    // max function overloads.
    // For floating-point types, `float` or `double` arguments are acceptable.
    // For integer types, `std::uint32_t`, `std::int32_t`, `std::uint64_t` or
    // `std::int64_t` type arguments are acceptable.
    అంతఃవరుస ద్విగుణ max(స్థిరరాశి ద్విగుణ a, స్థిరరాశి దశాంశం b)
    {
        తిరిగి sycl::fmax(a, static_cast<ద్విగుణ>(b));
    }
    అంతఃవరుస ద్విగుణ max(స్థిరరాశి దశాంశం a, స్థిరరాశి ద్విగుణ b)
    {
        తిరిగి sycl::fmax(static_cast<ద్విగుణ>(a), b);
    }
    అంతఃవరుస దశాంశం max(స్థిరరాశి దశాంశం a, స్థిరరాశి దశాంశం b) { తిరిగి sycl::fmax(a, b); }
    అంతఃవరుస ద్విగుణ max(స్థిరరాశి ద్విగుణ a, స్థిరరాశి ద్విగుణ b) { తిరిగి sycl::fmax(a, b); }
    అంతఃవరుస std::uint32_t max(స్థిరరాశి std::uint32_t a, స్థిరరాశి std::int32_t b)
    {
        తిరిగి sycl::max(a, static_cast<std::uint32_t>(b));
    }
    అంతఃవరుస std::uint32_t max(స్థిరరాశి std::int32_t a, స్థిరరాశి std::uint32_t b)
    {
        తిరిగి sycl::max(static_cast<std::uint32_t>(a), b);
    }
    అంతఃవరుస std::int32_t max(స్థిరరాశి std::int32_t a, స్థిరరాశి std::int32_t b)
    {
        తిరిగి sycl::max(a, b);
    }
    అంతఃవరుస std::uint32_t max(స్థిరరాశి std::uint32_t a, స్థిరరాశి std::uint32_t b)
    {
        తిరిగి sycl::max(a, b);
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::int64_t b)
    {
        తిరిగి sycl::max(a, static_cast<std::uint64_t>(b));
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::int64_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::max(static_cast<std::uint64_t>(a), b);
    }
    అంతఃవరుస std::int64_t max(స్థిరరాశి std::int64_t a, స్థిరరాశి std::int64_t b)
    {
        తిరిగి sycl::max(a, b);
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::max(a, b);
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::int32_t b)
    {
        తిరిగి sycl::max(a, static_cast<std::uint64_t>(b));
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::int32_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::max(static_cast<std::uint64_t>(a), b);
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::uint64_t a, స్థిరరాశి std::uint32_t b)
    {
        తిరిగి sycl::max(a, static_cast<std::uint64_t>(b));
    }
    అంతఃవరుస std::uint64_t max(స్థిరరాశి std::uint32_t a, స్థిరరాశి std::uint64_t b)
    {
        తిరిగి sycl::max(static_cast<std::uint64_t>(a), b);
    }

    అంతఃవరుస శూన్య
    has_capability_or_fail(స్థిరరాశి sycl::device &dev,
                           స్థిరరాశి std::initializer_list<sycl::aspect> &props)
    {
        కొరకు (స్థిరరాశి స్వయంచాలక &it : props)
        {
            ఒకవేళ (dev.has(it))
                కొనసాగు;
            ఎంపిక (it)
            {
            స్థితి sycl::aspect::fp64:
                విసురు std::runtime_error("'double' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ఆపు;
            స్థితి sycl::aspect::fp16:
                విసురు std::runtime_error("'half' is not supported in '" +
                                         dev.get_info<sycl::info::device::name>() +
                                         "' device");
                ఆపు;
            అప్రమేయ:
#నిర్వచించు __SYCL_ASPECT(ASPECT, ID) \
    స్థితి sycl::aspect::ASPECT:    \
        తిరిగి #ASPECT;
#నిర్వచించు __SYCL_ASPECT_DEPRECATED(ASPECT, ID, సందేశం_3) __SYCL_ASPECT(ASPECT, ID)
#నిర్వచించు __SYCL_ASPECT_DEPRECATED_ALIAS(ASPECT, ID, సందేశం_3)
                స్వయంచాలక getAspectNameStr = [](sycl::aspect Aspect_అంకె) -> std::string
                {
                    ఎంపిక (Aspect_అంకె)
                    {
#అన్తర్భావ <sycl/info/aspects.def>
#అన్తర్భావ <sycl/info/aspects_deprecated.def>
                    అప్రమేయ:
                        తిరిగి "unknown aspect";
                    }
                };
#నిర్వచనం_రద్దు __SYCL_ASPECT_DEPRECATED_ALIAS
#నిర్వచనం_రద్దు __SYCL_ASPECT_DEPRECATED
#నిర్వచనం_రద్దు __SYCL_ASPECT
                విసురు std::runtime_error(
                    "'" + getAspectNameStr(it) + "' is not supported in '" +
                    dev.get_info<sycl::info::device::name>() + "' device");
            }
            ఆపు;
        }
    }

    స్థిర అంతఃవరుస గుర్తులేని పూర్ణాంకం get_current_device_id()
    {
        తిరిగి dev_mgr::instance().current_device_id();
    }

    స్థిర అంతఃవరుస device_ext &get_current_device()
    {
        తిరిగి dev_mgr::instance().current_device();
    }

    స్థిర అంతఃవరుస device_ext &get_device(గుర్తులేని పూర్ణాంకం id)
    {
        తిరిగి dev_mgr::instance().get_device(id);
    }

    స్థిర అంతఃవరుస sycl::queue &get_లోపల_క్రమం_వరుస()
    {
        తిరిగి dev_mgr::instance().current_device().లోపల_క్రమం_వరుస();
    }

    స్థిర sycl::event
    dpct_memcpy(sycl::queue &q, శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr, size_t size,
                memcpy_direction direction,
                స్థిరరాశి std::vector<sycl::event> &dep_events = {})
    {
        ఒకవేళ (!size)
            తిరిగి sycl::event{};
        తిరిగి q.జ్ఞాపకం_నకలు(గమ్యం_ptr, from_ptr, size, dep_events);
        DPCT_వాడని(direction);
    }

    // Get actual copy range and make sure it will not exceed range.
    స్థిర అంతఃవరుస size_t get_copy_range(sycl::range<3> size, size_t ముక్క_2,
                                        size_t pitch)
    {
        తిరిగి ముక్క_2 * (size.get(2) - 1) + pitch * (size.get(1) - 1) + size.get(0);
    }

    స్థిర అంతఃవరుస size_t get_స్థానభ్రంశం(sycl::id<3> id, size_t ముక్క_2,
                                    size_t pitch)
    {
        తిరిగి ముక్క_2 * id.get(2) + pitch * id.get(1) + id.get(0);
    }

    /// copy 3D matrix specified by \p size from 3D matrix specified by \p from_ptr
    /// and \p from_range to another specified by \p to_ptr and \p to_range.
    స్థిర అంతఃవరుస std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr,
                sycl::range<3> గమ్యం_range, sycl::range<3> from_range,
                sycl::id<3> గమ్యం_id, sycl::id<3> from_id,
                sycl::range<3> size, memcpy_direction direction,
                స్థిరరాశి std::vector<sycl::event> &dep_events = {})
    {
        // RAII for host pointer
        తరగతి host_బఫరు
        {
            శూన్య *_buf;
            size_t కొలత_2;
            sycl::queue &_q;
            స్థిరరాశి std::vector<sycl::event> &_deps; // free operation depends

        బహిరంగ:
            host_బఫరు(size_t size, sycl::queue &q,
                        స్థిరరాశి std::vector<sycl::event> &deps)
                : _buf(std::జ్ఞాపకం_ఇవ్వు(size)), కొలత_2(size), _q(q), _deps(deps) {}
            శూన్య *get_ptr() స్థిరరాశి { తిరిగి _buf; }
            size_t get_కొలత() స్థిరరాశి { తిరిగి కొలత_2; }
            ~host_బఫరు()
            {
                ఒకవేళ (_buf)
                {
                    _q.submit([&](sycl::handler &cgh)
                              {
            cgh.depends_on(_deps);
            cgh.host_task([buf = _buf] { std::విడుదల(buf); }); });
                }
            }
        };
        std::vector<sycl::event> event_list;

        size_t గమ్యం_ముక్క = గమ్యం_range.get(1) * గమ్యం_range.get(0),
               from_ముక్క = from_range.get(1) * from_range.get(0);
        గుర్తులేని అక్షరం *గమ్యం_surface =
            (గుర్తులేని అక్షరం *)గమ్యం_ptr + get_స్థానభ్రంశం(గమ్యం_id, గమ్యం_ముక్క, గమ్యం_range.get(0));
        స్థిరరాశి గుర్తులేని అక్షరం *from_surface =
            (స్థిరరాశి గుర్తులేని అక్షరం *)from_ptr +
            get_స్థానభ్రంశం(from_id, from_ముక్క, from_range.get(0));

        ఒకవేళ (గమ్యం_ముక్క == from_ముక్క && గమ్యం_ముక్క == size.get(1) * size.get(0))
        {
            తిరిగి {dpct_memcpy(q, గమ్యం_surface, from_surface, గమ్యం_ముక్క * size.get(2),
                                direction, dep_events)};
        }
        direction = detail::deduce_memcpy_direction(q, గమ్యం_ptr, from_ptr, direction);
        size_t కొలత_ముక్క = size.get(1) * size.get(0);
        ఎంపిక (direction)
        {
        స్థితి host_గమ్యం_host:
            కొరకు (size_t z = 0; z < size.get(2); ++z)
            {
                గుర్తులేని అక్షరం *గమ్యం_ptr = గమ్యం_surface;
                స్థిరరాశి గుర్తులేని అక్షరం *from_ptr = from_surface;
                ఒకవేళ (గమ్యం_range.get(0) == from_range.get(0) &&
                    గమ్యం_range.get(0) == size.get(0))
                {
                    event_list.push_back(dpct_memcpy(q, గమ్యం_ptr, from_ptr, కొలత_ముక్క,
                                                     direction, dep_events));
                }
                లేకపోతే
                {
                    కొరకు (size_t y = 0; y < size.get(1); ++y)
                    {
                        event_list.push_back(dpct_memcpy(q, గమ్యం_ptr, from_ptr, size.get(0),
                                                         direction, dep_events));
                        గమ్యం_ptr += గమ్యం_range.get(0);
                        from_ptr += from_range.get(0);
                    }
                }
                గమ్యం_surface += గమ్యం_ముక్క;
                from_surface += from_ముక్క;
            }
            ఆపు;
        స్థితి host_గమ్యం_device:
        {
            host_బఫరు buf(get_copy_range(size, గమ్యం_ముక్క, గమ్యం_range.get(0)), q,
                            event_list);
            std::vector<sycl::event> host_events;
            ఒకవేళ (గమ్యం_ముక్క == కొలత_ముక్క)
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events =
                    dpct_memcpy(q, buf.get_ptr(), from_surface, గమ్యం_range, from_range,
                                sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size,
                                host_గమ్యం_host, dep_events);
            }
            లేకపోతే
            {
                // Copy host data to a temp host buffer with the shape of target.
                host_events = dpct_memcpy(
                    q, buf.get_ptr(), from_surface, గమ్యం_range, from_range,
                    sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0), size, host_గమ్యం_host,
                    // If has padding data, not sure whether it is useless. So fill temp
                    // buffer with it.
                    std::vector<sycl::event>{
                        dpct_memcpy(q, buf.get_ptr(), గమ్యం_surface, buf.get_కొలత(),
                                    device_గమ్యం_host, dep_events)});
            }
            // Copy from temp host buffer to device with only one submit.
            event_list.push_back(dpct_memcpy(q, గమ్యం_surface, buf.get_ptr(),
                                             buf.get_కొలత(), host_గమ్యం_device,
                                             host_events));
            ఆపు;
        }
        స్థితి device_గమ్యం_host:
        {
            host_బఫరు buf(get_copy_range(size, from_ముక్క, from_range.get(0)), q,
                            event_list);
            // Copy from host temp buffer to host target with reshaping.
            event_list = dpct_memcpy(
                q, గమ్యం_surface, buf.get_ptr(), గమ్యం_range, from_range, sycl::id<3>(0, 0, 0),
                sycl::id<3>(0, 0, 0), size, host_గమ్యం_host,
                // Copy from device to temp host buffer with only one submit.
                std::vector<sycl::event>{dpct_memcpy(q, buf.get_ptr(), from_surface,
                                                     buf.get_కొలత(),
                                                     device_గమ్యం_host, dep_events)});
            ఆపు;
        }
        స్థితి device_గమ్యం_device:
            event_list.push_back(q.submit([&](sycl::handler &cgh)
                                          {
        cgh.depends_on(dep_events);
        cgh.parallel_for<తరగతి dpct_memcpy_3d_detail>(
            size,
            [=](sycl::id<3> id) {
                గమ్యం_surface[get_స్థానభ్రంశం(id, గమ్యం_ముక్క, గమ్యం_range.get(0))] =
                    from_surface[get_స్థానభ్రంశం(id, from_ముక్క, from_range.get(0))];
            }); }));
        ఆపు;
        అప్రమేయ:
            విసురు std::runtime_error("dpct_memcpy: invalid direction value");
        }
        తిరిగి event_list;
    }

    /// memcpy 2D/3D matrix specified by pitched_data.
    స్థిర అంతఃవరుస std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, pitched_దత్తాంశం గమ్యం, sycl::id<3> గమ్యం_id,
                pitched_దత్తాంశం from, sycl::id<3> from_id, sycl::range<3> size,
                memcpy_direction direction = automatic)
    {
        తిరిగి dpct_memcpy(q, గమ్యం.get_దత్తాంశం_ptr(), from.get_దత్తాంశం_ptr(),
                           sycl::range<3>(గమ్యం.get_pitch(), గమ్యం.get_y(), 1),
                           sycl::range<3>(from.get_pitch(), from.get_y(), 1), గమ్యం_id, from_id,
                           size, direction);
    }

    /// memcpy 2D matrix with pitch.
    స్థిర అంతఃవరుస std::vector<sycl::event>
    dpct_memcpy(sycl::queue &q, శూన్య *గమ్యం_ptr, స్థిరరాశి శూన్య *from_ptr,
                size_t గమ్యం_pitch, size_t from_pitch, size_t x, size_t y,
                memcpy_direction direction = automatic)
    {
        తిరిగి dpct_memcpy(q, గమ్యం_ptr, from_ptr, sycl::range<3>(గమ్యం_pitch, y, 1),
                           sycl::range<3>(from_pitch, y, 1),
                           sycl::id<3>(0, 0, 0), sycl::id<3>(0, 0, 0),
                           sycl::range<3>(x, y, 1), direction);
    }

    అంతఃవరుస శూన్య gemm(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, పూర్ణాంకం m, పూర్ణాంకం n,
                     పూర్ణాంకం k, స్థిరరాశి శూన్య * alpha, స్థిరరాశి శూన్య * a, library_దత్తాంశం_t a_type, పూర్ణాంకం lda, స్థిరరాశి శూన్య * b,
                     library_దత్తాంశం_t b_type, పూర్ణాంకం ldb, స్థిరరాశి శూన్య * beta, శూన్య * c, library_దత్తాంశం_t c_type, పూర్ణాంకం ldc,
                     library_దత్తాంశం_t scaling_type) {
        ఒకవేళ (scaling_type == library_దత్తాంశం_t::real_float &&
            c_type == library_దత్తాంశం_t::complex_float)
        {
            scaling_type = library_దత్తాంశం_t::complex_float;
        }
        లేకపోతే ఒకవేళ (scaling_type == library_దత్తాంశం_t::real_double &&
                 c_type == library_దత్తాంశం_t::complex_double)
        {
            scaling_type = library_దత్తాంశం_t::complex_double;
        }

        std::uint64_t కీలకం =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ఎంపిక (కీలకం)
        {
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_impl<దశాంశం, దశాంశం, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_double, library_దత్తాంశం_t::real_double,
            library_దత్తాంశం_t::real_double, library_దత్తాంశం_t::real_double):
        {
            detail::gemm_impl<ద్విగుణ, ద్విగుణ, ద్విగుణ, ద్విగుణ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::complex_float, library_దత్తాంశం_t::complex_float,
            library_దత్తాంశం_t::complex_float, library_దత్తాంశం_t::complex_float):
        {
            detail::gemm_impl<std::complex<దశాంశం>, std::complex<దశాంశం>,
                              std::complex<దశాంశం>, std::complex<దశాంశం>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::complex_double, library_దత్తాంశం_t::complex_double,
            library_దత్తాంశం_t::complex_double, library_దత్తాంశం_t::complex_double):
        {
            detail::gemm_impl<std::complex<ద్విగుణ>, std::complex<ద్విగుణ>,
                              std::complex<ద్విగుణ>, std::complex<ద్విగుణ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half):
        {
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, alpha, a,
                                          lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
#నిర్వచితమైతే __INTEL_MKL__
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_bfloat16,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_impl<sycl::half, sycl::half, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_float):
        {
            దశాంశం alpha_విలువ_2 =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి దశాంశం *>(alpha), q);
            దశాంశం beta_విలువ =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి దశాంశం *>(beta), q);
            sycl::half alpha_half(alpha_విలువ_2);
            sycl::half beta_half(beta_విలువ);
            detail::gemm_impl<sycl::half, sycl::half, sycl::half,
                              sycl::half>(q, a_trans, b_trans, m, n, k, &alpha_half,
                                          a, lda, b, ldb, &beta_half, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_int8, library_దత్తాంశం_t::real_int8,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_impl<std::int8_t, std::int8_t, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_bfloat16,
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_int8, library_దత్తాంశం_t::real_int8,
            library_దత్తాంశం_t::real_int32, library_దత్తాంశం_t::real_int32):
        {
            దశాంశం alpha_float =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి std::int32_t *>(alpha), q);
            దశాంశం beta_float =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి std::int32_t *>(beta), q);
            detail::gemm_impl<std::int8_t, std::int8_t, std::int32_t, దశాంశం>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc);
            ఆపు;
        }
#అయితే_ముగింపు // __INTEL_MKL__
        అప్రమేయ:
            విసురు std::runtime_error("the combination of data type is unsupported");
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
    అంతఃవరుస శూన్య gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, పూర్ణాంకం m,
                           పూర్ణాంకం n, పూర్ణాంకం k, స్థిరరాశి శూన్య * alpha, స్థిరరాశి శూన్య * a[], library_దత్తాంశం_t a_type, పూర్ణాంకం lda,
                           స్థిరరాశి శూన్య * b[], library_దత్తాంశం_t b_type, పూర్ణాంకం ldb, స్థిరరాశి శూన్య * beta, శూన్య * c[],
                           library_దత్తాంశం_t c_type, పూర్ణాంకం ldc, పూర్ణాంకం batch_కొలత, library_దత్తాంశం_t scaling_type,
                           మాతృక_info_t<దశాంశం> * మాతృక_info) {
        std::uint64_t కీలకం =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ఎంపిక (కీలకం)
        {
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<దశాంశం, దశాంశం, దశాంశం, దశాంశం>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_double, library_దత్తాంశం_t::real_double,
            library_దత్తాంశం_t::real_double, library_దత్తాంశం_t::real_double):
        {
            detail::gemm_batch_impl<ద్విగుణ, ద్విగుణ, ద్విగుణ, ద్విగుణ>(q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb,
                                                                    beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
#నిర్వచితమైతే __INTEL_MKL__
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_bfloat16,
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_bfloat16,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
#అయితే_ముగింపు
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_int8, library_దత్తాంశం_t::real_int8,
            library_దత్తాంశం_t::real_int32, library_దత్తాంశం_t::real_int32):
        {
            దశాంశం alpha_float =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి std::int32_t *>(alpha), q);
            దశాంశం beta_float =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి std::int32_t *>(beta), q);
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t, దశాంశం>(
                q, a_trans, b_trans, m, n, k, &alpha_float, a, lda, b, ldb, &beta_float, c, ldc, batch_కొలత,
                మాతృక_info);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_int8, library_దత్తాంశం_t::real_int8,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_float):
        {
            దశాంశం alpha_విలువ_2 =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి దశాంశం *>(alpha), q);
            దశాంశం beta_విలువ =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి దశాంశం *>(beta), q);
            sycl::half alpha_half(alpha_విలువ_2);
            sycl::half beta_half(beta_విలువ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, b, ldb, &beta_half, c, ldc, batch_కొలత, మాతృక_info);
            ఆపు;
        }
        అప్రమేయ:
            విసురు std::runtime_error("the combination of data type is unsupported");
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
    అంతఃవరుస శూన్య gemm_batch(sycl::queue & q, oneapi::mkl::transpose a_trans, oneapi::mkl::transpose b_trans, పూర్ణాంకం m,
                           పూర్ణాంకం n, పూర్ణాంకం k, స్థిరరాశి శూన్య * alpha, స్థిరరాశి శూన్య * a, library_దత్తాంశం_t a_type, పూర్ణాంకం lda,
                           దీర్ఘ దీర్ఘ పూర్ణాంకం stride_a, స్థిరరాశి శూన్య * b, library_దత్తాంశం_t b_type, పూర్ణాంకం ldb,
                           దీర్ఘ దీర్ఘ పూర్ణాంకం stride_b, స్థిరరాశి శూన్య * beta, శూన్య * c, library_దత్తాంశం_t c_type, పూర్ణాంకం ldc,
                           దీర్ఘ దీర్ఘ పూర్ణాంకం stride_c, పూర్ణాంకం batch_కొలత, library_దత్తాంశం_t scaling_type) {
        ఒకవేళ (scaling_type == library_దత్తాంశం_t::real_float &&
            c_type == library_దత్తాంశం_t::complex_float)
        {
            scaling_type = library_దత్తాంశం_t::complex_float;
        }
        లేకపోతే ఒకవేళ (scaling_type == library_దత్తాంశం_t::real_double &&
                 c_type == library_దత్తాంశం_t::complex_double)
        {
            scaling_type = library_దత్తాంశం_t::complex_double;
        }

        std::uint64_t కీలకం =
            detail::get_type_combination_id(a_type, b_type, c_type, scaling_type);
        ఎంపిక (కీలకం)
        {
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<దశాంశం, దశాంశం, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_double, library_దత్తాంశం_t::real_double,
            library_దత్తాంశం_t::real_double, library_దత్తాంశం_t::real_double):
        {
            detail::gemm_batch_impl<ద్విగుణ, ద్విగుణ, ద్విగుణ, ద్విగుణ>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::complex_float, library_దత్తాంశం_t::complex_float,
            library_దత్తాంశం_t::complex_float, library_దత్తాంశం_t::complex_float):
        {
            detail::gemm_batch_impl<std::complex<దశాంశం>, std::complex<దశాంశం>,
                                    std::complex<దశాంశం>, std::complex<దశాంశం>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::complex_double, library_దత్తాంశం_t::complex_double,
            library_దత్తాంశం_t::complex_double, library_దత్తాంశం_t::complex_double):
        {
            detail::gemm_batch_impl<std::complex<ద్విగుణ>, std::complex<ద్విగుణ>,
                                    std::complex<ద్విగుణ>, std::complex<ద్విగుణ>>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half,
                                    sycl::half>(q, a_trans, b_trans, m, n, k, alpha,
                                                a, lda, stride_a, b, ldb, stride_b,
                                                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
#నిర్వచితమైతే __INTEL_MKL__
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_bfloat16,
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_bfloat16, library_దత్తాంశం_t::real_bfloat16,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<oneapi::mkl::bfloat16, oneapi::mkl::bfloat16, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b, beta, c, ldc, stride_c,
                batch_కొలత);
            ఆపు;
        }
#అయితే_ముగింపు
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_int8, library_దత్తాంశం_t::real_int8,
            library_దత్తాంశం_t::real_int32, library_దత్తాంశం_t::real_int32):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, std::int32_t,
                                    std::int32_t>(q, a_trans, b_trans, m, n, k, alpha,
                                                  a, lda, stride_a, b, ldb, stride_b,
                                                  beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_int8, library_దత్తాంశం_t::real_int8,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<std::int8_t, std::int8_t, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_float, library_దత్తాంశం_t::real_float):
        {
            detail::gemm_batch_impl<sycl::half, sycl::half, దశాంశం, దశాంశం>(
                q, a_trans, b_trans, m, n, k, alpha, a, lda, stride_a, b, ldb, stride_b,
                beta, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        స్థితి detail::get_type_combination_id(
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_half,
            library_దత్తాంశం_t::real_half, library_దత్తాంశం_t::real_float):
        {
            దశాంశం alpha_విలువ_2 =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి దశాంశం *>(alpha), q);
            దశాంశం beta_విలువ =
                dpct::get_విలువ(reinterpret_cast<స్థిరరాశి దశాంశం *>(beta), q);
            sycl::half alpha_half(alpha_విలువ_2);
            sycl::half beta_half(beta_విలువ);
            detail::gemm_batch_impl<sycl::half, sycl::half, sycl::half, sycl::half>(
                q, a_trans, b_trans, m, n, k, &alpha_half, a, lda, stride_a, b, ldb, stride_b,
                &beta_half, c, ldc, stride_c, batch_కొలత);
            ఆపు;
        }
        అప్రమేయ:
            విసురు std::runtime_error("the combination of data type is unsupported");
        }
    }

    స్థిర అంతఃవరుస శూన్య
    async_dpct_memcpy(శూన్య *గమ్యం_ptr, size_t గమ్యం_pitch, స్థిరరాశి శూన్య *from_ptr,
                      size_t from_pitch, size_t x, size_t y,
                      memcpy_direction direction = automatic,
                      sycl::queue &q = get_default_వరుస())
    {
        detail::dpct_memcpy(q, గమ్యం_ptr, from_ptr, గమ్యం_pitch, from_pitch, x, y,
                            direction);
    }

    ఉపయోగించి err0 = detail::generic_error_type<నిర్మాణం err0_ట్యాగ్, పూర్ణాంకం>;
    ఉపయోగించి err1 = detail::generic_error_type<నిర్మాణం err1_ట్యాగ్, పూర్ణాంకం>;

    స్థిర అంతఃవరుస శూన్య dpct_free(శూన్య *ptr, sycl::queue &q = get_default_వరుస()) {
        detail::dpct_free(ptr, q);
    }

    /// dpct accessor used as device function parameter.
    మూస <తరగతి T, జ్ఞాపకం_region జ్ఞాపకం_2, size_t Dimension> తరగతి accessor;
    మూస <తరగతి T, జ్ఞాపకం_region జ్ఞాపకం_2> తరగతి accessor<T, జ్ఞాపకం_2, 3> {
    బహిరంగ:
        ఉపయోగించి జ్ఞాపకం_t = detail::జ్ఞాపకం_traits<జ్ఞాపకం_2, T>;
        ఉపయోగించి element_t = రకం_పేరు జ్ఞాపకం_t::element_t;
        ఉపయోగించి సూచిక_t = రకం_పేరు జ్ఞాపకం_t::సూచిక_t;
        ఉపయోగించి accessor_t = రకం_పేరు జ్ఞాపకం_t::మూస accessor_t<3>;
        accessor(సూచిక_t దత్తాంశం, స్థిరరాశి sycl::range<3> &లోపల_range)
            : దత్తాంశం_2(దత్తాంశం), _range(లోపల_range) {}
        మూస <జ్ఞాపకం_region M = జ్ఞాపకం_2>
        accessor(రకం_పేరు std::enable_if<M != local, స్థిరరాశి accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(స్థిరరాశి accessor_t &acc, స్థిరరాశి sycl::range<3> &లోపల_range)
            : accessor(acc.get_సూచిక(), లోపల_range) {}
        accessor<T, జ్ఞాపకం_2, 2> క్రియాకారకం[](size_t సూచీ) స్థిరరాశి {
            sycl::range<2> sub(_range.get(1), _range.get(2));
            తిరిగి accessor<T, జ్ఞాపకం_2, 2>(దత్తాంశం_2 + సూచీ * sub.size(), sub);
        }

        సూచిక_t get_ptr() స్థిరరాశి { తిరిగి దత్తాంశం_2; }

    అంతరంగ:
        సూచిక_t దత్తాంశం_2;
        sycl::range<3> _range;
    };
    మూస <తరగతి T, జ్ఞాపకం_region జ్ఞాపకం_2> తరగతి accessor<T, జ్ఞాపకం_2, 2> {
    బహిరంగ:
        ఉపయోగించి జ్ఞాపకం_t = detail::జ్ఞాపకం_traits<జ్ఞాపకం_2, T>;
        ఉపయోగించి element_t = రకం_పేరు జ్ఞాపకం_t::element_t;
        ఉపయోగించి సూచిక_t = రకం_పేరు జ్ఞాపకం_t::సూచిక_t;
        ఉపయోగించి accessor_t = రకం_పేరు జ్ఞాపకం_t::మూస accessor_t<2>;
        accessor(సూచిక_t దత్తాంశం, స్థిరరాశి sycl::range<2> &లోపల_range)
            : దత్తాంశం_2(దత్తాంశం), _range(లోపల_range) {}
        మూస <జ్ఞాపకం_region M = జ్ఞాపకం_2>
        accessor(రకం_పేరు std::enable_if<M != local, స్థిరరాశి accessor_t>::type &acc)
            : accessor(acc, acc.get_range()) {}
        accessor(స్థిరరాశి accessor_t &acc, స్థిరరాశి sycl::range<2> &లోపల_range)
            : accessor(acc.get_సూచిక(), లోపల_range) {}

        సూచిక_t క్రియాకారకం[](size_t సూచీ) స్థిరరాశి {
            తిరిగి దత్తాంశం_2 + _range.get(1) * సూచీ;
        }

        సూచిక_t get_ptr() స్థిరరాశి { తిరిగి దత్తాంశం_2; }

    అంతరంగ:
        సూచిక_t దత్తాంశం_2;
        sycl::range<2> _range;
    };

    నామాకాశ detail {
        /// Device variable with address space of shared, global or constant.
        మూస <తరగతి T, జ్ఞాపకం_region జ్ఞాపకం_2, size_t Dimension> తరగతి device_జ్ఞాపకం {
        బహిరంగ:
            ఉపయోగించి accessor_t =
                రకం_పేరు detail::జ్ఞాపకం_traits<జ్ఞాపకం_2,
                                            T>::మూస accessor_t<Dimension>;
            ఉపయోగించి విలువ_t = రకం_పేరు detail::జ్ఞాపకం_traits<జ్ఞాపకం_2, T>::విలువ_t;
            ఉపయోగించి dpct_accessor_t = dpct::accessor<T, జ్ఞాపకం_2, Dimension>;

            device_జ్ఞాపకం() : device_జ్ఞాపకం(sycl::range<Dimension>(1)) {}

            /// Constructor of 1-D array with initializer list
            device_జ్ఞాపకం(స్థిరరాశి sycl::range<Dimension> &లోపల_range,
                        std::initializer_list<విలువ_t> &&init_list)
                : device_జ్ఞాపకం(లోపల_range) {
                ప్రతిజ్ఞా(init_list.size() <= లోపల_range.size());
                _host_ptr = (విలువ_t *)std::జ్ఞాపకం_ఇవ్వు(కొలత_2);
                std::memset(_host_ptr, 0, కొలత_2);
                std::జ్ఞాపకం_నకలు(_host_ptr, init_list.begin(), init_list.size() * పరిమాణం(T));
            }

            /// Constructor of 2-D array with initializer list
            మూస <size_t D = Dimension>
            device_జ్ఞాపకం(
                స్థిరరాశి రకం_పేరు std::enable_if<D == 2, sycl::range<2>>::type &లోపల_range,
                std::initializer_list<std::initializer_list<విలువ_t>> &&init_list)
                : device_జ్ఞాపకం(లోపల_range) {
                ప్రతిజ్ఞా(init_list.size() <= లోపల_range[0]);
                _host_ptr = (విలువ_t *)std::జ్ఞాపకం_ఇవ్వు(కొలత_2);
                std::memset(_host_ptr, 0, కొలత_2);
                స్వయంచాలక tmp_దత్తాంశం = _host_ptr;
                కొరకు (స్వయంచాలక sub_list : init_list) {
                    ప్రతిజ్ఞా(sub_list.size() <= లోపల_range[1]);
                    std::జ్ఞాపకం_నకలు(tmp_దత్తాంశం, sub_list.begin(),
                                sub_list.size() * పరిమాణం(T));
                    tmp_దత్తాంశం += లోపల_range[1];
                }
            }

            /// Constructor with range
            device_జ్ఞాపకం(స్థిరరాశి sycl::range<Dimension> &range_లోపల)
                : కొలత_2(range_లోపల.size() * పరిమాణం(T)), _range(range_లోపల),
                _reference(అబద్ధం), _host_ptr(శూన్యసూచిక), _device_ptr(శూన్యసూచిక) {
                static_assert(
                    (జ్ఞాపకం_2 == global) || (జ్ఞాపకం_2 == constant) || (జ్ఞాపకం_2 == shared),
                    "device memory region should be global, constant or shared");
                // Make sure that singleton class mem_mgr and dev_mgr will destruct
                // later than this.
                detail::mem_mgr::instance();
                dev_mgr::instance();
            }

            /// Constructor with range
            మూస <తరగతి... వాదనలు>
            device_జ్ఞాపకం(వాదనలు... Arguments)
                : device_జ్ఞాపకం(sycl::range<Dimension>(Arguments...)) {}

            ~device_జ్ఞాపకం() {
                ఒకవేళ (_device_ptr && !_reference)
                    dpct::dpct_free(_device_ptr);
                ఒకవేళ (_host_ptr)
                    std::విడుదల(_host_ptr);
            }

            /// Allocate memory with default queue, and init memory if has initial
            /// value.
            శూన్య init() { init(dpct::get_default_వరుస()); }
            /// Allocate memory with specified queue, and init memory if has initial
            /// value.
            శూన్య init(sycl::queue &q) {
                ఒకవేళ (_device_ptr)
                    తిరిగి;
                ఒకవేళ (!కొలత_2)
                    తిరిగి;
                allocate_device(q);
                ఒకవేళ (_host_ptr)
                    detail::dpct_memcpy(q, _device_ptr, _host_ptr, కొలత_2,
                                        host_గమ్యం_device);
            }

            /// The variable is assigned to a device pointer.
            శూన్య కేటాయించు(విలువ_t *src, size_t size) {
                ఇది->~device_జ్ఞాపకం();
                కొత్త (ఇది) device_జ్ఞాపకం(src, size);
            }

            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            విలువ_t *get_ptr() { తిరిగి get_ptr(get_default_వరుస()); }
            /// Get memory pointer of the memory object, which is virtual pointer when
            /// usm is not used, and device pointer when usm is used.
            విలువ_t *get_ptr(sycl::queue &q) {
                init(q);
                తిరిగి _device_ptr;
            }

            /// Get the device memory object size in bytes.
            size_t get_కొలత() { తిరిగి కొలత_2; }

            మూస <size_t D = Dimension>
            రకం_పేరు std::enable_if<D == 1, T>::type &క్రియాకారకం[](size_t సూచీ) {
                init();
                తిరిగి _device_ptr[సూచీ];
            }

            /// Get dpct::accessor with dimension info for the device memory object
            /// when usm is used and dimension is greater than 1.
            మూస <size_t D = Dimension>
            రకం_పేరు std::enable_if<D != 1, dpct_accessor_t>::type
            get_access([[maybe_unused]] sycl::handler &cgh) {
                తిరిగి dpct_accessor_t((T *)_device_ptr, _range);
            }

        అంతరంగ:
            device_జ్ఞాపకం(విలువ_t *జ్ఞాపకం_ptr, size_t size)
                : కొలత_2(size), _range(size / పరిమాణం(T)), _reference(నిజం),
                _device_ptr(జ్ఞాపకం_ptr) {}

            శూన్య allocate_device(sycl::queue &q) {
        #నిర్వచితం_కాకపోతే DPCT_USM_LEVEL_NONE
                ఒకవేళ (జ్ఞాపకం_2 == shared) {
                    _device_ptr = (విలువ_t *)sycl::malloc_shared(కొలత_2, q.get_device(),
                                                                q.get_context());
                    తిరిగి;
                }
        #నిర్వచితమైతే SYCL_EXT_ONEAPI_USM_DEVICE_READ_ONLY
                ఒకవేళ (జ్ఞాపకం_2 == constant) {
                    _device_ptr = (విలువ_t *)sycl::malloc_device(
                        కొలత_2, q.get_device(), q.get_context(),
                        sycl::ext::oneapi::property::usm::device_read_only());
                    తిరిగి;
                }
        #అయితే_ముగింపు
        #అయితే_ముగింపు
                _device_ptr = (విలువ_t *)detail::dpct_malloc(కొలత_2, q);
            }

            size_t కొలత_2;
            sycl::range<Dimension> _range;
            తార్కిక _reference;
            విలువ_t *_host_ptr;
            విలువ_t *_device_ptr;
        };
        మూస <తరగతి T, జ్ఞాపకం_region జ్ఞాపకం_2>
        తరగతి device_జ్ఞాపకం<T, జ్ఞాపకం_2, 0> : బహిరంగ device_జ్ఞాపకం<T, జ్ఞాపకం_2, 1> {
        బహిరంగ:
            ఉపయోగించి పునాది = device_జ్ఞాపకం<T, జ్ఞాపకం_2, 1>;
            ఉపయోగించి విలువ_t = రకం_పేరు పునాది::విలువ_t;
            ఉపయోగించి accessor_t =
                రకం_పేరు detail::జ్ఞాపకం_traits<జ్ఞాపకం_2, T>::మూస accessor_t<0>;

            /// Constructor with initial value.
            device_జ్ఞాపకం(స్థిరరాశి విలువ_t &val) : పునాది(sycl::range<1>(1), {val}) {}

            /// Default constructor
            device_జ్ఞాపకం() : పునాది(1) {}
        };
        } // namespace detail

    మూస <తరగతి T, size_t Dimension>
    ఉపయోగించి global_జ్ఞాపకం = detail::device_జ్ఞాపకం<T, global, Dimension>;
    మూస <తరగతి T, size_t Dimension>
    ఉపయోగించి constant_జ్ఞాపకం = detail::device_జ్ఞాపకం<T, constant, Dimension>;
    మూస <తరగతి T, size_t Dimension>
    ఉపయోగించి shared_జ్ఞాపకం = detail::device_జ్ఞాపకం<T, shared, Dimension>;


    మూస <రకం_పేరు T,
            sycl::access::చిరునామా_space చిరునామా_Space =
                sycl::access::చిరునామా_space::global_space,
            sycl::memory_order జ్ఞాపకం_క్రమం = sycl::memory_order::relaxed,
            sycl::జ్ఞాపకం_scope జ్ఞాపకం_Scope = sycl::జ్ఞాపకం_scope::device>
    అంతఃవరుస T atomic_fetch_add(T *addr, T నిర్వాహ్యం) {
    స్వయంచాలక atm =
        sycl::atomic_ref<T, జ్ఞాపకం_క్రమం, జ్ఞాపకం_Scope, చిరునామా_Space>(addr[0]);
    తిరిగి atm.fetch_add(నిర్వాహ్యం);
    }

    మూస <sycl::access::చిరునామా_space చిరునామా_Space =
                sycl::access::చిరునామా_space::global_space,
            sycl::memory_order జ్ఞాపకం_క్రమం = sycl::memory_order::relaxed,
            sycl::జ్ఞాపకం_scope జ్ఞాపకం_Scope = sycl::జ్ఞాపకం_scope::device,
            రకం_పేరు T1, రకం_పేరు T2>
    అంతఃవరుస T1 atomic_fetch_add(T1 *addr, T2 నిర్వాహ్యం) {
    స్వయంచాలక atm =
        sycl::atomic_ref<T1, జ్ఞాపకం_క్రమం, జ్ఞాపకం_Scope, చిరునామా_Space>(addr[0]);
    తిరిగి atm.fetch_add(నిర్వాహ్యం);
    }

    మూస <రకం_పేరు T, sycl::access::చిరునామా_space చిరునామా_Space =
                            sycl::access::చిరునామా_space::global_space>
    అంతఃవరుస T atomic_fetch_add(T *addr, T నిర్వాహ్యం,
                            sycl::memory_order జ్ఞాపకం_క్రమం) {
    ఎంపిక (జ్ఞాపకం_క్రమం) {
        స్థితి sycl::memory_order::relaxed:
            తిరిగి atomic_fetch_add<T, చిరునామా_Space, sycl::memory_order::relaxed,
                                    sycl::జ్ఞాపకం_scope::device>(addr, నిర్వాహ్యం);
        స్థితి sycl::memory_order::acq_rel:
            తిరిగి atomic_fetch_add<T, చిరునామా_Space, sycl::memory_order::acq_rel,
                                    sycl::జ్ఞాపకం_scope::device>(addr, నిర్వాహ్యం);
        స్థితి sycl::memory_order::seq_cst:
            తిరిగి atomic_fetch_add<T, చిరునామా_Space, sycl::memory_order::seq_cst,
                                    sycl::జ్ఞాపకం_scope::device>(addr, నిర్వాహ్యం);
        అప్రమేయ:
            ప్రతిజ్ఞా(అబద్ధం && "Invalid memory_order for atomics. Valid memory_order for "
                            "atomics are: sycl::memory_order::relaxed, "
                            "sycl::memory_order::acq_rel, sycl::memory_order::seq_cst!");
        }
    }

    మూస <sycl::access::చిరునామా_space చిరునామా_Space =
                sycl::access::చిరునామా_space::global_space,
            రకం_పేరు T1, రకం_పేరు T2>
    అంతఃవరుస T1 atomic_fetch_add(T1 *addr, T2 నిర్వాహ్యం,
                            sycl::memory_order జ్ఞాపకం_క్రమం) {
    atomic_fetch_add<T1, చిరునామా_Space>(addr, నిర్వాహ్యం, జ్ఞాపకం_క్రమం);
    }

    అంతఃవరుస గుర్తులేని పూర్ణాంకం byte_level_క్రమమార్పు(
        గుర్తులేని పూర్ణాంకం a, గుర్తులేని పూర్ణాంకం b, గుర్తులేని పూర్ణాంకం s) {
      గుర్తులేని పూర్ణాంకం ret;
      ret = ((((std::uint64_t)b << 32 | a) >> (s & 0x7) * 8) & 0xff) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 4) & 0x7) * 8) & 0xff)
             << 8) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 8) & 0x7) * 8) & 0xff)
             << 16) |
            (((((std::uint64_t)b << 32 | a) >> ((s >> 12) & 0x7) * 8) & 0xff)
             << 24);
      తిరిగి ret;
    }

    అంతఃవరుస uint32_t byte_level_క్రమమార్పు_custom(
        uint32_t low32, uint32_t high32, uint32_t sel, పూర్ణాంకం mode = 0) {
      constexpr uint16_t lookup[6][4] = {
          {0x3210, 0x4321, 0x5432, 0x6543},  // Forward 4-byte extract
          {0x5670, 0x6701, 0x7012, 0x0123},  // Backward 4-byte extract
          {0x0000, 0x1111, 0x2222, 0x3333},  // Replicate 8-bit values
          {0x3210, 0x3211, 0x3222, 0x3333},  // Edge clamp left
          {0x0000, 0x1110, 0x2210, 0x3210},  // Edge clamp right
          {0x1010, 0x3232, 0x1010, 0x3232}   // Replicate 16-bit values
      };

      ఒకవేళ (mode >= 1 && mode <= 6) {
        తిరిగి byte_level_క్రమమార్పు(low32, high32, lookup[mode - 1][sel & 0x3]);
      } లేకపోతే ఒకవేళ (!mode) {
        తిరిగి byte_level_క్రమమార్పు(low32, high32, sel);
      }
      తిరిగి 0;
    }

    మూస <పూర్ణాంకం n_nondefault_params, పూర్ణాంకం n_default_params, రకం_పేరు T>
    తరగతి వాదనలు_selector;

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
    మూస <పూర్ణాంకం n_nondefault_params, పూర్ణాంకం n_default_params, రకం_పేరు R,
              రకం_పేరు... Ts>
    తరగతి వాదనలు_selector<n_nondefault_params, n_default_params, R(Ts...)> {
      అంతరంగ:
        శూన్య **kernel_params;
        అక్షరం *వాదనలు_బఫరు;

        మూస <పూర్ణాంకం i> స్థిర constexpr పూర్ణాంకం account_for_default_params() {
            constexpr పూర్ణాంకం n_మొత్తం_params = పరిమాణం...(Ts);
            ఒకవేళ constexpr (i >= n_nondefault_params) {
                తిరిగి n_మొత్తం_params - n_default_params +
                       (i - n_nondefault_params);
            } లేకపోతే {
                తిరిగి i;
            }
        }

      బహిరంగ:
        /// Get the type of the ith argument of R(Ts...)
        /// \param [in] i Index of parameter to get
        /// \returns Type of ith parameter
        మూస <పూర్ణాంకం i>
        ఉపయోగించి arg_type = std::tuple_element_t<account_for_default_params<i>(),
                                              std::tuple<Ts...>>;
        స్థిర constexpr పూర్ణాంకం params_అంకె = పరిమాణం...(Ts);

      అంతరంగ:
        మూస <పూర్ణాంకం i> స్థిర constexpr పూర్ణాంకం get_స్థానభ్రంశం() {
            ఒకవేళ constexpr (i == 0) {
                // we can assume args_buffer is properly aligned to the
                // first argument
                తిరిగి 0;
            } లేకపోతే {
                constexpr పూర్ణాంకం prev_off = get_స్థానభ్రంశం<i - 1>();
                constexpr పూర్ణాంకం prev_past_చివర =
                    prev_off + పరిమాణం(arg_type<i - 1>);
                ఉపయోగించి T = arg_type<i>;
                // is the past-the-end of the i-1st element properly aligned
                // with the ith element's alignment?
                ఒకవేళ constexpr (prev_past_చివర % alignof(T) == 0) {
                    తిరిగి prev_past_చివర;
                }
                // otherwise bump prev_past_end to match alignment
                లేకపోతే {
                    తిరిగి prev_past_చివర +
                           (alignof(T) - (prev_past_చివర % alignof(T)));
                }
            }
        }

        స్థిర అక్షరం *get_వాదనలు_బఫరు(శూన్య **extra) {
            ఒకవేళ (!extra)
                తిరిగి శూన్యసూచిక;
            కొరకు (; (std::size_t)*extra != 0; ++extra) {
                ఒకవేళ ((std::size_t)*extra == 1) {
                    తిరిగి static_cast<అక్షరం *>(*(extra + 1));
                }
            }
            తిరిగి శూన్యసూచిక;
        }

      బహిరంగ:
        /// If kernel_params is nonnull, then args_selector will
        /// extract arguments from kernel_params. Otherwise, it
        /// will extract them from extra.
        /// \param [in] kernel_params Array of pointers to arguments
        /// a or null pointer.
        /// \param [in] extra Array containing pointer to argument buffer.
        వాదనలు_selector(శూన్య **kernel_params, శూన్య **extra)
            : kernel_params(kernel_params),
              వాదనలు_బఫరు(get_వాదనలు_బఫరు(extra)) {}

        /// Get a reference to the ith argument extracted from kernel_params
        /// or extra.
        /// \param [in] i Index of argument to get
        /// \returns Reference to the ith argument
        మూస <పూర్ణాంకం i> arg_type<i> &get() {
            ఒకవేళ (kernel_params) {
                తిరిగి *static_cast<arg_type<i> *>(kernel_params[i]);
            } లేకపోతే {
                తిరిగి *reinterpret_cast<arg_type<i> *>(వాదనలు_బఫరు +
                                                        get_స్థానభ్రంశం<i>());
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
    తరగతి kernel_launcher {
        మూస <రకం_పేరు FuncT, రకం_పేరు ArgSelector, std::size_t... సూచీ_2>
        స్థిర శూన్య launch_helper(FuncT &&func, ArgSelector &selector,
                                  std::index_sequence<సూచీ_2...>) {
            func(selector.మూస get<సూచీ_2>()...);
        }
        స్థిర శూన్య సమితి_execution_config(dim3 సమూహం_range, dim3 local_range,
                                         గుర్తులేని పూర్ణాంకం local_mem_కొలత_2,
                                         వరుస_ptr que) {
            ఒకవేళ (que) {
                _que = que;
            } లేకపోతే {
                _que = &get_default_వరుస();
            }
            _nr = sycl::nd_range<3>(
                static_cast<sycl::range<3>>(సమూహం_range * local_range),
                static_cast<sycl::range<3>>(local_range));
            local_mem_కొలత = local_mem_కొలత_2;


        }
        స్థిర అంతఃవరుస std::mutex kernel_function_ptr_map_mutex;

      బహిరంగ:
        /// Variables for storing execution configuration.
        స్థిర అంతఃవరుస thread_local sycl::queue *_que = శూన్యసూచిక;
        స్థిర అంతఃవరుస thread_local sycl::nd_range<3> _nr = sycl::nd_range<3>();
        స్థిర అంతఃవరుస thread_local గుర్తులేని పూర్ణాంకం local_mem_కొలత = 0;
        /// Map for retrieving launchable functor from a raw pointer.
        స్థిర అంతఃవరుస std::map<
            స్థిరరాశి శూన్య *,
            std::function<శూన్య(dim3, dim3, శూన్య **, గుర్తులేని పూర్ణాంకం, వరుస_ptr)>>
            kernel_function_ptr_map = {};

        /// Registers a kernel function pointer with a corresponding launchable
        /// functor.
        /// \param [in] func Pointer to the kernel function.
        /// \param [in] launcher Functor to handle kernel invocation.
        స్థిర శూన్య రిజిస్టరు_kernel_ptr(
            స్థిరరాశి శూన్య *func,
            std::function<శూన్య(dim3, dim3, శూన్య **, గుర్తులేని పూర్ణాంకం, వరుస_ptr)>
                launcher) {
            std::lock_guard<std::mutex> తాళం(kernel_function_ptr_map_mutex);
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
        మూస <రకం_పేరు FuncT, రకం_పేరు... వాదనలు_T>
        స్థిర std::enable_if_t<std::is_invocable_v<FuncT *, వాదనలు_T...>, శూన్య>
        launch(FuncT *func, dim3 సమూహం_range, dim3 local_range,
               గుర్తులేని పూర్ణాంకం local_mem_కొలత_2, వరుస_ptr que, వాదనలు_T... వాదనలు_2) {
            సమితి_execution_config(సమూహం_range, local_range, local_mem_కొలత_2, que);
            func(వాదనలు_2...);
        }
        /// Launches a kernel function through registered kernel function
        /// wrapper. \param [in] func Pointer to the registered kernel function
        /// wrapper. \param [in] group_range SYCL group range. \param [in]
        /// local_range SYCL local range. \param [in] args Array of pointers to
        /// kernel arguments. \param [in] local_mem_size The size of local
        /// memory required by the kernel function. \param [in] que SYCL queue
        /// used to execute kernel.
        స్థిర శూన్య launch(స్థిరరాశి శూన్య *func, dim3 సమూహం_range, dim3 local_range,
                           శూన్య **వాదనలు_2, గుర్తులేని పూర్ణాంకం local_mem_కొలత_2,
                           వరుస_ptr que) {
            std::lock_guard<std::mutex> తాళం(kernel_function_ptr_map_mutex);
            స్వయంచాలక Iter = kernel_function_ptr_map.find(func);
            ఒకవేళ (Iter == kernel_function_ptr_map.end()) {
                విసురు std::runtime_error("dpct::launch() : no registered "
                                         "kernel function wrapper found.");
            }
            (Iter->second)(సమూహం_range, local_range, వాదనలు_2, local_mem_కొలత_2, que);
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
        మూస <రకం_పేరు FuncT>
        స్థిర std::enable_if_t<std::is_function_v<FuncT>, శూన్య>
        launch(FuncT *func, dim3 సమూహం_range, dim3 local_range, శూన్య **వాదనలు_2,
               గుర్తులేని పూర్ణాంకం local_mem_కొలత_2, వరుస_ptr que) {
            constexpr size_t p_అంకె = వాదనలు_selector<0, 0, FuncT>::params_అంకె;
            సమితి_execution_config(సమూహం_range, local_range, local_mem_కొలత_2, que);
            వాదనలు_selector<p_అంకె, p_అంకె, FuncT> selector(వాదనలు_2, శూన్యసూచిక);
            launch_helper(func, selector, std::make_index_sequence<p_అంకె>{});
        }
    }; // COPY from DPCT head file
       // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/kernel.hpp

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/util.hpp
    మూస <రకం_పేరు T>
    T select_from_sub_సమూహం(
        sycl::sub_సమూహం g,
        T x,
        పూర్ణాంకం remote_local_id,
        పూర్ణాంకం logical_sub_సమూహం_కొలత = 32) {
      గుర్తులేని పూర్ణాంకం ప్రారంభం_సూచీ = g.get_local_రేఖీయ_id() /
                                 logical_sub_సమూహం_కొలత *
                                 logical_sub_సమూహం_కొలత;
      తిరిగి sycl::select_from_సమూహం(
          g, x, ప్రారంభం_సూచీ + remote_local_id % logical_sub_సమూహం_కొలత);
    }

    // /opt/intel/oneapi/dpcpp-ct/latest/include/dpct/math.hpp
    మూస <రకం_పేరు T>
    శూన్య ldmatrix(uintptr_t addr, T* m, తార్కిక trans = అబద్ధం, గుర్తులేని mat = 0) {
      స్వయంచాలక sg = sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం();
      పూర్ణాంకం lane = sg.get_local_రేఖీయ_id();

      పూర్ణాంకం lane_group8_అడ్డువరుస = lane / 8;
      పూర్ణాంకం lane_group8_నిలువు = lane % 8;

      ఒకవేళ (!trans) {
        // calculate the source lane
        పూర్ణాంకం src_lane = 2 * lane_group8_అడ్డువరుస;
        ఒకవేళ (lane_group8_నిలువు >= 4)
          src_lane += 1;

        // Broadcast the address from the source lane
        స్వయంచాలక recv_addr_uintp =
            dpct::select_from_sub_సమూహం(sg, addr, mat * 8 + src_lane);

        // Cast the received address from uintptr_t to the type of 'm'
        స్వయంచాలక recv_addr = reinterpret_cast<T*>(recv_addr_uintp);

        // Non-transposed load
        *m = recv_addr[lane_group8_నిలువు % 4];
      } లేకపోతే {
        // calculate the source lane
        పూర్ణాంకం src_lane = (lane % 4) * 2;

        // Broadcast the address from the source lane
        స్వయంచాలక recv_addr_uintp_1 =
            dpct::select_from_sub_సమూహం(sg, addr, mat * 8 + src_lane);
        స్వయంచాలక recv_addr_uintp_2 =
            dpct::select_from_sub_సమూహం(sg, addr, mat * 8 + src_lane + 1);

        // Cast the received address from uintptr_t to 'half *'
        స్వయంచాలక recv_addr_1 = reinterpret_cast<sycl::half*>(recv_addr_uintp_1);
        స్వయంచాలక recv_addr_2 = reinterpret_cast<sycl::half*>(recv_addr_uintp_2);

        // Transposed load
        పూర్ణాంకం సూచీ = lane / 4;
        sycl::half val0 = recv_addr_1[సూచీ];
        sycl::half val1 = recv_addr_2[సూచీ];

        // Combine the two 16-bits into one 32-bit value
        sycl::half2 val = sycl::half2(val0, val1);
        *m = *reinterpret_cast<T*>(&val);
      }
    }

    మూస <రకం_పేరు T>
    శూన్య ldmatrix(uintptr_t addr, T* m1, T* m2, తార్కిక trans = అబద్ధం) {
      // Load 1st matrix
      ldmatrix(addr, m1, trans, 0);
      // Load 2nd matrix
      ldmatrix(addr, m2, trans, 1);
    }

    మూస <రకం_పేరు T>
    శూన్య ldmatrix(
        uintptr_t addr, T* m1, T* m2, T* m3, T* m4, తార్కిక trans = అబద్ధం) {
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
    మూస <రకం_పేరు T>
    నిర్మాణం MMAType {
      ఉపయోగించి PackType = uint32_t;
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
    మూస <పూర్ణాంకం M, పూర్ణాంకం N, పూర్ణాంకం K, రకం_పేరు ABType, రకం_పేరు CDType>
    శూన్య mma(
        అస్థిర శూన్య** d_mat_frag,
        శూన్య* a_mat_frag,
        శూన్య* b_mat_frag,
        శూన్య* c_mat_frag) {
      స్వయంచాలక d = reinterpret_cast<అస్థిర CDType**>(d_mat_frag);
      స్వయంచాలక a =
          reinterpret_cast<రకం_పేరు MMAType<ABType>::PackType*>(a_mat_frag);
      స్వయంచాలక b =
          reinterpret_cast<రకం_పేరు MMAType<ABType>::PackType*>(b_mat_frag);
      స్వయంచాలక c = reinterpret_cast<CDType*>(c_mat_frag);

      స్వయంచాలక sg = sycl::ext::oneapi::this_work_అంశం::get_sub_సమూహం();
      పూర్ణాంకం lane = sg.get_local_రేఖీయ_id();

      static_assert(
          (M == 8 && N == 8 && K == 4) || (M == 8 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 8) || (M == 16 && N == 8 && K == 16) ||
              (M == 16 && N == 8 && K == 32),
          "Unsupported MMA shape!");

      పొట్టి అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం = 4 * (lane >> 2);
      పొట్టి నిలువు_ఎక్కించు_స్థానభ్రంశం = 8 * (lane % 4);

      ఒకవేళ constexpr (M == 8 && N == 8 && K == 4) {
        ఒకవేళ constexpr (std::is_floating_point_v<CDType>) {
          నిలువు_ఎక్కించు_స్థానభ్రంశం = అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం % 16;

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
          పూర్ణాంకం r_ind = (lane % 2) ? 1 : 0;
          పూర్ణాంకం c_ind = ((lane % 4) / 2) ? 2 : 0;

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
          రకం_పేరు MMAType<ABType>::PackType recv_a[2], recv_b[2];

          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            // Load partial fragment from col0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_సమూహం(sg, a[0], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_సమూహం(sg, a[1], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);

            // Load partial fragment from row0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[1], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);

            స్వయంచాలక ra = reinterpret_cast<ABType*>(recv_a);
            స్వయంచాలక rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment (for
            // even work item indices) d0 += col0{ a0 } * row0{ b0 } d1 += col0{
            // a0 } * row0{ b1 } d2 += col1{ a2 } * row0{ b0 } d3 += col1{ a2 }
            // * row0{ b1 } (for odd work item indices) d0 += col0{ a1 } * row0{
            // b2 } d1 += col0{ a1 } * row0{ b3 } d2 += col1{ a3 } * row0{ b2 }
            // d3 += col1{ a3 } * row0{ b3 }
            *d[0] +=
                static_cast<దశాంశం>(ra[r_ind]) * static_cast<దశాంశం>(rb[c_ind]);
            *d[1] += static_cast<దశాంశం>(ra[r_ind]) *
                     static_cast<దశాంశం>(rb[c_ind + 1]);
            *d[2] += static_cast<దశాంశం>(ra[r_ind + 2]) *
                     static_cast<దశాంశం>(rb[c_ind]);
            *d[3] += static_cast<దశాంశం>(ra[r_ind + 2]) *
                     static_cast<దశాంశం>(rb[c_ind + 1]);

            // Load partial fragment from row1 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 16);
            // Load partial fragment from row1 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[1], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 16);

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
                static_cast<దశాంశం>(ra[r_ind]) * static_cast<దశాంశం>(rb[c_ind]);
            *d[5] += static_cast<దశాంశం>(ra[r_ind]) *
                     static_cast<దశాంశం>(rb[c_ind + 1]);
            *d[6] += static_cast<దశాంశం>(ra[r_ind + 2]) *
                     static_cast<దశాంశం>(rb[c_ind]);
            *d[7] += static_cast<దశాంశం>(ra[r_ind + 2]) *
                     static_cast<దశాంశం>(rb[c_ind + 1]);
          }
        }
      } లేకపోతే ఒకవేళ constexpr (M == 8 && N == 8 && K == 16) {
        ఒకవేళ constexpr (std::is_integral_v<ABType>) {
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
          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            రకం_పేరు MMAType<ABType>::PackType recv_a, recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a = dpct::select_from_sub_సమూహం(sg, a[0], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 4);

            స్వయంచాలక a = reinterpret_cast<ABType*>(&recv_a);
            స్వయంచాలక b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row0{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row0{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
            }
          }
        }
      } లేకపోతే ఒకవేళ constexpr (M == 16 && N == 8 && K == 8) {
        ఒకవేళ constexpr (std::is_floating_point_v<CDType>) {
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
          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            రకం_పేరు MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_సమూహం(sg, a[0], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_సమూహం(sg, a[1], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 4);

            స్వయంచాలక ra = reinterpret_cast<ABType*>(recv_a);
            స్వయంచాలక rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1 } * col0{ b0, b1 } d1 += row0{ a0, a1 } * col1{
            // b0, b1 } d2 += row1{ a2, a3 } * col0{ b0, b1 } d3 += row1{ a2, a3
            // } * col1{ b0, b1 }
            కొరకు (పూర్ణాంకం j = 0; j < 2; j++) {
              *d[0] += static_cast<దశాంశం>(ra[j]) * static_cast<దశాంశం>(rb[j]);
              *d[1] +=
                  static_cast<దశాంశం>(ra[j]) * static_cast<దశాంశం>(rb[j + 2]);
              *d[2] +=
                  static_cast<దశాంశం>(ra[j + 2]) * static_cast<దశాంశం>(rb[j]);
              *d[3] +=
                  static_cast<దశాంశం>(ra[j + 2]) * static_cast<దశాంశం>(rb[j + 2]);
            }
          }
        }
      } లేకపోతే ఒకవేళ constexpr (M == 16 && N == 8 && K == 16) {
        ఒకవేళ constexpr (std::is_floating_point_v<CDType>) {
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
          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            రకం_పేరు MMAType<ABType>::PackType recv_a[4], recv_b[4];

            // Load partial fragment from row0 of matrix A ({a0, a1})
            recv_a[0] =
                dpct::select_from_sub_సమూహం(sg, a[0], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row0 of matrix A ({a2, a3})
            recv_a[1] =
                dpct::select_from_sub_సమూహం(sg, a[2], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row1 of matrix A ({a0, a1})
            recv_a[2] =
                dpct::select_from_sub_సమూహం(sg, a[1], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row1 of matrix A ({a2, a3})
            recv_a[3] =
                dpct::select_from_sub_సమూహం(sg, a[3], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);

            // Load partial fragment from col0 of matrix B ({b0, b1})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix B ({b2, b3})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[1], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col1 of matrix B ({b0, b1})
            recv_b[2] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + 4 + i);
            // Load partial fragment from col1 of matrix B ({b2, b3})
            recv_b[3] =
                dpct::select_from_sub_సమూహం(sg, b[1], నిలువు_ఎక్కించు_స్థానభ్రంశం + 4 + i);

            స్వయంచాలక ra = reinterpret_cast<ABType*>(recv_a);
            స్వయంచాలక rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a0, a1, a2,
            // a3 } * col0{ b0, b1, b2, b3 } d3 += row1{ a0, a1, a2, a3 } *
            // col1{ b0, b1, b2, b3 }
            కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
              *d[0] += static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j]);
              *d[1] +=
                  static_cast<CDType>(ra[j]) * static_cast<CDType>(rb[j + 4]);
              *d[2] +=
                  static_cast<CDType>(ra[j + 4]) * static_cast<CDType>(rb[j]);
              *d[3] += static_cast<CDType>(ra[j + 4]) *
                       static_cast<CDType>(rb[j + 4]);
            }
          }
        } లేకపోతే ఒకవేళ constexpr (std::is_integral_v<ABType>) {
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
          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            రకం_పేరు MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_సమూహం(sg, a[0], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_సమూహం(sg, a[1], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 4);

            స్వయంచాలక ra = reinterpret_cast<ABType*>(recv_a);
            స్వయంచాలక rb = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b4, b5, b6, b7 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b4, b5, b6, b7 }
            కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
              *d[0] += ra[i] * rb[i];
              *d[1] += ra[i] * rb[i + 4];
              *d[2] += ra[i + 4] * rb[i];
              *d[3] += ra[i + 4] * rb[i + 4];
            }
          }
        }
      } లేకపోతే ఒకవేళ constexpr (M == 16 && N == 8 && K == 32) {
        ఒకవేళ constexpr (std::is_integral_v<ABType>) {
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
          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            రకం_పేరు MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a0, a1, a2, a3})
            recv_a[0] =
                dpct::select_from_sub_సమూహం(sg, a[0], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row1 of matrix A ({a4, a5, a6, a7})
            recv_a[1] =
                dpct::select_from_sub_సమూహం(sg, a[1], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix B ({b0, b1, b2, b3})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col1 of matrix B ({b0, b1, b2, b3})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[0], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 4);

            స్వయంచాలక a = reinterpret_cast<ABType*>(recv_a);
            స్వయంచాలక b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a0, a1, a2, a3 } * col0{ b0, b1, b2, b3 } d1 += row0{
            // a0, a1, a2, a3 } * col1{ b0, b1, b2, b3 } d2 += row1{ a4, a5, a6,
            // a7 } * col0{ b0, b1, b2, b3 } d3 += row1{ a4, a5, a6, a7 } *
            // col1{ b0, b1, b2, b3 }
            కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
              *d[0] += a[j] * b[j];
              *d[1] += a[j] * b[j + 4];
              *d[2] += a[j + 4] * b[j];
              *d[3] += a[j + 4] * b[j + 4];
            }
          }

          కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            రకం_పేరు MMAType<ABType>::PackType recv_a[2], recv_b[2];

            // Load partial fragment from row0 of matrix A ({a8, a9, a10, a11})
            recv_a[0] =
                dpct::select_from_sub_సమూహం(sg, a[2], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from row1 of matrix A ({a12, a13, a14,
            // a15})
            recv_a[1] =
                dpct::select_from_sub_సమూహం(sg, a[3], అడ్డువరుస_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col0 of matrix B ({b4, b5, b6, b7})
            recv_b[0] =
                dpct::select_from_sub_సమూహం(sg, b[1], నిలువు_ఎక్కించు_స్థానభ్రంశం + i);
            // Load partial fragment from col1 of matrix B ({b4, b5, b6, b7})
            recv_b[1] =
                dpct::select_from_sub_సమూహం(sg, b[1], నిలువు_ఎక్కించు_స్థానభ్రంశం + i + 4);

            స్వయంచాలక a = reinterpret_cast<ABType*>(recv_a);
            స్వయంచాలక b = reinterpret_cast<ABType*>(recv_b);

            // Each work item calculates a partial product of A & B matrix
            // fragments and adds it to the corresponding D matrix fragment d0
            // += row0{ a8, a9, a10, a11 } * col0{ b4, b5, b6, b7 } d1 += row0{
            // a8, a9, a10, a11 } * col1{ b4, b5, b6, b7 } d2 += row1{ a12, a13,
            // a14, a15 } * col0{ b4, b5, b6, b7 } d3 += row1{ a12, a13, a14,
            // a15 } * col1{ b4, b5, b6, b7 }
            కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
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

#అయితే_ముగింపు // GGML_SYCL_DPCT_HELPER_HPP

#ضمن "ggml-et.h"

#ضمن "ggml-backend-impl.h"
#ضمن "ggml-backend.h"
#ضمن "ggml-et-common.h"
#ضمن "ggml-et-kernels.h"
#ضمن "ggml-et-memops.h"
#ضمن "ggml-et-ops.h"
#ضمن "ggml-impl.h"
#ضمن "ggml.h"

#ضمن <stdarg.h>

#ضمن <cstdarg>
#ضمن <cstdio>
#ضمن <cstdlib>
#ضمن <cstring>
#ضمن <string>
#ضمن <vector>

#إذا __has_include(<filesystem>)
#    include <filesystem>
نطاق fs = std::filesystem;
#وإلاإذا __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
نطاق fs = std::experimental::filesystem;
#وإلا
#    error "cannot include the filesystem library"
#د_که_پای

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
ساكن فراغ ggml_et_dump_tensor_metadata(ثابت ggml_tensor * ggtensor, size_t indent_level, ثابت حرف * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (اوږد اوږد) ggtensor->ne[0],
            (اوږد اوږد) ggtensor->ne[1], (اوږد اوږد) ggtensor->ne[2], (اوږد اوږد) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->معلومات, spaces.c_str(), (فراغ *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
ساكن فراغ ggml_et_dump_operator_metadata(ثابت ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != عدم);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    لكل (صحيح i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        حرف arr[16];
        صحيح  n = snprintf(arr, حجم(arr), "src[%i]->name", i);
        GGML_ASSERT((بینښه) n < حجم(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    إذا (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

ساكن بنية ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_بهیر;
    std::unique_ptr<std::ofstream>                    kernel_id_بهیر;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    منطقي                                              profiling_enabled = خطأ;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
ساكن std::string ggml_et_get_default_et_لار() {
    // List of environment variables to check in order of preference
    ثابت حرف * ثابت env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    لكل (ثابت حرف * var : env_vars) {
        إذا (ثابت حرف * et_لار = std::getenv(var)) {
            إذا (et_لار && *et_لار != '\0') {
                أرجع fs::path(et_لار).string();
            }
        }
    }

    // Otherwise assume default
    أرجع fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
ساكن مضمن خپلکاری ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_دورې        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_ماسک = 0x1FFFFFFFFu;
    ثابت std::string  et_لار                 = ggml_et_get_default_et_لار() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_منزل_BL2_Elf_لار =
        et_لار + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_لار =
        et_لار + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_لار = et_لار + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_لار  = et_لار + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.کارګر_Minion_Elf_لار  = et_لار + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_لار       = et_لار + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    ثابت std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_منزل_BL2_Elf_لار, sysEmuOptions.sp_BL2_Elf_لار,
        sysEmuOptions.machine_Minion_Elf_لار,          sysEmuOptions.master_Minion_Elf_لار,
        sysEmuOptions.کارګر_Minion_Elf_لار,           sysEmuOptions.executable_لار,
    };

    لكل (ثابت خپلکاری & file : required_files) {
        إذا (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.پرلپسې_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_دورې        = k_Sys_Emu_Max_دورې;
    sysEmuOptions.minion_Shires_ماسک = k_Sys_Emu_Minion_Shires_ماسک;
    sysEmuOptions.pu_Uart0_لار      = sysEmuOptions.پرلپسې_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_لار      = sysEmuOptions.پرلپسې_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_لار      = sysEmuOptions.پرلپسې_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_لار      = sysEmuOptions.پرلپسې_Dir + "spio_uart1_tx.log";
    sysEmuOptions.پیل_Gdb         = خطأ;
    sysEmuOptions.memcheck         = خطأ;

    أرجع sysEmuOptions;
}

// Forward declaration
ساكن فراغ ggml_et_driver_cleanup();

ساكن منطقي ggml_et_driver_init() {
    إذا (_drv.runtime != تش_اشاره) {
        أكد(_drv.device_layer != تش_اشاره);
    } وإلا {
        حاول {
#إذا defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#وإلا
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#د_که_پای  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ثابت حرف * profile_لار = getenv("GGML_ET_PROFILE");
            إذا (profile_لار) {
                std::string output_لار    = std::string(profile_لار) + "/et_runtime_trace.json";
                std::string kernel_id_لار = std::string(profile_لار) + "/kernel_id.json";

                _drv.profile_بهیر   = std::make_unique<std::ofstream>(output_لار);
                _drv.kernel_id_بهیر = std::make_unique<std::ofstream>(kernel_id_لار);
                إذا (!_drv.profile_بهیر->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_لار.c_str());
                    abort();
                }
                إذا (!_drv.kernel_id_بهیر->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_لار.c_str());
                    abort();
                }

                خپلکاری * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_بهیر, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = ریښتیا;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } امسك (ثابت std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            إذا (_drv.device_layer != تش_اشاره) {
                _drv.device_layer.reset();
            }
            إذا (_drv.runtime != تش_اشاره) {
                _drv.runtime.reset();
            }
            أرجع خطأ;
        }
    }
    أرجع ریښتیا;
}

ساكن std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    أرجع _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    أرجع _drv.runtime;
}

ساكن فراغ ggml_et_driver_cleanup() {
    إذا (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        خپلکاری * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = خطأ;

        إذا (_drv.profile_بهیر) {
            _drv.profile_بهیر->close();
            _drv.profile_بهیر.reset();
        }

        // Save kernel map
        إذا (_drv.kernel_id_بهیر && !_drv.kernel_map.تشه()) {
            خپلکاری & os = *_drv.kernel_id_بهیر;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            لكل (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].لومړی << "\": " << (صحيح) _drv.kernel_map[i].second;
                إذا (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_بهیر->close();
            _drv.kernel_id_بهیر.reset();
        }
    }
}

ساكن ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

ساكن فراغ ggml_backend_et_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_et_بفر_context * ctx = (ggml_backend_et_بفر_context *) بفر->context;
    إذا (ctx->معلومات != تش_اشاره) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        إذا (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->معلومات));
        }
    }
    ړنګول ctx;
}

ساكن فراغ * ggml_backend_et_بفر_get_بنسټ(ggml_backend_بفر_t بفر) {
    ggml_backend_et_بفر_context * ctx = (ggml_backend_et_بفر_context *) بفر->context;
    أرجع ctx->معلومات;
}

ساكن ggml_حالت ggml_backend_et_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    إذا (tensor->view_src != عدم) {
        أرجع GGML_حالت_SUCCESS;
    }

    ثابت size_t original_اندازه = ggml_nbytes(tensor);
    ثابت size_t padded_اندازه   = ggml_backend_buft_get_alloc_اندازه(بفر->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    إذا (padded_اندازه > original_اندازه) {
        ثابت size_t padding_اندازه = padded_اندازه - original_اندازه;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
        إذا (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            أرجع GGML_حالت_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->معلومات) + original_اندازه;
        إذا (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_اندازه)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            أرجع GGML_حالت_FAILED;
        }
    }

    أرجع GGML_حالت_SUCCESS;
}

ساكن فراغ ggml_backend_et_بفر_سېټ_tensor(ggml_backend_بفر_t بفر,
                                              ggml_tensor *         tensor,
                                              ثابت فراغ *          معلومات,
                                              size_t                دننه_واټن,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
    rt::بهیر_Id                     stream  = dev_ctx->default_بهیر;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->معلومات) + دننه_واټن;
    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(معلومات);

    rt::EventId event = runtime->memcpy_Host_منزل_Device(stream, src_ptr, dst_ptr, size, ریښتیا /*barrier*/);

    runtime->waitForEvent(event);
}

ساكن فراغ ggml_backend_et_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                              ثابت ggml_tensor *   tensor,
                                              فراغ *                معلومات,
                                              size_t                دننه_واټن,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
    rt::بهیر_Id                     stream  = dev_ctx->default_بهیر;

    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(tensor->معلومات) + دننه_واټن;
    std::byte *       dst_ptr = static_cast<std::byte *>(معلومات);

    rt::EventId event = runtime->memcpy_Device_منزل_Host(stream, src_ptr, dst_ptr, size, ریښتیا /*barrier*/);

    runtime->waitForEvent(event);
}

ساكن منطقي ggml_backend_et_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                              ثابت ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_ناکارول_شوی(بفر);
    GGML_ناکارول_شوی(src);
    GGML_ناکارول_شوی(dst);
    أرجع خطأ;
}

ساكن فراغ ggml_backend_et_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_backend_et_بفر_context * ctx = (ggml_backend_et_بفر_context *) بفر->context;

    إذا (ctx->size == 0 || ctx->معلومات == تش_اشاره) {
        أرجع;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
    إذا (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        أرجع;
    }

    // Use device-side memset kernel for efficient clearing
    إذا (!ggml_et_memset(dev_ctx, ctx->معلومات, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        أرجع;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

ساكن ثابت بنية ggml_backend_بفر_i ggml_backend_et_بفر_i = {
    /* .free_buffer     = */ ggml_backend_et_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_et_بفر_get_بنسټ,
    /* .init_tensor     = */ ggml_backend_et_بفر_init_tensor,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_et_بفر_سېټ_tensor,
    /* .get_tensor      = */ ggml_backend_et_بفر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_et_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_et_بفر_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت حرف * ggml_backend_et_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_ناکارول_شوی(buft);
    أرجع GGML_ET_NAME;
}

ساكن ggml_backend_بفر_t ggml_backend_et_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_et_بفر_type_context * btctx = (ggml_backend_et_بفر_type_context *) buft->context;

    ggml_backend_et_بفر_context * ctx = جديد ggml_backend_et_بفر_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        ړنګول ctx;
        أرجع تش_اشاره;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    إذا (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        ړنګول ctx;
        أرجع تش_اشاره;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->معلومات = runtime->mallocDevice(ctx->rtid, size);
    إذا (ctx->معلومات == تش_اشاره) {
        ړنګول ctx;
        أرجع تش_اشاره;
    }

    أرجع ggml_backend_بفر_init(buft, ggml_backend_et_بفر_i, ctx, size);
}

ساكن size_t ggml_backend_et_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime || !buft->device) {
        أرجع GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    أرجع prop.cache_کرښه_اندازه;
}

ساكن size_t ggml_backend_et_بفر_type_get_max_اندازه(ggml_backend_بفر_type_t buft) {
    إذا (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        أرجع dev_ctx->ټول_mem;
    }
    أرجع SIZE_MAX;
}

ساكن size_t ggml_backend_et_بفر_type_get_alloc_اندازه(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    GGML_ناکارول_شوی(buft);
    أرجع ggml_nbytes_pad(tensor);
}

ساكن منطقي ggml_backend_et_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    GGML_ناکارول_شوی(buft);
    أرجع خطأ;
}

ساكن ثابت بنية ggml_backend_بفر_type_i ggml_backend_et_بفر_type_i = {
    /* .get_name            = */ ggml_backend_et_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_et_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_بفر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_et_بفر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_et_بفر_type_is_host,
};

ساكن ثابت حرف * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_ناکارول_شوی(backend);
    أرجع GGML_ET_NAME;
}

ساكن فراغ ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    إذا (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        إذا (_drv.profiling_enabled) {
            خپلکاری kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        إذا (runtime) {
            إذا (dev_ctx->trace_بفر) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_بفر);
                dev_ctx->trace_بفر = تش_اشاره;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_بهیر(dev_ctx->default_بهیر);
            لكل (خپلکاری & خونه : dev_ctx->uberkernel.slots) {
                إذا (خونه.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, خونه.device_insts);
                    خونه.device_insts = تش_اشاره;
                }
                إذا (خونه.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, خونه.device_params);
                    خونه.device_params = تش_اشاره;
                }
                خونه.has_pending = خطأ;
            }
        }
    }

    ړنګول et_ctx;
    ړنګول backend;
}

ساكن ggml_backend_بفر_type_t ggml_backend_et_get_default_بفر_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    أرجع ggml_backend_et_بفر_type(et_ctx->devidx);
}

ساكن فراغ ggml_backend_et_سېټ_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ثابت فراغ *   معلومات,
                                             size_t         دننه_واټن,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::بهیر_Id                     stream  = dev_ctx->default_بهیر;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->معلومات) + دننه_واټن;
    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(معلومات);

    runtime->memcpy_Host_منزل_Device(stream, src_ptr, dst_ptr, size, ریښتیا /*barrier*/);
}

ساكن فراغ ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ثابت ggml_tensor * tensor,
                                             فراغ *              معلومات,
                                             size_t              دننه_واټن,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::بهیر_Id                     stream  = dev_ctx->default_بهیر;

    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(tensor->معلومات) + دننه_واټن;
    std::byte *       dst_ptr = static_cast<std::byte *>(معلومات);

    runtime->memcpy_Device_منزل_Host(stream, src_ptr, dst_ptr, size, ریښتیا /*barrier*/);
}

ساكن منطقي ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ثابت ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_ناکارول_شوی(backend_src);
    GGML_ناکارول_شوی(backend_dst);
    GGML_ناکارول_شوی(src);
    GGML_ناکارول_شوی(dst);
    أرجع خطأ;
}

ساكن فراغ ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_بهیر(dev_ctx->default_بهیر);

    خپلکاری errors = runtime->retrieve_بهیر_Errors(dev_ctx->default_بهیر);
    إذا (errors.تشه()) {
        أرجع;
    }
    لكل (ثابت خپلکاری & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (صحيح) err.error_کوډ,
                       (صحيح) err.errorContext_.value()[0].type_);
    }
    abort();
}

ساكن منطقي ggml_et_can_fuse(ثابت ggml_cgraph * cgraph, صحيح غوټه_idx, std::initializer_list<ggml_op> ops) {
    إذا (!ggml_can_fuse(cgraph, غوټه_idx, ops)) {
        أرجع خطأ;
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_جمع_کړه) {
        ثابت ggml_tensor * mm  = cgraph->nodes[غوټه_idx];
        ثابت ggml_tensor * جمع_کړه_2 = cgraph->nodes[غوټه_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        إذا (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        إذا (جمع_کړه_2->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        إذا (جمع_کړه_2->src[0] != mm && جمع_کړه_2->src[1] != mm) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * bias = (جمع_کړه_2->src[0] == mm) ? جمع_کړه_2->src[1] : جمع_کړه_2->src[0];

        إذا (bias->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // No broadcasting: bias shape must equal MM output shape.
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            إذا (bias->ne[i] != mm->ne[i]) {
                أرجع خطأ;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        إذا (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            أرجع خطأ;
        }
        لكل (صحيح i = 0; i < GGML_MAX_DIMS; ++i) {
            إذا ((int64_t) bias->nb[i] != (int64_t) جمع_کړه_2->nb[i]) {
                أرجع خطأ;
            }
        }
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ثابت ggml_tensor * rms_norm = cgraph->nodes[غوټه_idx];
        ثابت ggml_tensor * mul      = cgraph->nodes[غوټه_idx + 1];

        // ET only supports F32
        إذا (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        ثابت ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        إذا (weights->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        إذا (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_لیکې(weights)) {
            أرجع خطأ;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        إذا (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            أرجع خطأ;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        إذا (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            أرجع خطأ;
        }
    }

    أرجع ریښتیا;
}

ساكن ggml_حالت ggml_backend_et_ګراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_ګراف(&dev_ctx->uberkernel);

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * غوټه = cgraph->nodes[i];

        إذا (غوټه->op == GGML_OP_NONE || غوټه->op == GGML_OP_VIEW || غوټه->op == GGML_OP_RESHAPE ||
            غوټه->op == GGML_OP_ترتیب_بدلول || غوټه->op == GGML_OP_TRANSPOSE) {
            استمر;
        }

        // --- Fusion checks (before regular dispatch) ---
        إذا (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, غوټه, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            استمر;
        }
        إذا (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جمع_کړه })) {
            ggml_et_op_mul_mat(dev_ctx, غوټه, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            استمر;
        }

        بدل (غوټه->op) {
            حالة GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_مجموعه_لیکې:
                ggml_et_op_مجموعه_لیکې(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_منځنۍ:
                ggml_et_op_منځنۍ(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_جمع_کړه:
                ggml_et_op_جمع_کړه(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_ډله_NORM:
                ggml_et_op_ډله_norm(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_IM2_ستون:
                ggml_et_op_im2col(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_GET_لیکې:
                ggml_et_op_get_لیکې(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_SSM_لټون:
                ggml_et_op_ssm_لټون(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_سېټ_لیکې:
                ggml_et_op_سېټ_لیکې(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_سېټ:
                ggml_et_op_سېټ(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, غوټه);
                اكسر;

            حالة GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, غوټه);
                اكسر;

            افتراضي:
                ggml_et_uberkernel_abort_ګراف(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(غوټه->op));
                أرجع GGML_حالت_FAILED;
        }

        إذا (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_ګراف(&dev_ctx->uberkernel);
            أرجع GGML_حالت_FAILED;
        }
    }

    إذا (!ggml_et_uberkernel_پای_ټکی_ګراف(dev_ctx)) {
        ggml_et_uberkernel_abort_ګراف(&dev_ctx->uberkernel);
        أرجع GGML_حالت_FAILED;
    }

    أرجع GGML_حالت_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
ساكن منطقي et_ggml_is_لیکه_contiguous(ثابت ggml_tensor * t) {
    أرجع t->nb[0] == ggml_type_اندازه(t->type);
}

ساكن منطقي ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_ناکارول_شوی(dev);

    منطقي supported = خطأ;
    بدل (op->op) {
        حالة GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == حجم(عائم) && ggml_is_contiguous(op);
            اكسر;
        حالة GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_مجموعه_لیکې:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_منځنۍ:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == حجم(عائم) && op->src[0]->nb[0] == حجم(عائم)) {
                بدل (ggml_get_unary_op(op)) {
                    حالة GGML_UNARY_OP_ABS:
                    حالة GGML_UNARY_OP_SGN:
                    حالة GGML_UNARY_OP_NEG:
                    حالة GGML_UNARY_OP_ګام:
                    حالة GGML_UNARY_OP_TANH:
                    حالة GGML_UNARY_OP_ELU:
                    حالة GGML_UNARY_OP_RELU:
                    حالة GGML_UNARY_OP_SIGMOID:
                    حالة GGML_UNARY_OP_GELU:
                    حالة GGML_UNARY_OP_GELU_چټک:
                    حالة GGML_UNARY_OP_SILU:
                    حالة GGML_UNARY_OP_HARDSWISH:
                    حالة GGML_UNARY_OP_HARDSIGMOID:
                    حالة GGML_UNARY_OP_EXP:
                    حالة GGML_UNARY_OP_EXPM1:
                    حالة GGML_UNARY_OP_SOFTPLUS:
                    حالة GGML_UNARY_OP_GELU_ERF:
                    حالة GGML_UNARY_OP_FLOOR:
                    حالة GGML_UNARY_OP_CEIL:
                    حالة GGML_UNARY_OP_پړاو:
                    حالة GGML_UNARY_OP_TRUNC:
                        supported = ریښتیا;
                        اكسر;
                    افتراضي:
                        اكسر;
                }
            }
            اكسر;
        حالة GGML_OP_MUL:
        حالة GGML_OP_جمع_کړه:
        حالة GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == حجم(عائم) &&
                        op->src[0]->nb[0] == حجم(عائم) &&
                        (op->src[1]->nb[0] == حجم(عائم) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * حجم(عائم);
            اكسر;
        حالة GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            إذا (op->type == GGML_TYPE_F32 &&
                ((op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                 (op->src[0]->type == GGML_TYPE_F16 && op->src[1]->type == GGML_TYPE_F16)) &&
                op->ne[0] % 16 == 0 &&          // dst row length for tensor-store path
                op->src[0]->ne[1] % 16 == 0 &&  // m
                op->src[0]->ne[0] % 16 == 0 &&  // k
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Special path for the FP32 TensorFMA kernel
                // Limitation - generic kernels can tolerate non-cache-aligned dst rows
                // because they publish each output element atomically. The matrix
                // engine path still uses tiled tensor stores, so keep dst rows aligned.
                // The m edge is difficult to do because of the 4 conseqtive load hardware limitation
                // And the k edge is impossible because that is encoded as `stride & 0xFFFFFFFFFFC0ULL` which becomes 0 for stride 16 (4x FP32) :(
                // FIXME: Right now this overwrites the mul_mat_f32 kernel - whatever. Fix later. Demo code
                supported = ریښتیا;
            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                منطقي src0_لومړی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_لومړی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي dst_لومړی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                // Check destination stride ordering (only for dimensions with ne > 1)
                منطقي dst_properly_ordered = ریښتیا;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                supported = src0_لومړی_dim_contiguous && src1_لومړی_dim_contiguous && dst_لومړی_dim_contiguous &&
                            dst_properly_ordered;
            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                منطقي src0_لومړی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_لومړی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي dst_لومړی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                منطقي dst_properly_ordered = ریښتیا;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                supported = src0_لومړی_dim_contiguous && src1_لومړی_dim_contiguous && dst_لومړی_dim_contiguous &&
                            dst_properly_ordered;

            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                منطقي src0_لومړی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_لومړی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي dst_لومړی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                منطقي dst_properly_ordered = ریښتیا;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                supported = src0_لومړی_dim_contiguous && src1_لومړی_dim_contiguous && dst_لومړی_dim_contiguous &&
                            dst_properly_ordered;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            إذا (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                منطقي src0_لومړی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_لومړی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي src2_لومړی_dim_contiguous = (op->src[2]->nb[0] == ggml_type_اندازه(op->src[2]->type));
                منطقي dst_لومړی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                // Check destination stride ordering (only for dimensions with ne > 1)
                منطقي dst_properly_ordered = ریښتیا;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                منطقي dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_لومړی_dim_contiguous && src1_لومړی_dim_contiguous && src2_لومړی_dim_contiguous &&
                            dst_لومړی_dim_contiguous && dst_properly_ordered && dims_valid;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_لیکه_contiguous(op->src[0])) {
                ثابت صحيح  mode             = ggml_get_op_params_i32(op, 2);
                ثابت صحيح  ndims            = ggml_get_op_params_i32(op, 1);
                ثابت منطقي is_نورمال        = mode == GGML_ROPE_TYPE_نورمال;
                ثابت منطقي is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ثابت منطقي is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ثابت منطقي zero_view_دننه_واټن = op->src[0]->view_src == تش_اشاره || op->src[0]->view_offs == 0;
                ثابت منطقي has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ثابت منطقي zero_rot_دننه_واټن  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_دننه_واټن && zero_rot_دننه_واټن && ndims <= 512 &&
                    (is_نورمال || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_لیکه_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_لیکه_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_لیکه_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_ډله_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_لیکه_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            اكسر;
        حالة GGML_OP_IM2_ستون:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_اندازه(op->type) && op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type);
            اكسر;
        حالة GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                إذا (!op->src[0] || !op->src[1]) {
                    supported = خطأ;
                    اكسر;
                }
                إذا (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = خطأ;
                    اكسر;
                }
                إذا (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = خطأ;
                    اكسر;
                }

                ثابت ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                ثابت ggml_tensor * دننه  = op->src[1];  // [W,  H,  Cin, N]
                ثابت int32_t       s0  = ggml_get_op_params_i32(op, 0);
                ثابت int32_t       s1  = ggml_get_op_params_i32(op, 1);
                ثابت int32_t       p0  = ggml_get_op_params_i32(op, 2);
                ثابت int32_t       p1  = ggml_get_op_params_i32(op, 3);
                ثابت int32_t       d0  = ggml_get_op_params_i32(op, 4);
                ثابت int32_t       d1  = ggml_get_op_params_i32(op, 5);

                ثابت int64_t Kw   = flt->ne[0];
                ثابت int64_t Kh   = flt->ne[1];
                ثابت int64_t Cin  = flt->ne[2];
                ثابت int64_t Cout = flt->ne[3];
                ثابت int64_t H    = دننه->ne[1];
                (فراغ) دننه->ne[0];

                إذا (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || دننه->ne[3] != 1) {
                    supported = خطأ;
                    اكسر;
                }
                ثابت int64_t OW = op->ne[0];
                ثابت int64_t OH = op->ne[1];
                إذا (OW <= 0 || OH <= 0) {
                    supported = خطأ;
                    اكسر;
                }
                (فراغ) p0;
                (فراغ) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                ثابت int64_t Hp            = H + 2 * p1;
                ثابت int64_t OW_pad        = (OW + 15) & ~15;
                ثابت int64_t Wp_a          = OW_pad;
                ثابت منطقي    need_stage    = (OW % 16 != 0);
                ثابت int64_t stage_بایټونه   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ثابت int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ثابت int64_t scratch_بایټونه = 32 * 2 * 16 * 16 * 4;
                ثابت int64_t budget        = L2SCP_BUDGET - stage_بایټونه - scratch_بایټونه;
                ثابت int64_t فی_KT_بایټونه  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ثابت int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                إذا (K_TILES * فی_KT_بایټونه <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } وإلا {
                    chunk_KT_calc = K_TILES;
                    طالما (chunk_KT_calc > 1 && 2 * chunk_KT_calc * فی_KT_بایټونه > budget) {
                        chunk_KT_calc--;
                    }
                    طالما (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    إذا (chunk_KT_calc < 1) {
                        supported = خطأ;
                        اكسر;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                إذا (n_chunks_calc > 1) {
                    ثابت int64_t M_TILES     = Cout / 16;
                    ثابت int64_t w_tiles     = (OW + 15) / 16;
                    ثابت int64_t ټول_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ثابت int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    إذا (ټول_tiles > max_workers) {
                        supported = خطأ;
                        اكسر;
                    }
                }

                supported = ریښتیا;
                اكسر;
            }
        حالة GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            اكسر;
        حالة GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                ثابت منطقي  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_چټک || glu_type == GGML_GLU_OP_REGLU;

                إذا (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } وإلا {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_SOFT_MAX:
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                إذا (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    إذا (!supported) {
                        اكسر;
                    }
                }
                // Check optional sinks tensor (F32 only)
                إذا (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } وإلا {
                    supported = ریښتیا;
                }
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_SSM_لټون:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == حجم(عائم) && op->src[4]->nb[0] == حجم(عائم) &&
                        op->src[5]->nb[0] == حجم(عائم) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * حجم(عائم) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * حجم(عائم) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * حجم(عائم) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            اكسر;
        حالة GGML_OP_FLASH_ATTN_EXT:
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == تش_اشاره &&
                ggml_is_contiguous_لیکې(op) && ggml_is_contiguous_لیکې(op->src[0])) {
                عائم max_bias      = 0.0f;
                عائم logit_softcap = 0.0f;
                حافظه_نقل(&max_bias, (ثابت عائم *) op->op_params + 1, حجم(max_bias));
                حافظه_نقل(&logit_softcap, (ثابت عائم *) op->op_params + 2, حجم(logit_softcap));

                ثابت ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                منطقي ماسک_ok = (op->src[3] == تش_اشاره) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                ثابت int64_t nhq = op->src[0]->ne[2];
                ثابت int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                ثابت size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                ثابت size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                ثابت منطقي me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && ماسک_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == حجم(عائم) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == حجم(عائم) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_GET_لیکې:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            إذا (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            إذا ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                إذا (op->معلومات && op->src[0]->معلومات && op->معلومات == op->src[0]->معلومات) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = خطأ;
                } وإلا {
                    supported = ریښتیا;
                }
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            إذا (op->src[0]) {
                ثابت int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                إذا (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = ریښتیا;
                } وإلا إذا ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    إذا (op->معلومات && op->src[0]->معلومات && op->معلومات == op->src[0]->معلومات) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = خطأ;
                    } وإلا {
                        supported = ریښتیا;
                    }
                } وإلا إذا (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = ریښتیا;
                } وإلا {
                    supported = خطأ;
                }
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_CONCAT:
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                ثابت int32_t dim = ((ثابت int32_t *) op->op_params)[0];
                إذا (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = ریښتیا;
                } وإلا إذا (dim == 0 && ((op->src[0]->nb[0] % حجم(عائم) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % حجم(عائم) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = ریښتیا;
                } وإلا إذا (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = ریښتیا;
                }
            }
            اكسر;
        حالة GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == حجم(عائم) &&
                        op->src[1]->nb[0] == حجم(عائم) && op->src[0]->nb[1] == op->src[0]->ne[0] * حجم(عائم) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * حجم(عائم) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            اكسر;
        حالة GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == حجم(عائم)) {
                ثابت int32_t lp0      = ((ثابت int32_t *) op->op_params)[0];
                ثابت int32_t rp0      = ((ثابت int32_t *) op->op_params)[1];
                ثابت منطقي    circular = (منطقي) ((ثابت int32_t *) op->op_params)[8];
                إذا (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = ریښتیا;
                } وإلا {
                    supported = خطأ;
                }
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = ریښتیا;
            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = ریښتیا;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            اكسر;
        حالة GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            اكسر;
        حالة GGML_OP_سېټ:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ثابت منطقي   inplace = (منطقي) ((ثابت int32_t *) op->op_params)[4];
                ثابت size_t nb1     = ((ثابت int32_t *) op->op_params)[0];
                ثابت size_t nb2     = ((ثابت int32_t *) op->op_params)[1];
                ثابت size_t nb3     = ((ثابت int32_t *) op->op_params)[2];
                ثابت size_t دننه_واټن  = ((ثابت int32_t *) op->op_params)[3];
                ثابت size_t nb0     = ggml_element_اندازه(op);
                ثابت size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ثابت size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ثابت size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ثابت size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ثابت منطقي view_bounds_ok = دننه_واټن + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ثابت منطقي cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (دننه_واټن % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            اكسر;
        حالة GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = ریښتیا;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = ریښتیا;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&                               // k
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&                               // v
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&                               // g
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&                               // beta
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&                               // state
                op->src[2]->ne[0] % 8 == 0 &&                                                    // S_v multiple of 8
                (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[2]->ne[0]) &&  // g is scalar or per-element
                op->src[4]->ne[0] == 1 &&                                              // beta is scalar per position
                et_ggml_is_لیکه_contiguous(op->src[0]) && et_ggml_is_لیکه_contiguous(op->src[1]) &&
                et_ggml_is_لیکه_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = ریښتیا;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_VIEW:
        حالة GGML_OP_ترتیب_بدلول:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = ریښتیا;
            اكسر;
        حالة GGML_OP_سېټ_لیکې:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            إذا (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_لیکې(op) &&
                ggml_is_contiguous_لیکې(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = ریښتیا;
            اكسر;
        افتراضي:
            supported = خطأ;
            اكسر;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    أرجع supported;
}

ساكن منطقي ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    GGML_ناکارول_شوی(dev);
    أرجع buft->iface.get_name == ggml_backend_et_بفر_type_get_name;
}

ساكن منطقي ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    إذا (op->op == GGML_OP_GET_لیکې) {
        أرجع خطأ;
    }
    أرجع ریښتیا;

    GGML_ناکارول_شوی(dev);
}

ساكن ثابت بنية ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_سېټ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_et_ګراف_compute,
    /* .event_record            = */ عدم,
    /* .event_wait              = */ عدم,
    /* .graph_optimize          = */ عدم,
};

ساكن ثابت حرف * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع dev_ctx->name.c_str();
}

ساكن ثابت حرف * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع dev_ctx->desc.c_str();
}

ساكن فراغ ggml_backend_et_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * ټول) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *حرر                                    = dev_ctx->ټول_mem;
    *ټول                                   = dev_ctx->ټول_mem;
}

ساكن تعداد ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_ناکارول_شوی(dev);
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_et_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    GGML_ناکارول_شوی(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_ټول);
    props->device_id = عدم;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ ریښتیا,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ خطأ,
        /* .mmap_support          = */ ریښتیا,
    };
}

ساكن ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_ناکارول_شوی(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع ggml_backend_et_init(dev_ctx->devidx);
}

ساكن ggml_backend_بفر_type_t ggml_backend_et_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع dev_ctx->buftype;
}

ساكن ggml_backend_بفر_type_t ggml_backend_et_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    GGML_ناکارول_شوی(dev);
    أرجع ggml_backend_cpu_بفر_type();
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_حافظه,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ عدم,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ عدم,
    /* .event_free        = */ عدم,
    /* .event_synchronize = */ عدم,
};

/*
  Backend Registry.
*/

ساكن ثابت حرف * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ناکارول_شوی(reg);
    أرجع GGML_ET_NAME;
}

ساكن size_t ggml_backend_et_reg_get_device_شمېر(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    أرجع ctx->devices.size();
}

ساكن ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    إذا (devidx >= ctx->devices.size()) {
        أرجع تش_اشاره;
    }
    أرجع ctx->devices[devidx];
}

ساكن فراغ * ggml_backend_et_get_proc_پته(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_ناکارول_شوی(reg);
    GGML_ناکارول_شوی(name);
    أرجع تش_اشاره;
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_شمېر,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_پته,
};

ggml_backend_reg_t ggml_backend_et_reg(فراغ) {
    ساكن ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = جديد ggml_backend_et_reg_ctx;

        إذا (!ggml_et_driver_init()) {
            أرجع تش_اشاره;
        }

        ggml_backend_reg_t r = جديد ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_نسخه,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ تش_اشاره,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        لكل (صحيح i = 0; i < ggml_et_devicelayer()->get_Devices_شمېر(); i++) {
            ggml_backend_dev_t dev = جديد ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ تش_اشاره  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = جديد ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->ټول_mem                       = static_cast<size_t>(prop.حافظه_اندازه);
            {
                ثابت حرف * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_بفر_type_context * bufty_ctx = جديد ggml_backend_et_بفر_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = جديد ggml_backend_بفر_type{ /* .iface   = */ ggml_backend_et_بفر_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_بهیر = ggml_et_runtime()->create_بهیر(rtid);

            dev_ctx->trace_بفر = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_بفر_اندازه);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            لكل (خپلکاری & خونه : dev_ctx->uberkernel.slots) {
                خونه.insts.reserve(256);
                خونه.params_blob.reserve(1 << 20);
                خونه.device_insts_ظرفیت  = 256 * حجم(ggml_et_uberkernel_inst);
                خونه.device_params_ظرفیت = 1 << 20;
                خونه.device_insts           = ggml_et_runtime()->mallocDevice(rtid, خونه.device_insts_ظرفیت);
                خونه.device_params          = ggml_et_runtime()->mallocDevice(rtid, خونه.device_params_ظرفیت);
                إذا (خونه.device_insts == تش_اشاره) {
                    خونه.device_insts_ظرفیت = 0;
                }
                إذا (خونه.device_params == تش_اشاره) {
                    خونه.device_params_ظرفیت = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        أرجع r;
    }();

    أرجع _reg;
}

ggml_guid_t ggml_backend_et_guid(فراغ) {
    ساكن ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    أرجع &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    إذا (!ggml_et_driver_init()) {
        أرجع تش_اشاره;
    }

    إذا (devidx >= (size_t) ggml_backend_et_get_device_شمېر()) {
        أرجع تش_اشاره;
    }

    ggml_backend_et_context * ctx = جديد ggml_backend_et_context;
    ctx->devidx                   = (صحيح) devidx;

    ggml_backend_t backend = جديد ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    أرجع backend;
}

منطقي ggml_backend_is_et(ggml_backend_t backend) {
    أرجع backend != عدم && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

صحيح ggml_backend_et_get_device_شمېر(فراغ) {
    أرجع ggml_backend_et_reg_get_device_شمېر(ggml_backend_et_reg());
}

فراغ ggml_backend_et_get_device_description(صحيح devidx, حرف * description, size_t description_اندازه) {
    إذا (devidx < 0 || devidx >= ggml_backend_et_get_device_شمېر()) {
        snprintf(description, description_اندازه, "ET Device %d (invalid)", devidx);
        أرجع;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_اندازه, "%s", dev_ctx->desc.c_str());
}

فراغ ggml_backend_et_get_device_حافظه(صحيح devidx, size_t * حرر, size_t * ټول) {
    إذا (devidx < 0 || devidx >= ggml_backend_et_get_device_شمېر()) {
        *حرر  = 0;
        *ټول = 0;
        أرجع;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_حافظه(dev, حرر, ټول);
}

ggml_backend_بفر_type_t ggml_backend_et_بفر_type(size_t dev_عدد) {
    إذا (dev_عدد >= (size_t) ggml_backend_et_get_device_شمېر()) {
        أرجع تش_اشاره;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_عدد);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع dev_ctx->buftype;
}

ggml_backend_بفر_type_t ggml_backend_et_host_بفر_type(فراغ) {
    ساكن ggml_backend_بفر_type host_بفر_type = {
        /* .iface   = */ ggml_backend_et_بفر_type_i,
        /* .device  = */ تش_اشاره,
        /* .context = */ تش_اشاره,
    };
    أرجع &host_بفر_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

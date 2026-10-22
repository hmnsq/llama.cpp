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
#پایان_اگر

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
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (دراز دراز) ggtensor->ne[0],
            (دراز دراز) ggtensor->ne[1], (دراز دراز) ggtensor->ne[2], (دراز دراز) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->داده, spaces.c_str(), (فراغ *) ggtensor->src[0]);
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
        GGML_ASSERT((بیعلامت) n < حجم(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    إذا (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

ساكن بنية ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_جریان;
    std::unique_ptr<std::ofstream>                    kernel_id_جریان;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    منطقي                                              profiling_enabled = خطأ;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
ساكن std::string ggml_et_get_default_et_مسیر() {
    // List of environment variables to check in order of preference
    ثابت حرف * ثابت env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    لكل (ثابت حرف * var : env_vars) {
        إذا (ثابت حرف * et_مسیر = std::getenv(var)) {
            إذا (et_مسیر && *et_مسیر != '\0') {
                أرجع fs::path(et_مسیر).string();
            }
        }
    }

    // Otherwise assume default
    أرجع fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
ساكن مضمن خودکار ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_چرخهها        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_ماسک = 0x1FFFFFFFFu;
    ثابت std::string  et_مسیر                 = ggml_et_get_default_et_مسیر() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_مقصد_BL2_Elf_مسیر =
        et_مسیر + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_مسیر =
        et_مسیر + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_مسیر = et_مسیر + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_مسیر  = et_مسیر + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.کارگر_Minion_Elf_مسیر  = et_مسیر + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_مسیر       = et_مسیر + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    ثابت std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_مقصد_BL2_Elf_مسیر, sysEmuOptions.sp_BL2_Elf_مسیر,
        sysEmuOptions.machine_Minion_Elf_مسیر,          sysEmuOptions.master_Minion_Elf_مسیر,
        sysEmuOptions.کارگر_Minion_Elf_مسیر,           sysEmuOptions.executable_مسیر,
    };

    لكل (ثابت خودکار & file : required_files) {
        إذا (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.پیاپی_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_چرخهها        = k_Sys_Emu_Max_چرخهها;
    sysEmuOptions.minion_Shires_ماسک = k_Sys_Emu_Minion_Shires_ماسک;
    sysEmuOptions.pu_Uart0_مسیر      = sysEmuOptions.پیاپی_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_مسیر      = sysEmuOptions.پیاپی_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_مسیر      = sysEmuOptions.پیاپی_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_مسیر      = sysEmuOptions.پیاپی_Dir + "spio_uart1_tx.log";
    sysEmuOptions.شروع_Gdb         = خطأ;
    sysEmuOptions.memcheck         = خطأ;

    أرجع sysEmuOptions;
}

// Forward declaration
ساكن فراغ ggml_et_driver_cleanup();

ساكن منطقي ggml_et_driver_init() {
    إذا (_drv.runtime != اشاره_تهی) {
        أكد(_drv.device_layer != اشاره_تهی);
    } وإلا {
        حاول {
#إذا defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#وإلا
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#پایان_اگر  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ثابت حرف * profile_مسیر = getenv("GGML_ET_PROFILE");
            إذا (profile_مسیر) {
                std::string output_مسیر    = std::string(profile_مسیر) + "/et_runtime_trace.json";
                std::string kernel_id_مسیر = std::string(profile_مسیر) + "/kernel_id.json";

                _drv.profile_جریان   = std::make_unique<std::ofstream>(output_مسیر);
                _drv.kernel_id_جریان = std::make_unique<std::ofstream>(kernel_id_مسیر);
                إذا (!_drv.profile_جریان->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_مسیر.c_str());
                    abort();
                }
                إذا (!_drv.kernel_id_جریان->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_مسیر.c_str());
                    abort();
                }

                خودکار * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_جریان, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = درست;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } امسك (ثابت std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            إذا (_drv.device_layer != اشاره_تهی) {
                _drv.device_layer.reset();
            }
            إذا (_drv.runtime != اشاره_تهی) {
                _drv.runtime.reset();
            }
            أرجع خطأ;
        }
    }
    أرجع درست;
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
        خودکار * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = خطأ;

        إذا (_drv.profile_جریان) {
            _drv.profile_جریان->close();
            _drv.profile_جریان.reset();
        }

        // Save kernel map
        إذا (_drv.kernel_id_جریان && !_drv.kernel_map.خالی()) {
            خودکار & os = *_drv.kernel_id_جریان;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            لكل (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].اولی << "\": " << (صحيح) _drv.kernel_map[i].second;
                إذا (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_جریان->close();
            _drv.kernel_id_جریان.reset();
        }
    }
}

ساكن ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

ساكن فراغ ggml_backend_et_میانگیر_free_میانگیر(ggml_backend_میانگیر_t میانگیر) {
    ggml_backend_et_میانگیر_context * ctx = (ggml_backend_et_میانگیر_context *) میانگیر->context;
    إذا (ctx->داده != اشاره_تهی) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        إذا (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->داده));
        }
    }
    حذف ctx;
}

ساكن فراغ * ggml_backend_et_میانگیر_get_پایه(ggml_backend_میانگیر_t میانگیر) {
    ggml_backend_et_میانگیر_context * ctx = (ggml_backend_et_میانگیر_context *) میانگیر->context;
    أرجع ctx->داده;
}

ساكن ggml_وضعیت ggml_backend_et_میانگیر_init_tensor(ggml_backend_میانگیر_t میانگیر, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    إذا (tensor->view_src != عدم) {
        أرجع GGML_وضعیت_SUCCESS;
    }

    ثابت size_t original_اندازه = ggml_nbytes(tensor);
    ثابت size_t padded_اندازه   = ggml_backend_buft_get_alloc_اندازه(میانگیر->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    إذا (padded_اندازه > original_اندازه) {
        ثابت size_t padding_اندازه = padded_اندازه - original_اندازه;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) میانگیر->buft->device->context;
        إذا (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            أرجع GGML_وضعیت_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->داده) + original_اندازه;
        إذا (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_اندازه)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            أرجع GGML_وضعیت_FAILED;
        }
    }

    أرجع GGML_وضعیت_SUCCESS;
}

ساكن فراغ ggml_backend_et_میانگیر_مجموعه_tensor(ggml_backend_میانگیر_t میانگیر,
                                              ggml_tensor *         tensor,
                                              ثابت فراغ *          داده,
                                              size_t                جابه_جایی,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) میانگیر->buft->device->context;
    rt::جریان_Id                     stream  = dev_ctx->default_جریان;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->داده) + جابه_جایی;
    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(داده);

    rt::EventId event = runtime->memcpy_Host_مقصد_Device(stream, src_ptr, dst_ptr, size, درست /*barrier*/);

    runtime->waitForEvent(event);
}

ساكن فراغ ggml_backend_et_میانگیر_get_tensor(ggml_backend_میانگیر_t میانگیر,
                                              ثابت ggml_tensor *   tensor,
                                              فراغ *                داده,
                                              size_t                جابه_جایی,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) میانگیر->buft->device->context;
    rt::جریان_Id                     stream  = dev_ctx->default_جریان;

    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(tensor->داده) + جابه_جایی;
    std::byte *       dst_ptr = static_cast<std::byte *>(داده);

    rt::EventId event = runtime->memcpy_Device_مقصد_Host(stream, src_ptr, dst_ptr, size, درست /*barrier*/);

    runtime->waitForEvent(event);
}

ساكن منطقي ggml_backend_et_میانگیر_cpy_tensor(ggml_backend_میانگیر_t میانگیر,
                                              ثابت ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_استفادهنشده(میانگیر);
    GGML_استفادهنشده(src);
    GGML_استفادهنشده(dst);
    أرجع خطأ;
}

ساكن فراغ ggml_backend_et_میانگیر_clear(ggml_backend_میانگیر_t میانگیر, uint8_t value) {
    ggml_backend_et_میانگیر_context * ctx = (ggml_backend_et_میانگیر_context *) میانگیر->context;

    إذا (ctx->size == 0 || ctx->داده == اشاره_تهی) {
        أرجع;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) میانگیر->buft->device->context;
    إذا (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        أرجع;
    }

    // Use device-side memset kernel for efficient clearing
    إذا (!ggml_et_memset(dev_ctx, ctx->داده, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        أرجع;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

ساكن ثابت بنية ggml_backend_میانگیر_i ggml_backend_et_میانگیر_i = {
    /* .free_buffer     = */ ggml_backend_et_میانگیر_free_میانگیر,
    /* .get_base        = */ ggml_backend_et_میانگیر_get_پایه,
    /* .init_tensor     = */ ggml_backend_et_میانگیر_init_tensor,
    /* .memset_tensor   = */ عدم,
    /* .set_tensor      = */ ggml_backend_et_میانگیر_مجموعه_tensor,
    /* .get_tensor      = */ ggml_backend_et_میانگیر_get_tensor,
    /* .set_tensor_2d   = */ عدم,
    /* .get_tensor_2d   = */ عدم,
    /* .cpy_tensor      = */ ggml_backend_et_میانگیر_cpy_tensor,
    /* .clear           = */ ggml_backend_et_میانگیر_clear,
    /* .reset           = */ عدم,
};

ساكن ثابت حرف * ggml_backend_et_میانگیر_type_get_name(ggml_backend_میانگیر_type_t buft) {
    GGML_استفادهنشده(buft);
    أرجع GGML_ET_NAME;
}

ساكن ggml_backend_میانگیر_t ggml_backend_et_میانگیر_type_alloc_میانگیر(ggml_backend_میانگیر_type_t buft, size_t size) {
    ggml_backend_et_میانگیر_type_context * btctx = (ggml_backend_et_میانگیر_type_context *) buft->context;

    ggml_backend_et_میانگیر_context * ctx = جديد ggml_backend_et_میانگیر_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        حذف ctx;
        أرجع اشاره_تهی;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    إذا (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        حذف ctx;
        أرجع اشاره_تهی;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->داده = runtime->mallocDevice(ctx->rtid, size);
    إذا (ctx->داده == اشاره_تهی) {
        حذف ctx;
        أرجع اشاره_تهی;
    }

    أرجع ggml_backend_میانگیر_init(buft, ggml_backend_et_میانگیر_i, ctx, size);
}

ساكن size_t ggml_backend_et_میانگیر_type_get_alignment(ggml_backend_میانگیر_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime || !buft->device) {
        أرجع GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    أرجع prop.cache_سطر_اندازه;
}

ساكن size_t ggml_backend_et_میانگیر_type_get_max_اندازه(ggml_backend_میانگیر_type_t buft) {
    إذا (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        أرجع dev_ctx->مجموع_mem;
    }
    أرجع SIZE_MAX;
}

ساكن size_t ggml_backend_et_میانگیر_type_get_alloc_اندازه(ggml_backend_میانگیر_type_t buft, ثابت ggml_tensor * tensor) {
    GGML_استفادهنشده(buft);
    أرجع ggml_nbytes_pad(tensor);
}

ساكن منطقي ggml_backend_et_میانگیر_type_is_host(ggml_backend_میانگیر_type_t buft) {
    GGML_استفادهنشده(buft);
    أرجع خطأ;
}

ساكن ثابت بنية ggml_backend_میانگیر_type_i ggml_backend_et_میانگیر_type_i = {
    /* .get_name            = */ ggml_backend_et_میانگیر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_میانگیر_type_alloc_میانگیر,
    /* .alloc_buffer_n      = */ عدم,
    /* .get_alignment       = */ ggml_backend_et_میانگیر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_میانگیر_type_get_max_اندازه,
    /* .get_alloc_size      = */ ggml_backend_et_میانگیر_type_get_alloc_اندازه,
    /* .get_alloc_size_n    = */ عدم,
    /* .is_host             = */ ggml_backend_et_میانگیر_type_is_host,
};

ساكن ثابت حرف * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_استفادهنشده(backend);
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
            خودکار kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        إذا (runtime) {
            إذا (dev_ctx->trace_میانگیر) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_میانگیر);
                dev_ctx->trace_میانگیر = اشاره_تهی;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_جریان(dev_ctx->default_جریان);
            لكل (خودکار & جایگاه : dev_ctx->uberkernel.slots) {
                إذا (جایگاه.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, جایگاه.device_insts);
                    جایگاه.device_insts = اشاره_تهی;
                }
                إذا (جایگاه.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, جایگاه.device_params);
                    جایگاه.device_params = اشاره_تهی;
                }
                جایگاه.has_pending = خطأ;
            }
        }
    }

    حذف et_ctx;
    حذف backend;
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_et_get_default_میانگیر_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    أرجع ggml_backend_et_میانگیر_type(et_ctx->devidx);
}

ساكن فراغ ggml_backend_et_مجموعه_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ثابت فراغ *   داده,
                                             size_t         جابه_جایی,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::جریان_Id                     stream  = dev_ctx->default_جریان;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->داده) + جابه_جایی;
    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(داده);

    runtime->memcpy_Host_مقصد_Device(stream, src_ptr, dst_ptr, size, درست /*barrier*/);
}

ساكن فراغ ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ثابت ggml_tensor * tensor,
                                             فراغ *              داده,
                                             size_t              جابه_جایی,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::جریان_Id                     stream  = dev_ctx->default_جریان;

    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(tensor->داده) + جابه_جایی;
    std::byte *       dst_ptr = static_cast<std::byte *>(داده);

    runtime->memcpy_Device_مقصد_Host(stream, src_ptr, dst_ptr, size, درست /*barrier*/);
}

ساكن منطقي ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ثابت ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_استفادهنشده(backend_src);
    GGML_استفادهنشده(backend_dst);
    GGML_استفادهنشده(src);
    GGML_استفادهنشده(dst);
    أرجع خطأ;
}

ساكن فراغ ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    إذا (!runtime) {
        أرجع;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_جریان(dev_ctx->default_جریان);

    خودکار errors = runtime->retrieve_جریان_Errors(dev_ctx->default_جریان);
    إذا (errors.خالی()) {
        أرجع;
    }
    لكل (ثابت خودکار & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (صحيح) err.error_کد,
                       (صحيح) err.errorContext_.value()[0].type_);
    }
    abort();
}

ساكن منطقي ggml_et_can_fuse(ثابت ggml_cgraph * cgraph, صحيح گره_idx, std::initializer_list<ggml_op> ops) {
    إذا (!ggml_can_fuse(cgraph, گره_idx, ops)) {
        أرجع خطأ;
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_جمع_کن) {
        ثابت ggml_tensor * mm  = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * جمع_کن_2 = cgraph->nodes[گره_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        إذا (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        إذا (جمع_کن_2->type != GGML_TYPE_F32) {
            أرجع خطأ;
        }
        إذا (جمع_کن_2->src[0] != mm && جمع_کن_2->src[1] != mm) {
            أرجع خطأ;
        }

        ثابت ggml_tensor * bias = (جمع_کن_2->src[0] == mm) ? جمع_کن_2->src[1] : جمع_کن_2->src[0];

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
            إذا ((int64_t) bias->nb[i] != (int64_t) جمع_کن_2->nb[i]) {
                أرجع خطأ;
            }
        }
    }

    إذا (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ثابت ggml_tensor * rms_norm = cgraph->nodes[گره_idx];
        ثابت ggml_tensor * mul      = cgraph->nodes[گره_idx + 1];

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
        إذا (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_ردیفها(weights)) {
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

    أرجع درست;
}

ساكن ggml_وضعیت ggml_backend_et_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_گراف(&dev_ctx->uberkernel);

    لكل (صحيح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گره = cgraph->nodes[i];

        إذا (گره->op == GGML_OP_NONE || گره->op == GGML_OP_VIEW || گره->op == GGML_OP_RESHAPE ||
            گره->op == GGML_OP_جایگشت || گره->op == GGML_OP_TRANSPOSE) {
            استمر;
        }

        // --- Fusion checks (before regular dispatch) ---
        إذا (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, گره, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            استمر;
        }
        إذا (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جمع_کن })) {
            ggml_et_op_mul_mat(dev_ctx, گره, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            استمر;
        }

        بدل (گره->op) {
            حالة GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_جمع_ردیفها:
                ggml_et_op_جمع_ردیفها(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_میانگین:
                ggml_et_op_میانگین(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_جمع_کن:
                ggml_et_op_جمع_کن(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_گروه_NORM:
                ggml_et_op_گروه_norm(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_IM2_ستون:
                ggml_et_op_im2col(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_GET_ردیفها:
                ggml_et_op_get_ردیفها(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_SSM_پیمایش:
                ggml_et_op_ssm_پیمایش(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_مجموعه_ردیفها:
                ggml_et_op_مجموعه_ردیفها(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_مجموعه:
                ggml_et_op_مجموعه(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, گره);
                اكسر;

            حالة GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, گره);
                اكسر;

            افتراضي:
                ggml_et_uberkernel_abort_گراف(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(گره->op));
                أرجع GGML_وضعیت_FAILED;
        }

        إذا (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_گراف(&dev_ctx->uberkernel);
            أرجع GGML_وضعیت_FAILED;
        }
    }

    إذا (!ggml_et_uberkernel_انتها_گراف(dev_ctx)) {
        ggml_et_uberkernel_abort_گراف(&dev_ctx->uberkernel);
        أرجع GGML_وضعیت_FAILED;
    }

    أرجع GGML_وضعیت_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
ساكن منطقي et_ggml_is_ردیف_contiguous(ثابت ggml_tensor * t) {
    أرجع t->nb[0] == ggml_type_اندازه(t->type);
}

ساكن منطقي ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_استفادهنشده(dev);

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
        حالة GGML_OP_جمع_ردیفها:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_میانگین:
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
                    حالة GGML_UNARY_OP_گام:
                    حالة GGML_UNARY_OP_TANH:
                    حالة GGML_UNARY_OP_ELU:
                    حالة GGML_UNARY_OP_RELU:
                    حالة GGML_UNARY_OP_SIGMOID:
                    حالة GGML_UNARY_OP_GELU:
                    حالة GGML_UNARY_OP_GELU_سریع:
                    حالة GGML_UNARY_OP_SILU:
                    حالة GGML_UNARY_OP_HARDSWISH:
                    حالة GGML_UNARY_OP_HARDSIGMOID:
                    حالة GGML_UNARY_OP_EXP:
                    حالة GGML_UNARY_OP_EXPM1:
                    حالة GGML_UNARY_OP_SOFTPLUS:
                    حالة GGML_UNARY_OP_GELU_ERF:
                    حالة GGML_UNARY_OP_FLOOR:
                    حالة GGML_UNARY_OP_CEIL:
                    حالة GGML_UNARY_OP_دور:
                    حالة GGML_UNARY_OP_TRUNC:
                        supported = درست;
                        اكسر;
                    افتراضي:
                        اكسر;
                }
            }
            اكسر;
        حالة GGML_OP_MUL:
        حالة GGML_OP_جمع_کن:
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
                supported = درست;
            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                منطقي src0_اولی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_اولی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي dst_اولی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                // Check destination stride ordering (only for dimensions with ne > 1)
                منطقي dst_properly_ordered = درست;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                supported = src0_اولی_dim_contiguous && src1_اولی_dim_contiguous && dst_اولی_dim_contiguous &&
                            dst_properly_ordered;
            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                منطقي src0_اولی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_اولی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي dst_اولی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                منطقي dst_properly_ordered = درست;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                supported = src0_اولی_dim_contiguous && src1_اولی_dim_contiguous && dst_اولی_dim_contiguous &&
                            dst_properly_ordered;

            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                منطقي src0_اولی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_اولی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي dst_اولی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                منطقي dst_properly_ordered = درست;
                لكل (صحيح d = 0; d < 3; d++) {
                    إذا (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = خطأ;
                    }
                }

                supported = src0_اولی_dim_contiguous && src1_اولی_dim_contiguous && dst_اولی_dim_contiguous &&
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
                منطقي src0_اولی_dim_contiguous = (op->src[0]->nb[0] == ggml_type_اندازه(op->src[0]->type));
                منطقي src1_اولی_dim_contiguous = (op->src[1]->nb[0] == ggml_type_اندازه(op->src[1]->type));
                منطقي src2_اولی_dim_contiguous = (op->src[2]->nb[0] == ggml_type_اندازه(op->src[2]->type));
                منطقي dst_اولی_dim_contiguous  = (op->nb[0] == حجم(عائم));

                // Check destination stride ordering (only for dimensions with ne > 1)
                منطقي dst_properly_ordered = درست;
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

                supported = src0_اولی_dim_contiguous && src1_اولی_dim_contiguous && src2_اولی_dim_contiguous &&
                            dst_اولی_dim_contiguous && dst_properly_ordered && dims_valid;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_ردیف_contiguous(op->src[0])) {
                ثابت صحيح  mode             = ggml_get_op_params_i32(op, 2);
                ثابت صحيح  ndims            = ggml_get_op_params_i32(op, 1);
                ثابت منطقي is_نرمال        = mode == GGML_ROPE_TYPE_نرمال;
                ثابت منطقي is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ثابت منطقي is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ثابت منطقي zero_view_جابه_جایی = op->src[0]->view_src == اشاره_تهی || op->src[0]->view_offs == 0;
                ثابت منطقي has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ثابت منطقي zero_rot_جابه_جایی  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_جابه_جایی && zero_rot_جابه_جایی && ndims <= 512 &&
                    (is_نرمال || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ردیف_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ردیف_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ردیف_contiguous(op->src[0]);
            اكسر;
        حالة GGML_OP_گروه_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_ردیف_contiguous(op->src[0]) &&
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
                ثابت ggml_tensor * درون  = op->src[1];  // [W,  H,  Cin, N]
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
                ثابت int64_t H    = درون->ne[1];
                (فراغ) درون->ne[0];

                إذا (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || درون->ne[3] != 1) {
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
                ثابت int64_t stage_بایتها   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ثابت int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ثابت int64_t scratch_بایتها = 32 * 2 * 16 * 16 * 4;
                ثابت int64_t budget        = L2SCP_BUDGET - stage_بایتها - scratch_بایتها;
                ثابت int64_t در_هر_KT_بایتها  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ثابت int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                إذا (K_TILES * در_هر_KT_بایتها <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } وإلا {
                    chunk_KT_calc = K_TILES;
                    طالما (chunk_KT_calc > 1 && 2 * chunk_KT_calc * در_هر_KT_بایتها > budget) {
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
                    ثابت int64_t مجموع_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ثابت int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    إذا (مجموع_tiles > max_workers) {
                        supported = خطأ;
                        اكسر;
                    }
                }

                supported = درست;
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
                                                glu_type == GGML_GLU_OP_GEGLU_سریع || glu_type == GGML_GLU_OP_REGLU;

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
                    supported = درست;
                }
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_SSM_پیمایش:
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
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == اشاره_تهی &&
                ggml_is_contiguous_ردیفها(op) && ggml_is_contiguous_ردیفها(op->src[0])) {
                عائم max_bias      = 0.0f;
                عائم logit_softcap = 0.0f;
                کپی_حافظه(&max_bias, (ثابت عائم *) op->op_params + 1, حجم(max_bias));
                کپی_حافظه(&logit_softcap, (ثابت عائم *) op->op_params + 2, حجم(logit_softcap));

                ثابت ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                منطقي ماسک_ok = (op->src[3] == اشاره_تهی) || (op->src[3]->type == GGML_TYPE_F32) ||
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
        حالة GGML_OP_GET_ردیفها:
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
                إذا (op->داده && op->src[0]->داده && op->داده == op->src[0]->داده) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = خطأ;
                } وإلا {
                    supported = درست;
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
                    supported = درست;
                } وإلا إذا ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    إذا (op->داده && op->src[0]->داده && op->داده == op->src[0]->داده) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = خطأ;
                    } وإلا {
                        supported = درست;
                    }
                } وإلا إذا (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = درست;
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
                    supported = درست;
                } وإلا إذا (dim == 0 && ((op->src[0]->nb[0] % حجم(عائم) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % حجم(عائم) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = درست;
                } وإلا إذا (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = درست;
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
                    supported = درست;
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
                supported = درست;
            } وإلا إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = درست;
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
        حالة GGML_OP_مجموعه:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            إذا (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ثابت منطقي   inplace = (منطقي) ((ثابت int32_t *) op->op_params)[4];
                ثابت size_t nb1     = ((ثابت int32_t *) op->op_params)[0];
                ثابت size_t nb2     = ((ثابت int32_t *) op->op_params)[1];
                ثابت size_t nb3     = ((ثابت int32_t *) op->op_params)[2];
                ثابت size_t جابه_جایی  = ((ثابت int32_t *) op->op_params)[3];
                ثابت size_t nb0     = ggml_element_اندازه(op);
                ثابت size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ثابت size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ثابت size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ثابت size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ثابت منطقي view_bounds_ok = جابه_جایی + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ثابت منطقي cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (جابه_جایی % 64 == 0);

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
                supported = درست;
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
                supported = درست;
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
                et_ggml_is_ردیف_contiguous(op->src[0]) && et_ggml_is_ردیف_contiguous(op->src[1]) &&
                et_ggml_is_ردیف_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = درست;
            } وإلا {
                supported = خطأ;
            }
            اكسر;
        حالة GGML_OP_VIEW:
        حالة GGML_OP_جایگشت:
        حالة GGML_OP_TRANSPOSE:
        حالة GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = درست;
            اكسر;
        حالة GGML_OP_مجموعه_ردیفها:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            إذا (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_ردیفها(op) &&
                ggml_is_contiguous_ردیفها(op->src[0]) && ggml_is_contiguous(op->src[1])) {
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
            supported = درست;
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

ساكن منطقي ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_میانگیر_type_t buft) {
    GGML_استفادهنشده(dev);
    أرجع buft->iface.get_name == ggml_backend_et_میانگیر_type_get_name;
}

ساكن منطقي ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    إذا (op->op == GGML_OP_GET_ردیفها) {
        أرجع خطأ;
    }
    أرجع درست;

    GGML_استفادهنشده(dev);
}

ساكن ثابت بنية ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_مجموعه_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ عدم,
    /* .get_tensor_2d_async     = */ عدم,
    /* .cpy_tensor_async        = */ عدم,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ عدم,
    /* .graph_plan_free         = */ عدم,
    /* .graph_plan_update       = */ عدم,
    /* .graph_plan_compute      = */ عدم,
    /* .graph_compute           = */ ggml_backend_et_گراف_compute,
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

ساكن فراغ ggml_backend_et_device_get_حافظه(ggml_backend_dev_t dev, size_t * حرر, size_t * مجموع) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *حرر                                    = dev_ctx->مجموع_mem;
    *مجموع                                   = dev_ctx->مجموع_mem;
}

ساكن تعداد ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_استفادهنشده(dev);
    أرجع GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساكن فراغ ggml_backend_et_device_get_props(ggml_backend_dev_t dev, بنية ggml_backend_dev_props * props) {
    GGML_استفادهنشده(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_حافظه(dev, &props->حافظه_free, &props->حافظه_مجموع);
    props->device_id = عدم;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ درست,
        /* .host_buffer           = */ خطأ,
        /* .buffer_from_host_ptr  = */ خطأ,
        /* .events                = */ خطأ,
        /* .mmap_support          = */ درست,
    };
}

ساكن ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_استفادهنشده(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع ggml_backend_et_init(dev_ctx->devidx);
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_et_device_get_میانگیر_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع dev_ctx->buftype;
}

ساكن ggml_backend_میانگیر_type_t ggml_backend_et_device_get_host_میانگیر_type(ggml_backend_dev_t dev) {
    GGML_استفادهنشده(dev);
    أرجع ggml_backend_cpu_میانگیر_type();
}

ساكن ثابت بنية ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_حافظه,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_میانگیر_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_میانگیر_type,
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
    GGML_استفادهنشده(reg);
    أرجع GGML_ET_NAME;
}

ساكن size_t ggml_backend_et_reg_get_device_شمار(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    أرجع ctx->devices.size();
}

ساكن ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    إذا (devidx >= ctx->devices.size()) {
        أرجع اشاره_تهی;
    }
    أرجع ctx->devices[devidx];
}

ساكن فراغ * ggml_backend_et_get_proc_آدرس(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_استفادهنشده(reg);
    GGML_استفادهنشده(name);
    أرجع اشاره_تهی;
}

ساكن ثابت بنية ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_شمار,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_آدرس,
};

ggml_backend_reg_t ggml_backend_et_reg(فراغ) {
    ساكن ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = جديد ggml_backend_et_reg_ctx;

        إذا (!ggml_et_driver_init()) {
            أرجع اشاره_تهی;
        }

        ggml_backend_reg_t r = جديد ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_نسخه,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ اشاره_تهی,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        لكل (صحيح i = 0; i < ggml_et_devicelayer()->get_Devices_شمار(); i++) {
            ggml_backend_dev_t dev = جديد ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ اشاره_تهی  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = جديد ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->مجموع_mem                       = static_cast<size_t>(prop.حافظه_اندازه);
            {
                ثابت حرف * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_میانگیر_type_context * bufty_ctx = جديد ggml_backend_et_میانگیر_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = جديد ggml_backend_میانگیر_type{ /* .iface   = */ ggml_backend_et_میانگیر_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_جریان = ggml_et_runtime()->create_جریان(rtid);

            dev_ctx->trace_میانگیر = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_میانگیر_اندازه);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            لكل (خودکار & جایگاه : dev_ctx->uberkernel.slots) {
                جایگاه.insts.reserve(256);
                جایگاه.params_blob.reserve(1 << 20);
                جایگاه.device_insts_ظرفیت  = 256 * حجم(ggml_et_uberkernel_inst);
                جایگاه.device_params_ظرفیت = 1 << 20;
                جایگاه.device_insts           = ggml_et_runtime()->mallocDevice(rtid, جایگاه.device_insts_ظرفیت);
                جایگاه.device_params          = ggml_et_runtime()->mallocDevice(rtid, جایگاه.device_params_ظرفیت);
                إذا (جایگاه.device_insts == اشاره_تهی) {
                    جایگاه.device_insts_ظرفیت = 0;
                }
                إذا (جایگاه.device_params == اشاره_تهی) {
                    جایگاه.device_params_ظرفیت = 0;
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
        أرجع اشاره_تهی;
    }

    إذا (devidx >= (size_t) ggml_backend_et_get_device_شمار()) {
        أرجع اشاره_تهی;
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

صحيح ggml_backend_et_get_device_شمار(فراغ) {
    أرجع ggml_backend_et_reg_get_device_شمار(ggml_backend_et_reg());
}

فراغ ggml_backend_et_get_device_description(صحيح devidx, حرف * description, size_t description_اندازه) {
    إذا (devidx < 0 || devidx >= ggml_backend_et_get_device_شمار()) {
        snprintf(description, description_اندازه, "ET Device %d (invalid)", devidx);
        أرجع;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_اندازه, "%s", dev_ctx->desc.c_str());
}

فراغ ggml_backend_et_get_device_حافظه(صحيح devidx, size_t * حرر, size_t * مجموع) {
    إذا (devidx < 0 || devidx >= ggml_backend_et_get_device_شمار()) {
        *حرر  = 0;
        *مجموع = 0;
        أرجع;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_حافظه(dev, حرر, مجموع);
}

ggml_backend_میانگیر_type_t ggml_backend_et_میانگیر_type(size_t dev_رقم) {
    إذا (dev_رقم >= (size_t) ggml_backend_et_get_device_شمار()) {
        أرجع اشاره_تهی;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_رقم);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    أرجع dev_ctx->buftype;
}

ggml_backend_میانگیر_type_t ggml_backend_et_host_میانگیر_type(فراغ) {
    ساكن ggml_backend_میانگیر_type host_میانگیر_type = {
        /* .iface   = */ ggml_backend_et_میانگیر_type_i,
        /* .device  = */ اشاره_تهی,
        /* .context = */ اشاره_تهی,
    };
    أرجع &host_میانگیر_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

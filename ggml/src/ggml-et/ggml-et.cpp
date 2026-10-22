#شامل "ggml-et.h"

#شامل "ggml-backend-impl.h"
#شامل "ggml-backend.h"
#شامل "ggml-et-common.h"
#شامل "ggml-et-kernels.h"
#شامل "ggml-et-memops.h"
#شامل "ggml-et-ops.h"
#شامل "ggml-impl.h"
#شامل "ggml.h"

#شامل <stdarg.h>

#شامل <cstdarg>
#شامل <cstdio>
#شامل <cstdlib>
#شامل <cstring>
#شامل <string>
#شامل <vector>

#اگر __has_include(<filesystem>)
#    include <filesystem>
نامخلا fs = std::filesystem;
#وگرنہ_اگر __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
نامخلا fs = std::experimental::filesystem;
#ورنہ
#    error "cannot include the filesystem library"
#اگر_ختم

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
ساکن خالی ggml_et_dump_tensor_metadata(ثابت ggml_tensor * ggtensor, size_t indent_level, ثابت حرف * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (طویل طویل) ggtensor->ne[0],
            (طویل طویل) ggtensor->ne[1], (طویل طویل) ggtensor->ne[2], (طویل طویل) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->ڈیٹا, spaces.c_str(), (خالی *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
ساکن خالی ggml_et_dump_operator_metadata(ثابت ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != تہی);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    برائے (صحیح i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        حرف arr[16];
        صحیح  n = snprintf(arr, حجم(arr), "src[%i]->name", i);
        GGML_ASSERT((غیرعلامتی) n < حجم(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    اگر (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

ساکن ساخت ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_دھارا;
    std::unique_ptr<std::ofstream>                    kernel_id_دھارا;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    منطقی                                              profiling_enabled = جھوٹ;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
ساکن std::string ggml_et_get_default_et_راستہ() {
    // List of environment variables to check in order of preference
    ثابت حرف * ثابت env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    برائے (ثابت حرف * var : env_vars) {
        اگر (ثابت حرف * et_راستہ = std::getenv(var)) {
            اگر (et_راستہ && *et_راستہ != '\0') {
                واپس fs::path(et_راستہ).string();
            }
        }
    }

    // Otherwise assume default
    واپس fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
ساکن اندرونی خودکار ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_ادوار        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_نقاب = 0x1FFFFFFFFu;
    ثابت std::string  et_راستہ                 = ggml_et_get_default_et_راستہ() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_مقصد_BL2_Elf_راستہ =
        et_راستہ + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_راستہ =
        et_راستہ + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_راستہ = et_راستہ + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_راستہ  = et_راستہ + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.کارکن_Minion_Elf_راستہ  = et_راستہ + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_راستہ       = et_راستہ + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    ثابت std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_مقصد_BL2_Elf_راستہ, sysEmuOptions.sp_BL2_Elf_راستہ,
        sysEmuOptions.machine_Minion_Elf_راستہ,          sysEmuOptions.master_Minion_Elf_راستہ,
        sysEmuOptions.کارکن_Minion_Elf_راستہ,           sysEmuOptions.executable_راستہ,
    };

    برائے (ثابت خودکار & file : required_files) {
        اگر (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.تسلسل_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_ادوار        = k_Sys_Emu_Max_ادوار;
    sysEmuOptions.minion_Shires_نقاب = k_Sys_Emu_Minion_Shires_نقاب;
    sysEmuOptions.pu_Uart0_راستہ      = sysEmuOptions.تسلسل_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_راستہ      = sysEmuOptions.تسلسل_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_راستہ      = sysEmuOptions.تسلسل_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_راستہ      = sysEmuOptions.تسلسل_Dir + "spio_uart1_tx.log";
    sysEmuOptions.آغاز_Gdb         = جھوٹ;
    sysEmuOptions.memcheck         = جھوٹ;

    واپس sysEmuOptions;
}

// Forward declaration
ساکن خالی ggml_et_driver_cleanup();

ساکن منطقی ggml_et_driver_init() {
    اگر (_drv.runtime != خالی_اشاریہ) {
        أكد(_drv.device_layer != خالی_اشاریہ);
    } ورنہ {
        کوشش {
#اگر defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#ورنہ
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#اگر_ختم  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ثابت حرف * profile_راستہ = getenv("GGML_ET_PROFILE");
            اگر (profile_راستہ) {
                std::string output_راستہ    = std::string(profile_راستہ) + "/et_runtime_trace.json";
                std::string kernel_id_راستہ = std::string(profile_راستہ) + "/kernel_id.json";

                _drv.profile_دھارا   = std::make_unique<std::ofstream>(output_راستہ);
                _drv.kernel_id_دھارا = std::make_unique<std::ofstream>(kernel_id_راستہ);
                اگر (!_drv.profile_دھارا->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_راستہ.c_str());
                    abort();
                }
                اگر (!_drv.kernel_id_دھارا->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_راستہ.c_str());
                    abort();
                }

                خودکار * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_دھارا, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = سچ;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } پکڑو (ثابت std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            اگر (_drv.device_layer != خالی_اشاریہ) {
                _drv.device_layer.reset();
            }
            اگر (_drv.runtime != خالی_اشاریہ) {
                _drv.runtime.reset();
            }
            واپس جھوٹ;
        }
    }
    واپس سچ;
}

ساکن std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    واپس _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    واپس _drv.runtime;
}

ساکن خالی ggml_et_driver_cleanup() {
    اگر (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        خودکار * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = جھوٹ;

        اگر (_drv.profile_دھارا) {
            _drv.profile_دھارا->close();
            _drv.profile_دھارا.reset();
        }

        // Save kernel map
        اگر (_drv.kernel_id_دھارا && !_drv.kernel_map.خالی_خانے()) {
            خودکار & os = *_drv.kernel_id_دھارا;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            برائے (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].پہلا << "\": " << (صحیح) _drv.kernel_map[i].second;
                اگر (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_دھارا->close();
            _drv.kernel_id_دھارا.reset();
        }
    }
}

ساکن ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

ساکن خالی ggml_backend_et_بفر_free_بفر(ggml_backend_بفر_t بفر) {
    ggml_backend_et_بفر_context * ctx = (ggml_backend_et_بفر_context *) بفر->context;
    اگر (ctx->ڈیٹا != خالی_اشاریہ) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        اگر (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->ڈیٹا));
        }
    }
    حذف ctx;
}

ساکن خالی * ggml_backend_et_بفر_get_بنیاد(ggml_backend_بفر_t بفر) {
    ggml_backend_et_بفر_context * ctx = (ggml_backend_et_بفر_context *) بفر->context;
    واپس ctx->ڈیٹا;
}

ساکن ggml_حالت ggml_backend_et_بفر_init_tensor(ggml_backend_بفر_t بفر, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    اگر (tensor->view_src != تہی) {
        واپس GGML_حالت_SUCCESS;
    }

    ثابت size_t original_سائز = ggml_nbytes(tensor);
    ثابت size_t padded_سائز   = ggml_backend_buft_get_alloc_سائز(بفر->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    اگر (padded_سائز > original_سائز) {
        ثابت size_t padding_سائز = padded_سائز - original_سائز;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
        اگر (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            واپس GGML_حالت_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->ڈیٹا) + original_سائز;
        اگر (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_سائز)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            واپس GGML_حالت_FAILED;
        }
    }

    واپس GGML_حالت_SUCCESS;
}

ساکن خالی ggml_backend_et_بفر_سیٹ_tensor(ggml_backend_بفر_t بفر,
                                              ggml_tensor *         tensor,
                                              ثابت خالی *          ڈیٹا,
                                              size_t                انحراف,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        واپس;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
    rt::دھارا_Id                     stream  = dev_ctx->default_دھارا;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->ڈیٹا) + انحراف;
    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(ڈیٹا);

    rt::EventId event = runtime->memcpy_Host_مقصد_Device(stream, src_ptr, dst_ptr, size, سچ /*barrier*/);

    runtime->waitForEvent(event);
}

ساکن خالی ggml_backend_et_بفر_get_tensor(ggml_backend_بفر_t بفر,
                                              ثابت ggml_tensor *   tensor,
                                              خالی *                ڈیٹا,
                                              size_t                انحراف,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        واپس;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
    rt::دھارا_Id                     stream  = dev_ctx->default_دھارا;

    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(tensor->ڈیٹا) + انحراف;
    std::byte *       dst_ptr = static_cast<std::byte *>(ڈیٹا);

    rt::EventId event = runtime->memcpy_Device_مقصد_Host(stream, src_ptr, dst_ptr, size, سچ /*barrier*/);

    runtime->waitForEvent(event);
}

ساکن منطقی ggml_backend_et_بفر_cpy_tensor(ggml_backend_بفر_t بفر,
                                              ثابت ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_غیر_مستعمل(بفر);
    GGML_غیر_مستعمل(src);
    GGML_غیر_مستعمل(dst);
    واپس جھوٹ;
}

ساکن خالی ggml_backend_et_بفر_clear(ggml_backend_بفر_t بفر, uint8_t value) {
    ggml_backend_et_بفر_context * ctx = (ggml_backend_et_بفر_context *) بفر->context;

    اگر (ctx->size == 0 || ctx->ڈیٹا == خالی_اشاریہ) {
        واپس;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) بفر->buft->device->context;
    اگر (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        واپس;
    }

    // Use device-side memset kernel for efficient clearing
    اگر (!ggml_et_memset(dev_ctx, ctx->ڈیٹا, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        واپس;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

ساکن ثابت ساخت ggml_backend_بفر_i ggml_backend_et_بفر_i = {
    /* .free_buffer     = */ ggml_backend_et_بفر_free_بفر,
    /* .get_base        = */ ggml_backend_et_بفر_get_بنیاد,
    /* .init_tensor     = */ ggml_backend_et_بفر_init_tensor,
    /* .memset_tensor   = */ تہی,
    /* .set_tensor      = */ ggml_backend_et_بفر_سیٹ_tensor,
    /* .get_tensor      = */ ggml_backend_et_بفر_get_tensor,
    /* .set_tensor_2d   = */ تہی,
    /* .get_tensor_2d   = */ تہی,
    /* .cpy_tensor      = */ ggml_backend_et_بفر_cpy_tensor,
    /* .clear           = */ ggml_backend_et_بفر_clear,
    /* .reset           = */ تہی,
};

ساکن ثابت حرف * ggml_backend_et_بفر_type_get_name(ggml_backend_بفر_type_t buft) {
    GGML_غیر_مستعمل(buft);
    واپس GGML_ET_NAME;
}

ساکن ggml_backend_بفر_t ggml_backend_et_بفر_type_alloc_بفر(ggml_backend_بفر_type_t buft, size_t size) {
    ggml_backend_et_بفر_type_context * btctx = (ggml_backend_et_بفر_type_context *) buft->context;

    ggml_backend_et_بفر_context * ctx = نیا ggml_backend_et_بفر_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        حذف ctx;
        واپس خالی_اشاریہ;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    اگر (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        حذف ctx;
        واپس خالی_اشاریہ;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->ڈیٹا = runtime->mallocDevice(ctx->rtid, size);
    اگر (ctx->ڈیٹا == خالی_اشاریہ) {
        حذف ctx;
        واپس خالی_اشاریہ;
    }

    واپس ggml_backend_بفر_init(buft, ggml_backend_et_بفر_i, ctx, size);
}

ساکن size_t ggml_backend_et_بفر_type_get_alignment(ggml_backend_بفر_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime || !buft->device) {
        واپس GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    واپس prop.cache_سطر_سائز;
}

ساکن size_t ggml_backend_et_بفر_type_get_max_سائز(ggml_backend_بفر_type_t buft) {
    اگر (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        واپس dev_ctx->کل_mem;
    }
    واپس SIZE_MAX;
}

ساکن size_t ggml_backend_et_بفر_type_get_alloc_سائز(ggml_backend_بفر_type_t buft, ثابت ggml_tensor * tensor) {
    GGML_غیر_مستعمل(buft);
    واپس ggml_nbytes_pad(tensor);
}

ساکن منطقی ggml_backend_et_بفر_type_is_host(ggml_backend_بفر_type_t buft) {
    GGML_غیر_مستعمل(buft);
    واپس جھوٹ;
}

ساکن ثابت ساخت ggml_backend_بفر_type_i ggml_backend_et_بفر_type_i = {
    /* .get_name            = */ ggml_backend_et_بفر_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_بفر_type_alloc_بفر,
    /* .alloc_buffer_n      = */ تہی,
    /* .get_alignment       = */ ggml_backend_et_بفر_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_بفر_type_get_max_سائز,
    /* .get_alloc_size      = */ ggml_backend_et_بفر_type_get_alloc_سائز,
    /* .get_alloc_size_n    = */ تہی,
    /* .is_host             = */ ggml_backend_et_بفر_type_is_host,
};

ساکن ثابت حرف * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_غیر_مستعمل(backend);
    واپس GGML_ET_NAME;
}

ساکن خالی ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    اگر (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        اگر (_drv.profiling_enabled) {
            خودکار kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        اگر (runtime) {
            اگر (dev_ctx->trace_بفر) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_بفر);
                dev_ctx->trace_بفر = خالی_اشاریہ;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_دھارا(dev_ctx->default_دھارا);
            برائے (خودکار & خانہ : dev_ctx->uberkernel.slots) {
                اگر (خانہ.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, خانہ.device_insts);
                    خانہ.device_insts = خالی_اشاریہ;
                }
                اگر (خانہ.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, خانہ.device_params);
                    خانہ.device_params = خالی_اشاریہ;
                }
                خانہ.has_pending = جھوٹ;
            }
        }
    }

    حذف et_ctx;
    حذف backend;
}

ساکن ggml_backend_بفر_type_t ggml_backend_et_get_default_بفر_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    واپس ggml_backend_et_بفر_type(et_ctx->devidx);
}

ساکن خالی ggml_backend_et_سیٹ_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ثابت خالی *   ڈیٹا,
                                             size_t         انحراف,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        واپس;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::دھارا_Id                     stream  = dev_ctx->default_دھارا;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->ڈیٹا) + انحراف;
    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(ڈیٹا);

    runtime->memcpy_Host_مقصد_Device(stream, src_ptr, dst_ptr, size, سچ /*barrier*/);
}

ساکن خالی ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ثابت ggml_tensor * tensor,
                                             خالی *              ڈیٹا,
                                             size_t              انحراف,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        واپس;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::دھارا_Id                     stream  = dev_ctx->default_دھارا;

    ثابت std::byte * src_ptr = static_cast<ثابت std::byte *>(tensor->ڈیٹا) + انحراف;
    std::byte *       dst_ptr = static_cast<std::byte *>(ڈیٹا);

    runtime->memcpy_Device_مقصد_Host(stream, src_ptr, dst_ptr, size, سچ /*barrier*/);
}

ساکن منطقی ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ثابت ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_غیر_مستعمل(backend_src);
    GGML_غیر_مستعمل(backend_dst);
    GGML_غیر_مستعمل(src);
    GGML_غیر_مستعمل(dst);
    واپس جھوٹ;
}

ساکن خالی ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    اگر (!runtime) {
        واپس;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_دھارا(dev_ctx->default_دھارا);

    خودکار errors = runtime->retrieve_دھارا_Errors(dev_ctx->default_دھارا);
    اگر (errors.خالی_خانے()) {
        واپس;
    }
    برائے (ثابت خودکار & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (صحیح) err.error_رمز,
                       (صحیح) err.errorContext_.value()[0].type_);
    }
    abort();
}

ساکن منطقی ggml_et_can_fuse(ثابت ggml_cgraph * cgraph, صحیح گرہ_idx, std::initializer_list<ggml_op> ops) {
    اگر (!ggml_can_fuse(cgraph, گرہ_idx, ops)) {
        واپس جھوٹ;
    }

    اگر (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_جوڑو) {
        ثابت ggml_tensor * mm  = cgraph->nodes[گرہ_idx];
        ثابت ggml_tensor * جوڑو_2 = cgraph->nodes[گرہ_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        اگر (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        اگر (جوڑو_2->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }
        اگر (جوڑو_2->src[0] != mm && جوڑو_2->src[1] != mm) {
            واپس جھوٹ;
        }

        ثابت ggml_tensor * bias = (جوڑو_2->src[0] == mm) ? جوڑو_2->src[1] : جوڑو_2->src[0];

        اگر (bias->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }

        // No broadcasting: bias shape must equal MM output shape.
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
            اگر (bias->ne[i] != mm->ne[i]) {
                واپس جھوٹ;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        اگر (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            واپس جھوٹ;
        }
        برائے (صحیح i = 0; i < GGML_MAX_DIMS; ++i) {
            اگر ((int64_t) bias->nb[i] != (int64_t) جوڑو_2->nb[i]) {
                واپس جھوٹ;
            }
        }
    }

    اگر (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ثابت ggml_tensor * rms_norm = cgraph->nodes[گرہ_idx];
        ثابت ggml_tensor * mul      = cgraph->nodes[گرہ_idx + 1];

        // ET only supports F32
        اگر (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        ثابت ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        اگر (weights->type != GGML_TYPE_F32) {
            واپس جھوٹ;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        اگر (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_صفیں(weights)) {
            واپس جھوٹ;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        اگر (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            واپس جھوٹ;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        اگر (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            واپس جھوٹ;
        }
    }

    واپس سچ;
}

ساکن ggml_حالت ggml_backend_et_گراف_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_گراف(&dev_ctx->uberkernel);

    برائے (صحیح i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * گرہ = cgraph->nodes[i];

        اگر (گرہ->op == GGML_OP_NONE || گرہ->op == GGML_OP_VIEW || گرہ->op == GGML_OP_RESHAPE ||
            گرہ->op == GGML_OP_ترتیب_بدلو || گرہ->op == GGML_OP_TRANSPOSE) {
            جاری;
        }

        // --- Fusion checks (before regular dispatch) ---
        اگر (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, گرہ, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            جاری;
        }
        اگر (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_جوڑو })) {
            ggml_et_op_mul_mat(dev_ctx, گرہ, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            جاری;
        }

        انتخاب (گرہ->op) {
            حالت GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_مجموع_صفیں:
                ggml_et_op_مجموع_صفیں(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_اوسط:
                ggml_et_op_اوسط(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_جوڑو:
                ggml_et_op_جوڑو(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_گروہ_NORM:
                ggml_et_op_گروہ_norm(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_IM2_ستون:
                ggml_et_op_im2col(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_GET_صفیں:
                ggml_et_op_get_صفیں(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_SSM_جائزہ:
                ggml_et_op_ssm_جائزہ(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_سیٹ_صفیں:
                ggml_et_op_سیٹ_صفیں(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_سیٹ:
                ggml_et_op_سیٹ(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, گرہ);
                توڑو;

            حالت GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, گرہ);
                توڑو;

            طےشدہ:
                ggml_et_uberkernel_abort_گراف(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(گرہ->op));
                واپس GGML_حالت_FAILED;
        }

        اگر (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_گراف(&dev_ctx->uberkernel);
            واپس GGML_حالت_FAILED;
        }
    }

    اگر (!ggml_et_uberkernel_انتہا_گراف(dev_ctx)) {
        ggml_et_uberkernel_abort_گراف(&dev_ctx->uberkernel);
        واپس GGML_حالت_FAILED;
    }

    واپس GGML_حالت_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
ساکن منطقی et_ggml_is_صف_contiguous(ثابت ggml_tensor * t) {
    واپس t->nb[0] == ggml_type_سائز(t->type);
}

ساکن منطقی ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    GGML_غیر_مستعمل(dev);

    منطقی supported = جھوٹ;
    انتخاب (op->op) {
        حالت GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == حجم(اعشاری) && ggml_is_contiguous(op);
            توڑو;
        حالت GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_مجموع_صفیں:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_اوسط:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == حجم(اعشاری) && op->src[0]->nb[0] == حجم(اعشاری)) {
                انتخاب (ggml_get_unary_op(op)) {
                    حالت GGML_UNARY_OP_ABS:
                    حالت GGML_UNARY_OP_SGN:
                    حالت GGML_UNARY_OP_NEG:
                    حالت GGML_UNARY_OP_قدم:
                    حالت GGML_UNARY_OP_TANH:
                    حالت GGML_UNARY_OP_ELU:
                    حالت GGML_UNARY_OP_RELU:
                    حالت GGML_UNARY_OP_SIGMOID:
                    حالت GGML_UNARY_OP_GELU:
                    حالت GGML_UNARY_OP_GELU_تیز:
                    حالت GGML_UNARY_OP_SILU:
                    حالت GGML_UNARY_OP_HARDSWISH:
                    حالت GGML_UNARY_OP_HARDSIGMOID:
                    حالت GGML_UNARY_OP_EXP:
                    حالت GGML_UNARY_OP_EXPM1:
                    حالت GGML_UNARY_OP_SOFTPLUS:
                    حالت GGML_UNARY_OP_GELU_ERF:
                    حالت GGML_UNARY_OP_FLOOR:
                    حالت GGML_UNARY_OP_CEIL:
                    حالت GGML_UNARY_OP_باری:
                    حالت GGML_UNARY_OP_TRUNC:
                        supported = سچ;
                        توڑو;
                    طےشدہ:
                        توڑو;
                }
            }
            توڑو;
        حالت GGML_OP_MUL:
        حالت GGML_OP_جوڑو:
        حالت GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == حجم(اعشاری) &&
                        op->src[0]->nb[0] == حجم(اعشاری) &&
                        (op->src[1]->nb[0] == حجم(اعشاری) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * حجم(اعشاری);
            توڑو;
        حالت GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            اگر (op->type == GGML_TYPE_F32 &&
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
                supported = سچ;
            } ورنہ اگر (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                منطقی src0_پہلا_dim_contiguous = (op->src[0]->nb[0] == ggml_type_سائز(op->src[0]->type));
                منطقی src1_پہلا_dim_contiguous = (op->src[1]->nb[0] == ggml_type_سائز(op->src[1]->type));
                منطقی dst_پہلا_dim_contiguous  = (op->nb[0] == حجم(اعشاری));

                // Check destination stride ordering (only for dimensions with ne > 1)
                منطقی dst_properly_ordered = سچ;
                برائے (صحیح d = 0; d < 3; d++) {
                    اگر (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = جھوٹ;
                    }
                }

                supported = src0_پہلا_dim_contiguous && src1_پہلا_dim_contiguous && dst_پہلا_dim_contiguous &&
                            dst_properly_ordered;
            } ورنہ اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                منطقی src0_پہلا_dim_contiguous = (op->src[0]->nb[0] == ggml_type_سائز(op->src[0]->type));
                منطقی src1_پہلا_dim_contiguous = (op->src[1]->nb[0] == ggml_type_سائز(op->src[1]->type));
                منطقی dst_پہلا_dim_contiguous  = (op->nb[0] == حجم(اعشاری));

                منطقی dst_properly_ordered = سچ;
                برائے (صحیح d = 0; d < 3; d++) {
                    اگر (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = جھوٹ;
                    }
                }

                supported = src0_پہلا_dim_contiguous && src1_پہلا_dim_contiguous && dst_پہلا_dim_contiguous &&
                            dst_properly_ordered;

            } ورنہ اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                منطقی src0_پہلا_dim_contiguous = (op->src[0]->nb[0] == ggml_type_سائز(op->src[0]->type));
                منطقی src1_پہلا_dim_contiguous = (op->src[1]->nb[0] == ggml_type_سائز(op->src[1]->type));
                منطقی dst_پہلا_dim_contiguous  = (op->nb[0] == حجم(اعشاری));

                منطقی dst_properly_ordered = سچ;
                برائے (صحیح d = 0; d < 3; d++) {
                    اگر (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = جھوٹ;
                    }
                }

                supported = src0_پہلا_dim_contiguous && src1_پہلا_dim_contiguous && dst_پہلا_dim_contiguous &&
                            dst_properly_ordered;
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            اگر (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                منطقی src0_پہلا_dim_contiguous = (op->src[0]->nb[0] == ggml_type_سائز(op->src[0]->type));
                منطقی src1_پہلا_dim_contiguous = (op->src[1]->nb[0] == ggml_type_سائز(op->src[1]->type));
                منطقی src2_پہلا_dim_contiguous = (op->src[2]->nb[0] == ggml_type_سائز(op->src[2]->type));
                منطقی dst_پہلا_dim_contiguous  = (op->nb[0] == حجم(اعشاری));

                // Check destination stride ordering (only for dimensions with ne > 1)
                منطقی dst_properly_ordered = سچ;
                برائے (صحیح d = 0; d < 3; d++) {
                    اگر (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = جھوٹ;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                منطقی dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_پہلا_dim_contiguous && src1_پہلا_dim_contiguous && src2_پہلا_dim_contiguous &&
                            dst_پہلا_dim_contiguous && dst_properly_ordered && dims_valid;
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_صف_contiguous(op->src[0])) {
                ثابت صحیح  mode             = ggml_get_op_params_i32(op, 2);
                ثابت صحیح  ndims            = ggml_get_op_params_i32(op, 1);
                ثابت منطقی is_معمول        = mode == GGML_ROPE_TYPE_معمول;
                ثابت منطقی is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ثابت منطقی is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ثابت منطقی zero_view_انحراف = op->src[0]->view_src == خالی_اشاریہ || op->src[0]->view_offs == 0;
                ثابت منطقی has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ثابت منطقی zero_rot_انحراف  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_انحراف && zero_rot_انحراف && ndims <= 512 &&
                    (is_معمول || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_صف_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_صف_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_صف_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_گروہ_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_صف_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            توڑو;
        حالت GGML_OP_IM2_ستون:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_سائز(op->type) && op->src[1]->nb[0] == ggml_type_سائز(op->src[1]->type);
            توڑو;
        حالت GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                اگر (!op->src[0] || !op->src[1]) {
                    supported = جھوٹ;
                    توڑو;
                }
                اگر (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = جھوٹ;
                    توڑو;
                }
                اگر (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = جھوٹ;
                    توڑو;
                }

                ثابت ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                ثابت ggml_tensor * اندر  = op->src[1];  // [W,  H,  Cin, N]
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
                ثابت int64_t H    = اندر->ne[1];
                (خالی) اندر->ne[0];

                اگر (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || اندر->ne[3] != 1) {
                    supported = جھوٹ;
                    توڑو;
                }
                ثابت int64_t OW = op->ne[0];
                ثابت int64_t OH = op->ne[1];
                اگر (OW <= 0 || OH <= 0) {
                    supported = جھوٹ;
                    توڑو;
                }
                (خالی) p0;
                (خالی) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                ثابت int64_t Hp            = H + 2 * p1;
                ثابت int64_t OW_pad        = (OW + 15) & ~15;
                ثابت int64_t Wp_a          = OW_pad;
                ثابت منطقی    need_stage    = (OW % 16 != 0);
                ثابت int64_t stage_بائٹیں   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ثابت int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ثابت int64_t scratch_بائٹیں = 32 * 2 * 16 * 16 * 4;
                ثابت int64_t budget        = L2SCP_BUDGET - stage_بائٹیں - scratch_بائٹیں;
                ثابت int64_t فی_KT_بائٹیں  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ثابت int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                اگر (K_TILES * فی_KT_بائٹیں <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } ورنہ {
                    chunk_KT_calc = K_TILES;
                    جب (chunk_KT_calc > 1 && 2 * chunk_KT_calc * فی_KT_بائٹیں > budget) {
                        chunk_KT_calc--;
                    }
                    جب (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    اگر (chunk_KT_calc < 1) {
                        supported = جھوٹ;
                        توڑو;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                اگر (n_chunks_calc > 1) {
                    ثابت int64_t M_TILES     = Cout / 16;
                    ثابت int64_t w_tiles     = (OW + 15) / 16;
                    ثابت int64_t کل_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ثابت int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    اگر (کل_tiles > max_workers) {
                        supported = جھوٹ;
                        توڑو;
                    }
                }

                supported = سچ;
                توڑو;
            }
        حالت GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            توڑو;
        حالت GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                ثابت منطقی  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_تیز || glu_type == GGML_GLU_OP_REGLU;

                اگر (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } ورنہ {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_SOFT_MAX:
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                اگر (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    اگر (!supported) {
                        توڑو;
                    }
                }
                // Check optional sinks tensor (F32 only)
                اگر (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } ورنہ {
                    supported = سچ;
                }
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_SSM_جائزہ:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == حجم(اعشاری) && op->src[4]->nb[0] == حجم(اعشاری) &&
                        op->src[5]->nb[0] == حجم(اعشاری) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * حجم(اعشاری) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * حجم(اعشاری) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * حجم(اعشاری) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            توڑو;
        حالت GGML_OP_FLASH_ATTN_EXT:
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == خالی_اشاریہ &&
                ggml_is_contiguous_صفیں(op) && ggml_is_contiguous_صفیں(op->src[0])) {
                اعشاری max_bias      = 0.0f;
                اعشاری logit_softcap = 0.0f;
                حافظہ_نقل(&max_bias, (ثابت اعشاری *) op->op_params + 1, حجم(max_bias));
                حافظہ_نقل(&logit_softcap, (ثابت اعشاری *) op->op_params + 2, حجم(logit_softcap));

                ثابت ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                منطقی نقاب_ok = (op->src[3] == خالی_اشاریہ) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                ثابت int64_t nhq = op->src[0]->ne[2];
                ثابت int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                ثابت size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                ثابت size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                ثابت منطقی me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && نقاب_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == حجم(اعشاری) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == حجم(اعشاری) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_GET_صفیں:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            اگر (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            اگر ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                اگر (op->ڈیٹا && op->src[0]->ڈیٹا && op->ڈیٹا == op->src[0]->ڈیٹا) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = جھوٹ;
                } ورنہ {
                    supported = سچ;
                }
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            اگر (op->src[0]) {
                ثابت int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                اگر (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = سچ;
                } ورنہ اگر ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    اگر (op->ڈیٹا && op->src[0]->ڈیٹا && op->ڈیٹا == op->src[0]->ڈیٹا) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = جھوٹ;
                    } ورنہ {
                        supported = سچ;
                    }
                } ورنہ اگر (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = سچ;
                } ورنہ {
                    supported = جھوٹ;
                }
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_CONCAT:
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                ثابت int32_t dim = ((ثابت int32_t *) op->op_params)[0];
                اگر (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = سچ;
                } ورنہ اگر (dim == 0 && ((op->src[0]->nb[0] % حجم(اعشاری) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % حجم(اعشاری) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = سچ;
                } ورنہ اگر (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = سچ;
                }
            }
            توڑو;
        حالت GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == حجم(اعشاری) &&
                        op->src[1]->nb[0] == حجم(اعشاری) && op->src[0]->nb[1] == op->src[0]->ne[0] * حجم(اعشاری) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * حجم(اعشاری) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            توڑو;
        حالت GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == حجم(اعشاری)) {
                ثابت int32_t lp0      = ((ثابت int32_t *) op->op_params)[0];
                ثابت int32_t rp0      = ((ثابت int32_t *) op->op_params)[1];
                ثابت منطقی    circular = (منطقی) ((ثابت int32_t *) op->op_params)[8];
                اگر (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = سچ;
                } ورنہ {
                    supported = جھوٹ;
                }
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = سچ;
            } ورنہ اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = سچ;
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            توڑو;
        حالت GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            توڑو;
        حالت GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            توڑو;
        حالت GGML_OP_سیٹ:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ثابت منطقی   inplace = (منطقی) ((ثابت int32_t *) op->op_params)[4];
                ثابت size_t nb1     = ((ثابت int32_t *) op->op_params)[0];
                ثابت size_t nb2     = ((ثابت int32_t *) op->op_params)[1];
                ثابت size_t nb3     = ((ثابت int32_t *) op->op_params)[2];
                ثابت size_t انحراف  = ((ثابت int32_t *) op->op_params)[3];
                ثابت size_t nb0     = ggml_element_سائز(op);
                ثابت size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ثابت size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ثابت size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ثابت size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ثابت منطقی view_bounds_ok = انحراف + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ثابت منطقی cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (انحراف % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            توڑو;
        حالت GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = سچ;
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = سچ;
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            اگر (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&                               // k
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&                               // v
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&                               // g
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&                               // beta
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&                               // state
                op->src[2]->ne[0] % 8 == 0 &&                                                    // S_v multiple of 8
                (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[2]->ne[0]) &&  // g is scalar or per-element
                op->src[4]->ne[0] == 1 &&                                              // beta is scalar per position
                et_ggml_is_صف_contiguous(op->src[0]) && et_ggml_is_صف_contiguous(op->src[1]) &&
                et_ggml_is_صف_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = سچ;
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_VIEW:
        حالت GGML_OP_ترتیب_بدلو:
        حالت GGML_OP_TRANSPOSE:
        حالت GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = سچ;
            توڑو;
        حالت GGML_OP_سیٹ_صفیں:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            اگر (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_صفیں(op) &&
                ggml_is_contiguous_صفیں(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } ورنہ {
                supported = جھوٹ;
            }
            توڑو;
        حالت GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = سچ;
            توڑو;
        طےشدہ:
            supported = جھوٹ;
            توڑو;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    واپس supported;
}

ساکن منطقی ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_بفر_type_t buft) {
    GGML_غیر_مستعمل(dev);
    واپس buft->iface.get_name == ggml_backend_et_بفر_type_get_name;
}

ساکن منطقی ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ثابت ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    اگر (op->op == GGML_OP_GET_صفیں) {
        واپس جھوٹ;
    }
    واپس سچ;

    GGML_غیر_مستعمل(dev);
}

ساکن ثابت ساخت ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_سیٹ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ تہی,
    /* .get_tensor_2d_async     = */ تہی,
    /* .cpy_tensor_async        = */ تہی,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ تہی,
    /* .graph_plan_free         = */ تہی,
    /* .graph_plan_update       = */ تہی,
    /* .graph_plan_compute      = */ تہی,
    /* .graph_compute           = */ ggml_backend_et_گراف_compute,
    /* .event_record            = */ تہی,
    /* .event_wait              = */ تہی,
    /* .graph_optimize          = */ تہی,
};

ساکن ثابت حرف * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    واپس dev_ctx->name.c_str();
}

ساکن ثابت حرف * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    واپس dev_ctx->desc.c_str();
}

ساکن خالی ggml_backend_et_device_get_حافظہ(ggml_backend_dev_t dev, size_t * آزاد, size_t * کل) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *آزاد                                    = dev_ctx->کل_mem;
    *کل                                   = dev_ctx->کل_mem;
}

ساکن شمار ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_غیر_مستعمل(dev);
    واپس GGML_BACKEND_DEVICE_TYPE_GPU;
}

ساکن خالی ggml_backend_et_device_get_props(ggml_backend_dev_t dev, ساخت ggml_backend_dev_props * props) {
    GGML_غیر_مستعمل(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_حافظہ(dev, &props->حافظہ_free, &props->حافظہ_کل);
    props->device_id = تہی;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ سچ,
        /* .host_buffer           = */ جھوٹ,
        /* .buffer_from_host_ptr  = */ جھوٹ,
        /* .events                = */ جھوٹ,
        /* .mmap_support          = */ سچ,
    };
}

ساکن ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ثابت حرف * params) {
    GGML_غیر_مستعمل(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    واپس ggml_backend_et_init(dev_ctx->devidx);
}

ساکن ggml_backend_بفر_type_t ggml_backend_et_device_get_بفر_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    واپس dev_ctx->buftype;
}

ساکن ggml_backend_بفر_type_t ggml_backend_et_device_get_host_بفر_type(ggml_backend_dev_t dev) {
    GGML_غیر_مستعمل(dev);
    واپس ggml_backend_cpu_بفر_type();
}

ساکن ثابت ساخت ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_حافظہ,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_بفر_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_بفر_type,
    /* .buffer_from_host_ptr = */ تہی,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ تہی,
    /* .event_free        = */ تہی,
    /* .event_synchronize = */ تہی,
};

/*
  Backend Registry.
*/

ساکن ثابت حرف * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_غیر_مستعمل(reg);
    واپس GGML_ET_NAME;
}

ساکن size_t ggml_backend_et_reg_get_device_گنتی(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    واپس ctx->devices.size();
}

ساکن ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    اگر (devidx >= ctx->devices.size()) {
        واپس خالی_اشاریہ;
    }
    واپس ctx->devices[devidx];
}

ساکن خالی * ggml_backend_et_get_proc_پتہ(ggml_backend_reg_t reg, ثابت حرف * name) {
    GGML_غیر_مستعمل(reg);
    GGML_غیر_مستعمل(name);
    واپس خالی_اشاریہ;
}

ساکن ثابت ساخت ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_گنتی,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_پتہ,
};

ggml_backend_reg_t ggml_backend_et_reg(خالی) {
    ساکن ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = نیا ggml_backend_et_reg_ctx;

        اگر (!ggml_et_driver_init()) {
            واپس خالی_اشاریہ;
        }

        ggml_backend_reg_t r = نیا ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_نسخہ,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ خالی_اشاریہ,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        برائے (صحیح i = 0; i < ggml_et_devicelayer()->get_Devices_گنتی(); i++) {
            ggml_backend_dev_t dev = نیا ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ خالی_اشاریہ  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = نیا ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->کل_mem                       = static_cast<size_t>(prop.حافظہ_سائز);
            {
                ثابت حرف * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_بفر_type_context * bufty_ctx = نیا ggml_backend_et_بفر_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = نیا ggml_backend_بفر_type{ /* .iface   = */ ggml_backend_et_بفر_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_دھارا = ggml_et_runtime()->create_دھارا(rtid);

            dev_ctx->trace_بفر = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_بفر_سائز);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            برائے (خودکار & خانہ : dev_ctx->uberkernel.slots) {
                خانہ.insts.reserve(256);
                خانہ.params_blob.reserve(1 << 20);
                خانہ.device_insts_استعداد  = 256 * حجم(ggml_et_uberkernel_inst);
                خانہ.device_params_استعداد = 1 << 20;
                خانہ.device_insts           = ggml_et_runtime()->mallocDevice(rtid, خانہ.device_insts_استعداد);
                خانہ.device_params          = ggml_et_runtime()->mallocDevice(rtid, خانہ.device_params_استعداد);
                اگر (خانہ.device_insts == خالی_اشاریہ) {
                    خانہ.device_insts_استعداد = 0;
                }
                اگر (خانہ.device_params == خالی_اشاریہ) {
                    خانہ.device_params_استعداد = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        واپس r;
    }();

    واپس _reg;
}

ggml_guid_t ggml_backend_et_guid(خالی) {
    ساکن ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    واپس &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    اگر (!ggml_et_driver_init()) {
        واپس خالی_اشاریہ;
    }

    اگر (devidx >= (size_t) ggml_backend_et_get_device_گنتی()) {
        واپس خالی_اشاریہ;
    }

    ggml_backend_et_context * ctx = نیا ggml_backend_et_context;
    ctx->devidx                   = (صحیح) devidx;

    ggml_backend_t backend = نیا ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    واپس backend;
}

منطقی ggml_backend_is_et(ggml_backend_t backend) {
    واپس backend != تہی && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

صحیح ggml_backend_et_get_device_گنتی(خالی) {
    واپس ggml_backend_et_reg_get_device_گنتی(ggml_backend_et_reg());
}

خالی ggml_backend_et_get_device_description(صحیح devidx, حرف * description, size_t description_سائز) {
    اگر (devidx < 0 || devidx >= ggml_backend_et_get_device_گنتی()) {
        snprintf(description, description_سائز, "ET Device %d (invalid)", devidx);
        واپس;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_سائز, "%s", dev_ctx->desc.c_str());
}

خالی ggml_backend_et_get_device_حافظہ(صحیح devidx, size_t * آزاد, size_t * کل) {
    اگر (devidx < 0 || devidx >= ggml_backend_et_get_device_گنتی()) {
        *آزاد  = 0;
        *کل = 0;
        واپس;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_حافظہ(dev, آزاد, کل);
}

ggml_backend_بفر_type_t ggml_backend_et_بفر_type(size_t dev_عدد) {
    اگر (dev_عدد >= (size_t) ggml_backend_et_get_device_گنتی()) {
        واپس خالی_اشاریہ;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_عدد);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    واپس dev_ctx->buftype;
}

ggml_backend_بفر_type_t ggml_backend_et_host_بفر_type(خالی) {
    ساکن ggml_backend_بفر_type host_بفر_type = {
        /* .iface   = */ ggml_backend_et_بفر_type_i,
        /* .device  = */ خالی_اشاریہ,
        /* .context = */ خالی_اشاریہ,
    };
    واپس &host_بفر_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

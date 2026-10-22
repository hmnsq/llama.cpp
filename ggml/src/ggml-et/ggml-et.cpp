#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-backend.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-kernels.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-memops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-et-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdarg.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdarg>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdio>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>
#ਅਨ੍ਤਰ੍ਭਾਵ <string>
#ਅਨ੍ਤਰ੍ਭਾਵ <vector>

#ਜੇ __has_include(<filesystem>)
#    include <filesystem>
ਨਾਮਾਕਾਸ਼ fs = std::filesystem;
#ਜਾਂਜੇ __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
ਨਾਮਾਕਾਸ਼ fs = std::experimental::filesystem;
#ਨਹੀਂਤਾਂ
#    error "cannot include the filesystem library"
#ਜੇ_ਅੰਤ

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
ਸਥਿਰ ਖਾਲੀ ggml_et_dump_tensor_metadata(ਅਚਲ ggml_tensor * ggtensor, size_t indent_level, ਅਚਲ ਅੱਖਰ * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (ਲੰਮਾ ਲੰਮਾ) ggtensor->ne[0],
            (ਲੰਮਾ ਲੰਮਾ) ggtensor->ne[1], (ਲੰਮਾ ਲੰਮਾ) ggtensor->ne[2], (ਲੰਮਾ ਲੰਮਾ) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->ਅੰਕੜੇ, spaces.c_str(), (ਖਾਲੀ *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
ਸਥਿਰ ਖਾਲੀ ggml_et_dump_operator_metadata(ਅਚਲ ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != ਨਿਰਾ);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        ਅੱਖਰ arr[16];
        ਪੂਰਨ  n = snprintf(arr, ਆਕਾਰ(arr), "src[%i]->name", i);
        GGML_ASSERT((ਬਿਨਾਨਿਸ਼ਾਨੀ) n < ਆਕਾਰ(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    ਜੇ (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

ਸਥਿਰ ਬਣਤਰ ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_ਧਾਰਾ;
    std::unique_ptr<std::ofstream>                    kernel_id_ਧਾਰਾ;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    ਤਾਰਕਿਕ                                              profiling_enabled = ਝੂਠ;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
ਸਥਿਰ std::string ggml_et_get_default_et_ਰਸਤਾ() {
    // List of environment variables to check in order of preference
    ਅਚਲ ਅੱਖਰ * ਅਚਲ env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    ਲਈ (ਅਚਲ ਅੱਖਰ * var : env_vars) {
        ਜੇ (ਅਚਲ ਅੱਖਰ * et_ਰਸਤਾ = std::getenv(var)) {
            ਜੇ (et_ਰਸਤਾ && *et_ਰਸਤਾ != '\0') {
                ਵਾਪਸ fs::path(et_ਰਸਤਾ).string();
            }
        }
    }

    // Otherwise assume default
    ਵਾਪਸ fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
ਸਥਿਰ ਸਤਰਅੰਦਰ ਸਵੈਚਾਲਿਤ ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_ਚੱਕਰਾਂ        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_ਨਕਾਬ = 0x1FFFFFFFFu;
    ਅਚਲ std::string  et_ਰਸਤਾ                 = ggml_et_get_default_et_ਰਸਤਾ() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_ਮੰਜ਼ਿਲ_BL2_Elf_ਰਸਤਾ =
        et_ਰਸਤਾ + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_ਰਸਤਾ =
        et_ਰਸਤਾ + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_ਰਸਤਾ = et_ਰਸਤਾ + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_ਰਸਤਾ  = et_ਰਸਤਾ + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.ਕਾਮਾ_Minion_Elf_ਰਸਤਾ  = et_ਰਸਤਾ + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_ਰਸਤਾ       = et_ਰਸਤਾ + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    ਅਚਲ std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_ਮੰਜ਼ਿਲ_BL2_Elf_ਰਸਤਾ, sysEmuOptions.sp_BL2_Elf_ਰਸਤਾ,
        sysEmuOptions.machine_Minion_Elf_ਰਸਤਾ,          sysEmuOptions.master_Minion_Elf_ਰਸਤਾ,
        sysEmuOptions.ਕਾਮਾ_Minion_Elf_ਰਸਤਾ,           sysEmuOptions.executable_ਰਸਤਾ,
    };

    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & file : required_files) {
        ਜੇ (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.ਲੜੀਵਾਰ_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_ਚੱਕਰਾਂ        = k_Sys_Emu_Max_ਚੱਕਰਾਂ;
    sysEmuOptions.minion_Shires_ਨਕਾਬ = k_Sys_Emu_Minion_Shires_ਨਕਾਬ;
    sysEmuOptions.pu_Uart0_ਰਸਤਾ      = sysEmuOptions.ਲੜੀਵਾਰ_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_ਰਸਤਾ      = sysEmuOptions.ਲੜੀਵਾਰ_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_ਰਸਤਾ      = sysEmuOptions.ਲੜੀਵਾਰ_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_ਰਸਤਾ      = sysEmuOptions.ਲੜੀਵਾਰ_Dir + "spio_uart1_tx.log";
    sysEmuOptions.ਸ਼ੁਰੂ_Gdb         = ਝੂਠ;
    sysEmuOptions.memcheck         = ਝੂਠ;

    ਵਾਪਸ sysEmuOptions;
}

// Forward declaration
ਸਥਿਰ ਖਾਲੀ ggml_et_driver_cleanup();

ਸਥਿਰ ਤਾਰਕਿਕ ggml_et_driver_init() {
    ਜੇ (_drv.runtime != ਖਾਲੀ_ਸੂਚਕ) {
        ਪ੍ਰਤਿਜ੍ਞਾ(_drv.device_layer != ਖਾਲੀ_ਸੂਚਕ);
    } ਨਹੀਂਤਾਂ {
        ਕੋਸ਼ਿਸ਼ {
#ਜੇ defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#ਨਹੀਂਤਾਂ
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#ਜੇ_ਅੰਤ  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ਅਚਲ ਅੱਖਰ * profile_ਰਸਤਾ = getenv("GGML_ET_PROFILE");
            ਜੇ (profile_ਰਸਤਾ) {
                std::string output_ਰਸਤਾ    = std::string(profile_ਰਸਤਾ) + "/et_runtime_trace.json";
                std::string kernel_id_ਰਸਤਾ = std::string(profile_ਰਸਤਾ) + "/kernel_id.json";

                _drv.profile_ਧਾਰਾ   = std::make_unique<std::ofstream>(output_ਰਸਤਾ);
                _drv.kernel_id_ਧਾਰਾ = std::make_unique<std::ofstream>(kernel_id_ਰਸਤਾ);
                ਜੇ (!_drv.profile_ਧਾਰਾ->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_ਰਸਤਾ.c_str());
                    abort();
                }
                ਜੇ (!_drv.kernel_id_ਧਾਰਾ->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_ਰਸਤਾ.c_str());
                    abort();
                }

                ਸਵੈਚਾਲਿਤ * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_ਧਾਰਾ, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = ਸੱਚ;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } ਫੜੋ (ਅਚਲ std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            ਜੇ (_drv.device_layer != ਖਾਲੀ_ਸੂਚਕ) {
                _drv.device_layer.reset();
            }
            ਜੇ (_drv.runtime != ਖਾਲੀ_ਸੂਚਕ) {
                _drv.runtime.reset();
            }
            ਵਾਪਸ ਝੂਠ;
        }
    }
    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    ਵਾਪਸ _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    ਵਾਪਸ _drv.runtime;
}

ਸਥਿਰ ਖਾਲੀ ggml_et_driver_cleanup() {
    ਜੇ (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        ਸਵੈਚਾਲਿਤ * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = ਝੂਠ;

        ਜੇ (_drv.profile_ਧਾਰਾ) {
            _drv.profile_ਧਾਰਾ->close();
            _drv.profile_ਧਾਰਾ.reset();
        }

        // Save kernel map
        ਜੇ (_drv.kernel_id_ਧਾਰਾ && !_drv.kernel_map.ਖਾਲੀ_ਥਾਵਾਂ()) {
            ਸਵੈਚਾਲਿਤ & os = *_drv.kernel_id_ਧਾਰਾ;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            ਲਈ (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].ਪਹਿਲਾ << "\": " << (ਪੂਰਨ) _drv.kernel_map[i].second;
                ਜੇ (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_ਧਾਰਾ->close();
            _drv.kernel_id_ਧਾਰਾ.reset();
        }
    }
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_ਬਫਰ_free_ਬਫਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_et_ਬਫਰ_context * ctx = (ggml_backend_et_ਬਫਰ_context *) ਬਫਰ->context;
    ਜੇ (ctx->ਅੰਕੜੇ != ਖਾਲੀ_ਸੂਚਕ) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        ਜੇ (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->ਅੰਕੜੇ));
        }
    }
    ਮਿਟਾਓ ctx;
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_et_ਬਫਰ_get_ਅਧਾਰ(ggml_backend_ਬਫਰ_t ਬਫਰ) {
    ggml_backend_et_ਬਫਰ_context * ctx = (ggml_backend_et_ਬਫਰ_context *) ਬਫਰ->context;
    ਵਾਪਸ ctx->ਅੰਕੜੇ;
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_et_ਬਫਰ_init_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    ਜੇ (tensor->view_src != ਨਿਰਾ) {
        ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
    }

    ਅਚਲ size_t original_ਮਾਪ = ggml_nbytes(tensor);
    ਅਚਲ size_t padded_ਮਾਪ   = ggml_backend_buft_get_alloc_ਮਾਪ(ਬਫਰ->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    ਜੇ (padded_ਮਾਪ > original_ਮਾਪ) {
        ਅਚਲ size_t padding_ਮਾਪ = padded_ਮਾਪ - original_ਮਾਪ;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ਬਫਰ->buft->device->context;
        ਜੇ (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->ਅੰਕੜੇ) + original_ਮਾਪ;
        ਜੇ (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_ਮਾਪ)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
        }
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_ਬਫਰ_ਸੈੱਟ_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                              ggml_tensor *         tensor,
                                              ਅਚਲ ਖਾਲੀ *          ਅੰਕੜੇ,
                                              size_t                ਵਿੱਥ,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਵਾਪਸ;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ਬਫਰ->buft->device->context;
    rt::ਧਾਰਾ_Id                     stream  = dev_ctx->default_ਧਾਰਾ;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->ਅੰਕੜੇ) + ਵਿੱਥ;
    ਅਚਲ std::byte * src_ptr = static_cast<ਅਚਲ std::byte *>(ਅੰਕੜੇ);

    rt::EventId event = runtime->memcpy_Host_ਮੰਜ਼ਿਲ_Device(stream, src_ptr, dst_ptr, size, ਸੱਚ /*barrier*/);

    runtime->waitForEvent(event);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_ਬਫਰ_get_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                              ਅਚਲ ggml_tensor *   tensor,
                                              ਖਾਲੀ *                ਅੰਕੜੇ,
                                              size_t                ਵਿੱਥ,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਵਾਪਸ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ਬਫਰ->buft->device->context;
    rt::ਧਾਰਾ_Id                     stream  = dev_ctx->default_ਧਾਰਾ;

    ਅਚਲ std::byte * src_ptr = static_cast<ਅਚਲ std::byte *>(tensor->ਅੰਕੜੇ) + ਵਿੱਥ;
    std::byte *       dst_ptr = static_cast<std::byte *>(ਅੰਕੜੇ);

    rt::EventId event = runtime->memcpy_Device_ਮੰਜ਼ਿਲ_Host(stream, src_ptr, dst_ptr, size, ਸੱਚ /*barrier*/);

    runtime->waitForEvent(event);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_et_ਬਫਰ_cpy_tensor(ggml_backend_ਬਫਰ_t ਬਫਰ,
                                              ਅਚਲ ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_ਅਣਵਰਤਿਆ(ਬਫਰ);
    GGML_ਅਣਵਰਤਿਆ(src);
    GGML_ਅਣਵਰਤਿਆ(dst);
    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_ਬਫਰ_clear(ggml_backend_ਬਫਰ_t ਬਫਰ, uint8_t value) {
    ggml_backend_et_ਬਫਰ_context * ctx = (ggml_backend_et_ਬਫਰ_context *) ਬਫਰ->context;

    ਜੇ (ctx->size == 0 || ctx->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ) {
        ਵਾਪਸ;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ਬਫਰ->buft->device->context;
    ਜੇ (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        ਵਾਪਸ;
    }

    // Use device-side memset kernel for efficient clearing
    ਜੇ (!ggml_et_memset(dev_ctx, ctx->ਅੰਕੜੇ, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        ਵਾਪਸ;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ_i ggml_backend_et_ਬਫਰ_i = {
    /* .free_buffer     = */ ggml_backend_et_ਬਫਰ_free_ਬਫਰ,
    /* .get_base        = */ ggml_backend_et_ਬਫਰ_get_ਅਧਾਰ,
    /* .init_tensor     = */ ggml_backend_et_ਬਫਰ_init_tensor,
    /* .memset_tensor   = */ ਨਿਰਾ,
    /* .set_tensor      = */ ggml_backend_et_ਬਫਰ_ਸੈੱਟ_tensor,
    /* .get_tensor      = */ ggml_backend_et_ਬਫਰ_get_tensor,
    /* .set_tensor_2d   = */ ਨਿਰਾ,
    /* .get_tensor_2d   = */ ਨਿਰਾ,
    /* .cpy_tensor      = */ ggml_backend_et_ਬਫਰ_cpy_tensor,
    /* .clear           = */ ggml_backend_et_ਬਫਰ_clear,
    /* .reset           = */ ਨਿਰਾ,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_et_ਬਫਰ_type_get_name(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ਅਣਵਰਤਿਆ(buft);
    ਵਾਪਸ GGML_ET_NAME;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_t ggml_backend_et_ਬਫਰ_type_alloc_ਬਫਰ(ggml_backend_ਬਫਰ_type_t buft, size_t size) {
    ggml_backend_et_ਬਫਰ_type_context * btctx = (ggml_backend_et_ਬਫਰ_type_context *) buft->context;

    ggml_backend_et_ਬਫਰ_context * ctx = ਨਵਾਂ ggml_backend_et_ਬਫਰ_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਮਿਟਾਓ ctx;
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    ਜੇ (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        ਮਿਟਾਓ ctx;
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->ਅੰਕੜੇ = runtime->mallocDevice(ctx->rtid, size);
    ਜੇ (ctx->ਅੰਕੜੇ == ਖਾਲੀ_ਸੂਚਕ) {
        ਮਿਟਾਓ ctx;
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਵਾਪਸ ggml_backend_ਬਫਰ_init(buft, ggml_backend_et_ਬਫਰ_i, ctx, size);
}

ਸਥਿਰ size_t ggml_backend_et_ਬਫਰ_type_get_alignment(ggml_backend_ਬਫਰ_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime || !buft->device) {
        ਵਾਪਸ GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    ਵਾਪਸ prop.cache_ਸਤਰ_ਮਾਪ;
}

ਸਥਿਰ size_t ggml_backend_et_ਬਫਰ_type_get_max_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft) {
    ਜੇ (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        ਵਾਪਸ dev_ctx->ਕੁੱਲ_mem;
    }
    ਵਾਪਸ SIZE_MAX;
}

ਸਥਿਰ size_t ggml_backend_et_ਬਫਰ_type_get_alloc_ਮਾਪ(ggml_backend_ਬਫਰ_type_t buft, ਅਚਲ ggml_tensor * tensor) {
    GGML_ਅਣਵਰਤਿਆ(buft);
    ਵਾਪਸ ggml_nbytes_pad(tensor);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_et_ਬਫਰ_type_is_host(ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ਅਣਵਰਤਿਆ(buft);
    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_ਬਫਰ_type_i ggml_backend_et_ਬਫਰ_type_i = {
    /* .get_name            = */ ggml_backend_et_ਬਫਰ_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_ਬਫਰ_type_alloc_ਬਫਰ,
    /* .alloc_buffer_n      = */ ਨਿਰਾ,
    /* .get_alignment       = */ ggml_backend_et_ਬਫਰ_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_ਬਫਰ_type_get_max_ਮਾਪ,
    /* .get_alloc_size      = */ ggml_backend_et_ਬਫਰ_type_get_alloc_ਮਾਪ,
    /* .get_alloc_size_n    = */ ਨਿਰਾ,
    /* .is_host             = */ ggml_backend_et_ਬਫਰ_type_is_host,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_ਅਣਵਰਤਿਆ(backend);
    ਵਾਪਸ GGML_ET_NAME;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    ਜੇ (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        ਜੇ (_drv.profiling_enabled) {
            ਸਵੈਚਾਲਿਤ kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        ਜੇ (runtime) {
            ਜੇ (dev_ctx->trace_ਬਫਰ) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_ਬਫਰ);
                dev_ctx->trace_ਬਫਰ = ਖਾਲੀ_ਸੂਚਕ;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_ਧਾਰਾ(dev_ctx->default_ਧਾਰਾ);
            ਲਈ (ਸਵੈਚਾਲਿਤ & ਥਾਂ : dev_ctx->uberkernel.slots) {
                ਜੇ (ਥਾਂ.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, ਥਾਂ.device_insts);
                    ਥਾਂ.device_insts = ਖਾਲੀ_ਸੂਚਕ;
                }
                ਜੇ (ਥਾਂ.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, ਥਾਂ.device_params);
                    ਥਾਂ.device_params = ਖਾਲੀ_ਸੂਚਕ;
                }
                ਥਾਂ.has_pending = ਝੂਠ;
            }
        }
    }

    ਮਿਟਾਓ et_ctx;
    ਮਿਟਾਓ backend;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_et_get_default_ਬਫਰ_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    ਵਾਪਸ ggml_backend_et_ਬਫਰ_type(et_ctx->devidx);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_ਸੈੱਟ_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ਅਚਲ ਖਾਲੀ *   ਅੰਕੜੇ,
                                             size_t         ਵਿੱਥ,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਵਾਪਸ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::ਧਾਰਾ_Id                     stream  = dev_ctx->default_ਧਾਰਾ;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->ਅੰਕੜੇ) + ਵਿੱਥ;
    ਅਚਲ std::byte * src_ptr = static_cast<ਅਚਲ std::byte *>(ਅੰਕੜੇ);

    runtime->memcpy_Host_ਮੰਜ਼ਿਲ_Device(stream, src_ptr, dst_ptr, size, ਸੱਚ /*barrier*/);
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ਅਚਲ ggml_tensor * tensor,
                                             ਖਾਲੀ *              ਅੰਕੜੇ,
                                             size_t              ਵਿੱਥ,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਵਾਪਸ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::ਧਾਰਾ_Id                     stream  = dev_ctx->default_ਧਾਰਾ;

    ਅਚਲ std::byte * src_ptr = static_cast<ਅਚਲ std::byte *>(tensor->ਅੰਕੜੇ) + ਵਿੱਥ;
    std::byte *       dst_ptr = static_cast<std::byte *>(ਅੰਕੜੇ);

    runtime->memcpy_Device_ਮੰਜ਼ਿਲ_Host(stream, src_ptr, dst_ptr, size, ਸੱਚ /*barrier*/);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ਅਚਲ ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_ਅਣਵਰਤਿਆ(backend_src);
    GGML_ਅਣਵਰਤਿਆ(backend_dst);
    GGML_ਅਣਵਰਤਿਆ(src);
    GGML_ਅਣਵਰਤਿਆ(dst);
    ਵਾਪਸ ਝੂਠ;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ਜੇ (!runtime) {
        ਵਾਪਸ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_ਧਾਰਾ(dev_ctx->default_ਧਾਰਾ);

    ਸਵੈਚਾਲਿਤ errors = runtime->retrieve_ਧਾਰਾ_Errors(dev_ctx->default_ਧਾਰਾ);
    ਜੇ (errors.ਖਾਲੀ_ਥਾਵਾਂ()) {
        ਵਾਪਸ;
    }
    ਲਈ (ਅਚਲ ਸਵੈਚਾਲਿਤ & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (ਪੂਰਨ) err.error_ਕੋਡ,
                       (ਪੂਰਨ) err.errorContext_.value()[0].type_);
    }
    abort();
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_et_can_fuse(ਅਚਲ ggml_cgraph * cgraph, ਪੂਰਨ ਗੰਢ_idx, std::initializer_list<ggml_op> ops) {
    ਜੇ (!ggml_can_fuse(cgraph, ਗੰਢ_idx, ops)) {
        ਵਾਪਸ ਝੂਠ;
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ਜੋੜੋ) {
        ਅਚਲ ggml_tensor * mm  = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * ਜੋੜੋ_2 = cgraph->nodes[ਗੰਢ_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        ਜੇ (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        ਜੇ (ਜੋੜੋ_2->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਜੇ (ਜੋੜੋ_2->src[0] != mm && ਜੋੜੋ_2->src[1] != mm) {
            ਵਾਪਸ ਝੂਠ;
        }

        ਅਚਲ ggml_tensor * bias = (ਜੋੜੋ_2->src[0] == mm) ? ਜੋੜੋ_2->src[1] : ਜੋੜੋ_2->src[0];

        ਜੇ (bias->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        // No broadcasting: bias shape must equal MM output shape.
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
            ਜੇ (bias->ne[i] != mm->ne[i]) {
                ਵਾਪਸ ਝੂਠ;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        ਜੇ (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            ਵਾਪਸ ਝੂਠ;
        }
        ਲਈ (ਪੂਰਨ i = 0; i < GGML_MAX_DIMS; ++i) {
            ਜੇ ((int64_t) bias->nb[i] != (int64_t) ਜੋੜੋ_2->nb[i]) {
                ਵਾਪਸ ਝੂਠ;
            }
        }
    }

    ਜੇ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ਅਚਲ ggml_tensor * rms_norm = cgraph->nodes[ਗੰਢ_idx];
        ਅਚਲ ggml_tensor * mul      = cgraph->nodes[ਗੰਢ_idx + 1];

        // ET only supports F32
        ਜੇ (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        ਅਚਲ ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        ਜੇ (weights->type != GGML_TYPE_F32) {
            ਵਾਪਸ ਝੂਠ;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        ਜੇ (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_ਪੰਕਤੀਆਂ(weights)) {
            ਵਾਪਸ ਝੂਠ;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        ਜੇ (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            ਵਾਪਸ ਝੂਠ;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        ਜੇ (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            ਵਾਪਸ ਝੂਠ;
        }
    }

    ਵਾਪਸ ਸੱਚ;
}

ਸਥਿਰ ggml_ਸਥਿਤੀ ggml_backend_et_ਗ੍ਰਾਫ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_ਗ੍ਰਾਫ(&dev_ctx->uberkernel);

    ਲਈ (ਪੂਰਨ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ਗੰਢ = cgraph->nodes[i];

        ਜੇ (ਗੰਢ->op == GGML_OP_NONE || ਗੰਢ->op == GGML_OP_VIEW || ਗੰਢ->op == GGML_OP_RESHAPE ||
            ਗੰਢ->op == GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ || ਗੰਢ->op == GGML_OP_TRANSPOSE) {
            ਜਾਰੀ;
        }

        // --- Fusion checks (before regular dispatch) ---
        ਜੇ (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            ਜਾਰੀ;
        }
        ਜੇ (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ਜੋੜੋ })) {
            ggml_et_op_mul_mat(dev_ctx, ਗੰਢ, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            ਜਾਰੀ;
        }

        ਚੋਣ (ਗੰਢ->op) {
            ਹਾਲਤ GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
                ggml_et_op_ਜੋੜ_ਪੰਕਤੀਆਂ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਔਸਤ:
                ggml_et_op_ਔਸਤ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਜੋੜੋ:
                ggml_et_op_ਜੋੜੋ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
                ggml_et_op_ਸਮੂਹ_norm(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
                ggml_et_op_im2col(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
                ggml_et_op_get_ਪੰਕਤੀਆਂ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
                ggml_et_op_ssm_ਜਾਂਚ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
                ggml_et_op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_ਸੈੱਟ:
                ggml_et_op_ਸੈੱਟ(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, ਗੰਢ);
                ਰੋਕੋ;

            ਮੂਲ:
                ggml_et_uberkernel_abort_ਗ੍ਰਾਫ(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(ਗੰਢ->op));
                ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
        }

        ਜੇ (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_ਗ੍ਰਾਫ(&dev_ctx->uberkernel);
            ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
        }
    }

    ਜੇ (!ggml_et_uberkernel_ਸਿਰਾ_ਗ੍ਰਾਫ(dev_ctx)) {
        ggml_et_uberkernel_abort_ਗ੍ਰਾਫ(&dev_ctx->uberkernel);
        ਵਾਪਸ GGML_ਸਥਿਤੀ_FAILED;
    }

    ਵਾਪਸ GGML_ਸਥਿਤੀ_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
ਸਥਿਰ ਤਾਰਕਿਕ et_ggml_is_ਪੰਕਤੀ_contiguous(ਅਚਲ ggml_tensor * t) {
    ਵਾਪਸ t->nb[0] == ggml_type_ਮਾਪ(t->type);
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    GGML_ਅਣਵਰਤਿਆ(dev);

    ਤਾਰਕਿਕ supported = ਝੂਠ;
    ਚੋਣ (op->op) {
        ਹਾਲਤ GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && ggml_is_contiguous(op);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਜੋੜ_ਪੰਕਤੀਆਂ:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਔਸਤ:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
                ਚੋਣ (ggml_get_unary_op(op)) {
                    ਹਾਲਤ GGML_UNARY_OP_ABS:
                    ਹਾਲਤ GGML_UNARY_OP_SGN:
                    ਹਾਲਤ GGML_UNARY_OP_NEG:
                    ਹਾਲਤ GGML_UNARY_OP_ਕਦਮ:
                    ਹਾਲਤ GGML_UNARY_OP_TANH:
                    ਹਾਲਤ GGML_UNARY_OP_ELU:
                    ਹਾਲਤ GGML_UNARY_OP_RELU:
                    ਹਾਲਤ GGML_UNARY_OP_SIGMOID:
                    ਹਾਲਤ GGML_UNARY_OP_GELU:
                    ਹਾਲਤ GGML_UNARY_OP_GELU_ਤੇਜ਼:
                    ਹਾਲਤ GGML_UNARY_OP_SILU:
                    ਹਾਲਤ GGML_UNARY_OP_HARDSWISH:
                    ਹਾਲਤ GGML_UNARY_OP_HARDSIGMOID:
                    ਹਾਲਤ GGML_UNARY_OP_EXP:
                    ਹਾਲਤ GGML_UNARY_OP_EXPM1:
                    ਹਾਲਤ GGML_UNARY_OP_SOFTPLUS:
                    ਹਾਲਤ GGML_UNARY_OP_GELU_ERF:
                    ਹਾਲਤ GGML_UNARY_OP_FLOOR:
                    ਹਾਲਤ GGML_UNARY_OP_CEIL:
                    ਹਾਲਤ GGML_UNARY_OP_ਗੇੜਾ:
                    ਹਾਲਤ GGML_UNARY_OP_TRUNC:
                        supported = ਸੱਚ;
                        ਰੋਕੋ;
                    ਮੂਲ:
                        ਰੋਕੋ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL:
        ਹਾਲਤ GGML_OP_ਜੋੜੋ:
        ਹਾਲਤ GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        (op->src[1]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            ਜੇ (op->type == GGML_TYPE_F32 &&
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
                supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                ਤਾਰਕਿਕ src0_ਪਹਿਲਾ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ਮਾਪ(op->src[0]->type));
                ਤਾਰਕਿਕ src1_ਪਹਿਲਾ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ਮਾਪ(op->src[1]->type));
                ਤਾਰਕਿਕ dst_ਪਹਿਲਾ_dim_contiguous  = (op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                // Check destination stride ordering (only for dimensions with ne > 1)
                ਤਾਰਕਿਕ dst_properly_ordered = ਸੱਚ;
                ਲਈ (ਪੂਰਨ d = 0; d < 3; d++) {
                    ਜੇ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ਝੂਠ;
                    }
                }

                supported = src0_ਪਹਿਲਾ_dim_contiguous && src1_ਪਹਿਲਾ_dim_contiguous && dst_ਪਹਿਲਾ_dim_contiguous &&
                            dst_properly_ordered;
            } ਨਹੀਂਤਾਂ ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                ਤਾਰਕਿਕ src0_ਪਹਿਲਾ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ਮਾਪ(op->src[0]->type));
                ਤਾਰਕਿਕ src1_ਪਹਿਲਾ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ਮਾਪ(op->src[1]->type));
                ਤਾਰਕਿਕ dst_ਪਹਿਲਾ_dim_contiguous  = (op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ਤਾਰਕਿਕ dst_properly_ordered = ਸੱਚ;
                ਲਈ (ਪੂਰਨ d = 0; d < 3; d++) {
                    ਜੇ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ਝੂਠ;
                    }
                }

                supported = src0_ਪਹਿਲਾ_dim_contiguous && src1_ਪਹਿਲਾ_dim_contiguous && dst_ਪਹਿਲਾ_dim_contiguous &&
                            dst_properly_ordered;

            } ਨਹੀਂਤਾਂ ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                ਤਾਰਕਿਕ src0_ਪਹਿਲਾ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ਮਾਪ(op->src[0]->type));
                ਤਾਰਕਿਕ src1_ਪਹਿਲਾ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ਮਾਪ(op->src[1]->type));
                ਤਾਰਕਿਕ dst_ਪਹਿਲਾ_dim_contiguous  = (op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                ਤਾਰਕਿਕ dst_properly_ordered = ਸੱਚ;
                ਲਈ (ਪੂਰਨ d = 0; d < 3; d++) {
                    ਜੇ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ਝੂਠ;
                    }
                }

                supported = src0_ਪਹਿਲਾ_dim_contiguous && src1_ਪਹਿਲਾ_dim_contiguous && dst_ਪਹਿਲਾ_dim_contiguous &&
                            dst_properly_ordered;
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                ਤਾਰਕਿਕ src0_ਪਹਿਲਾ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ਮਾਪ(op->src[0]->type));
                ਤਾਰਕਿਕ src1_ਪਹਿਲਾ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ਮਾਪ(op->src[1]->type));
                ਤਾਰਕਿਕ src2_ਪਹਿਲਾ_dim_contiguous = (op->src[2]->nb[0] == ggml_type_ਮਾਪ(op->src[2]->type));
                ਤਾਰਕਿਕ dst_ਪਹਿਲਾ_dim_contiguous  = (op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                // Check destination stride ordering (only for dimensions with ne > 1)
                ਤਾਰਕਿਕ dst_properly_ordered = ਸੱਚ;
                ਲਈ (ਪੂਰਨ d = 0; d < 3; d++) {
                    ਜੇ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ਝੂਠ;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                ਤਾਰਕਿਕ dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_ਪਹਿਲਾ_dim_contiguous && src1_ਪਹਿਲਾ_dim_contiguous && src2_ਪਹਿਲਾ_dim_contiguous &&
                            dst_ਪਹਿਲਾ_dim_contiguous && dst_properly_ordered && dims_valid;
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[0])) {
                ਅਚਲ ਪੂਰਨ  mode             = ggml_get_op_params_i32(op, 2);
                ਅਚਲ ਪੂਰਨ  ndims            = ggml_get_op_params_i32(op, 1);
                ਅਚਲ ਤਾਰਕਿਕ is_ਸਾਧਾਰਨ        = mode == GGML_ROPE_TYPE_ਸਾਧਾਰਨ;
                ਅਚਲ ਤਾਰਕਿਕ is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ਅਚਲ ਤਾਰਕਿਕ is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ਅਚਲ ਤਾਰਕਿਕ zero_view_ਵਿੱਥ = op->src[0]->view_src == ਖਾਲੀ_ਸੂਚਕ || op->src[0]->view_offs == 0;
                ਅਚਲ ਤਾਰਕਿਕ has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ਅਚਲ ਤਾਰਕਿਕ zero_rot_ਵਿੱਥ  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_ਵਿੱਥ && zero_rot_ਵਿੱਥ && ndims <= 512 &&
                    (is_ਸਾਧਾਰਨ || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸਮੂਹ_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_IM2_ਖਾਨਾ:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_ਮਾਪ(op->type) && op->src[1]->nb[0] == ggml_type_ਮਾਪ(op->src[1]->type);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                ਜੇ (!op->src[0] || !op->src[1]) {
                    supported = ਝੂਠ;
                    ਰੋਕੋ;
                }
                ਜੇ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = ਝੂਠ;
                    ਰੋਕੋ;
                }
                ਜੇ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = ਝੂਠ;
                    ਰੋਕੋ;
                }

                ਅਚਲ ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                ਅਚਲ ggml_tensor * ਵਿੱਚ  = op->src[1];  // [W,  H,  Cin, N]
                ਅਚਲ int32_t       s0  = ggml_get_op_params_i32(op, 0);
                ਅਚਲ int32_t       s1  = ggml_get_op_params_i32(op, 1);
                ਅਚਲ int32_t       p0  = ggml_get_op_params_i32(op, 2);
                ਅਚਲ int32_t       p1  = ggml_get_op_params_i32(op, 3);
                ਅਚਲ int32_t       d0  = ggml_get_op_params_i32(op, 4);
                ਅਚਲ int32_t       d1  = ggml_get_op_params_i32(op, 5);

                ਅਚਲ int64_t Kw   = flt->ne[0];
                ਅਚਲ int64_t Kh   = flt->ne[1];
                ਅਚਲ int64_t Cin  = flt->ne[2];
                ਅਚਲ int64_t Cout = flt->ne[3];
                ਅਚਲ int64_t H    = ਵਿੱਚ->ne[1];
                (ਖਾਲੀ) ਵਿੱਚ->ne[0];

                ਜੇ (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || ਵਿੱਚ->ne[3] != 1) {
                    supported = ਝੂਠ;
                    ਰੋਕੋ;
                }
                ਅਚਲ int64_t OW = op->ne[0];
                ਅਚਲ int64_t OH = op->ne[1];
                ਜੇ (OW <= 0 || OH <= 0) {
                    supported = ਝੂਠ;
                    ਰੋਕੋ;
                }
                (ਖਾਲੀ) p0;
                (ਖਾਲੀ) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                ਅਚਲ int64_t Hp            = H + 2 * p1;
                ਅਚਲ int64_t OW_pad        = (OW + 15) & ~15;
                ਅਚਲ int64_t Wp_a          = OW_pad;
                ਅਚਲ ਤਾਰਕਿਕ    need_stage    = (OW % 16 != 0);
                ਅਚਲ int64_t stage_ਬਾਈਟਾਂ   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ਅਚਲ int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ਅਚਲ int64_t scratch_ਬਾਈਟਾਂ = 32 * 2 * 16 * 16 * 4;
                ਅਚਲ int64_t budget        = L2SCP_BUDGET - stage_ਬਾਈਟਾਂ - scratch_ਬਾਈਟਾਂ;
                ਅਚਲ int64_t ਪ੍ਰਤੀ_KT_ਬਾਈਟਾਂ  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ਅਚਲ int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                ਜੇ (K_TILES * ਪ੍ਰਤੀ_KT_ਬਾਈਟਾਂ <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } ਨਹੀਂਤਾਂ {
                    chunk_KT_calc = K_TILES;
                    ਜਦੋਂ (chunk_KT_calc > 1 && 2 * chunk_KT_calc * ਪ੍ਰਤੀ_KT_ਬਾਈਟਾਂ > budget) {
                        chunk_KT_calc--;
                    }
                    ਜਦੋਂ (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    ਜੇ (chunk_KT_calc < 1) {
                        supported = ਝੂਠ;
                        ਰੋਕੋ;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                ਜੇ (n_chunks_calc > 1) {
                    ਅਚਲ int64_t M_TILES     = Cout / 16;
                    ਅਚਲ int64_t w_tiles     = (OW + 15) / 16;
                    ਅਚਲ int64_t ਕੁੱਲ_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ਅਚਲ int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    ਜੇ (ਕੁੱਲ_tiles > max_workers) {
                        supported = ਝੂਠ;
                        ਰੋਕੋ;
                    }
                }

                supported = ਸੱਚ;
                ਰੋਕੋ;
            }
        ਹਾਲਤ GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                ਅਚਲ ਤਾਰਕਿਕ  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_ਤੇਜ਼ || glu_type == GGML_GLU_OP_REGLU;

                ਜੇ (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } ਨਹੀਂਤਾਂ {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOFT_MAX:
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                ਜੇ (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    ਜੇ (!supported) {
                        ਰੋਕੋ;
                    }
                }
                // Check optional sinks tensor (F32 only)
                ਜੇ (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } ਨਹੀਂਤਾਂ {
                    supported = ਸੱਚ;
                }
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_ਜਾਂਚ:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && op->src[4]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[5]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FLASH_ATTN_EXT:
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == ਖਾਲੀ_ਸੂਚਕ &&
                ggml_is_contiguous_ਪੰਕਤੀਆਂ(op) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0])) {
                ਦਸ਼ਮਲਵ max_bias      = 0.0f;
                ਦਸ਼ਮਲਵ logit_softcap = 0.0f;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&max_bias, (ਅਚਲ ਦਸ਼ਮਲਵ *) op->op_params + 1, ਆਕਾਰ(max_bias));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(&logit_softcap, (ਅਚਲ ਦਸ਼ਮਲਵ *) op->op_params + 2, ਆਕਾਰ(logit_softcap));

                ਅਚਲ ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                ਤਾਰਕਿਕ ਨਕਾਬ_ok = (op->src[3] == ਖਾਲੀ_ਸੂਚਕ) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                ਅਚਲ int64_t nhq = op->src[0]->ne[2];
                ਅਚਲ int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                ਅਚਲ size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                ਅਚਲ size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                ਅਚਲ ਤਾਰਕਿਕ me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && ਨਕਾਬ_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GET_ਪੰਕਤੀਆਂ:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            ਜੇ ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                ਜੇ (op->ਅੰਕੜੇ && op->src[0]->ਅੰਕੜੇ && op->ਅੰਕੜੇ == op->src[0]->ਅੰਕੜੇ) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = ਝੂਠ;
                } ਨਹੀਂਤਾਂ {
                    supported = ਸੱਚ;
                }
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            ਜੇ (op->src[0]) {
                ਅਚਲ int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                ਜੇ (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = ਸੱਚ;
                } ਨਹੀਂਤਾਂ ਜੇ ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    ਜੇ (op->ਅੰਕੜੇ && op->src[0]->ਅੰਕੜੇ && op->ਅੰਕੜੇ == op->src[0]->ਅੰਕੜੇ) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = ਝੂਠ;
                    } ਨਹੀਂਤਾਂ {
                        supported = ਸੱਚ;
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = ਸੱਚ;
                } ਨਹੀਂਤਾਂ {
                    supported = ਝੂਠ;
                }
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_CONCAT:
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                ਅਚਲ int32_t dim = ((ਅਚਲ int32_t *) op->op_params)[0];
                ਜੇ (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = ਸੱਚ;
                } ਨਹੀਂਤਾਂ ਜੇ (dim == 0 && ((op->src[0]->nb[0] % ਆਕਾਰ(ਦਸ਼ਮਲਵ) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % ਆਕਾਰ(ਦਸ਼ਮਲਵ) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = ਸੱਚ;
                } ਨਹੀਂਤਾਂ ਜੇ (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = ਸੱਚ;
                }
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[1]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ) && op->src[0]->nb[1] == op->src[0]->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == ਆਕਾਰ(ਦਸ਼ਮਲਵ)) {
                ਅਚਲ int32_t lp0      = ((ਅਚਲ int32_t *) op->op_params)[0];
                ਅਚਲ int32_t rp0      = ((ਅਚਲ int32_t *) op->op_params)[1];
                ਅਚਲ ਤਾਰਕਿਕ    circular = (ਤਾਰਕਿਕ) ((ਅਚਲ int32_t *) op->op_params)[8];
                ਜੇ (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = ਸੱਚ;
                } ਨਹੀਂਤਾਂ {
                    supported = ਝੂਠ;
                }
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ਅਚਲ ਤਾਰਕਿਕ   inplace = (ਤਾਰਕਿਕ) ((ਅਚਲ int32_t *) op->op_params)[4];
                ਅਚਲ size_t nb1     = ((ਅਚਲ int32_t *) op->op_params)[0];
                ਅਚਲ size_t nb2     = ((ਅਚਲ int32_t *) op->op_params)[1];
                ਅਚਲ size_t nb3     = ((ਅਚਲ int32_t *) op->op_params)[2];
                ਅਚਲ size_t ਵਿੱਥ  = ((ਅਚਲ int32_t *) op->op_params)[3];
                ਅਚਲ size_t nb0     = ggml_element_ਮਾਪ(op);
                ਅਚਲ size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ਅਚਲ size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ਅਚਲ size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ਅਚਲ size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ਅਚਲ ਤਾਰਕਿਕ view_bounds_ok = ਵਿੱਥ + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ਅਚਲ ਤਾਰਕਿਕ cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (ਵਿੱਥ % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            ਜੇ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&                               // k
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&                               // v
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&                               // g
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&                               // beta
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&                               // state
                op->src[2]->ne[0] % 8 == 0 &&                                                    // S_v multiple of 8
                (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[2]->ne[0]) &&  // g is scalar or per-element
                op->src[4]->ne[0] == 1 &&                                              // beta is scalar per position
                et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[0]) && et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[1]) &&
                et_ggml_is_ਪੰਕਤੀ_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = ਸੱਚ;
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_VIEW:
        ਹਾਲਤ GGML_OP_ਕ੍ਰਮ_ਬਦਲੋ:
        ਹਾਲਤ GGML_OP_TRANSPOSE:
        ਹਾਲਤ GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = ਸੱਚ;
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_ਸੈੱਟ_ਪੰਕਤੀਆਂ:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            ਜੇ (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_ਪੰਕਤੀਆਂ(op) &&
                ggml_is_contiguous_ਪੰਕਤੀਆਂ(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } ਨਹੀਂਤਾਂ {
                supported = ਝੂਠ;
            }
            ਰੋਕੋ;
        ਹਾਲਤ GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = ਸੱਚ;
            ਰੋਕੋ;
        ਮੂਲ:
            supported = ਝੂਠ;
            ਰੋਕੋ;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    ਵਾਪਸ supported;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ਬਫਰ_type_t buft) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ buft->iface.get_name == ggml_backend_et_ਬਫਰ_type_get_name;
}

ਸਥਿਰ ਤਾਰਕਿਕ ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ਅਚਲ ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    ਜੇ (op->op == GGML_OP_GET_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ ਝੂਠ;
    }
    ਵਾਪਸ ਸੱਚ;

    GGML_ਅਣਵਰਤਿਆ(dev);
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_ਸੈੱਟ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ ਨਿਰਾ,
    /* .get_tensor_2d_async     = */ ਨਿਰਾ,
    /* .cpy_tensor_async        = */ ਨਿਰਾ,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ ਨਿਰਾ,
    /* .graph_plan_free         = */ ਨਿਰਾ,
    /* .graph_plan_update       = */ ਨਿਰਾ,
    /* .graph_plan_compute      = */ ਨਿਰਾ,
    /* .graph_compute           = */ ggml_backend_et_ਗ੍ਰਾਫ_compute,
    /* .event_record            = */ ਨਿਰਾ,
    /* .event_wait              = */ ਨਿਰਾ,
    /* .graph_optimize          = */ ਨਿਰਾ,
};

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ਵਾਪਸ dev_ctx->name.c_str();
}

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ਵਾਪਸ dev_ctx->desc.c_str();
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_device_get_ਸਮ੍ਰਿਤੀ(ggml_backend_dev_t dev, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *ਮੁਕਤ                                    = dev_ctx->ਕੁੱਲ_mem;
    *ਕੁੱਲ                                   = dev_ctx->ਕੁੱਲ_mem;
}

ਸਥਿਰ ਗਿਣਤੀ ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ਸਥਿਰ ਖਾਲੀ ggml_backend_et_device_get_props(ggml_backend_dev_t dev, ਬਣਤਰ ggml_backend_dev_props * props) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_ਸਮ੍ਰਿਤੀ(dev, &props->ਸਮ੍ਰਿਤੀ_free, &props->ਸਮ੍ਰਿਤੀ_ਕੁੱਲ);
    props->device_id = ਨਿਰਾ;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ ਸੱਚ,
        /* .host_buffer           = */ ਝੂਠ,
        /* .buffer_from_host_ptr  = */ ਝੂਠ,
        /* .events                = */ ਝੂਠ,
        /* .mmap_support          = */ ਸੱਚ,
    };
}

ਸਥਿਰ ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ਅਚਲ ਅੱਖਰ * params) {
    GGML_ਅਣਵਰਤਿਆ(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ਵਾਪਸ ggml_backend_et_init(dev_ctx->devidx);
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_et_device_get_ਬਫਰ_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ਵਾਪਸ dev_ctx->buftype;
}

ਸਥਿਰ ggml_backend_ਬਫਰ_type_t ggml_backend_et_device_get_host_ਬਫਰ_type(ggml_backend_dev_t dev) {
    GGML_ਅਣਵਰਤਿਆ(dev);
    ਵਾਪਸ ggml_backend_cpu_ਬਫਰ_type();
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_ਸਮ੍ਰਿਤੀ,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_ਬਫਰ_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_ਬਫਰ_type,
    /* .buffer_from_host_ptr = */ ਨਿਰਾ,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ ਨਿਰਾ,
    /* .event_free        = */ ਨਿਰਾ,
    /* .event_synchronize = */ ਨਿਰਾ,
};

/*
  Backend Registry.
*/

ਸਥਿਰ ਅਚਲ ਅੱਖਰ * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ਅਣਵਰਤਿਆ(reg);
    ਵਾਪਸ GGML_ET_NAME;
}

ਸਥਿਰ size_t ggml_backend_et_reg_get_device_ਗਿਣਤੀ(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    ਵਾਪਸ ctx->devices.size();
}

ਸਥਿਰ ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    ਜੇ (devidx >= ctx->devices.size()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }
    ਵਾਪਸ ctx->devices[devidx];
}

ਸਥਿਰ ਖਾਲੀ * ggml_backend_et_get_proc_ਪਤਾ(ggml_backend_reg_t reg, ਅਚਲ ਅੱਖਰ * name) {
    GGML_ਅਣਵਰਤਿਆ(reg);
    GGML_ਅਣਵਰਤਿਆ(name);
    ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
}

ਸਥਿਰ ਅਚਲ ਬਣਤਰ ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_ਗਿਣਤੀ,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_ਪਤਾ,
};

ggml_backend_reg_t ggml_backend_et_reg(ਖਾਲੀ) {
    ਸਥਿਰ ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = ਨਵਾਂ ggml_backend_et_reg_ctx;

        ਜੇ (!ggml_et_driver_init()) {
            ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
        }

        ggml_backend_reg_t r = ਨਵਾਂ ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_ਸੰਸਕਰਣ,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ ਖਾਲੀ_ਸੂਚਕ,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        ਲਈ (ਪੂਰਨ i = 0; i < ggml_et_devicelayer()->get_Devices_ਗਿਣਤੀ(); i++) {
            ggml_backend_dev_t dev = ਨਵਾਂ ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ ਖਾਲੀ_ਸੂਚਕ  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = ਨਵਾਂ ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->ਕੁੱਲ_mem                       = static_cast<size_t>(prop.ਸਮ੍ਰਿਤੀ_ਮਾਪ);
            {
                ਅਚਲ ਅੱਖਰ * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_ਬਫਰ_type_context * bufty_ctx = ਨਵਾਂ ggml_backend_et_ਬਫਰ_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = ਨਵਾਂ ggml_backend_ਬਫਰ_type{ /* .iface   = */ ggml_backend_et_ਬਫਰ_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_ਧਾਰਾ = ggml_et_runtime()->create_ਧਾਰਾ(rtid);

            dev_ctx->trace_ਬਫਰ = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_ਬਫਰ_ਮਾਪ);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            ਲਈ (ਸਵੈਚਾਲਿਤ & ਥਾਂ : dev_ctx->uberkernel.slots) {
                ਥਾਂ.insts.reserve(256);
                ਥਾਂ.params_blob.reserve(1 << 20);
                ਥਾਂ.device_insts_ਸਮਰੱਥਾ  = 256 * ਆਕਾਰ(ggml_et_uberkernel_inst);
                ਥਾਂ.device_params_ਸਮਰੱਥਾ = 1 << 20;
                ਥਾਂ.device_insts           = ggml_et_runtime()->mallocDevice(rtid, ਥਾਂ.device_insts_ਸਮਰੱਥਾ);
                ਥਾਂ.device_params          = ggml_et_runtime()->mallocDevice(rtid, ਥਾਂ.device_params_ਸਮਰੱਥਾ);
                ਜੇ (ਥਾਂ.device_insts == ਖਾਲੀ_ਸੂਚਕ) {
                    ਥਾਂ.device_insts_ਸਮਰੱਥਾ = 0;
                }
                ਜੇ (ਥਾਂ.device_params == ਖਾਲੀ_ਸੂਚਕ) {
                    ਥਾਂ.device_params_ਸਮਰੱਥਾ = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        ਵਾਪਸ r;
    }();

    ਵਾਪਸ _reg;
}

ggml_guid_t ggml_backend_et_guid(ਖਾਲੀ) {
    ਸਥਿਰ ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    ਵਾਪਸ &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    ਜੇ (!ggml_et_driver_init()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ਜੇ (devidx >= (size_t) ggml_backend_et_get_device_ਗਿਣਤੀ()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_backend_et_context * ctx = ਨਵਾਂ ggml_backend_et_context;
    ctx->devidx                   = (ਪੂਰਨ) devidx;

    ggml_backend_t backend = ਨਵਾਂ ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    ਵਾਪਸ backend;
}

ਤਾਰਕਿਕ ggml_backend_is_et(ggml_backend_t backend) {
    ਵਾਪਸ backend != ਨਿਰਾ && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

ਪੂਰਨ ggml_backend_et_get_device_ਗਿਣਤੀ(ਖਾਲੀ) {
    ਵਾਪਸ ggml_backend_et_reg_get_device_ਗਿਣਤੀ(ggml_backend_et_reg());
}

ਖਾਲੀ ggml_backend_et_get_device_description(ਪੂਰਨ devidx, ਅੱਖਰ * description, size_t description_ਮਾਪ) {
    ਜੇ (devidx < 0 || devidx >= ggml_backend_et_get_device_ਗਿਣਤੀ()) {
        snprintf(description, description_ਮਾਪ, "ET Device %d (invalid)", devidx);
        ਵਾਪਸ;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_ਮਾਪ, "%s", dev_ctx->desc.c_str());
}

ਖਾਲੀ ggml_backend_et_get_device_ਸਮ੍ਰਿਤੀ(ਪੂਰਨ devidx, size_t * ਮੁਕਤ, size_t * ਕੁੱਲ) {
    ਜੇ (devidx < 0 || devidx >= ggml_backend_et_get_device_ਗਿਣਤੀ()) {
        *ਮੁਕਤ  = 0;
        *ਕੁੱਲ = 0;
        ਵਾਪਸ;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_ਸਮ੍ਰਿਤੀ(dev, ਮੁਕਤ, ਕੁੱਲ);
}

ggml_backend_ਬਫਰ_type_t ggml_backend_et_ਬਫਰ_type(size_t dev_ਅੰਕ) {
    ਜੇ (dev_ਅੰਕ >= (size_t) ggml_backend_et_get_device_ਗਿਣਤੀ()) {
        ਵਾਪਸ ਖਾਲੀ_ਸੂਚਕ;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_ਅੰਕ);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ਵਾਪਸ dev_ctx->buftype;
}

ggml_backend_ਬਫਰ_type_t ggml_backend_et_host_ਬਫਰ_type(ਖਾਲੀ) {
    ਸਥਿਰ ggml_backend_ਬਫਰ_type host_ਬਫਰ_type = {
        /* .iface   = */ ggml_backend_et_ਬਫਰ_type_i,
        /* .device  = */ ਖਾਲੀ_ਸੂਚਕ,
        /* .context = */ ਖਾਲੀ_ਸੂਚਕ,
    };
    ਵਾਪਸ &host_ਬਫਰ_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

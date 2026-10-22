#ಅನ್ತರ್ಭಾವ "ggml-et.h"

#ಅನ್ತರ್ಭಾವ "ggml-backend-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml-backend.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-kernels.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-memops.h"
#ಅನ್ತರ್ಭಾವ "ggml-et-ops.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ <stdarg.h>

#ಅನ್ತರ್ಭಾವ <cstdarg>
#ಅನ್ತರ್ಭಾವ <cstdio>
#ಅನ್ತರ್ಭಾವ <cstdlib>
#ಅನ್ತರ್ಭಾವ <cstring>
#ಅನ್ತರ್ಭಾವ <string>
#ಅನ್ತರ್ಭಾವ <vector>

#ಆದರೆ __has_include(<filesystem>)
#    include <filesystem>
ನಾಮಾಕಾಶ fs = std::filesystem;
#ಅಥವಾಆದರೆ __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
ನಾಮಾಕಾಶ fs = std::experimental::filesystem;
#ಇಲ್ಲದಿದ್ದರೆ
#    error "cannot include the filesystem library"
#ಆದರೆ_ಅಂತ್ಯ

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_dump_tensor_metadata(ಸ್ಥಿರಾಂಕ ggml_tensor * ggtensor, size_t indent_level, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (ದೀರ್ಘ ದೀರ್ಘ) ggtensor->ne[0],
            (ದೀರ್ಘ ದೀರ್ಘ) ggtensor->ne[1], (ದೀರ್ಘ ದೀರ್ಘ) ggtensor->ne[2], (ದೀರ್ಘ ದೀರ್ಘ) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->ದತ್ತಾಂಶ, spaces.c_str(), (ಶೂನ್ಯ *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_dump_operator_metadata(ಸ್ಥಿರಾಂಕ ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != ಖಾಲಿ);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        ಅಕ್ಷರ arr[16];
        ಪೂರ್ಣಾಂಕ  n = snprintf(arr, ಗಾತ್ರ(arr), "src[%i]->name", i);
        GGML_ASSERT((ಚಿಹ್ನೆರಹಿತ) n < ಗಾತ್ರ(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    ಆದರೆ (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

ಸ್ಥಿರ ರಚನೆ ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_ಹರಿವು;
    std::unique_ptr<std::ofstream>                    kernel_id_ಹರಿವು;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    ತಾರ್ಕಿಕ                                              profiling_enabled = ಸುಳ್ಳು;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
ಸ್ಥಿರ std::string ggml_et_get_default_et_ಮಾರ್ಗ() {
    // List of environment variables to check in order of preference
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಸ್ಥಿರಾಂಕ env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * var : env_vars) {
        ಆದರೆ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * et_ಮಾರ್ಗ = std::getenv(var)) {
            ಆದರೆ (et_ಮಾರ್ಗ && *et_ಮಾರ್ಗ != '\0') {
                ಹಿಂತಿರುಗಿ fs::path(et_ಮಾರ್ಗ).string();
            }
        }
    }

    // Otherwise assume default
    ಹಿಂತಿರುಗಿ fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ವಯಂಚಾಲಿತ ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_ಚಕ್ರಗಳು        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_ಮುಖವಾಡ = 0x1FFFFFFFFu;
    ಸ್ಥಿರಾಂಕ std::string  et_ಮಾರ್ಗ                 = ggml_et_get_default_et_ಮಾರ್ಗ() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_ಗಮ್ಯ_BL2_Elf_ಮಾರ್ಗ =
        et_ಮಾರ್ಗ + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_ಮಾರ್ಗ =
        et_ಮಾರ್ಗ + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_ಮಾರ್ಗ = et_ಮಾರ್ಗ + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_ಮಾರ್ಗ  = et_ಮಾರ್ಗ + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.ಕೆಲಸಗಾರ_Minion_Elf_ಮಾರ್ಗ  = et_ಮಾರ್ಗ + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_ಮಾರ್ಗ       = et_ಮಾರ್ಗ + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    ಸ್ಥಿರಾಂಕ std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_ಗಮ್ಯ_BL2_Elf_ಮಾರ್ಗ, sysEmuOptions.sp_BL2_Elf_ಮಾರ್ಗ,
        sysEmuOptions.machine_Minion_Elf_ಮಾರ್ಗ,          sysEmuOptions.master_Minion_Elf_ಮಾರ್ಗ,
        sysEmuOptions.ಕೆಲಸಗಾರ_Minion_Elf_ಮಾರ್ಗ,           sysEmuOptions.executable_ಮಾರ್ಗ,
    };

    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & file : required_files) {
        ಆದರೆ (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.ಸತತ_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_ಚಕ್ರಗಳು        = k_Sys_Emu_Max_ಚಕ್ರಗಳು;
    sysEmuOptions.minion_Shires_ಮುಖವಾಡ = k_Sys_Emu_Minion_Shires_ಮುಖವಾಡ;
    sysEmuOptions.pu_Uart0_ಮಾರ್ಗ      = sysEmuOptions.ಸತತ_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_ಮಾರ್ಗ      = sysEmuOptions.ಸತತ_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_ಮಾರ್ಗ      = sysEmuOptions.ಸತತ_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_ಮಾರ್ಗ      = sysEmuOptions.ಸತತ_Dir + "spio_uart1_tx.log";
    sysEmuOptions.ಆರಂಭ_Gdb         = ಸುಳ್ಳು;
    sysEmuOptions.memcheck         = ಸುಳ್ಳು;

    ಹಿಂತಿರುಗಿ sysEmuOptions;
}

// Forward declaration
ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_driver_cleanup();

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_driver_init() {
    ಆದರೆ (_drv.runtime != ಶೂನ್ಯಸೂಚಕ) {
        ಪ್ರತಿಜ್ಞಾ(_drv.device_layer != ಶೂನ್ಯಸೂಚಕ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಪ್ರಯತ್ನಿಸು {
#ಆದರೆ defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#ಇಲ್ಲದಿದ್ದರೆ
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#ಆದರೆ_ಅಂತ್ಯ  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * profile_ಮಾರ್ಗ = getenv("GGML_ET_PROFILE");
            ಆದರೆ (profile_ಮಾರ್ಗ) {
                std::string output_ಮಾರ್ಗ    = std::string(profile_ಮಾರ್ಗ) + "/et_runtime_trace.json";
                std::string kernel_id_ಮಾರ್ಗ = std::string(profile_ಮಾರ್ಗ) + "/kernel_id.json";

                _drv.profile_ಹರಿವು   = std::make_unique<std::ofstream>(output_ಮಾರ್ಗ);
                _drv.kernel_id_ಹರಿವು = std::make_unique<std::ofstream>(kernel_id_ಮಾರ್ಗ);
                ಆದರೆ (!_drv.profile_ಹರಿವು->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_ಮಾರ್ಗ.c_str());
                    abort();
                }
                ಆದರೆ (!_drv.kernel_id_ಹರಿವು->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_ಮಾರ್ಗ.c_str());
                    abort();
                }

                ಸ್ವಯಂಚಾಲಿತ * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_ಹರಿವು, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = ನಿಜ;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } ಹಿಡಿ (ಸ್ಥಿರಾಂಕ std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            ಆದರೆ (_drv.device_layer != ಶೂನ್ಯಸೂಚಕ) {
                _drv.device_layer.reset();
            }
            ಆದರೆ (_drv.runtime != ಶೂನ್ಯಸೂಚಕ) {
                _drv.runtime.reset();
            }
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    ಹಿಂತಿರುಗಿ _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    ಹಿಂತಿರುಗಿ _drv.runtime;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_et_driver_cleanup() {
    ಆದರೆ (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        ಸ್ವಯಂಚಾಲಿತ * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = ಸುಳ್ಳು;

        ಆದರೆ (_drv.profile_ಹರಿವು) {
            _drv.profile_ಹರಿವು->close();
            _drv.profile_ಹರಿವು.reset();
        }

        // Save kernel map
        ಆದರೆ (_drv.kernel_id_ಹರಿವು && !_drv.kernel_map.ಖಾಲಿಗಳು()) {
            ಸ್ವಯಂಚಾಲಿತ & os = *_drv.kernel_id_ಹರಿವು;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            ಗಾಗಿ (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].ಮೊದಲ << "\": " << (ಪೂರ್ಣಾಂಕ) _drv.kernel_map[i].second;
                ಆದರೆ (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_ಹರಿವು->close();
            _drv.kernel_id_ಹರಿವು.reset();
        }
    }
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_ಬಫರ್_free_ಬಫರ್(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_et_ಬಫರ್_context * ctx = (ggml_backend_et_ಬಫರ್_context *) ಬಫರ್->context;
    ಆದರೆ (ctx->ದತ್ತಾಂಶ != ಶೂನ್ಯಸೂಚಕ) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        ಆದರೆ (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->ದತ್ತಾಂಶ));
        }
    }
    ಅಳಿಸು ctx;
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_et_ಬಫರ್_get_ತಳ(ggml_backend_ಬಫರ್_t ಬಫರ್) {
    ggml_backend_et_ಬಫರ್_context * ctx = (ggml_backend_et_ಬಫರ್_context *) ಬಫರ್->context;
    ಹಿಂತಿರುಗಿ ctx->ದತ್ತಾಂಶ;
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_et_ಬಫರ್_init_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    ಆದರೆ (tensor->view_src != ಖಾಲಿ) {
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
    }

    ಸ್ಥಿರಾಂಕ size_t original_ಗಾತ್ರ = ggml_nbytes(tensor);
    ಸ್ಥಿರಾಂಕ size_t padded_ಗಾತ್ರ   = ggml_backend_buft_get_alloc_ಗಾತ್ರ(ಬಫರ್->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    ಆದರೆ (padded_ಗಾತ್ರ > original_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ size_t padding_ಗಾತ್ರ = padded_ಗಾತ್ರ - original_ಗಾತ್ರ;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ಬಫರ್->buft->device->context;
        ಆದರೆ (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->ದತ್ತಾಂಶ) + original_ಗಾತ್ರ;
        ಆದರೆ (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_ಗಾತ್ರ)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
        }
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_ಬಫರ್_ಗಣ_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                              ggml_tensor *         tensor,
                                              ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *          ದತ್ತಾಂಶ,
                                              size_t                ಸ್ಥಾನಾಂತರ,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಹಿಂತಿರುಗಿ;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ಬಫರ್->buft->device->context;
    rt::ಹರಿವು_Id                     stream  = dev_ctx->default_ಹರಿವು;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->ದತ್ತಾಂಶ) + ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ std::byte * src_ptr = static_cast<ಸ್ಥಿರಾಂಕ std::byte *>(ದತ್ತಾಂಶ);

    rt::EventId event = runtime->memcpy_Host_ಗಮ್ಯ_Device(stream, src_ptr, dst_ptr, size, ನಿಜ /*barrier*/);

    runtime->waitForEvent(event);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_ಬಫರ್_get_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                              ಸ್ಥಿರಾಂಕ ggml_tensor *   tensor,
                                              ಶೂನ್ಯ *                ದತ್ತಾಂಶ,
                                              size_t                ಸ್ಥಾನಾಂತರ,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ಬಫರ್->buft->device->context;
    rt::ಹರಿವು_Id                     stream  = dev_ctx->default_ಹರಿವು;

    ಸ್ಥಿರಾಂಕ std::byte * src_ptr = static_cast<ಸ್ಥಿರಾಂಕ std::byte *>(tensor->ದತ್ತಾಂಶ) + ಸ್ಥಾನಾಂತರ;
    std::byte *       dst_ptr = static_cast<std::byte *>(ದತ್ತಾಂಶ);

    rt::EventId event = runtime->memcpy_Device_ಗಮ್ಯ_Host(stream, src_ptr, dst_ptr, size, ನಿಜ /*barrier*/);

    runtime->waitForEvent(event);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_et_ಬಫರ್_cpy_tensor(ggml_backend_ಬಫರ್_t ಬಫರ್,
                                              ಸ್ಥಿರಾಂಕ ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_ಬಳಸದ(ಬಫರ್);
    GGML_ಬಳಸದ(src);
    GGML_ಬಳಸದ(dst);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_ಬಫರ್_clear(ggml_backend_ಬಫರ್_t ಬಫರ್, uint8_t value) {
    ggml_backend_et_ಬಫರ್_context * ctx = (ggml_backend_et_ಬಫರ್_context *) ಬಫರ್->context;

    ಆದರೆ (ctx->size == 0 || ctx->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        ಹಿಂತಿರುಗಿ;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) ಬಫರ್->buft->device->context;
    ಆದರೆ (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        ಹಿಂತಿರುಗಿ;
    }

    // Use device-side memset kernel for efficient clearing
    ಆದರೆ (!ggml_et_memset(dev_ctx, ctx->ದತ್ತಾಂಶ, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        ಹಿಂತಿರುಗಿ;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್_i ggml_backend_et_ಬಫರ್_i = {
    /* .free_buffer     = */ ggml_backend_et_ಬಫರ್_free_ಬಫರ್,
    /* .get_base        = */ ggml_backend_et_ಬಫರ್_get_ತಳ,
    /* .init_tensor     = */ ggml_backend_et_ಬಫರ್_init_tensor,
    /* .memset_tensor   = */ ಖಾಲಿ,
    /* .set_tensor      = */ ggml_backend_et_ಬಫರ್_ಗಣ_tensor,
    /* .get_tensor      = */ ggml_backend_et_ಬಫರ್_get_tensor,
    /* .set_tensor_2d   = */ ಖಾಲಿ,
    /* .get_tensor_2d   = */ ಖಾಲಿ,
    /* .cpy_tensor      = */ ggml_backend_et_ಬಫರ್_cpy_tensor,
    /* .clear           = */ ggml_backend_et_ಬಫರ್_clear,
    /* .reset           = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_et_ಬಫರ್_type_get_name(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ಬಳಸದ(buft);
    ಹಿಂತಿರುಗಿ GGML_ET_NAME;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_t ggml_backend_et_ಬಫರ್_type_alloc_ಬಫರ್(ggml_backend_ಬಫರ್_type_t buft, size_t size) {
    ggml_backend_et_ಬಫರ್_type_context * btctx = (ggml_backend_et_ಬಫರ್_type_context *) buft->context;

    ggml_backend_et_ಬಫರ್_context * ctx = ಹೊಸ ggml_backend_et_ಬಫರ್_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಅಳಿಸು ctx;
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    ಆದರೆ (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        ಅಳಿಸು ctx;
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->ದತ್ತಾಂಶ = runtime->mallocDevice(ctx->rtid, size);
    ಆದರೆ (ctx->ದತ್ತಾಂಶ == ಶೂನ್ಯಸೂಚಕ) {
        ಅಳಿಸು ctx;
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಹಿಂತಿರುಗಿ ggml_backend_ಬಫರ್_init(buft, ggml_backend_et_ಬಫರ್_i, ctx, size);
}

ಸ್ಥಿರ size_t ggml_backend_et_ಬಫರ್_type_get_alignment(ggml_backend_ಬಫರ್_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime || !buft->device) {
        ಹಿಂತಿರುಗಿ GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    ಹಿಂತಿರುಗಿ prop.cache_ಸಾಲು_ಗಾತ್ರ;
}

ಸ್ಥಿರ size_t ggml_backend_et_ಬಫರ್_type_get_max_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft) {
    ಆದರೆ (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        ಹಿಂತಿರುಗಿ dev_ctx->ಒಟ್ಟು_mem;
    }
    ಹಿಂತಿರುಗಿ SIZE_MAX;
}

ಸ್ಥಿರ size_t ggml_backend_et_ಬಫರ್_type_get_alloc_ಗಾತ್ರ(ggml_backend_ಬಫರ್_type_t buft, ಸ್ಥಿರಾಂಕ ggml_tensor * tensor) {
    GGML_ಬಳಸದ(buft);
    ಹಿಂತಿರುಗಿ ggml_nbytes_pad(tensor);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_et_ಬಫರ್_type_is_host(ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ಬಳಸದ(buft);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_ಬಫರ್_type_i ggml_backend_et_ಬಫರ್_type_i = {
    /* .get_name            = */ ggml_backend_et_ಬಫರ್_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_ಬಫರ್_type_alloc_ಬಫರ್,
    /* .alloc_buffer_n      = */ ಖಾಲಿ,
    /* .get_alignment       = */ ggml_backend_et_ಬಫರ್_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_ಬಫರ್_type_get_max_ಗಾತ್ರ,
    /* .get_alloc_size      = */ ggml_backend_et_ಬಫರ್_type_get_alloc_ಗಾತ್ರ,
    /* .get_alloc_size_n    = */ ಖಾಲಿ,
    /* .is_host             = */ ggml_backend_et_ಬಫರ್_type_is_host,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_ಬಳಸದ(backend);
    ಹಿಂತಿರುಗಿ GGML_ET_NAME;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    ಆದರೆ (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        ಆದರೆ (_drv.profiling_enabled) {
            ಸ್ವಯಂಚಾಲಿತ kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        ಆದರೆ (runtime) {
            ಆದರೆ (dev_ctx->trace_ಬಫರ್) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_ಬಫರ್);
                dev_ctx->trace_ಬಫರ್ = ಶೂನ್ಯಸೂಚಕ;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_ಹರಿವು(dev_ctx->default_ಹರಿವು);
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ಸ್ಥಾನ : dev_ctx->uberkernel.slots) {
                ಆದರೆ (ಸ್ಥಾನ.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, ಸ್ಥಾನ.device_insts);
                    ಸ್ಥಾನ.device_insts = ಶೂನ್ಯಸೂಚಕ;
                }
                ಆದರೆ (ಸ್ಥಾನ.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, ಸ್ಥಾನ.device_params);
                    ಸ್ಥಾನ.device_params = ಶೂನ್ಯಸೂಚಕ;
                }
                ಸ್ಥಾನ.has_pending = ಸುಳ್ಳು;
            }
        }
    }

    ಅಳಿಸು et_ctx;
    ಅಳಿಸು backend;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_et_get_default_ಬಫರ್_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    ಹಿಂತಿರುಗಿ ggml_backend_et_ಬಫರ್_type(et_ctx->devidx);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_ಗಣ_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *   ದತ್ತಾಂಶ,
                                             size_t         ಸ್ಥಾನಾಂತರ,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::ಹರಿವು_Id                     stream  = dev_ctx->default_ಹರಿವು;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->ದತ್ತಾಂಶ) + ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ std::byte * src_ptr = static_cast<ಸ್ಥಿರಾಂಕ std::byte *>(ದತ್ತಾಂಶ);

    runtime->memcpy_Host_ಗಮ್ಯ_Device(stream, src_ptr, dst_ptr, size, ನಿಜ /*barrier*/);
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ಸ್ಥಿರಾಂಕ ggml_tensor * tensor,
                                             ಶೂನ್ಯ *              ದತ್ತಾಂಶ,
                                             size_t              ಸ್ಥಾನಾಂತರ,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::ಹರಿವು_Id                     stream  = dev_ctx->default_ಹರಿವು;

    ಸ್ಥಿರಾಂಕ std::byte * src_ptr = static_cast<ಸ್ಥಿರಾಂಕ std::byte *>(tensor->ದತ್ತಾಂಶ) + ಸ್ಥಾನಾಂತರ;
    std::byte *       dst_ptr = static_cast<std::byte *>(ದತ್ತಾಂಶ);

    runtime->memcpy_Device_ಗಮ್ಯ_Host(stream, src_ptr, dst_ptr, size, ನಿಜ /*barrier*/);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ಸ್ಥಿರಾಂಕ ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_ಬಳಸದ(backend_src);
    GGML_ಬಳಸದ(backend_dst);
    GGML_ಬಳಸದ(src);
    GGML_ಬಳಸದ(dst);
    ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ಆದರೆ (!runtime) {
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_ಹರಿವು(dev_ctx->default_ಹರಿವು);

    ಸ್ವಯಂಚಾಲಿತ errors = runtime->retrieve_ಹರಿವು_Errors(dev_ctx->default_ಹರಿವು);
    ಆದರೆ (errors.ಖಾಲಿಗಳು()) {
        ಹಿಂತಿರುಗಿ;
    }
    ಗಾಗಿ (ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (ಪೂರ್ಣಾಂಕ) err.error_ಸಂಕೇತ,
                       (ಪೂರ್ಣಾಂಕ) err.errorContext_.value()[0].type_);
    }
    abort();
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_et_can_fuse(ಸ್ಥಿರಾಂಕ ggml_cgraph * cgraph, ಪೂರ್ಣಾಂಕ ಗಂಟು_idx, std::initializer_list<ggml_op> ops) {
    ಆದರೆ (!ggml_can_fuse(cgraph, ಗಂಟು_idx, ops)) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_ಕೂಡಿಸು) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * mm  = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * ಕೂಡಿಸು_2 = cgraph->nodes[ಗಂಟು_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        ಆದರೆ (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        ಆದರೆ (ಕೂಡಿಸು_2->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಆದರೆ (ಕೂಡಿಸು_2->src[0] != mm && ಕೂಡಿಸು_2->src[1] != mm) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        ಸ್ಥಿರಾಂಕ ggml_tensor * bias = (ಕೂಡಿಸು_2->src[0] == mm) ? ಕೂಡಿಸು_2->src[1] : ಕೂಡಿಸು_2->src[0];

        ಆದರೆ (bias->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // No broadcasting: bias shape must equal MM output shape.
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
            ಆದರೆ (bias->ne[i] != mm->ne[i]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        ಆದರೆ (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < GGML_MAX_DIMS; ++i) {
            ಆದರೆ ((int64_t) bias->nb[i] != (int64_t) ಕೂಡಿಸು_2->nb[i]) {
                ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
            }
        }
    }

    ಆದರೆ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ಸ್ಥಿರಾಂಕ ggml_tensor * rms_norm = cgraph->nodes[ಗಂಟು_idx];
        ಸ್ಥಿರಾಂಕ ggml_tensor * mul      = cgraph->nodes[ಗಂಟು_idx + 1];

        // ET only supports F32
        ಆದರೆ (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        ಸ್ಥಿರಾಂಕ ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        ಆದರೆ (weights->type != GGML_TYPE_F32) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        ಆದರೆ (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(weights)) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        ಆದರೆ (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        ಆದರೆ (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
        }
    }

    ಹಿಂತಿರುಗಿ ನಿಜ;
}

ಸ್ಥಿರ ggml_ಸ್ಥಿತಿ ggml_backend_et_ನಕ್ಷೆ_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_ನಕ್ಷೆ(&dev_ctx->uberkernel);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ಗಂಟು = cgraph->nodes[i];

        ಆದರೆ (ಗಂಟು->op == GGML_OP_NONE || ಗಂಟು->op == GGML_OP_VIEW || ಗಂಟು->op == GGML_OP_RESHAPE ||
            ಗಂಟು->op == GGML_OP_ಕ್ರಮಬದಲಾವಣೆ || ಗಂಟು->op == GGML_OP_TRANSPOSE) {
            ಮುಂದುವರಿ;
        }

        // --- Fusion checks (before regular dispatch) ---
        ಆದರೆ (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            ಮುಂದುವರಿ;
        }
        ಆದರೆ (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ಕೂಡಿಸು })) {
            ggml_et_op_mul_mat(dev_ctx, ಗಂಟು, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            ಮುಂದುವರಿ;
        }

        ಆಯ್ಕೆ (ಗಂಟು->op) {
            ಸ್ಥಿತಿ GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
                ggml_et_op_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
                ggml_et_op_ಸರಾಸರಿ(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
                ggml_et_op_ಕೂಡಿಸು(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
                ggml_et_op_ಗುಂಪು_norm(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
                ggml_et_op_im2col(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
                ggml_et_op_get_ಅಡ್ಡಸಾಲುಗಳು(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
                ggml_et_op_ssm_ಪರಿಶೀಲನೆ(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
                ggml_et_op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_ಗಣ:
                ggml_et_op_ಗಣ(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, ಗಂಟು);
                ನಿಲ್ಲಿಸು;

            ಪೂರ್ವನಿಯೋಜಿತ:
                ggml_et_uberkernel_abort_ನಕ್ಷೆ(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(ಗಂಟು->op));
                ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
        }

        ಆದರೆ (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_ನಕ್ಷೆ(&dev_ctx->uberkernel);
            ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
        }
    }

    ಆದರೆ (!ggml_et_uberkernel_ಕೊನೆ_ನಕ್ಷೆ(dev_ctx)) {
        ggml_et_uberkernel_abort_ನಕ್ಷೆ(&dev_ctx->uberkernel);
        ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_FAILED;
    }

    ಹಿಂತಿರುಗಿ GGML_ಸ್ಥಿತಿ_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
ಸ್ಥಿರ ತಾರ್ಕಿಕ et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(ಸ್ಥಿರಾಂಕ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ t->nb[0] == ggml_type_ಗಾತ್ರ(t->type);
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    GGML_ಬಳಸದ(dev);

    ತಾರ್ಕಿಕ supported = ಸುಳ್ಳು;
    ಆಯ್ಕೆ (op->op) {
        ಸ್ಥಿತಿ GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) && ggml_is_contiguous(op);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಮೊತ್ತ_ಅಡ್ಡಸಾಲುಗಳು:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಸರಾಸರಿ:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) && op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
                ಆಯ್ಕೆ (ggml_get_unary_op(op)) {
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ABS:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SGN:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_NEG:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ಹೆಜ್ಜೆ:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_TANH:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ELU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_RELU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SIGMOID:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ಶೀಘ್ರ:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SILU:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSWISH:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_HARDSIGMOID:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_EXP:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_EXPM1:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_SOFTPLUS:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_GELU_ERF:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_FLOOR:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_CEIL:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_ಸುತ್ತು:
                    ಸ್ಥಿತಿ GGML_UNARY_OP_TRUNC:
                        supported = ನಿಜ;
                        ನಿಲ್ಲಿಸು;
                    ಪೂರ್ವನಿಯೋಜಿತ:
                        ನಿಲ್ಲಿಸು;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL:
        ಸ್ಥಿತಿ GGML_OP_ಕೂಡಿಸು:
        ಸ್ಥಿತಿ GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        (op->src[1]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            ಆದರೆ (op->type == GGML_TYPE_F32 &&
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
                supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                ತಾರ್ಕಿಕ src0_ಮೊದಲ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[0]->type));
                ತಾರ್ಕಿಕ src1_ಮೊದಲ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[1]->type));
                ತಾರ್ಕಿಕ dst_ಮೊದಲ_dim_contiguous  = (op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

                // Check destination stride ordering (only for dimensions with ne > 1)
                ತಾರ್ಕಿಕ dst_properly_ordered = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < 3; d++) {
                    ಆದರೆ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ಸುಳ್ಳು;
                    }
                }

                supported = src0_ಮೊದಲ_dim_contiguous && src1_ಮೊದಲ_dim_contiguous && dst_ಮೊದಲ_dim_contiguous &&
                            dst_properly_ordered;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                ತಾರ್ಕಿಕ src0_ಮೊದಲ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[0]->type));
                ತಾರ್ಕಿಕ src1_ಮೊದಲ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[1]->type));
                ತಾರ್ಕಿಕ dst_ಮೊದಲ_dim_contiguous  = (op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

                ತಾರ್ಕಿಕ dst_properly_ordered = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < 3; d++) {
                    ಆದರೆ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ಸುಳ್ಳು;
                    }
                }

                supported = src0_ಮೊದಲ_dim_contiguous && src1_ಮೊದಲ_dim_contiguous && dst_ಮೊದಲ_dim_contiguous &&
                            dst_properly_ordered;

            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                ತಾರ್ಕಿಕ src0_ಮೊದಲ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[0]->type));
                ತಾರ್ಕಿಕ src1_ಮೊದಲ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[1]->type));
                ತಾರ್ಕಿಕ dst_ಮೊದಲ_dim_contiguous  = (op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

                ತಾರ್ಕಿಕ dst_properly_ordered = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < 3; d++) {
                    ಆದರೆ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ಸುಳ್ಳು;
                    }
                }

                supported = src0_ಮೊದಲ_dim_contiguous && src1_ಮೊದಲ_dim_contiguous && dst_ಮೊದಲ_dim_contiguous &&
                            dst_properly_ordered;
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                ತಾರ್ಕಿಕ src0_ಮೊದಲ_dim_contiguous = (op->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[0]->type));
                ತಾರ್ಕಿಕ src1_ಮೊದಲ_dim_contiguous = (op->src[1]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[1]->type));
                ತಾರ್ಕಿಕ src2_ಮೊದಲ_dim_contiguous = (op->src[2]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[2]->type));
                ತಾರ್ಕಿಕ dst_ಮೊದಲ_dim_contiguous  = (op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ));

                // Check destination stride ordering (only for dimensions with ne > 1)
                ತಾರ್ಕಿಕ dst_properly_ordered = ನಿಜ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ d = 0; d < 3; d++) {
                    ಆದರೆ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = ಸುಳ್ಳು;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                ತಾರ್ಕಿಕ dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_ಮೊದಲ_dim_contiguous && src1_ಮೊದಲ_dim_contiguous && src2_ಮೊದಲ_dim_contiguous &&
                            dst_ಮೊದಲ_dim_contiguous && dst_properly_ordered && dims_valid;
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[0])) {
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  mode             = ggml_get_op_params_i32(op, 2);
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ  ndims            = ggml_get_op_params_i32(op, 1);
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_ಸಾಮಾನ್ಯ        = mode == GGML_ROPE_TYPE_ಸಾಮಾನ್ಯ;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ zero_view_ಸ್ಥಾನಾಂತರ = op->src[0]->view_src == ಶೂನ್ಯಸೂಚಕ || op->src[0]->view_offs == 0;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ zero_rot_ಸ್ಥಾನಾಂತರ  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_ಸ್ಥಾನಾಂತರ && zero_rot_ಸ್ಥಾನಾಂತರ && ndims <= 512 &&
                    (is_ಸಾಮಾನ್ಯ || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗುಂಪು_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_IM2_ಲಂಬಸಾಲು:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_ಗಾತ್ರ(op->type) && op->src[1]->nb[0] == ggml_type_ಗಾತ್ರ(op->src[1]->type);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                ಆದರೆ (!op->src[0] || !op->src[1]) {
                    supported = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                ಆದರೆ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                ಆದರೆ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }

                ಸ್ಥಿರಾಂಕ ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳ  = op->src[1];  // [W,  H,  Cin, N]
                ಸ್ಥಿರಾಂಕ int32_t       s0  = ggml_get_op_params_i32(op, 0);
                ಸ್ಥಿರಾಂಕ int32_t       s1  = ggml_get_op_params_i32(op, 1);
                ಸ್ಥಿರಾಂಕ int32_t       p0  = ggml_get_op_params_i32(op, 2);
                ಸ್ಥಿರಾಂಕ int32_t       p1  = ggml_get_op_params_i32(op, 3);
                ಸ್ಥಿರಾಂಕ int32_t       d0  = ggml_get_op_params_i32(op, 4);
                ಸ್ಥಿರಾಂಕ int32_t       d1  = ggml_get_op_params_i32(op, 5);

                ಸ್ಥಿರಾಂಕ int64_t Kw   = flt->ne[0];
                ಸ್ಥಿರಾಂಕ int64_t Kh   = flt->ne[1];
                ಸ್ಥಿರಾಂಕ int64_t Cin  = flt->ne[2];
                ಸ್ಥಿರಾಂಕ int64_t Cout = flt->ne[3];
                ಸ್ಥಿರಾಂಕ int64_t H    = ಒಳ->ne[1];
                (ಶೂನ್ಯ) ಒಳ->ne[0];

                ಆದರೆ (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || ಒಳ->ne[3] != 1) {
                    supported = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                ಸ್ಥಿರಾಂಕ int64_t OW = op->ne[0];
                ಸ್ಥಿರಾಂಕ int64_t OH = op->ne[1];
                ಆದರೆ (OW <= 0 || OH <= 0) {
                    supported = ಸುಳ್ಳು;
                    ನಿಲ್ಲಿಸು;
                }
                (ಶೂನ್ಯ) p0;
                (ಶೂನ್ಯ) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                ಸ್ಥಿರಾಂಕ int64_t Hp            = H + 2 * p1;
                ಸ್ಥಿರಾಂಕ int64_t OW_pad        = (OW + 15) & ~15;
                ಸ್ಥಿರಾಂಕ int64_t Wp_a          = OW_pad;
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    need_stage    = (OW % 16 != 0);
                ಸ್ಥಿರಾಂಕ int64_t stage_ಬೈಟುಗಳು   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ಸ್ಥಿರಾಂಕ int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ಸ್ಥಿರಾಂಕ int64_t scratch_ಬೈಟುಗಳು = 32 * 2 * 16 * 16 * 4;
                ಸ್ಥಿರಾಂಕ int64_t budget        = L2SCP_BUDGET - stage_ಬೈಟುಗಳು - scratch_ಬೈಟುಗಳು;
                ಸ್ಥಿರಾಂಕ int64_t ಪ್ರತಿ_KT_ಬೈಟುಗಳು  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ಸ್ಥಿರಾಂಕ int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                ಆದರೆ (K_TILES * ಪ್ರತಿ_KT_ಬೈಟುಗಳು <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    chunk_KT_calc = K_TILES;
                    ಆಗ (chunk_KT_calc > 1 && 2 * chunk_KT_calc * ಪ್ರತಿ_KT_ಬೈಟುಗಳು > budget) {
                        chunk_KT_calc--;
                    }
                    ಆಗ (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    ಆದರೆ (chunk_KT_calc < 1) {
                        supported = ಸುಳ್ಳು;
                        ನಿಲ್ಲಿಸು;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                ಆದರೆ (n_chunks_calc > 1) {
                    ಸ್ಥಿರಾಂಕ int64_t M_TILES     = Cout / 16;
                    ಸ್ಥಿರಾಂಕ int64_t w_tiles     = (OW + 15) / 16;
                    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ಸ್ಥಿರಾಂಕ int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    ಆದರೆ (ಒಟ್ಟು_tiles > max_workers) {
                        supported = ಸುಳ್ಳು;
                        ನಿಲ್ಲಿಸು;
                    }
                }

                supported = ನಿಜ;
                ನಿಲ್ಲಿಸು;
            }
        ಸ್ಥಿತಿ GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_ಶೀಘ್ರ || glu_type == GGML_GLU_OP_REGLU;

                ಆದರೆ (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } ಇಲ್ಲದಿದ್ದರೆ {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOFT_MAX:
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                ಆದರೆ (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    ಆದರೆ (!supported) {
                        ನಿಲ್ಲಿಸು;
                    }
                }
                // Check optional sinks tensor (F32 only)
                ಆದರೆ (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    supported = ನಿಜ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_ಪರಿಶೀಲನೆ:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) && op->src[4]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[5]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FLASH_ATTN_EXT:
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == ಶೂನ್ಯಸೂಚಕ &&
                ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0])) {
                ದಶಮಾಂಶ max_bias      = 0.0f;
                ದಶಮಾಂಶ logit_softcap = 0.0f;
                ಸ್ಮರಣೆ_ನಕಲು(&max_bias, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) op->op_params + 1, ಗಾತ್ರ(max_bias));
                ಸ್ಮರಣೆ_ನಕಲು(&logit_softcap, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) op->op_params + 2, ಗಾತ್ರ(logit_softcap));

                ಸ್ಥಿರಾಂಕ ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                ತಾರ್ಕಿಕ ಮುಖವಾಡ_ok = (op->src[3] == ಶೂನ್ಯಸೂಚಕ) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                ಸ್ಥಿರಾಂಕ int64_t nhq = op->src[0]->ne[2];
                ಸ್ಥಿರಾಂಕ int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                ಸ್ಥಿರಾಂಕ size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                ಸ್ಥಿರಾಂಕ size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && ಮುಖವಾಡ_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            ಆದರೆ ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                ಆದರೆ (op->ದತ್ತಾಂಶ && op->src[0]->ದತ್ತಾಂಶ && op->ದತ್ತಾಂಶ == op->src[0]->ದತ್ತಾಂಶ) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = ಸುಳ್ಳು;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    supported = ನಿಜ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            ಆದರೆ (op->src[0]) {
                ಸ್ಥಿರಾಂಕ int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                ಆದರೆ (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    ಆದರೆ (op->ದತ್ತಾಂಶ && op->src[0]->ದತ್ತಾಂಶ && op->ದತ್ತಾಂಶ == op->src[0]->ದತ್ತಾಂಶ) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = ಸುಳ್ಳು;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        supported = ನಿಜ;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    supported = ಸುಳ್ಳು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_CONCAT:
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                ಸ್ಥಿರಾಂಕ int32_t dim = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[0];
                ಆದರೆ (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dim == 0 && ((op->src[0]->nb[0] % ಗಾತ್ರ(ದಶಮಾಂಶ) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % ಗಾತ್ರ(ದಶಮಾಂಶ) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = ನಿಜ;
                }
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[1]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ) && op->src[0]->nb[1] == op->src[0]->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == ಗಾತ್ರ(ದಶಮಾಂಶ)) {
                ಸ್ಥಿರಾಂಕ int32_t lp0      = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[0];
                ಸ್ಥಿರಾಂಕ int32_t rp0      = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[1];
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ    circular = (ತಾರ್ಕಿಕ) ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[8];
                ಆದರೆ (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = ನಿಜ;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    supported = ಸುಳ್ಳು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ   inplace = (ತಾರ್ಕಿಕ) ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[4];
                ಸ್ಥಿರಾಂಕ size_t nb1     = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[0];
                ಸ್ಥಿರಾಂಕ size_t nb2     = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[1];
                ಸ್ಥಿರಾಂಕ size_t nb3     = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[2];
                ಸ್ಥಿರಾಂಕ size_t ಸ್ಥಾನಾಂತರ  = ((ಸ್ಥಿರಾಂಕ int32_t *) op->op_params)[3];
                ಸ್ಥಿರಾಂಕ size_t nb0     = ggml_element_ಗಾತ್ರ(op);
                ಸ್ಥಿರಾಂಕ size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ಸ್ಥಿರಾಂಕ size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ಸ್ಥಿರಾಂಕ size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ಸ್ಥಿರಾಂಕ size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ view_bounds_ok = ಸ್ಥಾನಾಂತರ + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (ಸ್ಥಾನಾಂತರ % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            ಆದರೆ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&                               // k
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&                               // v
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&                               // g
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&                               // beta
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&                               // state
                op->src[2]->ne[0] % 8 == 0 &&                                                    // S_v multiple of 8
                (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[2]->ne[0]) &&  // g is scalar or per-element
                op->src[4]->ne[0] == 1 &&                                              // beta is scalar per position
                et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[0]) && et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[1]) &&
                et_ggml_is_ಅಡ್ಡಸಾಲು_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = ನಿಜ;
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_VIEW:
        ಸ್ಥಿತಿ GGML_OP_ಕ್ರಮಬದಲಾವಣೆ:
        ಸ್ಥಿತಿ GGML_OP_TRANSPOSE:
        ಸ್ಥಿತಿ GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            ಆದರೆ (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op) &&
                ggml_is_contiguous_ಅಡ್ಡಸಾಲುಗಳು(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } ಇಲ್ಲದಿದ್ದರೆ {
                supported = ಸುಳ್ಳು;
            }
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = ನಿಜ;
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            supported = ಸುಳ್ಳು;
            ನಿಲ್ಲಿಸು;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    ಹಿಂತಿರುಗಿ supported;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_ಬಫರ್_type_t buft) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ buft->iface.get_name == ggml_backend_et_ಬಫರ್_type_get_name;
}

ಸ್ಥಿರ ತಾರ್ಕಿಕ ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    ಆದರೆ (op->op == GGML_OP_GET_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ ಸುಳ್ಳು;
    }
    ಹಿಂತಿರುಗಿ ನಿಜ;

    GGML_ಬಳಸದ(dev);
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_ಗಣ_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ ಖಾಲಿ,
    /* .get_tensor_2d_async     = */ ಖಾಲಿ,
    /* .cpy_tensor_async        = */ ಖಾಲಿ,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ ಖಾಲಿ,
    /* .graph_plan_free         = */ ಖಾಲಿ,
    /* .graph_plan_update       = */ ಖಾಲಿ,
    /* .graph_plan_compute      = */ ಖಾಲಿ,
    /* .graph_compute           = */ ggml_backend_et_ನಕ್ಷೆ_compute,
    /* .event_record            = */ ಖಾಲಿ,
    /* .event_wait              = */ ಖಾಲಿ,
    /* .graph_optimize          = */ ಖಾಲಿ,
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ dev_ctx->name.c_str();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ dev_ctx->desc.c_str();
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_device_get_ಸ್ಮರಣೆ(ggml_backend_dev_t dev, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *ಬಿಡುಗಡೆ                                    = dev_ctx->ಒಟ್ಟು_mem;
    *ಒಟ್ಟು                                   = dev_ctx->ಒಟ್ಟು_mem;
}

ಸ್ಥಿರ ಎಣಿಕೆ ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ GGML_BACKEND_DEVICE_TYPE_GPU;
}

ಸ್ಥಿರ ಶೂನ್ಯ ggml_backend_et_device_get_props(ggml_backend_dev_t dev, ರಚನೆ ggml_backend_dev_props * props) {
    GGML_ಬಳಸದ(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_ಸ್ಮರಣೆ(dev, &props->ಸ್ಮರಣೆ_free, &props->ಸ್ಮರಣೆ_ಒಟ್ಟು);
    props->device_id = ಖಾಲಿ;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ ನಿಜ,
        /* .host_buffer           = */ ಸುಳ್ಳು,
        /* .buffer_from_host_ptr  = */ ಸುಳ್ಳು,
        /* .events                = */ ಸುಳ್ಳು,
        /* .mmap_support          = */ ನಿಜ,
    };
}

ಸ್ಥಿರ ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * params) {
    GGML_ಬಳಸದ(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ ggml_backend_et_init(dev_ctx->devidx);
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_et_device_get_ಬಫರ್_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ dev_ctx->buftype;
}

ಸ್ಥಿರ ggml_backend_ಬಫರ್_type_t ggml_backend_et_device_get_host_ಬಫರ್_type(ggml_backend_dev_t dev) {
    GGML_ಬಳಸದ(dev);
    ಹಿಂತಿರುಗಿ ggml_backend_cpu_ಬಫರ್_type();
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_ಸ್ಮರಣೆ,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_ಬಫರ್_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_ಬಫರ್_type,
    /* .buffer_from_host_ptr = */ ಖಾಲಿ,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ ಖಾಲಿ,
    /* .event_free        = */ ಖಾಲಿ,
    /* .event_synchronize = */ ಖಾಲಿ,
};

/*
  Backend Registry.
*/

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_ಬಳಸದ(reg);
    ಹಿಂತಿರುಗಿ GGML_ET_NAME;
}

ಸ್ಥಿರ size_t ggml_backend_et_reg_get_device_ಎಣಿಕೆ(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    ಹಿಂತಿರುಗಿ ctx->devices.size();
}

ಸ್ಥಿರ ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    ಆದರೆ (devidx >= ctx->devices.size()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }
    ಹಿಂತಿರುಗಿ ctx->devices[devidx];
}

ಸ್ಥಿರ ಶೂನ್ಯ * ggml_backend_et_get_proc_ವಿಳಾಸ(ggml_backend_reg_t reg, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * name) {
    GGML_ಬಳಸದ(reg);
    GGML_ಬಳಸದ(name);
    ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_ಎಣಿಕೆ,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_ವಿಳಾಸ,
};

ggml_backend_reg_t ggml_backend_et_reg(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = ಹೊಸ ggml_backend_et_reg_ctx;

        ಆದರೆ (!ggml_et_driver_init()) {
            ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
        }

        ggml_backend_reg_t r = ಹೊಸ ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_ಆವೃತ್ತಿ,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ ಶೂನ್ಯಸೂಚಕ,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ggml_et_devicelayer()->get_Devices_ಎಣಿಕೆ(); i++) {
            ggml_backend_dev_t dev = ಹೊಸ ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ ಶೂನ್ಯಸೂಚಕ  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = ಹೊಸ ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->ಒಟ್ಟು_mem                       = static_cast<size_t>(prop.ಸ್ಮರಣೆ_ಗಾತ್ರ);
            {
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_ಬಫರ್_type_context * bufty_ctx = ಹೊಸ ggml_backend_et_ಬಫರ್_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = ಹೊಸ ggml_backend_ಬಫರ್_type{ /* .iface   = */ ggml_backend_et_ಬಫರ್_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_ಹರಿವು = ggml_et_runtime()->create_ಹರಿವು(rtid);

            dev_ctx->trace_ಬಫರ್ = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_ಬಫರ್_ಗಾತ್ರ);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            ಗಾಗಿ (ಸ್ವಯಂಚಾಲಿತ & ಸ್ಥಾನ : dev_ctx->uberkernel.slots) {
                ಸ್ಥಾನ.insts.reserve(256);
                ಸ್ಥಾನ.params_blob.reserve(1 << 20);
                ಸ್ಥಾನ.device_insts_ಸಾಮರ್ಥ್ಯ  = 256 * ಗಾತ್ರ(ggml_et_uberkernel_inst);
                ಸ್ಥಾನ.device_params_ಸಾಮರ್ಥ್ಯ = 1 << 20;
                ಸ್ಥಾನ.device_insts           = ggml_et_runtime()->mallocDevice(rtid, ಸ್ಥಾನ.device_insts_ಸಾಮರ್ಥ್ಯ);
                ಸ್ಥಾನ.device_params          = ggml_et_runtime()->mallocDevice(rtid, ಸ್ಥಾನ.device_params_ಸಾಮರ್ಥ್ಯ);
                ಆದರೆ (ಸ್ಥಾನ.device_insts == ಶೂನ್ಯಸೂಚಕ) {
                    ಸ್ಥಾನ.device_insts_ಸಾಮರ್ಥ್ಯ = 0;
                }
                ಆದರೆ (ಸ್ಥಾನ.device_params == ಶೂನ್ಯಸೂಚಕ) {
                    ಸ್ಥಾನ.device_params_ಸಾಮರ್ಥ್ಯ = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        ಹಿಂತಿರುಗಿ r;
    }();

    ಹಿಂತಿರುಗಿ _reg;
}

ggml_guid_t ggml_backend_et_guid(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    ಹಿಂತಿರುಗಿ &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    ಆದರೆ (!ggml_et_driver_init()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ಆದರೆ (devidx >= (size_t) ggml_backend_et_get_device_ಎಣಿಕೆ()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_backend_et_context * ctx = ಹೊಸ ggml_backend_et_context;
    ctx->devidx                   = (ಪೂರ್ಣಾಂಕ) devidx;

    ggml_backend_t backend = ಹೊಸ ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    ಹಿಂತಿರುಗಿ backend;
}

ತಾರ್ಕಿಕ ggml_backend_is_et(ggml_backend_t backend) {
    ಹಿಂತಿರುಗಿ backend != ಖಾಲಿ && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

ಪೂರ್ಣಾಂಕ ggml_backend_et_get_device_ಎಣಿಕೆ(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ ggml_backend_et_reg_get_device_ಎಣಿಕೆ(ggml_backend_et_reg());
}

ಶೂನ್ಯ ggml_backend_et_get_device_description(ಪೂರ್ಣಾಂಕ devidx, ಅಕ್ಷರ * description, size_t description_ಗಾತ್ರ) {
    ಆದರೆ (devidx < 0 || devidx >= ggml_backend_et_get_device_ಎಣಿಕೆ()) {
        snprintf(description, description_ಗಾತ್ರ, "ET Device %d (invalid)", devidx);
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_ಗಾತ್ರ, "%s", dev_ctx->desc.c_str());
}

ಶೂನ್ಯ ggml_backend_et_get_device_ಸ್ಮರಣೆ(ಪೂರ್ಣಾಂಕ devidx, size_t * ಬಿಡುಗಡೆ, size_t * ಒಟ್ಟು) {
    ಆದರೆ (devidx < 0 || devidx >= ggml_backend_et_get_device_ಎಣಿಕೆ()) {
        *ಬಿಡುಗಡೆ  = 0;
        *ಒಟ್ಟು = 0;
        ಹಿಂತಿರುಗಿ;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_ಸ್ಮರಣೆ(dev, ಬಿಡುಗಡೆ, ಒಟ್ಟು);
}

ggml_backend_ಬಫರ್_type_t ggml_backend_et_ಬಫರ್_type(size_t dev_ಅಂಕೆ) {
    ಆದರೆ (dev_ಅಂಕೆ >= (size_t) ggml_backend_et_get_device_ಎಣಿಕೆ()) {
        ಹಿಂತಿರುಗಿ ಶೂನ್ಯಸೂಚಕ;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_ಅಂಕೆ);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    ಹಿಂತಿರುಗಿ dev_ctx->buftype;
}

ggml_backend_ಬಫರ್_type_t ggml_backend_et_host_ಬಫರ್_type(ಶೂನ್ಯ) {
    ಸ್ಥಿರ ggml_backend_ಬಫರ್_type host_ಬಫರ್_type = {
        /* .iface   = */ ggml_backend_et_ಬಫರ್_type_i,
        /* .device  = */ ಶೂನ್ಯಸೂಚಕ,
        /* .context = */ ಶೂನ್ಯಸೂಚಕ,
    };
    ಹಿಂತಿರುಗಿ &host_ಬಫರ್_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

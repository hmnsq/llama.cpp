#అన్తర్భావ "ggml-et.h"

#అన్తర్భావ "ggml-backend-impl.h"
#అన్తర్భావ "ggml-backend.h"
#అన్తర్భావ "ggml-et-common.h"
#అన్తర్భావ "ggml-et-kernels.h"
#అన్తర్భావ "ggml-et-memops.h"
#అన్తర్భావ "ggml-et-ops.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ <stdarg.h>

#అన్తర్భావ <cstdarg>
#అన్తర్భావ <cstdio>
#అన్తర్భావ <cstdlib>
#అన్తర్భావ <cstring>
#అన్తర్భావ <string>
#అన్తర్భావ <vector>

#ఒకవేళ __has_include(<filesystem>)
#    include <filesystem>
నామాకాశ fs = std::filesystem;
#లేదావేళ __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
నామాకాశ fs = std::experimental::filesystem;
#లేకపోతే
#    error "cannot include the filesystem library"
#అయితే_ముగింపు

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
స్థిర శూన్య ggml_et_dump_tensor_metadata(స్థిరరాశి ggml_tensor * ggtensor, size_t indent_level, స్థిరరాశి అక్షరం * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (దీర్ఘ దీర్ఘ) ggtensor->ne[0],
            (దీర్ఘ దీర్ఘ) ggtensor->ne[1], (దీర్ఘ దీర్ఘ) ggtensor->ne[2], (దీర్ఘ దీర్ఘ) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->దత్తాంశం, spaces.c_str(), (శూన్య *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
స్థిర శూన్య ggml_et_dump_operator_metadata(స్థిరరాశి ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != ఖాళీ);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        అక్షరం arr[16];
        పూర్ణాంకం  n = snprintf(arr, పరిమాణం(arr), "src[%i]->name", i);
        GGML_ASSERT((గుర్తులేని) n < పరిమాణం(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    ఒకవేళ (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

స్థిర నిర్మాణం ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_ప్రవాహం;
    std::unique_ptr<std::ofstream>                    kernel_id_ప్రవాహం;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    తార్కిక                                              profiling_enabled = అబద్ధం;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
స్థిర std::string ggml_et_get_default_et_మార్గం() {
    // List of environment variables to check in order of preference
    స్థిరరాశి అక్షరం * స్థిరరాశి env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    కొరకు (స్థిరరాశి అక్షరం * var : env_vars) {
        ఒకవేళ (స్థిరరాశి అక్షరం * et_మార్గం = std::getenv(var)) {
            ఒకవేళ (et_మార్గం && *et_మార్గం != '\0') {
                తిరిగి fs::path(et_మార్గం).string();
            }
        }
    }

    // Otherwise assume default
    తిరిగి fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
స్థిర అంతఃవరుస స్వయంచాలక ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_చక్రాలు        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_ముసుగు = 0x1FFFFFFFFu;
    స్థిరరాశి std::string  et_మార్గం                 = ggml_et_get_default_et_మార్గం() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_గమ్యం_BL2_Elf_మార్గం =
        et_మార్గం + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_మార్గం =
        et_మార్గం + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_మార్గం = et_మార్గం + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_మార్గం  = et_మార్గం + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.కార్మికుడు_Minion_Elf_మార్గం  = et_మార్గం + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_మార్గం       = et_మార్గం + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    స్థిరరాశి std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_గమ్యం_BL2_Elf_మార్గం, sysEmuOptions.sp_BL2_Elf_మార్గం,
        sysEmuOptions.machine_Minion_Elf_మార్గం,          sysEmuOptions.master_Minion_Elf_మార్గం,
        sysEmuOptions.కార్మికుడు_Minion_Elf_మార్గం,           sysEmuOptions.executable_మార్గం,
    };

    కొరకు (స్థిరరాశి స్వయంచాలక & file : required_files) {
        ఒకవేళ (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.నిరంతరం_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_చక్రాలు        = k_Sys_Emu_Max_చక్రాలు;
    sysEmuOptions.minion_Shires_ముసుగు = k_Sys_Emu_Minion_Shires_ముసుగు;
    sysEmuOptions.pu_Uart0_మార్గం      = sysEmuOptions.నిరంతరం_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_మార్గం      = sysEmuOptions.నిరంతరం_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_మార్గం      = sysEmuOptions.నిరంతరం_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_మార్గం      = sysEmuOptions.నిరంతరం_Dir + "spio_uart1_tx.log";
    sysEmuOptions.ప్రారంభం_Gdb         = అబద్ధం;
    sysEmuOptions.memcheck         = అబద్ధం;

    తిరిగి sysEmuOptions;
}

// Forward declaration
స్థిర శూన్య ggml_et_driver_cleanup();

స్థిర తార్కిక ggml_et_driver_init() {
    ఒకవేళ (_drv.runtime != శూన్యసూచిక) {
        ప్రతిజ్ఞా(_drv.device_layer != శూన్యసూచిక);
    } లేకపోతే {
        ప్రయత్నించు {
#ఒకవేళ defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#లేకపోతే
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#అయితే_ముగింపు  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            స్థిరరాశి అక్షరం * profile_మార్గం = getenv("GGML_ET_PROFILE");
            ఒకవేళ (profile_మార్గం) {
                std::string output_మార్గం    = std::string(profile_మార్గం) + "/et_runtime_trace.json";
                std::string kernel_id_మార్గం = std::string(profile_మార్గం) + "/kernel_id.json";

                _drv.profile_ప్రవాహం   = std::make_unique<std::ofstream>(output_మార్గం);
                _drv.kernel_id_ప్రవాహం = std::make_unique<std::ofstream>(kernel_id_మార్గం);
                ఒకవేళ (!_drv.profile_ప్రవాహం->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_మార్గం.c_str());
                    abort();
                }
                ఒకవేళ (!_drv.kernel_id_ప్రవాహం->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_మార్గం.c_str());
                    abort();
                }

                స్వయంచాలక * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_ప్రవాహం, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = నిజం;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } పట్టుకో (స్థిరరాశి std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            ఒకవేళ (_drv.device_layer != శూన్యసూచిక) {
                _drv.device_layer.reset();
            }
            ఒకవేళ (_drv.runtime != శూన్యసూచిక) {
                _drv.runtime.reset();
            }
            తిరిగి అబద్ధం;
        }
    }
    తిరిగి నిజం;
}

స్థిర std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    తిరిగి _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    తిరిగి _drv.runtime;
}

స్థిర శూన్య ggml_et_driver_cleanup() {
    ఒకవేళ (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        స్వయంచాలక * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = అబద్ధం;

        ఒకవేళ (_drv.profile_ప్రవాహం) {
            _drv.profile_ప్రవాహం->close();
            _drv.profile_ప్రవాహం.reset();
        }

        // Save kernel map
        ఒకవేళ (_drv.kernel_id_ప్రవాహం && !_drv.kernel_map.ఖాళీలు()) {
            స్వయంచాలక & os = *_drv.kernel_id_ప్రవాహం;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            కొరకు (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].మొదటి << "\": " << (పూర్ణాంకం) _drv.kernel_map[i].second;
                ఒకవేళ (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_ప్రవాహం->close();
            _drv.kernel_id_ప్రవాహం.reset();
        }
    }
}

స్థిర ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

స్థిర శూన్య ggml_backend_et_బఫరు_free_బఫరు(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_et_బఫరు_context * ctx = (ggml_backend_et_బఫరు_context *) బఫరు->context;
    ఒకవేళ (ctx->దత్తాంశం != శూన్యసూచిక) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        ఒకవేళ (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->దత్తాంశం));
        }
    }
    తొలగించు ctx;
}

స్థిర శూన్య * ggml_backend_et_బఫరు_get_పునాది(ggml_backend_బఫరు_t బఫరు) {
    ggml_backend_et_బఫరు_context * ctx = (ggml_backend_et_బఫరు_context *) బఫరు->context;
    తిరిగి ctx->దత్తాంశం;
}

స్థిర ggml_స్థితి ggml_backend_et_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    ఒకవేళ (tensor->view_src != ఖాళీ) {
        తిరిగి GGML_స్థితి_SUCCESS;
    }

    స్థిరరాశి size_t original_కొలత = ggml_nbytes(tensor);
    స్థిరరాశి size_t padded_కొలత   = ggml_backend_buft_get_alloc_కొలత(బఫరు->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    ఒకవేళ (padded_కొలత > original_కొలత) {
        స్థిరరాశి size_t padding_కొలత = padded_కొలత - original_కొలత;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) బఫరు->buft->device->context;
        ఒకవేళ (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            తిరిగి GGML_స్థితి_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->దత్తాంశం) + original_కొలత;
        ఒకవేళ (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_కొలత)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            తిరిగి GGML_స్థితి_FAILED;
        }
    }

    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_et_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు,
                                              ggml_tensor *         tensor,
                                              స్థిరరాశి శూన్య *          దత్తాంశం,
                                              size_t                స్థానభ్రంశం,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తిరిగి;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) బఫరు->buft->device->context;
    rt::ప్రవాహం_Id                     stream  = dev_ctx->default_ప్రవాహం;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->దత్తాంశం) + స్థానభ్రంశం;
    స్థిరరాశి std::byte * src_ptr = static_cast<స్థిరరాశి std::byte *>(దత్తాంశం);

    rt::EventId event = runtime->memcpy_Host_గమ్యం_Device(stream, src_ptr, dst_ptr, size, నిజం /*barrier*/);

    runtime->waitForEvent(event);
}

స్థిర శూన్య ggml_backend_et_బఫరు_get_tensor(ggml_backend_బఫరు_t బఫరు,
                                              స్థిరరాశి ggml_tensor *   tensor,
                                              శూన్య *                దత్తాంశం,
                                              size_t                స్థానభ్రంశం,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తిరిగి;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) బఫరు->buft->device->context;
    rt::ప్రవాహం_Id                     stream  = dev_ctx->default_ప్రవాహం;

    స్థిరరాశి std::byte * src_ptr = static_cast<స్థిరరాశి std::byte *>(tensor->దత్తాంశం) + స్థానభ్రంశం;
    std::byte *       dst_ptr = static_cast<std::byte *>(దత్తాంశం);

    rt::EventId event = runtime->memcpy_Device_గమ్యం_Host(stream, src_ptr, dst_ptr, size, నిజం /*barrier*/);

    runtime->waitForEvent(event);
}

స్థిర తార్కిక ggml_backend_et_బఫరు_cpy_tensor(ggml_backend_బఫరు_t బఫరు,
                                              స్థిరరాశి ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_వాడని(బఫరు);
    GGML_వాడని(src);
    GGML_వాడని(dst);
    తిరిగి అబద్ధం;
}

స్థిర శూన్య ggml_backend_et_బఫరు_clear(ggml_backend_బఫరు_t బఫరు, uint8_t value) {
    ggml_backend_et_బఫరు_context * ctx = (ggml_backend_et_బఫరు_context *) బఫరు->context;

    ఒకవేళ (ctx->size == 0 || ctx->దత్తాంశం == శూన్యసూచిక) {
        తిరిగి;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) బఫరు->buft->device->context;
    ఒకవేళ (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        తిరిగి;
    }

    // Use device-side memset kernel for efficient clearing
    ఒకవేళ (!ggml_et_memset(dev_ctx, ctx->దత్తాంశం, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        తిరిగి;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_బఫరు_i ggml_backend_et_బఫరు_i = {
    /* .free_buffer     = */ ggml_backend_et_బఫరు_free_బఫరు,
    /* .get_base        = */ ggml_backend_et_బఫరు_get_పునాది,
    /* .init_tensor     = */ ggml_backend_et_బఫరు_init_tensor,
    /* .memset_tensor   = */ ఖాళీ,
    /* .set_tensor      = */ ggml_backend_et_బఫరు_సమితి_tensor,
    /* .get_tensor      = */ ggml_backend_et_బఫరు_get_tensor,
    /* .set_tensor_2d   = */ ఖాళీ,
    /* .get_tensor_2d   = */ ఖాళీ,
    /* .cpy_tensor      = */ ggml_backend_et_బఫరు_cpy_tensor,
    /* .clear           = */ ggml_backend_et_బఫరు_clear,
    /* .reset           = */ ఖాళీ,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_et_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    GGML_వాడని(buft);
    తిరిగి GGML_ET_NAME;
}

స్థిర ggml_backend_బఫరు_t ggml_backend_et_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    ggml_backend_et_బఫరు_type_context * btctx = (ggml_backend_et_బఫరు_type_context *) buft->context;

    ggml_backend_et_బఫరు_context * ctx = కొత్త ggml_backend_et_బఫరు_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తొలగించు ctx;
        తిరిగి శూన్యసూచిక;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    ఒకవేళ (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        తొలగించు ctx;
        తిరిగి శూన్యసూచిక;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->దత్తాంశం = runtime->mallocDevice(ctx->rtid, size);
    ఒకవేళ (ctx->దత్తాంశం == శూన్యసూచిక) {
        తొలగించు ctx;
        తిరిగి శూన్యసూచిక;
    }

    తిరిగి ggml_backend_బఫరు_init(buft, ggml_backend_et_బఫరు_i, ctx, size);
}

స్థిర size_t ggml_backend_et_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime || !buft->device) {
        తిరిగి GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    తిరిగి prop.cache_పంక్తి_కొలత;
}

స్థిర size_t ggml_backend_et_బఫరు_type_get_max_కొలత(ggml_backend_బఫరు_type_t buft) {
    ఒకవేళ (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        తిరిగి dev_ctx->మొత్తం_mem;
    }
    తిరిగి SIZE_MAX;
}

స్థిర size_t ggml_backend_et_బఫరు_type_get_alloc_కొలత(ggml_backend_బఫరు_type_t buft, స్థిరరాశి ggml_tensor * tensor) {
    GGML_వాడని(buft);
    తిరిగి ggml_nbytes_pad(tensor);
}

స్థిర తార్కిక ggml_backend_et_బఫరు_type_is_host(ggml_backend_బఫరు_type_t buft) {
    GGML_వాడని(buft);
    తిరిగి అబద్ధం;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_బఫరు_type_i ggml_backend_et_బఫరు_type_i = {
    /* .get_name            = */ ggml_backend_et_బఫరు_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_బఫరు_type_alloc_బఫరు,
    /* .alloc_buffer_n      = */ ఖాళీ,
    /* .get_alignment       = */ ggml_backend_et_బఫరు_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_బఫరు_type_get_max_కొలత,
    /* .get_alloc_size      = */ ggml_backend_et_బఫరు_type_get_alloc_కొలత,
    /* .get_alloc_size_n    = */ ఖాళీ,
    /* .is_host             = */ ggml_backend_et_బఫరు_type_is_host,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_వాడని(backend);
    తిరిగి GGML_ET_NAME;
}

స్థిర శూన్య ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    ఒకవేళ (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        ఒకవేళ (_drv.profiling_enabled) {
            స్వయంచాలక kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        ఒకవేళ (runtime) {
            ఒకవేళ (dev_ctx->trace_బఫరు) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_బఫరు);
                dev_ctx->trace_బఫరు = శూన్యసూచిక;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_ప్రవాహం(dev_ctx->default_ప్రవాహం);
            కొరకు (స్వయంచాలక & స్థానం : dev_ctx->uberkernel.slots) {
                ఒకవేళ (స్థానం.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, స్థానం.device_insts);
                    స్థానం.device_insts = శూన్యసూచిక;
                }
                ఒకవేళ (స్థానం.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, స్థానం.device_params);
                    స్థానం.device_params = శూన్యసూచిక;
                }
                స్థానం.has_pending = అబద్ధం;
            }
        }
    }

    తొలగించు et_ctx;
    తొలగించు backend;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_et_get_default_బఫరు_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    తిరిగి ggml_backend_et_బఫరు_type(et_ctx->devidx);
}

స్థిర శూన్య ggml_backend_et_సమితి_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             స్థిరరాశి శూన్య *   దత్తాంశం,
                                             size_t         స్థానభ్రంశం,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తిరిగి;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::ప్రవాహం_Id                     stream  = dev_ctx->default_ప్రవాహం;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->దత్తాంశం) + స్థానభ్రంశం;
    స్థిరరాశి std::byte * src_ptr = static_cast<స్థిరరాశి std::byte *>(దత్తాంశం);

    runtime->memcpy_Host_గమ్యం_Device(stream, src_ptr, dst_ptr, size, నిజం /*barrier*/);
}

స్థిర శూన్య ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             స్థిరరాశి ggml_tensor * tensor,
                                             శూన్య *              దత్తాంశం,
                                             size_t              స్థానభ్రంశం,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తిరిగి;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::ప్రవాహం_Id                     stream  = dev_ctx->default_ప్రవాహం;

    స్థిరరాశి std::byte * src_ptr = static_cast<స్థిరరాశి std::byte *>(tensor->దత్తాంశం) + స్థానభ్రంశం;
    std::byte *       dst_ptr = static_cast<std::byte *>(దత్తాంశం);

    runtime->memcpy_Device_గమ్యం_Host(stream, src_ptr, dst_ptr, size, నిజం /*barrier*/);
}

స్థిర తార్కిక ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             స్థిరరాశి ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_వాడని(backend_src);
    GGML_వాడని(backend_dst);
    GGML_వాడని(src);
    GGML_వాడని(dst);
    తిరిగి అబద్ధం;
}

స్థిర శూన్య ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    ఒకవేళ (!runtime) {
        తిరిగి;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_ప్రవాహం(dev_ctx->default_ప్రవాహం);

    స్వయంచాలక errors = runtime->retrieve_ప్రవాహం_Errors(dev_ctx->default_ప్రవాహం);
    ఒకవేళ (errors.ఖాళీలు()) {
        తిరిగి;
    }
    కొరకు (స్థిరరాశి స్వయంచాలక & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (పూర్ణాంకం) err.error_సంకేతం,
                       (పూర్ణాంకం) err.errorContext_.value()[0].type_);
    }
    abort();
}

స్థిర తార్కిక ggml_et_can_fuse(స్థిరరాశి ggml_cgraph * cgraph, పూర్ణాంకం కణుపు_idx, std::initializer_list<ggml_op> ops) {
    ఒకవేళ (!ggml_can_fuse(cgraph, కణుపు_idx, ops)) {
        తిరిగి అబద్ధం;
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_కూడు) {
        స్థిరరాశి ggml_tensor * mm  = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * కూడు_2 = cgraph->nodes[కణుపు_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        ఒకవేళ (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        ఒకవేళ (కూడు_2->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }
        ఒకవేళ (కూడు_2->src[0] != mm && కూడు_2->src[1] != mm) {
            తిరిగి అబద్ధం;
        }

        స్థిరరాశి ggml_tensor * bias = (కూడు_2->src[0] == mm) ? కూడు_2->src[1] : కూడు_2->src[0];

        ఒకవేళ (bias->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        // No broadcasting: bias shape must equal MM output shape.
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
            ఒకవేళ (bias->ne[i] != mm->ne[i]) {
                తిరిగి అబద్ధం;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        ఒకవేళ (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            తిరిగి అబద్ధం;
        }
        కొరకు (పూర్ణాంకం i = 0; i < GGML_MAX_DIMS; ++i) {
            ఒకవేళ ((int64_t) bias->nb[i] != (int64_t) కూడు_2->nb[i]) {
                తిరిగి అబద్ధం;
            }
        }
    }

    ఒకవేళ (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        స్థిరరాశి ggml_tensor * rms_norm = cgraph->nodes[కణుపు_idx];
        స్థిరరాశి ggml_tensor * mul      = cgraph->nodes[కణుపు_idx + 1];

        // ET only supports F32
        ఒకవేళ (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        స్థిరరాశి ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        ఒకవేళ (weights->type != GGML_TYPE_F32) {
            తిరిగి అబద్ధం;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        ఒకవేళ (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_అడ్డువరుసలు(weights)) {
            తిరిగి అబద్ధం;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        ఒకవేళ (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            తిరిగి అబద్ధం;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        ఒకవేళ (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            తిరిగి అబద్ధం;
        }
    }

    తిరిగి నిజం;
}

స్థిర ggml_స్థితి ggml_backend_et_లేఖాచిత్రం_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_లేఖాచిత్రం(&dev_ctx->uberkernel);

    కొరకు (పూర్ణాంకం i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * కణుపు = cgraph->nodes[i];

        ఒకవేళ (కణుపు->op == GGML_OP_NONE || కణుపు->op == GGML_OP_VIEW || కణుపు->op == GGML_OP_RESHAPE ||
            కణుపు->op == GGML_OP_క్రమమార్పు || కణుపు->op == GGML_OP_TRANSPOSE) {
            కొనసాగు;
        }

        // --- Fusion checks (before regular dispatch) ---
        ఒకవేళ (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, కణుపు, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            కొనసాగు;
        }
        ఒకవేళ (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_కూడు })) {
            ggml_et_op_mul_mat(dev_ctx, కణుపు, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            కొనసాగు;
        }

        ఎంపిక (కణుపు->op) {
            స్థితి GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
                ggml_et_op_కూడిక_అడ్డువరుసలు(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_సగటు:
                ggml_et_op_సగటు(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_కూడు:
                ggml_et_op_కూడు(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_సమూహం_NORM:
                ggml_et_op_సమూహం_norm(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_IM2_నిలువు:
                ggml_et_op_im2col(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_GET_అడ్డువరుసలు:
                ggml_et_op_get_అడ్డువరుసలు(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_SSM_పరిశీలన:
                ggml_et_op_ssm_పరిశీలన(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
                ggml_et_op_సమితి_అడ్డువరుసలు(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_సమితి:
                ggml_et_op_సమితి(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, కణుపు);
                ఆపు;

            స్థితి GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, కణుపు);
                ఆపు;

            అప్రమేయ:
                ggml_et_uberkernel_abort_లేఖాచిత్రం(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(కణుపు->op));
                తిరిగి GGML_స్థితి_FAILED;
        }

        ఒకవేళ (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_లేఖాచిత్రం(&dev_ctx->uberkernel);
            తిరిగి GGML_స్థితి_FAILED;
        }
    }

    ఒకవేళ (!ggml_et_uberkernel_చివర_లేఖాచిత్రం(dev_ctx)) {
        ggml_et_uberkernel_abort_లేఖాచిత్రం(&dev_ctx->uberkernel);
        తిరిగి GGML_స్థితి_FAILED;
    }

    తిరిగి GGML_స్థితి_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
స్థిర తార్కిక et_ggml_is_అడ్డువరుస_contiguous(స్థిరరాశి ggml_tensor * t) {
    తిరిగి t->nb[0] == ggml_type_కొలత(t->type);
}

స్థిర తార్కిక ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    GGML_వాడని(dev);

    తార్కిక supported = అబద్ధం;
    ఎంపిక (op->op) {
        స్థితి GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == పరిమాణం(దశాంశం) && ggml_is_contiguous(op);
            ఆపు;
        స్థితి GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_కూడిక_అడ్డువరుసలు:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_సగటు:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == పరిమాణం(దశాంశం) && op->src[0]->nb[0] == పరిమాణం(దశాంశం)) {
                ఎంపిక (ggml_get_unary_op(op)) {
                    స్థితి GGML_UNARY_OP_ABS:
                    స్థితి GGML_UNARY_OP_SGN:
                    స్థితి GGML_UNARY_OP_NEG:
                    స్థితి GGML_UNARY_OP_అడుగు:
                    స్థితి GGML_UNARY_OP_TANH:
                    స్థితి GGML_UNARY_OP_ELU:
                    స్థితి GGML_UNARY_OP_RELU:
                    స్థితి GGML_UNARY_OP_SIGMOID:
                    స్థితి GGML_UNARY_OP_GELU:
                    స్థితి GGML_UNARY_OP_GELU_శీఘ్ర:
                    స్థితి GGML_UNARY_OP_SILU:
                    స్థితి GGML_UNARY_OP_HARDSWISH:
                    స్థితి GGML_UNARY_OP_HARDSIGMOID:
                    స్థితి GGML_UNARY_OP_EXP:
                    స్థితి GGML_UNARY_OP_EXPM1:
                    స్థితి GGML_UNARY_OP_SOFTPLUS:
                    స్థితి GGML_UNARY_OP_GELU_ERF:
                    స్థితి GGML_UNARY_OP_FLOOR:
                    స్థితి GGML_UNARY_OP_CEIL:
                    స్థితి GGML_UNARY_OP_విడత:
                    స్థితి GGML_UNARY_OP_TRUNC:
                        supported = నిజం;
                        ఆపు;
                    అప్రమేయ:
                        ఆపు;
                }
            }
            ఆపు;
        స్థితి GGML_OP_MUL:
        స్థితి GGML_OP_కూడు:
        స్థితి GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == పరిమాణం(దశాంశం) &&
                        op->src[0]->nb[0] == పరిమాణం(దశాంశం) &&
                        (op->src[1]->nb[0] == పరిమాణం(దశాంశం) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * పరిమాణం(దశాంశం);
            ఆపు;
        స్థితి GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            ఒకవేళ (op->type == GGML_TYPE_F32 &&
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
                supported = నిజం;
            } లేకపోతే ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                తార్కిక src0_మొదటి_dim_contiguous = (op->src[0]->nb[0] == ggml_type_కొలత(op->src[0]->type));
                తార్కిక src1_మొదటి_dim_contiguous = (op->src[1]->nb[0] == ggml_type_కొలత(op->src[1]->type));
                తార్కిక dst_మొదటి_dim_contiguous  = (op->nb[0] == పరిమాణం(దశాంశం));

                // Check destination stride ordering (only for dimensions with ne > 1)
                తార్కిక dst_properly_ordered = నిజం;
                కొరకు (పూర్ణాంకం d = 0; d < 3; d++) {
                    ఒకవేళ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = అబద్ధం;
                    }
                }

                supported = src0_మొదటి_dim_contiguous && src1_మొదటి_dim_contiguous && dst_మొదటి_dim_contiguous &&
                            dst_properly_ordered;
            } లేకపోతే ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                తార్కిక src0_మొదటి_dim_contiguous = (op->src[0]->nb[0] == ggml_type_కొలత(op->src[0]->type));
                తార్కిక src1_మొదటి_dim_contiguous = (op->src[1]->nb[0] == ggml_type_కొలత(op->src[1]->type));
                తార్కిక dst_మొదటి_dim_contiguous  = (op->nb[0] == పరిమాణం(దశాంశం));

                తార్కిక dst_properly_ordered = నిజం;
                కొరకు (పూర్ణాంకం d = 0; d < 3; d++) {
                    ఒకవేళ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = అబద్ధం;
                    }
                }

                supported = src0_మొదటి_dim_contiguous && src1_మొదటి_dim_contiguous && dst_మొదటి_dim_contiguous &&
                            dst_properly_ordered;

            } లేకపోతే ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                తార్కిక src0_మొదటి_dim_contiguous = (op->src[0]->nb[0] == ggml_type_కొలత(op->src[0]->type));
                తార్కిక src1_మొదటి_dim_contiguous = (op->src[1]->nb[0] == ggml_type_కొలత(op->src[1]->type));
                తార్కిక dst_మొదటి_dim_contiguous  = (op->nb[0] == పరిమాణం(దశాంశం));

                తార్కిక dst_properly_ordered = నిజం;
                కొరకు (పూర్ణాంకం d = 0; d < 3; d++) {
                    ఒకవేళ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = అబద్ధం;
                    }
                }

                supported = src0_మొదటి_dim_contiguous && src1_మొదటి_dim_contiguous && dst_మొదటి_dim_contiguous &&
                            dst_properly_ordered;
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                తార్కిక src0_మొదటి_dim_contiguous = (op->src[0]->nb[0] == ggml_type_కొలత(op->src[0]->type));
                తార్కిక src1_మొదటి_dim_contiguous = (op->src[1]->nb[0] == ggml_type_కొలత(op->src[1]->type));
                తార్కిక src2_మొదటి_dim_contiguous = (op->src[2]->nb[0] == ggml_type_కొలత(op->src[2]->type));
                తార్కిక dst_మొదటి_dim_contiguous  = (op->nb[0] == పరిమాణం(దశాంశం));

                // Check destination stride ordering (only for dimensions with ne > 1)
                తార్కిక dst_properly_ordered = నిజం;
                కొరకు (పూర్ణాంకం d = 0; d < 3; d++) {
                    ఒకవేళ (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = అబద్ధం;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                తార్కిక dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_మొదటి_dim_contiguous && src1_మొదటి_dim_contiguous && src2_మొదటి_dim_contiguous &&
                            dst_మొదటి_dim_contiguous && dst_properly_ordered && dims_valid;
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_అడ్డువరుస_contiguous(op->src[0])) {
                స్థిరరాశి పూర్ణాంకం  mode             = ggml_get_op_params_i32(op, 2);
                స్థిరరాశి పూర్ణాంకం  ndims            = ggml_get_op_params_i32(op, 1);
                స్థిరరాశి తార్కిక is_సాధారణ        = mode == GGML_ROPE_TYPE_సాధారణ;
                స్థిరరాశి తార్కిక is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                స్థిరరాశి తార్కిక is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                స్థిరరాశి తార్కిక zero_view_స్థానభ్రంశం = op->src[0]->view_src == శూన్యసూచిక || op->src[0]->view_offs == 0;
                స్థిరరాశి తార్కిక has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                స్థిరరాశి తార్కిక zero_rot_స్థానభ్రంశం  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_స్థానభ్రంశం && zero_rot_స్థానభ్రంశం && ndims <= 512 &&
                    (is_సాధారణ || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_అడ్డువరుస_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_అడ్డువరుస_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_అడ్డువరుస_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_సమూహం_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_అడ్డువరుస_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            ఆపు;
        స్థితి GGML_OP_IM2_నిలువు:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_కొలత(op->type) && op->src[1]->nb[0] == ggml_type_కొలత(op->src[1]->type);
            ఆపు;
        స్థితి GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                ఒకవేళ (!op->src[0] || !op->src[1]) {
                    supported = అబద్ధం;
                    ఆపు;
                }
                ఒకవేళ (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = అబద్ధం;
                    ఆపు;
                }
                ఒకవేళ (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = అబద్ధం;
                    ఆపు;
                }

                స్థిరరాశి ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                స్థిరరాశి ggml_tensor * లోపల  = op->src[1];  // [W,  H,  Cin, N]
                స్థిరరాశి int32_t       s0  = ggml_get_op_params_i32(op, 0);
                స్థిరరాశి int32_t       s1  = ggml_get_op_params_i32(op, 1);
                స్థిరరాశి int32_t       p0  = ggml_get_op_params_i32(op, 2);
                స్థిరరాశి int32_t       p1  = ggml_get_op_params_i32(op, 3);
                స్థిరరాశి int32_t       d0  = ggml_get_op_params_i32(op, 4);
                స్థిరరాశి int32_t       d1  = ggml_get_op_params_i32(op, 5);

                స్థిరరాశి int64_t Kw   = flt->ne[0];
                స్థిరరాశి int64_t Kh   = flt->ne[1];
                స్థిరరాశి int64_t Cin  = flt->ne[2];
                స్థిరరాశి int64_t Cout = flt->ne[3];
                స్థిరరాశి int64_t H    = లోపల->ne[1];
                (శూన్య) లోపల->ne[0];

                ఒకవేళ (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || లోపల->ne[3] != 1) {
                    supported = అబద్ధం;
                    ఆపు;
                }
                స్థిరరాశి int64_t OW = op->ne[0];
                స్థిరరాశి int64_t OH = op->ne[1];
                ఒకవేళ (OW <= 0 || OH <= 0) {
                    supported = అబద్ధం;
                    ఆపు;
                }
                (శూన్య) p0;
                (శూన్య) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                స్థిరరాశి int64_t Hp            = H + 2 * p1;
                స్థిరరాశి int64_t OW_pad        = (OW + 15) & ~15;
                స్థిరరాశి int64_t Wp_a          = OW_pad;
                స్థిరరాశి తార్కిక    need_stage    = (OW % 16 != 0);
                స్థిరరాశి int64_t stage_బైట్లు   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                స్థిరరాశి int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                స్థిరరాశి int64_t scratch_బైట్లు = 32 * 2 * 16 * 16 * 4;
                స్థిరరాశి int64_t budget        = L2SCP_BUDGET - stage_బైట్లు - scratch_బైట్లు;
                స్థిరరాశి int64_t చొప్పున_KT_బైట్లు  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                స్థిరరాశి int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                ఒకవేళ (K_TILES * చొప్పున_KT_బైట్లు <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } లేకపోతే {
                    chunk_KT_calc = K_TILES;
                    అయితే (chunk_KT_calc > 1 && 2 * chunk_KT_calc * చొప్పున_KT_బైట్లు > budget) {
                        chunk_KT_calc--;
                    }
                    అయితే (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    ఒకవేళ (chunk_KT_calc < 1) {
                        supported = అబద్ధం;
                        ఆపు;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                ఒకవేళ (n_chunks_calc > 1) {
                    స్థిరరాశి int64_t M_TILES     = Cout / 16;
                    స్థిరరాశి int64_t w_tiles     = (OW + 15) / 16;
                    స్థిరరాశి int64_t మొత్తం_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    స్థిరరాశి int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    ఒకవేళ (మొత్తం_tiles > max_workers) {
                        supported = అబద్ధం;
                        ఆపు;
                    }
                }

                supported = నిజం;
                ఆపు;
            }
        స్థితి GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            ఆపు;
        స్థితి GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                స్థిరరాశి తార్కిక  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_శీఘ్ర || glu_type == GGML_GLU_OP_REGLU;

                ఒకవేళ (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } లేకపోతే {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_SOFT_MAX:
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                ఒకవేళ (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    ఒకవేళ (!supported) {
                        ఆపు;
                    }
                }
                // Check optional sinks tensor (F32 only)
                ఒకవేళ (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } లేకపోతే {
                    supported = నిజం;
                }
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_SSM_పరిశీలన:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == పరిమాణం(దశాంశం) && op->src[4]->nb[0] == పరిమాణం(దశాంశం) &&
                        op->src[5]->nb[0] == పరిమాణం(దశాంశం) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * పరిమాణం(దశాంశం) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * పరిమాణం(దశాంశం) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * పరిమాణం(దశాంశం) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            ఆపు;
        స్థితి GGML_OP_FLASH_ATTN_EXT:
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == శూన్యసూచిక &&
                ggml_is_contiguous_అడ్డువరుసలు(op) && ggml_is_contiguous_అడ్డువరుసలు(op->src[0])) {
                దశాంశం max_bias      = 0.0f;
                దశాంశం logit_softcap = 0.0f;
                జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) op->op_params + 1, పరిమాణం(max_bias));
                జ్ఞాపకం_నకలు(&logit_softcap, (స్థిరరాశి దశాంశం *) op->op_params + 2, పరిమాణం(logit_softcap));

                స్థిరరాశి ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                తార్కిక ముసుగు_ok = (op->src[3] == శూన్యసూచిక) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                స్థిరరాశి int64_t nhq = op->src[0]->ne[2];
                స్థిరరాశి int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                స్థిరరాశి size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                స్థిరరాశి size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                స్థిరరాశి తార్కిక me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && ముసుగు_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == పరిమాణం(దశాంశం) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == పరిమాణం(దశాంశం) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_GET_అడ్డువరుసలు:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            ఒకవేళ ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                ఒకవేళ (op->దత్తాంశం && op->src[0]->దత్తాంశం && op->దత్తాంశం == op->src[0]->దత్తాంశం) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = అబద్ధం;
                } లేకపోతే {
                    supported = నిజం;
                }
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            ఒకవేళ (op->src[0]) {
                స్థిరరాశి int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                ఒకవేళ (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = నిజం;
                } లేకపోతే ఒకవేళ ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    ఒకవేళ (op->దత్తాంశం && op->src[0]->దత్తాంశం && op->దత్తాంశం == op->src[0]->దత్తాంశం) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = అబద్ధం;
                    } లేకపోతే {
                        supported = నిజం;
                    }
                } లేకపోతే ఒకవేళ (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = నిజం;
                } లేకపోతే {
                    supported = అబద్ధం;
                }
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_CONCAT:
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                స్థిరరాశి int32_t dim = ((స్థిరరాశి int32_t *) op->op_params)[0];
                ఒకవేళ (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = నిజం;
                } లేకపోతే ఒకవేళ (dim == 0 && ((op->src[0]->nb[0] % పరిమాణం(దశాంశం) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % పరిమాణం(దశాంశం) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = నిజం;
                } లేకపోతే ఒకవేళ (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = నిజం;
                }
            }
            ఆపు;
        స్థితి GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == పరిమాణం(దశాంశం) &&
                        op->src[1]->nb[0] == పరిమాణం(దశాంశం) && op->src[0]->nb[1] == op->src[0]->ne[0] * పరిమాణం(దశాంశం) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * పరిమాణం(దశాంశం) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            ఆపు;
        స్థితి GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == పరిమాణం(దశాంశం)) {
                స్థిరరాశి int32_t lp0      = ((స్థిరరాశి int32_t *) op->op_params)[0];
                స్థిరరాశి int32_t rp0      = ((స్థిరరాశి int32_t *) op->op_params)[1];
                స్థిరరాశి తార్కిక    circular = (తార్కిక) ((స్థిరరాశి int32_t *) op->op_params)[8];
                ఒకవేళ (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = నిజం;
                } లేకపోతే {
                    supported = అబద్ధం;
                }
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = నిజం;
            } లేకపోతే ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = నిజం;
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            ఆపు;
        స్థితి GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            ఆపు;
        స్థితి GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            ఆపు;
        స్థితి GGML_OP_సమితి:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                స్థిరరాశి తార్కిక   inplace = (తార్కిక) ((స్థిరరాశి int32_t *) op->op_params)[4];
                స్థిరరాశి size_t nb1     = ((స్థిరరాశి int32_t *) op->op_params)[0];
                స్థిరరాశి size_t nb2     = ((స్థిరరాశి int32_t *) op->op_params)[1];
                స్థిరరాశి size_t nb3     = ((స్థిరరాశి int32_t *) op->op_params)[2];
                స్థిరరాశి size_t స్థానభ్రంశం  = ((స్థిరరాశి int32_t *) op->op_params)[3];
                స్థిరరాశి size_t nb0     = ggml_element_కొలత(op);
                స్థిరరాశి size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                స్థిరరాశి size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                స్థిరరాశి size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                స్థిరరాశి size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                స్థిరరాశి తార్కిక view_bounds_ok = స్థానభ్రంశం + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                స్థిరరాశి తార్కిక cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (స్థానభ్రంశం % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            ఆపు;
        స్థితి GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = నిజం;
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = నిజం;
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            ఒకవేళ (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&                               // k
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&                               // v
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&                               // g
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&                               // beta
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&                               // state
                op->src[2]->ne[0] % 8 == 0 &&                                                    // S_v multiple of 8
                (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[2]->ne[0]) &&  // g is scalar or per-element
                op->src[4]->ne[0] == 1 &&                                              // beta is scalar per position
                et_ggml_is_అడ్డువరుస_contiguous(op->src[0]) && et_ggml_is_అడ్డువరుస_contiguous(op->src[1]) &&
                et_ggml_is_అడ్డువరుస_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = నిజం;
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_VIEW:
        స్థితి GGML_OP_క్రమమార్పు:
        స్థితి GGML_OP_TRANSPOSE:
        స్థితి GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = నిజం;
            ఆపు;
        స్థితి GGML_OP_సమితి_అడ్డువరుసలు:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            ఒకవేళ (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_అడ్డువరుసలు(op) &&
                ggml_is_contiguous_అడ్డువరుసలు(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } లేకపోతే {
                supported = అబద్ధం;
            }
            ఆపు;
        స్థితి GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = నిజం;
            ఆపు;
        అప్రమేయ:
            supported = అబద్ధం;
            ఆపు;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    తిరిగి supported;
}

స్థిర తార్కిక ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_బఫరు_type_t buft) {
    GGML_వాడని(dev);
    తిరిగి buft->iface.get_name == ggml_backend_et_బఫరు_type_get_name;
}

స్థిర తార్కిక ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, స్థిరరాశి ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    ఒకవేళ (op->op == GGML_OP_GET_అడ్డువరుసలు) {
        తిరిగి అబద్ధం;
    }
    తిరిగి నిజం;

    GGML_వాడని(dev);
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_సమితి_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ ఖాళీ,
    /* .get_tensor_2d_async     = */ ఖాళీ,
    /* .cpy_tensor_async        = */ ఖాళీ,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ ఖాళీ,
    /* .graph_plan_free         = */ ఖాళీ,
    /* .graph_plan_update       = */ ఖాళీ,
    /* .graph_plan_compute      = */ ఖాళీ,
    /* .graph_compute           = */ ggml_backend_et_లేఖాచిత్రం_compute,
    /* .event_record            = */ ఖాళీ,
    /* .event_wait              = */ ఖాళీ,
    /* .graph_optimize          = */ ఖాళీ,
};

స్థిర స్థిరరాశి అక్షరం * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    తిరిగి dev_ctx->name.c_str();
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    తిరిగి dev_ctx->desc.c_str();
}

స్థిర శూన్య ggml_backend_et_device_get_జ్ఞాపకం(ggml_backend_dev_t dev, size_t * విడుదల, size_t * మొత్తం) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *విడుదల                                    = dev_ctx->మొత్తం_mem;
    *మొత్తం                                   = dev_ctx->మొత్తం_mem;
}

స్థిర గణన ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_వాడని(dev);
    తిరిగి GGML_BACKEND_DEVICE_TYPE_GPU;
}

స్థిర శూన్య ggml_backend_et_device_get_props(ggml_backend_dev_t dev, నిర్మాణం ggml_backend_dev_props * props) {
    GGML_వాడని(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_జ్ఞాపకం(dev, &props->జ్ఞాపకం_free, &props->జ్ఞాపకం_మొత్తం);
    props->device_id = ఖాళీ;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ నిజం,
        /* .host_buffer           = */ అబద్ధం,
        /* .buffer_from_host_ptr  = */ అబద్ధం,
        /* .events                = */ అబద్ధం,
        /* .mmap_support          = */ నిజం,
    };
}

స్థిర ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, స్థిరరాశి అక్షరం * params) {
    GGML_వాడని(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    తిరిగి ggml_backend_et_init(dev_ctx->devidx);
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_et_device_get_బఫరు_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    తిరిగి dev_ctx->buftype;
}

స్థిర ggml_backend_బఫరు_type_t ggml_backend_et_device_get_host_బఫరు_type(ggml_backend_dev_t dev) {
    GGML_వాడని(dev);
    తిరిగి ggml_backend_cpu_బఫరు_type();
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_జ్ఞాపకం,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_బఫరు_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_బఫరు_type,
    /* .buffer_from_host_ptr = */ ఖాళీ,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ ఖాళీ,
    /* .event_free        = */ ఖాళీ,
    /* .event_synchronize = */ ఖాళీ,
};

/*
  Backend Registry.
*/

స్థిర స్థిరరాశి అక్షరం * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_వాడని(reg);
    తిరిగి GGML_ET_NAME;
}

స్థిర size_t ggml_backend_et_reg_get_device_లెక్క(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    తిరిగి ctx->devices.size();
}

స్థిర ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    ఒకవేళ (devidx >= ctx->devices.size()) {
        తిరిగి శూన్యసూచిక;
    }
    తిరిగి ctx->devices[devidx];
}

స్థిర శూన్య * ggml_backend_et_get_proc_చిరునామా(ggml_backend_reg_t reg, స్థిరరాశి అక్షరం * name) {
    GGML_వాడని(reg);
    GGML_వాడని(name);
    తిరిగి శూన్యసూచిక;
}

స్థిర స్థిరరాశి నిర్మాణం ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_లెక్క,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_చిరునామా,
};

ggml_backend_reg_t ggml_backend_et_reg(శూన్య) {
    స్థిర ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = కొత్త ggml_backend_et_reg_ctx;

        ఒకవేళ (!ggml_et_driver_init()) {
            తిరిగి శూన్యసూచిక;
        }

        ggml_backend_reg_t r = కొత్త ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_సంస్కరణ,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ శూన్యసూచిక,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        కొరకు (పూర్ణాంకం i = 0; i < ggml_et_devicelayer()->get_Devices_లెక్క(); i++) {
            ggml_backend_dev_t dev = కొత్త ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ శూన్యసూచిక  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = కొత్త ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->మొత్తం_mem                       = static_cast<size_t>(prop.జ్ఞాపకం_కొలత);
            {
                స్థిరరాశి అక్షరం * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && strcmp(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_బఫరు_type_context * bufty_ctx = కొత్త ggml_backend_et_బఫరు_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = కొత్త ggml_backend_బఫరు_type{ /* .iface   = */ ggml_backend_et_బఫరు_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_ప్రవాహం = ggml_et_runtime()->create_ప్రవాహం(rtid);

            dev_ctx->trace_బఫరు = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_బఫరు_కొలత);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            కొరకు (స్వయంచాలక & స్థానం : dev_ctx->uberkernel.slots) {
                స్థానం.insts.reserve(256);
                స్థానం.params_blob.reserve(1 << 20);
                స్థానం.device_insts_సామర్థ్యం  = 256 * పరిమాణం(ggml_et_uberkernel_inst);
                స్థానం.device_params_సామర్థ్యం = 1 << 20;
                స్థానం.device_insts           = ggml_et_runtime()->mallocDevice(rtid, స్థానం.device_insts_సామర్థ్యం);
                స్థానం.device_params          = ggml_et_runtime()->mallocDevice(rtid, స్థానం.device_params_సామర్థ్యం);
                ఒకవేళ (స్థానం.device_insts == శూన్యసూచిక) {
                    స్థానం.device_insts_సామర్థ్యం = 0;
                }
                ఒకవేళ (స్థానం.device_params == శూన్యసూచిక) {
                    స్థానం.device_params_సామర్థ్యం = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        తిరిగి r;
    }();

    తిరిగి _reg;
}

ggml_guid_t ggml_backend_et_guid(శూన్య) {
    స్థిర ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    తిరిగి &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    ఒకవేళ (!ggml_et_driver_init()) {
        తిరిగి శూన్యసూచిక;
    }

    ఒకవేళ (devidx >= (size_t) ggml_backend_et_get_device_లెక్క()) {
        తిరిగి శూన్యసూచిక;
    }

    ggml_backend_et_context * ctx = కొత్త ggml_backend_et_context;
    ctx->devidx                   = (పూర్ణాంకం) devidx;

    ggml_backend_t backend = కొత్త ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    తిరిగి backend;
}

తార్కిక ggml_backend_is_et(ggml_backend_t backend) {
    తిరిగి backend != ఖాళీ && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

పూర్ణాంకం ggml_backend_et_get_device_లెక్క(శూన్య) {
    తిరిగి ggml_backend_et_reg_get_device_లెక్క(ggml_backend_et_reg());
}

శూన్య ggml_backend_et_get_device_description(పూర్ణాంకం devidx, అక్షరం * description, size_t description_కొలత) {
    ఒకవేళ (devidx < 0 || devidx >= ggml_backend_et_get_device_లెక్క()) {
        snprintf(description, description_కొలత, "ET Device %d (invalid)", devidx);
        తిరిగి;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_కొలత, "%s", dev_ctx->desc.c_str());
}

శూన్య ggml_backend_et_get_device_జ్ఞాపకం(పూర్ణాంకం devidx, size_t * విడుదల, size_t * మొత్తం) {
    ఒకవేళ (devidx < 0 || devidx >= ggml_backend_et_get_device_లెక్క()) {
        *విడుదల  = 0;
        *మొత్తం = 0;
        తిరిగి;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_జ్ఞాపకం(dev, విడుదల, మొత్తం);
}

ggml_backend_బఫరు_type_t ggml_backend_et_బఫరు_type(size_t dev_అంకె) {
    ఒకవేళ (dev_అంకె >= (size_t) ggml_backend_et_get_device_లెక్క()) {
        తిరిగి శూన్యసూచిక;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_అంకె);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    తిరిగి dev_ctx->buftype;
}

ggml_backend_బఫరు_type_t ggml_backend_et_host_బఫరు_type(శూన్య) {
    స్థిర ggml_backend_బఫరు_type host_బఫరు_type = {
        /* .iface   = */ ggml_backend_et_బఫరు_type_i,
        /* .device  = */ శూన్యసూచిక,
        /* .context = */ శూన్యసూచిక,
    };
    తిరిగి &host_బఫరు_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)

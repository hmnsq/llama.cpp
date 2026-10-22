#अन्तर्भाव "ggml-et.h"

#अन्तर्भाव "ggml-backend-impl.h"
#अन्तर्भाव "ggml-backend.h"
#अन्तर्भाव "ggml-et-common.h"
#अन्तर्भाव "ggml-et-kernels.h"
#अन्तर्भाव "ggml-et-memops.h"
#अन्तर्भाव "ggml-et-ops.h"
#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव <stdarg.h>

#अन्तर्भाव <cstdarg>
#अन्तर्भाव <cstdio>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>
#अन्तर्भाव <string>
#अन्तर्भाव <vector>

#यदि __has_include(<filesystem>)
#    include <filesystem>
नामाकाश fs = std::filesystem;
#नोचेद्यदि __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
नामाकाश fs = std::experimental::filesystem;
#अन्यथा
#    error "cannot include the filesystem library"
#यदि_अन्तः

/*
 * ggml_et_dump_tensor_metadata
 * @brief prints the metadata of a single tensorf
 */
स्थिर शून्य ggml_et_dump_tensor_metadata(ध्रुव ggml_tensor * ggtensor, size_t indent_level, ध्रुव वर्ण * title) {
    std::string spaces(indent_level, ' ');
    fprintf(stderr,
            "%s%s: %s\n"
            "%s  type: %s\n"
            "%s  ne: %lld %lld %lld %lld\n"
            "%s  nb: %zu %zu %zu %zu\n"
            "%s  op: %s\n"
            "%s  data: %p\n"
            "%s  src0: %p\n",
            spaces.c_str(), title, ggtensor->name, spaces.c_str(), ggml_type_name(ggtensor->type), spaces.c_str(), (दीर्घ दीर्घ) ggtensor->ne[0],
            (दीर्घ दीर्घ) ggtensor->ne[1], (दीर्घ दीर्घ) ggtensor->ne[2], (दीर्घ दीर्घ) ggtensor->ne[3], spaces.c_str(),
            ggtensor->nb[0], ggtensor->nb[1], ggtensor->nb[2], ggtensor->nb[3], spaces.c_str(), ggml_op_name(ggtensor->op),
            spaces.c_str(), ggtensor->दत्त, spaces.c_str(), (शून्य *) ggtensor->src[0]);
}

/*
 * ggml_et_dump_operator_metadata
 * @brief prints the metadata of a single tensor (or operator) including it's input and views
 */
स्थिर शून्य ggml_et_dump_operator_metadata(ध्रुव ggml_tensor * ggtensor) {
    GGML_ASSERT(ggtensor != रिक्त);
    ggml_et_dump_tensor_metadata(ggtensor, 0, "GGML tensor");
    प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_SRC && ggtensor->src[i]; i++) {
        वर्ण arr[16];
        पूर्णाङ्क  n = snprintf(arr, परिमाण(arr), "src[%i]->name", i);
        GGML_ASSERT((अचिह्नित) n < परिमाण(arr) && "printed too much data to stack buffer");
        ggml_et_dump_tensor_metadata(ggtensor->src[i], 2, arr);
    }
    यदि (ggtensor->view_src) {
        ggml_et_dump_tensor_metadata(ggtensor, 2, "view_src");
    }
}

स्थिर संरचना ggml_et_driver {
    std::shared_ptr<dev::IDeviceLayer>                device_layer;
    std::shared_ptr<rt::IRuntime>                     runtime;
    std::unique_ptr<std::ofstream>                    profile_प्रवाहः;
    std::unique_ptr<std::ofstream>                    kernel_id_प्रवाहः;
    std::vector<std::pair<std::string, rt::KernelId>> kernel_map;
    तार्किक                                              profiling_enabled = असत्य;
} _drv;

// Check at runtime environment variables for paths likely holding ET toolchain with sysemu elf files
स्थिर std::string ggml_et_get_default_et_मार्गः() {
    // List of environment variables to check in order of preference
    ध्रुव वर्ण * ध्रुव env_vars[] = { "ET_TOOLCHAIN", "TOOLCHAIN_ROOT" };

    प्रति (ध्रुव वर्ण * var : env_vars) {
        यदि (ध्रुव वर्ण * et_मार्गः = std::getenv(var)) {
            यदि (et_मार्गः && *et_मार्गः != '\0') {
                प्रत्यागम fs::path(et_मार्गः).string();
            }
        }
    }

    // Otherwise assume default
    प्रत्यागम fs::path("/opt/et").string();
}

// config when using sysemu instead of PCIe hardware device
// adapted from `ainekko/et-platform/esperanto-tools-libs/tools/src/bench.cpp`
स्थिर अन्तःस्थ स्वचालित ggml_et_get_default_sysemu_options() {
    constexpr uint64_t k_Sys_Emu_Max_आवृत्तयः        = std::numeric_limits<uint64_t>::max();
    constexpr uint64_t k_Sys_Emu_Minion_Shires_आवरणम् = 0x1FFFFFFFFu;
    ध्रुव std::string  et_मार्गः                 = ggml_et_get_default_et_मार्गः() + "/";

    emu::SysEmuOptions sysEmuOptions;

    // Construct all paths
    sysEmuOptions.bootrom_Trampoline_गन्तव्यम्_BL2_Elf_मार्गः =
        et_मार्गः + "lib/esperanto-fw/BootromTrampolineToBL2/BootromTrampolineToBL2.elf";
    sysEmuOptions.sp_BL2_Elf_मार्गः =
        et_मार्गः + "lib/esperanto-fw/ServiceProcessorBL2/fast-boot/ServiceProcessorBL2_fast-boot.elf";
    sysEmuOptions.machine_Minion_Elf_मार्गः = et_मार्गः + "lib/esperanto-fw/MachineMinion/MachineMinion.elf";
    sysEmuOptions.master_Minion_Elf_मार्गः  = et_मार्गः + "lib/esperanto-fw/MasterMinion/MasterMinion.elf";
    sysEmuOptions.कार्यकर्ता_Minion_Elf_मार्गः  = et_मार्गः + "lib/esperanto-fw/WorkerMinion/WorkerMinion.elf";
    sysEmuOptions.executable_मार्गः       = et_मार्गः + "bin/sys_emu";

    // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
    ध्रुव std::vector<std::string> required_files = {
        sysEmuOptions.bootrom_Trampoline_गन्तव्यम्_BL2_Elf_मार्गः, sysEmuOptions.sp_BL2_Elf_मार्गः,
        sysEmuOptions.machine_Minion_Elf_मार्गः,          sysEmuOptions.master_Minion_Elf_मार्गः,
        sysEmuOptions.कार्यकर्ता_Minion_Elf_मार्गः,           sysEmuOptions.executable_मार्गः,
    };

    प्रति (ध्रुव स्वचालित & file : required_files) {
        यदि (!fs::exists(file) || fs::file_size(file) == 0) {
            // Check that each path has a valid existing non-zero file otherwise emulator just silently hangs
            GGML_LOG_ERROR("ET: Unable to find required sysemu file: %s", file.c_str());
            GGML_LOG_ERROR("ET: Confirm et-platform is correctly installed at configured path.");
            abort();
        }
    }

    sysEmuOptions.सततम्_Dir           = (fs::current_path().string() + "/");
    sysEmuOptions.max_आवृत्तयः        = k_Sys_Emu_Max_आवृत्तयः;
    sysEmuOptions.minion_Shires_आवरणम् = k_Sys_Emu_Minion_Shires_आवरणम्;
    sysEmuOptions.pu_Uart0_मार्गः      = sysEmuOptions.सततम्_Dir + "pu_uart0_tx.log";
    sysEmuOptions.pu_Uart1_मार्गः      = sysEmuOptions.सततम्_Dir + "pu_uart1_tx.log";
    sysEmuOptions.sp_Uart0_मार्गः      = sysEmuOptions.सततम्_Dir + "spio_uart0_tx.log";
    sysEmuOptions.sp_Uart1_मार्गः      = sysEmuOptions.सततम्_Dir + "spio_uart1_tx.log";
    sysEmuOptions.startGdb         = असत्य;
    sysEmuOptions.memcheck         = असत्य;

    प्रत्यागम sysEmuOptions;
}

// Forward declaration
स्थिर शून्य ggml_et_driver_cleanup();

स्थिर तार्किक ggml_et_driver_init() {
    यदि (_drv.runtime != शून्यसूचकः) {
        प्रतिज्ञा(_drv.device_layer != शून्यसूचकः);
    } अन्यथा {
        यत्न {
#यदि defined GGML_ET_SYSEMU && GGML_ET_SYSEMU
            // For emulator device using sysEmuOptions provided by function above enabled compiling with `-DGGML_ET_SYSEMU=ON`
            _drv.device_layer = dev::IDeviceLayer::createSysEmuDeviceLayer(ggml_et_get_default_sysemu_options());
#अन्यथा
            // For physical PCIe device
            _drv.device_layer = dev::IDeviceLayer::createPcieDeviceLayer();
#यदि_अन्तः  // GGML_ET_SYSEMU

            _drv.runtime = rt::IRuntime::create(_drv.device_layer);

            // Initialize profiler if requested via environment variable
            ध्रुव वर्ण * profile_मार्गः = getenv("GGML_ET_PROFILE");
            यदि (profile_मार्गः) {
                std::string output_मार्गः    = std::string(profile_मार्गः) + "/et_runtime_trace.json";
                std::string kernel_id_मार्गः = std::string(profile_मार्गः) + "/kernel_id.json";

                _drv.profile_प्रवाहः   = std::make_unique<std::ofstream>(output_मार्गः);
                _drv.kernel_id_प्रवाहः = std::make_unique<std::ofstream>(kernel_id_मार्गः);
                यदि (!_drv.profile_प्रवाहः->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling output file: %s", output_मार्गः.c_str());
                    abort();
                }
                यदि (!_drv.kernel_id_प्रवाहः->is_open()) {
                    GGML_LOG_ERROR("ET: Failed to open profiling kernel map: %s", kernel_id_मार्गः.c_str());
                    abort();
                }

                स्वचालित * profiler = _drv.runtime->getProfiler();
                profiler->start(*_drv.profile_प्रवाहः, rt::IProfiler::OutputType::Json);
                _drv.profiling_enabled = सत्य;
                GGML_LOG_INFO("ET: Runtime profiler started (JSON format)");

                // Register cleanup at program exit
                std::atexit(ggml_et_driver_cleanup);
            }
        } गृहाण (ध्रुव std::exception & e) {
            GGML_LOG_ERROR("ggml_et: %s", e.what());
            यदि (_drv.device_layer != शून्यसूचकः) {
                _drv.device_layer.reset();
            }
            यदि (_drv.runtime != शून्यसूचकः) {
                _drv.runtime.reset();
            }
            प्रत्यागम असत्य;
        }
    }
    प्रत्यागम सत्य;
}

स्थिर std::shared_ptr<dev::IDeviceLayer> ggml_et_devicelayer() {
    प्रत्यागम _drv.device_layer;
}

std::shared_ptr<rt::IRuntime> ggml_et_runtime() {
    प्रत्यागम _drv.runtime;
}

स्थिर शून्य ggml_et_driver_cleanup() {
    यदि (_drv.profiling_enabled && _drv.runtime) {
        GGML_LOG_INFO("ET: Stopping runtime profiler");
        स्वचालित * profiler = _drv.runtime->getProfiler();
        profiler->stop();
        _drv.profiling_enabled = असत्य;

        यदि (_drv.profile_प्रवाहः) {
            _drv.profile_प्रवाहः->close();
            _drv.profile_प्रवाहः.reset();
        }

        // Save kernel map
        यदि (_drv.kernel_id_प्रवाहः && !_drv.kernel_map.रिक्तानि()) {
            स्वचालित & os = *_drv.kernel_id_प्रवाहः;
            // XXX: Manual JSON construction. Not pretty but removes dependency
            os << "{\n";
            प्रति (size_t i = 0; i < _drv.kernel_map.size(); i++) {
                os << "  \"" << _drv.kernel_map[i].first << "\": " << (पूर्णाङ्क) _drv.kernel_map[i].second;
                यदि (i + 1 < _drv.kernel_map.size()) {
                    os << ",";
                }
                os << "\n";
            }
            os << "}\n";
            _drv.kernel_id_प्रवाहः->close();
            _drv.kernel_id_प्रवाहः.reset();
        }
    }
}

स्थिर ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx);

स्थिर शून्य ggml_backend_et_बफर_free_बफर(ggml_backend_बफर_t बफर) {
    ggml_backend_et_बफर_context * ctx = (ggml_backend_et_बफर_context *) बफर->context;
    यदि (ctx->दत्त != शून्यसूचकः) {
        std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
        यदि (runtime) {
            runtime->freeDevice(ctx->rtid, static_cast<std::byte *>(ctx->दत्त));
        }
    }
    विलोपय ctx;
}

स्थिर शून्य * ggml_backend_et_बफर_get_आधारः(ggml_backend_बफर_t बफर) {
    ggml_backend_et_बफर_context * ctx = (ggml_backend_et_बफर_context *) बफर->context;
    प्रत्यागम ctx->दत्त;
}

स्थिर ggml_स्थितिः ggml_backend_et_बफर_init_tensor(ggml_backend_बफर_t बफर, ggml_tensor * tensor) {
    // View tensors share buffer with their view_src, no additional initialization needed
    यदि (tensor->view_src != रिक्त) {
        प्रत्यागम GGML_स्थितिः_SUCCESS;
    }

    ध्रुव size_t original_आकार = ggml_nbytes(tensor);
    ध्रुव size_t padded_आकार   = ggml_backend_buft_get_alloc_आकार(बफर->buft, tensor);

    // Clear padding bytes to avoid NaN values
    // XXX: Martin - do we need this?
    यदि (padded_आकार > original_आकार) {
        ध्रुव size_t padding_आकार = padded_आकार - original_आकार;

        // Get device context to access memops kernel
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
        यदि (!dev_ctx) {
            GGML_LOG_ERROR("ET: Failed to get device context for padding clear");
            प्रत्यागम GGML_स्थितिः_FAILED;
        }

        // Use device-side memset kernel for efficient padding clear
        std::byte * padding_ptr = static_cast<std::byte *>(tensor->दत्त) + original_आकार;
        यदि (!ggml_et_memset(dev_ctx, padding_ptr, 0, padding_आकार)) {
            GGML_LOG_ERROR("ET: Failed to clear padding using memset kernel for tensor %s", tensor->name);
            प्रत्यागम GGML_स्थितिः_FAILED;
        }
    }

    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

स्थिर शून्य ggml_backend_et_बफर_समुच्चयः_tensor(ggml_backend_बफर_t बफर,
                                              ggml_tensor *         tensor,
                                              ध्रुव शून्य *          दत्त,
                                              size_t                विस्थापनम्,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    // Create short-lived stream for this transfer
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
    rt::प्रवाहः_Id                     stream  = dev_ctx->default_प्रवाहः;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->दत्त) + विस्थापनम्;
    ध्रुव std::byte * src_ptr = static_cast<ध्रुव std::byte *>(दत्त);

    rt::EventId event = runtime->memcpy_Host_गन्तव्यम्_Device(stream, src_ptr, dst_ptr, size, सत्य /*barrier*/);

    runtime->waitForEvent(event);
}

स्थिर शून्य ggml_backend_et_बफर_get_tensor(ggml_backend_बफर_t बफर,
                                              ध्रुव ggml_tensor *   tensor,
                                              शून्य *                दत्त,
                                              size_t                विस्थापनम्,
                                              size_t                size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
    rt::प्रवाहः_Id                     stream  = dev_ctx->default_प्रवाहः;

    ध्रुव std::byte * src_ptr = static_cast<ध्रुव std::byte *>(tensor->दत्त) + विस्थापनम्;
    std::byte *       dst_ptr = static_cast<std::byte *>(दत्त);

    rt::EventId event = runtime->memcpy_Device_गन्तव्यम्_Host(stream, src_ptr, dst_ptr, size, सत्य /*barrier*/);

    runtime->waitForEvent(event);
}

स्थिर तार्किक ggml_backend_et_बफर_cpy_tensor(ggml_backend_बफर_t बफर,
                                              ध्रुव ggml_tensor *   src,
                                              ggml_tensor *         dst) {
    GGML_अप्रयुक्तम्(बफर);
    GGML_अप्रयुक्तम्(src);
    GGML_अप्रयुक्तम्(dst);
    प्रत्यागम असत्य;
}

स्थिर शून्य ggml_backend_et_बफर_clear(ggml_backend_बफर_t बफर, uint8_t value) {
    ggml_backend_et_बफर_context * ctx = (ggml_backend_et_बफर_context *) बफर->context;

    यदि (ctx->size == 0 || ctx->दत्त == शून्यसूचकः) {
        प्रत्यागम;
    }

    // Get device context to access memops kernel
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) बफर->buft->device->context;
    यदि (!dev_ctx) {
        GGML_LOG_ERROR("ET: Failed to get device context for buffer clear");
        प्रत्यागम;
    }

    // Use device-side memset kernel for efficient clearing
    यदि (!ggml_et_memset(dev_ctx, ctx->दत्त, value, ctx->size)) {
        GGML_LOG_ERROR("ET: buffer_clear failed using memset kernel");
        प्रत्यागम;
    }

    GGML_LOG_DEBUG("ET: Buffer cleared successfully using memops kernel");
}

स्थिर ध्रुव संरचना ggml_backend_बफर_i ggml_backend_et_बफर_i = {
    /* .free_buffer     = */ ggml_backend_et_बफर_free_बफर,
    /* .get_base        = */ ggml_backend_et_बफर_get_आधारः,
    /* .init_tensor     = */ ggml_backend_et_बफर_init_tensor,
    /* .memset_tensor   = */ रिक्त,
    /* .set_tensor      = */ ggml_backend_et_बफर_समुच्चयः_tensor,
    /* .get_tensor      = */ ggml_backend_et_बफर_get_tensor,
    /* .set_tensor_2d   = */ रिक्त,
    /* .get_tensor_2d   = */ रिक्त,
    /* .cpy_tensor      = */ ggml_backend_et_बफर_cpy_tensor,
    /* .clear           = */ ggml_backend_et_बफर_clear,
    /* .reset           = */ रिक्त,
};

स्थिर ध्रुव वर्ण * ggml_backend_et_बफर_type_get_name(ggml_backend_बफर_type_t buft) {
    GGML_अप्रयुक्तम्(buft);
    प्रत्यागम GGML_ET_NAME;
}

स्थिर ggml_backend_बफर_t ggml_backend_et_बफर_type_alloc_बफर(ggml_backend_बफर_type_t buft, size_t size) {
    ggml_backend_et_बफर_type_context * btctx = (ggml_backend_et_बफर_type_context *) buft->context;

    ggml_backend_et_बफर_context * ctx = नूतन ggml_backend_et_बफर_context;
    ctx->devidx                          = btctx->devidx;
    ctx->size                            = size;

    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        विलोपय ctx;
        प्रत्यागम शून्यसूचकः;
    }

    std::vector<rt::DeviceId> rtids = runtime->getDevices();
    यदि (static_cast<size_t>(btctx->devidx) >= rtids.size()) {
        विलोपय ctx;
        प्रत्यागम शून्यसूचकः;
    }
    ctx->rtid = rtids[btctx->devidx];

    ctx->दत्त = runtime->mallocDevice(ctx->rtid, size);
    यदि (ctx->दत्त == शून्यसूचकः) {
        विलोपय ctx;
        प्रत्यागम शून्यसूचकः;
    }

    प्रत्यागम ggml_backend_बफर_init(buft, ggml_backend_et_बफर_i, ctx, size);
}

स्थिर size_t ggml_backend_et_बफर_type_get_alignment(ggml_backend_बफर_type_t buft) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime || !buft->device) {
        प्रत्यागम GGML_MEM_ALIGN;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
    rt::DeviceProperties             prop    = runtime->getDeviceProperties(dev_ctx->rtid);
    प्रत्यागम prop.cache_पङ्क्तिः_आकार;
}

स्थिर size_t ggml_backend_et_बफर_type_get_max_आकार(ggml_backend_बफर_type_t buft) {
    यदि (buft->device) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) buft->device->context;
        प्रत्यागम dev_ctx->योग_mem;
    }
    प्रत्यागम SIZE_MAX;
}

स्थिर size_t ggml_backend_et_बफर_type_get_alloc_आकार(ggml_backend_बफर_type_t buft, ध्रुव ggml_tensor * tensor) {
    GGML_अप्रयुक्तम्(buft);
    प्रत्यागम ggml_nbytes_pad(tensor);
}

स्थिर तार्किक ggml_backend_et_बफर_type_is_host(ggml_backend_बफर_type_t buft) {
    GGML_अप्रयुक्तम्(buft);
    प्रत्यागम असत्य;
}

स्थिर ध्रुव संरचना ggml_backend_बफर_type_i ggml_backend_et_बफर_type_i = {
    /* .get_name            = */ ggml_backend_et_बफर_type_get_name,
    /* .alloc_buffer        = */ ggml_backend_et_बफर_type_alloc_बफर,
    /* .alloc_buffer_n      = */ रिक्त,
    /* .get_alignment       = */ ggml_backend_et_बफर_type_get_alignment,
    /* .get_max_size        = */ ggml_backend_et_बफर_type_get_max_आकार,
    /* .get_alloc_size      = */ ggml_backend_et_बफर_type_get_alloc_आकार,
    /* .get_alloc_size_n    = */ रिक्त,
    /* .is_host             = */ ggml_backend_et_बफर_type_is_host,
};

स्थिर ध्रुव वर्ण * ggml_backend_et_get_name(ggml_backend_t backend) {
    GGML_अप्रयुक्तम्(backend);
    प्रत्यागम GGML_ET_NAME;
}

स्थिर शून्य ggml_backend_et_free(ggml_backend_t backend) {
    ggml_backend_et_context *     et_ctx  = (ggml_backend_et_context *) backend->context;
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();

    // Clean up kernels on this device before freeing backend
    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), et_ctx->devidx);
    यदि (dev && dev->context) {
        ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;

        यदि (_drv.profiling_enabled) {
            स्वचालित kernels = ggml_et_get_loaded_kernels(dev_ctx);
            _drv.kernel_map.insert(_drv.kernel_map.end(), kernels.begin(), kernels.end());
        }

        ggml_et_unload_all_kernels(dev_ctx);

        यदि (runtime) {
            यदि (dev_ctx->trace_बफर) {
                runtime->freeDevice(dev_ctx->rtid, dev_ctx->trace_बफर);
                dev_ctx->trace_बफर = शून्यसूचकः;
            }
            // Drain any in-flight uberkernel launches before freeing the
            // device buffers they read from.
            runtime->wait_For_प्रवाहः(dev_ctx->default_प्रवाहः);
            प्रति (स्वचालित & slot : dev_ctx->uberkernel.slots) {
                यदि (slot.device_insts) {
                    runtime->freeDevice(dev_ctx->rtid, slot.device_insts);
                    slot.device_insts = शून्यसूचकः;
                }
                यदि (slot.device_params) {
                    runtime->freeDevice(dev_ctx->rtid, slot.device_params);
                    slot.device_params = शून्यसूचकः;
                }
                slot.has_pending = असत्य;
            }
        }
    }

    विलोपय et_ctx;
    विलोपय backend;
}

स्थिर ggml_backend_बफर_type_t ggml_backend_et_get_default_बफर_type(ggml_backend_t backend) {
    ggml_backend_et_context * et_ctx = (ggml_backend_et_context *) backend->context;

    प्रत्यागम ggml_backend_et_बफर_type(et_ctx->devidx);
}

स्थिर शून्य ggml_backend_et_समुच्चयः_tensor_async(ggml_backend_t backend,
                                             ggml_tensor *  tensor,
                                             ध्रुव शून्य *   दत्त,
                                             size_t         विस्थापनम्,
                                             size_t         size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::प्रवाहः_Id                     stream  = dev_ctx->default_प्रवाहः;

    std::byte *       dst_ptr = static_cast<std::byte *>(tensor->दत्त) + विस्थापनम्;
    ध्रुव std::byte * src_ptr = static_cast<ध्रुव std::byte *>(दत्त);

    runtime->memcpy_Host_गन्तव्यम्_Device(stream, src_ptr, dst_ptr, size, सत्य /*barrier*/);
}

स्थिर शून्य ggml_backend_et_get_tensor_async(ggml_backend_t      backend,
                                             ध्रुव ggml_tensor * tensor,
                                             शून्य *              दत्त,
                                             size_t              विस्थापनम्,
                                             size_t              size) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    rt::प्रवाहः_Id                     stream  = dev_ctx->default_प्रवाहः;

    ध्रुव std::byte * src_ptr = static_cast<ध्रुव std::byte *>(tensor->दत्त) + विस्थापनम्;
    std::byte *       dst_ptr = static_cast<std::byte *>(दत्त);

    runtime->memcpy_Device_गन्तव्यम्_Host(stream, src_ptr, dst_ptr, size, सत्य /*barrier*/);
}

स्थिर तार्किक ggml_backend_et_cpy_tensor_async(ggml_backend_t      backend_src,
                                             ggml_backend_t      backend_dst,
                                             ध्रुव ggml_tensor * src,
                                             ggml_tensor *       dst) {
    GGML_अप्रयुक्तम्(backend_src);
    GGML_अप्रयुक्तम्(backend_dst);
    GGML_अप्रयुक्तम्(src);
    GGML_अप्रयुक्तम्(dst);
    प्रत्यागम असत्य;
}

स्थिर शून्य ggml_backend_et_synchronize(ggml_backend_t backend) {
    std::shared_ptr<rt::IRuntime> runtime = ggml_et_runtime();
    यदि (!runtime) {
        प्रत्यागम;
    }

    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    runtime->wait_For_प्रवाहः(dev_ctx->default_प्रवाहः);

    स्वचालित errors = runtime->retrieve_प्रवाहः_Errors(dev_ctx->default_प्रवाहः);
    यदि (errors.रिक्तानि()) {
        प्रत्यागम;
    }
    प्रति (ध्रुव स्वचालित & err : errors) {
        GGML_LOG_ERROR("ET: stream error detected at synchronization point. Code: %d,Type: %d\n", (पूर्णाङ्क) err.error_संहिता,
                       (पूर्णाङ्क) err.errorContext_.value()[0].type_);
    }
    abort();
}

स्थिर तार्किक ggml_et_can_fuse(ध्रुव ggml_cgraph * cgraph, पूर्णाङ्क ग्रन्थिः_idx, std::initializer_list<ggml_op> ops) {
    यदि (!ggml_can_fuse(cgraph, ग्रन्थिः_idx, ops)) {
        प्रत्यागम असत्य;
    }

    यदि (ops.size() == 2 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_सङ्कलय) {
        ध्रुव ggml_tensor * mm  = cgraph->nodes[ग्रन्थिः_idx];
        ध्रुव ggml_tensor * सङ्कलय_2 = cgraph->nodes[ग्रन्थिः_idx + 1];

        // Only Q8_0 weights x F32 activations -> F32 (the kernel that has
        // the bias path).  Other MM variants must wait for their own kernel
        // bias support.
        यदि (mm->type != GGML_TYPE_F32 || mm->src[0]->type != GGML_TYPE_Q8_0 || mm->src[1]->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        // ADD must be F32 and one of its operands must be the MM output.
        यदि (सङ्कलय_2->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }
        यदि (सङ्कलय_2->src[0] != mm && सङ्कलय_2->src[1] != mm) {
            प्रत्यागम असत्य;
        }

        ध्रुव ggml_tensor * bias = (सङ्कलय_2->src[0] == mm) ? सङ्कलय_2->src[1] : सङ्कलय_2->src[0];

        यदि (bias->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        // No broadcasting: bias shape must equal MM output shape.
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
            यदि (bias->ne[i] != mm->ne[i]) {
                प्रत्यागम असत्य;
            }
        }

        // Bias and dst must be contiguous and have identical strides - the
        // kernel uses dst-style offset arithmetic against bias's nb[].
        यदि (!ggml_is_contiguous(bias) || !ggml_is_contiguous(mm)) {
            प्रत्यागम असत्य;
        }
        प्रति (पूर्णाङ्क i = 0; i < GGML_MAX_DIMS; ++i) {
            यदि ((int64_t) bias->nb[i] != (int64_t) सङ्कलय_2->nb[i]) {
                प्रत्यागम असत्य;
            }
        }
    }

    यदि (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        ध्रुव ggml_tensor * rms_norm = cgraph->nodes[ग्रन्थिः_idx];
        ध्रुव ggml_tensor * mul      = cgraph->nodes[ग्रन्थिः_idx + 1];

        // ET only supports F32
        यदि (rms_norm->src[0]->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        // Identify the weights tensor (the MUL operand that isn't rms_norm output)
        ध्रुव ggml_tensor * weights = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];

        यदि (weights->type != GGML_TYPE_F32) {
            प्रत्यागम असत्य;
        }

        // Both inputs must be contiguous (ET hardware requirement)
        यदि (!ggml_is_contiguous(rms_norm->src[0]) || !ggml_is_contiguous_पङ्क्तिकाः(weights)) {
            प्रत्यागम असत्य;
        }

        // ET requires cache-aligned rows (ne[0] % 16 == 0)
        यदि (rms_norm->src[0]->ne[0] % 16 != 0 || weights->ne[0] % 16 != 0) {
            प्रत्यागम असत्य;
        }

        // Fused kernel doesn't handle dim-0 broadcasting
        यदि (weights->ne[0] != rms_norm->src[0]->ne[0]) {
            प्रत्यागम असत्य;
        }
    }

    प्रत्यागम सत्य;
}

स्थिर ggml_स्थितिः ggml_backend_et_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) backend->device->context;
    ggml_et_uberkernel_begin_graph(&dev_ctx->uberkernel);

    प्रति (पूर्णाङ्क i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * ग्रन्थिः = cgraph->nodes[i];

        यदि (ग्रन्थिः->op == GGML_OP_NONE || ग्रन्थिः->op == GGML_OP_VIEW || ग्रन्थिः->op == GGML_OP_RESHAPE ||
            ग्रन्थिः->op == GGML_OP_PERMUTE || ग्रन्थिः->op == GGML_OP_TRANSPOSE) {
            अनुवर्त;
        }

        // --- Fusion checks (before regular dispatch) ---
        यदि (ggml_et_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL })) {
            ggml_et_op_rms_norm_mul(dev_ctx, ग्रन्थिः, cgraph->nodes[i + 1]);
            i++;  // skip the MUL node
            अनुवर्त;
        }
        यदि (ggml_et_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_सङ्कलय })) {
            ggml_et_op_mul_mat(dev_ctx, ग्रन्थिः, cgraph->nodes[i + 1]);
            i++;  // skip the ADD node
            अनुवर्त;
        }

        विकल्प (ग्रन्थिः->op) {
            पक्ष GGML_OP_SQR:
                ggml_et_op_sqr(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_UNARY:
                ggml_et_op_unary(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_योग_पङ्क्तिकाः:
                ggml_et_op_योग_पङ्क्तिकाः(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_माध्यम्:
                ggml_et_op_माध्यम्(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_CLAMP:
                ggml_et_op_clamp(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_MUL:
                ggml_et_op_mul(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_सङ्कलय:
                ggml_et_op_सङ्कलय(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_SUB:
                ggml_et_op_sub(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_CUMSUM:
                ggml_et_op_cumsum(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_MUL_MAT:
                ggml_et_op_mul_mat(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_MUL_MAT_ID:
                ggml_et_op_mul_mat_id(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_ROPE:
                ggml_et_op_rope(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_RMS_NORM:
                ggml_et_op_rms_norm(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_NORM:
                ggml_et_op_norm(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_L2_NORM:
                ggml_et_op_l2_norm(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_गणः_NORM:
                ggml_et_op_गणः_norm(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_SCALE:
                ggml_et_op_scale(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_GLU:
                ggml_et_op_glu(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_SOFT_MAX:
                ggml_et_op_softmax(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_IM2COL:
                ggml_et_op_im2col(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_CONV_2D:
                ggml_et_op_conv_2d(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_FLASH_ATTN_EXT:
                ggml_et_op_flash_attn_ext(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_GET_पङ्क्तिकाः:
                ggml_et_op_get_पङ्क्तिकाः(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_CONT:
                ggml_et_op_cont(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_CPY:
                ggml_et_op_cpy(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_CONCAT:
                ggml_et_op_concat(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_REPEAT:
                ggml_et_op_repeat(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_SSM_CONV:
                ggml_et_op_ssm_conv(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_SSM_अवलोकनम्:
                ggml_et_op_ssm_अवलोकनम्(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_PAD:
                ggml_et_op_pad(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
                ggml_et_op_समुच्चयः_पङ्क्तिकाः(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_FILL:
                ggml_et_op_fill(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_DIAG:
                ggml_et_op_diag(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_TRI:
                ggml_et_op_tri(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_SOLVE_TRI:
                ggml_et_op_solve_tri(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_समुच्चयः:
                ggml_et_op_समुच्चयः(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_RWKV_WKV6:
                ggml_et_op_rwkv_wkv6(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_RWKV_WKV7:
                ggml_et_op_rwkv_wkv7(dev_ctx, ग्रन्थिः);
                भङ्ग;

            पक्ष GGML_OP_GATED_DELTA_NET:
                ggml_et_op_gated_delta_net(dev_ctx, ग्रन्थिः);
                भङ्ग;

            स्वभाव:
                ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
                GGML_LOG_ERROR("ET: Unsupported operation in graph: %s", ggml_op_name(ग्रन्थिः->op));
                प्रत्यागम GGML_स्थितिः_FAILED;
        }

        यदि (ggml_et_uberkernel_failed(&dev_ctx->uberkernel)) {
            ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
            प्रत्यागम GGML_स्थितिः_FAILED;
        }
    }

    यदि (!ggml_et_uberkernel_अन्तः_graph(dev_ctx)) {
        ggml_et_uberkernel_abort_graph(&dev_ctx->uberkernel);
        प्रत्यागम GGML_स्थितिः_FAILED;
    }

    प्रत्यागम GGML_स्थितिः_SUCCESS;
}

// Check that elements within each row are contiguous (nb[0] == type_size).
// Higher-dim strides can be arbitrary - kernels navigate them via byte offsets.
स्थिर तार्किक et_ggml_is_row_contiguous(ध्रुव ggml_tensor * t) {
    प्रत्यागम t->nb[0] == ggml_type_आकार(t->type);
}

स्थिर तार्किक ggml_backend_et_device_supports_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    GGML_अप्रयुक्तम्(dev);

    तार्किक supported = असत्य;
    विकल्प (op->op) {
        पक्ष GGML_OP_CUMSUM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->nb[0] == परिमाण(भिन्न) && ggml_is_contiguous(op);
            भङ्ग;
        पक्ष GGML_OP_SQR:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_योग_पङ्क्तिकाः:
            // dst has ne[0]=1, src0 row length must be cache-aligned
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->src[0]->ne[0] % 16 == 0 && ggml_is_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_माध्यम्:
            // Kernel handles arbitrary ne00 (per-row alignment guard with
            // scalar tail), so no row-length divisibility constraint here.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_CLAMP:
            // Element-wise; kernel distributes by cache lines and handles a
            // scalar tail, so any contiguous F32 size is fine - including the
            // 1x1x1x1 scalar case.
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_UNARY:
            // Only require dim-0 contiguity (nb[0] == sizeof(float)). Higher
            // dims may be arbitrarily strided views; the kernel walks per-row
            // using all four nb[] values. See unary_f32.c entry_point.
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && op->nb[0] == परिमाण(भिन्न) && op->src[0]->nb[0] == परिमाण(भिन्न)) {
                विकल्प (ggml_get_unary_op(op)) {
                    पक्ष GGML_UNARY_OP_ABS:
                    पक्ष GGML_UNARY_OP_SGN:
                    पक्ष GGML_UNARY_OP_NEG:
                    पक्ष GGML_UNARY_OP_चरणः:
                    पक्ष GGML_UNARY_OP_TANH:
                    पक्ष GGML_UNARY_OP_ELU:
                    पक्ष GGML_UNARY_OP_RELU:
                    पक्ष GGML_UNARY_OP_SIGMOID:
                    पक्ष GGML_UNARY_OP_GELU:
                    पक्ष GGML_UNARY_OP_GELU_शीघ्र:
                    पक्ष GGML_UNARY_OP_SILU:
                    पक्ष GGML_UNARY_OP_HARDSWISH:
                    पक्ष GGML_UNARY_OP_HARDSIGMOID:
                    पक्ष GGML_UNARY_OP_EXP:
                    पक्ष GGML_UNARY_OP_EXPM1:
                    पक्ष GGML_UNARY_OP_SOFTPLUS:
                    पक्ष GGML_UNARY_OP_GELU_ERF:
                    पक्ष GGML_UNARY_OP_FLOOR:
                    पक्ष GGML_UNARY_OP_CEIL:
                    पक्ष GGML_UNARY_OP_आवर्तनम्:
                    पक्ष GGML_UNARY_OP_TRUNC:
                        supported = सत्य;
                        भङ्ग;
                    स्वभाव:
                        भङ्ग;
                }
            }
            भङ्ग;
        पक्ष GGML_OP_MUL:
        पक्ष GGML_OP_सङ्कलय:
        पक्ष GGML_OP_SUB:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->nb[0] == परिमाण(भिन्न) &&
                        op->src[0]->nb[0] == परिमाण(भिन्न) &&
                        (op->src[1]->nb[0] == परिमाण(भिन्न) || op->src[1]->ne[0] == 1) &&
                        op->nb[1] == op->ne[0] * परिमाण(भिन्न);
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT:
            // Support Q8_0 x F32 -> F32, F16 x F32 -> F32, F16 x F16 -> F32, and F32 x F32 -> F32 matrix multiplication
            // Stride requirements: first dimension must be contiguous for all tensors
            यदि (op->type == GGML_TYPE_F32 &&
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
                supported = सत्य;
            } अन्यथा यदि (op->type == GGML_TYPE_F32 && op->src[0] &&
                       (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) && op->src[1] &&
                       (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32)) {
                // Check first dimension contiguity requirements
                तार्किक src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                तार्किक src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                तार्किक dst_first_dim_contiguous  = (op->nb[0] == परिमाण(भिन्न));

                // Check destination stride ordering (only for dimensions with ne > 1)
                तार्किक dst_properly_ordered = सत्य;
                प्रति (पूर्णाङ्क d = 0; d < 3; d++) {
                    यदि (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = असत्य;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;
            } अन्यथा यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                तार्किक src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                तार्किक src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                तार्किक dst_first_dim_contiguous  = (op->nb[0] == परिमाण(भिन्न));

                तार्किक dst_properly_ordered = सत्य;
                प्रति (पूर्णाङ्क d = 0; d < 3; d++) {
                    यदि (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = असत्य;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;

            } अन्यथा यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_Q4_0 && op->src[1] &&
                       op->src[1]->type == GGML_TYPE_F32) {
                // Keep the existing quantized path constraints separate from the
                // relaxed non-quant generic fallback.
                तार्किक src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                तार्किक src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                तार्किक dst_first_dim_contiguous  = (op->nb[0] == परिमाण(भिन्न));

                तार्किक dst_properly_ordered = सत्य;
                प्रति (पूर्णाङ्क d = 0; d < 3; d++) {
                    यदि (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = असत्य;
                    }
                }

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && dst_first_dim_contiguous &&
                            dst_properly_ordered;
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_MUL_MAT_ID:
            // Support MUL_MAT_ID for Mixture of Experts: (Q8_0/Q4_0/F16/F32) x F32 -> F32 with I32 expert indices
            // src0 (as): [K, M, n_expert] - expert weight matrices (can be quantized)
            // src1 (b):  [K, n_expert_used, batch] - activations (F32)
            // src2 (ids): [n_expert_used, batch] - expert selection indices (I32)
            // dst: [M, n_expert_used, batch, 1] - output (F32)
            यदि (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_Q8_0 || op->src[0]->type == GGML_TYPE_Q4_0 ||
                 op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_I32) {
                // Check first dimension contiguity requirements (matching CPU backend)
                तार्किक src0_first_dim_contiguous = (op->src[0]->nb[0] == ggml_type_आकार(op->src[0]->type));
                तार्किक src1_first_dim_contiguous = (op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type));
                तार्किक src2_first_dim_contiguous = (op->src[2]->nb[0] == ggml_type_आकार(op->src[2]->type));
                तार्किक dst_first_dim_contiguous  = (op->nb[0] == परिमाण(भिन्न));

                // Check destination stride ordering (only for dimensions with ne > 1)
                तार्किक dst_properly_ordered = सत्य;
                प्रति (पूर्णाङ्क d = 0; d < 3; d++) {
                    यदि (op->ne[d] > 1 && op->ne[d + 1] > 1 && op->nb[d] > op->nb[d + 1]) {
                        dst_properly_ordered = असत्य;
                    }
                }

                // Validate tensor dimension constraints from GGML definition
                तार्किक dims_valid = (op->src[0]->ne[3] == 1) &&  // as is 3d (one matrix per expert)
                                  (op->src[1]->ne[3] == 1) &&  // b is 3d
                                  (op->src[2]->ne[2] == 1 && op->src[2]->ne[3] == 1) &&  // ids is 2d
                                  (op->src[2]->ne[1] == op->src[1]->ne[2]) &&    // must have expert list per b row
                                  (op->src[0]->ne[0] == op->src[1]->ne[0]) &&    // K dimension must match
                                  (op->src[2]->ne[0] % op->src[1]->ne[1] == 0);  // can broadcast

                supported = src0_first_dim_contiguous && src1_first_dim_contiguous && src2_first_dim_contiguous &&
                            dst_first_dim_contiguous && dst_properly_ordered && dims_valid;
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_ROPE:
            // Support F32 x I32 -> F32 RoPE for the modes implemented by rope_f32.
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0])) {
                ध्रुव पूर्णाङ्क  mode             = ggml_get_op_params_i32(op, 2);
                ध्रुव पूर्णाङ्क  ndims            = ggml_get_op_params_i32(op, 1);
                ध्रुव तार्किक is_प्रसामान्यम्        = mode == GGML_ROPE_TYPE_प्रसामान्यम्;
                ध्रुव तार्किक is_neox          = mode == GGML_ROPE_TYPE_NEOX;
                ध्रुव तार्किक is_imrope        = mode == GGML_ROPE_TYPE_IMROPE;
                ध्रुव तार्किक zero_view_विस्थापनम् = op->src[0]->view_src == शून्यसूचकः || op->src[0]->view_offs == 0;
                ध्रुव तार्किक has_sections = ggml_get_op_params_i32(op, 11) > 0 || ggml_get_op_params_i32(op, 12) > 0 ||
                                          ggml_get_op_params_i32(op, 13) > 0;
                // FIXME: support ggml_rope_set_offset
                ध्रुव तार्किक zero_rot_विस्थापनम्  = ggml_get_op_params_i32(op, 15) == 0;

                supported =
                    zero_view_विस्थापनम् && zero_rot_विस्थापनम् && ndims <= 512 &&
                    (is_प्रसामान्यम् || (is_neox && ndims % 16 == 0) || (is_imrope && ndims % 16 == 0 && has_sections));
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_RMS_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_L2_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_गणः_NORM:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && et_ggml_is_row_contiguous(op->src[0]) &&
                        ggml_get_op_params_i32(op, 0) > 0;
            भङ्ग;
        पक्ष GGML_OP_IM2COL:
            supported = op->src[0] && op->src[1] &&
                        ((op->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32) ||
                         (op->type == GGML_TYPE_F16 &&
                          (op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32))) &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[1]) &&
                        op->nb[0] == ggml_type_आकार(op->type) && op->src[1]->nb[0] == ggml_type_आकार(op->src[1]->type);
            भङ्ग;
        पक्ष GGML_OP_CONV_2D:
            {
                // First-cut conv_2d_f32_me kernel constraints. Anything outside
                // this falls back to CPU (it's a strict subset on purpose).
                यदि (!op->src[0] || !op->src[1]) {
                    supported = असत्य;
                    भङ्ग;
                }
                यदि (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 ||
                    op->src[1]->type != GGML_TYPE_F32) {
                    supported = असत्य;
                    भङ्ग;
                }
                यदि (!ggml_is_contiguous(op) || !ggml_is_contiguous(op->src[0]) || !ggml_is_contiguous(op->src[1])) {
                    supported = असत्य;
                    भङ्ग;
                }

                ध्रुव ggml_tensor * flt = op->src[0];  // [Kw, Kh, Cin, Cout]
                ध्रुव ggml_tensor * अन्तर्  = op->src[1];  // [W,  H,  Cin, N]
                ध्रुव int32_t       s0  = ggml_get_op_params_i32(op, 0);
                ध्रुव int32_t       s1  = ggml_get_op_params_i32(op, 1);
                ध्रुव int32_t       p0  = ggml_get_op_params_i32(op, 2);
                ध्रुव int32_t       p1  = ggml_get_op_params_i32(op, 3);
                ध्रुव int32_t       d0  = ggml_get_op_params_i32(op, 4);
                ध्रुव int32_t       d1  = ggml_get_op_params_i32(op, 5);

                ध्रुव int64_t Kw   = flt->ne[0];
                ध्रुव int64_t Kh   = flt->ne[1];
                ध्रुव int64_t Cin  = flt->ne[2];
                ध्रुव int64_t Cout = flt->ne[3];
                ध्रुव int64_t H    = अन्तर्->ne[1];
                (शून्य) अन्तर्->ne[0];

                यदि (s0 < 1 || s1 < 1 || !(d0 == 1 && d1 == 1) || Cin % 16 != 0 || Cout % 16 != 0 || अन्तर्->ne[3] != 1) {
                    supported = असत्य;
                    भङ्ग;
                }
                ध्रुव int64_t OW = op->ne[0];
                ध्रुव int64_t OH = op->ne[1];
                यदि (OW <= 0 || OH <= 0) {
                    supported = असत्य;
                    भङ्ग;
                }
                (शून्य) p0;
                (शून्य) p1;

                // Mirror the kernel's sizing:
                //   if K_TILES * per_KT_bytes <= budget: 1 buffer, n_chunks=1
                //   else: 2 buffers (double-buffer), shrink chunk_KT until
                //         2*chunk_KT*per_KT_bytes <= budget.
                ध्रुव int64_t Hp            = H + 2 * p1;
                ध्रुव int64_t OW_pad        = (OW + 15) & ~15;
                ध्रुव int64_t Wp_a          = OW_pad;
                ध्रुव तार्किक    need_stage    = (OW % 16 != 0);
                ध्रुव int64_t stage_अष्टकानि   = need_stage ? (Cout * OH * OW_pad * 4) : 0;
                ध्रुव int64_t L2SCP_BUDGET  = 1500 * 1024;
                // Per-hart partial-TenC scratch (mirrors kernel MAX_TILES_PER_HART=2):
                // 32 minions x 2 tiles x 1024 bytes = 64 KB per shire.
                ध्रुव int64_t scratch_अष्टकानि = 32 * 2 * 16 * 16 * 4;
                ध्रुव int64_t budget        = L2SCP_BUDGET - stage_अष्टकानि - scratch_अष्टकानि;
                ध्रुव int64_t प्रत्येकम्_KT_अष्टकानि  = Kh * Kw * Cout * 16 * 4 + Kw * 16 * Hp * Wp_a * 4;
                ध्रुव int64_t K_TILES       = Cin / 16;

                int64_t chunk_KT_calc;
                int64_t n_chunks_calc;
                यदि (K_TILES * प्रत्येकम्_KT_अष्टकानि <= budget) {
                    chunk_KT_calc = K_TILES;
                    n_chunks_calc = 1;
                } अन्यथा {
                    chunk_KT_calc = K_TILES;
                    यावत् (chunk_KT_calc > 1 && 2 * chunk_KT_calc * प्रत्येकम्_KT_अष्टकानि > budget) {
                        chunk_KT_calc--;
                    }
                    यावत् (chunk_KT_calc > 1 && K_TILES % chunk_KT_calc != 0) {
                        chunk_KT_calc--;
                    }
                    यदि (chunk_KT_calc < 1) {
                        supported = असत्य;
                        भङ्ग;
                    }
                    n_chunks_calc = K_TILES / chunk_KT_calc;
                }

                यदि (n_chunks_calc > 1) {
                    ध्रुव int64_t M_TILES     = Cout / 16;
                    ध्रुव int64_t w_tiles     = (OW + 15) / 16;
                    ध्रुव int64_t योग_tiles = OH * w_tiles * M_TILES;
                    // MAX_TILES_PER_HART = 2 (mirrors kernel constant).
                    ध्रुव int64_t max_workers = (need_stage ? 32 : 1024) * 2;
                    यदि (योग_tiles > max_workers) {
                        supported = असत्य;
                        भङ्ग;
                    }
                }

                supported = सत्य;
                भङ्ग;
            }
        पक्ष GGML_OP_SCALE:
            // F32 contiguous, total elements must be cache line aligned (16 floats)
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && (ggml_nelements(op) % 16 == 0);
            भङ्ग;
        पक्ष GGML_OP_GLU:
            // Note: we only require row-wise contiguity (ggml_is_contiguous_1) so that
            // strided views over a packed up_proj tensor (the common split-GLU layout)
            // are accepted. The kernel walks rows via nb[1] strides, so the inner
            // dimension just needs to be densely packed.
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_nelements(op) % 16 == 0 && ggml_is_contiguous_1(op) && ggml_is_contiguous_1(op->src[0])) {
                // Check GLU variant - support SWIGLU, SWIGLU_OAI, GEGLU, GEGLU_ERF, GEGLU_QUICK, REGLU
                ggml_glu_op glu_type          = ggml_get_glu_op(op);
                ध्रुव तार्किक  supported_variant = glu_type == GGML_GLU_OP_SWIGLU || glu_type == GGML_GLU_OP_SWIGLU_OAI ||
                                                glu_type == GGML_GLU_OP_SWIGLU_CLAMP || glu_type == GGML_GLU_OP_GEGLU ||
                                                glu_type == GGML_GLU_OP_GEGLU_ERF ||
                                                glu_type == GGML_GLU_OP_GEGLU_शीघ्र || glu_type == GGML_GLU_OP_REGLU;

                यदि (op->src[1]) {
                    supported = supported_variant && op->src[1]->type == GGML_TYPE_F32 &&
                                ggml_is_contiguous_1(op->src[1]) && op->src[0]->ne[0] == op->ne[0] &&
                                op->src[1]->ne[0] == op->ne[0];
                } अन्यथा {
                    supported = supported_variant && op->src[0]->ne[0] == 2 * op->ne[0];
                }
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_SOFT_MAX:
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->src[0]->ne[0] > 1) {
                // Check optional mask tensor (F32 only)
                यदि (op->src[1]) {
                    supported = op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[1]);
                    यदि (!supported) {
                        भङ्ग;
                    }
                }
                // Check optional sinks tensor (F32 only)
                यदि (op->src[2]) {
                    supported = op->src[2]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[2]);
                } अन्यथा {
                    supported = सत्य;
                }
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_SSM_अवलोकनम्:
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->src[0] &&
                        op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0]) && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[2]) && op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op->src[3]) && op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&
                        op->src[5] && op->src[5]->type == GGML_TYPE_F32 && op->src[6] &&
                        op->src[6]->type == GGML_TYPE_I32 && ggml_is_contiguous(op->src[6]) &&
                        op->src[1]->nb[0] == परिमाण(भिन्न) && op->src[4]->nb[0] == परिमाण(भिन्न) &&
                        op->src[5]->nb[0] == परिमाण(भिन्न) &&
                        op->src[1]->nb[1] == (size_t) op->src[1]->ne[0] * परिमाण(भिन्न) &&
                        op->src[4]->nb[1] == (size_t) op->src[4]->ne[0] * परिमाण(भिन्न) &&
                        op->src[5]->nb[1] == (size_t) op->src[5]->ne[0] * परिमाण(भिन्न) &&
                        op->src[0]->ne[0] == op->src[4]->ne[0] && op->src[0]->ne[1] == op->src[1]->ne[0] &&
                        op->src[0]->ne[2] == op->src[1]->ne[1] && op->src[1]->ne[2] == op->src[2]->ne[1] &&
                        op->src[1]->ne[3] == op->src[2]->ne[2] && op->src[4]->ne[2] == op->src[1]->ne[2] &&
                        op->src[4]->ne[3] == op->src[1]->ne[3] && ggml_are_same_shape(op->src[4], op->src[5]) &&
                        op->src[6]->ne[0] == op->src[1]->ne[3] && op->src[3]->ne[1] == op->src[1]->ne[1] &&
                        (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[0]->ne[0]) &&
                        (op->src[1]->ne[1] % op->src[4]->ne[1] == 0);
            भङ्ग;
        पक्ष GGML_OP_FLASH_ATTN_EXT:
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) && op->src[2] &&
                (op->src[2]->type == GGML_TYPE_F32 || op->src[2]->type == GGML_TYPE_F16) && op->src[4] == शून्यसूचकः &&
                ggml_is_contiguous_पङ्क्तिकाः(op) && ggml_is_contiguous_पङ्क्तिकाः(op->src[0])) {
                भिन्न max_bias      = 0.0f;
                भिन्न logit_softcap = 0.0f;
                स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) op->op_params + 1, परिमाण(max_bias));
                स्मृतिं_प्रतिलिख(&logit_softcap, (ध्रुव भिन्न *) op->op_params + 2, परिमाण(logit_softcap));

                ध्रुव ggml_prec prec = ggml_flash_attn_ext_get_prec(op);

                // Mask must be F16 or F32 if present
                तार्किक आवरणम्_ok = (op->src[3] == शून्यसूचकः) || (op->src[3]->type == GGML_TYPE_F32) ||
                               (op->src[3]->type == GGML_TYPE_F16);

                // GQA: n_head_q must be a multiple of n_head_kv
                ध्रुव int64_t nhq = op->src[0]->ne[2];
                ध्रुव int64_t nhk = op->src[1]->ne[2];

                // K/V row stride must match element size
                ध्रुव size_t k_elem = op->src[1]->type == GGML_TYPE_F16 ? 2 : 4;
                ध्रुव size_t v_elem = op->src[2]->type == GGML_TYPE_F16 ? 2 : 4;

                // Only support matrix engine path (F16 K/V, dk%32==0);
                // mask scalar F32 fallback to get baseline perf readings
                ध्रुव तार्किक me_eligible = op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16 &&
                                         (op->src[0]->ne[0] % 32) == 0;

                supported = me_eligible && आवरणम्_ok && (prec == GGML_PREC_F32 || prec == GGML_PREC_DEFAULT) &&
                            max_bias == 0.0f && logit_softcap == 0.0f && op->src[0]->nb[0] == परिमाण(भिन्न) &&
                            op->src[1]->nb[0] == k_elem && op->src[2]->nb[0] == v_elem && op->nb[0] == परिमाण(भिन्न) &&
                            op->src[0]->ne[0] == op->src[1]->ne[0] &&  // dk matches
                            op->src[2]->ne[0] == op->ne[0] &&          // dv matches
                            op->src[2]->ne[0] <= 512 &&                // dv limit
                            op->src[0]->ne[0] <= 512 &&                // dk limit
                            nhq % nhk == 0 &&                          // GQA ratio is integer
                            op->src[0]->ne[1] == op->ne[2] && op->src[0]->ne[2] == op->ne[1] &&
                            op->src[0]->ne[3] == op->ne[3] && op->src[1]->ne[1] == op->src[2]->ne[1] &&
                            op->src[1]->ne[2] == op->src[2]->ne[2] && op->src[1]->ne[3] == op->src[2]->ne[3] &&
                            op->src[0]->ne[3] == op->src[1]->ne[3];
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_GET_पङ्क्तिकाः:
            // Support F32/F16/Q4_0/Q8_0/Q4_K data with I32 indices -> F32 output
            यदि (op->type == GGML_TYPE_F32 && op->src[0] &&
                (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 ||
                 op->src[0]->type == GGML_TYPE_Q4_0 || op->src[0]->type == GGML_TYPE_Q8_0 ||
                 op->src[0]->type == GGML_TYPE_Q4_K) &&
                op->src[1] && op->src[1]->type == GGML_TYPE_I32 && ggml_is_contiguous(op) &&
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->src[0]->ne[2] == op->src[1]->ne[1]) && (op->src[1]->ne[3] == 1);
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_CONT:
            // Support F32->F32 and F16->F16 CONT operations (rearrange non-contiguous to contiguous)
            यदि ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0] &&
                op->src[0]->type == op->type && ggml_is_contiguous(op)) {
                // Defensive check: ensure dst and src0 are not aliased (separate buffers)
                // While GGML design currently guarantees this, check for future robustness
                यदि (op->दत्त && op->src[0]->दत्त && op->दत्त == op->src[0]->दत्त) {
                    GGML_LOG_WARN("ET: CONT operation detected aliased tensors (dst == src0), unsupported");
                    supported = असत्य;
                } अन्यथा {
                    supported = सत्य;
                }
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_CPY:
            // CPY copies src[0] data into dst layout (same as CONT for same-type)
            // Special path: zero-element tensors (scalars) are accepted as no-ops
            यदि (op->src[0]) {
                ध्रुव int64_t nelements = op->ne[0] * op->ne[1] * op->ne[2] * op->ne[3];
                यदि (nelements == 0) {
                    // Zero-element / scalar no-op case - always supported
                    supported = सत्य;
                } अन्यथा यदि ((op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && op->src[0]->type == op->type &&
                           ggml_is_contiguous(op)) {
                    // Same-type with contiguous dst - reuse CONT kernel
                    यदि (op->दत्त && op->src[0]->दत्त && op->दत्त == op->src[0]->दत्त) {
                        GGML_LOG_WARN("ET: CPY operation detected aliased tensors, unsupported");
                        supported = असत्य;
                    } अन्यथा {
                        supported = सत्य;
                    }
                } अन्यथा यदि (op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                    // F32 -> F16 conversion copy
                    supported = सत्य;
                } अन्यथा {
                    supported = असत्य;
                }
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_CONCAT:
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op)) {
                ध्रुव int32_t dim = ((ध्रुव int32_t *) op->op_params)[0];
                यदि (dim == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                    ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Fast dim==0 path: both source row segments are cacheline-aligned
                    // and contiguous, so the kernel can use vector row copies.
                    supported = सत्य;
                } अन्यथा यदि (dim == 0 && ((op->src[0]->nb[0] % परिमाण(भिन्न) == 0) || op->src[0]->ne[0] == 1) &&
                           ((op->src[1]->nb[0] % परिमाण(भिन्न) == 0) || op->src[1]->ne[0] == 1)) {
                    // Slow dim==0 path: scalar, stride-aware copies for non-contiguous
                    // or non-aligned source row segments. Destination remains contiguous.
                    supported = सत्य;
                } अन्यथा यदि (op->ne[0] % 16 == 0 && op->src[0]->ne[0] % 16 == 0 && op->src[1]->ne[0] % 16 == 0 &&
                           ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    // Dim >= 1 path: full aligned row copies from one source or the other.
                    supported = सत्य;
                }
            }
            भङ्ग;
        पक्ष GGML_OP_SSM_CONV:
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->nb[0] == परिमाण(भिन्न) &&
                        op->src[1]->nb[0] == परिमाण(भिन्न) && op->src[0]->nb[1] == op->src[0]->ne[0] * परिमाण(भिन्न) &&
                        op->src[1]->nb[1] == op->src[1]->ne[0] * परिमाण(भिन्न) && ggml_is_contiguous(op) &&
                        op->src[1]->ne[1] == op->src[0]->ne[1] && op->ne[0] == op->src[0]->ne[1] &&
                        op->ne[1] == op->src[0]->ne[0] - op->src[1]->ne[0] + 1 && op->ne[2] == op->src[0]->ne[2];
            भङ्ग;
        पक्ष GGML_OP_PAD:
            // F32 zero-pad only, no dim0 padding, dst contiguous
            // ne[0] must be CL-aligned (% 16 == 0) or evenly divide a CL (16 % ne[0] == 0)
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && (op->ne[0] % 16 == 0 || 16 % op->ne[0] == 0) &&
                op->src[0]->nb[0] == परिमाण(भिन्न)) {
                ध्रुव int32_t lp0      = ((ध्रुव int32_t *) op->op_params)[0];
                ध्रुव int32_t rp0      = ((ध्रुव int32_t *) op->op_params)[1];
                ध्रुव तार्किक    circular = (तार्किक) ((ध्रुव int32_t *) op->op_params)[8];
                यदि (lp0 == 0 && rp0 == 0 && !circular) {
                    supported = सत्य;
                } अन्यथा {
                    supported = असत्य;
                }
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_REPEAT:
            // Two acceptable shapes:
            //   1. No-op REPEAT (src and dst have identical shape): dispatched
            //      to cont_f32, which handles arbitrary contiguous sizes.
            //   2. Real REPEAT via repeat_f32 kernel: dst ne[0] cacheline-aligned,
            //      src0 ne[0] cacheline-aligned or 1, dst.ne[i] % src0.ne[i] == 0.
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op)) {
                supported = सत्य;
            } अन्यथा यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[0]->ne[0] == 1 || op->src[0]->ne[0] % 16 == 0) && op->ne[0] % 16 == 0 &&
                       ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && op->ne[0] % op->src[0]->ne[0] == 0 &&
                       op->ne[1] % op->src[0]->ne[1] == 0 && op->ne[2] % op->src[0]->ne[2] == 0 &&
                       op->ne[3] % op->src[0]->ne[3] == 0) {
                supported = सत्य;
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_FILL:
            // F32 contiguous, ne[0] cacheline-aligned for SIMD fill
            supported = op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && op->ne[0] % 16 == 0;
            भङ्ग;
        पक्ष GGML_OP_DIAG:
            // F32 contiguous dst, src0 is 1D vector [N,1,...], dst is [N,N,...]
            // ne[0] must be cacheline-aligned for SIMD zeroing
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        op->ne[0] % 16 == 0 && op->ne[0] == op->ne[1] && op->src[0]->ne[0] == op->ne[0] &&
                        op->src[0]->ne[1] == 1 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_TRI:
            // F32 contiguous, same shape in/out
            // Kernel handles arbitrary ne[0] with aligned fast path + scalar fallback
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]);
            भङ्ग;
        पक्ष GGML_OP_SOLVE_TRI:
            // F32 contiguous, A square, shapes compatible
            // Only lower-triangular left-side non-unit variant
            // Require k % 16 == 0 for cache-line-safe column parallelism
            supported = op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                        op->src[1]->type == GGML_TYPE_F32 && op->src[0]->ne[0] == op->src[0]->ne[1] &&
                        op->src[0]->ne[1] == op->src[1]->ne[1] && op->src[1]->ne[0] % 16 == 0 &&
                        ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः:
            // Minimal useful support: inplace F32 SET of a contiguous src1 view into
            // a contiguous dst/base tensor using explicit destination view strides.
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && ggml_is_contiguous(op) && ggml_is_contiguous(op->src[0]) &&
                ggml_is_contiguous(op->src[1]) && ggml_are_same_shape(op, op->src[0]) && op->src[1]->ne[0] % 16 == 0) {
                ध्रुव तार्किक   inplace = (तार्किक) ((ध्रुव int32_t *) op->op_params)[4];
                ध्रुव size_t nb1     = ((ध्रुव int32_t *) op->op_params)[0];
                ध्रुव size_t nb2     = ((ध्रुव int32_t *) op->op_params)[1];
                ध्रुव size_t nb3     = ((ध्रुव int32_t *) op->op_params)[2];
                ध्रुव size_t विस्थापनम्  = ((ध्रुव int32_t *) op->op_params)[3];
                ध्रुव size_t nb0     = ggml_element_आकार(op);
                ध्रुव size_t im0     = op->src[1]->ne[0] == 0 ? 0 : op->src[1]->ne[0] - 1;
                ध्रुव size_t im1     = op->src[1]->ne[1] == 0 ? 0 : op->src[1]->ne[1] - 1;
                ध्रुव size_t im2     = op->src[1]->ne[2] == 0 ? 0 : op->src[1]->ne[2] - 1;
                ध्रुव size_t im3     = op->src[1]->ne[3] == 0 ? 0 : op->src[1]->ne[3] - 1;

                ध्रुव तार्किक view_bounds_ok = विस्थापनम् + im0 * nb0 + im1 * nb1 + im2 * nb2 + im3 * nb3 <= ggml_nbytes(op);

                ध्रुव तार्किक cacheline_aligned =
                    (nb1 % 64 == 0) && (nb2 % 64 == 0) && (nb3 % 64 == 0) && (विस्थापनम् % 64 == 0);

                supported = inplace && view_bounds_ok && cacheline_aligned;
            }
            भङ्ग;
        पक्ष GGML_OP_RWKV_WKV6:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            // 6 sources: k, v, r, tf, td, state
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[0]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = सत्य;
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_RWKV_WKV7:
            // F32 contiguous, head_size must be multiple of 8 for vectorization
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] &&
                op->src[1]->type == GGML_TYPE_F32 && op->src[2] && op->src[2]->type == GGML_TYPE_F32 && op->src[3] &&
                op->src[3]->type == GGML_TYPE_F32 && op->src[4] && op->src[4]->type == GGML_TYPE_F32 && op->src[5] &&
                op->src[5]->type == GGML_TYPE_F32 && op->src[6] && op->src[6]->type == GGML_TYPE_F32 &&
                op->src[2]->ne[0] % 8 == 0 &&  // head_size multiple of 8
                ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && ggml_is_contiguous(op->src[2]) &&
                ggml_is_contiguous(op->src[3]) && ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5]) &&
                ggml_is_contiguous(op->src[6])) {
                supported = सत्य;
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_GATED_DELTA_NET:
            // F32, S_v must be multiple of 8 for vectorization
            // q, k, v may be row-contiguous with strided higher dimensions.
            // g, beta, state stay contiguous.
            यदि (op->type == GGML_TYPE_F32 && op->src[0] && op->src[0]->type == GGML_TYPE_F32 &&  // q
                op->src[1] && op->src[1]->type == GGML_TYPE_F32 &&                               // k
                op->src[2] && op->src[2]->type == GGML_TYPE_F32 &&                               // v
                op->src[3] && op->src[3]->type == GGML_TYPE_F32 &&                               // g
                op->src[4] && op->src[4]->type == GGML_TYPE_F32 &&                               // beta
                op->src[5] && op->src[5]->type == GGML_TYPE_F32 &&                               // state
                op->src[2]->ne[0] % 8 == 0 &&                                                    // S_v multiple of 8
                (op->src[3]->ne[0] == 1 || op->src[3]->ne[0] == op->src[2]->ne[0]) &&  // g is scalar or per-element
                op->src[4]->ne[0] == 1 &&                                              // beta is scalar per position
                et_ggml_is_row_contiguous(op->src[0]) && et_ggml_is_row_contiguous(op->src[1]) &&
                et_ggml_is_row_contiguous(op->src[2]) && ggml_is_contiguous(op->src[3]) &&
                ggml_is_contiguous(op->src[4]) && ggml_is_contiguous(op->src[5])) {
                supported = सत्य;
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_VIEW:
        पक्ष GGML_OP_PERMUTE:
        पक्ष GGML_OP_TRANSPOSE:
        पक्ष GGML_OP_RESHAPE:
            // Metadata-only no-ops, accept any type
            supported = सत्य;
            भङ्ग;
        पक्ष GGML_OP_समुच्चयः_पङ्क्तिकाः:
            // Support F32 data with I64 indices -> F16/F32 output (scatter operation)
            यदि (op->src[0] && op->src[0]->type == GGML_TYPE_F32 && op->src[1] && op->src[1]->type == GGML_TYPE_I64 &&
                (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16) && ggml_is_contiguous_पङ्क्तिकाः(op) &&
                ggml_is_contiguous_पङ्क्तिकाः(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                // Validate dimension constraints from ggml implementation
                supported = (op->ne[0] == op->src[0]->ne[0]) &&              // same number of columns
                            (op->ne[2] == op->src[0]->ne[2]) &&              // same batch size
                            (op->ne[3] == op->src[0]->ne[3]) &&              // same outer dimension
                            (op->src[0]->ne[1] == op->src[1]->ne[0]) &&      // src rows = index count
                            (op->src[0]->ne[2] % op->src[1]->ne[1] == 0) &&  // batch constraint
                            (op->src[0]->ne[3] % op->src[1]->ne[2] == 0) &&  // outer constraint
                            (op->src[1]->ne[3] == 1);                        // indices tensor constraint
            } अन्यथा {
                supported = असत्य;
            }
            भङ्ग;
        पक्ष GGML_OP_NONE:
            // Always support NONE operations - they represent leaf nodes (parameters, inputs, constants)
            // No computation needed, just memory management
            supported = सत्य;
            भङ्ग;
        स्वभाव:
            supported = असत्य;
            भङ्ग;
    }
    // if(!supported) {
    //     ggml_et_dump_operator_metadata(op);
    // }
    प्रत्यागम supported;
}

स्थिर तार्किक ggml_backend_et_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_बफर_type_t buft) {
    GGML_अप्रयुक्तम्(dev);
    प्रत्यागम buft->iface.get_name == ggml_backend_et_बफर_type_get_name;
}

स्थिर तार्किक ggml_backend_et_device_offload_op(ggml_backend_dev_t dev, ध्रुव ggml_tensor * op) {
    // GET_ROWS (embedding lookup) uses a large weight (tok_embd) that lives on CPU (dev_input).
    // The scheduler has no mechanism to cache cross-backend weight copies - it re-copies split
    // inputs every graph_compute call. For GET_ROWS this means copying the entire embedding table
    // (e.g. 266MB for Llama 3.1 1B) from host to device on every token, just to look up a few rows.
    // Keep GET_ROWS on CPU and let the scheduler copy only the small result to the device.
    // The other backends either only offload if the tensor lives on device or is large enough to
    // justify the copy cost.
    यदि (op->op == GGML_OP_GET_पङ्क्तिकाः) {
        प्रत्यागम असत्य;
    }
    प्रत्यागम सत्य;

    GGML_अप्रयुक्तम्(dev);
}

स्थिर ध्रुव संरचना ggml_backend_i ggml_backend_et_i = {
    /* .get_name                = */ ggml_backend_et_get_name,
    /* .free                    = */ ggml_backend_et_free,
    /* .set_tensor_async        = */ ggml_backend_et_समुच्चयः_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_et_get_tensor_async,
    /* .set_tensor_2d_async     = */ रिक्त,
    /* .get_tensor_2d_async     = */ रिक्त,
    /* .cpy_tensor_async        = */ रिक्त,
    /* .synchronize             = */ ggml_backend_et_synchronize,
    /* .graph_plan_create       = */ रिक्त,
    /* .graph_plan_free         = */ रिक्त,
    /* .graph_plan_update       = */ रिक्त,
    /* .graph_plan_compute      = */ रिक्त,
    /* .graph_compute           = */ ggml_backend_et_graph_compute,
    /* .event_record            = */ रिक्त,
    /* .event_wait              = */ रिक्त,
    /* .graph_optimize          = */ रिक्त,
};

स्थिर ध्रुव वर्ण * ggml_backend_et_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    प्रत्यागम dev_ctx->name.c_str();
}

स्थिर ध्रुव वर्ण * ggml_backend_et_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    प्रत्यागम dev_ctx->desc.c_str();
}

स्थिर शून्य ggml_backend_et_device_get_स्मृतिः(ggml_backend_dev_t dev, size_t * मुक्त, size_t * योग) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    // Currently getFreeMemory is not available on a runtime without server.
    // For now, report total memory as free.
    *मुक्त                                    = dev_ctx->योग_mem;
    *योग                                   = dev_ctx->योग_mem;
}

स्थिर गणना ggml_backend_dev_type ggml_backend_et_device_get_type(ggml_backend_dev_t dev) {
    GGML_अप्रयुक्तम्(dev);
    प्रत्यागम GGML_BACKEND_DEVICE_TYPE_GPU;
}

स्थिर शून्य ggml_backend_et_device_get_props(ggml_backend_dev_t dev, संरचना ggml_backend_dev_props * props) {
    GGML_अप्रयुक्तम्(dev);
    props->name        = ggml_backend_et_device_get_name(dev);
    props->description = ggml_backend_et_device_get_description(dev);
    props->type        = ggml_backend_et_device_get_type(dev);
    ggml_backend_et_device_get_स्मृतिः(dev, &props->स्मृतिः_free, &props->स्मृतिः_योग);
    props->device_id = रिक्त;  // No PCI device ID available
    props->caps      = {
        /* .async                 = */ सत्य,
        /* .host_buffer           = */ असत्य,
        /* .buffer_from_host_ptr  = */ असत्य,
        /* .events                = */ असत्य,
        /* .mmap_support          = */ सत्य,
    };
}

स्थिर ggml_backend_t ggml_backend_et_device_init_backend(ggml_backend_dev_t dev, ध्रुव वर्ण * params) {
    GGML_अप्रयुक्तम्(params);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    प्रत्यागम ggml_backend_et_init(dev_ctx->devidx);
}

स्थिर ggml_backend_बफर_type_t ggml_backend_et_device_get_बफर_type(ggml_backend_dev_t dev) {
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    प्रत्यागम dev_ctx->buftype;
}

स्थिर ggml_backend_बफर_type_t ggml_backend_et_device_get_host_बफर_type(ggml_backend_dev_t dev) {
    GGML_अप्रयुक्तम्(dev);
    प्रत्यागम ggml_backend_cpu_बफर_type();
}

स्थिर ध्रुव संरचना ggml_backend_device_i ggml_backend_et_device_i = {
    /* .get_name          = */ ggml_backend_et_device_get_name,
    /* .get_description   = */ ggml_backend_et_device_get_description,
    /* .get_memory        = */ ggml_backend_et_device_get_स्मृतिः,
    /* .get_type          = */ ggml_backend_et_device_get_type,
    /* .get_props         = */ ggml_backend_et_device_get_props,
    /* .init_backend      = */ ggml_backend_et_device_init_backend,
    /* .get_buffer_type   = */ ggml_backend_et_device_get_बफर_type,
    /* .get_host_buffer_type = */ ggml_backend_et_device_get_host_बफर_type,
    /* .buffer_from_host_ptr = */ रिक्त,
    /* .supports_op       = */ ggml_backend_et_device_supports_op,
    /* .supports_buft     = */ ggml_backend_et_device_supports_buft,
    /* .offload_op        = */ ggml_backend_et_device_offload_op,
    /* .event_new         = */ रिक्त,
    /* .event_free        = */ रिक्त,
    /* .event_synchronize = */ रिक्त,
};

/*
  Backend Registry.
*/

स्थिर ध्रुव वर्ण * ggml_backend_et_reg_get_name(ggml_backend_reg_t reg) {
    GGML_अप्रयुक्तम्(reg);
    प्रत्यागम GGML_ET_NAME;
}

स्थिर size_t ggml_backend_et_reg_get_device_गणनम्(ggml_backend_reg_t reg) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    प्रत्यागम ctx->devices.size();
}

स्थिर ggml_backend_dev_t ggml_backend_et_reg_get_device(ggml_backend_reg_t reg, size_t devidx) {
    ggml_backend_et_reg_ctx * ctx = (ggml_backend_et_reg_ctx *) reg->context;
    यदि (devidx >= ctx->devices.size()) {
        प्रत्यागम शून्यसूचकः;
    }
    प्रत्यागम ctx->devices[devidx];
}

स्थिर शून्य * ggml_backend_et_get_proc_स्थानाङ्कः(ggml_backend_reg_t reg, ध्रुव वर्ण * name) {
    GGML_अप्रयुक्तम्(reg);
    GGML_अप्रयुक्तम्(name);
    प्रत्यागम शून्यसूचकः;
}

स्थिर ध्रुव संरचना ggml_backend_reg_i ggml_backend_et_reg_i = {
    /* .get_name         = */ ggml_backend_et_reg_get_name,
    /* .get_device_count = */ ggml_backend_et_reg_get_device_गणनम्,
    /* .get_device       = */ ggml_backend_et_reg_get_device,
    /* .get_proc_address = */ ggml_backend_et_get_proc_स्थानाङ्कः,
};

ggml_backend_reg_t ggml_backend_et_reg(शून्य) {
    स्थिर ggml_backend_reg_t _reg = []() -> ggml_backend_reg_t {
        ggml_backend_et_reg_ctx * ctx = नूतन ggml_backend_et_reg_ctx;

        यदि (!ggml_et_driver_init()) {
            प्रत्यागम शून्यसूचकः;
        }

        ggml_backend_reg_t r = नूतन ggml_backend_reg{
            /* .api_version = */ GGML_BACKEND_API_संस्करणम्,
            /* .iface       = */ ggml_backend_et_reg_i,
            /* .context     = */ शून्यसूचकः,  // Set later
        };

        std::vector<rt::DeviceId> rtids = ggml_et_runtime()->getDevices();

        प्रति (पूर्णाङ्क i = 0; i < ggml_et_devicelayer()->get_Devices_गणनम्(); i++) {
            ggml_backend_dev_t dev = नूतन ggml_backend_device{
                /* .iface   = */ ggml_backend_et_device_i,
                /* .reg     = */ r,
                /* .context = */ शून्यसूचकः  // Set later
            };

            rt::DeviceId         rtid = rtids[i];
            rt::DeviceProperties prop = ggml_et_runtime()->getDeviceProperties(rtid);

            // Create device context.
            ggml_backend_et_device_context * dev_ctx = नूतन ggml_backend_et_device_context;
            dev_ctx->devidx                          = i;
            dev_ctx->rtid                            = rtid;
            dev_ctx->name                            = GGML_ET_NAME + std::to_string(i);
            dev_ctx->desc                            = "ET device " + std::to_string(i);
            dev_ctx->योग_mem                       = static_cast<size_t>(prop.स्मृतिः_आकार);
            {
                ध्रुव वर्ण * env            = getenv("GGML_ET_UBERKERNEL");
                dev_ctx->uberkernel_enabled = env && env[0] != '\0' && वाक्यं_तुलय(env, "0") != 0;
            }
            // Add buffer type for device to device context.
            ggml_backend_et_बफर_type_context * bufty_ctx = नूतन ggml_backend_et_बफर_type_context;
            bufty_ctx->devidx                               = i;
            bufty_ctx->name                                 = GGML_ET_NAME + std::to_string(i);
            dev_ctx->buftype = नूतन ggml_backend_बफर_type{ /* .iface   = */ ggml_backend_et_बफर_type_i,
                                                             /* .device  = */ dev,
                                                             /* .context = */ bufty_ctx };

            // Create default stream for ordered execution on this device
            dev_ctx->default_प्रवाहः = ggml_et_runtime()->create_प्रवाहः(rtid);

            dev_ctx->trace_बफर = ggml_et_runtime()->mallocDevice(rtid, ET_TRACE_बफर_आकार);
            // Pre-size each slot's host buffers and device-side scratch so the
            // first few graph_compute calls don't pay a malloc/grow penalty.
            प्रति (स्वचालित & slot : dev_ctx->uberkernel.slots) {
                slot.insts.reserve(256);
                slot.params_blob.reserve(1 << 20);
                slot.device_insts_क्षमता  = 256 * परिमाण(ggml_et_uberkernel_inst);
                slot.device_params_क्षमता = 1 << 20;
                slot.device_insts           = ggml_et_runtime()->mallocDevice(rtid, slot.device_insts_क्षमता);
                slot.device_params          = ggml_et_runtime()->mallocDevice(rtid, slot.device_params_क्षमता);
                यदि (slot.device_insts == शून्यसूचकः) {
                    slot.device_insts_क्षमता = 0;
                }
                यदि (slot.device_params == शून्यसूचकः) {
                    slot.device_params_क्षमता = 0;
                }
            }

            dev->context = dev_ctx;

            ctx->devices.push_back(dev);
        }

        r->context = ctx;
        प्रत्यागम r;
    }();

    प्रत्यागम _reg;
}

ggml_guid_t ggml_backend_et_guid(शून्य) {
    स्थिर ggml_guid guid = { 0x4b, 0xe0, 0x72, 0x88, 0xc0, 0xf6, 0x29, 0xb4,
                              0x79, 0x9f, 0x70, 0x68, 0x71, 0x0f, 0x6d, 0xc8 };
    प्रत्यागम &guid;
}

ggml_backend_t ggml_backend_et_init(size_t devidx) {
    यदि (!ggml_et_driver_init()) {
        प्रत्यागम शून्यसूचकः;
    }

    यदि (devidx >= (size_t) ggml_backend_et_get_device_गणनम्()) {
        प्रत्यागम शून्यसूचकः;
    }

    ggml_backend_et_context * ctx = नूतन ggml_backend_et_context;
    ctx->devidx                   = (पूर्णाङ्क) devidx;

    ggml_backend_t backend = नूतन ggml_backend{
        /* .guid    = */ ggml_backend_et_guid(),
        /* .iface   = */ ggml_backend_et_i,
        /* .device  = */ ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx),
        /* .context = */ ctx,
    };

    प्रत्यागम backend;
}

तार्किक ggml_backend_is_et(ggml_backend_t backend) {
    प्रत्यागम backend != रिक्त && ggml_guid_matches(backend->guid, ggml_backend_et_guid());
}

पूर्णाङ्क ggml_backend_et_get_device_गणनम्(शून्य) {
    प्रत्यागम ggml_backend_et_reg_get_device_गणनम्(ggml_backend_et_reg());
}

शून्य ggml_backend_et_get_device_description(पूर्णाङ्क devidx, वर्ण * description, size_t description_आकार) {
    यदि (devidx < 0 || devidx >= ggml_backend_et_get_device_गणनम्()) {
        snprintf(description, description_आकार, "ET Device %d (invalid)", devidx);
        प्रत्यागम;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    snprintf(description, description_आकार, "%s", dev_ctx->desc.c_str());
}

शून्य ggml_backend_et_get_device_स्मृतिः(पूर्णाङ्क devidx, size_t * मुक्त, size_t * योग) {
    यदि (devidx < 0 || devidx >= ggml_backend_et_get_device_गणनम्()) {
        *मुक्त  = 0;
        *योग = 0;
        प्रत्यागम;
    }

    ggml_backend_dev_t dev = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), devidx);
    ggml_backend_et_device_get_स्मृतिः(dev, मुक्त, योग);
}

ggml_backend_बफर_type_t ggml_backend_et_बफर_type(size_t dev_अङ्कः) {
    यदि (dev_अङ्कः >= (size_t) ggml_backend_et_get_device_गणनम्()) {
        प्रत्यागम शून्यसूचकः;
    }

    ggml_backend_dev_t               dev     = ggml_backend_et_reg_get_device(ggml_backend_et_reg(), dev_अङ्कः);
    ggml_backend_et_device_context * dev_ctx = (ggml_backend_et_device_context *) dev->context;
    प्रत्यागम dev_ctx->buftype;
}

ggml_backend_बफर_type_t ggml_backend_et_host_बफर_type(शून्य) {
    स्थिर ggml_backend_बफर_type host_बफर_type = {
        /* .iface   = */ ggml_backend_et_बफर_type_i,
        /* .device  = */ शून्यसूचकः,
        /* .context = */ शून्यसूचकः,
    };
    प्रत्यागम &host_बफर_type;
}

GGML_BACKEND_DL_IMPL(ggml_backend_et_reg)
